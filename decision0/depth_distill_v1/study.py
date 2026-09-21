"""A bounded across-depth distillation experiment, not an official JevBench run."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
from prompt_variants import messages,encode_checked
from flow_study import Runtime,corpus as old_corpus,save,filehash,prediction,softmax
from cases import corpus,checks,digest,visible,FAMILIES

PROTOCOL={'id':'decision0-depth-distill-v1','seeds':[73019,73037],'arms':['reference','repeat_distill'],
 'rank':8,'scale':2.0,'lr':3e-5,'weight_decay':0.0,'epochs':1,'microbatch':1,'accumulation':4,
 'training_rows':96,'updates':24,'teacher_weight':0.3,'train_token_limit':1024,'eval_token_limit':8192,
 'teacher_gate':'eligible family and unique repeated-teacher argmax equals construction reference',
 'teacher_families':['policy','join','intent','entailment'],'selection':'fixed final checkpoint; no test or development selection',
 'student_input':'baseline single-copy','teacher_input':'repeat_full','readout':'stock normalized hidden; FP32 native answer rows',
 'base':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'benchmark':'c6004e008ffba24aec091261ca1a5c02f7324702','official_score':None,
 'data_hashes':{'train':'eef49e64c696d07a86b39f303e6e803f625bdf61e8194e7eb4fbf906fac327e1','held':'81bfb471bc1e4719482a25cb9277d3dd2ea761adddef90ddbeef0b9f29c1243a'}}

def emit(**x):print(json.dumps(x,allow_nan=False),flush=True)
def memory():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def verified_data(split):
    rows=corpus(split)
    if digest(rows)!=PROTOCOL['data_hashes'][split]:raise RuntimeError('Frozen data mismatch')
    return rows

def init(out):
    p=Path(out);p.mkdir(parents=True,exist_ok=False);save(p/'protocol.json',PROTOCOL);save(p/'data_checks.json',checks());return p

def prepare(args):
    p=init(args.out);rows=verified_data('train');rt=Runtime();save(p/'runtime.json',rt.meta)
    rows=rows[args.shard::4]
    with (p/'teacher.jsonl').open('w') as f:
      for i,r in enumerate(rows):
        base=rt.score(r,'baseline');teacher=rt.score(r,'repeat_full')
        rec={'id':r['id'],'source':r['source'],'row_hash':digest(r),'base':base,'teacher':teacher}
        f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush()
        emit(phase='teacher',shard=args.shard,done=i+1,total=len(rows))
    save(p/'receipt.json',{'records':len(rows),'sha256':filehash(p/'teacher.jsonl'),'protocol_hash':digest(PROTOCOL)})

def teachers(root):
    out={}
    for p in Path(root).rglob('teacher.jsonl'):
      receipt=json.loads((p.parent/'receipt.json').read_text())
      if filehash(p)!=receipt['sha256'] or receipt['protocol_hash']!=digest(PROTOCOL):raise RuntimeError('Teacher artifact mismatch')
      for l in p.read_text().splitlines():
        r=json.loads(l)
        if r['id'] in out:raise RuntimeError('Duplicate teacher')
        out[r['id']]=r
    rows=verified_data('train')
    if set(out)!={r['id'] for r in rows}:raise RuntimeError('Incomplete teachers')
    for r in rows:
      t=out[r['id']]
      if t['row_hash']!=digest(r) or t['teacher']['labels']!=[o['id'] for o in r['options']]:raise RuntimeError('Teacher label alignment')
    return out

def target_for(r,t,arm):
    labels=[o['id'] for o in r['options']];q=[r['target_probs'][x] for x in labels]
    p=t['teacher']['probabilities'];m=max(p)
    use=bool(r['teacher_eligible'] and sum(x==m for x in p)==1 and t['teacher']['predicted']==r['expected'])
    w=PROTOCOL['teacher_weight'] if arm=='repeat_distill' and use else 0.0
    return [(1-w)*a+w*b for a,b in zip(q,p)],use,w

class Adapter:
    def __init__(self,rt,seed):
      import torch
      self.rt=rt;self.enabled=True;torch.manual_seed(seed)
      class Factors(torch.nn.Module):
        def __init__(self,down):
          super().__init__();self.a=torch.nn.Parameter(torch.empty(8,down.in_features));self.b=torch.nn.Parameter(torch.zeros(down.out_features,8));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):return (2*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
      self.targets=[(n,m) for n,m in rt.model.named_modules() if n.startswith('model.language_model.layers.') and n.endswith('.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
      if len(self.targets)!=32:raise RuntimeError('Wrong target count')
      self.factors=torch.nn.ModuleList([Factors(m) for n,m in self.targets]);self.hooks=[]
      for (name,module),factor in zip(self.targets,self.factors):
        def hook(m,args,result,factor=factor):return result+factor(args[0]) if self.enabled else result
        self.hooks.append(module.register_forward_hook(hook))
    def state(self):return {k:v.detach().cpu().contiguous().clone() for k,v in self.factors.state_dict().items()}
    def load(self,path):
      from safetensors.torch import load_file
      self.factors.load_state_dict(load_file(str(path)))

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def train(args):
    import torch
    from safetensors.torch import save_file
    p=init(args.out);rows=verified_data('train');ts=teachers(args.teacher_root)
    rt=Runtime();save(p/'runtime.json',rt.meta)
    # Replace the inference-only capture with a differentiable view. No detach here.
    rt.hook.remove()
    def capture(m,ins):rt.hidden=ins[0][:,-1,:] if ins[0].ndim==3 else ins[0]
    rt.hook=rt.head.register_forward_pre_hook(capture)
    enc=[encode_checked(rt.tokenizer,visible(r),'baseline',PROTOCOL['train_token_limit']) for r in rows]
    before=rt.score(rows[0],'baseline')['logits'];ad=Adapter(rt,args.seed)
    initial_hash=tensor_hash(ad.state());zero=rt.score(rows[0],'baseline')['logits']
    if max(abs(a-b) for a,b in zip(zero,before))>1e-4:raise RuntimeError('Zero update changes model')
    save(p/'initialized.json',{'parameters':sum(x.numel() for x in ad.factors.parameters()),'seed':args.seed,'arm':args.arm,'initial_sha256':initial_hash,'tokens':[x['input_tokens'] for x in enc],'layers':[n for n,m in ad.targets]})
    rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads();rt.model.train()
    opt=torch.optim.AdamW(ad.factors.parameters(),lr=PROTOCOL['lr'],weight_decay=0)
    order=list(range(len(rows)));random.Random(args.seed).shuffle(order);updates=0;begin=time.perf_counter()
    targets=[];seen=[];steps=[]
    for start in range(0,len(order),4):
      opt.zero_grad(set_to_none=True);loss_value=0.0;tick=time.perf_counter()
      ids=order[start:start+4]
      for i in ids:
        r=rows[i];e=enc[i];x=torch.tensor([e['input_ids']]);rt.hidden=None
        out=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
        if rt.hidden is None or not rt.hidden.requires_grad:raise RuntimeError('Gradient path detached')
        slots=e['answer_token_ids'];w=rt.head.weight[slots].float();b=rt.head.bias[slots].float() if rt.head.bias is not None else None
        z=torch.nn.functional.linear(rt.hidden.float(),w,b)[0];q,use,mix=target_for(r,ts[r['id']],args.arm)
        loss=-(torch.tensor(q)*torch.log_softmax(z,-1)).sum()/len(ids)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss_value+=float(loss.detach());loss.backward();seen.append(r['id'])
        targets.append({'id':r['id'],'target':q,'teacher_eligible_correct':use,'mixture_weight':mix})
        del out,loss,z,x;rt.hidden=None
      norms=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
      if any(v is None or not math.isfinite(v) or v<=0 for v in norms):raise RuntimeError('Missing across-depth gradient')
      if any(x.grad is not None for x in rt.model.parameters()):raise RuntimeError('Base parameter acquired gradient')
      gn=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1.0));opt.step();updates+=1
      row={'update':updates,'loss':loss_value,'gradient_norm':gn,'B_gradient_norms':norms,'seconds':time.perf_counter()-tick,'peak_rss_mib':memory()};steps.append(row);emit(phase='train',arm=args.arm,seed=args.seed,**row)
      save(p/'progress.json',{'updates':updates,'steps':steps,'seen':seen})
      if updates%8==0:save_file(ad.state(),str(p/'recovery.safetensors'))
    if updates!=24 or len(set(seen))!=96:raise RuntimeError('Training budget/coverage mismatch')
    rt.model.eval();rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads()
    checkpoint=p/'adapter.safetensors';save_file(ad.state(),str(checkpoint));weight_hash=filehash(checkpoint)
    probe=rt.score(rows[0],'baseline')['logits']
    ad.enabled=False;restored=rt.score(rows[0],'baseline')['logits'];restore_err=max(abs(a-b) for a,b in zip(restored,before))
    with torch.no_grad():
      for f in ad.factors:f.b.zero_()
    ad.load(checkpoint);ad.enabled=True;reloaded=rt.score(rows[0],'baseline')['logits'];reload_err=max(abs(a-b) for a,b in zip(reloaded,probe))
    if restore_err>1e-4 or reload_err>1e-4:raise RuntimeError('Checkpoint restoration mismatch')
    if (p/'recovery.safetensors').exists():(p/'recovery.safetensors').unlink()
    save(p/'targets.json',targets)
    receipt={'arm':args.arm,'seed':args.seed,'updates':updates,'examples':len(seen),'initial_sha256':initial_hash,'checkpoint_sha256':weight_hash,'updated_blocks':sum(bool(f.b.abs().sum()>0) for f in ad.factors),'training_seconds':time.perf_counter()-begin,'peak_rss_mib':memory(),'base_restoration_error':restore_err,'same_process_reload_error':reload_err,'protocol_hash':digest(PROTOCOL),'new_capability_established':False}
    save(p/'receipt.json',receipt);emit(phase='trained',**receipt)


def checkpoints(root):
    cps={}
    for p in Path(root).rglob('adapter.safetensors'):
      r=json.loads((p.parent/'receipt.json').read_text())
      if filehash(p)!=r['checkpoint_sha256'] or r['protocol_hash']!=digest(PROTOCOL):raise RuntimeError('Checkpoint integrity')
      key=f"{r['arm']}-{r['seed']}"
      if key in cps:raise RuntimeError('Duplicate checkpoint')
      cps[key]=p
    expected={f'{a}-{s}' for a in PROTOCOL['arms'] for s in PROTOCOL['seeds']}
    if set(cps)!=expected:raise RuntimeError('Missing trained checkpoint')
    return cps

def eval_rows():
    held=verified_data('held')
    for r in held:r['tier']='held'
    return old_corpus('bench')+held

def assignment(rows,n=12):
    loads=[0]*n;buckets=[[] for _ in range(n)]
    for i in sorted(range(len(rows)),key=lambda i:(-len(json.dumps(visible(rows[i]))),i)):
      j=min(range(n),key=lambda j:(loads[j],j));buckets[j].append(i);loads[j]+=len(json.dumps(visible(rows[i])))
    return [sorted(b) for b in buckets]

def evaluate(args):
    p=init(args.out);rows=eval_rows();ids=assignment(rows)[args.shard];cps=checkpoints(args.checkpoint_root);rt=Runtime();ad=Adapter(rt,PROTOCOL['seeds'][0]);save(p/'runtime.json',rt.meta)
    from safetensors.torch import load_file
    states={k:load_file(str(v)) for k,v in cps.items()}
    variants=['base','repeat']+sorted(cps);save(p/'assignment.json',{'ids':[rows[i]['id'] for i in ids],'variants':variants,'corpus_hash':digest(rows)})
    n=0
    with (p/'records.jsonl').open('w') as f:
      for i in ids:
        r=rows[i]
        for variant in variants[i%len(variants):]+variants[:i%len(variants)]:
          if variant in ('base','repeat'):ad.enabled=False
          else:ad.factors.load_state_dict(states[variant]);ad.enabled=True
          layout='repeat_full' if variant=='repeat' else 'baseline'
          rec=rt.score(r,layout);rec['variant']=variant;rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit')});rec['correct']=rec['predicted']==r['expected']
          f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();n+=1
        emit(phase='eval',shard=args.shard,done=n,total=len(ids)*len(variants))
    save(p/'receipt.json',{'records':n,'sha256':filehash(p/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'full_model_inference':True})

def metrics(rs):
    n=len(rs);m={'n':n,'correct':sum(r['correct'] for r in rs)};m['accuracy']=m['correct']/n
    m['nll']=sum(-math.log(max(dict(zip(r['labels'],r['probabilities']))[r['expected']],1e-15)) for r in rs)/n
    ps=[r for r in rs if r.get('gold_probs') or (r.get('family')=='probability' and r.get('target_probs'))]
    m['probability_n']=len(ps);m['tvd']=sum(sum(abs(dict(zip(r['labels'],r['probabilities'])).get(k,0)-v) for k,v in (r.get('gold_probs') or r['target_probs']).items())/2 for r in ps)/len(ps) if ps else None
    m['tokens']=sum(r['input_tokens'] for r in rs)
    return m

def aggregate(args):
    p=init(args.out);rows=eval_rows();records=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
      receipt=json.loads((f.parent/'receipt.json').read_text())
      if filehash(f)!=receipt['sha256'] or receipt['protocol_hash']!=digest(PROTOCOL):raise RuntimeError('Evaluation artifact mismatch')
      for l in f.read_text().splitlines():
        r=json.loads(l);key=(r['id'],r['variant'])
        if key in seen:raise RuntimeError('Duplicate inference')
        seen.add(key);records.append(r)
    variants=['base','repeat']+[f'{a}-{s}' for a in PROTOCOL['arms'] for s in PROTOCOL['seeds']]
    if seen!={(r['id'],v) for r in rows for v in variants}:raise RuntimeError('Incomplete evaluation; refusing summary')
    summary={}
    for v in variants:
      rs=[r for r in records if r['variant']==v];parts={}
      for split in ('public','held'):
        rr=[r for r in rs if (r['tier']=='held')==(split=='held')];parts[split]=metrics(rr)
        parts[split]['families']={fam:metrics([r for r in rr if r['family']==fam]) for fam in sorted({r['family'] for r in rr})}
        if split=='public':parts[split]['tiers']={t:metrics([r for r in rr if r['tier']==t]) for t in ('easy','standard','hard')}
        base={r['id']:r for r in records if r['variant']=='base'}
        parts[split]['repairs']=sum(r['correct'] and not base[r['id']]['correct'] for r in rr);parts[split]['regressions']=sum(not r['correct'] and base[r['id']]['correct'] for r in rr)
      summary[v]=parts
    (p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in sorted(records,key=lambda r:(r['id'],r['variant']))))
    save(p/'results.json',{'summary':summary,'official_score':None,'record_count':len(records),'protocol':PROTOCOL,'records_hash':filehash(p/'records.jsonl')});emit(phase='complete',summary=summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('teacher');a.add_argument('--shard',type=int,required=True)
    b=sub.add_parser('train');b.add_argument('--seed',type=int,choices=PROTOCOL['seeds'],required=True);b.add_argument('--arm',choices=PROTOCOL['arms'],required=True);b.add_argument('--teacher-root',required=True)
    c=sub.add_parser('evaluate');c.add_argument('--shard',type=int,required=True);c.add_argument('--checkpoint-root',required=True)
    d=sub.add_parser('aggregate');d.add_argument('--root',required=True)
    for parser in (a,b,c,d):parser.add_argument('--out',required=True)
    args=ap.parse_args()
    try:{'teacher':prepare,'train':train,'evaluate':evaluate,'aggregate':aggregate}[args.mode](args)
    except Exception as e:
      p=Path(args.out);p.mkdir(parents=True,exist_ok=True);save(p/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'peak_rss_mib':memory()});raise
