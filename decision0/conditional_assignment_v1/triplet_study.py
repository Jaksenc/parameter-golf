"""Conditional assignment supervision on verified counterfactual triplets.

A memory-bounded two-pass vector-Jacobian product exactly differentiates the
coupled loss for deterministic forwards, without retaining three model graphs.
The endpoint-only control performs the same forwards and uses coefficient zero.
No triplet constraint, reference assignment or oracle is used at inference.
"""
from __future__ import annotations
import argparse, itertools, json, math, random, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent
from triplet_data import corpus,groups,checks,digest,visible

PROTOCOL={'id':'decision0-conditional-assignment-v1','seed':91823,'arms':['endpoint','conditional'],
 'rank':8,'scale':2.0,'lr':3e-5,'epochs':1,'updates':32,'rows':96,
 'conditional_coefficient':1.0,'group_size':3,'student_input':'baseline single copy',
 'objective':'Endpoint CE plus conditional likelihood of the reference bijection, given the three observed outcome labels.',
 'gradient_method':'Two-pass deterministic VJP; both arms have identical forward count.',
 'train_limit':2048,'eval_limit':8192,'checkpoint':'fixed final; no model or hyperparameter selection on public scores',
 'train_sha256':'bab8c4df8b59f64018b63d66322daff4a7c65393f4a15d84f48fe9e26178626d',
 'held_sha256':'28051a929ddb46b8c0954590914a64c9298568c825bed3d56fdedbfb882aa695',
 'source_inputs':'24 constructed counterfactual triplets plus 24 original SNLI training pairs',
 'base_model':'Qwen/Qwen3.5-4B','base_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'benchmark_revision':'c6004e008ffba24aec091261ca1a5c02f7324702','official_score':None}

def validate_data():
    result=checks()
    for split in ('train','held'):
        if result[split]['sha256']!=PROTOCOL[split+'_sha256']:raise RuntimeError('Frozen data mismatch')
    return result

def assignment_loss(logits,labels):
    """Negative log P(correct assignment | outcome multiset), for matched rows.

    logits[i] contains scores in a common semantic outcome order. labels[i]
    identifies the correct outcome for row i. All correct outcomes must differ.
    """
    import torch
    if len(logits)!=3 or len(labels)!=3 or len(set(labels))!=3:raise ValueError('Require a verified three-outcome triplet')
    if any(z.ndim!=1 for z in logits) or len({len(z) for z in logits})!=1:raise ValueError('Unaligned outcomes')
    terms=[]
    for perm in itertools.permutations(labels):terms.append(sum(logits[i][j] for i,j in enumerate(perm)))
    correct=sum(z[j] for z,j in zip(logits,labels))
    return torch.logsumexp(torch.stack(terms),dim=0)-correct

def init(out):
    p=Path(out);p.mkdir(parents=True,exist_ok=False)
    parent.save(p/'protocol.json',PROTOCOL);parent.save(p/'data_checks.json',validate_data());return p

def forward(rt,enc):
    import torch
    x=torch.tensor([enc['input_ids']]);rt.hidden=None
    output=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
    if rt.hidden is None:raise RuntimeError('Missing normalized hidden state')
    slots=enc['answer_token_ids'];w=rt.head.weight[slots].float();b=rt.head.bias[slots].float() if rt.head.bias is not None else None
    z=torch.nn.functional.linear(rt.hidden.float(),w,b)[0]
    del output,x
    return z

def train(args):
    import torch
    from safetensors.torch import save_file
    p=init(args.out);gs=groups('train');rows=corpus('train');rt=parent.Runtime();parent.save(p/'runtime.json',rt.meta)
    rt.hook.remove()
    def capture(m,ins):rt.hidden=ins[0][:,-1,:] if ins[0].ndim==3 else ins[0]
    rt.hook=rt.head.register_forward_pre_hook(capture)
    enc={r['id']:parent.encode_checked(rt.tokenizer,visible(r),'baseline',PROTOCOL['train_limit']) for r in rows}
    before=rt.score(rows[0],'baseline')['logits'];ad=parent.Adapter(rt,PROTOCOL['seed'])
    initial=parent.tensor_hash(ad.state());zero=rt.score(rows[0],'baseline')['logits']
    if max(abs(x-y) for x,y in zip(before,zero))>1e-4:raise RuntimeError('Zero adapter changed base')
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for n,m in rt.model.named_modules() if 'language_model' in n):raise RuntimeError('Two-pass exactness requires deterministic model')
    parent.save(p/'initialized.json',{'parameters':sum(x.numel() for x in ad.factors.parameters()),'initial_sha256':initial,'seed':PROTOCOL['seed'],'arm':args.arm,'tokens':{r['id']:enc[r['id']]['input_tokens'] for r in rows},'source_groups':[[r['id'] for r in g] for g in gs]})
    rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads();rt.model.train()
    opt=torch.optim.AdamW(ad.factors.parameters(),lr=PROTOCOL['lr'],weight_decay=0)
    order=list(range(len(gs)));random.Random(PROTOCOL['seed']).shuffle(order)
    coefficient=PROTOCOL['conditional_coefficient'] if args.arm=='conditional' else 0.0
    steps=[];seen=[];begin=time.perf_counter();max_forward_replay=0.0
    for update,gi in enumerate(order,1):
        group=gs[gi];tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        # First pass computes only the small coupled objective's inputs.
        detached=[]
        with torch.no_grad():
            for r in group:
                z=forward(rt,enc[r['id']]);detached.append(z.detach().clone());rt.hidden=None
        leaves=[z.clone().requires_grad_(True) for z in detached]
        labels=[[o['id'] for o in r['options']].index(r['expected']) for r in group]
        ce=torch.stack([-torch.log_softmax(z,-1)[y] for z,y in zip(leaves,labels)]).mean()
        conditional=assignment_loss(leaves,labels) if group[0]['matched'] else sum(z.sum()*0 for z in leaves)
        loss=ce+coefficient*conditional
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite group objective')
        adjoints=torch.autograd.grad(loss,leaves)
        for r,old,adj in zip(group,detached,adjoints):
            z=forward(rt,enc[r['id']]);err=float((z.detach()-old).abs().max());max_forward_replay=max(max_forward_replay,err)
            if err>1e-4:raise RuntimeError(f'Nondeterministic coupled-gradient replay: {err}')
            if not z.requires_grad:raise RuntimeError('Model gradient detached')
            torch.sum(z*adj.detach()).backward();seen.append(r['id']);rt.hidden=None
            del z
        norms=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
        if any(v is None or not math.isfinite(v) or v<=0 for v in norms):raise RuntimeError('Missing across-depth gradients')
        if any(x.grad is not None for x in rt.model.parameters()):raise RuntimeError('Base weight acquired gradient')
        gn=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1.0));opt.step()
        rec={'update':update,'group':gi,'ids':[r['id'] for r in group],'matched':group[0]['matched'],'loss':float(loss.detach()),'endpoint_ce':float(ce.detach()),'conditional_loss':float(conditional.detach()),'gradient_norm':gn,'B_gradient_norms':norms,'seconds':time.perf_counter()-tick,'peak_rss_mib':parent.memory()}
        steps.append(rec);parent.emit(phase='triplet_train',arm=args.arm,**rec)
        parent.save(p/'progress.json',{'steps':steps,'seen':seen,'forward_replay_max_error':max_forward_replay})
        if update%8==0:save_file(ad.state(),str(p/'recovery.safetensors'))
    if len(steps)!=32 or len(seen)!=96 or len(set(seen))!=96:raise RuntimeError('Training coverage mismatch')
    rt.model.eval();rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads()
    cp=p/'adapter.safetensors';save_file(ad.state(),str(cp));probe=rt.score(rows[0],'baseline')['logits']
    ad.enabled=False;restored=rt.score(rows[0],'baseline')['logits'];restore_err=max(abs(x-y) for x,y in zip(before,restored))
    with torch.no_grad():
        for factor in ad.factors:factor.b.zero_()
    ad.load(cp);ad.enabled=True;loaded=rt.score(rows[0],'baseline')['logits'];load_err=max(abs(x-y) for x,y in zip(probe,loaded))
    if max(restore_err,load_err)>1e-4:raise RuntimeError('Checkpoint round-trip failed')
    if (p/'recovery.safetensors').exists():(p/'recovery.safetensors').unlink()
    parent.save(p/'training_rows.json',rows)
    receipt={'arm':args.arm,'seed':PROTOCOL['seed'],'checkpoint_sha256':parent.filehash(cp),'protocol_hash':digest(PROTOCOL),'initial_sha256':initial,'updates':len(steps),'unique_rows':len(rows),'train_forwards':2*len(rows),'student_tokens_per_pass':sum(e['input_tokens'] for e in enc.values()),'max_student_tokens':max(e['input_tokens'] for e in enc.values()),'elapsed_seconds':time.perf_counter()-begin,'peak_rss_mib':parent.memory(),'updated_blocks':sum(bool(f.b.abs().sum()>0) for f in ad.factors),'same_process_reload_error':load_err,'base_restoration_error':restore_err,'forward_replay_max_error':max_forward_replay,'new_capability_established':False}
    parent.save(p/'receipt.json',receipt);parent.emit(phase='triplet_trained',**receipt)

def checkpoints(root):
    out={}
    for p in Path(root).rglob('adapter.safetensors'):
        r=json.loads((p.parent/'receipt.json').read_text())
        if parent.filehash(p)!=r['checkpoint_sha256'] or r['protocol_hash']!=digest(PROTOCOL):raise RuntimeError('Invalid checkpoint')
        if r['arm'] in out:raise RuntimeError('Duplicate checkpoint')
        out[r['arm']]=p
    if set(out)!=set(PROTOCOL['arms']):raise RuntimeError('Missing trained arm')
    return out

def eval_rows():return parent.old_corpus('bench')+corpus('held')

def evaluate(args):
    from safetensors.torch import load_file
    p=init(args.out);rows=eval_rows();ids=parent.assignment(rows,12)[args.shard]
    states={k:load_file(str(v)) for k,v in checkpoints(args.checkpoint_root).items()}
    rt=parent.Runtime();ad=parent.Adapter(rt,PROTOCOL['seed']);parent.save(p/'runtime.json',rt.meta)
    variants=['base']+PROTOCOL['arms'];count=0
    with (p/'records.jsonl').open('w') as f:
        for i in ids:
            r=rows[i]
            for variant in variants[i%3:]+variants[:i%3]:
                ad.enabled=variant!='base'
                if ad.enabled:ad.factors.load_state_dict(states[variant])
                rec=rt.score(r,'baseline');rec['variant']=variant
                rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit','matched','long_context')})
                rec['correct']=rec['predicted']==r['expected'];f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
            parent.emit(phase='triplet_eval',shard=args.shard,done=count,total=3*len(ids))
    parent.save(p/'receipt.json',{'records':count,'sha256':parent.filehash(p/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'full_model_inference':True})

def aggregate(args):
    p=init(args.out);records=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
        receipt=json.loads((f.parent/'receipt.json').read_text())
        if receipt['sha256']!=parent.filehash(f) or receipt['protocol_hash']!=digest(PROTOCOL):raise RuntimeError('Invalid output shard')
        for l in f.read_text().splitlines():
            r=json.loads(l);key=(r['id'],r['variant'])
            if key in seen:raise RuntimeError('Duplicate prediction')
            seen.add(key);records.append(r)
    variants=['base']+PROTOCOL['arms']
    if seen!={(r['id'],v) for r in eval_rows() for v in variants}:raise RuntimeError('Incomplete evaluation')
    base={r['id']:r for r in records if r['variant']=='base'};summary={}
    for variant in variants:
        summary[variant]={}
        for split in ('public','held'):
            rs=[r for r in records if r['variant']==variant and (r['tier']=='new-held')==(split=='held')]
            m=parent.metrics(rs);m['families']={fam:parent.metrics([r for r in rs if r['family']==fam]) for fam in sorted({r['family'] for r in rs})}
            m['repairs']=sum(r['correct'] and not base[r['id']]['correct'] for r in rs);m['regressions']=sum(not r['correct'] and base[r['id']]['correct'] for r in rs)
            if split=='held':
                matched=[r for r in rs if r.get('matched')];sources={r['source'] for r in matched}
                m['triplets']={fam:{'n':len({r['source'] for r in matched if r['family']==fam}),'all_correct':sum(all(r['correct'] for r in matched if r['source']==s) for s in sources if next(r['family'] for r in matched if r['source']==s)==fam)} for fam in ('amount','elapsed','join','policy')}
            summary[variant][split]=m
    records.sort(key=lambda r:(r['id'],r['variant']));(p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    parent.save(p/'results.json',{'summary':summary,'protocol':PROTOCOL,'records':len(records),'official_score':None});parent.emit(phase='triplet_complete',summary=summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--arm',choices=PROTOCOL['arms'],required=True)
    e=sub.add_parser('evaluate');e.add_argument('--shard',type=int,required=True);e.add_argument('--checkpoint-root',required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',required=True)
    for parser in (t,e,a):parser.add_argument('--out',required=True)
    args=ap.parse_args()
    try:{'train':train,'evaluate':evaluate,'aggregate':aggregate}[args.mode](args)
    except Exception as exc:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True);parent.save(p/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
