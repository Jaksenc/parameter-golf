"""Real 4B diagnostic: original versus blocked answer-code schedule.
No public benchmark inputs or labels. Both arms see an identical multiset.
Fixed final step 36; checkpoints are written before potentially slow evaluation.
This tests small-set learnability, not generalization or benchmark performance.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,math,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent
SEED=94231
PERMS=tuple(itertools.permutations(range(3)))

def perm_for(step,example,mode):
    if step<1 or example not in range(3) or mode not in ('original','blocked'):raise ValueError('Invalid schedule')
    return PERMS[(step-1+(example if mode=='original' else 0))%6]

def examples(split):
    n,p,f,c=(4,121,151,27) if split=='seen' else (7,143,127,35)
    center=n*p+f-c
    options=[{'id':f'value{i}','description':f'The exact result is {center+d} cents.'} for i,d in enumerate([-1,0,1])]
    options=[options[2],options[0],options[1]]
    return [{'id':f'{split}-{i}','state':f'Quantity {n}. Unit price {p} cents. Handling fee {f+d} cents. Credit {c} cents.',
      'question':'Compute quantity times unit price, plus handling fee, minus credit. Select the exact result in cents.',
      'options':copy.deepcopy(options),'expected':f'value{i}'} for i,d in enumerate([-1,0,1])]

def view(row,step,example,mode):
    r=copy.deepcopy(row);r['options']=[row['options'][j] for j in perm_for(step,example,mode)];return r

def self_test():
    from collections import Counter
    rows=examples('seen');counts={};mse={}
    for mode in ('original','blocked'):
        counts[mode]=Counter();sq=[]
        for step in range(1,37):
            pos=[]
            for i,row in enumerate(rows):
                v=view(row,step,i,mode);j=[o['id'] for o in v['options']].index(v['expected']);pos.append(j)
                counts[mode][(i,perm_for(step,i,mode))]+=1
            if mode=='blocked':assert sorted(pos)==[0,1,2]
            sq.append(sum((1/3-pos.count(j)/3)**2 for j in range(3)))
        mse[mode]=sum(sq)/len(sq)
    assert counts['original']==counts['blocked']
    assert abs(mse['original']-1/3)<1e-12 and mse['blocked']==0
    for split in ('seen','unseen'):
        n,p,f,c=(4,121,151,27) if split=='seen' else (7,143,127,35)
        for i,r in enumerate(examples(split)):
            expected=n*p+f+(i-1)-c
            desc=next(o['description'] for o in r['options'] if o['id']==r['expected'])
            assert desc==f'The exact result is {expected} cents.'
    return {'same_training_multiset':True,'uniform_bias_mse':mse,'reference_arithmetic_checked':6,'benchmark_calls':0}

def protocol(mode):
    return {'id':'decision0-frontier-balanced-microfit-v1','seed':SEED,'updates':36,'learning_rate':0.0003,
      'accumulation':3,'rank':8,'schedule':mode,'save':'after every optimizer update, before evaluation',
      'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
      'selection':'fixed final step 36; no data-dependent endpoint or seed selection',
      'evaluation':'seen and near-unseen examples under all six orders at steps 0,12,24,36',
      'scope':'three-example learning diagnostic, not benchmark performance','benchmark_calls':0}

def save_state(path,step,ad,opt,pr):
    import torch
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    state={'step':step,'factors':ad.state(),'optimizer':opt.state_dict(),'protocol':pr,'torch_rng':torch.get_rng_state().clone()}
    try:
        with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def run(args):
    import torch
    from safetensors.torch import save_file
    p=Path(args.out);p.mkdir(parents=True,exist_ok=True);pr=protocol(args.schedule)
    if (p/'protocol.json').exists() and not args.resume:raise ValueError('Refusing to overwrite an experiment')
    parent.save(p/'protocol.json',pr);parent.save(p/'schedule_checks.json',self_test())
    rows=examples('seen');held=examples('unseen');rt=parent.Runtime();parent.save(p/'runtime.json',rt.meta)
    rt.hook.remove()
    def capture(m,ins):rt.hidden=ins[0][:,-1,:] if ins[0].ndim==3 else ins[0]
    rt.hook=rt.head.register_forward_pre_hook(capture)
    before=[rt.score(r,'baseline')['logits'] for r in rows];ad=parent.Adapter(rt,SEED)
    initial=parent.tensor_hash(ad.state());opt=torch.optim.AdamW(ad.factors.parameters(),lr=pr['learning_rate'],weight_decay=0)
    start_step=1;latest=p/'latest.pt';history=[]
    if args.resume:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['protocol']!=pr:raise RuntimeError('Resume protocol mismatch')
        ad.factors.load_state_dict(saved['factors']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);start_step=saved['step']+1
        if (p/'history.json').exists():history=json.loads((p/'history.json').read_text())
    parent.save(p/'inputs.json',{'seen':rows,'unseen':held});views={};encoded={}
    for s in range(1,7):
        for i,r in enumerate(rows):
            key=(s,i);views[key]=view(r,s,i,args.schedule);encoded[key]=parent.encode_checked(rt.tokenizer,views[key],'baseline',512)
    parent.save(p/'initialized.json',{'initial_sha256':initial,'parameters':sum(x.numel() for x in ad.factors.parameters()),'input_lengths':{str(k):e['input_tokens'] for k,e in encoded.items()}})
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for n,m in rt.model.named_modules() if 'language_model' in n):raise RuntimeError('Requires deterministic forwards')
    rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads()
    def evaluate(step):
        rt.model.eval();out=[]
        for split,rs in [('seen',rows),('unseen',held)]:
            for row in rs:
                for perm in PERMS:
                    q=copy.deepcopy(row);q['options']=[row['options'][j] for j in perm]
                    rec=rt.score(q,'baseline');rec.update({'step':step,'split':split,'order':'perm-'+''.join(map(str,perm)),'expected':row['expected'],'correct':rec['predicted']==row['expected']});out.append(rec)
        parent.save(p/f'evaluation-step-{step}.json',out)
        parent.emit(phase='evaluation',step=step,schedule=args.schedule,seen_correct=sum(r['correct'] for r in out if r['split']=='seen'),unseen_correct=sum(r['correct'] for r in out if r['split']=='unseen'),denominator_each=18)
        rt.model.train()
    if start_step==1:evaluate(0)
    else:
        persisted=start_step-1
        if persisted in (12,24,36) and not (p/f'evaluation-step-{persisted}.json').exists():evaluate(persisted)
        else:rt.model.train()
    for update in range(start_step,37):
        opt.zero_grad(set_to_none=True);loss_total=0.;tick=time.perf_counter();order=[0,1,2];random.Random(SEED+update).shuffle(order)
        for i in order:
            key=((update-1)%6+1,i);r=views[key];e=encoded[key];x=torch.tensor([e['input_ids']]);rt.hidden=None
            raw=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
            if rt.hidden is None or not rt.hidden.requires_grad:raise RuntimeError('Detached gradient')
            slots=e['answer_token_ids'];b=rt.head.bias[slots].float() if rt.head.bias is not None else None
            z=torch.nn.functional.linear(rt.hidden.float(),rt.head.weight[slots].float(),b)[0];y=[o['id'] for o in r['options']].index(r['expected'])
            loss=-torch.log_softmax(z,-1)[y]/3
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss_total+=float(loss.detach());loss.backward();rt.hidden=None;del raw,z,x,loss
        grads=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
        if any(g is None or not math.isfinite(g) for g in grads):raise RuntimeError('Invalid across-depth gradients')
        if any(x.grad is not None for x in rt.model.parameters()):raise RuntimeError('Frozen base acquired gradient')
        gn=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1.0));opt.step();save_state(latest,update,ad,opt,pr)
        rec={'persisted_step':update,'training_loss':loss_total,'gradient_norm':gn,'B_gradient_norms':grads,'seconds':time.perf_counter()-tick,'peak_rss_mib':parent.memory()};history.append(rec)
        parent.save(p/'history.json',history);parent.emit(phase='training',schedule=args.schedule,**rec)
        if update in (12,24,36):evaluate(update)
    rt.model.eval();rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads()
    cp=p/'adapter.safetensors';save_file(ad.state(),str(cp));probe=rt.score(rows[0],'baseline')['logits']
    ad.enabled=False;restore=max(abs(a-b) for row,z in zip(rows,before) for a,b in zip(rt.score(row,'baseline')['logits'],z))
    with torch.no_grad():
        for f in ad.factors:f.b.zero_()
    ad.load(cp);ad.enabled=True;reload=max(abs(a-b) for a,b in zip(probe,rt.score(rows[0],'baseline')['logits']))
    if max(restore,reload)>1e-4:raise RuntimeError('Restoration mismatch')
    parent.save(p/'receipt.json',{'completed_updates':36,'initial_sha256':initial,'checkpoint_sha256':parent.filehash(cp),'same_process_reload_error':reload,'base_restoration_error':restore,'updated_blocks':sum(bool(f.b.abs().sum()>0) for f in ad.factors),'benchmark_calls':0,'general_capability_established':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--schedule',choices=['original','blocked'],default='blocked');ap.add_argument('--out',default='result');ap.add_argument('--resume',action='store_true');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:
        try:run(args)
        except Exception as exc:
            p=Path(args.out);p.mkdir(parents=True,exist_ok=True);parent.save(p/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc(),'final_result_valid':False});raise
