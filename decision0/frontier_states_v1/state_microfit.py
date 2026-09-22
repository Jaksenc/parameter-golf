"""Two predeclared output curricula on real Qwen3.5-4B.
No benchmark calls. No numerical values from the reference enter generation.
This is a six-example fit study, not a general arithmetic or release claim.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent

PROTOCOL={'id':'decision0-verified-state-microfit-v1','seed':108731,'arms':['repeat_total','process'],
          'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
          'updates':36,'accumulation':2,'lr':0.0003,'rank':8,'scale':2.0,
          'train_cases':6,'held_cases':6,'selection':'fixed step 36; no best-checkpoint selection',
          'base_frozen':True,'train_blocks':32,'max_new_tokens':40,'max_train_tokens':512,
          'loss':'native full-vocabulary assistant-token CE, including EOS; both arms emit a,b,c',
          'output_length_control':'same JSON keys and 3-digit values at all training targets; tokenizer equality is required',
          'primary':'free-running exact c value; all required fields and all triplet endpoints also reported',
          'matched':'same numerical inputs, seed, ordering, updates, target-token counts; mode instruction and target values differ',
          'scope':'small learnability study; two procedural source situations per split, no claim of broad generalization',
          'benchmark_calls':0,'official_score':None}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def cases(split):
    specs=[(3,137,91,26),(6,89,73,48)] if split=='train' else [(5,113,86,39),(9,127,107,54)]
    out=[]
    for source,(q,p,f,c) in enumerate(specs):
        for d in (-1,0,1):
            prod=q*p;sub=prod+f+d;total=sub-c
            out.append({'id':f'{split}-{source}-{d+1}','source':f'{split}-{source}',
                        'quantity':q,'price':p,'fee':f+d,'credit':c,
                        'expected':{'a':prod,'b':sub,'c':total}})
    return out

def target(row,arm):
    e=row['expected']
    return dict(e) if arm=='process' else {'a':e['c'],'b':e['c'],'c':e['c']}

def conversation(row,arm):
    if arm not in PROTOCOL['arms']:raise ValueError(arm)
    mapping='a is product, b is subtotal, and c is total.' if arm=='process' else 'a is total, b is total, and c is total.'
    return [{'role':'system','content':'Calculate using exact integer cents. Product is quantity times unit price. Subtotal is product plus handling fee. Total is subtotal minus credit. Return only one JSON object with integer fields a, b, c in that order; '+mapping},
            {'role':'user','content':f"Quantity {row['quantity']}. Unit price {row['price']} cents. Handling fee {row['fee']} cents. Credit {row['credit']} cents."}]

def parse_output(text):
    try:
        obj=json.loads(text)
        if not isinstance(obj,dict) or set(obj)!={'a','b','c'} or any(type(v) is not int for v in obj.values()):raise ValueError('wrong schema')
        return obj,None
    except (ValueError,TypeError) as ex:return None,str(ex)

def checks():
    from decimal import Decimal
    rows=cases('train')+cases('held')
    for r in rows:
        prod=int(Decimal(r['quantity'])*Decimal(r['price']));sub=prod+r['fee'];total=sub-r['credit']
        assert r['expected']=={'a':prod,'b':sub,'c':total}
        for arm in PROTOCOL['arms']:
            msg=conversation(r,arm)
            assert r['id'] not in json.dumps(msg)
            assert set(target(r,arm))=={'a','b','c'}
    assert len({(r['quantity'],r['price'],r['fee'],r['credit']) for r in rows})==12
    assert all(len(str(x))==3 for r in cases('train') for x in r['expected'].values())
    return {'cases':12,'train_sha256':digest(cases('train')),'held_sha256':digest(cases('held'))}

def run(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file,load_file
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    parent.save(out/'protocol.json',PROTOCOL);parent.save(out/'case_checks.json',checks())
    rows=cases('train');held=cases('held');parent.save(out/'cases.json',{'train':rows,'held':held})
    rt=parent.Runtime();parent.save(out/'runtime.json',rt.meta)
    rt.hook.remove();native=rt.model.get_output_embeddings()
    class Capture(torch.nn.Module):
        def __init__(self):super().__init__();self.hidden=None
        def forward(self,h):self.hidden=h;return h[...,:1]*0
    capture=Capture()
    def prep(row,arm):
        text=rt.tokenizer.apply_chat_template(conversation(row,arm),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ans=json.dumps(target(row,arm),separators=(',',':'))
        prefix=rt.tokenizer.encode(text,add_special_tokens=False);suf=rt.tokenizer.encode(ans,add_special_tokens=False)
        if rt.tokenizer.encode(text+ans,add_special_tokens=False)!=prefix+suf:raise RuntimeError('Unstable target boundary')
        eos=rt.tokenizer.eos_token_id
        if type(eos) is not int:raise RuntimeError('Missing EOS')
        full=prefix+suf+[eos]
        if len(full)>PROTOCOL['max_train_tokens']:raise RuntimeError('Oversized training sequence')
        return {'ids':full,'prefix':len(prefix),'target_count':len(suf)+1,'text':text,'target':ans}
    allenc={(r['id'],arm):prep(r,arm) for r in rows for arm in PROTOCOL['arms']}
    for r in rows:
        if allenc[(r['id'],'process')]['target_count']!=allenc[(r['id'],'repeat_total')]['target_count']:raise RuntimeError('Target token budgets differ')
    parent.save(out/'tokenization.json',{'training':{f'{i}/{a}':{k:v for k,v in e.items() if k not in ('ids','text')} for (i,a),e in allenc.items()},
                'number_tokens':{s:rt.tokenizer.encode(s,add_special_tokens=False) for s in ('475','607','1143','6 0 7')},'same_targets_per_case':True})
    ad=parent.Adapter(rt,PROTOCOL['seed']);init=parent.tensor_hash(ad.state())
    def evaluate(step):
        rt.model.set_output_embeddings(native);rt.model.eval();rt.model.gradient_checkpointing_disable()
        recs=[]
        with torch.no_grad():
            for split,rs in [('train',rows),('held',held)]:
                for row in rs:
                    text=rt.tokenizer.apply_chat_template(conversation(row,args.arm),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    ids=rt.tokenizer.encode(text,add_special_tokens=False);x=torch.tensor([ids]);start=time.perf_counter()
                    y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=PROTOCOL['max_new_tokens'],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
                    generated=y[0,len(ids):].tolist();raw=rt.tokenizer.decode(generated,skip_special_tokens=True);parsed,error=parse_output(raw);gold=target(row,args.arm)
                    rec={'id':row['id'],'source':row['source'],'split':split,'step':step,'arm':args.arm,
                         'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':len(ids),
                         'generated_tokens':len(generated),'token_limit':len(generated)>=PROTOCOL['max_new_tokens'],
                         'raw_output':raw,'parsed':parsed,'expected':gold,'schema_error':error,
                         'total_correct':parsed is not None and parsed['c']==row['expected']['c'],
                         'all_fields_correct':parsed==gold,'seconds':time.perf_counter()-start}
                    recs.append(rec);parent.emit(phase='state_eval',**rec)
        parent.save(out/f'evaluation-{step}.json',recs)
        rt.model.set_output_embeddings(capture);rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.train()
        return recs
    baseline=evaluate(0)
    opt=torch.optim.AdamW(ad.factors.parameters(),lr=PROTOCOL['lr'],weight_decay=0)
    rt.model.enable_input_require_grads()
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for n,m in rt.model.named_modules() if 'language_model' in n):raise RuntimeError('Unexpected training dropout')
    def loss_for(row):
        e=allenc[(row['id'],args.arm)];x=torch.tensor([e['ids']]);capture.hidden=None
        rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=0)
        h=capture.hidden
        if h is None or h.shape[:2]!=x.shape or not h.requires_grad:raise RuntimeError('Detached process supervision')
        hs=h[0,e['prefix']-1:-1];ys=x[0,e['prefix']:]
        if len(ys)!=e['target_count'] or len(hs)!=len(ys):raise RuntimeError('Wrong causal loss alignment')
        loss=h.sum()*0
        for i in range(0,len(ys),4):
            z=F.linear(hs[i:i+4].to(native.weight.dtype),native.weight,native.bias).float()
            loss=loss+F.cross_entropy(z,ys[i:i+4],reduction='sum')
        return loss/len(ys)
    history=[];sequence=[];rng=random.Random(PROTOCOL['seed'])
    for epoch in range(12):
        indices=list(range(6));rng.shuffle(indices);sequence+=indices
    assert len(sequence)==PROTOCOL['updates']*PROTOCOL['accumulation']
    parent.save(out/'initialization.json',{'sha256':init,'order':[rows[i]['id'] for i in sequence],
             'trainable_parameters':sum(p.numel() for p in ad.factors.parameters()),
             'target_tokens_total':sum(allenc[(rows[i]['id'],args.arm)]['target_count'] for i in sequence),
             'input_tokens_total':sum(allenc[(rows[i]['id'],args.arm)]['prefix'] for i in sequence)})
    for step in range(1,PROTOCOL['updates']+1):
        tick=time.perf_counter();opt.zero_grad(set_to_none=True);s=0.
        for i in sequence[(step-1)*2:step*2]:
            loss=loss_for(rows[i])/2
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            s+=float(loss.detach());loss.backward();capture.hidden=None
        norms=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
        if any(g is None or not math.isfinite(g) for g in norms):raise RuntimeError('Invalid layer gradients')
        if any(p.grad is not None for p in rt.model.parameters()):raise RuntimeError('Frozen base acquired gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1.));opt.step()
        save_file(ad.state(),str(out/'recovery.safetensors'))
        tmp=out/'optimizer.tmp';torch.save({'step':step,'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'protocol':PROTOCOL},tmp);tmp.replace(out/'optimizer.pt')
        history.append({'step':step,'loss':s,'gradient_norm':norm,'B_gradient_norms':norms,'seconds':time.perf_counter()-tick,'peak_rss_mib':parent.memory()})
        parent.save(out/'history.json',history);parent.emit(phase='state_train',arm=args.arm,**history[-1])
    final=out/'adapter.safetensors';save_file(ad.state(),str(final))
    rt.model.disable_input_require_grads();after=evaluate(PROTOCOL['updates'])
    saved=load_file(str(final));now=ad.state()
    if set(saved)!=set(now) or any(not torch.equal(saved[k],now[k]) for k in now):raise RuntimeError('Saved checkpoint mismatch')
    summary={}
    for step,rs in [(0,baseline),(PROTOCOL['updates'],after)]:
        summary[str(step)]={}
        for split in ('train','held'):
            sub=[r for r in rs if r['split']==split]
            summary[str(step)][split]={'n':len(sub),'total_correct':sum(r['total_correct'] for r in sub),
              'all_fields_correct':sum(r['all_fields_correct'] for r in sub),
              'triplets_all_total_correct':sum(all(r['total_correct'] for r in sub if r['source']==src) for src in {r['source'] for r in sub})}
    parent.save(out/'receipt.json',{'arm':args.arm,'updates':36,'initial_sha256':init,'checkpoint_sha256':sha(final),'summary':summary,
        'changed_blocks':sum(bool(f.b.abs().sum()>0) for f in ad.factors),'benchmark_calls':0,'official_score':None,
        'new_knowledge_claim':'none beyond this bounded supervised fit and free-running test'})
    parent.emit(phase='state_completed',arm=args.arm,summary=summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=PROTOCOL['arms'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    if args.check_only:print(json.dumps(checks()));raise SystemExit(0)
    try:run(args)
    except Exception as ex:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True);parent.save(p/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc(),'final_result_valid':False});raise
