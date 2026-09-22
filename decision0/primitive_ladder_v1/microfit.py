"""Learnability diagnostic, intentionally three seen arithmetic examples.

This is not a benchmark candidate. Fixed 32 optimizer updates, no early stopping,
no label-dependent checkpoint selection. No benchmark data are loaded.
"""
from __future__ import annotations
import argparse, copy, itertools, json, math, random, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent

PROTOCOL={'id':'decision0-execution-microfit-v1','seed':94231,'updates':32,'rank':8,
 'learning_rate':3e-4,'accumulation':3,'training_examples':3,'epochs_equivalent':32,'outcome_orders':'all six cycled during training',
 'check_steps':[0,1,2,4,8,16,32],'layers':'all 32 FFN down projections',
 'selection':'fixed final; all learning-curve checkpoints are diagnostic only',
 'purpose':'Test fit on three seen problems, not arithmetic generalization.',
 'input':'single-copy unchanged decision interface','benchmark_calls':0,
 'base_model':'Qwen/Qwen3.5-4B','base_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}

def examples(split):
    params=(4,121,151,27) if split=='seen' else (7,143,127,35)
    n,p,f,c=params;center=n*p+f-c
    options=[{'id':f'value{i}','description':f'The exact result is {center+d} cents.'} for i,d in enumerate([-1,0,1])]
    # Fixed nonnumeric order. Exact references never enter the model input.
    options=[options[2],options[0],options[1]]
    return [{'id':f'{split}-{i}','state':f'Quantity {n}. Unit price {p} cents. Handling fee {f+d} cents. Credit {c} cents.',
      'question':'Compute quantity times unit price, plus handling fee, minus credit. Select the exact result in cents.',
      'options':copy.deepcopy(options),'expected':f'value{i}','result':n*p+f+d-c}
      for i,d in enumerate([-1,0,1])]

def run(args):
    import torch
    from safetensors.torch import save_file,load_file
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);parent.save(out/'protocol.json',PROTOCOL)
    rows=examples('seen');held=examples('unseen');rt=parent.Runtime();parent.save(out/'runtime.json',rt.meta)
    rt.hook.remove()
    def capture(m,ins):rt.hidden=ins[0][:,-1,:] if ins[0].ndim==3 else ins[0]
    rt.hook=rt.head.register_forward_pre_hook(capture)
    perms=list(itertools.permutations(range(3)));views=[];enc=[]
    for r in rows:
      vv=[];ee=[]
      for perm in perms:
        v=copy.deepcopy(r);v['options']=[r['options'][j] for j in perm];vv.append(v);ee.append(parent.encode_checked(rt.tokenizer,v,'baseline',512))
      views.append(vv);enc.append(ee)
    before=[rt.score(r,'baseline')['logits'] for r in rows];ad=parent.Adapter(rt,PROTOCOL['seed'])
    init_hash=parent.tensor_hash(ad.state());parent.save(out/'inputs.json',{'seen':rows,'unseen':held,'tokens':[[e['input_tokens'] for e in es] for es in enc],'initial_hash':init_hash})
    opt=torch.optim.AdamW(ad.factors.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=0)
    rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads()
    curve=[];logs=[]
    def evaluate(step):
      rt.model.eval();records=[]
      for split,rs in [('seen',rows),('unseen',held)]:
        for r in rs:
          for order in ('original','reversed'):
            q=copy.deepcopy(r)
            if order=='reversed':q['options'].reverse()
            z=rt.score(q,'baseline');z.update({'step':step,'split':split,'order':order,'expected':r['expected'],'correct':z['predicted']==r['expected']});records.append(z)
      curve.extend(records);parent.save(out/'learning_curve.json',curve)
      parent.emit(phase='microfit_curve',step=step,counts={f'{s}/{o}':sum(r['correct'] for r in records if r['split']==s and r['order']==o) for s in ('seen','unseen') for o in ('original','reversed')})
      rt.model.train()
    evaluate(0);begin=time.perf_counter()
    for update in range(1,33):
      opt.zero_grad(set_to_none=True);lossval=0.;tick=time.perf_counter();order=[0,1,2];random.Random(PROTOCOL['seed']+update).shuffle(order)
      for i in order:
        vi=(update-1+i)%6;r=views[i][vi];e=enc[i][vi];x=torch.tensor([e['input_ids']]);rt.hidden=None
        raw=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
        slots=e['answer_token_ids'];w=rt.head.weight[slots].float();b=rt.head.bias[slots].float() if rt.head.bias is not None else None
        z=torch.nn.functional.linear(rt.hidden.float(),w,b)[0];y=[o['id'] for o in r['options']].index(r['expected'])
        loss=-torch.log_softmax(z,-1)[y]/3
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        lossval+=float(loss.detach());loss.backward();rt.hidden=None;del raw,z,x,loss
      grad=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
      if any(g is None or not math.isfinite(g) for g in grad):raise RuntimeError('Invalid gradient')
      norm=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1));opt.step()
      record={'step':update,'ce_before_update':lossval,'gradient_norm':norm,'block_B_gradients':grad,'seconds':time.perf_counter()-tick,'peak_rss_mib':parent.memory()};logs.append(record);parent.save(out/'progress.json',logs);parent.emit(phase='microfit_update',**record)
      if update in PROTOCOL['check_steps']:evaluate(update)
      if update%8==0:save_file(ad.state(),str(out/'recovery.safetensors'))
    rt.model.eval();rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads()
    save_file(ad.state(),str(out/'adapter.safetensors'));ad.enabled=False
    restore=max(abs(a-b) for r,old in zip(rows,before) for a,b in zip(rt.score(r,'baseline')['logits'],old))
    ad.enabled=True;last=rt.score(rows[0],'baseline')['logits']
    with torch.no_grad():
      for f in ad.factors:f.b.zero_()
    ad.factors.load_state_dict(load_file(str(out/'adapter.safetensors')));reload=max(abs(a-b) for a,b in zip(last,rt.score(rows[0],'baseline')['logits']))
    if max(restore,reload)>1e-4:raise RuntimeError('Restoration mismatch')
    (out/'recovery.safetensors').unlink(missing_ok=True)
    parent.save(out/'receipt.json',{'updates':32,'train_presentations':96,'curve_records':len(curve),'initial_hash':init_hash,'adapter_sha256':parent.filehash(out/'adapter.safetensors'),'restoration_error':restore,'reload_error':reload,'elapsed_seconds':time.perf_counter()-begin,'updated_blocks':sum(bool(f.b.abs().sum()) for f in ad.factors),'benchmark_score':None,'release_candidate':False})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
    try:run(args)
    except Exception as exc:
      out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'FAILED.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}));raise
