"""Matched 2x2 causal training pilot. No relationship loss, generation or oracle notes.
Both task gradients are computed in every arm; event weight stays one half.
Complete CPU checkpoints make restart exact within the fixed runtime contract.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,time,traceback
from pathlib import Path
from fractions import Fraction
import causal_v17_data as data
import learn_v16 as prior

def hfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def snapshot(torch,factors,optimizer):
 import causal_v17_state as ck
 return (copy.deepcopy(factors.state_dict()),copy.deepcopy(optimizer.state_dict()),ck.random_state())
def restore(factors,optimizer,state):
 import causal_v17_state as ck
 factors.load_state_dict(state[0]);optimizer.load_state_dict(state[1]);ck.restore_random(state[2])
def update(torch,factors,optimizer,g,cap=1.):
 optimizer.zero_grad(set_to_none=True)
 for p,v in zip(factors.parameters(),g):p.grad=v.clone()
 norm=float(torch.nn.utils.clip_grad_norm_(factors.parameters(),cap));optimizer.step()
 return norm

def run(root,out,seed,arm,stop):
 import torch,numpy as np
 from safetensors.torch import save_file
 from reconstruct_v1 import Runtime
 import measure_v14
 import causal_v17_state as ck
 root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
 r=json.loads((root/'causal-prepared/records.json').read_text());m=json.loads((root/'causal-prepared/manifest.json').read_text())
 if data.digest(r)!=m['record_hash'] or seed not in data.SEEDS or arm not in data.ARMS:raise ValueError('Frozen data/arm mismatch')
 modal_weight,lr=data.ARMS[arm];schedule=m['schedules'][str(seed)]
 if stop not in (16,32):raise ValueError('Only fixed segment endpoints supported')
 np.random.seed(seed);random.seed(seed)
 rt=Runtime(root/'reconstruction-inputs');model=rt.model;fixture=rt.check()
 original_parameters=list(model.parameters());model.requires_grad_(False);model.eval()
 if any(isinstance(x,torch.nn.Dropout) and x.p>0 for x in model.modules()):raise RuntimeError('Expected dropout-free pinned backbone')
 use=set(m['evaluation_indices']+m['probe_indices']+[i for pair in schedule for i in pair])
 cache={}
 for i in sorted(use):
  row=r[i]['input'];msg=measure_v14.arm_messages(row,'','semantic_codes')
  text=rt.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=rt.tokenizer.encode(text,add_special_tokens=False);codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(row['labels']))]
  if len(ids)>768 or any(len(x)!=1 for x in codes):raise ValueError('Prompt/code bound')
  for j,c in enumerate(codes):
   if rt.tokenizer.encode(text+chr(65+j),add_special_tokens=False)!=ids+c:raise ValueError('Boundary mismatch')
  cache[i]={'ids':torch.tensor([ids]),'mask':torch.ones((1,len(ids)),dtype=torch.long),'codes':[x[0] for x in codes],
            'hash':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(ids)}
  if r[i]['target'] is None:raise ValueError('Missing target')
 targets,factors,hooks=prior.make_factors(torch,model,seed)
 params=list(factors.parameters());optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 initial_hash=prior.tensors_digest(factors.state_dict())
 bindings={'model_revision':prior.filehash(root/'reconstruct_v1.py'),'trainer':hfile(__file__),'data':m['record_hash'],
           'seed':seed,'arm':arm,'schedule':data.digest(schedule),'runtime_model_revision':rt.receipt['revision']}
 calls={'training':0,'evaluation':0,'counterfactual':0,'integrity':0}
 def score(i,phase='evaluation'):
  c=cache[i];rt.head.codes=c['codes'];calls[phase]+=1
  z=model(input_ids=c['ids'],attention_mask=c['mask'],logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
  if not torch.isfinite(z).all() or z.numel()!=len(r[i]['target']):raise RuntimeError('Bad logits')
  return z
 def evaluate(indices,path,step):
  model.eval();rows=[]
  with torch.inference_mode():
   for n,i in enumerate(indices):
    tick=time.perf_counter();z=score(i);p=torch.softmax(z,-1)
    row={'id':r[i]['input']['id'],'record_index':i,'step':step,'seed':seed,'arm':arm,'logits':z.tolist(),'probabilities':p.tolist(),
         'prompt_sha256':cache[i]['hash'],'input_sha256':data.digest(r[i]['input']),'tokens':cache[i]['tokens'],'seconds':time.perf_counter()-tick}
    rows.append(row)
  data.write(path,rows);model.train();return rows
 with torch.inference_mode():initial=score(m['probe_indices'][0],'integrity').clone()
 if (out/'full-state/latest.json').exists():
  load=ck.load(out/'full-state',factors,optimizer,bindings=bindings);completed=load['completed_steps']
  if load['schedule_state']['next_index']!=completed or completed>stop:raise ValueError('Resume progress mismatch')
 else:
  completed=0
  evaluate(m['probe_indices'],out/'probe-step-0.json',0)
  pre={'seed':seed,'arm':arm,'lr':lr,'modal_weight':modal_weight,'event_weight':.5,'parameters':sum(p.numel() for p in params),
       'initial_sha256':initial_hash,'runtime':rt.receipt,'fixture':fixture,'max_tokens':max(c['tokens'] for c in cache.values()),
       'bindings':bindings,'schedule':schedule,'frozen_base_gradients':True}
  data.write(out/'preflight.json',pre)
  ck.save(out/'full-state',factors,optimizer,completed_steps=0,bindings=bindings,schedule_state={'next_index':0})
 if not 0<=completed<=stop:raise ValueError('Unexpected resume boundary')
 if completed and (out/f'probe-step-{completed}.json').exists():
  expected=json.loads((out/f'probe-step-{completed}.json').read_text())[0]['logits']
  with torch.inference_mode():actual=score(m['probe_indices'][0],'integrity')
  if float((actual-torch.tensor(expected)).abs().max())>1e-4:raise RuntimeError('Cross-process resumed logits changed')
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
 train_start=time.perf_counter()
 for step in range(completed+1,stop+1):
  pair=schedule[step-1]
  if [r[i]['question_type'] for i in pair]!=['event','mode'] or any(r[i]['split']!='fit' for i in pair):raise ValueError('Training population leakage')
  tick=time.perf_counter();grads=[];losses=[];observed=[]
  for i in pair:
   optimizer.zero_grad(set_to_none=True);q=torch.tensor([float(Fraction(x)) for x in r[i]['target']],dtype=torch.float32)
   z=score(i,'training');loss=-(q*torch.log_softmax(z,-1)).sum();loss.backward()
   if any(p.grad is not None for p in original_parameters):raise RuntimeError('Original weight received gradient')
   gs=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
   if any(not torch.isfinite(g).all() for g in gs):raise RuntimeError('Nonfinite task gradient')
   grads.append(gs);losses.append(float(loss.detach()));observed.append(z.detach().tolist())
  e,mgrad=grads
  dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(e,mgrad))
  en=math.sqrt(sum(float((a.double()**2).sum()) for a in e));mn=math.sqrt(sum(float((a.double()**2).sum()) for a in mgrad))
  combined=[.5*a+.5*modal_weight*b for a,b in zip(e,mgrad)]
  counterfactuals=None
  if step in (1,16,32):
   state=snapshot(torch,factors,optimizer);counterfactuals={}
   for label,g in [('event_update',[.5*a for a in e]),('modal_update',[.5*a for a in mgrad])]:
    restore(factors,optimizer,state);norm=update(torch,factors,optimizer,g)
    with torch.no_grad():zz=[score(i,'counterfactual') for i in pair]
    after=[float(-(torch.tensor([float(Fraction(x)) for x in r[i]['target']])*torch.log_softmax(z,-1)).sum()) for i,z in zip(pair,zz)]
    counterfactuals[label]={'losses_before':losses,'losses_after':after,'gradient_norm':norm}
   restore(factors,optimizer,state)
  norm=update(torch,factors,optimizer,combined)
  if any(not torch.isfinite(p).all() for p in params):raise RuntimeError('Nonfinite updated weights')
  cp=ck.save(out/'full-state',factors,optimizer,completed_steps=step,bindings=bindings,schedule_state={'next_index':step})
  for old in (out/'full-state').glob('step-*.pt'):
   if old.name!=cp['file']:old.unlink()
  row={'step':step,'pair':pair,'world':r[pair[0]]['group'],'event_ce':losses[0],'mode_ce':losses[1],
       'weighted_loss':.5*(losses[0]+modal_weight*losses[1]),'event_grad_norm':en,'mode_grad_norm':mn,
       'gradient_cosine':dot/(en*mn) if en*mn>0 else None,'combined_preclip_norm':norm,'counterfactuals':counterfactuals,
       'observed_logits':observed,'checkpoint_sha256':cp['sha256'],'seconds':time.perf_counter()-tick}
  append(out/'training.jsonl',row)
  if step in (1,8,16,32):
   save_file({k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()},str(out/f'adapter-step-{step}.safetensors'))
   evaluate(m['probe_indices'],out/f'probe-step-{step}.json',step)
  if step%4==0:print(json.dumps({'seed':seed,'arm':arm,'step':step,'stop':stop,'ce':losses,'seconds':row['seconds']}),flush=True)
 saved_hash=prior.tensors_digest(factors.state_dict());opt_state=copy.deepcopy(optimizer.state_dict())
 with torch.no_grad():before=score(m['probe_indices'][0],'integrity').clone();factors[0].b.add_(.001)
 restored=ck.load(out/'full-state',factors,optimizer,bindings=bindings)
 if prior.tensors_digest(factors.state_dict())!=saved_hash or restored['completed_steps']!=stop:raise RuntimeError('State roundtrip failure')
 with torch.no_grad():after=score(m['probe_indices'][0],'integrity')
 err=float((before-after).abs().max())
 if err>1e-4:raise RuntimeError('Model roundtrip output mismatch')
 model.gradient_checkpointing_disable();model.disable_input_require_grads();model.eval()
 if stop==32:
  evaluate(m['evaluation_indices'],out/'final-predictions.json',stop)
  for f in factors:f.enabled=False
  with torch.no_grad():base=score(m['probe_indices'][0],'integrity')
  restore_error=float((base-initial).abs().max())
  if restore_error>1e-4:raise RuntimeError('Base restoration mismatch')
  for f in factors:f.enabled=True
 else:restore_error=None
 data.write(out/f'segment-{stop}.json',{'complete':True,'from_step':completed,'through_step':stop,'seed':seed,'arm':arm,
           'new_training_updates':stop-completed,'counterfactual_updates_discarded':2*sum(1 for s in (1,16,32) if completed<s<=stop),
           'roundtrip_logit_error':err,'base_restoration_error':restore_error,'initial_tensor_sha256':initial_hash,
           'final_tensor_sha256':saved_hash,'calls':calls,'bindings':bindings,'seconds_excluding_load':time.perf_counter()-train_start,
           'worlds_presented_total':stop,'readout':'unmodified semantic_codes, T=1','scope':'fixed 32-step same-grammar causal pilot, not converged training'})
 for hook in hooks:hook.remove()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=list(data.ARMS),required=True);p.add_argument('--stop',type=int,choices=[16,32],required=True);a=p.parse_args()
 try:run(a.root,a.out,a.seed,a.arm,a.stop)
 except Exception as e:
  Path(a.out).mkdir(parents=True,exist_ok=True);data.write(Path(a.out)/'FAILED.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise
