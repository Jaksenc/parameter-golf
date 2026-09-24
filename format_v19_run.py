"""Matched direct adaptation; diagnostics never take trial optimizer steps."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback
from fractions import Fraction
from pathlib import Path
import format_v19_data as data
import learn_v16 as parent

LR=1e-4
MAX_TOKENS=1536

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(p,r):
 with Path(p).open('a') as f:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def lock(root):
 m=read(root/'format-prepared/manifest.json');r=read(root/'format-prepared/records.json')
 if data.digest(r)!=m['records_hash']:raise ValueError('Data changed')
 for n,h in read(root/'FORMAT_V19_SOURCE_LOCK.json').items():
  if sha(root/n)!=h:raise ValueError('Source changed: '+n)
 return r,m

def encoding(rt,row):
 import measure_v14
 torch=rt.torch
 text=rt.tokenizer.apply_chat_template(measure_v14.arm_messages(row,'','semantic_codes'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=rt.tokenizer.encode(text,add_special_tokens=False)
 cs=[rt.tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(len(row['labels']))]
 if len(ids)>MAX_TOKENS or any(len(c)!=1 for c in cs) or len({c[0] for c in cs})!=len(cs):raise ValueError('Input/code bound')
 for i,c in enumerate(cs):
  if rt.tokenizer.encode(text+chr(65+i),add_special_tokens=False)!=ids+c:raise ValueError('Code boundary')
 return {'ids':torch.tensor([ids]),'mask':torch.ones((1,len(ids)),dtype=torch.long),'codes':[c[0] for c in cs],
         'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(ids)}

def forward(rt,c):
 rt.head.codes=c['codes']
 z=rt.model(input_ids=c['ids'],attention_mask=c['mask'],logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
 if len(z)!=len(c['codes']) or not rt.torch.isfinite(z).all():raise ValueError('Invalid logits')
 return z

def anchor(rt,root,number):
 row=read(root/'transfer-prepared/anchors.json')[number]
 c=encoding(rt,row['input'])
 with rt.torch.inference_mode():z=forward(rt,c)
 err=float((z-rt.torch.tensor(row['record']['logits'])).abs().max())
 if err>1e-4 or c['prompt_sha256']!=row['record']['prompt_sha256']:raise ValueError('Anchor mismatch')
 return {'id':row['input']['id'],'max_error':err,'prompt_sha256':c['prompt_sha256']}

def train(root,out,seed,arm,stop):
 import torch,numpy as np,random
 from safetensors.torch import save_file
 from reconstruct_v1 import Runtime
 import causal_v17_state as ck
 root,out=Path(root),Path(out);r,m=lock(root)
 if seed not in data.SEEDS or arm not in data.ARMS or stop not in (16,32):raise ValueError('Run contract')
 schedule=m['schedules'][str(seed)][arm];random.seed(seed);np.random.seed(seed)
 rt=Runtime(root/'reconstruction-inputs');fixture=rt.check();ref=anchor(rt,root,0)
 baseparams=list(rt.model.parameters());rt.model.requires_grad_(False);rt.model.eval()
 if any(isinstance(x,torch.nn.Dropout) and x.p>0 for x in rt.model.modules()):raise ValueError('Nonzero dropout unsupported')
 cache={i:encoding(rt,r[i]['input']) for i in set(schedule+m['probe_indices'])}
 first=m['probe_indices'][0]
 with torch.inference_mode():baseprobe=forward(rt,cache[first]).clone()
 _,factors,hooks=parent.make_factors(torch,rt.model,seed);initial=parent.tensors_digest(factors.state_dict())
 with torch.inference_mode():zeroerr=float((forward(rt,cache[first])-baseprobe).abs().max())
 if zeroerr>1e-4:raise ValueError('Zero factors change output')
 optimizer=torch.optim.AdamW(factors.parameters(),lr=LR,weight_decay=.01)
 bindings={'source':sha(__file__),'data':m['records_hash'],'schedule':data.digest(schedule),'seed':seed,'arm':arm,
           'runtime_revision':rt.receipt['revision'],'lr':LR,'steps':32}
 out.mkdir(parents=True,exist_ok=True);history=[];start=0
 if (out/'full-state/latest.json').exists():
  restored=ck.load(out/'full-state',factors,optimizer,bindings=bindings);start=restored['completed_steps'];history=restored['schedule_state']['history']
  if len(history)!=start or restored['schedule_state']['next_index']!=start:raise ValueError('Resume ledger mismatch')
  (out/'training.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in history))
 else:
  data.write(out/'preflight.json',{'seed':seed,'arm':arm,'source':sha(__file__),'initial_adapter_sha256':initial,
    'parameters':sum(p.numel() for p in factors.parameters()),'runtime':rt.receipt,'fixture':fixture,'anchor':ref,'zero_error':zeroerr,'bindings':bindings,'schedule':schedule})
 if start>stop:raise ValueError('Cannot rewind')
 def probes(step):
  rt.model.eval();rows=[]
  with torch.inference_mode():
   for i in m['probe_indices']:
    z=forward(rt,cache[i]);rows.append({'id':r[i]['input']['id'],'index':i,'logits':z.tolist(),'probabilities':torch.softmax(z,-1).tolist(),'step':step})
  data.write(out/f'probes-{step}.json',rows)
 if start==0:probes(0)
 else:
  p=read(out/f'probes-{start}.json')[0]
  with torch.inference_mode():err=float((forward(rt,cache[first])-torch.tensor(p['logits'])).abs().max())
  if err>1e-4:raise ValueError('Cross-process resume logit mismatch')
 rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads();rt.model.train()
 for step in range(start+1,stop+1):
  i=schedule[step-1];item=r[i]
  if item['split']!='fit' or (arm=='single' and item['format']!='prose') or item['format'] not in data.TRAIN_FORMATS:raise ValueError('Fit scope')
  tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
  z=forward(rt,cache[i]);q=torch.tensor([float(Fraction(x)) for x in item['target']],dtype=torch.float32)
  ce=-(q*torch.log_softmax(z,-1)).sum();(.5*ce).backward()
  if any(p.grad is not None for p in baseparams):raise ValueError('Backbone received gradient')
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in factors.parameters()):raise ValueError('Bad gradient')
  norm=float(torch.nn.utils.clip_grad_norm_(factors.parameters(),1.));optimizer.step()
  if any(int(optimizer.state[p]['step'])!=step for p in factors.parameters()):raise ValueError('Optimizer counter')
  if any(not torch.isfinite(p).all() for p in factors.parameters()):raise ValueError('Bad weights')
  row={'step':step,'index':i,'group':item['group'],'format':item['format'],'target':item['target'],'logits':z.detach().tolist(),
       'ce':float(ce.detach()),'weighted_loss':float(.5*ce.detach()),'preclip_norm':norm,'tokens':cache[i]['tokens'],'optimizer_step':step,'seconds':time.perf_counter()-tick}
  history.append(row)
  receipt=ck.save(out/'full-state',factors,optimizer,completed_steps=step,bindings=bindings,schedule_state={'next_index':step,'history':history})
  for p in (out/'full-state').glob('step-*.pt'):
   if p.name!=receipt['file']:p.unlink()
  append(out/'training.jsonl',row)
  if step in (8,16,32):
   save_file({k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()},str(out/f'adapter-{step}.safetensors'))
   probes(step);rt.model.train()
  if step%4==0:print(json.dumps({'seed':seed,'arm':arm,'step':step,'ce':row['ce']}),flush=True)
 saved=parent.tensors_digest(factors.state_dict())
 with torch.no_grad():factors[0].b.add_(.001)
 restored=ck.load(out/'full-state',factors,optimizer,bindings=bindings)
 if saved!=parent.tensors_digest(factors.state_dict()) or restored['completed_steps']!=stop:raise ValueError('Checkpoint roundtrip')
 rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads();rt.model.eval()
 for f in factors:f.enabled=False
 with torch.inference_mode():restoreerr=float((forward(rt,cache[first])-baseprobe).abs().max())
 if restoreerr>1e-4:raise ValueError('Base restoration')
 for f in factors:f.enabled=True
 data.write(out/f'complete-{stop}.json',{'complete':True,'seed':seed,'arm':arm,'from_step':start,'through_step':stop,
  'adapter_sha256':sha(out/f'adapter-{stop}.safetensors'),'tensor_hash':saved,'source_sha256':sha(__file__),'data_hash':m['records_hash'],
  'initial_adapter_sha256':initial,'base_restoration_error':restoreerr,'real_updates':stop-start,
  'training_tokens':sum(x['tokens'] for x in history),'history_hash':data.digest(history),'format_counts':dict(__import__('collections').Counter(x['format'] for x in history))})
 for h in hooks:h.remove()

def evaluate(root,out,shard):
 import torch
 from safetensors.torch import load_file
 from reconstruct_v1 import Runtime
 root,out=Path(root),Path(out);r,m=lock(root)
 if shard not in range(16):raise ValueError('Shard')
 ids=m['evaluation_indices'][shard::16];rt=Runtime(root/'reconstruction-inputs');fixture=rt.check();ref=anchor(rt,root,shard)
 rt.model.requires_grad_(False);rt.model.eval()
 cache={i:encoding(rt,r[i]['input']) for i in ids}
 _,factors,hooks=parent.make_factors(torch,rt.model,data.SEEDS[0]);factors.requires_grad_(False)
 states={};hashes={}
 for seed in data.SEEDS:
  for arm in data.ARMS:
   key=f'{seed}-{arm}';d=root/'models'/f'format-v19-model-{key}';c=read(d/'complete-32.json');p=d/'adapter-32.safetensors'
   if c['through_step']!=32 or c['data_hash']!=m['records_hash'] or c['adapter_sha256']!=sha(p):raise ValueError('Incomplete/checkpoint mismatch')
   states[key]=load_file(str(p));hashes[key]=sha(p)
 out.mkdir(parents=True,exist_ok=False);path=out/'records.jsonl';path.write_text('')
 data.write(out/'preflight.json',{'runtime':rt.receipt,'fixture':fixture,'anchor':ref,'checkpoints':hashes,'data_hash':m['records_hash']})
 for n,i in enumerate(ids):
  outputs={}
  for f in factors:f.enabled=False
  for key in ['unchanged']+list(states):
   if key!='unchanged':
    factors.load_state_dict(states[key],strict=True)
    for f in factors:f.enabled=True
   tick=time.perf_counter()
   with torch.inference_mode():z=forward(rt,cache[i])
   outputs[key]={'logits':z.tolist(),'probabilities':torch.softmax(z,-1).tolist(),'seconds':time.perf_counter()-tick}
  for f in factors:f.enabled=False
  with torch.inference_mode():err=float((forward(rt,cache[i])-torch.tensor(outputs['unchanged']['logits'])).abs().max())
  if err>1e-4:raise ValueError('Baseline restoration')
  append(path,{'id':r[i]['input']['id'],'index':i,'input_sha256':data.digest(r[i]['input']),'prompt_sha256':cache[i]['prompt_sha256'],
               'tokens':cache[i]['tokens'],'code_ids':cache[i]['codes'],'outputs':outputs,'restoration_error':err})
  if n%4==0:print(json.dumps({'shard':shard,'done':n+1,'planned':len(ids)}),flush=True)
 data.write(out/'complete.json',{'complete':True,'count':len(ids),'indices':ids,'records_sha256':sha(path),'data_hash':m['records_hash'],'models':list(states)})
 for h in hooks:h.remove()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--root',default='.');p.add_argument('--out',required=True)
 p.add_argument('--seed',type=int);p.add_argument('--arm',choices=data.ARMS);p.add_argument('--stop',type=int,choices=[16,32],default=32);p.add_argument('--shard',type=int)
 a=p.parse_args()
 try:
  if a.mode=='train':train(a.root,a.out,a.seed,a.arm,a.stop)
  else:evaluate(a.root,a.out,a.shard)
 except Exception as exc:
  Path(a.out).mkdir(parents=True,exist_ok=True);data.write(Path(a.out)/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
