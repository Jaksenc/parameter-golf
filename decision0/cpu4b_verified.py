from __future__ import annotations
import argparse, hashlib, json, math, os, random, time, gc
from pathlib import Path

MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
EXPECTED={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
SYSTEM='Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.'
SEED=47; RANK=8; SCALE=2.0; MAX_TOKENS=8192

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def jhash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False))
def read_jsonl(path): return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def emit(kind,**kw): print(json.dumps({'kind':kind,**kw},sort_keys=True,allow_nan=False),flush=True)

def sem_row_from_bench(r):
 q=r['question']; typ=q['type']; crit=q.get('criteria'); opts=[]
 if typ=='noul':
  for key,label in [('true','yes'),('false','no')]: opts.append({'id':label,'description':key+': '+(crit or {}).get(key,f'The proposition is {key}.')})
 elif typ=='choice':
  for k,v in crit.items(): opts.append({'id':k,'description':k+': '+(v or k)})
 elif typ=='score':
  for i,v in enumerate(crit): opts.append({'id':str(i),'description':str(i)+': '+str(v)})
 else: raise ValueError(typ)
 return {'id':r['id'],'state':r['state'],'question':q['instructions'],'options':opts}

def direct_messages(row):
 payload={'evidence':row['state'],'criterion':row['question'],'options':[{'letter':chr(65+i),'description':o['description']} for i,o in enumerate(row['options'])]}
 return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]

class RT:
 def __init__(self):
  import torch, transformers
  from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
  from huggingface_hub import snapshot_download
  self.torch=torch; torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
  self.snap=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=4)
  hashes={p.name:sha(p) for p in Path(self.snap).glob('*.safetensors')}
  if hashes!=EXPECTED: raise RuntimeError('base hash mismatch '+str(hashes))
  self.tok=AutoTokenizer.from_pretrained(self.snap,local_files_only=True,trust_remote_code=False)
  self.model,info=Qwen3_5ForConditionalGeneration.from_pretrained(self.snap,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
  issues={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
  if issues: raise RuntimeError(str(issues))
  self.model.eval(); self.head=self.model.get_output_embeddings(); self.cap={}
  names=[n for n,m in self.model.named_modules() if n.endswith('.layers.31.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
  if len(names)!=1: raise RuntimeError('terminal layer mismatch '+str(names))
  self.layer_name=names[0]; self.down=self.model.get_submodule(self.layer_name); layer=self.model.get_submodule(self.layer_name.rsplit('.mlp.',1)[0]); body=self.model.get_submodule(self.layer_name.split('.layers.')[0]); self.norm=body.norm
  def pre_r(m,args): self.cap['r']=args[0][:,-1,:].detach().clone()
  def pre_x(m,args): self.cap['x']=args[0][:,-1,:].detach().clone()
  def post_y(m,args,y): self.cap['y']=y[:,-1,:].detach().clone()
  def pre_n(m,args): self.cap['norm_in']=args[0][:,-1,:].detach().clone()
  self.hooks=[layer.post_attention_layernorm.register_forward_pre_hook(pre_r),self.down.register_forward_pre_hook(pre_x),self.down.register_forward_hook(post_y),self.norm.register_forward_pre_hook(pre_n)]
  self.meta={'model':MODEL,'revision':REV,'hashes':hashes,'torch':torch.__version__,'transformers':transformers.__version__,'layer':self.layer_name,'dtype':'bf16 backbone'}
 def encode(self,row):
  text=self.tok.apply_chat_template(direct_messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=self.tok.encode(text,add_special_tokens=False)
  if not ids or len(ids)>MAX_TOKENS: raise ValueError(f"{row['id']} tokens {len(ids)}")
  slots=[]
  for i in range(len(row['options'])):
   letter=chr(65+i); t=self.tok.encode(letter,add_special_tokens=False)
   if len(t)!=1 or self.tok.encode(text+letter,add_special_tokens=False)!=ids+t: raise ValueError('boundary '+row['id'])
   slots.append(t[0])
  return {'input_ids':self.torch.tensor([ids]),'attention_mask':self.torch.ones((1,len(ids)),dtype=self.torch.long)},slots,len(ids),hashlib.sha256(text.encode()).hexdigest()
 def cache(self,row):
  import inspect, torch
  enc,slots,n,ph=self.encode(row); self.cap.clear(); args=dict(enc,use_cache=False,return_dict=True); sig=inspect.signature(self.model.forward).parameters
  if 'logits_to_keep' in sig: args['logits_to_keep']=1
  with torch.no_grad(): out=self.model(**args).logits[:, -1, :]
  if not torch.equal(self.cap['r']+self.cap['y'],self.cap['norm_in']): raise RuntimeError('residual mismatch')
  w=self.head.weight[slots].float().detach().cpu(); b=self.head.bias[slots].float().detach().cpu() if self.head.bias is not None else None
  z_native=out[0,slots].float().detach().cpu()
  h0=self.norm(self.cap['norm_in']).float().detach(); z_fp32=torch.nn.functional.linear(h0,w,b)[0].cpu()
  return {'id':row['id'],'tokens':n,'prompt':ph,'slots':slots,'x':self.cap['x'].cpu(),'r':self.cap['r'].cpu(),'y':self.cap['y'].cpu(),'w':w,'b':b,'native':z_native,'fp32':z_fp32}
 def close(self):
  for h in self.hooks:h.remove()

class Adapter:
 def __init__(self,torch,down,norm,hidden):
  self.torch=torch; self.norm=norm; self.hidden=hidden
  class M(torch.nn.Module):
   def __init__(self):
    super().__init__(); self.a=torch.nn.Parameter(torch.empty(RANK,down.in_features)); self.b=torch.nn.Parameter(torch.zeros(down.out_features,RANK)); torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    self.opa=torch.nn.Linear(hidden,64); self.opb=torch.nn.Linear(hidden,64); self.stage=torch.nn.Linear(hidden,4); self.support=torch.nn.Linear(hidden,8)
   def delta(self,x): return (torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)*SCALE).to(x.dtype)
  self.m=M()
 def logits_hidden(self,c):
  torch=self.torch; x=c['x']; r=c['r']; y=c['y']; h=self.norm(r+(y+self.m.delta(x))).float(); z=torch.nn.functional.linear(h,c['w'],c['b'])[0]; return z,h
 def state(self): return {k:v.detach().cpu().contiguous() for k,v in self.m.state_dict().items()}
 def load(self,s): self.m.load_state_dict(s)

def ce_soft(torch,z,target):
 q=torch.tensor(target,dtype=torch.float32); return -(q*torch.nn.functional.log_softmax(z.float(),dim=-1)).sum()
def aux_loss(torch,m,h,a):
 terms=[]
 for key,head in [('operand_a',m.opa),('operand_b',m.opb),('stage',m.stage)]:
  y=a.get(key,-1)
  if isinstance(y,int) and y>=0: terms.append(torch.nn.functional.cross_entropy(head(h),torch.tensor([y])))
 if 'support_mask' in a:
  logits=m.support(h)[0]; valid=torch.tensor(a['support_valid'],dtype=torch.float32); y=torch.tensor(a['support_mask'],dtype=torch.float32)
  if valid.sum()>0: terms.append((torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*valid).sum()/valid.sum())
 return torch.stack(terms).mean() if terms else h.sum()*0

def predict(torch,adapter,c):
 with torch.no_grad(): z,_=adapter.logits_hidden(c); p=z.double().softmax(-1).tolist(); return z.tolist(),p
def eval_rows(torch,adapter,rows,caches,bench=False):
 out=[]
 for r,c in zip(rows,caches):
  z,p=predict(torch,adapter,c); pred=max(range(len(p)),key=p.__getitem__); rec={'id':r['id'],'probabilities':p,'logits':z,'pred_index':pred}
  if bench:
   opts=sem_row_from_bench(r)['options']; ids=[o['id'] for o in opts]; expected=r['expected']; rec.update({'tier':r['_tier'],'labels':ids,'predicted':ids[pred],'expected':expected,'correct':ids[pred]==expected,'family':r.get('family')})
  else:
   target=r['target_probs']; rec.update({'correct':pred==max(range(len(target)),key=target.__getitem__),'target_probs':target,'family':r['family']})
  out.append(rec)
 return out
def stats(rows,bench=False):
 d={'n':len(rows),'correct':sum(bool(r['correct']) for r in rows),'accuracy':sum(bool(r['correct']) for r in rows)/len(rows)}
 if bench: d['tiers']={t:{'n':sum(r['tier']==t for r in rows),'correct':sum(r['correct'] for r in rows if r['tier']==t)} for t in ('easy','standard','hard')}
 return d

def train_arm(torch,rt,train,trainc,dev,devc,arm,out):
 torch.manual_seed(SEED); ad=Adapter(torch,rt.down,rt.norm,rt.model.config.get_text_config().hidden_size); opt=torch.optim.AdamW(ad.m.parameters(),lr=5e-4,weight_decay=.01)
 rng=random.Random(SEED); best=None; best_key=None; history=[]
 for epoch in range(1,17):
  order=list(range(len(train))); rng.shuffle(order)
  for start in range(0,len(order),32):
   ids=order[start:start+32]; opt.zero_grad(set_to_none=True); loss=0
   for i in ids:
    r=train[i]; z,h=ad.logits_hidden(trainc[i]); target=r['target_probs']; hard=[0.]*len(target); hard[max(range(len(target)),key=target.__getitem__)]=1.
    main=ce_soft(torch,z,target if arm=='verified' else hard); extra=aux_loss(torch,ad.m,h,r.get('annotations',{})) if arm=='verified' else h.sum()*0
    loss=loss+(main+.2*extra)/len(ids)
   loss.backward(); gn=torch.nn.utils.clip_grad_norm_(ad.m.parameters(),1.); opt.step()
  preds=eval_rows(torch,ad,dev,devc); s=stats(preds); nll=sum(-math.log(max(p['probabilities'][max(range(len(p['target_probs'])),key=p['target_probs'].__getitem__)],1e-30)) for p in preds)/len(preds); key=(s['accuracy'],-nll)
  history.append({'epoch':epoch,**s,'nll':nll,'grad_norm':float(gn)}); emit('epoch',arm=arm,**history[-1])
  if best_key is None or key>best_key: best_key=key; best=ad.state()
 ad.load(best); import safetensors.torch as st; st.save_file(ad.state(),str(Path(out)/f'{arm}.safetensors')); dump(Path(out)/f'{arm}-history.json',history); return ad

def make_row(family,split,i,seed):
 import random, datetime
 rng=random.Random((seed+1)*1000003+i*97+sum(map(ord,family+split))); sid=f"{family[:2].upper()}-{split[:2]}-{i:04d}"; ann={'operand_a':-1,'operand_b':-1,'stage':0,'support_mask':[0]*8,'support_valid':[0]*8}
 if family=='money':
  count=rng.randint(2,15); price=rng.randint(15,80); fee=rng.randint(0,20); credit=rng.randint(0,20); total=count*price+fee-credit; inclusive=bool(i%2); delta=[-1,0,1][i%3]; limit=total+delta
  state=f"Record A: {sid} has {count} units at ${price/100:.2f} each. Record B: add fee ${fee/100:.2f}. Record C: subtract credit ${credit/100:.2f}. Record D: the limit is ${limit/100:.2f}. Record E concerns another order and is irrelevant."; question=f"For {sid}, compute units times unit price, add the fee, subtract the credit. Is the result {'at most' if inclusive else 'strictly less than'} the limit?"
  ok=total<=limit if inclusive else total<limit; opts=[('yes','The computed amount satisfies the stated comparison.'),('no','The computed amount does not satisfy the stated comparison.')]; targ=[1.,0.] if ok else [0.,1.]; ann.update(operand_a=count,operand_b=price,stage=0,support_mask=[1,1,1,1,0,0,0,0],support_valid=[1,1,1,1,1,0,0,0])
 elif family=='time':
  base=datetime.datetime(2026,1+(i%10),1+(i*3)%20,8+(i%8),15,tzinfo=datetime.timezone.utc); hours=rng.randint(12,120); event=base+datetime.timedelta(hours=hours); threshold=hours+[-1,0,1][i%3]; inclusive=bool(i%2)
  state=f"Window starts {base.isoformat()}. Event occurs {event.isoformat()}. An unrelated timestamp is 2020-01-01T00:00:00+00:00."; question=f"Is the elapsed time {'at most' if inclusive else 'strictly less than'} {threshold} hours? Use the timestamps exactly."
  ok=hours<=threshold if inclusive else hours<threshold; opts=[('yes','The elapsed-time condition is satisfied.'),('no','The elapsed-time condition is not satisfied.')]; targ=[1.,0.] if ok else [0.,1.]; ann.update(operand_a=min(hours,63),operand_b=min(threshold,63),stage=1,support_mask=[1,1,0,0,0,0,0,0],support_valid=[1,1,1,0,0,0,0,0])
 elif family=='join':
  people=[('Ava','red','alpha'),('Noah','blue','beta'),('Mia','green','gamma')]; person,team,pool=people[i%3]; wrong=people[(i+1)%3]; state=f"Person record: {person} belongs to team {team}. Team record: {team} maps to pool {pool}. Distractor: {wrong[0]} belongs to {wrong[1]}, which maps to {wrong[2]}."; opts=[(p,f"Route to pool {p}.") for p in ['alpha','beta','gamma']]; targ=[1. if p==pool else 0. for p,_ in opts]; question=f"Which pool should {person} be routed to by following the person-to-team and team-to-pool records?"; ann.update(operand_a=i%3,operand_b=(i+1)%3,stage=2,support_mask=[1,1,0,0,0,0,0,0],support_valid=[1,1,1,0,0,0,0,0])
 elif family=='policy':
  active=bool(i%2); exception=bool((i//2)%2); blocked=bool((i//4)%2); version=2 if (i//8)%2 else 1; state=f"Current policy version is v{version}. Base rule: active accounts may export. v2 amendment: blocked accounts may not export. Exception: approved-audit status overrides the block. Account active={str(active).lower()}, blocked={str(blocked).lower()}, approved-audit={str(exception).lower()}."; ok=active and (version==1 or not blocked or exception); opts=[('allow','The policy permits export.'),('deny','The policy does not permit export.')]; targ=[1.,0.] if ok else [0.,1.]; question="Under the current policy version, may this account export? Apply amendments and exceptions before deciding."; ann.update(operand_a=int(active),operand_b=int(blocked),stage=3,support_mask=[1,1,1,1,0,0,0,0],support_valid=[1,1,1,1,0,0,0,0])
 elif family=='probability':
  a=rng.randint(5,45); b=rng.randint(5,45); c=rng.randint(5,45); total=a+b+c; state=f"Complete cohort counts: alpha={a}, beta={b}, gamma={c}. No cases are omitted."; opts=[('alpha','Outcome alpha.'),('beta','Outcome beta.'),('gamma','Outcome gamma.')]; targ=[a/total,b/total,c/total]; question="For a uniformly selected member of this complete cohort, what is the outcome distribution?"; ann.update(operand_a=min(a,63),operand_b=min(b,63),stage=0,support_mask=[1,0,0,0,0,0,0,0],support_valid=[1,0,0,0,0,0,0,0])
 elif family=='judge':
  x=rng.randint(10,80); y=rng.randint(2,12); true=x*y; err=[0,1,-1,5][i%4]; cand=true+err; state=f"Question: compute {x} multiplied by {y}. Candidate answer: {cand}."; ok=cand==true; opts=[('correct','The candidate answer is correct.'),('incorrect','The candidate answer is incorrect.')]; targ=[1.,0.] if ok else [0.,1.]; question="Judge the candidate answer using exact arithmetic."; ann.update(operand_a=x if x<64 else 63,operand_b=y,stage=0,support_mask=[1,0,0,0,0,0,0,0],support_valid=[1,0,0,0,0,0,0,0])
 else: raise ValueError(family)
 order=list(range(len(opts))); rng.shuffle(order); opts=[opts[j] for j in order]; targ=[targ[j] for j in order]
 return {'id':sid,'family':family,'split':split,'public':{'id':sid,'state':state,'question':question,'options':[{'id':k,'description':d} for k,d in opts]},'target_probs':targ,'annotations':ann,'replay':False}

def make_data(split,n_each,seed): return [make_row(f,split,i,seed) for f in ('money','time','join','policy','probability','judge') for i in range(n_each)]
def fetch_bench():
 import urllib.request
 base='https://raw.githubusercontent.com/fstandhartinger/jevbench/c6004e008ffba24aec091261ca1a5c02f7324702/datasets/public/'; rows=[]
 for tier,fn in [('easy','easy.jsonl'),('standard','original.jsonl'),('hard','hard.jsonl')]:
  raw=urllib.request.urlopen(base+fn,timeout=60).read().decode()
  for line in raw.splitlines():
   if line.strip(): r=json.loads(line); r['_tier']=tier; rows.append(r)
 if len(rows)!=231: raise RuntimeError(f'benchmark rows {len(rows)}')
 return rows

def main():
 import torch
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='result'); a=ap.parse_args(); out=Path(a.out); out.mkdir(exist_ok=True)
 train=make_data('train',64,11); dev=make_data('development',16,23); held=make_data('heldout',16,37); bench=fetch_bench(); dump(out/'data_receipt.json',{'train':len(train),'development':len(dev),'heldout':len(held),'train_hash':jhash(train),'dev_hash':jhash(dev),'heldout_hash':jhash(held),'benchmark_hash':jhash(bench),'benchmark_training_used':False})
 rt=RT(); dump(out/'runtime.json',rt.meta); sets=[('train',train,[r['public'] for r in train]),('dev',dev,[r['public'] for r in dev]),('held',held,[r['public'] for r in held]),('bench',bench,[sem_row_from_bench(r) for r in bench])]; cached={}; t=time.perf_counter()
 for name,rows,vis in sets:
  arr=[]
  for i,r in enumerate(vis):
   arr.append(rt.cache(r))
   if (i+1)%25==0 or i+1==len(vis): emit('cache',set=name,done=i+1,total=len(vis),seconds=time.perf_counter()-t)
  cached[name]=arr
 torch.manual_seed(SEED); base=Adapter(torch,rt.down,rt.norm,rt.model.config.get_text_config().hidden_size)
 with torch.no_grad(): base.m.b.zero_()
 base_b=eval_rows(torch,base,bench,cached['bench'],True); base_h=eval_rows(torch,base,held,cached['held']); results={'base':{'benchmark':stats(base_b,True),'heldout':stats(base_h)}}; dump(out/'base-benchmark.json',base_b)
 for arm in ('hard','verified'):
  ad=train_arm(torch,rt,train,cached['train'],dev,cached['dev'],arm,out); b=eval_rows(torch,ad,bench,cached['bench'],True); h=eval_rows(torch,ad,held,cached['held']); results[arm]={'benchmark':stats(b,True),'heldout':stats(h)}; dump(out/f'{arm}-benchmark.json',b); dump(out/f'{arm}-heldout.json',h)
 checks=[]
 for arm in ('hard','verified'):
  import safetensors.torch as st; state=st.load_file(str(out/f'{arm}.safetensors')); ad=Adapter(torch,rt.down,rt.norm,rt.model.config.get_text_config().hidden_size); ad.load(state)
  def hook(m,args,res): return res+ad.m.delta(args[0])
  hh=rt.down.register_forward_hook(hook)
  try:
   for idx in (0,80,160,230):
    r=sem_row_from_bench(bench[idx]); full=rt.cache(r)['native']; cached_z,_=ad.logits_hidden(cached['bench'][idx]); err=float((full-cached_z.detach()).abs().max()); checks.append({'arm':arm,'id':bench[idx]['id'],'max_logit_diff':err})
  finally: hh.remove()
 dump(out/'checks.json',checks); dump(out/'results.json',{'results':results,'checks':checks,'benchmark_public_only':True,'official_score':None,'training_uses_jevbench':False,'seed':SEED,'method':'cached terminal FFN low-rank adaptation, exact full-model equivalence spot checks'}); rt.close(); emit('complete',results=results,checks=checks)

if __name__=='__main__': main()
