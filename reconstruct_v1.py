"""Jev-like reconstruction, NOT recovered Jev. No Jev API or autoregressive generation.
Research-only experiment: SciQ CC-BY-NC-3.0 and BoolQ CC-BY-SA-3.0.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
from collections import Counter
MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
EXPECTED={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
SHARDS=16
SEED=73129
MAX_TOKENS=8192
TRAIN_STEPS=256
PRIMARY='reconstructed_head_calibrated'
CONFIGS=['native_readout','reconstructed_head_raw',PRIMARY,'reconstructed_head_sym2']
def emit(kind,**kw):print(json.dumps({'kind':kind,**kw},sort_keys=True,allow_nan=False),flush=True)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def filehash(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False));q.replace(p)
def input_only(r):return {k:copy.deepcopy(r[k]) for k in ('id','state','question','labels')}
def probabilities(z,temperature=1.):
 if not z or not math.isfinite(temperature) or temperature<=0 or any(not math.isfinite(v) for v in z):raise ValueError('Invalid probability arguments')
 m=max(z);w=[math.exp((v-m)/temperature) for v in z];s=sum(w);return [v/s for v in w]
def target(r):return r['target_probs'] if r.get('target_probs') is not None else [float(k==str(r['expected'])) for k in r['labels']]
def synthetic(n=96):
 rng=random.Random(SEED);rows=[]
 for i in range(n):
  a,b=rng.sample(range(2,70),2);kind=i%3
  if kind==0:
   q={'type':'score','instructions':'Classify the numeric measurement using these thresholds.','criteria':['Below 20','At least 20 but below 40','At least 40 but below 60','At least 60']};labels=['0','1','2','3'];state={'measurement':a};gold=str(sum(a>=v for v in (20,40,60)));soft=None
  elif kind==1:
   labels=['amber','cobalt','jade','violet'];counts=[rng.randint(1,9) for _ in labels];s=sum(counts);state={'bag_counts':dict(zip(labels,counts)),'draw':'One object uniformly at random from the bag.'};q={'type':'choice','instructions':'Which color is drawn? Report its categorical distribution.','criteria':{k:'The selected object has color '+k for k in labels}};soft=[v/s for v in counts];gold=labels[max(range(4),key=lambda j:soft[j])]
  else:
   labels=['no','yes'];state={'rule':'Approve only when both current and verified are true.','current':bool(a%2),'verified':bool(b%2)};q={'type':'noul','instructions':'Is approval allowed under the stated rule?','criteria':{'false':'Approval is not allowed','true':'Approval is allowed'}};gold='yes' if state['current'] and state['verified'] else 'no';soft=None
  rows.append({'id':f'synthetic-{i:04d}','state':state,'question':q,'labels':labels,'expected':gold,'target_probs':soft,'source':'independent-executable-v1','partition':'train' if i<64 else 'development' if i<80 else 'calibration'})
 return rows

def prepare(root):
 import jevbench_public_v1 as upstream
 from huggingface_hub import HfApi,hf_hub_download
 import pyarrow.parquet as pq
 root=Path(root);root.mkdir(parents=True,exist_ok=True);tasks,bm=upstream.prepare(root/'benchmark',adapters=False);api=HfApi();samples=[];sources=[]
 for repo in ['allenai/sciq','google/boolq']:
  info=api.dataset_info(repo,files_metadata=False);paths=sorted(s.rfilename for s in info.siblings if s.rfilename.endswith('.parquet') and 'train' in s.rfilename)
  if not paths:raise RuntimeError('No public training parquet in '+repo)
  path=hf_hub_download(repo,paths[0],repo_type='dataset',revision=info.sha);data=pq.read_table(path).to_pylist();rng=random.Random(SEED+len(samples));indices=rng.sample(range(len(data)),192)
  for j,ix in enumerate(indices):
   d=data[ix];part='train' if j<128 else 'development' if j<160 else 'calibration'
   if repo=='allenai/sciq':
    choices=[d['correct_answer'],d['distractor1'],d['distractor2'],d['distractor3']];order=list(range(4));rng.shuffle(order);labels=['a','b','c','d'];q={'type':'choice','instructions':d['question'],'criteria':{k:choices[ix] for k,ix in zip(labels,order)}};gold=labels[order.index(0)];state=d.get('support') or 'Use your scientific knowledge to answer the question.'
   else:
    labels=['no','yes'];q={'type':'noul','instructions':d['question'],'criteria':{'false':'The answer is no','true':'The answer is yes'}};gold='yes' if d['answer'] else 'no';state=d['passage']
   samples.append({'id':repo.replace('/','-')+'-'+str(ix),'state':state,'question':q,'labels':labels,'expected':gold,'target_probs':None,'source':repo,'partition':part})
  sources.append({'repo':repo,'revision':info.sha,'path':paths[0],'sha256':filehash(path),'selected_indices':indices,'rows':len(data)})
 samples+=synthetic();bench_text={digest({k:v for k,v in input_only(r).items() if k!='id'}) for r in tasks}
 for r in samples:
  if digest({k:v for k,v in input_only(r).items() if k!='id'}) in bench_text:raise RuntimeError('Exact evaluation overlap')
  upstream.convert(input_only(r))
 combined=[{**input_only(r),'partition':r['partition']} for r in samples]+[{**input_only(r),'partition':'benchmark'} for r in tasks];bins=[[] for _ in range(SHARDS)];loads=[0.]*SHARDS
 for r in sorted(combined,key=lambda r:(-len(json.dumps(input_only(r)))**1.3,r['id'])):
  k=min(range(SHARDS),key=lambda k:(loads[k],k));bins[k].append(r['id']);loads[k]+=len(json.dumps(input_only(r)))**1.3
 manifest={'model':MODEL,'model_revision':REV,'benchmark_revision':bm['benchmark_revision'],'benchmark_data_hash':bm['data_hash'],'input_hash':digest(combined),'sources':sources,'samples':Counter(r['partition'] for r in combined),'partitions':bins,'source_sha256':filehash(__file__),'primary':PRIMARY,'configs':CONFIGS,'training':{'steps':TRAIN_STEPS,'seed':SEED,'learning_rate':0.001,'batch_size':64,'max_logit_correction':1.0,'kl_anchor':0.5,'brier_weight':0.1,'weight_decay':0.01},'public_only':True,'full_leaderboard_score':None,'limits':['Public benchmark previously inspected and evaluated; not a fresh sealed test.','Exact input disjointness checked; semantic and pretraining contamination not excluded.','SciQ CC-BY-NC-3.0; research-only experiment.']}
 write(root/'training.json',samples);write(root/'inputs.json',combined);write(root/'manifest.json',manifest);emit('prepared',samples=manifest['samples'],benchmark_hash=bm['data_hash'],source_hash=manifest['source_sha256'])

class Runtime:
 def __init__(self,root):
  import torch
  from huggingface_hub import snapshot_download
  from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
  self.torch=torch;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);self.snapshot=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4);hashes={p.name:filehash(p) for p in Path(self.snapshot).glob('*.safetensors')}
  if hashes!=EXPECTED:raise RuntimeError('Checkpoint hash mismatch')
  self.tokenizer=AutoTokenizer.from_pretrained(self.snapshot,trust_remote_code=False,local_files_only=True);start=time.perf_counter();self.model,info=Qwen3_5ForConditionalGeneration.from_pretrained(self.snapshot,dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False,local_files_only=True,output_loading_info=True);issues={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
  if issues:raise RuntimeError(str(issues))
  self.model.eval();self.original_head=self.model.get_output_embeddings();original=self.original_head
  class SelectedHead(torch.nn.Module):
   def __init__(self):super().__init__();self.base=original;self.codes=[];self.hidden=None
   @property
   def weight(self):return self.base.weight
   def forward(self,h):
    if not self.codes:raise RuntimeError('No configured candidates')
    self.hidden=h[:,-1,:].detach().float();bias=self.base.bias[self.codes].float() if self.base.bias is not None else None
    return torch.nn.functional.linear(h.float(),self.base.weight[self.codes].float(),bias)
  self.head=SelectedHead();self.model.set_output_embeddings(self.head);self.receipt={'model':MODEL,'revision':REV,'weight_hashes':hashes,'torch':torch.__version__,'parameters':sum(p.numel() for p in self.model.parameters()),'load_seconds':time.perf_counter()-start,'dtype':'BF16 backbone; FP32 selected-row projection','no_generation':True};sys.path.insert(0,str((Path(root)/'benchmark/vendor').resolve()));emit('loaded',**self.receipt)
 def encode(self,inp,reverse=False):
  import jevbench_public_v1 as upstream
  from semif_phase1.direct import encode_prompt
  _,sem=upstream.convert(input_only(inp))
  if reverse:sem['options']=list(reversed(sem['options']))
  return encode_prompt(self.tokenizer,sem,MAX_TOKENS)
 def score(self,inp,reverse=False):
  torch=self.torch;t=time.perf_counter();ids,codes,ph=self.encode(inp,reverse);self.head.codes=codes
  with torch.inference_mode():
   out=self.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),logits_to_keep=1,use_cache=False,return_dict=True);z=out.logits[0,-1].float().cpu();h=self.head.hidden[0].cpu().clone()
  if not torch.isfinite(z).all() or not torch.isfinite(h).all():raise RuntimeError('Nonfinite output')
  return {'logits':z.tolist(),'seconds':time.perf_counter()-t,'tokens':len(ids),'prompt_hash':ph,'input_hash':digest(input_only(inp))},h
 def check(self):
  fixture={'id':'readout-verification','state':'The named color is cobalt.','question':{'type':'choice','instructions':'Select the stated color.','criteria':{'amber':'Amber','cobalt':'Cobalt'}},'labels':['amber','cobalt']};out,h=self.score(fixture);torch=self.torch;codes=self.head.codes;full=[]
  with torch.inference_mode():
   for i in range(0,self.original_head.weight.shape[0],4096):
    b=self.original_head.bias[i:i+4096].float() if self.original_head.bias is not None else None;full.append(torch.nn.functional.linear(h[None],self.original_head.weight[i:i+4096].float(),b)[0])
   ref=torch.cat(full)[codes];err=float((ref-torch.tensor(out['logits'])).abs().max())
  if err>1e-4:raise RuntimeError('Full vocabulary reference mismatch: '+str(err))
  return {'full_vocabulary_fp32_error':err,'predicted':fixture['labels'][max(range(2),key=lambda j:out['logits'][j])],'fixture_expected':'cobalt'}

def extract(root,out,shard):
 from safetensors.torch import save_file
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False);manifest=json.loads((root/'manifest.json').read_text());allrows=json.loads((root/'inputs.json').read_text());idx={r['id']:r for r in allrows}
 if filehash(__file__)!=manifest['source_sha256']:raise RuntimeError('Source changed')
 rt=Runtime(root);check=rt.check();write(out/'preflight.json',{'runtime':rt.receipt,'check':check});hs={};records=[]
 for i,rid in enumerate(manifest['partitions'][shard]):
  inp=idx[rid]
  for rev in [False,True]:
   rec,h=rt.score(input_only(inp),rev);key=f'row{i:04d}_order{int(rev)}';hs[key]=h;records.append({**rec,'id':rid,'partition':inp['partition'],'reverse':rev,'feature_key':key,'labels':inp['labels'],'shard':shard})
  if i%4==0:emit('progress',shard=shard,completed=i+1,planned=len(manifest['partitions'][shard]))
 save_file(hs,str(out/'features.safetensors'));write(out/'records.json',records);write(out/'complete.json',{'shard':shard,'input_hash':manifest['input_hash'],'count':len(records),'feature_sha256':filehash(out/'features.safetensors'),'source_sha256':manifest['source_sha256'],'task_ids':manifest['partitions'][shard]});emit('complete',shard=shard,records=len(records),feature_sha256=filehash(out/'features.safetensors'))

def build_training_batch(root,feature_root):
 from safetensors.torch import load_file
 root=Path(root);feature_root=Path(feature_root);manifest=json.loads((root/'manifest.json').read_text());labels=json.loads((root/'training.json').read_text());idx={r['id']:r for r in labels};fit=[];benchmark=[];seen=set()
 for shard in range(SHARDS):
  d=feature_root/('shard-'+str(shard));c=json.loads((d/'complete.json').read_text())
  if c['input_hash']!=manifest['input_hash'] or c['source_sha256']!=manifest['source_sha256'] or c['task_ids']!=manifest['partitions'][shard]:raise RuntimeError('Provenance mismatch')
  if filehash(d/'features.safetensors')!=c['feature_sha256']:raise RuntimeError('Feature mismatch')
  hs=load_file(str(d/'features.safetensors'));rr=json.loads((d/'records.json').read_text())
  if len(rr)!=c['count'] or len(rr)!=2*len(c['task_ids']):raise RuntimeError('Incomplete shard')
  for r in rr:
   key=(r['id'],r['reverse'])
   if key in seen:raise RuntimeError('Duplicate record')
   seen.add(key);r['hidden']=hs[r['feature_key']]
   if r['partition']=='benchmark':benchmark.append(r);continue
   src=idx[r['id']]
   if r['partition']!=src['partition'] or r['input_hash']!=digest(input_only(src)):raise RuntimeError('Training input mismatch')
   y=target(src);r['target']=list(reversed(y)) if r['reverse'] else y;fit.append(r)
 if len(benchmark)!=462 or len(fit)!=960:raise RuntimeError('Incomplete population')
 return fit,benchmark

def fit_head(fit,out):
 import torch
 from safetensors.torch import save_file
 torch.manual_seed(SEED);torch.set_num_threads(4)
 if any(r['partition'] not in ('train','development','calibration') for r in fit):raise ValueError('Evaluation record in fitting')
 dim=fit[0]['hidden'].numel();delta=torch.nn.Parameter(torch.zeros(16,dim));opt=torch.optim.AdamW([delta],lr=.001,weight_decay=.01);tr=[r for r in fit if r['partition']=='train'];dev=[r for r in fit if r['partition']=='development'];cal=[r for r in fit if r['partition']=='calibration']
 def batch(rows):
  h=torch.stack([r['hidden'] for r in rows]);h=h/torch.sqrt((h*h).mean(-1,keepdim=True)+1e-8);z=torch.full((len(rows),16),-1e9);y=torch.zeros_like(z)
  for i,r in enumerate(rows):k=len(r['logits']);z[i,:k]=torch.tensor(r['logits']);y[i,:k]=torch.tensor(r['target'])
  return h,z,y
 ht,zt,yt=batch(tr);hd,zd,yd=batch(dev);hc,zc,yc=batch(cal)
 def scored(h,z):return z+torch.tanh(torch.nn.functional.linear(h,delta))
 def loss(z,y,original=None):
  lp=torch.log_softmax(z,-1);p=lp.exp();ce=-(y*lp).sum(-1).mean();br=((p-y)**2).sum(-1).mean();total=ce+.1*br
  if original is not None:
   q=torch.softmax(original,-1);kl=(q*(torch.log_softmax(original,-1)-lp)).sum(-1).mean();total=total+.5*kl
  return total
 best=delta.detach().clone();best_loss=float(loss(zd,yd));trace=[];rng=torch.Generator().manual_seed(SEED)
 for step in range(1,TRAIN_STEPS+1):
  ii=torch.randint(len(tr),(64,),generator=rng);opt.zero_grad(set_to_none=True);l=loss(scored(ht[ii],zt[ii]),yt[ii],zt[ii]);l.backward();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
  if step%8==0:
   with torch.no_grad():dl=float(loss(scored(hd,zd),yd));trace.append({'step':step,'train_loss':float(l.detach()),'development_loss':dl})
   if dl<best_loss:best_loss=dl;best=delta.detach().clone()
 with torch.no_grad():delta.copy_(best);calz=scored(hc,zc)
 temps=[math.exp(math.log(.25)+i/120*math.log(16)) for i in range(121)]
 def get_temp(z):
  vals=[float(-(yc*torch.log_softmax(z/t,-1)).sum(-1).mean()) for t in temps];j=min(range(len(vals)),key=lambda i:vals[i]);return temps[j],vals[j]
 native_t,native_nll=get_temp(zc);t,nll=get_temp(calz);save_file({'delta':best},str(Path(out)/'decision_head.safetensors'))
 cfg={'format':'bounded-residual-native-readout-v1','base_model':MODEL,'revision':REV,'hidden_size':dim,'slots':16,'residual_limit':1.0,'normalization':'per-example root mean square + 1e-8','temperature':t,'native_comparison_temperature':native_t,'calibration_nll':nll,'native_calibration_nll':native_nll,'trainable_parameters':delta.numel(),'nonzero_weights':int((best!=0).sum()),'max_abs_parameter':float(best.abs().max()),'steps':TRAIN_STEPS,'selection':'minimum development CE + 0.1 Brier, including zero-update checkpoint','partition_examples':dict(Counter(r['partition'] for r in fit)),'trace':trace,'sha256':filehash(Path(out)/'decision_head.safetensors'),'scope':'Frozen pretrained backbone; residual decision head and scalar temperature fitted. Not RLCD.'};write(Path(out)/'head_config.json',cfg);return best,cfg

def metrics(rows,tasks):
 import numpy as np
 from jevbench.scoring import score_task
 from jevbench.tasks import Task
 evaluated=[]
 for r in rows:
  t=tasks[r['id']];p=dict(zip(t['labels'],r['probabilities']));s=score_task(p,Task.from_dict(t));evaluated.append({**r,**s,'tier':t['_tier'],'type':t['question']['type'],'expected':t['expected']})
 def summary(ss):
  correct=sum(bool(r['correct']) for r in ss);n=len(ss);nll=[];br=[];conf=[];acc=[]
  for r in ss:
   p=r['probs'];y=str(r['expected']);nll.append(-math.log(max(p[y],1e-30)));br.append(sum((v-float(k==y))**2 for k,v in p.items()));conf.append(max(p.values()));acc.append(float(r['correct']))
  ece=0.
  for b in range(10):
   ix=[i for i,c in enumerate(conf) if int(min(c*10,9))==b]
   if ix:ece+=len(ix)/n*abs(sum(conf[i] for i in ix)/len(ix)-sum(acc[i] for i in ix)/len(ix))
  return {'n':n,'correct':correct,'accuracy':correct/n,'nll':float(np.mean(nll)),'brier':float(np.mean(br)),'ece_10_equal_width':ece,'strict_valid':sum(r['strict_valid'] for r in ss)}
 out={'overall':summary(evaluated),'by_tier':{t:summary([r for r in evaluated if r['tier']==t]) for t in ('easy','standard','hard')},'by_type':{t:summary([r for r in evaluated if r['type']==t]) for t in ('choice','noul','score')}};soft=[]
 for r in evaluated:
  g=tasks[r['id']].get('provenance',{}).get('gold_probs')
  if g:soft.append(sum((r['probs'].get(k,0.)-v)**2 for k,v in g.items()))
 out['gold_distribution_brier']={'n':len(soft),'mean':float(np.mean(soft)) if soft else None};return out,evaluated

def finalize(root,features,out):
 import torch
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((root/'manifest.json').read_text());fit,bench=build_training_batch(root,features)
 # Fitting receives no benchmark object or labels; persist checkpoint before scoring.
 delta,cfg=fit_head(fit,out);sys.path.insert(0,str((root/'benchmark/vendor').resolve()));tasks={r['id']:r for r in json.loads((root/'benchmark/tasks.json').read_text())};byid={rid:[] for rid in tasks}
 for r in bench:
  if r['input_hash']!=digest(input_only(tasks[r['id']])):raise RuntimeError('Evaluation input mismatch')
  byid[r['id']].append(r)
 results={c:[] for c in CONFIGS}
 for rid,rr in byid.items():
  rr=sorted(rr,key=lambda r:r['reverse']);p=[];base=[]
  for r in rr:
   h=r['hidden'];hn=h/torch.sqrt((h*h).mean()+1e-8);k=len(r['logits']);z=torch.tensor(r['logits']);new=z+torch.tanh(torch.nn.functional.linear(hn,delta[:k]));ps=probabilities(new.tolist(),cfg['temperature']);p.append(list(reversed(ps)) if r['reverse'] else ps);base.append(probabilities(r['logits']))
   if not r['reverse']:raw=probabilities(new.tolist())
  distributions={'native_readout':base[0],'reconstructed_head_raw':raw,PRIMARY:p[0],'reconstructed_head_sym2':[(x+y)/2 for x,y in zip(*p)]}
  for c,ps in distributions.items():results[c].append({'id':rid,'probabilities':ps,'seconds':sum(r['seconds'] for r in rr) if c.endswith('sym2') else rr[0]['seconds']})
 summaries={};detail={}
 for c,rr in results.items():summaries[c],detail[c]=metrics(rr,tasks)
 base={r['id']:r for r in detail['native_readout']};primary={r['id']:r for r in detail[PRIMARY]};wins=[i for i in tasks if primary[i]['correct'] and not base[i]['correct']];losses=[i for i in tasks if base[i]['correct'] and not primary[i]['correct']]
 final={'status':'completed','public_tasks':231,'official_full_score':None,'official_submission':False,'primary':PRIMARY,'metrics':summaries,'paired_primary_vs_native':{'wins':len(wins),'losses':len(losses),'win_ids':wins,'loss_ids':losses},'trained_checkpoint':cfg,'manifest':manifest,'runtime_scope':'GitHub ARM CPU BF16, serial per shard. Parallel shards are throughput execution, not per-request latency acceleration.','limitations':['Qwen3.5 public weights, not Jev weights.','Only 231 public benchmark items; 303 nonpublic decisions unavailable.','320 source training examples, 80 development, 80 calibration. Two orderings are augmentations, not independent examples.','No API distillation or recovered RLCD. No composite score or deployed pricing fabricated.','Single seed; no universal calibration claim.']};write(out/'results.json',final);write(out/'predictions.json',detail);write(out/'training_manifest.json',{'input_ids':[r['id'] for r in fit],'benchmark_ids':list(tasks),'disjoint':not set(r['id'] for r in fit)&set(tasks)});emit('FINAL_RESULTS',metrics=summaries,paired=final['paired_primary_vs_native'],head={'nonzero_weights':cfg['nonzero_weights'],'temperature':cfg['temperature'],'sha256':cfg['sha256']})

def selftest():
 r={'id':'x','state':'z','question':{'type':'choice'},'labels':['x','y'],'expected':'SECRET','provenance':{'answer':'SECRET'},'partition':'benchmark'};before=input_only(r);r['expected']='ALTERED';assert input_only(r)==before and 'SECRET' not in json.dumps(before);assert abs(sum(probabilities([10000,10001]))-1)<1e-12
 for bad in [float('nan'),float('inf')]:
  try:probabilities([bad,1.]);raise AssertionError('Nonfinite accepted')
  except ValueError:pass
 ss=synthetic();assert len(ss)==96 and len({r['id'] for r in ss})==96
 for r in ss:assert abs(sum(target(r))-1)<1e-8
 assert dict(Counter(r['partition'] for r in ss))=={'train':64,'development':16,'calibration':16};emit('selftest',passed=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['test','prepare','extract','finalize']);a.add_argument('--root',default='reconstruction-inputs');a.add_argument('--out',default='reconstruction-results');a.add_argument('--features',default='features');a.add_argument('--shard',type=int,default=0);x=a.parse_args();selftest()
 if x.mode=='prepare':prepare(x.root)
 elif x.mode=='extract':
  if x.shard not in range(SHARDS):raise ValueError('Invalid shard')
  extract(x.root,x.out,x.shard)
 elif x.mode=='finalize':finalize(x.root,x.features,x.out)
if __name__=='__main__':main()
