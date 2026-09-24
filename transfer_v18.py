"""Frozen transfer/retention test of six completed v17.1 adapters; no training."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,time,urllib.request
from fractions import Fraction as F
from pathlib import Path
import causal_v17_data as old
import contrast_v7 as external
import learn_v16 as learn
import measure_v14 as measure
SEED=180924271
SHARDS=16
SEEDS=(17101,17102,17103)
KINDS=('event_high','mixed_high')
CHECKPOINTS={
 'event_high/17101':'972b1868b7ccd1dd9665e54c67ceaffcb42bb90440cc1857be974bbe545f4d5e',
 'mixed_high/17101':'1b296908e7a0b285cc92e7df4d51bc84a940764c7eeada53defd900e65eda65e',
 'event_high/17102':'a16357219de710a19f067fd88fe2cd1b2c79ccce32911c57bc9b64cbea3a9711',
 'mixed_high/17102':'b888625bfe734cb0a3a7409ed7098be7be4a003b3177c4f518df7e41548fb5fb',
 'event_high/17103':'6750d5d0b9691106b03ac9d9e5cf131d5be0c977b08ad4039ae6c3f468cff603',
 'mixed_high/17103':'6076296064ffaeacad2bffe29d1ada7de2dc07f195e214d4107411ae2c55f621'}
TEMPERATURES={'unchanged/0':[1.8025009252216606,2.29739670999407],
 'event_high/17101':[1.4640856959456257,1.8025009252216606],
 'event_high/17102':[1.3660402567543957,2.9281713918912513],
 'event_high/17103':[1.5157165665103982,2.5491212546385245],
 'mixed_high/17101':[1.6245047927124712,2.1435469250725863],
 'mixed_high/17102':[1.9318726578496914,3.0314331330207964],
 'mixed_high/17103':[1.8660659830736153,3.363585661014858]}
EXCLUDED={'causal_judgement':[28,32,34,39,83,84,95,110,119,122,129,130,140,146,159,175,184],
 'date_understanding':[1,10,28,34,53,70,77,102,177,178,183,187,208,213,220,249],
 'disambiguation_qa':[5,16,27,32,67,99,118,154,157,194,198,212,214,221,235,244],
 'logical_deduction_seven_objects':[1,2,4,9,10,11,21,26,34,43,44,46,50,72,74,77,84,86,89,92,94,99,101,107,109,111,113,118,122,131,133,136,148,151,155,158,159,168,170,175,181,190,193,204,214,218,222,223,246],
 'sports_understanding':[27,55,76,92,102,104,110,132,134,140,155,182,188,216,218,230,231],
 'temporal_sequences':[6,37,77,78,94,107,113,144,150,151,164,165,167,185,213,244],
 'tracking_shuffled_objects_seven_objects':[7,8,19,20,35,55,63,74,87,98,106,109,116,121,144,206]}

def digest(x):return old.digest(x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):old.write(p,x)
def canonical(x):return json.loads(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False))
def evidence(w):
 rows='; '.join(f"{s}: sample I has {a}, sample II has {b}, archive has {n}, detection rate is {r}/10" for s,a,b,n,r in zip(w['labels'],w['a'],w['b'],w['noise'],w['rates']))
 f=w['family']
 if f=='counts':rule='A record is drawn uniformly from sample I. Sample II and the archive are inaccessible; detection rates are irrelevant.'
 elif f=='conditional':rule='Sample I contains exactly the records certified for use; sample II is uncertified. A record is selected uniformly from the certified population only. The archive and detection rates play no role.'
 elif f=='mixture':rule=f"Select sample I with probability {w['weight']}/10 and sample II with probability {10-w['weight']}/10, then select a record uniformly inside that sample. Neither choice has been observed. The archive and detection rates play no role."
 else:rule='A record was drawn uniformly from sample I and its detector signaled positive. For each category its listed detection rate is the exact probability of that positive signal. The category remains unknown. Sample II and the archive play no role.'
 return 'All entries are exact and exhaustive, not estimates. Category register: '+rows+'. Procedure: '+rule

def make_worlds(seed=SEED):
 rng=random.Random(seed);out=[]
 for fam in old.FAMILIES:
  for j,k in enumerate((2,3,4,5,6,8)):
   for _ in range(10000):
    w={'family':fam,'labels':rng.sample(['copper','quartz','opal','slate','pearl','onyx','basalt','flint','marble','garnet','beryl','topaz'],k),
       'a':[rng.randint(1,49) for _ in range(k)],'b':[rng.randint(1,43) for _ in range(k)],
       'noise':[rng.randint(20,190) for _ in range(k)],'rates':[rng.randint(1,9) for _ in range(k)],'weight':rng.randint(1,9),
       'id':f'transfer-{fam}-k{k}','zero_intervention':j in (0,4)}
    if w['zero_intervention']:w['a'][0]=0
    q=old.law(w)
    if q.count(max(q))==1:break
   else:raise ValueError('Unique mode not generated')
   out.append(w)
 return out

def generated():
 result=[]
 for w in make_worlds():
  for variant in ('familiar','prose','replication','relevant','mode'):
   v=copy.deepcopy(w)
   if variant=='replication':v['a']=[5*x for x in v['a']]
   if variant=='relevant':v['a'][0]+=19
   q=old.law(v);kind='mode' if variant=='mode' else 'event'
   target=[F(int(i==old.mode(q))) for i in range(len(q))] if kind=='mode' else q
   text=old.render(v) if variant=='familiar' else evidence(v)
   instruction=('Return the distribution of the category of the selected record, conditioned on every observation stated in the procedure. Do not substitute certainty about which category is most likely.' if kind=='event' else
                'Identify the unique category with the largest conditional probability under the exact procedure. Report the distribution over the correct answer to that question, not the law of a random selected record.')
   inp=canonical({'id':w['id']+'-'+variant,'state':text,'question':{'type':'choice','instructions':instruction,'criteria':{s:'The category is '+s+'.' for s in w['labels']}},'labels':w['labels']})
   result.append({'input':inp,'target':[str(x) for x in target],'cohort':'probability','family':w['family'],'group':w['id'],
                  'variant':variant,'quantity':kind,'mechanism':v,'labels_seen_in_training':False})
 return result

def retention(source_dir):
 out=[];receipts=[];source_dir=Path(source_dir);source_dir.mkdir(parents=True,exist_ok=True)
 for fam,excluded in sorted(EXCLUDED.items()):
  p=source_dir/(fam+'.json')
  if not p.exists():
   url=f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/bbh/{fam}.json'
   with urllib.request.urlopen(url,timeout=60) as r:p.write_bytes(r.read())
  raw=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
  if blob!=external.SOURCES[fam]:raise ValueError('External bytes changed')
  xs=json.loads(raw)['examples'];order=list(range(len(xs)));random.Random(SEED+int(hashlib.sha256(fam.encode()).hexdigest()[:8],16)).shuffle(order)
  chosen=[i for i in order if i not in excluded][:8]
  for i in chosen:
   r=external.external_row(fam,i,xs[i]);inp=canonical({k:r[k] for k in ('id','state','question','labels')});inp['id']=f'transfer-bbh-{fam}-{i:03d}'
   y=str(r['expected']);target=[str(int(s==y)) for s in inp['labels']]
   if sum(map(int,target))!=1:raise ValueError('Invalid source label mapping')
   out.append({'input':inp,'target':target,'cohort':'retention','family':fam,'group':inp['id'],'variant':'external','quantity':'deterministic','source_index':i,'source_blob':blob})
  receipts.append({'family':fam,'selected_indices':chosen,'excluded_indices':excluded,'git_blob':blob,'sha256':sha(p)})
 return out,receipts

def prepare(root,out):
 root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=False)
 ext,receipts=retention(root/'transfer-sources');rows=generated()+ext
 previous=json.loads((root/'causal-prepared/records.json').read_text())
 sig=lambda r:digest({k:r['input'][k] for k in ('state','question','labels')})
 if len({sig(r) for r in rows})!=176 or {sig(r) for r in rows}&{sig(r) for r in previous}:raise ValueError('Input overlap')
 jobs=[[] for _ in range(SHARDS)];cost=[0]*SHARDS
 for row in sorted(rows,key=lambda r:(-len(json.dumps(r['input'])),r['input']['id'])):
  i=min(range(SHARDS),key=lambda k:(cost[k],k));jobs[i].append(row['input']);cost[i]+=len(json.dumps(row['input']))+1800
 weights={}
 for model,h in CHECKPOINTS.items():
  arm,seed=model.split('/');p=root/f'models/causal-v17-clean-model-{seed}-{arm}/adapter-step-32.safetensors'
  if sha(p)!=h:raise ValueError('Saved adapter hash mismatch')
  weights[model]=str(p.relative_to(root))
 oldbase=[json.loads(s) for s in (root/'baseline/predictions.jsonl').read_text().splitlines()]
 byid={r['input']['id']:r['input'] for r in previous}
 anchors=[{'input':byid[r['id']],'record':r} for r in oldbase[:SHARDS]]
 parents={p.name:sha(p) for p in root.glob('*.py') if p.name in ('causal_v17_data.py','learn_v16.py','learn_data.py','relational_objective.py','bridge_v15.py','measure_v14.py','contrast_v7.py','handoff_v5.py','reconstruct_v1.py','evidence_v4.py','jevbench_public_v1.py')}
 m={'version':'transfer-v18.0','source_sha256':sha(__file__),'parent_hashes':parents,'jobs_hash':digest(jobs),'evaluation_hash':digest(rows),'source_receipts':receipts,
    'weights':weights,'checkpoint_hashes':CHECKPOINTS,'temperatures':TEMPERATURES,'seed':SEED,'models':['unchanged/0']+sorted(CHECKPOINTS),
    'tasks':176,'new_predictions':1232,'generated_worlds':24,'retention_items':56,'shard_counts':list(map(len,jobs)),
    'primary':'frozen-calibrated event_high versus unchanged on prose event CE and squared loss; raw external retention accuracy is guardrail',
    'calibration':'v17 temperatures retained exactly; no test fitting. Modal temperature on external tasks is a secondary transport diagnostic.'}
 for name,val in [('jobs',jobs),('evaluation',rows),('anchors',anchors),('manifest',m)]:write(out/(name+'.json'),val)
 print(json.dumps(m),flush=True)

def run(root,prepared,out,shard):
 import torch
 from safetensors.torch import load_file
 from reconstruct_v1 import Runtime
 root,prepared,out=map(Path,(root,prepared,out));out.mkdir(parents=True,exist_ok=False)
 if shard not in range(SHARDS):raise ValueError('Invalid shard')
 m=json.loads((prepared/'manifest.json').read_text());jobs=json.loads((prepared/'jobs.json').read_text())
 if m['source_sha256']!=sha(__file__) or m['jobs_hash']!=digest(jobs):raise ValueError('Freeze changed')
 for p,h in m['parent_hashes'].items():
  if sha(root/p)!=h:raise ValueError('Parent code changed')
 rt=Runtime(root/'reconstruction-inputs');rt.model.requires_grad_(False);rt.model.eval();fixture=rt.check();cache={}
 def encode(inp):
  msg=measure.arm_messages(inp,'','semantic_codes');text=rt.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=rt.tokenizer.encode(text,add_special_tokens=False);codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(inp['labels']))]
  if len(ids)>1600 or any(len(x)!=1 for x in codes):raise ValueError('Prompt/codes out of fixed domain; no truncation')
  for j,c in enumerate(codes):
   if rt.tokenizer.encode(text+chr(65+j),add_special_tokens=False)!=ids+c:raise ValueError('Code boundary mismatch')
  return {'ids':torch.tensor([ids]),'mask':torch.ones((1,len(ids)),dtype=torch.long),'codes':[c[0] for c in codes],'hash':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(ids)}
 def forward(c):
  rt.head.codes=c['codes']
  with torch.inference_mode():z=rt.model(input_ids=c['ids'],attention_mask=c['mask'],logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
  if not torch.isfinite(z).all() or z.numel()!=len(c['codes']):raise ValueError('Bad logits')
  return z
 anchor=json.loads((prepared/'anchors.json').read_text())[shard];enc=encode(anchor['input']);ref=torch.tensor(anchor['record']['logits']);native=forward(enc)
 if enc['hash']!=anchor['record']['prompt_sha256'] or float((native-ref).abs().max())>1e-4:raise ValueError('Baseline anchor mismatch')
 _,factors,hooks=learn.make_factors(torch,rt.model,17101);factors.requires_grad_(False)
 if float((forward(enc)-native).abs().max())>1e-4:raise ValueError('Zero adapter mismatch')
 states={}
 for key,p in m['weights'].items():
  if sha(root/p)!=CHECKPOINTS[key]:raise ValueError('Checkpoint changed')
  states[key]=load_file(str(root/p))
 checks={}
 for key,state in states.items():
  factors.load_state_dict(state,strict=True)
  arm,seed=key.split('/');d=root/f'models/causal-v17-clean-model-{seed}-{arm}'
  oldpred=json.loads((d/'final-predictions.json').read_text());gold=next(r for r in oldpred if r['id']==anchor['input']['id'])
  err=float((forward(enc)-torch.tensor(gold['logits'])).abs().max());checks[key]=err
  if err>1e-4:raise ValueError('Adapted anchor mismatch')
 write(out/'preflight.json',{'model':rt.receipt,'fixture':fixture,'source_sha256':m['source_sha256'],'baseline_anchor_error':float((native-ref).abs().max()),'adapter_anchor_errors':checks,'weights':CHECKPOINTS})
 path=out/'records.jsonl';path.write_text('');done=[]
 for inp in jobs[shard]:
  c=encode(inp);order=list(m['models']);random.Random(int(digest(inp)[:8],16)).shuffle(order);outputs={}
  for model in order:
   for factor in factors:factor.enabled=model!='unchanged/0'
   if model in states:factors.load_state_dict(states[model],strict=True)
   tick=time.perf_counter();z=forward(c);p=torch.softmax(z,-1)
   outputs[model]={'logits':z.tolist(),'probabilities':p.tolist(),'label':inp['labels'][int(z.argmax())],'seconds':time.perf_counter()-tick}
  record={'id':inp['id'],'input_sha256':digest(inp),'prompt_sha256':c['hash'],'tokens':c['tokens'],'code_token_ids':c['codes'],'outputs':outputs,'execution_order':order,'shard':shard}
  with path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  done.append(inp['id']);print(json.dumps({'shard':shard,'done':len(done),'total':len(jobs[shard])}),flush=True)
 for factor in factors:factor.enabled=False
 err=float((forward(enc)-native).abs().max())
 if err>1e-4:raise ValueError('Baseline restoration mismatch')
 for h in hooks:h.remove()
 write(out/'complete.json',{'ids':done,'count':len(done),'source_sha256':m['source_sha256'],'jobs_hash':digest(jobs[shard]),'records_sha256':sha(path),'baseline_restoration_error':err,'new_optimizer_updates':0,'new_generations':0})

def aggregate(prepared,records,out):
 prepared,records,out=map(Path,(prepared,records,out));m=json.loads((prepared/'manifest.json').read_text());jobs=json.loads((prepared/'jobs.json').read_text());rows=[];seen=set()
 for i,plan in enumerate(jobs):
  p=records/f'transfer-v18-shard-{i}';c=json.loads((p/'complete.json').read_text());path=p/'records.jsonl'
  if c['source_sha256']!=m['source_sha256'] or c['jobs_hash']!=digest(plan) or c['records_sha256']!=sha(path):raise ValueError('Shard provenance')
  rr=[json.loads(s) for s in path.read_text().splitlines()];wanted={r['id']:r for r in plan}
  if len(rr)!=len(plan) or set(c['ids'])!=set(wanted):raise ValueError('Incomplete shard')
  for r in rr:
   if r['id'] in seen or r['id'] not in wanted or r['input_sha256']!=digest(wanted[r['id']]) or set(r['outputs'])!=set(m['models']):raise ValueError('Missing or changed observation')
   seen.add(r['id']);rows.append(r)
 if len(rows)!=176:raise ValueError('Incomplete population')
 write(out/'all_records.json',sorted(rows,key=lambda r:r['id']));write(out/'completion.json',{'complete':True,'inputs':176,'predictions':1232,'trained_models':6,'baseline_models':1,'new_training_updates':0})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--root',default='.');p.add_argument('--prepared',default='transfer-prepared');p.add_argument('--out',default='.');p.add_argument('--records',default='records');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.mode=='prepare':prepare(a.root,a.prepared)
 elif a.mode=='run':run(a.root,a.prepared,a.out,a.shard)
 else:aggregate(a.prepared,a.records,a.out)
