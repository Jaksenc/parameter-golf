"""Independent recorded-output reduction; no inference or parameter selection."""
from __future__ import annotations
import argparse,hashlib,json,math,re,random
from decimal import Decimal,localcontext
from fractions import Fraction as F
from pathlib import Path
import numpy as np
SEED=180924271
SEEDS=(17101,17102,17103)
ARMS=('event_high','mixed_high')

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))
def probs(z,T=1):
 z=np.asarray(z,dtype=np.float64)/T;z-=z.max();p=np.exp(z);return p/p.sum()
def metric(z,q,T=1):
 z=np.asarray(z,dtype=np.float64)/T;p=probs(z);q=np.array(q,dtype=float);logsum=float(z.max()+np.log(np.exp(z-z.max()).sum()));lp=z-logsum
 return {'ce':float(-(q*lp).sum()),'squared':float(((p-q)**2).sum()),'tvd':float(np.abs(p-q).sum()/2),
         'correct':float(p.argmax()==q.argmax()),'entropy':float(-(p[p>0]*np.log(p[p>0])).sum()),'false_zero':int(any((q>0)&(p==0))),
         'zero_target_mass':float(p[q==0].sum())}

def parse_source(row):
 text=row['input']['state'];labels=row['input']['labels'];fam=row['family']
 found=re.findall(r'([a-z]+): sample I has (\d+), sample II has (\d+), archive has (\d+), detection rate is (\d+)/10',text)
 if found:
  if [r[0] for r in found]!=labels:raise ValueError('Text category order')
  a=[int(r[1]) for r in found];b=[int(r[2]) for r in found];rates=[int(r[4]) for r in found]
  match=re.search(r'Select sample I with probability (\d+)/10',text);weight=int(match[1]) if match else None
 else:
  def section(name):
   match=re.search(re.escape(name)+r'([^.]*)',text)
   if not match:raise ValueError('Missing source section: '+name)
   vals=dict((s,int(n)) for s,n in re.findall(r'([a-z]+)=(\d+)',match[1]))
   if set(vals)!=set(labels):raise ValueError('Source section domain')
   return [vals[s] for s in labels]
  a=section({'counts':'ACTIVE bag:','conditional':'FLAGGED counts:','mixture':'L counts:','bayes':'Initial class counts:'}[fam])
  b=section('R counts:') if fam=='mixture' else None
  rates=section('Pass probabilities have numerators ') if fam=='bayes' else None
  weight=int(re.search(r'Choose L with probability (\d+)/10',text)[1]) if fam=='mixture' else None
 # Independent integer multiplicities of eligible elementary events.
 if fam in ('counts','conditional'):counts=a
 elif fam=='bayes':counts=[sum(int(u<rate) for _ in range(n) for u in range(10)) for n,rate in zip(a,rates)]
 else:
  counts=[0]*len(a)
  for selector in range(10):
   pool=a if selector<weight else b;rep=sum(b) if selector<weight else sum(a)
   for j,n in enumerate(pool):counts[j]+=n*rep
 q=[F(n,sum(counts)) for n in counts]
 if row['quantity']=='mode':
  if q.count(max(q))!=1:raise ValueError('Nonunique modal target')
  q=[F(int(i==q.index(max(q)))) for i in range(len(q))]
 return q

def resample_comparison(values,rows,seed=SEED,draws=10000):
 # values is paired per-seed difference, shape [3, N]; baseline replicated only for pairing.
 values=np.asarray(values,dtype=float);strata={}
 for i,r in enumerate(rows):strata.setdefault(r['family'],[]).append(i)
 rng=np.random.default_rng(seed);ss=rng.integers(0,3,size=(draws,3));boot=np.zeros(draws)
 for family,idx in sorted(strata.items()):
  selected=np.asarray(idx)[rng.integers(len(idx),size=(draws,len(idx)))];sums=np.zeros(draws)
  for j in range(3):sums+=values[ss[:,j,None],selected].sum(axis=1)
  boot+=sums/3
 boot/=len(rows)
 return {'mean_difference':float(values.mean()),'ci95':list(map(float,np.quantile(boot,[.025,.975]))),'seed_differences':list(map(float,values.mean(1))),
         'worlds_or_items':len(rows),'seeds':3,'strata':{k:len(v) for k,v in strata.items()},'resamples':draws,
         'limitations':'Descriptive crossed paired-seed/within-family-world bootstrap; no multiplicity correction or calibration uncertainty propagation.'}

def summarize(ms):return {k:float(np.mean([m[k] for m in ms])) for k in ms[0]}

def run(root):
 root=Path(root);prep=root/'transfer-prepared';out=root/'results';m=json.load(open(prep/'manifest.json'));ev=json.load(open(prep/'evaluation.json'));jobs=json.load(open(prep/'jobs.json'))
 if digest(ev)!=m['evaluation_hash'] or digest(jobs)!=m['jobs_hash'] or sha(root/'transfer_v18.py')!=m['source_sha256']:raise ValueError('Frozen data/code mismatch')
 raw=json.load(open(root/'all_records.json'));idx={r['id']:r for r in raw};wanted={r['input']['id']:r for r in ev}
 if len(idx)!=len(raw) or set(idx)!=set(wanted) or len(raw)!=176:raise ValueError('Incomplete population')
 source_checked=0;num_checked=0;maxstored=0.;maxdecimal=0.;metricrows=[];predictions=[]
 for r in ev:
  rid=r['input']['id'];obs=idx[rid];q=list(map(F,r['target']))
  if sum(q)!=1 or min(q)<0:raise ValueError('Target law invalid')
  if r['cohort']=='probability':
   if parse_source(r)!=q:raise ValueError('Source target mismatch')
  else:
   p=root/'transfer-sources'/(r['family']+'.json');rawsource=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(rawsource)).encode()+b'\0'+rawsource).hexdigest()
   if blob!=r['source_blob']:raise ValueError('External bytes mismatch')
   ex=json.loads(rawsource)['examples'][r['source_index']]
   if ex['input']!=r['input']['state']:raise ValueError('External text mismatch')
   y=str(ex['target']).lower();labels=r['input']['labels'];gold=[int(s.lower()==y) for s in labels]
   if sum(gold)!=1 or list(map(F,gold))!=q:raise ValueError('External answer mapping')
  source_checked+=1
  if obs['input_sha256']!=digest(r['input']) or set(obs['outputs'])!=set(m['models']):raise ValueError('Output mismatch')
  if len(obs['code_token_ids'])!=len(set(obs['code_token_ids'])) or len(obs['code_token_ids'])!=len(q):raise ValueError('Code map domain')
  for model,output in obs['outputs'].items():
   z=output['logits'];p=probs(z)
   if len(z)!=len(q) or not np.isfinite(z).all():raise ValueError('Invalid output')
   if output['label']!=r['input']['labels'][int(np.argmax(z))]:raise ValueError('Misaligned answer')
   err=float(np.abs(p-output['probabilities']).max());maxstored=max(maxstored,err)
   if err>2e-6:raise ValueError('FP32 probability mismatch')
   with localcontext() as ctx:
    ctx.prec=60;dz=[Decimal(str(x))-Decimal(str(max(z))) for x in z];dz=[x.exp() for x in dz];den=sum(dz);dp=[float(x/den) for x in dz]
   maxdecimal=max(maxdecimal,float(np.abs(p-dp).max()));num_checked+=1
   for cal in ('raw','frozen_calibrated'):
    T=1. if cal=='raw' else m['temperatures'][model][0 if r['quantity']=='event' else 1]
    mm=metric(z,q,T);metricrows.append({'id':rid,'model':model,'calibration':cal,'temperature':T,**mm})
   predictions.append({'id':rid,'model':model,'label':output['label'],'raw_probabilities':list(p),'frozen_calibrated_probabilities':list(probs(z,m['temperatures'][model][0 if r['quantity']=='event' else 1]))})
 # Validate every original shard completion, all checkpoint-anchor tests and source selection.
 receipts=[]
 for i,plan in enumerate(jobs):
  folder=root/'records'/f'transfer-v18-shard-{i}';comp=json.load(open(folder/'complete.json'));pre=json.load(open(folder/'preflight.json'))
  if comp['source_sha256']!=m['source_sha256'] or comp['jobs_hash']!=digest(plan) or comp['records_sha256']!=sha(folder/'records.jsonl'):raise ValueError('Shard corrupt')
  if comp['count']!=len(plan) or set(comp['ids'])!={r['id'] for r in plan}:raise ValueError('Shard incomplete')
  if max(pre['adapter_anchor_errors'].values())>1e-4 or pre['baseline_anchor_error']>1e-4 or comp['baseline_restoration_error']>1e-4:raise ValueError('Anchor failure')
  receipts.append(pre)
 if len({json.dumps(r['model']['weight_hashes'],sort_keys=True) for r in receipts})!=1:raise ValueError('Model weights differ')
 for rec in m['source_receipts']:
  rawsource=(root/'transfer-sources'/(rec['family']+'.json')).read_bytes();xs=json.loads(rawsource)['examples'];order=list(range(len(xs)))
  random.Random(SEED+int(hashlib.sha256(rec['family'].encode()).hexdigest()[:8],16)).shuffle(order)
  if [i for i in order if i not in rec['excluded_indices']][:8]!=rec['selected_indices']:raise ValueError('Selection changed')
  if sha(root/'transfer-sources'/(rec['family']+'.json'))!=rec['sha256']:raise ValueError('Source changed')
 lu={(r['id'],r['model'],r['calibration']):r for r in metricrows}
 cohorts={v:[r for r in ev if r['variant']==v] for v in ('familiar','prose','replication','relevant','mode','external')}
 cohorts.update({'prose_seen_counts':[r for r in cohorts['prose'] if len(r['target'])<=5],
                 'prose_unseen_counts':[r for r in cohorts['prose'] if len(r['target'])>5]})
 summary={'status':'complete','models':m['models'],'cohorts':{},'per_seed':{},'contrasts':{},'limitations':m['calibration']}
 metricnames=('ce','squared','tvd','correct','entropy','false_zero','zero_target_mass')
 for name,rr in cohorts.items():
  if not rr:raise ValueError('Missing cohort')
  summary['cohorts'][name]={'n':len(rr),'raw':{},'frozen_calibrated':{}}
  summary['per_seed'][name]={}
  for cal in ('raw','frozen_calibrated'):
   for arm in ('unchanged',)+ARMS:
    models=['unchanged/0'] if arm=='unchanged' else [f'{arm}/{s}' for s in SEEDS]
    result={met:float(np.mean([lu[r['input']['id'],model,cal][met] for r in rr for model in models])) for met in metricnames}
    summary['cohorts'][name][cal][arm]=result
   uniform=[metric([0]*len(r['target']),list(map(F,r['target']))) for r in rr];summary['cohorts'][name][cal]['uniform']=summarize(uniform)
  for model in m['models']:
   summary['per_seed'][name][model]={cal:{met:float(np.mean([lu[r['input']['id'],model,cal][met] for r in rr])) for met in metricnames} for cal in ('raw','frozen_calibrated')}
  for cal in ('raw','frozen_calibrated'):
   for a,b in [('event_high','unchanged'),('mixed_high','unchanged'),('mixed_high','event_high')]:
    for met in ('ce','squared','correct'):
     v=[[lu[r['input']['id'],f'{a}/{seed}',cal][met]-lu[r['input']['id'],'unchanged/0' if b=='unchanged' else f'{b}/{seed}',cal][met] for r in rr] for seed in SEEDS]
     summary['contrasts'][f'{name}/{cal}/{a}-minus-{b}/{met}']=resample_comparison(v,rr)
 # Mathematical intervention measurements with absolute errors retained above.
 grouped={}
 for r in ev:
  if r['cohort']=='probability':grouped.setdefault(r['group'],{})[r['variant']]=r
 interventions={}
 for model in m['models']:
  interventions[model]={}
  for cal in ('raw','frozen_calibrated'):
   temp=1. if cal=='raw' else m['temperatures'][model][0];vals={'representation_squared':[],'replication_squared':[],'relevant_delta_squared':[]}
   for group in grouped.values():
    def p(v):return probs(idx[group[v]['input']['id']]['outputs'][model]['logits'],temp)
    def q(v):return np.array(list(map(float,map(F,group[v]['target']))))
    vals['representation_squared'].append(float(((p('prose')-p('familiar'))**2).sum()))
    vals['replication_squared'].append(float(((p('replication')-p('prose'))**2).sum()))
    vals['relevant_delta_squared'].append(float(((p('relevant')-p('prose')-(q('relevant')-q('prose')))**2).sum()))
   interventions[model][cal]={key:float(np.mean(v)) for key,v in vals.items()}
 # Each external change remains attributable to one model, never an averaged ensemble answer.
 retention={}
 for model in m['models']:
  fixes=[];breaks=[]
  for r in cohorts['external']:
   rid=r['input']['id'];a=lu[rid,model,'raw']['correct'];b=lu[rid,'unchanged/0','raw']['correct']
   if a>b:fixes.append(rid)
   if a<b:breaks.append(rid)
  retention[model]={'correct':int(sum(lu[r['input']['id'],model,'raw']['correct'] for r in cohorts['external'])),'n':56,'repair_ids':fixes,'regression_ids':breaks}
 cost={}
 for model in m['models']:
  vals=[r['outputs'][model]['seconds'] for r in raw];cost[model]={'mean_forward_seconds':float(np.mean(vals)),'median_forward_seconds':float(np.median(vals)),'sum_forward_seconds':float(np.sum(vals))}
 audit={'source_targets_checked':source_checked,'raw_vectors_checked':num_checked,'all_metric_vectors':len(metricrows),'max_stored_probability_error':maxstored,'max_decimal_probability_error':maxdecimal,'shards':len(receipts),'adapter_anchor_checks':sum(len(r['adapter_anchor_errors']) for r in receipts),'new_generations':0,'new_training_updates':0}
 write(out/'summary.json',summary);write(out/'all_metric_rows.json',metricrows);write(out/'predictions.json',predictions);write(out/'interventions.json',interventions);write(out/'retention.json',retention);write(out/'cost.json',cost);write(out/'audit.json',audit)
 print(json.dumps({'primary':summary['cohorts']['prose'],'retention':retention,'audit':audit},indent=2))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();run(a.root)
