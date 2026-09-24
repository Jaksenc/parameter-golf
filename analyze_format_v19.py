"""Independent all-output reducer: no pretrained model calls or selection by test score."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from test_format_v19 import reference,decode

SEEDS=(19101,19102,19103)
ARMS=('single','varied')
KEYS=('unchanged',)+tuple(f'{s}-{a}' for s in SEEDS for a in ARMS)
GRID=np.exp2(np.linspace(-2,4,121))

def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))
def probs(z,t=1.):
 z=np.asarray(z,dtype=np.float64)/t;z=z-z.max();w=np.exp(z);return w/w.sum()
def loss(z,q,t=1.):
 z=np.asarray(z,dtype=np.float64)/t;z-=z.max();l=z-np.log(np.exp(z).sum());p=np.exp(l);q=np.asarray(q)
 return {'ce':float(-q@l),'squared':float(((p-q)**2).sum()),'tvd':float(np.abs(p-q).sum()/2),
         'correct':int(np.argmax(p)==np.argmax(q)),'entropy':float(-np.dot(p,l))}
def independent_probs(z):
 with localcontext() as c:
  c.prec=60;v=[Decimal(str(x)) for x in z];m=max(v);e=[(x-m).exp() for x in v];s=sum(e)
  return np.asarray([float(x/s) for x in e])
def mean_metrics(values):
 return {key:float(np.mean([v[key] for v in values])) for key in values[0]} if values else {}

def interval(arr,groups,draws=10000):
 """arr=[paired seeds, unique worlds], strata preserve the family counts."""
 arr=np.asarray(arr,dtype=float);rng=np.random.default_rng(19240923);boots=np.zeros(draws);den=0
 seedix=rng.integers(arr.shape[0],size=(draws,arr.shape[0]))
 for family in sorted(set(groups)):
  ids=np.flatnonzero(np.asarray(groups)==family);worldix=rng.choice(ids,size=(draws,len(ids)),replace=True)
  sampled=arr[seedix[:,:,None],worldix[:,None,:]]
  boots+=sampled.sum(axis=(1,2));den+=len(ids)*arr.shape[0]
 return {'mean':float(arr.mean()),'interval95':list(map(float,np.quantile(boots/den,[.025,.975]))),'draws':draws,
         'unique_worlds':arr.shape[1],'paired_seeds':arr.shape[0]}

def analyze(root):
 root=Path(root);data=read(root/'format-prepared/records.json');m=read(root/'format-prepared/manifest.json');allrows=read(root/'all_records.json')
 if digest(data)!=m['records_hash']:raise ValueError('Data hash')
 wanted=set(m['evaluation_indices']);seen=set();index={};fp_error=0.;dec_error=0.;nvec=0;reconstructed=0
 for row in data:
  if row['split']!='retention':
   if reference(row['input'])!=list(map(Fraction,row['target'])):raise ValueError('Source target disagreement')
   reconstructed+=1
 for row in allrows:
  i=row['index']
  if i not in wanted or i in seen or set(row['outputs'])!=set(KEYS):raise ValueError('Missing/duplicate state')
  if row['id']!=data[i]['input']['id'] or row['input_sha256']!=digest(data[i]['input']):raise ValueError('Input mapping')
  if len(set(row['code_ids']))!=len(data[i]['target']) or row['restoration_error']>1e-4:raise ValueError('Code/restoration')
  for key,o in row['outputs'].items():
   if len(o['logits'])!=len(data[i]['target']) or not all(math.isfinite(x) for x in o['logits']):raise ValueError('Invalid vector')
   p=probs(o['logits']);fp_error=max(fp_error,float(np.max(abs(p-np.asarray(o['probabilities'])))))
   dec_error=max(dec_error,float(np.max(abs(p-independent_probs(o['logits'])))));nvec+=1
  seen.add(i);index[i]=row
 if seen!=wanted or nvec!=1764 or fp_error>4e-7 or dec_error>1e-12:raise ValueError('Population/probability mismatch')
 for shard in range(16):
  d=root/'evaluated'/f'format-v19-evaluation-{shard}';c=read(d/'complete.json')
  if c['records_sha256']!=sha(d/'records.jsonl') or c['indices']!=m['evaluation_indices'][shard::16]:raise ValueError('Shard hash')
  rows=[json.loads(x) for x in (d/'records.jsonl').read_text().splitlines()]
  if len(rows)!=c['count'] or any(index[x['index']]!=x for x in rows):raise ValueError('Shard content')
 train={};target_pairs=0
 for seed in SEEDS:
  inits=[]
  for arm in ARMS:
   key=f'{seed}-{arm}';folder=root/'models'/f'format-v19-model-{key}'
   c=read(folder/'complete-32.json');pf=read(folder/'preflight.json');ledger=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
   if c['from_step']!=16 or c['through_step']!=32 or c['adapter_sha256']!=sha(folder/'adapter-32.safetensors'):raise ValueError('Endpoint/checkpoint')
   if [x['step'] for x in ledger]!=list(range(1,33)) or [x['optimizer_step'] for x in ledger]!=list(range(1,33)):raise ValueError('Updates')
   if [x['index'] for x in ledger]!=m['schedules'][str(seed)][arm] or digest(ledger)!=c['history_hash']:raise ValueError('Schedule/history')
   for line in ledger:
    q=[float(Fraction(x)) for x in data[line['index']]['target']];ce=loss(line['logits'],q)['ce']
    if abs(ce-line['ce'])>1e-5 or abs(line['weighted_loss']-.5*line['ce'])>1e-6:raise ValueError('Observed loss')
   inits.append(c['initial_adapter_sha256'])
   train[key]={'steps':32,'unique_worlds':len({x['group'] for x in ledger}),'input_tokens':sum(x['tokens'] for x in ledger),
               'update_seconds':sum(x['seconds'] for x in ledger),'formats':dict(Counter(x['format'] for x in ledger)),
               'checkpoint_sha256':c['adapter_sha256'],'base_restoration_error':c['base_restoration_error']}
  if len(set(inits))!=1:raise ValueError('Unpaired initialization')
  for i,j in zip(m['schedules'][str(seed)]['single'],m['schedules'][str(seed)]['varied']):
   if data[i]['group']!=data[j]['group'] or data[i]['target']!=data[j]['target'] or decode(data[i]['input']['state'])!=decode(data[j]['input']['state']):raise ValueError('Unequal data treatment')
   target_pairs+=1
 calibration=[i for i in wanted if data[i]['split']=='calibration']
 if len(calibration)!=32:raise ValueError('Calibration count')
 temperatures={};grid_losses={}
 for key in KEYS:
  curve=[float(np.mean([loss(index[i]['outputs'][key]['logits'],list(map(lambda x:float(Fraction(x)),data[i]['target'])),t)['ce'] for i in calibration])) for t in GRID]
  temperatures[key]=float(GRID[int(np.argmin(curve))]);grid_losses[key]=curve
 scored={};vectors={}
 for scale in ('raw','calibrated'):
  scored[scale]={};vectors[scale]={}
  for i in wanted:
   q=[float(Fraction(x)) for x in data[i]['target']];scored[scale][i]={};vectors[scale][i]={}
   for key in KEYS:
    t=temperatures[key] if scale=='calibrated' and data[i]['split']!='retention' else 1.
    z=index[i]['outputs'][key]['logits'];scored[scale][i][key]=loss(z,q,t);vectors[scale][i][key]=probs(z,t)
   scored[scale][i]['uniform']=loss([0.]*len(q),q)
 def ids_for(fmt,noise):return sorted(i for i in wanted if data[i]['split']=='test' and data[i]['variant']=='base' and data[i]['format']==fmt and data[i]['noise']==noise)
 cells={f'{fmt}_noise{int(noise)}':ids_for(fmt,noise) for fmt in ('prose','table') for noise in (False,True)}
 primary=cells['table_noise1'];summary={'cells':{},'comparisons':{},'calibration':temperatures,'training':train,'scope':m['scope']}
 for scale in scored:
  summary['cells'][scale]={}
  for cell,ids in cells.items():
   modelmetrics={key:mean_metrics([scored[scale][i][key] for i in ids]) for key in (*KEYS,'uniform')}
   for arm in ARMS:modelmetrics[arm+'_mean']=mean_metrics([scored[scale][i][f'{s}-{arm}'] for s in SEEDS for i in ids])
   summary['cells'][scale][cell]=modelmetrics
  for a,b in [('varied','single'),('single','unchanged'),('varied','unchanged')]:
   for metric in ('ce','squared','correct'):
    arr=[[scored[scale][i][f'{s}-{a}'][metric]-scored[scale][i][f'{s}-{b}' if b!='unchanged' else b][metric] for i in primary] for s in SEEDS]
    summary['comparisons'][f'{scale}/{a}-{b}/{metric}']=interval(arr,[data[i]['family'] for i in primary])
 worldmap={}
 for i in wanted:
  r=data[i]
  if r['split']=='test':worldmap.setdefault(r['group'],{})[(r['format'],r['noise'],r['variant'])]=i
 diagnostics={}
 for scale in scored:
  diagnostics[scale]={}
  for key in KEYS:
   vals={x:[] for x in ['format_effect_ce','noise_effect_ce','interaction_ce','replication_squared','relevant_delta_squared','format_squared','noise_squared']}
   for group,lookup in worldmap.items():
    a=lookup['prose',False,'base'];b=lookup['table',False,'base'];c=lookup['prose',True,'base'];e=lookup['table',True,'base'];rep=lookup['table',True,'replicate'];edit=lookup['table',True,'relevant_edit']
    ce=lambda i:scored[scale][i][key]['ce'];p=lambda i:vectors[scale][i][key]
    vals['format_effect_ce'].append(ce(b)-ce(a));vals['noise_effect_ce'].append(ce(c)-ce(a));vals['interaction_ce'].append(ce(e)-ce(b)-ce(c)+ce(a))
    vals['replication_squared'].append(float(((p(rep)-p(e))**2).sum()))
    delta=np.array([float(Fraction(x)) for x in data[edit]['target']])-np.array([float(Fraction(x)) for x in data[e]['target']])
    vals['relevant_delta_squared'].append(float(((p(edit)-p(e)-delta)**2).sum()))
    vals['format_squared'].append(float(((p(b)-p(a))**2).sum()));vals['noise_squared'].append(float(((p(c)-p(a))**2).sum()))
   diagnostics[scale][key]={k:float(np.mean(v)) for k,v in vals.items()}
 retention_ids=sorted(i for i in wanted if data[i]['split']=='retention');retention={}
 for key in KEYS:
  y=[int(np.argmax([float(Fraction(x)) for x in data[i]['target']])) for i in retention_ids]
  preds=[int(np.argmax(vectors['raw'][i][key])) for i in retention_ids];basepred=[int(np.argmax(vectors['raw'][i]['unchanged'])) for i in retention_ids]
  retention[key]={'n':len(y),'correct':sum(a==b for a,b in zip(preds,y)),
   'repairs':[data[i]['input']['id'] for i,a,b,g in zip(retention_ids,preds,basepred,y) if a==g and b!=g],
   'regressions':[data[i]['input']['id'] for i,a,b,g in zip(retention_ids,preds,basepred,y) if a!=g and b==g],
   'metrics':mean_metrics([scored['raw'][i][key] for i in retention_ids])}
 timing={key:mean_metrics([{'seconds':index[i]['outputs'][key]['seconds'],'tokens':index[i]['tokens']} for i in wanted]) for key in KEYS}
 output=root/'results'
 save(output/'summary.json',summary);save(output/'diagnostics.json',diagnostics);save(output/'retention.json',retention);save(output/'calibration_grid.json',{'temperatures':temperatures,'grid':GRID.tolist(),'losses':grid_losses})
 save(output/'all_metrics.json',{s:{str(i):v for i,v in rows.items()} for s,rows in scored.items()});save(output/'timing.json',timing)
 save(output/'audit.json',{'complete':True,'source_targets_reconstructed':reconstructed,'probability_vectors':nvec,'max_fp32_difference':fp_error,'max_decimal_difference':dec_error,
  'training_updates':sum(x['steps'] for x in train.values()),'matched_world_presentations':target_pairs,'checkpoints':len(train),'evaluation_requests':len(index),
  'scope':'Independent recorded-data/source arithmetic; no pretrained rerun; probes excluded from capability sample.'})
 print(json.dumps({'primary':summary['cells']['raw']['table_noise1'],'comparisons':summary['comparisons'],'retention':{k:v['correct'] for k,v in retention.items()}},indent=2))
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();analyze(a.root)
