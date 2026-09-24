"""Exact-world matched rendering experiment. Targets never enter model requests."""
from __future__ import annotations
import argparse,copy,hashlib,json,random
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import causal_v17_data as parent
SEED=1909242630
SEEDS=(19101,19102,19103)
ARMS=('single','varied')
STEPS=32
TRAIN_FORMATS=('prose','records','json')
TEST_FORMAT='table'
digest=parent.digest
write=parent.write

def fields(w,noise=True):
 f=w['family']
 if f=='counts':r={'ACTIVE':w['a']}; n='SPARE'
 elif f=='conditional':r={'FLAGGED':w['a']};n='CLEAR'
 elif f=='mixture':r={'L':w['a'],'R':w['b']};n='UNUSED'
 elif f=='bayes':r={'INITIAL':w['a'],'PASS_NUMERATOR':w['rates']};n='UNRELATED'
 else:raise ValueError('Unknown mechanism')
 if noise:r[n]=w['noise']
 return r

def rule(w):
 f=w['family']
 if f=='counts':return 'Draw one item uniformly from ACTIVE only. Any SPARE inventory is excluded.'
 if f=='conditional':return 'Draw uniformly among FLAGGED items only. Any CLEAR items are ineligible.'
 if f=='mixture':return f'First choose L with probability {w["weight"]}/10, otherwise R. Draw uniformly within the chosen container. Any UNUSED inventory is excluded.'
 return 'Select uniformly from INITIAL. PASS_NUMERATOR gives the exact pass likelihood numerator for each category, with denominator ten. The selected item is known to have passed. Any UNRELATED inventory is excluded.'

def render(w,fmt,noise=True):
 cols=fields(w,noise);labels=w['labels']
 prefix=rule(w)+' All listed inventories are exhaustive and exact. The category of the selected item is unobserved.\n'
 if fmt=='prose':
  body=' '.join(name+': '+', '.join(f'{s}={v}' for s,v in zip(labels,a))+'.' for name,a in cols.items())
 elif fmt=='records':
  body='\n'.join('Category '+s+': '+'; '.join(name+'='+str(a[i]) for name,a in reversed(list(cols.items())))+'.' for i,s in enumerate(labels))
 elif fmt=='json':
  body=json.dumps({s:{name:a[i] for name,a in cols.items()} for i,s in enumerate(labels)},sort_keys=True,separators=(',',':'))
 elif fmt=='table':
  names=list(cols)
  body='| Category | '+' | '.join(names)+' |\n| '+ ' | '.join(['---']*(len(names)+1))+' |\n'
  body+='\n'.join('| '+s+' | '+' | '.join(str(cols[name][i]) for name in names)+' |' for i,s in enumerate(labels))
 else:raise ValueError('Unknown rendering')
 return prefix+body

def record(w,fmt='prose',noise=True,variant='base'):
 q=parent.law(w)
 r={'id':f'fmt19-{w["id"]}-{fmt}-n{int(noise)}-{variant}', 'state':render(w,fmt,noise),
    'question':{'type':'choice','instructions':'Return the categorical probability distribution of the unobserved selected item under the stated mechanism and conditioning. Estimate event probabilities, not confidence in the most likely label.',
                'criteria':{s:'The selected category is '+s+'.' for s in w['labels']}},'labels':w['labels'][:]}
 return {'input':json.loads(json.dumps(r,sort_keys=True)),'target':[str(x) for x in q],
         'group':w['id'],'split':w['split'],'family':w['family'],'format':fmt,'noise':bool(noise),'variant':variant,
         'quantity':'event','mechanism':copy.deepcopy(w)}

def build(root):
 root=Path(root);rng=random.Random(SEED);worlds=[]
 for split,reps in [('fit',4),('calibration',1),('test',1)]:
  for fam in parent.FAMILIES:
   for k in (2,3,4,5):
    for entropy in ('sharp','diffuse'):
     for rep in range(reps):
      w=parent.make_world(rng,split,fam,k,entropy,rep if split=='fit' else 1,len(worlds))
      w['id']='fmt19-'+w['id'];worlds.append(w)
 records=[]
 for w in worlds:
  if w['split']=='fit':records.extend(record(w,f,True) for f in TRAIN_FORMATS)
  elif w['split']=='calibration':records.append(record(w))
  else:
   for f in ('prose','table'):
    for noise in (False,True):records.append(record(w,f,noise))
   z=copy.deepcopy(w);z['a']=[5*v for v in z['a']]
   assert parent.law(z)==parent.law(w)
   records.append(record(z,'table',True,'replicate'))
   z=copy.deepcopy(w);z['a'][0]+=19
   assert parent.law(z)!=parent.law(w)
   records.append(record(z,'table',True,'relevant_edit'))
 old=json.loads((root/'transfer-prepared/evaluation.json').read_text())
 for fam in sorted({r['family'] for r in old if r['cohort']=='retention'}):
  cohort=sorted([r for r in old if r['cohort']=='retention' and r['family']==fam],key=lambda r:r['input']['id'])[:4]
  for r in cohort:records.append({**copy.deepcopy(r),'split':'retention','format':'external','noise':None,'variant':'historical_retention'})
 previous=[r.get('input',r) for r in old]
 if (root/'causal-prepared/records.json').exists():previous += [r['input'] for r in json.loads((root/'causal-prepared/records.json').read_text())]
 keys={digest({k:r[k] for k in ('state','question','labels')}) for r in previous}
 for r in records:
  if r['split']!='retention' and digest({k:r['input'][k] for k in ('state','question','labels')}) in keys:raise ValueError('Prior input overlap')
 idx={(r['group'],r['format']):i for i,r in enumerate(records) if r['split']=='fit'}
 cells={}
 for w in worlds:
  if w['split']=='fit':cells.setdefault((w['family'],len(w['a']),w['entropy_bin']),[]).append(w)
 schedules={}
 for seed in SEEDS:
  rr=random.Random(seed);selected=[rr.choice(cells[c]) for c in sorted(cells)];rr.shuffle(selected)
  assignments=list(TRAIN_FORMATS)*11;rr.shuffle(assignments)
  schedules[str(seed)]={arm:[idx[(w['id'],'prose' if arm=='single' else assignments[j])] for j,w in enumerate(selected)] for arm in ARMS}
  assert len(selected)==32 and len({w['id'] for w in selected})==32
 evaluation=[i for i,r in enumerate(records) if r['split']!='fit']
 probe=[i for i,r in enumerate(records) if r['split']=='test' and r['format']=='prose' and r['noise'] and r['variant']=='base' and len(r['input']['labels'])==2]
 manifest={'version':'format-v19.0','seed':SEED,'seeds':SEEDS,'arms':ARMS,'steps':STEPS,'schedules':schedules,
 'worlds_hash':digest(worlds),'records_hash':digest(records),'worlds':len(worlds),'records':len(records),
 'evaluation_indices':evaluation,'probe_indices':probe,'training_formats':TRAIN_FORMATS,'heldout_format':TEST_FORMAT,
 'expected_eval':len(evaluation),'retention_items':28,'primary':'raw table+noise event CE and squared loss; varied minus single at final32',
 'calibration':'one event temperature per checkpoint from 32 new canonical calibration worlds; secondary only',
 'scope':'Same-grammar mathematical mechanisms; table unseen during training; historical 28-item retention subset.'}
 assert len(evaluation)==252 and len(probe)==8
 return worlds,records,manifest

def prepare(root,out):
 w,r,m=build(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 for name,obj in [('worlds',w),('records',r),('manifest',m)]:write(out/(name+'.json'),obj)
 print(json.dumps({k:v for k,v in m.items() if k not in ('schedules','evaluation_indices','probe_indices')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',default='format-prepared');a=p.parse_args();prepare(a.root,a.out)
