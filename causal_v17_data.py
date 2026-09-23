"""Frozen, balanced probability worlds for a matched objective/step-size pilot.
Exact targets are produced from mechanisms; model input contains no oracle note.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
SEED=170923261
SEEDS=(17101,17102,17103)
STEPS=32
FAMILIES=('counts','conditional','mixture','bayes')
ARMS={'event_high':(0.,1e-4),'mixed_high':(1.,1e-4),'event_low':(0.,1e-5),'mixed_low':(1.,1e-5)}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False))
def law(w):
 a=w['a'];b=w['b'];f=w['family'];n=len(a)
 if f in ('counts','conditional'):return [F(x,sum(a)) for x in a]
 if f=='mixture':return [F(w['weight'],10)*F(a[i],sum(a))+F(10-w['weight'],10)*F(b[i],sum(b)) for i in range(n)]
 z=[x*r for x,r in zip(a,w['rates'])];return [F(x,sum(z)) for x in z]
def wrong_law(w):
 if w['family']=='counts':z=w['noise']
 elif w['family']=='conditional':z=[x+y for x,y in zip(w['a'],w['noise'])]
 elif w['family']=='mixture':z=[w['weight']*x+(10-w['weight'])*y for x,y in zip(w['a'],w['b'])]
 else:z=w['a']
 return [F(x,sum(z)) for x in z]
def mode(q):return max(range(len(q)),key=lambda i:q[i])
def entropy(q):return -sum(float(x)*math.log(float(x)) for x in q if x)/math.log(len(q))
def render(w,paraphrase=False):
 table=lambda a:', '.join(f'{s}={v}' for s,v in zip(w['labels'],a))
 f=w['family']
 if f=='counts':
  if paraphrase:return f"Draw uniformly from the in-use container, never the reserve. In-use inventory: {table(w['a'])}. Reserve inventory: {table(w['noise'])}. These are exhaustive exact counts; the draw is unobserved."
  return f"ACTIVE bag: {table(w['a'])}. SPARE bag: {table(w['noise'])}. Draw one ticket uniformly from ACTIVE only. Counts are exact and exhaustive. Color is unobserved."
 if f=='conditional':
  if paraphrase:return f"Only certified items are eligible, all equally likely. Certified inventory: {table(w['a'])}. Uncertified inventory: {table(w['noise'])}. No other items exist, and no color has been observed."
  return f"FLAGGED counts: {table(w['a'])}. CLEAR counts: {table(w['noise'])}. Select one item uniformly among FLAGGED items only. Counts are exact and exhaustive; its color is unobserved."
 if f=='mixture':
  if paraphrase:return f"A two-stage draw first chooses container X with probability {w['weight']}/10, otherwise Y. It then chooses uniformly within that container. X inventory: {table(w['a'])}. Y inventory: {table(w['b'])}. Ignored inventory elsewhere: {table(w['noise'])}. All values are exact. Neither choice has been observed."
  return f"Choose L with probability {w['weight']}/10, otherwise R; then draw uniformly within the chosen bag. L counts: {table(w['a'])}. R counts: {table(w['b'])}. An UNUSED bag: {table(w['noise'])}. Counts and selection weights are exact; no color is observed."
 if paraphrase:return f"Choose uniformly from inventory {table(w['a'])}. Acceptance likelihood numerators out of 10 by color: {table(w['rates'])}. The chosen item is known to be accepted. Unrelated inventory: {table(w['noise'])}. Infer its unobserved color using exact counts and likelihoods."
 return f"Initial class counts: {table(w['a'])}. Pass probabilities have numerators {table(w['rates'])}, each over 10. Select uniformly from the initial items; the selected item is known to PASS. Unrelated counts: {table(w['noise'])}. All counts and rates are exact; color is unobserved."
def record(w,question='event',variant='base',split=None):
 q=law(w);k=len(q);m=mode(q)
 instruction=('Return the distribution over the COLOR OF THE UNOBSERVED DRAW under the stated mechanism and conditioning. Report event probabilities, not certainty about the modal label.' if question=='event' else
              'Which color has the UNIQUE LARGEST event probability under these exact parameters? Return the distribution over the correct ANSWER to that modal-category question, not the random draw.')
 p=q if question=='event' else [F(int(i==m)) for i in range(k)]
 row={'id':f"v17-{w['id']}-{variant}-{question}",'state':render(w,variant=='paraphrase'),'question':{'type':'choice','instructions':instruction,'criteria':{s:'The color is '+s+'.' for s in w['labels']}},'labels':w['labels'][:]}
 return {'input':json.loads(json.dumps(row,sort_keys=True)),'target':[str(x) for x in p],'group':w['id'],'split':split or w['split'],'family':w['family'],'question_type':question,'variant':variant,'entropy_bin':w['entropy_bin'],'mechanism':copy.deepcopy(w)}
def make_world(rng,split,family,k,bin_name,rep,index):
 for attempt in range(100000):
  w={'family':family,'labels':rng.sample(['amber','cobalt','jade','violet','ochre','silver','ivory'],k),
     'a':[rng.randint(1,35) for _ in range(k)],'b':[rng.randint(1,35) for _ in range(k)],'noise':[rng.randint(1,60) for _ in range(k)],
     'rates':[rng.randint(1,9) for _ in range(k)],'weight':rng.randint(1,9)}
  if bin_name=='sharp':w['a'][rng.randrange(k)]*=rng.randint(4,12)
  q=law(w);h=entropy(q)
  if q.count(max(q))!=1:continue
  if not (.3<=h<=.75 if bin_name=='sharp' else .90<=h<=.995):continue
  # Half the worlds explicitly discriminate the intended operation from a shortcut.
  wrong=wrong_law(w)
  if rep%2==1 and (wrong.count(max(wrong))!=1 or mode(q)==mode(wrong)):continue
  # Balance modal output slots across strata without changing the law.
  shift=(mode(q)-(index%k))%k
  for a in ('a','b','noise','rates','labels'):w[a]=w[a][shift:]+w[a][:shift]
  w.update(id=f'{split}-{family}-k{k}-{bin_name}-{rep}',split=split,entropy_bin=bin_name)
  assert mode(law(w))==index%k
  return w
 raise RuntimeError('Unable to make prespecified world')
def build():
 rng=random.Random(SEED);worlds=[]
 for split,reps in [('fit',4),('calibration',1),('test',1)]:
  for family in FAMILIES:
   for k in (2,3,4,5):
    for bin_name in ('sharp','diffuse'):
     for rep in range(reps):
      # All test/cal worlds are also discriminating, without selecting on model outcomes.
      w=make_world(rng,split,family,k,bin_name,rep if split=='fit' else 1,len(worlds))
      worlds.append(w)
 records=[]
 for w in worlds:
  records.extend([record(w,'event'),record(w,'mode')])
  if w['split']=='test':
   z=copy.deepcopy(w);z['a']=[4*x for x in z['a']];assert law(z)==law(w);records.append(record(z,'event','replicate'))
   z=copy.deepcopy(w);z['noise']=[7*x+11 for x in z['noise']];assert law(z)==law(w);records.append(record(z,'event','distractor'))
   z=copy.deepcopy(w);z['a'][0]+=17;assert law(z)!=law(w);records.append(record(z,'event','relevant_edit'))
   records.append(record(w,'event','paraphrase'))
 indices={r['input']['id']:i for i,r in enumerate(records)}
 schedules={}
 cells={}
 for w in worlds:
  if w['split']=='fit':cells.setdefault((w['family'],len(w['a']),w['entropy_bin']),[]).append(w)
 for seed in SEEDS:
  rr=random.Random(seed);order=[]
  for cell in sorted(cells):
   w=rr.choice(cells[cell]);order.append([indices[record(w,q)['input']['id']] for q in ('event','mode')])
  rr.shuffle(order);assert len(order)==STEPS;schedules[str(seed)]=order
 # Eight fixed probe worlds; probes never determine an optimizer choice.
 probe_worlds=[w for w in worlds if w['split']=='test' and len(w['a'])==2]
 probe=[indices[record(w,q)['input']['id']] for w in probe_worlds for q in ('event','mode')]
 evaluation=[i for i,r in enumerate(records) if r['split']!='fit']
 keys=[digest({k:r['input'][k] for k in ('state','question','labels')}) for r in records]
 assert len(keys)==len(set(keys))
 manifest={'version':'causal-v17.0','worlds':len(worlds),'fit_worlds':128,'calibration_worlds':32,'test_worlds':32,
           'records':len(records),'record_hash':digest(records),'world_hash':digest(worlds),'steps':STEPS,'seeds':list(SEEDS),
           'arms':ARMS,'schedules':schedules,'evaluation_indices':evaluation,'probe_indices':probe,
           'notes':'32 unique fit worlds per seed, selected one per stratum. Four arms share each seed schedule. Final test never determines parameters.'}
 return worlds,records,manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='causal-prepared');a=p.parse_args();w,r,m=build()
 write(Path(a.out)/'worlds.json',w);write(Path(a.out)/'records.json',r);write(Path(a.out)/'manifest.json',m)
 print(json.dumps({k:v for k,v in m.items() if k not in ('schedules','evaluation_indices')},indent=2))
if __name__=='__main__':main()
