import copy,json,re,unittest,sys,math
from datetime import datetime,timedelta
from cases import corpus,checks,visible,digest,FAMILIES
from study import target_for,PROTOCOL,assignment,metrics,verified_data
from prompt_variants import messages

class DataTests(unittest.TestCase):
 def test_freeze(self):
  r=checks();self.assertEqual(r['hashes'],PROTOCOL['data_hashes'])
 def test_no_visible_refs(self):
  for r in corpus('train')+corpus('held'):
   original=messages(visible(r),'baseline')
   a=copy.deepcopy(r);a['expected']='POISON';a['oracle']={'SECRET':True};a['target_probs']={};a['source']='CHANGED';a['id']='HIDDEN-ID'
   self.assertEqual(messages(visible(a),'baseline'),original)
 def test_pair_bindings(self):
  for split in ('train','held'):
   rs=corpus(split)
   for group in {r['source'] for r in rs}:
    a,b=[r for r in rs if r['source']==group]
    self.assertEqual(a['options'],b['options']);self.assertEqual(a['question'],b['question'])
    self.assertNotEqual(a['state'],b['state']);self.assertNotEqual(a['target_probs'],b['target_probs'])
 def test_amount_text(self):
  for r in corpus('train')+corpus('held'):
   if r['family']!='amount':continue
   s=r['state'];n,pr=map(int,re.search(r'(\d+) units; unit price (\d+) cents',s).groups());f,c=map(int,re.search(r'fee (\d+) cents; credit (\d+) cents',s).groups());tag=re.search(r'For order ([A-Z]+),',r['question']).group(1);cap=int(re.search(tag+r': cap (\d+) cents\.',s).group(1));total=n*pr+f-c
   self.assertEqual(r['expected'],'below' if total<cap else 'above' if total>cap else 'equal')
 def test_elapsed_text(self):
  for r in corpus('train')+corpus('held'):
   if r['family']!='elapsed':continue
   s=r['state'];start=datetime.fromisoformat(re.search(r'start=([^,]+)',s).group(1));mins=int(re.search(r'interval=(\d+)',s).group(1));end=datetime.fromisoformat(re.search(r'submitted=([^\.]+)',s).group(1));delta=(end-(start+timedelta(minutes=mins))).total_seconds()
   self.assertEqual(r['expected'],'before' if delta<0 else 'after' if delta>0 else 'at')
 def test_policy_text(self):
  for r in corpus('train')+corpus('held'):
   if r['family']!='policy':continue
   tag=re.search(r'Current rule for ([A-Z]+):',r['question']).group(1)
   facts={field:re.search(tag+r': '+field+r'=(\w+)',r['state']).group(1) for field in ('active','hold','clearance')}
   admissible=[]
   import itertools
   for a,h,c in itertools.product((True,False),repeat=3):
    if all(v=='unrecorded' or (v=='yes')==b for v,b in zip(facts.values(),(a,h,c))):admissible.append(a and (not h or c))
   ans='allow' if all(admissible) else 'deny' if not any(admissible) else 'unknown'
   self.assertEqual(r['expected'],ans)
 def test_join_text(self):
  for r in corpus('train')+corpus('held'):
   if r['family']!='join':continue
   person=re.search(r'assigned to (\w+)\?',r['question']).group(1);team=re.search(person+r' belongs to (\w+)\.',r['state']).group(1);dest=re.search(r'Team '+team+r' uses destination (\w+)\.',r['state']).group(1)
   self.assertEqual(r['expected'],dest)
 def test_probability_text(self):
  for r in corpus('train')+corpus('held'):
   if r['family']!='probability':continue
   counts=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',r['state'].split('Quarantined')[0]));total=sum(counts.values())
   self.assertEqual(set(counts),set(r['target_probs']))
   for k in counts:self.assertAlmostEqual(r['target_probs'][k],counts[k]/total)
 def test_cardinality_and_families(self):
  for split in ('train','held'):
   rs=corpus(split);self.assertEqual({r['family'] for r in rs},set(FAMILIES));self.assertTrue({3,4,6} <= {len(r['options']) for r in rs})
 def test_reference_teacher_control(self):
  r=next(r for r in corpus('train') if r['teacher_eligible']);labels=[o['id'] for o in r['options']];n=len(labels);p=[.1/(n-1)]*n;p[labels.index(r['expected'])]=.9
  t={'teacher':{'probabilities':p,'predicted':r['expected']}}
  q,use,w=target_for(r,t,'reference');self.assertEqual(w,0);self.assertEqual(q,[r['target_probs'][k] for k in labels])
  q,use,w=target_for(r,t,'repeat_distill');self.assertEqual(w,.3);self.assertAlmostEqual(sum(q),1);self.assertAlmostEqual(max(q),.97)
 def test_wrong_teacher_does_not_replace_reference(self):
  r=next(r for r in corpus('train') if r['teacher_eligible']);labels=[o['id'] for o in r['options']];wrong=next(x for x in labels if x!=r['expected']);p=[float(x==wrong) for x in labels]
  q,use,w=target_for(r,{'teacher':{'probabilities':p,'predicted':wrong}},'repeat_distill');self.assertEqual(w,0);self.assertFalse(use)
 def test_numerical_teacher_excluded(self):
  for r in corpus('train'):
   if r['family'] not in ('amount','elapsed','probability','judge'):continue
   p=[r['target_probs'][o['id']] for o in r['options']];q,use,w=target_for(r,{'teacher':{'probabilities':p,'predicted':r['expected']}},'repeat_distill');self.assertEqual(w,0)
 def test_tied_teacher_excluded(self):
  r=next(r for r in corpus('train') if r['teacher_eligible']);p=[1/len(r['options'])]*len(r['options']);q,use,w=target_for(r,{'teacher':{'probabilities':p,'predicted':r['expected']}},'repeat_distill');self.assertEqual(w,0)
 def test_balanced_assignment(self):
  rows=corpus('held');a=assignment(rows);self.assertEqual(sorted(i for x in a for i in x),list(range(96)))
 def test_target_not_in_prompt(self):
  for r in corpus('train'):
   text=json.dumps(messages(r,'baseline'));self.assertNotIn('target_probs',text);self.assertNotIn('oracle',text)

if __name__=='__main__':unittest.main()
