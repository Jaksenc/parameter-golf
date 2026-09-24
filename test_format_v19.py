import copy,json,math,re,tempfile,unittest
from pathlib import Path
from fractions import Fraction as F
import format_v19_data as d

def decode(state):
 """Independent parser of the actual rendered controlled sources."""
 header,body=state.split('\n',1);out={}
 if body.startswith('{'):
  rows=json.loads(body)
  for label,fields in rows.items():
   for field,value in fields.items():out.setdefault(field,{})[label]=value
 elif body.startswith('|'):
  lines=body.splitlines();fields=[x.strip() for x in lines[0].strip('|').split('|')][1:]
  for line in lines[2:]:
   cells=[x.strip() for x in line.strip('|').split('|')]
   if len(cells)!=len(fields)+1:raise ValueError('Table width')
   for name,value in zip(fields,cells[1:]):out.setdefault(name,{})[cells[0]]=int(value)
 elif body.startswith('Category '):
  for line in body.splitlines():
   name,rest=line.removeprefix('Category ').split(': ',1)
   for field,value in re.findall(r'([A-Z_]+)=(\d+)',rest):out.setdefault(field,{})[name]=int(value)
 else:
  for part in body.split('. '):
   field,rest=part.split(': ',1)
   out[field]={name:int(n) for name,n in re.findall(r'([a-z]+)=(\d+)',rest)}
 return header,out

def reference(row):
 h,fs=decode(row['state']);labels=row['labels']
 def vec(name):return [fs[name][s] for s in labels]
 if 'ACTIVE' in fs:counts=vec('ACTIVE')
 elif 'FLAGGED' in fs:counts=vec('FLAGGED')
 elif 'INITIAL' in fs:counts=[n*v for n,v in zip(vec('INITIAL'),vec('PASS_NUMERATOR'))]
 else:
  weight=int(re.search(r'probability (\d+)/10',h)[1]);a,b=vec('L'),vec('R')
  counts=[weight*x*sum(b)+(10-weight)*y*sum(a) for x,y in zip(a,b)]
 return [F(n,sum(counts)) for n in counts]

class DataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.w,cls.r,cls.m=d.build('.')
 def test_all_actual_source_targets(self):
  for r in self.r:
   if r['split']!='retention':self.assertEqual(reference(r['input']),list(map(F,r['target'])))
 def test_population(self):self.assertEqual((len(self.w),len(self.r),self.m['expected_eval']),(192,636,252))
 def test_same_world_sequence(self):
  for s,arms in self.m['schedules'].items():
   self.assertEqual([self.r[i]['group'] for i in arms['single']],[self.r[i]['group'] for i in arms['varied']])
 def test_same_targets_and_labels(self):
  for arms in self.m['schedules'].values():
   for i,j in zip(arms['single'],arms['varied']):
    self.assertEqual(self.r[i]['target'],self.r[j]['target']);self.assertEqual(self.r[i]['input']['labels'],self.r[j]['input']['labels'])
 def test_same_source_facts(self):
  for arms in self.m['schedules'].values():
   for i,j in zip(arms['single'],arms['varied']):self.assertEqual(decode(self.r[i]['input']['state']),decode(self.r[j]['input']['state']))
 def test_distinct_fit_worlds(self):
  for arms in self.m['schedules'].values():self.assertEqual(len({self.r[i]['group'] for i in arms['single']}),32)
 def test_unseen_format(self):
  for arms in self.m['schedules'].values():
   for indices in arms.values():self.assertTrue(all(self.r[i]['format']!='table' for i in indices))
 def test_all_train_formats(self):
  for arms in self.m['schedules'].values():self.assertEqual({self.r[i]['format'] for i in arms['varied']},set(d.TRAIN_FORMATS))
 def test_split_is_world_disjoint(self):
  gs={s:{x['group'] for x in self.r if x['split']==s} for s in ('fit','calibration','test')}
  self.assertFalse(gs['fit']&gs['test']);self.assertFalse(gs['fit']&gs['calibration']);self.assertFalse(gs['test']&gs['calibration'])
 def test_unique_input_id(self):self.assertEqual(len({r['input']['id'] for r in self.r}),len(self.r))
 def test_semantics_of_noise(self):
  for w in self.w:
   self.assertEqual(reference(d.record(w,'table',False)['input']),reference(d.record(w,'table',True)['input']))
 def test_factorial_only_noise_field_differs(self):
  for w in self.w:
   _,clean=decode(d.render(w,'table',False));_,noisy=decode(d.render(w,'table',True))
   self.assertEqual({k:noisy[k] for k in clean},clean);self.assertEqual(len(noisy),len(clean)+1)
 def test_rule_unchanged(self):
  for w in self.w:
   self.assertEqual(len({d.render(w,f,n).split('\n')[0] for f in (*d.TRAIN_FORMATS,'table') for n in (False,True)}),1)
 def test_targets_sum(self):
  for r in self.r:self.assertEqual(sum(map(F,r['target'])),1)
 def test_no_labels_in_request_metadata(self):
  for r in self.r:self.assertEqual(set(r['input']),{'id','state','question','labels'})
 def test_retention_count(self):self.assertEqual(sum(r['split']=='retention' for r in self.r),28)
 def test_render_unknown(self):
  with self.assertRaises(ValueError):d.render(self.w[0],'bad')
 def test_no_world_mutation(self):
  w=copy.deepcopy(self.w[0]);before=copy.deepcopy(w)
  for f in (*d.TRAIN_FORMATS,'table'):d.record(w,f)
  self.assertEqual(w,before)
 def test_unseen_random_worlds(self):
  import random
  rng=random.Random(775)
  for i in range(40):
   w=d.parent.make_world(rng,'check',d.parent.FAMILIES[i%4],2+i%4,'diffuse',1,i)
   for f in (*d.TRAIN_FORMATS,'table'):self.assertEqual(reference(d.record(w,f)['input']),d.parent.law(w))

class OptimizationTests(unittest.TestCase):
 def test_event_weight(self):
  import torch
  z=torch.tensor([.2,-.4],requires_grad=True);q=torch.tensor([.3,.7]);loss=.5*(-q*torch.log_softmax(z,-1)).sum();loss.backward()
  self.assertTrue(torch.allclose(z.grad,.5*(torch.softmax(z.detach(),-1)-q)))
 def test_full_checkpoint_resumption(self):
  import torch
  import causal_v17_state as ck
  torch.manual_seed(91);a=torch.nn.Linear(2,2);b=copy.deepcopy(a)
  oa=torch.optim.AdamW(a.parameters(),lr=.01);ob=torch.optim.AdamW(b.parameters(),lr=.01)
  def step(m,o):
   o.zero_grad();m(torch.ones(1,2)).square().sum().backward();o.step()
  for _ in range(3):step(a,oa);step(b,ob)
  with tempfile.TemporaryDirectory() as td:
   ck.save(td,b,ob,completed_steps=3,bindings={'x':'test'},schedule_state={'history':[1,2,3]})
   fresh=copy.deepcopy(a);opt=torch.optim.AdamW(fresh.parameters(),lr=.01)
   data=ck.load(td,fresh,opt,bindings={'x':'test'})
   self.assertEqual(data['schedule_state']['history'],[1,2,3])
   for _ in range(3):step(a,oa);step(fresh,opt)
   self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),fresh.parameters())))
 def test_replication_counterexample(self):
  a,b=[9,1],[1,9];q=[F(4,10)*F(x,10)+F(6,10)*F(y,10) for x,y in zip(a,b)]
  wrong=[4*(10*x)+6*y for x,y in zip(a,b)]
  self.assertEqual(q[0],F(42,100));self.assertGreater(wrong[0],wrong[1])

if __name__=='__main__':unittest.main()
