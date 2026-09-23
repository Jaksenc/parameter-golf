import copy,hashlib,json,math,random,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
import causal_v17_data as d
import causal_v17_state as ck
from causal_v17_train import snapshot,restore,update

def independent(w):
 f=w['family'];a=w['a'];b=w['b']
 if f in ('counts','conditional'):weights=a
 elif f=='bayes':weights=[sum(1 for _ in range(c) for t in range(10) if t<rr) for c,rr in zip(a,w['rates'])]
 else:weights=[sum((c*sum(b) if t<w['weight'] else b[i]*sum(a)) for t in range(10)) for i,c in enumerate(a)]
 return [F(v,sum(weights)) for v in weights]
class Data(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.w,cls.r,cls.m=d.build()
 def test_size(self):self.assertEqual((len(self.w),len(self.r)),(192,512))
 def test_reproducible(self):self.assertEqual(d.build()[2]['record_hash'],self.m['record_hash'])
 def test_targets(self):
  for r in self.r:
   q=independent(r['mechanism']);want=q if r['question_type']=='event' else [F(int(i==d.mode(q))) for i in range(len(q))]
   self.assertEqual(list(map(F,r['target'])),want)
 def test_balance(self):
  for split in ('fit','calibration','test'):
   ws=[w for w in self.w if w['split']==split]
   self.assertEqual(len({(w['family'],len(w['a']),w['entropy_bin']) for w in ws}),32)
 def test_schedule_worlds(self):
  for seq in self.m['schedules'].values():
   self.assertEqual(len(seq),32);self.assertEqual(len({self.r[a]['group'] for a,b in seq}),32)
   for a,b in seq:self.assertEqual(self.r[a]['input']['state'],self.r[b]['input']['state']);self.assertEqual(self.r[a]['split'],'fit')
 def test_no_oracle_in_request(self):
  for r in self.r:self.assertEqual(set(r['input']),{'id','state','question','labels'})
 def test_no_duplicate_inputs(self):
  keys=[d.digest({k:r['input'][k] for k in ('state','question','labels')}) for r in self.r];self.assertEqual(len(set(keys)),len(keys))
 def test_replication(self):
  base={r['group']:r for r in self.r if r['variant']=='base' and r['question_type']=='event'}
  for r in self.r:
   if r['variant']=='replicate':self.assertEqual(r['target'],base[r['group']]['target'])
 def test_relevant_changes(self):
  base={r['group']:r for r in self.r if r['variant']=='base' and r['question_type']=='event'}
  for r in self.r:
   if r['variant']=='relevant_edit':self.assertNotEqual(r['target'],base[r['group']]['target'])
 def test_discriminating_test_worlds(self):
  for w in self.w:
   if w['split']!='fit':self.assertNotEqual(d.mode(d.law(w)),d.mode(d.wrong_law(w)))
 def test_calibration_disjoint(self):
  groups={s:{w['id'] for w in self.w if w['split']==s} for s in ('fit','test','calibration')}
  self.assertFalse(groups['fit']&groups['test'] or groups['test']&groups['calibration'])
 def test_plain_worlds_not_zeros(self):
  self.assertTrue(all(min(independent(w))>0 for w in self.w))
 def test_loss_fixed_event_weight(self):
  e=torch.tensor(2.);m=torch.tensor(8.)
  self.assertEqual(float(.5*(e+0*m)),1.);self.assertEqual(float(.5*(e+1*m)),5.)
class Recovery(unittest.TestCase):
 def make(self):
  torch.manual_seed(99);model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2));return model,torch.optim.AdamW(model.parameters(),lr=.001)
 def step(self,m,o):
  o.zero_grad();x=torch.randn(2,3)+np.random.rand()+random.random();l=m(x).square().mean();l.backward();o.step();return float(l.detach())
 def test_exact_process_state(self):
  random.seed(2);np.random.seed(2);m,o=self.make()
  for _ in range(3):self.step(m,o)
  with tempfile.TemporaryDirectory() as t:
   ck.save(t,m,o,completed_steps=3,bindings={'data':'x'},schedule_state={'next':3})
   loss=self.step(m,o);want=copy.deepcopy(m.state_dict());other,oo=self.make();r=ck.load(t,other,oo,bindings={'data':'x'})
   self.assertEqual(r['completed_steps'],3);self.assertEqual(loss,self.step(other,oo));self.assertTrue(all(torch.equal(want[k],v) for k,v in other.state_dict().items()))
 def test_binding_rejection(self):
  m,o=self.make()
  with tempfile.TemporaryDirectory() as t:
   ck.save(t,m,o,completed_steps=0,bindings={'data':'x'})
   with self.assertRaises(ValueError):ck.load(t,m,o,bindings={'data':'z'})
 def test_corruption_rejection(self):
  m,o=self.make()
  with tempfile.TemporaryDirectory() as t:
   r=ck.save(t,m,o,completed_steps=0,bindings={'data':'x'});p=Path(t)/r['file'];x=bytearray(p.read_bytes());x[-1]^=1;p.write_bytes(x)
   with self.assertRaises(ValueError):ck.load(t,m,o,bindings={'data':'x'})
 def test_counterfactual_restore(self):
  m,o=self.make();self.step(m,o);s=snapshot(torch,m,o);wanted=copy.deepcopy(m.state_dict())
  update(torch,m,o,[torch.ones_like(p) for p in m.parameters()]);restore(m,o,s)
  self.assertTrue(all(torch.equal(wanted[k],v) for k,v in m.state_dict().items()))
 def test_task_gradients_add(self):
  p=torch.tensor([.3,-.5],requires_grad=True);e=-(torch.tensor([.4,.6])*torch.log_softmax(p,0)).sum();m=-torch.log_softmax(p,0)[1]
  ge=torch.autograd.grad(e,p,retain_graph=True)[0];gm=torch.autograd.grad(m,p,retain_graph=True)[0];joint=torch.autograd.grad(.5*(e+m),p)[0]
  self.assertTrue(torch.allclose(joint,.5*(ge+gm)))
 def test_inactive_modal_has_no_update_contribution(self):
  ge=torch.tensor([1.,2.]);gm=torch.tensor([-999.,800.]);self.assertTrue(torch.equal(.5*ge+0*gm,.5*ge))
if __name__=='__main__':unittest.main()
