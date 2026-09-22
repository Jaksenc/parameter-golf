import itertools, unittest
import torch
import triplet_data as d
import triplet_study as s

class TestTriplet(unittest.TestCase):
 def test_corpus(self):
  report=s.validate_data();self.assertEqual(report['train']['rows'],96);self.assertEqual(report['held']['matched_sources'],32)
 def test_visible_exclusion(self):
  for r in d.corpus('train'): self.assertFalse({'oracle','expected','source','edit','target_probs'} & d.visible(r).keys())
 def test_metadata_randomization(self):
  for fam in d.FAMILIES:
   for i in range(6):
    g=d.make_group('train',fam,i);self.assertEqual(len({r['expected'] for r in g}),3)
    self.assertEqual(len({r['oracle']['key'] for r in g}),1)
 def test_primitive_references(self):
  for split in ('train','held'):
   for r in d.corpus(split):
    if r['family']!='snli':self.assertEqual(d.reference_from_text(r),r['expected'])
 def test_no_discrimination_is_log_factorial(self):
  z=torch.tensor([.3,-.7,1.2],dtype=torch.double)
  self.assertAlmostEqual(float(s.assignment_loss([z]*3,[0,1,2])),__import__('math').log(6),places=12)
 def test_row_column_offsets_cancel(self):
  torch.manual_seed(2);z=torch.randn(3,5,dtype=torch.double);y=[4,0,2]
  a=s.assignment_loss(list(z),y);b=s.assignment_loss(list(z+torch.randn(3,1)+torch.randn(1,5)),y)
  self.assertAlmostEqual(float(a),float(b),places=6)
 def test_permutation_equivariance(self):
  z=torch.tensor([[2.,1.,-1.],[0.,3.,1.],[-1.,1.,4.]],dtype=torch.double)
  original=s.assignment_loss(list(z),[0,1,2])
  for p in itertools.permutations(range(3)):
   new=z[:,p];labels=[p.index(i) for i in range(3)]
   self.assertAlmostEqual(float(original),float(s.assignment_loss(list(new),labels)),places=12)
 def test_input_order_invariance(self):
  z=torch.randn(3,4,dtype=torch.double);y=[2,3,0];p=[2,0,1]
  self.assertAlmostEqual(float(s.assignment_loss(list(z),y)),float(s.assignment_loss([z[i] for i in p],[y[i] for i in p])),places=12)
 def test_gradient_check(self):
  z=torch.randn(3,4,dtype=torch.double,requires_grad=True)
  self.assertTrue(torch.autograd.gradcheck(lambda x:s.assignment_loss(list(x),[0,1,2]),(z,),eps=1e-6))
 def test_two_pass_vjp(self):
  torch.manual_seed(4);w=torch.randn(5,3,dtype=torch.double,requires_grad=True);x=torch.randn(3,5,dtype=torch.double)
  z=x@w;L=s.assignment_loss(list(z),[0,1,2])+torch.nn.functional.cross_entropy(z,torch.tensor([0,1,2]));g=torch.autograd.grad(L,w)[0]
  leaves=[row.detach().clone().requires_grad_() for row in z];other=s.assignment_loss(leaves,[0,1,2])+torch.nn.functional.cross_entropy(torch.stack(leaves),torch.tensor([0,1,2]));adj=torch.autograd.grad(other,leaves)
  w2=w.detach().clone().requires_grad_()
  for xi,ai in zip(x,adj):torch.sum((xi@w2)*ai).backward()
  self.assertTrue(torch.allclose(g,w2.grad,atol=1e-12,rtol=1e-12))
 def test_high_correct_margin(self):
  z=torch.eye(3)*10
  self.assertLess(float(s.assignment_loss(list(z),[0,1,2])),1e-5)
 def test_duplicate_label_rejected(self):
  with self.assertRaises(ValueError):s.assignment_loss(list(torch.randn(3,4)),[1,1,2])

if __name__=='__main__':unittest.main()
