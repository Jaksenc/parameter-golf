import copy, json, os, random, sys, unittest
from fractions import Fraction as F
from pathlib import Path
import torch
import learn_data as d
import learn_v16 as train
from relational_objective import Relation,compute_loss

PARENT=Path(os.environ.get('BRIDGE_PARENT','.'))
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.records,cls.edges,cls.manifest=d.make_data(PARENT)
    def test_population(self):self.assertEqual((len(self.records),len(self.edges)),(152,100))
    def test_split_counts(self):self.assertEqual(sum(r['split']=='fit' for r in self.records),96)
    def test_world_separation(self):
        a={r['group'] for r in self.records if r['split']=='fit'};b={r['group'] for r in self.records if r['split']=='check'}
        self.assertFalse(a&b);self.assertEqual((len(a),len(b)),(16,10))
    def test_each_target_from_source(self):
        for r in self.records:
            row=r['input'];q=d.law(row['state'],row['labels'])
            if r['mode']=='mode':q=[F(int(v==max(q))) for v in q]
            self.assertEqual(q,list(map(F,r['target'])))
    def test_targets_not_in_request(self):
        for r in self.records:self.assertEqual(set(r['input']),{'id','state','question','labels'})
    def test_canonical_unique(self):
        keys=[d.digest({k:r['input'][k] for k in ('state','question','labels')}) for r in self.records];self.assertEqual(len(keys),len(set(keys)))
    def test_relationship_truth(self):
        for e in self.edges:
            a=list(map(F,self.records[e['left']]['target']));b=list(map(F,self.records[e['right']]['target']))
            self.assertEqual(b,[sum(F(e['mapping'][i][j])*a[j] for j in range(len(a)))+F(e['delta'][i]) for i in range(len(b))])
    def test_replication_present(self):self.assertEqual(sum(e['operation']=='left_urn_replicated' and e['split']=='fit' for e in self.edges),16)
    def test_distinct_paired_seeds(self):
        orders=[]
        for seed in train.SEEDS:
            order=list(self.manifest['train_edges']);random.Random(seed).shuffle(order);orders.append(order)
        self.assertEqual(len({d.digest(v) for v in orders}),3)
        self.assertTrue(all(set(v)==set(self.manifest['train_edges']) for v in orders))
    def test_no_test_edges(self):
        for i in self.manifest['train_edges']:self.assertEqual(self.edges[i]['split'],'fit')
    def test_probe_not_heldout(self):
        for i in self.manifest['probe_indices']:self.assertEqual(self.records[i]['split'],'fit')
    def test_order_stable(self):self.assertEqual(d.make_data(PARENT)[2],self.manifest)

class GradientTests(unittest.TestCase):
    def setup(self):
        z=[torch.tensor([.2,-.4,.1],requires_grad=True),torch.tensor([.9,-.1,.2],requires_grad=True)]
        q=[torch.tensor([.2,.3,.5]),torch.tensor([.3,.2,.5])]
        edge={'mapping':[[0,1,0],[1,0,0],[0,0,1]],'delta':['0','0','0'],'kind':'permutation'}
        return z,q,edge
    def test_replayed_output_derivative(self):
        z,q,e=self.setup();grads,_=train.loss_grads(torch,z,q,e,.25)
        rel=Relation(0,1,torch.tensor(e['mapping'],dtype=torch.float32),torch.zeros(3),'permutation')
        direct=torch.autograd.grad(compute_loss(z,q,[rel],.25).total,z)
        for a,b in zip(grads,direct):self.assertTrue(torch.equal(a,b))
    def test_zero_weight_equals_ce(self):
        z,q,e=self.setup();g,_=train.loss_grads(torch,z,q,e,0.)
        direct=torch.autograd.grad(sum(-(x*torch.log_softmax(y,-1)).sum() for x,y in zip(q,z))/2,z)
        for a,b in zip(g,direct):self.assertTrue(torch.equal(a,b))
    def test_parameter_gradient_replay(self):
        torch.manual_seed(47);model=torch.nn.Linear(4,3);xs=[torch.randn(4),torch.randn(4)];_,q,e=self.setup()
        z=[model(x) for x in xs];rel=Relation(0,1,torch.tensor(e['mapping'],dtype=torch.float32),torch.zeros(3),'permutation')
        compute_loss(z,q,[rel],.25).total.backward();direct=[p.grad.clone() for p in model.parameters()];model.zero_grad()
        with torch.no_grad():zs=[model(x) for x in xs]
        gs,_=train.loss_grads(torch,zs,q,e,.25)
        for x,g in zip(xs,gs):torch.autograd.backward(model(x),g)
        for a,p in zip(direct,model.parameters()):self.assertTrue(torch.allclose(a,p.grad,atol=1e-7,rtol=1e-6))
    def test_false_relation_rejected(self):
        z,q,e=self.setup();e['mapping']=[[1,0,0],[0,1,0],[0,0,1]]
        with self.assertRaises(ValueError):train.loss_grads(torch,z,q,e,.25)
    def test_nonfinite_rejected(self):
        z,q,e=self.setup();z[0]=torch.tensor([float('nan'),0.,1.])
        with self.assertRaises(ValueError):train.loss_grads(torch,z,q,e,.25)
    def test_wrong_consistent_prediction_not_truth(self):
        q=[torch.tensor([.8,.2]),torch.tensor([.8,.2])];z=[torch.tensor([0.,1.]),torch.tensor([0.,1.])]
        rel=Relation(0,1,torch.eye(2),torch.zeros(2),'irrelevant');out=compute_loss(z,q,[rel],.25)
        self.assertEqual(float(out.relational),0.);self.assertGreater(float(out.supervised),1.)
    def test_relation_does_change_gradient(self):
        z,q,e=self.setup();a,_=train.loss_grads(torch,z,q,e,0.);b,_=train.loss_grads(torch,z,q,e,.25)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a,b)))
    def test_relation_zero_at_verified_targets(self):
        _,q,e=self.setup();z=[v.log() for v in q];_,info=train.loss_grads(torch,z,q,e,.25)
        self.assertLess(info['relation_loss'],1e-12)

class AdapterTests(unittest.TestCase):
    def toy(self):
        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Module();self.model.language_model=torch.nn.Module();self.model.language_model.layers=torch.nn.ModuleList()
                for _ in range(32):
                    l=torch.nn.Module();l.mlp=torch.nn.Module();l.mlp.down_proj=torch.nn.Linear(8,8);self.model.language_model.layers.append(l)
        return Root()
    def test_zero_adapter_equivalent(self):
        m=self.toy();m.requires_grad_(False);x=torch.randn(2,8);module=m.model.language_model.layers[0].mlp.down_proj
        y=module(x);_,f,h=train.make_factors(torch,m,train.SEEDS[0]);self.assertTrue(torch.equal(y,module(x)))
        for hook in h:hook.remove()
    def test_seed_identity_across_arms(self):
        a=self.toy();b=self.toy();_,f,_=train.make_factors(torch,a,16101);_,g,_=train.make_factors(torch,b,16101)
        self.assertEqual(train.tensors_digest(f.state_dict()),train.tensors_digest(g.state_dict()))
    def test_trainable_not_base(self):
        m=self.toy();m.requires_grad_(False);before=list(m.parameters());_,f,_=train.make_factors(torch,m,16101)
        self.assertFalse({id(x) for x in before}&{id(x) for x in f.parameters()});self.assertTrue(all(x.requires_grad for x in f.parameters()))
    def test_first_step_updates_B_not_A(self):
        m=self.toy();m.requires_grad_(False);_,f,_=train.make_factors(torch,m,16101)
        out=m.model.language_model.layers[0].mlp.down_proj(torch.randn(3,8));out.square().sum().backward()
        self.assertGreater(float(f[0].b.grad.norm()),0.);self.assertEqual(float(f[0].a.grad.norm()),0.)
    def test_disabled_restores_base(self):
        m=self.toy();m.requires_grad_(False);x=torch.randn(2,8);module=m.model.language_model.layers[0].mlp.down_proj;y=module(x)
        _,f,_=train.make_factors(torch,m,16101)
        with torch.no_grad():f[0].b.fill_(.25)
        self.assertFalse(torch.equal(y,module(x)));f[0].enabled=False;self.assertTrue(torch.equal(y,module(x)))

if __name__=='__main__':unittest.main()
