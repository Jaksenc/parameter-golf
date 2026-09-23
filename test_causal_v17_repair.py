"""Regression tests for non-mutating counterfactual optimizer observations."""
import copy, random, tempfile, unittest
import numpy as np
import torch
import causal_v17_train as train
import causal_v17_state as ck

def build():
    torch.manual_seed(71); np.random.seed(71); random.seed(71)
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    return model,opt

def gradient(model,t):
    x=torch.tensor([[.4,.2,.9]]); y=torch.tensor([[.1,.8]])
    loss=(model(x)-y).square().sum()+ t*.0001*sum(p.square().sum() for p in model.parameters())
    return [g.detach().clone() for g in torch.autograd.grad(loss,list(model.parameters()))]

def advance(model,opt,t):train.update(torch,model,opt,gradient(model,t))

def unsafe_restore(model,opt,s):
    model.load_state_dict(s[0]);opt.load_state_dict(s[1]);ck.restore_random(s[2])

class RepairTests(unittest.TestCase):
    def test_original_restoration_mutates_snapshot(self):
        model,opt=build();advance(model,opt,1)
        s=train.snapshot(torch,model,opt);expected=train.optimizer_fingerprint(torch,opt)
        unsafe_restore(model,opt,s);advance(model,opt,2);unsafe_restore(model,opt,s)
        self.assertNotEqual(train.optimizer_fingerprint(torch,opt),expected)
        self.assertEqual({int(v['step']) for v in opt.state.values()},{2})
    def test_fixed_snapshot_survives_three_branches(self):
        model,opt=build();advance(model,opt,1)
        s=train.snapshot(torch,model,opt);before=train.optimizer_fingerprint(torch,opt)
        for branch in range(3):
            train.restore(model,opt,s);advance(model,opt,branch+2)
        train.restore(model,opt,s)
        self.assertEqual(train.optimizer_fingerprint(torch,opt),before)
        self.assertEqual({int(v['step']) for v in opt.state.values()},{1})
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,s[0][k]))
    def test_live_state_does_not_alias_snapshot(self):
        model,opt=build();advance(model,opt,1);s=train.snapshot(torch,model,opt);train.restore(model,opt,s)
        live=opt.state_dict()
        for key,vals in s[1]['state'].items():
            for name,v in vals.items():
                if isinstance(v,torch.Tensor):self.assertNotEqual(v.data_ptr(),live['state'][key][name].data_ptr())
    def test_full_32step_trajectory_matches_no_diagnostics(self):
        control,oc=build();test,ot=build()
        for step in range(1,33):
            advance(control,oc,step)
            g=gradient(test,step)
            if step in (1,16,32):
                s=train.snapshot(torch,test,ot)
                for direction in (1.,-1.):
                    train.restore(test,ot,s)
                    train.update(torch,test,ot,[direction*x for x in g])
                train.restore(test,ot,s)
            train.update(torch,test,ot,g)
        for a,b in zip(control.parameters(),test.parameters()):self.assertTrue(torch.equal(a,b))
        self.assertEqual(train.optimizer_fingerprint(torch,oc),train.optimizer_fingerprint(torch,ot))
    def test_rollback_restores_three_random_streams(self):
        model,opt=build();s=train.snapshot(torch,model,opt)
        expected=(random.random(),float(np.random.random()),torch.rand(8))
        train.restore(model,opt,s)
        actual=(random.random(),float(np.random.random()),torch.rand(8))
        self.assertEqual(actual[:2],expected[:2]);self.assertTrue(torch.equal(actual[2],expected[2]))
    def test_real_update_counter_excludes_discarded_updates(self):
        model,opt=build()
        for n in range(1,5):
            s=train.snapshot(torch,model,opt)
            for _ in range(3):train.restore(model,opt,s);advance(model,opt,9)
            train.restore(model,opt,s);advance(model,opt,n)
            self.assertEqual({int(v['step']) for v in opt.state.values()},{n})

if __name__=='__main__':unittest.main()
