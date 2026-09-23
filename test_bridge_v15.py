import copy,json,unittest
from fractions import Fraction as F
from unittest.mock import patch
import bridge_v15 as b

# Separate finite-ticket reference: all branches operate on integer multiplicities.
def enumerated(w):
    a=w['a'];k=len(a);counts=[0]*k
    if w['family'] in ('counts','conditional'):
        for i,n in enumerate(a):
            for _ in range(n):counts[i]+=1
    elif w['family']=='mixture':
        for selector in range(10):
            nums=a if selector<w['weight'] else w['b']
            copies=sum(w['b']) if selector<w['weight'] else sum(a)
            for i,n in enumerate(nums):
                for _ in range(n*copies):counts[i]+=1
    else:
        for i,n in enumerate(a):
            for item in range(n):
                for assay in range(10):
                    if assay<w['rates'][i]:counts[i]+=1
    return [F(n,sum(counts)) for n in counts]

class Data(unittest.TestCase):
    def test_size(self):self.assertEqual(len(b.worlds()),24)
    def test_seed(self):self.assertEqual(b.worlds(),b.worlds())
    def test_unique(self):self.assertEqual(len({w['state'] for w in b.worlds()}),24)
    def test_cardinality(self):self.assertEqual({len(w['labels']) for w in b.worlds()},{2,3,4,5})
    def test_ticket_oracle(self):
        for w in b.worlds():self.assertEqual(b.distribution(w),enumerated(w))
    def test_unseen_seed_oracles(self):
        for seed in range(11):
            for w in b.worlds(seed):self.assertEqual(b.distribution(w),enumerated(w))
    def test_distractor_invariance(self):
        for w in b.worlds():
            z=copy.deepcopy(w);z['noise']=[n*9+7 for n in z['noise']]
            self.assertNotEqual(b.render(w)[0],b.render(z)[0]);self.assertEqual(enumerated(w),enumerated(z))
    def test_evidence_change(self):
        for w in b.worlds():
            z=copy.deepcopy(w);z['a'][0]+=1
            self.assertNotEqual(enumerated(w),enumerated(z))
    def test_renaming_equivariance(self):
        for w in b.worlds():
            z=copy.deepcopy(w)
            for field in ('labels','a','b','noise','rates'):z[field]=list(reversed(z[field]))
            self.assertEqual(enumerated(z),list(reversed(enumerated(w))))
    def test_event_mode_pair(self):
        for w in b.worlds():
            e,m=b.task(w,'event'),b.task(w,'mode');self.assertEqual(e['input']['state'],m['input']['state']);self.assertEqual(e['input']['labels'],m['input']['labels'])
            self.assertEqual(sum(map(F,e['target'])),1);self.assertEqual(sum(map(F,m['target'])),1);self.assertEqual(m['target'].count('1'),1)
    def test_oracle_only_context(self):
        for w in b.worlds():
            r=b.task(w,'event');self.assertNotIn(r['oracle'],r['input']['state']);self.assertNotIn('target',r['input'])
    def test_zero_support_present(self):self.assertTrue(any(min(enumerated(w))==0 for w in b.worlds()))
    def test_mutation_isolation(self):
        w=b.worlds()[0];original=copy.deepcopy(w);b.task(w,'mode');self.assertEqual(w,original)
    def test_schema_rejects_target(self):
        r=b.task(b.worlds()[0],'event')['input'];r['target']=[.5,.5]
        with self.assertRaises(ValueError):b.parent.request(r)

class Channels(unittest.TestCase):
    def test_same_note_both_paths(self):
        row=b.task(b.worlds()[0],'event')['input'];note='Exact same test draft.'
        numerical={'parse_valid':True,'parsed':{'probabilities':[.2,.8],'label':row['labels'][1]},'trace':{'seconds':2}}
        legacy={'probabilities_uncalibrated':[.7,.3],'label':row['labels'][0],'seconds':1}
        with patch.object(b.parent,'numerical_readout',return_value=numerical) as n,patch.object(b.parent,'code_readout',return_value=legacy) as c:
            out=b.view(None,row,note)
            self.assertEqual(n.call_args.args[2],c.call_args.args[2]);self.assertEqual(n.call_args.args[1],c.call_args.args[1]);self.assertFalse(out['outputs']['numeric_mass']['fallback_used'])
    def test_invalid_numerical_fallback(self):
        row=b.task(b.worlds()[0],'event')['input'];n={'parse_valid':False,'trace':{'seconds':2}};l={'probabilities_uncalibrated':[.3,.7],'label':row['labels'][1],'seconds':1}
        with patch.object(b.parent,'numerical_readout',return_value=n),patch.object(b.parent,'code_readout',return_value=l):o=b.view(None,row,'draft')
        self.assertEqual(o['outputs']['numeric_mass']['probabilities'],[.3,.7]);self.assertEqual(o['outputs']['numeric_mass']['policy_seconds'],3)
    def test_normalization(self):self.assertEqual(b.parent.parse_weights('[2,3,5]',['a','b','c'])['probabilities'],[.2,.3,.5])
    def test_no_false_smoothing(self):self.assertEqual(b.parent.parse_weights('[0,1]',['a','b'])['probabilities'],[0.,1.])
    def test_reject_invalid(self):
        for x in ('[NaN,1]','[true,1]','[0,0]','[-1,2]','[1]','[".5", ".5"]','[1,2] prose'):
            with self.assertRaises(ValueError):b.parent.parse_weights(x,['a','b'])
    def test_source_data_same_prompts(self):
        r=b.task(b.worlds()[0],'event')['input']
        for c in b.CHANNELS:
            s=b.parent.arm_messages(r,'draft',c)[1]['content'];self.assertIn(json.dumps(r['labels']),s)

if __name__=='__main__':unittest.main()
