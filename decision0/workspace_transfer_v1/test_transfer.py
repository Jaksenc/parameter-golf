import copy, json, re, unittest
from datetime import datetime
from fractions import Fraction
import transfer_data as d
import transfer_study as s

def reference(r):
    text=r['state'];fam=r['family']
    if fam=='amount':
        name=re.search(r'For purchase ([A-Z]+)',r['question']).group(1)
        items=json.loads(re.search(rf'Purchase {name}: line items are (\[.*?\])\.',text).group(1))
        shipping,credit=map(int,re.search(rf'Purchase {name}: shipping (-?\d+) cents; credit (-?\d+) cents',text).groups())
        cap=int(re.search(rf'Purchase {name}: comparison budget (-?\d+) cents',text).group(1))
        amounts=[i['quantity']*i['unit_cents'] for i in items]
        instruction=re.search(rf'Calculation instruction applicable to {name}: ([^\n]+)',text).group(1)
        discount=re.search(r'by (\d+) percent',instruction)
        if discount:
            rate=Fraction(100-int(discount.group(1)),100)
            def round_half(x):return (2*x.numerator+x.denominator)//(2*x.denominator)
            value=(sum(round_half(a*rate) for a in amounts) if 'each individual line' in instruction else round_half(sum(amounts)*rate))
        else:value=sum(amounts)
        value+=shipping-credit
        return {'below':float(value<cap),'equal':float(value==cap),'above':float(value>cap)}
    if fam=='temporal':
        name=re.search(r'case ([A-Z]+)',r['question']).group(1)
        event=datetime.fromisoformat(re.search(rf'(?:Service case|Case) {name}: received timestamp ([^\s]+)\.',text).group(1))
        official=re.search(rf'Case {name}: official deadline is ([^\s]+)\.',text)
        if official:seconds=(event-datetime.fromisoformat(official.group(1))).total_seconds()
        else:
            start=datetime.fromisoformat(re.search(rf'Service case {name}: start ([^\s]+)\.',text).group(1))
            allowed=int(re.search(rf'Service case {name}: start [^\s]+\. Allowed counted time is (\d+) minutes',text).group(1))
            pause=int(re.search(rf'Case {name}: subtract (\d+) minutes',text).group(1))
            seconds=(event-start).total_seconds()-60*(allowed+pause)
        return {'before':float(seconds<0),'at':float(seconds==0),'after':float(seconds>0)}
    if fam=='linkage':
        name=re.search(r'customer ([A-Z]+)',r['question']).group(1)
        contract=re.search(rf'Customer {name}: current contract ([A-Z]+)\.',text).group(1)
        node=re.search(rf'Contract {contract}: current service node ([A-Z]+)\.',text).group(1)
        depot=re.search(rf'Node {node}: currently handled by depot ([A-Z]+)\.',text).group(1)
        destination=re.search(rf'Depot {depot}: its destination is ([a-z]+)\.',text).group(1)
        return {o['id']:float(o['id']==destination) for o in r['options']}
    if fam=='policy':
        name=re.search(r'subject ([A-Z]+)',r['question']).group(1)
        vals=dict(re.findall(rf'Current subject {name}: ([a-z_]+)=([a-z]+)\.',text))
        if 'active' in vals:
            possible=[]
            for a in ([True,False] if vals['active']=='unrecorded' else [vals['active']=='yes']):
                for h in ([True,False] if vals['hold']=='unrecorded' else [vals['hold']=='yes']):
                    for c in ([True,False] if vals['clearance']=='unrecorded' else [vals['clearance']=='yes']):possible.append(a and (not h or c))
            ans='allow' if all(possible) else 'deny' if not any(possible) else 'unknown'
        elif 'withdrawn' in vals:
            if vals['withdrawn']=='yes':ans='deny'
            elif vals['withdrawn']=='unrecorded':ans='unknown'
            elif vals['exception']=='yes':ans='allow'
            elif vals['exception']=='unrecorded':ans='unknown'
            elif vals['region'] not in ('east','west'):ans='review'
            else:ans={'yes':'allow','no':'deny','unrecorded':'unknown'}[vals['certified']]
        else:
            c=vals['class'];ans={'restricted':'deny','public':'allow','unrecorded':'unknown'}.get(c)
            if ans is None:ans={'yes':'allow','no':'deny','unrecorded':'unknown'}[vals['consent']]
        return {o['id']:float(o['id']==ans) for o in r['options']}
    if fam=='judging':
        cand=re.search(r'Candidate response: ([^\n]+)',text).group(1)
        if 'Reference invoice' in text:
            q,p,fee,credit=map(int,re.search(r'Reference invoice [A-Z]+: (\d+) units priced at (\d+) cents each, a handling fee of (\d+) cents, and a credit of (\d+) cents',text).groups())
            reported=int(re.search(r'final amount is (\d+) cents',cand).group(1));ans='correct' if reported==q*p+fee-credit else 'numeric_error'
        elif 'countersignature' in text:ans='correct' if cand.startswith('Not yet.') else 'rule_error'
        else:
            name=re.search(r'Question posed to the candidate: what is the current status of ([A-Z]+)',text).group(1)
            status=re.search(rf'Current record: request {name} is ([a-z]+)',text).group(1)
            reported=re.search(rf'Request {name} is ([a-z]+)',cand).group(1);ans='correct' if reported==status else 'wrong_evidence'
        return {o['id']:float(o['id']==ans) for o in r['options']}
    if fam=='probability':
        if 'screened members:' in text:
            a,b,c=map(int,re.search(r'Population [A-Z]+, screened members: alpha=(\d+), beta=(\d+), gamma=(\d+)',text).groups())
            return dict(zip(('alpha','beta','gamma'),(a/(a+b+c),b/(a+b+c),c/(a+b+c))))
        if 'Sampling protocol' in text:
            a=list(map(int,re.search(r'Station [A-Z]+-A has alpha=(\d+), beta=(\d+), gamma=(\d+)',text).groups()))
            b=list(map(int,re.search(r'Station [A-Z]+-B has alpha=(\d+), beta=(\d+), gamma=(\d+)',text).groups()))
            w=int(re.search(r'choose station A with probability (\d+) out of 10',text).group(1))
            return {k:float(Fraction(w*a[i],10*sum(a))+Fraction((10-w)*b[i],10*sum(b))) for i,k in enumerate(('alpha','beta','gamma'))}
        a,b,c=map(int,re.search(r'Container [A-Z]+ contains alpha=(\d+), beta=(\d+), gamma=(\d+)',text).groups())
        seq=['a']*a+['b']*b+['c']*c
        same=sum(x==y for i,x in enumerate(seq) for j,y in enumerate(seq) if i!=j)
        denom=len(seq)*(len(seq)-1)
        return {'same':same/denom,'different':1-same/denom}
    raise ValueError(fam)

class DataTests(unittest.TestCase):
    def test_frozen_data(self):self.assertEqual(s.check_data()['held']['rows'],72)
    def test_independent_reference(self):
        for split in ('development','held'):
            for r in d.corpus(split):
                actual=reference(r)
                for k,q in r['target_probs'].items():self.assertAlmostEqual(actual.get(k,0),q,places=12,msg=r['id'])
    def test_reference_fields_excluded(self):
        for r in d.corpus('development'):
            p=copy.deepcopy(r);p.update(expected='SECRET_GOLD_7182',family='SECRET_FAMILY_7182',source='SECRET_SOURCE_7182',oracle='SECRET_ORACLE_7182')
            for f in [s.work_messages,lambda x:s.final_messages(x,{'support':'draft','work':'draft','event_probs':None})]:self.assertNotIn('SECRET_',json.dumps(f(p)))
    def test_reference_poison_same_prompt(self):
        for r in d.corpus('held'):
            p=dict(r,expected='incorrect',oracle={'total':-999},gold_probs={'wrong':1},family='other');self.assertEqual(s.work_messages(r),s.work_messages(p))
    def test_all_shards_exact(self):
        rows=d.corpus('development')+d.corpus('held')
        for n in (1,8,12,16):self.assertEqual(sorted(sum(s.assignment(rows,n),[])),list(range(len(rows))))
    def test_id_invisible(self):
        r=d.corpus('development')[0];self.assertEqual(s.work_messages(r),s.work_messages(dict(r,id='not-the-same')))
    def test_options_have_all_reference_labels(self):
        for split in ('development','held'):
            for r in d.corpus(split):self.assertEqual({o['id'] for o in r['options']},set(r['target_probs']))

class ParserTests(unittest.TestCase):
    def test_null(self):self.assertIsNone(s.parse_workspace('{"support":"x","work":"y","event_probs":null}',['a','b'])[1])
    def test_distribution_alignment(self):
        _,q=s.parse_workspace('{"support":"x","work":"y","event_probs":{"B":0.7,"A":0.3}}',['z','a']);self.assertEqual(q,[.3,.7])
    def test_false_not_number(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":true,"B":0}}',['x','y'])
    def test_no_percentages(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":30,"B":70}}',['x','y'])
    def test_duplicate(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":0.3,"A":0.3,"B":0.7}}',['x','y'])
    def test_missing_outcome(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":1}}',['x','y'])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":NaN,"B":0}}',['x','y'])
    def test_schema(self):
        for raw in ('null','[]','{"answer":"A"}','```json\n{}\n```'):
            with self.assertRaises(ValueError):s.parse_workspace(raw,['x','y'])
    def test_negative(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":-0.1,"B":1.1}}',['x','y'])
    def test_not_normalized(self):
        with self.assertRaises(ValueError):s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":0.3,"B":0.3}}',['x','y'])
    def test_small_roundoff(self):
        _,q=s.parse_workspace('{"support":"x","work":"y","event_probs":{"A":0.3,"B":0.6999999}}',['x','y']);self.assertAlmostEqual(sum(q),1)

class MetricsTests(unittest.TestCase):
    def test_failure_counted(self):
        rows=[dict(id='x',variant='workspace',ok=False,correct=False,labels=['a','b'],probabilities=None,expected='a',gold_probs={'a':.7,'b':.3},request_seconds=1,total_input_tokens=10)]
        m=s.measure(rows);self.assertEqual(m['failures'],1);self.assertEqual(m['accuracy'],0);self.assertEqual(m['event_tvd'],1)
    def test_tvd(self):
        r={'ok':True,'labels':['b','a'],'probabilities':[.2,.8],'gold_probs':{'a':.7,'b':.3}};self.assertAlmostEqual(s.tvd(r),.1)
    def test_tie_lexicographic(self):self.assertEqual(s.prediction(['z','a'],[.5,.5]),'a')
    def test_percentile(self):self.assertEqual(s.percentile([1,2,3],.5),2)

if __name__=='__main__':unittest.main()
