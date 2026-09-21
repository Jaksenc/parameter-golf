import copy,json,math,re,unittest
from datetime import datetime
from decimal import Decimal
from flow_data import development,visible,FAMILIES,digest
from flow_study import PROTOCOL,assigned_indices,softmax,prediction,summary,tvd
from prompt_variants import messages,VARIANTS

class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=development()
    def test_deterministic_hash(self):self.assertEqual(digest(self.rows),PROTOCOL['development_hash'])
    def test_counts_and_semantic_uniqueness(self):
        self.assertEqual(len(self.rows),72);self.assertEqual(len({x['source'] for x in self.rows}),36)
        self.assertEqual(len({digest({'state':x['state'],'question':x['question'],'options':x['options']}) for x in self.rows}),72)
    def test_references_excluded(self):
        a=self.rows[0];b=copy.deepcopy(a);b['expected']='SECRET_TARGET';b['reference']={'answer':'SECRET_TARGET'};b['gold_probs']={'SECRET_TARGET':1}
        for v in VARIANTS:self.assertEqual(messages(visible(a),v),messages(visible(b),v))
    def test_pair_nuisance_and_changes(self):
        for a,b in zip(self.rows[::2],self.rows[1::2]):
            self.assertEqual(a['source'],b['source']);self.assertEqual(a['options'],b['options'])
            self.assertNotEqual(a['state']+a['question'],b['state']+b['question'])
            if a['family']!='probability':self.assertNotEqual(a['expected'],b['expected'])
    def test_amount_text_oracle(self):
        for r in self.rows:
            if r['family']!='amount':continue
            n=int(re.search(r'Units supplied: (\d+)',r['state'])[1]);d=[Decimal(x) for x in re.findall(r'\$(\d+\.\d{2})',r['state'])]
            total=n*d[0]+d[1]-d[2];yes=total<=d[3] if 'at or below' in r['question'] else total<d[3]
            self.assertEqual(r['expected'],'satisfies' if yes else 'exceeds')
    def test_time_text_oracle(self):
        for r in self.rows:
            if r['family']!='elapsed':continue
            ts=re.findall(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d',r['state']);mins=(datetime.fromisoformat(ts[1])-datetime.fromisoformat(ts[0])).total_seconds()/60
            limit=int(re.search(r'(\d+) minutes',r['question'])[1]);yes=mins<=limit if 'including exactly at' in r['question'] else mins<limit
            self.assertEqual(r['expected'],'inside' if yes else 'outside')
    def test_lookup_text_oracle(self):
        for r in self.rows:
            if r['family']!='lookup':continue
            entity=re.search(r'pool for ([A-Z-]+)',r['question'])[1];team=re.search(r'Active membership: '+entity+r' -> ([A-Z-]+)',r['state'])[1];pool=re.search(r'Current route: '+team+r' -> ([a-z]+)',r['state'])[1]
            self.assertEqual(pool,r['expected'])
    def test_policy_text_oracle(self):
        for r in self.rows:
            if r['family']!='policy':continue
            blocked='safety block=true' in r['state'];permit='required permit present=true' in r['state'];self.assertEqual(r['expected'],'hold' if blocked else 'release' if permit else 'permit-review')
    def test_probability_text_oracle(self):
        for r in self.rows:
            if r['family']!='probability':continue
            line=next(l for l in r['state'].splitlines() if l.startswith('Complete'));counts={k:int(v) for k,v in re.findall(r'(ash|birch|cedar|elm)=(\d+)',line)}
            for k,v in counts.items():self.assertAlmostEqual(r['target_probs'][k],v/sum(counts.values()))
    def test_judge_text_oracle(self):
        for r in self.rows:
            if r['family']!='judge':continue
            n,rate,fee=map(int,re.search(r'(\d+) units at (\d+) credits each, plus one fee of (\d+)',r['state']).groups());text=r['state'].split('Candidate answer: ')[1];num=int(re.search(r'result is (\d+)',text)[1]);answer='arithmetic' if num!=n*rate+fee else 'unsupported' if 'confirmed delivery' in text else 'missing' if 'one-time fee' not in text else 'valid';self.assertEqual(r['expected'],answer)
    def test_partitions_complete(self):
        for n in [1,8,16]:
            ids=sum(assigned_indices(self.rows,n),[]);self.assertEqual(sorted(ids),list(range(72)))
    def test_layout_semantics(self):
        r=visible(self.rows[0]);base=json.loads(messages(r,'baseline')[1]['content']);first=json.loads(messages(r,'schema_first')[1]['content']);bracket=json.loads(messages(r,'schema_bracket')[1]['content'])
        self.assertEqual(dict(base),dict(first));self.assertEqual(list(base),['evidence','criterion','options']);self.assertEqual(list(first),['criterion','options','evidence'])
        self.assertEqual(bracket['decision_reminder'],{'criterion':base['criterion'],'options':base['options']})
        self.assertEqual(messages(r,'repeat_full')[1]['content'],messages(r,'baseline')[1]['content']+'\n\n'+messages(r,'baseline')[1]['content'])
    def test_numeric_scoring_and_ties(self):
        self.assertEqual(prediction(['z','a'],[.5,.5]),'a');self.assertAlmostEqual(sum(softmax([1000,999])),1)
        with self.assertRaises(ValueError):softmax([math.nan,1])
    def test_gold_probability_metric(self):
        self.assertAlmostEqual(tvd({'ok':True,'labels':['x','y'],'probabilities':[.7,.3],'gold_probs':{'x':.7,'y':.3}}),0)
        self.assertEqual(tvd({'ok':False,'target_probs':{'x':1}}),1)
    def test_independent_id_twins(self):
        for a,b in zip(self.rows[::2],self.rows[1::2]):
            if a['family']=='amount':
                ida=re.search(r'Invoice (R-[A-Z]+)',a['state'])[1];idb=re.search(r'Invoice (R-[A-Z]+)',b['state'])[1];self.assertEqual(ida,idb);self.assertNotEqual(a['expected'],b['expected'])
    def test_variant_rejects_reference_via_transport(self):
        for r in self.rows:
            for v in VARIANTS:
                text=messages(visible(r),v)[1]['content'];self.assertNotIn(r['id'],text);self.assertNotIn('target_probs',text);self.assertNotIn('gold_probs',text)
if __name__=='__main__':unittest.main()
