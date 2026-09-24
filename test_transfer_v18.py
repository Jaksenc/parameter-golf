import copy,json,re,unittest
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import transfer_v18 as t

def independently_parsed_law(text):
 rows=re.findall(r'([a-z]+): sample I has (\d+), sample II has (\d+), archive has (\d+), detection rate is (\d+)/10',text)
 if not rows:raise ValueError('Unsupported source')
 labels=[r[0] for r in rows];a=[int(r[1]) for r in rows];b=[int(r[2]) for r in rows];rates=[int(r[4]) for r in rows]
 counts=[0]*len(rows)
 mix=re.search(r'Select sample I with probability (\d+)/10',text)
 if mix:
  weight=int(mix[1])
  for selector in range(10):
   pool=a if selector<weight else b;scale=sum(b) if selector<weight else sum(a)
   for j,quantity in enumerate(pool):counts[j]+=quantity*scale
 elif 'detector signaled positive' in text:
  for j,quantity in enumerate(a):
   for _ in range(quantity):
    for slot in range(10):counts[j]+=int(slot<rates[j])
 else:counts=a
 return labels,[F(n,sum(counts)) for n in counts]

class ScientificTests(unittest.TestCase):
 def test_generated_size(self):self.assertEqual(len(t.generated()),120)
 def test_worlds(self):self.assertEqual(len(t.make_worlds()),24)
 def test_deterministic(self):self.assertEqual(t.generated(),t.generated())
 def test_label_counts(self):self.assertEqual({len(w['labels']) for w in t.make_worlds()},{2,3,4,5,6,8})
 def test_no_answer_in_input(self):
  for r in t.generated():self.assertEqual(set(r['input']),{'id','state','question','labels'})
 def test_source_targets(self):
  for r in t.generated():
   if r['variant']=='familiar':continue
   labels,q=independently_parsed_law(r['input']['state']);self.assertEqual(labels,r['input']['labels'])
   if r['quantity']=='mode':q=[F(int(i==q.index(max(q)))) for i in range(len(q))]
   self.assertEqual(q,list(map(F,r['target'])))
 def test_oracle_seeds(self):
  for seed in range(5):
   for w in t.make_worlds(seed):self.assertEqual(independently_parsed_law(t.evidence(w))[1],t.old.law(w))
 def test_new_prose_same_law(self):
  for w in t.make_worlds():self.assertEqual(independently_parsed_law(t.evidence(w))[1],t.old.law(w))
 def test_replication(self):
  for w in t.make_worlds():
   z=copy.deepcopy(w);z['a']=[n*5 for n in z['a']];self.assertEqual(independently_parsed_law(t.evidence(z))[1],t.old.law(w))
 def test_relevant_changes(self):
  for w in t.make_worlds():
   z=copy.deepcopy(w);z['a'][0]+=19;self.assertNotEqual(t.old.law(z),t.old.law(w))
 def test_zero_support(self):self.assertTrue(any(0 in t.old.law(w) for w in t.make_worlds()))
 def test_unique_primary_modes(self):
  for w in t.make_worlds():self.assertEqual(t.old.law(w).count(max(t.old.law(w))),1)
 def test_no_duplicate_inputs(self):self.assertEqual(len({t.digest(r['input']) for r in t.generated()}),120)
 def test_temperatures_domain(self):self.assertEqual(set(t.TEMPERATURES),set(t.CHECKPOINTS)|{'unchanged/0'})
 def test_temperatures_positive(self):self.assertTrue(all(x>0 for p in t.TEMPERATURES.values() for x in p))
 def test_no_formal_retention(self):self.assertNotIn('formal_fallacies',t.EXCLUDED)
 def test_external_targets_and_exclusion(self):
  path=Path('transfer-sources')
  if not path.exists():self.skipTest('Source cache not present')
  rows,receipts=t.retention(path);self.assertEqual(len(rows),56)
  for r in rows:self.assertEqual(sum(map(int,r['target'])),1)
  for rec in receipts:self.assertFalse(set(rec['selected_indices'])&set(rec['excluded_indices']))
 def test_retention_reproducible(self):
  path=Path('transfer-sources')
  if not path.exists():self.skipTest('Source cache not present')
  self.assertEqual(t.retention(path),t.retention(path))

if __name__=='__main__':unittest.main()
