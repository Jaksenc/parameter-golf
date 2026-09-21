import unittest,copy,random,math
from unittest.mock import patch
import analyze_v8 as a
import countercase_v8 as c
import countercase_v8_decide as decide
from test_countercase import fixture

class AnalysisTests(unittest.TestCase):
 def test_parser_independent_fuzz(self):
  rng=random.Random(9183);labels=['no','yes','0','1','01','with space','(A)']
  for _ in range(10000):
   text='\n'.join(rng.choice(['FINAL: yes','FINAL: no','FINAL: 0','FINAL: 01','FINAL: invalid','FINAL: 1',' final: yes','text','FINAL: with space','FINAL: (A)','FINAL:\tyes','']) for _ in range(rng.randint(0,5)))
   cut=bool(rng.getrandbits(1))
   self.assertEqual(a.final(text,labels,cut),c.prior.h5.parse_final(text,labels,cut))
 def test_known_pairs(self):
  rows=[{'id':str(i),'predictions':{'a':'y' if i<8 else 'n','b':'y' if i<6 else 'n'}} for i in range(10)]
  tasks={r['id']:{'expected':'y','partition':'fresh_bbh','family':'x'} for r in rows}
  p=a.paired(rows,tasks,'a','b',100)
  self.assertEqual((p['repairs'],p['regressions']),(2,0));self.assertAlmostEqual(p['delta_percentage_points'],20)
 def test_no_difference(self):
  rows=[{'id':str(i),'predictions':{'a':'y','b':'y'}} for i in range(4)];tasks={r['id']:{'expected':'y','partition':'jevbench','group':'g'+str(i//2)} for i,r in enumerate(rows)}
  p=a.paired(rows,tasks,'a','b',100);self.assertEqual(p['bootstrap95_percentage_points'],[0.,0.]);self.assertEqual(p['discordant_pair_p'],1)
 def test_integer_gold(self):
  r=[{'id':'a','predictions':{'a':'0','b':'1'}}];t={'a':{'expected':0,'partition':'fresh_bbh','family':'x'}}
  p=a.paired(r,t,'a','b',100);self.assertEqual(p['repairs'],1)
 def test_corrupt_stop_flag(self):
  o={'trace':{'token_ids':[1]*480,'output_tokens':480,'eos_ids':[2],'hit_cap':False,'text':'FINAL: yes'},'readout':None,'answer':'yes'}
  with self.assertRaises(ValueError):a.replay(o,fixture(),True)
 def test_wrong_label(self):
  o={'trace':{'token_ids':[2],'output_tokens':1,'eos_ids':[2],'hit_cap':False,'text':'FINAL: yes','seconds':1.},'readout':None,'answer':'no'}
  with self.assertRaises(ValueError):a.replay(o,fixture(),True)
 def test_no_fabricated_probability(self):
  result={'answer':'yes','trace':{'output_tokens':3,'text':'FINAL: yes'},'readout':None}
  with patch.object(c,'solve',return_value=result):
   o=decide.decide(object(),fixture())
  self.assertIsNone(o['raw_uncalibrated_probabilities']);self.assertFalse(o['semantic_correctness_verified'])
 def test_entry_preserves_fallback(self):
  result={'answer':'no','trace':{'output_tokens':480,'text':'...'},'readout':{'probabilities_uncalibrated':[.7,.3]}}
  with patch.object(c,'solve',return_value=result):o=decide.decide(object(),fixture())
  self.assertEqual(o['answer'],'no');self.assertTrue(o['used_categorical_fallback'])
 def test_entry_rejects_oracle_before_solve(self):
  r=fixture();r['expected']='yes'
  with patch.object(c,'solve') as s,self.assertRaises(ValueError):decide.decide(object(),r)
  s.assert_not_called()

if __name__=='__main__':unittest.main()
