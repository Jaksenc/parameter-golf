from __future__ import annotations
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import countercase_v8 as c


def fixture():
 return {'id':'fixture','state':'All looms are green. The numbered loom is a loom.',
         'question':{'type':'choice','instructions':'Is it green?','criteria':{'no':'No','yes':'Yes'}},'labels':['no','yes']}

class SchemaTests(unittest.TestCase):
 def test_valid(self):self.assertEqual(c.canonical(fixture()),fixture())
 def test_missing_state(self):
  r=fixture();del r['state']
  with self.assertRaises(ValueError):c.canonical(r)
 def test_reject_answer(self):
  r=fixture();r['expected']='yes'
  with self.assertRaises(ValueError):c.canonical(r)
 def test_reject_oracle(self):
  r=fixture();r['oracle']={'true':1}
  with self.assertRaises(ValueError):c.canonical(r)
 def test_reject_training_metadata(self):
  r=fixture();r['partition']='test'
  with self.assertRaises(ValueError):c.canonical(r)
 def test_reject_duplicate_labels(self):
  r=fixture();r['labels']=['yes','yes']
  with self.assertRaises(ValueError):c.canonical(r)
 def test_reject_newline_label(self):
  r=fixture();r['labels']=['yes\n','no']
  with self.assertRaises(ValueError):c.canonical(r)
 def test_empty_label(self):
  r=fixture();r['labels']=['','no']
  with self.assertRaises(ValueError):c.canonical(r)
 def test_numeric_label(self):
  r=fixture();r['labels']=[0,1]
  with self.assertRaises(ValueError):c.canonical(r)
 def test_invalid_type(self):
  r=fixture();r['question']['type']='free'
  with self.assertRaises(ValueError):c.canonical(r)
 def test_missing_criteria(self):
  r=fixture();del r['question']['criteria']
  with self.assertRaises(ValueError):c.canonical(r)
 def test_no_mutation(self):
  r=fixture();a=copy.deepcopy(r);c.canonical(r);self.assertEqual(r,a)
 def test_canonical_order(self):
  r=fixture();a=dict(reversed(list(r.items())));a['question']=dict(reversed(list(a['question'].items())))
  self.assertEqual(json.dumps(c.canonical(r)),json.dumps(c.canonical(a)))
 def test_nonfinite(self):
  r=fixture();r['state']={'x':float('nan')}
  with self.assertRaises(ValueError):c.canonical(r)
 def test_size_bound(self):
  r=fixture();r['state']='x'*160000
  with self.assertRaises(ValueError):c.canonical(r)

class PromptTests(unittest.TestCase):
 def test_user_evidence_identical(self):
  self.assertEqual(c.messages(fixture(),c.PRIMARY)[1],c.messages(fixture(),'standard480')[1])
 def test_different_system(self):
  self.assertNotEqual(c.messages(fixture(),c.PRIMARY)[0],c.messages(fixture(),'standard480')[0])
 def test_no_prior_answer(self):
  m=c.messages(fixture(),c.PRIMARY);self.assertNotIn('fixture',m[1]['content'])
 def test_unknown_arm(self):
  with self.assertRaises(ValueError):c.messages(fixture(),'oracle')
 def test_schema_does_not_modify_evidence(self):
  self.assertEqual(json.loads(c.messages(fixture(),c.PRIMARY)[1]['content'])['state'],fixture()['state'])
 def test_counterexample_not_proof(self):self.assertIn('Failure to find a counterexample is not a proof',c.SYSTEM)
 def test_no_judges(self):self.assertEqual(c.ARMS,('standard480','countercase480'))

class SolverTests(unittest.TestCase):
 def run_mock(self,text,cut=False,label='no',arm=c.PRIMARY):
  trace={'text':text,'hit_cap':cut,'token_ids':[1,2],'output_tokens':2}
  with patch.object(c.prior,'generate',return_value=trace) as gen,patch.object(c.prior.h5,'readout',return_value={'label':label}) as rd:
   out=c.solve(object(),fixture(),arm);return out,gen.call_args,rd.call_count
 def test_complete_preserved(self):
  r,_,n=self.run_mock('Reason.\nFINAL: yes');self.assertEqual(r['answer'],'yes');self.assertEqual(n,0)
 def test_missing_fallback(self):
  r,_,n=self.run_mock('Reason only.');self.assertEqual(r['answer'],'no');self.assertEqual(n,1)
 def test_partial_final_fallback(self):
  r,_,n=self.run_mock('FINAL: yes',True);self.assertEqual(n,1)
 def test_closed_final_before_cap(self):
  r,_,n=self.run_mock('FINAL: yes\n',True);self.assertEqual(n,0)
 def test_multiple_final_fallback(self):
  r,_,n=self.run_mock('FINAL: yes\nFINAL: no\n');self.assertEqual(n,1)
 def test_unknown_final(self):
  r,_,n=self.run_mock('FINAL: maybe');self.assertEqual(n,1)
 def test_uppercase_not_normalized(self):
  r,_,n=self.run_mock('FINAL: YES');self.assertEqual(n,1)
 def test_same_budget(self):
  _,a,_=self.run_mock('FINAL: yes');_,b,_=self.run_mock('FINAL: yes',arm='standard480');self.assertEqual(a.args[-1],480);self.assertEqual(a.args[-1],b.args[-1])
 def test_illegal_readout_rejected(self):
  with self.assertRaises(ValueError):self.run_mock('reason',label='not-a-label')
 def test_semantic_status_false(self):
  r,_,_=self.run_mock('FINAL: yes');self.assertFalse(r['hypothesis_semantically_verified'])

class SelectionTests(unittest.TestCase):
 def make_source(self,root):
  data={'examples':[{'input':f'Different independent evidence statement for example number {i}: the conditional premise varies case {i}.','target':'valid' if i%2 else 'invalid'} for i in range(30)]}
  raw=json.dumps(data).encode();(root/'formal_fallacies.json').write_bytes(raw)
  return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),data
 def test_deterministic(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);b,_=self.make_source(p)
   with patch.dict(c.prior.SOURCES,{'formal_fallacies':b},clear=True):
    a=c.select_external(p,[]);self.assertEqual(a,c.select_external(p,[]));self.assertEqual(len(a[0]),8)
 def test_excludes_prior_stem(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);b,_=self.make_source(p)
   with patch.dict(c.prior.SOURCES,{'formal_fallacies':b},clear=True):
    initial,_=c.select_external(p,[]);new,_=c.select_external(p,initial)
    self.assertFalse({r['state'] for r in initial}&{r['state'] for r in new})
 def test_reject_changed_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);b,_=self.make_source(p);(p/'formal_fallacies.json').write_text('changed')
   with patch.dict(c.prior.SOURCES,{'formal_fallacies':b},clear=True),self.assertRaises(ValueError):c.select_external(p,[])
 def test_selection_does_not_depend_on_keys(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);b,data=self.make_source(p)
   with patch.dict(c.prior.SOURCES,{'formal_fallacies':b},clear=True):a=c.select_external(p,[])[1][0]['selected_indices']
   for x in data['examples']:x['target']='invalid' if x['target']=='valid' else 'valid'
   raw=json.dumps(data).encode();(p/'formal_fallacies.json').write_bytes(raw)
   b=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
   with patch.dict(c.prior.SOURCES,{'formal_fallacies':b},clear=True):self.assertEqual(a,c.select_external(p,[])[1][0]['selected_indices'])

if __name__=='__main__':unittest.main()
