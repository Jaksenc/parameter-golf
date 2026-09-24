"""Synthetic fixtures test the reducer only; no fixture is a neural observation."""
import contextlib,copy,io,json,math,tempfile,unittest
from pathlib import Path
from fractions import Fraction
import numpy as np
import format_v19_data as d
import analyze_format_v19 as a

def fixture(root):
 w,r,m=d.build('.');root=Path(root)
 for name,obj in [('worlds',w),('records',r),('manifest',m)]:a.save(root/'format-prepared'/(name+'.json'),obj)
 for seed in d.SEEDS:
  for arm in d.ARMS:
   folder=root/'models'/f'format-v19-model-{seed}-{arm}';folder.mkdir(parents=True)
   (folder/'adapter-32.safetensors').write_bytes(b'SYNTHETIC_FIXTURE_NOT_WEIGHTS')
   h=[]
   for step,i in enumerate(m['schedules'][str(seed)][arm],1):
    q=[float(Fraction(x)) for x in r[i]['target']];z=np.linspace(-.1,.1,len(q)).tolist();ce=a.loss(z,q)['ce']
    h.append({'step':step,'index':i,'optimizer_step':step,'group':r[i]['group'],'format':r[i]['format'],'target':r[i]['target'],'logits':z,'ce':ce,'weighted_loss':.5*ce,'tokens':17,'seconds':1.})
   (folder/'training.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in h))
   a.save(folder/'complete-32.json',{'from_step':16,'through_step':32,'initial_adapter_sha256':str(seed),'adapter_sha256':a.sha(folder/'adapter-32.safetensors'),'history_hash':a.digest(h),'base_restoration_error':0.})
   a.save(folder/'preflight.json',{'synthetic_fixture':True})
 rows=[]
 for i in m['evaluation_indices']:
  k=len(r[i]['target']);z=np.linspace(-.3,.3,k).tolist()
  outputs={name:{'logits':z,'probabilities':a.probs(z).astype(np.float32).astype(float).tolist(),'seconds':1.} for name in a.KEYS}
  rows.append({'id':r[i]['input']['id'],'index':i,'input_sha256':a.digest(r[i]['input']),'prompt_sha256':'fixture','tokens':17,'code_ids':list(range(k)),'outputs':outputs,'restoration_error':0.})
 a.save(root/'all_records.json',rows)
 for shard in range(16):
  folder=root/'evaluated'/f'format-v19-evaluation-{shard}';folder.mkdir(parents=True)
  sub=rows[shard::16];path=folder/'records.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in sub))
  a.save(folder/'complete.json',{'count':len(sub),'indices':m['evaluation_indices'][shard::16],'records_sha256':a.sha(path)})
 return rows

class Analysis(unittest.TestCase):
 def test_exact_loss_uniform(self):
  m=a.loss([0,0],[.5,.5]);self.assertAlmostEqual(m['ce'],math.log(2));self.assertEqual(m['squared'],0)
 def test_exact_loss_target(self):
  q=[.2,.3,.5];m=a.loss(np.log(q),q);self.assertLess(m['squared'],1e-28)
 def test_logit_shift(self):self.assertTrue(np.allclose(a.probs([2,3,4]),a.probs([102,103,104])))
 def test_temperature_preserves_ranking(self):self.assertEqual(a.probs([1,4,2],4).argmax(),1)
 def test_independent_decimal(self):self.assertTrue(np.allclose(a.probs([-20,0,9]),a.independent_probs([-20,0,9]),rtol=1e-14))
 def test_bootstrap_zero(self):
  c=a.interval(np.zeros((3,8)),['a']*4+['b']*4,100);self.assertEqual(c['interval95'],[0.,0.])
 def test_bootstrap_constant(self):
  c=a.interval(np.ones((3,8))*.25,['a']*4+['b']*4,100);self.assertEqual(c['interval95'],[.25,.25])
 def test_full_synthetic_replay(self):
  with tempfile.TemporaryDirectory() as td:
   fixture(td)
   with contextlib.redirect_stdout(io.StringIO()):a.analyze(td)
   audit=a.read(Path(td)/'results/audit.json');self.assertEqual(audit['probability_vectors'],1764);self.assertEqual(audit['training_updates'],192)
 def test_reject_missing_output(self):
  with tempfile.TemporaryDirectory() as td:
   rows=fixture(td);rows.pop();a.save(Path(td)/'all_records.json',rows)
   with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):a.analyze(td)
 def test_reject_wrong_target(self):
  with tempfile.TemporaryDirectory() as td:
   fixture(td);p=Path(td)/'format-prepared/records.json';r=a.read(p);r[0]['target']=['1','0'];a.save(p,r)
   with self.assertRaises(ValueError):a.analyze(td)
 def test_reject_modified_output(self):
  with tempfile.TemporaryDirectory() as td:
   rows=fixture(td);rows[0]['outputs']['unchanged']['probabilities'][0]=.99;a.save(Path(td)/'all_records.json',rows)
   with self.assertRaises(ValueError):a.analyze(td)

if __name__=='__main__':unittest.main()
