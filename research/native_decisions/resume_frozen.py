"""Complete missing commits, never retry a saved error or replace an earlier answer.
Original run 35913116027 retained 353 of 522 planned attempts. Prefix identities
below are independently reconstructed from its eight original durable journals.
No score, reference answer, or observed correctness controls pending selection.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import readout
import entry_exact
import experiment as exp
COUNTS=[70,37,35,41,38,64,31,37]
SOURCE={'experiment.py':'a2afc987a87e77b267181f07d5f26ed29d3ac2e96ad047e859b057bd2af2a782','readout.py':'0e12ec6914bb29c14222bb432e7bd9e901f35ee7c62325ca12b938dcfaf5b186','exact_backend.py':'7118b66d16667f68e3f08a2456a38c3e8d94dd7e13413c421c1ed0de0db4a8c2','entry_exact.py':'306bca7b4702c74442403313125c500b091e7a6f409edde08c1a6d31c61bfc8e','raw_logits.cpp':'788b29ec6c868c1b7027edc907d435e3b8108fb3c09a67bf64eb6652ebdda733'}
MISSING='fb9885c9e10bccf032041110669278b3d276e380e3188d4a562a892a7e582251'
PRESENT='288229b0e18870483d2e4cb67dea7eedf429e7fae2d3e4ddbfc3ad8805361973'
SHARDS=24

def pending():
 for name,h in SOURCE.items():
  if readout.digest(Path(__file__).with_name(name).read_bytes())!=h:raise ValueError('Frozen source differs: '+name)
 tasks,probe,manifest=exp.acquire();present=[];absent=[]
 for shard,n in enumerate(COUNTS):
  seq=[]
  for t in [t for i,t in enumerate(tasks) if i%8==shard]:
   arms=['native_input','native_canonical']+(['native_reverse','canonical_reverse','keyed'] if t['id'] in probe else [])
   off=int(readout.digest(t['id'])[:8],16)%len(arms);arms=arms[off:]+arms[:off]
   seq.extend([[t['id'],a] for a in arms])
  present+=seq[:n];absent+=seq[n:]
 present.sort();absent.sort()
 if len(absent)!=169 or readout.digest(readout.canonical(absent))!=MISSING:raise ValueError('Missing-key lock')
 if readout.digest(readout.canonical(present))!=PRESENT:raise ValueError('Present-key lock')
 return tasks,probe,manifest,absent

def worker(inp,out):
 packets=readout.loads(Path(inp).read_text());backend=readout.Backend()
 try:
  with Path(out).open('x') as f:
   for p in packets:
    if set(p)!={'id','arm','task'} or set(p['task'])!={'state','question','labels'}:raise ValueError('Worker schema')
    r=backend.solve(p['task'],p['arm']);r.update(id=p['id'],task_sha256=readout.digest(readout.canonical(p['task'])))
    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print('COMMITTED '+json.dumps({k:r.get(k) for k in ('id','arm','ok','error','seconds')}),flush=True)
 finally:backend.close()

def run(shard):
 if not 0<=shard<SHARDS:raise ValueError('shard')
 started=time.perf_counter();proc=log=None;out=Path(f'resumed-native-journal-{shard:02d}.jsonl')
 result={'status':'failed','run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'source_sha256':readout.digest(Path(__file__).read_bytes()),'shard':shard,'shards':SHARDS,'original_run':'35913116027','original_counts':COUNTS,'missing_keys_sha256':MISSING,'present_keys_sha256':PRESENT,'frozen_source':SOURCE,'new_generations_for_uncommitted_keys':True,'saved_failures_retried':False,'unknown_prior_uncommitted_compute':True}
 try:
  tasks,probe,manifest,absent=pending();lookup={t['id']:t for t in tasks}
  selected=[key for i,key in enumerate(absent) if i%SHARDS==shard]
  result.update(selected_keys=selected,manifest=manifest,probe_ids=sorted(probe),cases=[lookup[i] for i in sorted({k[0] for k in selected})])
  result['protocol']={'scope':'missing original commits only','model_pin':exp.PIN,'original_protocol_unchanged':True,'native_generated_tokens':0,'weights_trained':False,'memory':False,'calibration_fitted':False,'cost_usd':None,'paid_inference':False,'energy_measured':False,'worker_timeout_seconds':1440,'per_call_limits_unchanged':True,'official_score':False}
  print('LOCK '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
  inp=Path('/tmp/native-resume-questions.json');inp.write_text(json.dumps([{'id':i,'arm':a,'task':{k:lookup[i][k] for k in ('state','question','labels')}} for i,a in selected]))
  proc,log,info=exp.boot.setup(exp.PIN);result['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out.resolve())],env=env,timeout=1440).returncode
  except subprocess.TimeoutExpired:exitcode='worker_timeout'
  records=[readout.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
  result.update(status='completed',worker_exit=exitcode,records=records,planned_attempts=len(selected),returned_attempts=len(records))
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  exp.stop(proc,log);result['wall_seconds']=time.perf_counter()-started
  exp.write(f'resumed-native-{shard:02d}.json',result)
  with zipfile.ZipFile('resume-source.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(__file__,'resume_frozen.py')
  print('RESULT '+json.dumps({k:result.get(k) for k in ('status','error','planned_attempts','returned_attempts','worker_exit','wall_seconds')}),flush=True)
 if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.worker:worker(a.input,a.output)
 else:run(a.shard)
