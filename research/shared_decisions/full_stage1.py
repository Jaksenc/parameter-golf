"""Complete the public cohort without rerunning the 20 recovered pilot items.
No solver, prompt, temperature, token cap, or scoring rule is tuned in this continuation.
Only public tasks are supplied; labels/rationales are excluded from worker packets.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import remote_stage1 as original
import entry_v2  # installs the already evaluated readout-v2 source
SHA={'easy':'231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b','original':'5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180','hard':'89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb'}
CORE='946de0d3aa1ae65a547fe0e6efa700ac267d9ffc3636d38bf739d39b6006e3c9'
PILOT_RUN='35786433408'
SHARDS=24

def emit(path,obj):
 raw=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False).encode();Path(path).write_bytes(raw)
 encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
 print('RECEIPT '+json.dumps({'path':path,'sha256':original.sha(raw),'bytes':len(raw)}),flush=True)
 for i in range(0,len(encoded),3000):print('BACKUP_%03d '%(i//3000)+encoded[i:i+3000],flush=True)

def acquire():
 root,identity=original.setup_source()
 if original.sha((root/'shared_system/core.py').read_bytes())!=CORE:raise ValueError('core_changed')
 pilot,manifest=original.acquire();excluded={r['id'] for r in pilot if r['tier']!='authored'}
 if len(excluded)!=20:raise ValueError('pilot_cohort_changed')
 rows=[]
 for tier,h in SHA.items():
  raw=Path('sources',tier+'.jsonl').read_bytes()
  if original.sha(raw)!=h:raise ValueError('data_changed:'+tier)
  for line in raw.splitlines():
   if line.strip():r=json.loads(line);r['tier']=tier;rows.append(r)
 if len(rows)!=231 or len({r['id'] for r in rows})!=231:raise ValueError('public_count')
 remaining=[r for r in rows if r['id'] not in excluded]
 from shared_system.contracts import canonical
 remaining.sort(key=lambda r:original.sha(('adaptive-full-continuation-v1\0'+canonical({k:r[k] for k in ('state','question','labels')})).encode()))
 if len(remaining)!=211:raise ValueError('remaining_count')
 return root,identity,remaining,manifest,sorted(excluded)

def run(shard):
 proc=log=None;started=time.perf_counter()
 result={'status':'failed','shard':shard,'shards':SHARDS,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'driver_sha256':original.sha(Path(__file__).read_bytes()),'pilot_run':PILOT_RUN,'core_sha256':CORE}
 try:
  root,identity,allrows,manifest,excluded=acquire();rows=[r for i,r in enumerate(allrows) if i%SHARDS==shard]
  result.update(source_bundle_sha256=identity,manifest=manifest,pilot_excluded_ids=excluded,selected_ids=[r['id'] for r in allrows],cases=rows)
  result['protocol']={'scope':'remaining 211 of 231 public items; 20 pilot decisions retained separately','upstream_commit':original.JEV,'direct_cap':1280,'plan_cap':768,'reserved_final_cap':512,'model_pin':original.PIN,'temperature':0,'seed':1707,'native_thinking':False,'memory':False,'training':False,'cost_usd':None,'paid_model_api':False,'energy_measured':False,'official_composite':False,'no_test_feedback':True,'max_inference_jobs':SHARDS,'shards_parallel':12,'worker_timeout_seconds':1200,'no_automatic_retry':True}
  print('PROTOCOL_LOCK '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
  temp=Path('/tmp/full-shared-pilot');temp.mkdir(exist_ok=True);inp=temp/'questions.json';out=temp/'answers.json'
  inp.write_text(json.dumps([{'id':r['id'],'task':{k:r[k] for k in ('state','question','labels')}} for r in rows]))
  proc,log,info=original.boot.setup(original.PIN);result['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  try:exitcode=subprocess.run([sys.executable,str(Path(__file__).with_name('entry_v2.py')),'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=1200).returncode
  except subprocess.TimeoutExpired:exitcode='timeout'
  answers=json.loads(out.read_text()) if out.exists() else []
  result.update(status='completed',worker_exit=exitcode,answers=answers,planned_attempts=2*len(rows),returned_attempts=len(answers))
 except Exception as exc:result['error']=f'{type(exc).__name__}: {exc}'
 finally:
  original.stop(proc,log);result['wall_seconds']=time.perf_counter()-started
  emit(f'full-stage1-{shard:02d}.json',result)
 if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 if not 0<=a.shard<SHARDS:raise ValueError('shard')
 run(a.shard)
