"""Readout-only comparison. Frozen weights; no benchmark-trained selector or calibration.
No private/sealed data. Per-attempt durable records precede any offline scoring.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import readout
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'adaptive_language'))
import bootstrap as boot
import preflight_entry  # exact CPU-asset correction used by the previous experiment
PIN={'model_revision':'e87f176479d0855a907a41277aca2f8ee7a09523','weights':{'sha256':'00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'},'runtime':{'sha256':'9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'}}
JEV='51a8d73fa798aa337bb1b26abd10995c0ab847e9'
DATA_SHA={'easy':'231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b','original':'5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180','hard':'89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb'}
SALT='adaptive-native-v02-permutation-20260923\0'
SHARDS=8

def stop(proc,log):
 if proc:
  proc.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 if log:log.close()

def write(path,obj):
 raw=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False).encode()
 with Path(path).open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return readout.digest(raw)

def bundle():
 with zipfile.ZipFile('native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(HERE.glob('*.py')):z.write(p,'native_decisions/'+p.name)
  for name in ('bootstrap.py','preflight_entry.py'):
   p=HERE.parent/'adaptive_language'/name;z.write(p,'bootstrap/'+name)

def fixtures():
 return [
 {'state':'The sole approved color is green. Red and blue are not approved.','question':{'type':'choice','instructions':'Which color is approved?','criteria':{'blue':'Blue is approved.','green':'Green is approved.','red':'Red is approved.'}},'labels':['red','blue','green'],'expected':'green'},
 {'state':'The report explicitly states that the payment has not been received.','question':{'type':'noul','instructions':'Has the payment been received?','criteria':{'false':'The payment has not been received.','true':'The payment has been received.'}},'labels':['no','yes'],'expected':'no'},
 {'state':'Impact report: cosmetic only; every function works normally.','question':{'type':'score','instructions':'Choose the highest supported impact level.','criteria':['Cosmetic only; no functional impairment.','One function impaired.','All core functions unavailable.']},'labels':['0','1','2'],'expected':0}]

def preflight():
 result={'status':'failed','run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'unit_checks':readout.checks()};proc=log=None
 try:
  proc,log,info=boot.setup(PIN);result['runtime']=info;b=readout.Backend();records=[]
  for t in fixtures():
   task={k:t[k] for k in ('state','question','labels')}
   for arm in ('native_input','native_canonical','canonical_reverse'):
    r=b.solve(task,arm);r.update(task=task,expected=t['expected'])
    r['correct']=r['ok'] and min(r['probs'],key=lambda k:(-r['probs'][k],k))==str(t['expected']);records.append(r)
  result.update(status='completed',rows=records,ready=all(r['correct'] for r in records))
  # Transport/syntax alone is insufficient: the fixed readiness answers must be correct.
  if os.environ.get('GITHUB_OUTPUT'):
   with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('ready='+str(result['ready']).lower()+'\n')
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  stop(proc,log);write('native-preflight.json',result);bundle()
  print('PREFLIGHT '+json.dumps({k:v for k,v in result.items() if k not in ('rows','runtime')}),flush=True)
 if not result.get('ready'):raise SystemExit(1)

def acquire():
 rows=[];manifest=[]
 for tier,h in DATA_SHA.items():
  url=f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/datasets/public/{tier}.jsonl';raw=boot.get(url)
  if readout.digest(raw)!=h:raise ValueError('dataset_identity')
  rs=[json.loads(x) for x in raw.splitlines() if x.strip()]
  for r in rs:r['tier']=tier
  rows+=rs;manifest.append({'tier':tier,'sha256':h,'url':url,'n':len(rs)})
 if len(rows)!=231 or len({r['id'] for r in rows})!=231:raise ValueError('cohort')
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['tier'],r['family'])].append(r)
 probe={min(rs,key=lambda r:readout.digest(SALT+readout.canonical({k:r[k] for k in ('state','question','labels')})))['id'] for rs in grouped.values()}
 if len(probe)!=20:raise ValueError('probe_count')
 rows.sort(key=lambda r:readout.digest(SALT+'allocation'+r['id']))
 return rows,probe,manifest

def worker(inp,journal):
 packets=json.loads(Path(inp).read_text());b=readout.Backend()
 with Path(journal).open('x') as f:
  for t in packets:
   if set(t)!={'id','task','probe'} or set(t['task'])!={'state','question','labels'}:raise ValueError('solver_packet')
   arms=['native_input','native_canonical']+(['native_reverse','canonical_reverse','keyed'] if t['probe'] else [])
   offset=int(readout.digest(t['id'])[:8],16)%len(arms);arms=arms[offset:]+arms[:offset]
   for arm in arms:
    r=b.solve(t['task'],arm);r.update(id=t['id'],task_sha256=readout.digest(readout.canonical(t['task'])))
    f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print('ATTEMPT '+json.dumps({k:r.get(k) for k in ('id','arm','ok','error','seconds')}),flush=True)

def run(shard):
 proc=log=None;started=time.perf_counter();out=Path(f'native-journal-{shard:02d}.jsonl')
 result={'status':'failed','shard':shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'core_sha256':readout.digest(HERE.joinpath('readout.py').read_bytes()),'driver_sha256':readout.digest(Path(__file__).read_bytes())}
 try:
  readout.checks();allrows,probe,manifest=acquire();rows=[r for i,r in enumerate(allrows) if i%SHARDS==shard]
  packets=[{'id':r['id'],'task':{k:r[k] for k in ('state','question','labels')},'probe':r['id'] in probe} for r in rows]
  result.update(manifest=manifest,probe_ids=sorted(probe),selected_ids=[r['id'] for r in allrows],cases=rows)
  result['protocol']={'public_only':True,'upstream_commit':JEV,'model_pin':PIN,'arms_all':['native_input','native_canonical'],'probe_arms':['native_reverse','canonical_reverse','keyed'],'native_forward_tokens':1,'keyed_cap':1280,'no_token_filtering_except_option_grammar':True,'canonicalization':'criterion then label for choice/noul; numeric level for score','score_scope':'public development/regression; no official v1.4 composite','weights_trained':False,'calibration_fitted':False,'memory':False,'tools':False,'seed':1707,'native_temperature':1.,'keyed_temperature':0.,'input_labels_hidden_from_worker':True,'no_retries':True,'cpu_only':True,'worker_timeout_seconds':1500,'no_paid_inference':True,'cost_usd':None,'energy':None}
  print('PROTOCOL_LOCK '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
  tmp=Path('/tmp/native-decisions');tmp.mkdir(exist_ok=True);inp=tmp/'questions.json';inp.write_text(json.dumps(packets))
  proc,log,info=boot.setup(PIN);result['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--journal',str(out.resolve())],env=env,timeout=1500).returncode
  except subprocess.TimeoutExpired:exitcode='worker_timeout'
  answers=[json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
  result.update(status='completed',worker_exit=exitcode,records=answers,planned_attempts=sum(5 if p['probe'] else 2 for p in packets),returned_attempts=len(answers))
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  stop(proc,log);result['wall_seconds']=time.perf_counter()-started;write(f'native-shard-{shard:02d}.json',result);bundle()
  print('COMPLETE '+json.dumps({k:v for k,v in result.items() if k in ('status','error','planned_attempts','returned_attempts','worker_exit','wall_seconds')}),flush=True)
 if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--journal');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.preflight:preflight()
 elif a.worker:worker(a.input,a.journal)
 else:
  if not 0<=a.shard<SHARDS:raise ValueError('shard')
  run(a.shard)
