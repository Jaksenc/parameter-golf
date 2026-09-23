"""Recovery of absent outputs only, not score-driven retries or changed inference."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import full_stage1 as old
import remote_stage1 as original
MISSING=[['easy-extraction-02','direct'],['easy-extraction-02','planned'],['easy-extraction-06','direct'],['easy-extraction-06','planned'],['easy-extraction-08','direct'],['easy-extraction-08','planned'],['easy-fact-01','direct'],['easy-fact-01','planned'],['easy-fact-08','direct'],['easy-fact-08','planned'],['easy-fact-09','direct'],['easy-fact-09','planned'],['easy-tool_selection-06','planned'],['easy-tool_selection-08','direct'],['easy-tool_selection-08','planned'],['hard-opus-a-long_policy-01','direct'],['hard-opus-a-long_policy-01','planned'],['hard-opus-a-long_policy-04','direct'],['hard-opus-a-probability-04','direct'],['hard-opus-a-probability-04','planned'],['hard-opus-a-probability-07','direct'],['hard-opus-a-probability-07','planned'],['hard-opus-a-probability-08','direct'],['hard-opus-a-probability-08','planned'],['hard-opus-b-ambiguous-10','direct'],['hard-opus-b-ambiguous-10','planned'],['hard-opus-b-ambiguous-11','planned'],['hard-opus-b-multi_hop-03','direct'],['hard-opus-b-multi_hop-03','planned'],['hard-opus-c-long_policy-03','direct'],['hard-opus-c-long_policy-03','planned'],['hard-opus-c-long_policy-04','direct'],['hard-opus-c-long_policy-04','planned'],['hard-opus-c-long_policy-10','direct'],['hard-opus-c-long_policy-10','planned'],['hard-sol-a-multi_hop-01','direct'],['hard-sol-a-multi_hop-05','direct'],['hard-sol-a-multi_hop-05','planned'],['hard-sol-a-multi_hop-09','planned'],['hard-sol-a-trap-13','direct'],['hard-sol-a-trap-13','planned'],['hard-sol-b-routing_hard-01','direct'],['hard-sol-b-routing_hard-01','planned'],['hard-sol-c-judge_hard-09','direct'],['hard-sol-c-judge_hard-09','planned'],['original-adequacy-02-1','direct'],['original-extraction-06-1','direct'],['original-extraction-06-1','planned'],['original-policy-05-0','direct'],['original-policy-05-0','planned'],['original-policy-06-1','direct'],['original-policy-06-1','planned']]
LOCK='09ecb7b105198e07f56a2b60302f50a4517a9b0637dc8a6ecd3598944f92d566'
def worker(inp,out):
 root,_=original.setup_source()
 assert original.sha((root/'shared_system/core.py').read_bytes())==old.CORE
 from shared_system.core import LocalCore,first_pass
 from shared_system.contracts import Task
 core=LocalCore(model_identity='Qwen3.5-4B-Q4_K_M:'+original.PIN['weights']['sha256']);answers=[]
 for row in json.loads(Path(inp).read_text()):
  assert set(row)=={'id','source','task'} and row['source'] in ('direct','planned')
  value=first_pass(Task.create(row['task']),core,planned=row['source']=='planned');value['id']=row['id'];answers.append(value)
  Path(out).write_text(json.dumps(answers,ensure_ascii=False,allow_nan=False))
  old.emit('checkpoint.json',{'answer':value})
def run(shard):
 assert original.sha(json.dumps(MISSING,separators=(',',':')).encode())==LOCK
 proc=log=None;start=time.perf_counter();result={'status':'failed','shard':shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'missing_lock':LOCK,'original_run':35802769624,'recovery_not_new_independent_cohort':True,'only_absent_outputs_restarted':True,'existing_error_outputs_not_retried':True,'core_sha256':old.CORE}
 try:
  root,identity,remaining,manifest,excluded=old.acquire();lookup={r['id']:r for r in remaining}
  assert len(MISSING)==52 and all(i in lookup for i,a in MISSING)
  ordered=sorted(MISSING,key=lambda p:original.sha(('recovery-order\0'+json.dumps(p)).encode()))
  pairs=[p for i,p in enumerate(ordered) if i%13==shard]
  questions=[{'id':i,'source':a,'task':{k:lookup[i][k] for k in ('state','question','labels')}} for i,a in pairs]
  tmp=Path('/tmp/system1-missing');tmp.mkdir(exist_ok=True);inp=tmp/'input.json';out=tmp/'output.json';inp.write_text(json.dumps(questions))
  result.update(pairs=pairs,manifest=manifest,source_bundle_sha256=identity,protocol={'same_caps':[1280,768],'temperature':0,'native_thinking':False,'seed':1707,'request_timeout':240,'no_paid_api':True,'no_training':True,'no_memory':True,'no_score_feedback':True})
  print('RECOVERY_LOCK '+json.dumps(result),flush=True)
  proc,log,info=original.boot.setup(original.PIN);result['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  try:code=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=1100).returncode
  except subprocess.TimeoutExpired:code='timeout'
  result.update(status='completed',worker_exit=code,answers=json.loads(out.read_text()) if out.exists() else [])
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  original.stop(proc,log);result['wall_seconds']=time.perf_counter()-start
  old.emit(f'recovered-readouts-{shard:02d}.json',result)
  with zipfile.ZipFile(f'recovery-source-{shard:02d}.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in Path(__file__).parent.glob('*.py'):z.write(p,'research/shared_decisions/'+p.name)
   p=Path(__file__).with_name('source_bundle.json');z.write(p,'research/shared_decisions/'+p.name)
   for p in (Path('/tmp/adaptive-shared-core')).rglob('*.py'):z.write(p,str(p.relative_to('/tmp/adaptive-shared-core')))
 if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int);a=p.parse_args()
 if a.worker:worker(a.input,a.output)
 else:
  if a.shard is None or not 0<=a.shard<13:raise ValueError('shard')
  run(a.shard)
