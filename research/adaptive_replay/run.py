"""Bounded public-data experiment. No private user data or benchmark learning."""
from __future__ import annotations
import argparse,collections,importlib.util,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('prior_semantic_runner',HERE.parent/'adaptive_semantic/run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
import core
boot=prior.boot
SALT='adaptive-relational-replay-v09-fixed\0'
ARMS=('reasoning','replay','sham','native')

def cohort():
 old,sources=prior.cohort()
 excluded_gsm=prior.OLD_GSM|{int(r['id'].split(':')[1]) for r in old if r['suite']=='GSM8K'}
 raw=boot.get(f'https://raw.githubusercontent.com/openai/grade-school-math/{prior.GSM}/grade_school_math/data/test.jsonl')
 data=[json.loads(x) for x in raw.splitlines() if x.strip()]
 # Reserve also the eight selected but unexecuted v0.8 schema-follow-up questions.
 reserved=sorted(((i,r) for i,r in enumerate(data) if i not in excluded_gsm),key=lambda p:core.sha(('adaptive-schema-v08-followup\0'+p[1]['question']).encode()))[:8]
 excluded_gsm|={i for i,r in reserved};assert len(excluded_gsm)==104
 gsm=sorted(((i,r) for i,r in enumerate(data) if i not in excluded_gsm),key=lambda p:core.sha((SALT+p[1]['question']).encode()))[:12]
 tasks=[{'id':f'gsm8k:{i}','suite':'GSM8K','family':'gsm8k','question':r['question'],'reference':r['answer'].split('####')[-1].strip()} for i,r in gsm]
 bbhs=[r for r in old if r['suite']=='BBH']
 chosen_families=sorted(bbhs,key=lambda r:core.sha((SALT+r['family']).encode()))[:12]
 for earlier in chosen_families:
  family=earlier['family'];url=f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{prior.BBH}/bbh/{family}.json'
  raw=boot.get(url);rows=json.loads(raw)['examples']
  excluded={int(earlier['id'].split(':')[-1])}
  excluded|={i for i,r in sorted(enumerate(rows),key=lambda p:core.sha(('adaptive-language-v07-20260921\0'+p[1]['input']).encode()))[:2]}
  i,r=min(((i,r) for i,r in enumerate(rows) if i not in excluded),key=lambda p:core.sha((SALT+p[1]['input']).encode()))
  tasks.append({'id':f'bbh:{family}:{i}','suite':'BBH','family':family,'question':r['input'],'reference':r['target']})
 tasks.sort(key=lambda r:r['id']);assert len(tasks)==24 and len(set(r['id'] for r in tasks))==24
 assert not set(r['id'] for r in tasks)&set(r['id'] for r in old)
 return tasks,{'sources':sources,'excluded_gsm_ids':sorted(excluded_gsm),'selection_salt':SALT,'selected_bbh_families':[r['family'] for r in chosen_families]}

def memory(path):
 m=core.Memory(path,'frozen-research')
 if not m.read():
  for r in core.bank_records():m.add(r)
 m.close();m=core.Memory(path,'frozen-research');rs=m.read();m.close()
 assert rs==sorted(core.bank_records(),key=lambda r:r['id'])
 return rs

def preflight():
 checks=core.checks();proc=log=None;result={'status':'failed','checks':checks,'run_id':os.environ.get('GITHUB_RUN_ID')}
 try:
  rs=memory('/tmp/replay-preflight/memory.sqlite3');proc,log,info=boot.setup(prior.PIN);result['runtime']=info
  examples=[('Two batches take 18 and 30 minutes. What is their arithmetic mean duration in minutes?','24'),('Rina has placed half of a 420-piece puzzle. How many pieces has Rina placed?','210')]
  out=[]
  for q,answer in examples:
   for arm in ('reasoning','replay','native'):
    r=core.solve(q,arm,rs);r.update(question=q,expected=answer,correct=prior.score(r['answer'],answer,'GSM8K'));out.append(r)
  result['rows']=out;result['ready']=all(r['correct'] for r in out if r['arm']!='native')
  result['native_ready']=all(r['correct'] for r in out if r['arm']=='native')
  result['status']='completed'
  if os.environ.get('GITHUB_OUTPUT'):
   with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('ready='+str(result['ready']).lower()+'\nnative='+str(result['native_ready']).lower()+'\n')
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  prior.stop(proc,log);Path('replay-preflight.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
  with zipfile.ZipFile('replay-source.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in HERE.glob('*.py'):z.write(p,'replay/'+p.name)
  print('PREFLIGHT '+json.dumps({k:v for k,v in result.items() if k not in ('rows','runtime')}),flush=True)
 if result['status']!='completed' or not result.get('ready'):raise SystemExit(1)

def worker(inp,out,native):
 tasks=json.loads(Path(inp).read_text());rs=memory('/tmp/replay-worker/memory.sqlite3');before=core.sha(core.canon(rs).encode());results=[]
 arms=ARMS if native else ARMS[:-1]
 for t in tasks:
  if set(t)!={'id','question'}:raise ValueError('solver_input_schema')
  offset=int(core.sha(t['id'].encode())[:8],16)%len(arms)
  for arm in arms[offset:]+arms[:offset]:
   r=core.solve(t['question'],arm,rs);r['id']=t['id'];results.append(r)
   Path(out).write_text(json.dumps(results,ensure_ascii=False,allow_nan=False))
   print('ATTEMPT '+json.dumps({k:r[k] for k in ('id','arm','status','seconds','error')}),flush=True)
 assert core.sha(core.canon(memory('/tmp/replay-worker/memory.sqlite3')).encode())==before

def run(shard,native):
 core.checks();tasks,manifest=cohort();rows=[r for i,r in enumerate(tasks) if i%6==shard]
 arms=ARMS if native else ARMS[:-1]
 result={'status':'failed','run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'shard':shard,'source_hashes':{p.name:core.sha(p.read_bytes()) for p in HERE.glob('*.py')},'manifest':manifest,'selected_ids':[r['id'] for r in tasks]}
 result['protocol']={'arms':arms,'one_call_per_arm':True,'legacy_max_tokens':1536,'native_max_tokens':2048,'native_temperature':1.,'other_temperature':0.,'native_top_p':.95,'native_top_k':20,'native_presence_penalty':1.5,'seed':1707,'model_pin':prior.PIN,'persistent_advisory_memory':True,'memory_frozen_during_scores':True,'trained_llm_weights':False,'trained_retriever_used':False,'sham':'same memory count from lowest lexical relevance; token lengths not exactly matched','energy_measured':False,'frontier_reference_evaluated':False,'official_leaderboard':False,'comparability':'native arm changes both inference mode, sampling and cap; not a clean biology-only intervention'}
 print('LOCK '+json.dumps(result),flush=True)
 proc=log=None;started=time.perf_counter();temp=Path('/tmp/replay-benchmark');temp.mkdir(exist_ok=True)
 inp=temp/'questions.json';out=temp/'answers.json';inp.write_text(json.dumps([{k:r[k] for k in ('id','question')} for r in rows]))
 try:
  proc,log,info=boot.setup(prior.PIN);result['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  cmd=[sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)]+(['--native'] if native else [])
  try:exitcode=subprocess.run(cmd,env=env,timeout=20*60).returncode
  except subprocess.TimeoutExpired:exitcode='worker_timeout'
  predicted=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in predicted}
  for r in rows:
   for arm in arms:
    if (r['id'],arm) not in seen:predicted.append({'id':r['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'seconds':None,'memories':[]})
  lookup={r['id']:r for r in rows}
  for r in predicted:
   t=lookup[r['id']];r.update(question=t['question'],reference=t['reference'],suite=t['suite'],family=t['family'],correct=prior.score(r['answer'],t['reference'],t['suite']))
  result.update(status='completed',worker_exit=exitcode,rows=predicted)
  result['summary']={a:{'correct':sum(r['correct'] for r in predicted if r['arm']==a),'n':sum(r['arm']==a for r in predicted),'statuses':dict(collections.Counter(r['status'] for r in predicted if r['arm']==a))} for a in arms}
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  prior.stop(proc,log);result['wall_seconds']=time.perf_counter()-started
  Path(f'replay-shard-{shard:02d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
  print('SUMMARY '+json.dumps({k:v for k,v in result.items() if k in ('status','summary','error','worker_exit','wall_seconds')}),flush=True)
 if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--native',action='store_true');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.preflight:preflight()
 elif a.worker:worker(a.input,a.output,a.native)
 else:
  if not 0<=a.shard<6:raise ValueError('shard')
  run(a.shard,a.native)
