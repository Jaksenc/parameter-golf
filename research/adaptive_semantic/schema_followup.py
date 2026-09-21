"""Separate follow-up after preflight/partial primary diagnostics. No primary changes."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import run as primary
boot=primary.boot
SALT='adaptive-schema-v08-followup\0'
SCHEMA_HASH='b99c9e4324e0ded15d98b136651aa3abd3401d0f2c47365ddad303567389b530'
ARMS=('reasoning','strict_first','schema_first')
def sha(b):return hashlib.sha256(b).hexdigest()
def prepare():
 r=primary.prepare()
 assert sha(Path(__file__).with_name('schema.py').read_bytes())==SCHEMA_HASH
 return r

def solve(q,arm):
 from adaptive_language.engine import LocalChat,extract_final,TransportError
 from adaptive_language.program import ProgramError,parse_plan
 from semantic_v08.engine import CONTRACT,REASON
 from semantic_v08.contracts import execute_contract
 from semantic_v08.schema import ConstrainedChat,schema_for
 start=time.perf_counter();calls=[];answer=None;ex=None;error=None;status='missing'
 system=REASON if arm=='reasoning' else CONTRACT
 try:
  transport=ConstrainedChat(q) if arm=='schema_first' else LocalChat()
  c=transport([{'role':'system','content':system},{'role':'user','content':q}],1024,arm!='reasoning');calls.append(c)
  if c['finish_reason']!='stop':raise ProgramError('incomplete_generation')
  if arm=='reasoning':
   answer=extract_final(c['text']);status='model_answer_unverified' if answer is not None else 'missing_answer'
  else:
   plan=parse_plan(c['text']);ex=execute_contract(q,plan);ex['plan']=plan;answer=ex['answer'];status=ex['status']
 except (ProgramError,TransportError,ValueError) as e:error=str(e);status='rejected_or_failed'
 return {'arm':arm,'answer':answer,'status':status,'execution':ex,'error':error,'calls':calls,'seconds':time.perf_counter()-start,'schema':schema_for(q) if arm=='schema_first' else None,'semantic_verification':False}

def cohort():
 old,sources=primary.cohort();excluded=primary.OLD_GSM|{int(r['id'].split(':')[1]) for r in old if r['suite']=='GSM8K'}
 assert len(excluded)==96
 raw=boot.get(f'https://raw.githubusercontent.com/openai/grade-school-math/{primary.GSM}/grade_school_math/data/test.jsonl')
 data=[json.loads(x) for x in raw.splitlines() if x.strip()]
 chosen=sorted(((i,r) for i,r in enumerate(data) if i not in excluded),key=lambda p:sha((SALT+p[1]['question']).encode()))[:8]
 return [{'id':f'gsm8k:{i}','question':r['question'],'reference':r['answer'].split('####')[-1].strip()} for i,r in chosen],sorted(excluded),sha(raw)

def worker(inp,out):
 prepare();rs=[]
 for t in json.loads(Path(inp).read_text()):
  assert set(t)=={'id','question'}
  k=int(sha(t['id'].encode())[:8],16)%3
  for arm in ARMS[k:]+ARMS[:k]:
   r=solve(t['question'],arm);r['id']=t['id'];rs.append(r);Path(out).write_text(json.dumps(rs,ensure_ascii=False,allow_nan=False))
   print('FOLLOWUP_ATTEMPT '+json.dumps({k:r[k] for k in ('id','arm','status','seconds','error')}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.worker:return worker(a.input,a.output)
 assert a.shard in (0,1)
 receipt=prepare();allrows,excluded,datahash=cohort();tasks=[r for i,r in enumerate(allrows) if i%2==a.shard]
 result={'status':'failed','run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'source_hash':sha(Path(__file__).read_bytes()),'schema_source_hash':SCHEMA_HASH,'core':receipt,'shard':a.shard,'selected_ids':[r['id'] for r in allrows],'excluded_gsm_ids':excluded,'data_sha256':datahash,'protocol':{'arms':ARMS,'max_tokens':1024,'one_call_per_arm':True,'salt':SALT,'labels_solver_visible':False,'native_thinking':False,'memory':False,'training':False,'model_pin':primary.PIN,'scope':'Eight disjoint GSM8K diagnostic questions, after partial primary analysis; not a full benchmark.'}}
 print('LOCK '+json.dumps(result),flush=True)
 proc=log=None;started=time.perf_counter()
 try:
  proc,log,info=boot.setup(primary.PIN);result['runtime']=info
  q='There are 3 boxes of 12 pieces each. Half are removed. How many pieces remain?'
  pre=solve(q,'schema_first');result['preflight']=pre
  if pre['answer']!='18' or not (pre.get('execution') or {}).get('contract_checked'):
   result['status']='preflight_rejected';raise ValueError('Constrained numeric preflight did not produce a correct checked contract')
  temp=Path('/tmp/schema-eval');temp.mkdir(exist_ok=True);inp=temp/'questions.json';out=temp/'outputs.json'
  inp.write_text(json.dumps([{k:r[k] for k in ('id','question')} for r in tasks]))
  env={k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ('TOKEN','SECRET','KEY'))}
  try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=15*60).returncode
  except subprocess.TimeoutExpired:exitcode='worker_timeout'
  rs=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in rs}
  for t in tasks:
   for arm in ARMS:
    if (t['id'],arm) not in seen:rs.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'seconds':None})
  lookup={r['id']:r for r in tasks}
  for r in rs:
   t=lookup[r['id']];r.update(question=t['question'],reference=t['reference'],correct=primary.score(r['answer'],t['reference'],'GSM8K'))
  result.update(status='completed',worker_exit=exitcode,rows=rs)
 except Exception as e:result['error']=f'{type(e).__name__}: {e}'
 finally:
  primary.stop(proc,log);result['wall_seconds']=time.perf_counter()-started
  Path(f'schema-shard-{a.shard:02d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
  print('FOLLOWUP_RESULT '+json.dumps({k:v for k,v in result.items() if k not in ('rows','runtime','excluded_gsm_ids','preflight')}),flush=True)
 if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
