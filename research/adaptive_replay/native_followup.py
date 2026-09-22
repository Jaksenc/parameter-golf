"""Separate configuration study; does not edit or rescore primary replay outputs."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,urllib.request,hashlib,zipfile
from pathlib import Path
import core
import run as primary
PROMPT=core.REASON+' On the FINAL line, output only the answer itself: a bare number without units or explanatory text for numeric questions, or the exact requested label/text for other questions.'
ARMS=('matched_reasoning','native_thinking')

def transport(messages,native=False):
 copied=[dict(m) for m in messages];copied[0]['content']=PROMPT
 payload={'model':'local','messages':copied,'max_tokens':2048,'temperature':1.0 if native else 0,'seed':1707,'stream':False,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':native}}
 if native:payload.update(top_p=.95,top_k=20,min_p=0.,presence_penalty=1.5,repeat_penalty=1.)
 raw=core.canon(payload).encode();t=time.perf_counter()
 req=urllib.request.Request('http://127.0.0.1:8787/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=240) as response:
  data=response.read(2_000_001)
  if len(data)>2_000_000:raise ValueError('response_size')
  obj=json.loads(data)
 c=obj['choices'][0]
 return {'text':c['message'].get('content') or '', 'reasoning_content':c['message'].get('reasoning_content') or '', 'finish_reason':c['finish_reason'],'usage':obj.get('usage',{}),'timings':obj.get('timings',{}),'seconds':time.perf_counter()-t,'request_sha256':core.sha(raw),'request_parameters':{k:v for k,v in payload.items() if k!='messages'}}

def solve(q,arm):
 r=core.solve(q,'native' if arm=='native_thinking' else 'reasoning',[],transport)
 r.update(arm=arm,prompt_sha256=core.sha(PROMPT.encode()))
 return r

def preflight():
 proc=log=None;r={'status':'failed','source_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),'prompt':PROMPT,'source_hash':core.sha(Path(__file__).read_bytes())}
 try:
  proc,log,info=primary.boot.setup(primary.prior.PIN);r['runtime']=info;r['rows']=[]
  for q,expected in [('Two batches take 18 and 30 minutes. What is their arithmetic mean duration in minutes?','24'),('Rina has placed half of a 420-piece puzzle. How many pieces has Rina placed?','210')]:
   for arm in ARMS:
    row=solve(q,arm);row.update(question=q,expected=expected,correct=primary.prior.score(row['answer'],expected,'GSM8K'));r['rows'].append(row)
  r['ready']=all(row['correct'] for row in r['rows']);r['status']='completed'
  if os.environ.get('GITHUB_OUTPUT'):
   with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('ready='+str(r['ready']).lower()+'\n')
 except Exception as e:r['error']=str(e)
 finally:
  primary.prior.stop(proc,log);Path('native-preflight.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False))
  with zipfile.ZipFile('native-source.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(__file__,'native_followup.py')
  print('PREFLIGHT '+json.dumps({k:v for k,v in r.items() if k not in ('rows','runtime')}),flush=True)
 if r['status']!='completed' or not r.get('ready'):raise SystemExit(1)

def worker(inp,out):
 rows=[]
 for t in json.loads(Path(inp).read_text()):
  assert set(t)=={'id','question'}
  arms=ARMS if int(core.sha(t['id'].encode())[:8],16)%2==0 else ARMS[::-1]
  for arm in arms:
   r=solve(t['question'],arm);r['id']=t['id'];rows.append(r);Path(out).write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
   print('ATTEMPT '+json.dumps({k:r[k] for k in ('id','arm','answer','status','seconds')}),flush=True)

def run(shard):
 allrows,manifest=primary.cohort();tasks=[r for i,r in enumerate(allrows) if i%6==shard]
 r={'status':'failed','shard':shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'source_hash':core.sha(Path(__file__).read_bytes()),'prompt':PROMPT,'prompt_sha256':core.sha(PROMPT.encode()),'selected_ids':[t['id'] for t in allrows],'manifest':manifest,'protocol':{'arms':ARMS,'max_tokens_both':2048,'same_prompt':True,'native_sampling':'temperature1,top_p.95,top_k20,presence1.5','ordinary_sampling':'greedy','seed':1707,'memory':False,'training':False,'label_feedback':False,'official_benchmark':False,'scope':'Separate post-readiness-format diagnostic on same previously locked 24-question cohort, not additional independent task evidence. Cap/mode/sampling differ from primary study; this is not frontier maximum-performance configuration.'}}
 print('LOCK '+json.dumps(r),flush=True)
 proc=log=None;started=time.perf_counter();tmp=Path('/tmp/native-eval');tmp.mkdir(exist_ok=True);inp=tmp/'questions.json';out=tmp/'outputs.json'
 inp.write_text(json.dumps([{k:t[k] for k in ('id','question')} for t in tasks]))
 try:
  proc,log,info=primary.boot.setup(primary.prior.PIN);r['runtime']=info
  env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
  try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=20*60).returncode
  except subprocess.TimeoutExpired:exitcode='worker_timeout'
  rows=json.loads(out.read_text()) if out.exists() else [];seen={(x['id'],x['arm']) for x in rows};lookup={t['id']:t for t in tasks}
  for t in tasks:
   for arm in ARMS:
    if (t['id'],arm) not in seen:rows.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'seconds':None})
  for x in rows:
   t=lookup[x['id']];x.update(question=t['question'],reference=t['reference'],suite=t['suite'],family=t['family'],correct=primary.prior.score(x['answer'],t['reference'],t['suite']))
  r.update(status='completed',worker_exit=exitcode,rows=rows)
 except Exception as e:r['error']=f'{type(e).__name__}: {e}'
 finally:
  primary.prior.stop(proc,log);r['wall_seconds']=time.perf_counter()-started;Path(f'native-shard-{shard:02d}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False));print('STATUS '+r['status'],flush=True)
 if r['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.preflight:preflight()
 elif a.worker:worker(a.input,a.output)
 else:
  if not 0<=a.shard<6:raise ValueError('shard')
  run(a.shard)
