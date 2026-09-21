"""Frozen semantic pilot: public inputs only, no memory or training in scored runs."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,re,shutil,subprocess,sys,time,types
from pathlib import Path
HERE=Path(__file__).parent
OLD=HERE.parent/'adaptive_language'
sys.path.insert(0,str(OLD))
import bootstrap as boot
import preflight_entry
from prepare_core import unpack
from benchmark import PIN,GSM,BBH,score
HASHES={'contracts.py':'12a2a9b8d09b03ad2e82138b653bf449cda334e673ecec37fa5ccc11ee58bcfc','engine.py':'d8cd0397b3ca175e588f8d55a6f20e95c46f39137fc44da47456ed684f76fc51'}
OLD_GSM={1043,1055,1064,1074,1076,1093,1123,1127,1131,114,1168,1200,1205,1250,1274,1286,14,164,181,182,216,217,220,275,327,329,331,333,354,391,393,424,429,532,58,581,634,636,641,642,644,673,728,753,770,797,800,801,81,818,829,869,886,892,893,901,902,91,911,933,936,947,975,996}
SALT='adaptive-semantic-v08-locked\0'
ARMS=('reasoning','program','contract')

def sha(b):return hashlib.sha256(b).hexdigest()
def prepare():
    for name,h in HASHES.items():
        actual=sha((HERE/name).read_bytes())
        if actual!=h:raise ValueError(f'Frozen source mismatch {name}: {actual}')
    root,receipt=unpack();sys.path.insert(0,str(root))
    package=types.ModuleType('semantic_v08');package.__path__=[str(HERE)];sys.modules['semantic_v08']=package
    from semantic_v08.contracts import execute_contract,ProgramError
    q='There are 3 boxes with 12 pieces per box. Half are removed. How many pieces remain?'
    plan={'target':'How many pieces remain?','quantities':[{'name':'boxes','value':'3','unit':'box','source':'3 boxes'},{'name':'per_box','value':'12','unit':'piece/box','source':'12 pieces per box'},{'name':'fraction','value':'1/2','unit':'1','source':'Half'}],'steps':[{'name':'total','expr':'boxes*per_box'}],'result':'total*(1-fraction)','answer_unit':'piece'}
    assert execute_contract(q,plan)['answer']=='18'
    bad=json.loads(json.dumps(plan));bad['quantities'][0]['value']='4'
    try:execute_contract(q,bad)
    except ProgramError:pass
    else:raise AssertionError('Missing literal guard')
    return receipt

def cohort():
    sources=[]
    def get(url):
        raw=boot.get(url);sources.append({'url':url,'sha256':sha(raw),'bytes':len(raw)});return raw
    raw=get(f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/grade_school_math/data/test.jsonl')
    allgsm=[json.loads(x) for x in raw.splitlines() if x.strip()]
    assert len(allgsm)==1319 and len(OLD_GSM)==64
    exclusion_hashes={sha(allgsm[i]['question'].encode()) for i in OLD_GSM}
    gsm=sorted([(i,r) for i,r in enumerate(allgsm) if i not in OLD_GSM],key=lambda p:sha((SALT+p[1]['question']).encode()))[:32]
    tasks=[{'id':f'gsm8k:{i}','suite':'GSM8K','family':'gsm8k','question':r['question'],'reference':r['answer'].split('####')[-1].strip()} for i,r in gsm]
    tree=boot.json_get(f'https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/git/trees/{BBH}?recursive=1')
    if tree.get('truncated'):raise ValueError('Incomplete tree')
    paths=sorted(e['path'] for e in tree['tree'] if re.fullmatch(r'bbh/[^/]+\.json',e['path']))
    assert len(paths)==27
    for path in paths:
        rows=json.loads(get(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH}/{path}'))['examples']
        old=sorted(enumerate(rows),key=lambda p:sha(('adaptive-language-v07-20260921\0'+p[1]['input']).encode()))[:2]
        oldids={i for i,_ in old};exclusion_hashes.update(sha(r['input'].encode()) for _,r in old)
        i,r=min(((i,r) for i,r in enumerate(rows) if i not in oldids),key=lambda p:sha((SALT+p[1]['input']).encode()))
        family=Path(path).stem
        tasks.append({'id':f'bbh:{family}:{i}','suite':'BBH','family':family,'question':r['input'],'reference':r['target']})
    tasks.sort(key=lambda r:r['id']);assert len(tasks)==59
    assert not exclusion_hashes.intersection(sha(r['question'].encode()) for r in tasks)
    return tasks,sources

def stop(proc,log):
    if proc:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    if log:log.close()

def preflight():
    receipt=prepare();proc=log=None;r={'status':'failed','core':receipt,'source_hashes':HASHES,'run_id':os.environ.get('GITHUB_RUN_ID')}
    try:
        proc,log,info=boot.setup(PIN);r['runtime']=info
        from adaptive_language.engine import LocalChat
        from semantic_v08.engine import solve
        examples=[('A machine produces 15 items each hour for 4 hours. How many items does it produce?', '60'),('Mira completes half of a 600-piece puzzle. How many pieces has Mira placed?', '300'),('Put these words in alphabetical order: amber, cobalt, birch.','amber birch cobalt')]
        r['rows']=[dict(solve(q,'contract',LocalChat()),question=q,expected=a) for q,a in examples]
        r['status']='completed';r['all_transport_completed']=all(x['status']!='transport_error' for x in r['rows'])
    except Exception as e:r['error']=f'{type(e).__name__}: {e}'
    finally:
        stop(proc,log);Path('semantic-preflight.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False));print('PREFLIGHT '+json.dumps(r),flush=True)
    if r['status']!='completed' or not r.get('all_transport_completed'):raise SystemExit(1)

def worker(inp,out):
    prepare();from adaptive_language.engine import LocalChat
    from semantic_v08.engine import solve
    answers=[];transport=LocalChat(timeout=240)
    for task in json.loads(Path(inp).read_text()):
        if set(task)!={'id','question'}:raise ValueError('Unexpected solver input')
        offset=int(sha(task['id'].encode())[:8],16)%3;arms=ARMS[offset:]+ARMS[:offset]
        for arm in arms:
            t=time.perf_counter()
            try:r=solve(task['question'],arm,transport)
            except Exception as e:r={'arm':arm,'answer':None,'status':'unexpected_error','calls':[],'usage':{},'seconds':time.perf_counter()-t,'errors':[f'{type(e).__name__}: {e}']}
            r['id']=task['id'];answers.append(r)
            Path(out).write_text(json.dumps(answers,ensure_ascii=False,allow_nan=False))
            print('ATTEMPT '+json.dumps({k:r.get(k) for k in ('id','arm','status','seconds','usage','errors')}),flush=True)

def run(shard,shards):
    receipt=prepare();alltasks,sources=cohort();tasks=[r for i,r in enumerate(alltasks) if i%shards==shard]
    proc=log=None;start=time.perf_counter()
    result={'status':'failed','core':receipt,'source_hashes':HASHES,'runner_sha256':sha(Path(__file__).read_bytes()),'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'shard':shard,'shards':shards,'sources':sources,'selected_ids':[r['id'] for r in alltasks],'exclusion_gsm_ids':sorted(OLD_GSM)}
    lock={'source_hashes':HASHES,'runner_sha256':result['runner_sha256'],'model_pin':PIN,'selected_ids':result['selected_ids'],'sources':sources,'arms':ARMS,'max_completion_tokens':1536,'plan_tokens':1024,'repair_tokens':512,'native_thinking':False,'temperature':0,'memory':False,'training':False,'selection_salt':SALT,'same_token_cap_not_equal_compute':True}
    result['protocol_lock']=lock;print('PROTOCOL_LOCK '+json.dumps(lock),flush=True)
    temp=Path('/tmp/semantic-questions');temp.mkdir(exist_ok=True);inp=temp/'input.json';out=temp/'output.json'
    inp.write_text(json.dumps([{k:r[k] for k in ('id','question')} for r in tasks]))
    try:
        proc,log,info=boot.setup(PIN);result['runtime']=info
        env={k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ('TOKEN','SECRET','KEY'))}
        try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=20*60).returncode
        except subprocess.TimeoutExpired:exitcode='worker_timeout'
        rows=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in rows}
        for t in tasks:
            for arm in ARMS:
                if (t['id'],arm) not in seen:rows.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'usage':{},'seconds':None})
        lookup={r['id']:r for r in tasks}
        for r in rows:
            t=lookup[r['id']];r.update(question=t['question'],reference=t['reference'],suite=t['suite'],family=t['family'],correct=score(r['answer'],t['reference'],t['suite']))
        result.update(status='completed',worker_exit=exitcode,rows=rows)
        result['summary']={a:{'n':sum(r['arm']==a for r in rows),'correct':sum(r['correct'] for r in rows if r['arm']==a),'statuses':dict(collections.Counter(r['status'] for r in rows if r['arm']==a))} for a in ARMS}
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        stop(proc,log);result['wall_seconds']=time.perf_counter()-start
        Path(f'semantic-shard-{shard:02d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        print('SUMMARY '+json.dumps({k:v for k,v in result.items() if k in ('status','error','summary','worker_exit','wall_seconds')}),flush=True)
    if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=12);a=p.parse_args()
    if a.prepare:print('PREPARED '+json.dumps(prepare()),flush=True)
    elif a.preflight:preflight()
    elif a.worker:worker(a.input,a.output)
    else:run(a.shard,a.shards)
