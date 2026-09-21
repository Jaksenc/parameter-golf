"""Public benchmark controller. Each solver worker sees id and question only."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,re,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import bootstrap as boot
import preflight_entry
from prepare_core import unpack,EXPECTED
GSM='3101c7d5072418e28b9008a6636bde82a006892c'
BBH='9ee07bd481feebf959a6b59d61ea57bdcf30964d'
SALT='adaptive-language-v07-20260921\0'
PIN={'model_revision':'e87f176479d0855a907a41277aca2f8ee7a09523','weights':{'sha256':'00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'},'runtime':{'sha256':'9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'}}
def sha(x):return hashlib.sha256(x).hexdigest()
def source(url):
    raw=boot.get(url);return raw,{'url':url,'sha256':sha(raw),'bytes':len(raw)}
def acquire(shard,shards):
    manifests=[];data=[]
    raw,manifest=source(f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/grade_school_math/data/test.jsonl');manifests.append(manifest)
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows)!=1319:raise ValueError('Unexpected GSM8K test size')
    for i,row in sorted(enumerate(rows),key=lambda p:sha((SALT+p[1]['question']).encode()))[:32]:
        data.append({'id':f'gsm8k:{i}','suite':'GSM8K','family':'gsm8k','question':row['question'],'reference':row['answer'].split('####')[-1].strip()})
    tree=boot.json_get(f'https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/git/trees/{BBH}?recursive=1')
    if tree.get('truncated'):raise ValueError('Truncated BBH tree')
    tasks=sorted(e['path'] for e in tree['tree'] if re.fullmatch(r'bbh/[^/]+\.json',e['path']))
    if len(tasks)!=27:raise ValueError(f'Expected 27 BBH tasks; got {len(tasks)}')
    for path in tasks:
        raw,manifest=source(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH}/{path}');manifests.append(manifest)
        rows=json.loads(raw)['examples'];family=Path(path).stem
        for i,row in sorted(enumerate(rows),key=lambda p:sha((SALT+p[1]['input']).encode()))[:2]:
            data.append({'id':f'bbh:{family}:{i}','suite':'BBH','family':family,'question':row['input'],'reference':row['target']})
    data.sort(key=lambda r:r['id'])
    if len(data)!=86:raise ValueError('Expected 86 cases')
    return data,[r for i,r in enumerate(data) if i%shards==shard],manifests

def score(answer,reference,suite):
    if answer is None:return False
    def normalized(s):
        s=str(s).strip().lower().replace('**','')
        if s.endswith('.'):s=s[:-1].rstrip()
        return re.sub(r'\s+',' ',s)
    a=normalized(answer);b=normalized(reference)
    if suite=='GSM8K':
        a=a.replace(',','');b=b.replace(',','')
        if a.startswith('$'):a=a[1:]
        try:return Fraction(a)==Fraction(b)
        except (ValueError,ZeroDivisionError):return False
    if a==b:return True
    return bool(re.fullmatch(r'\(?[a-z]\)?',a) and re.fullmatch(r'\(?[a-z]\)?',b) and a.strip('()')==b.strip('()'))

def worker(inp,outp):
    root,receipt=unpack();sys.path.insert(0,str(root))
    from adaptive_language.engine import LocalChat,solve
    rows=json.loads(Path(inp).read_text());out=[];transport=LocalChat()
    for task in rows:
        if set(task)!={'id','question'}:raise ValueError('Unexpected solver fields')
        arms=['direct','reasoning','program'];offset=int(sha(task['id'].encode())[:8],16)%3;arms=arms[offset:]+arms[:offset]
        for arm in arms:
            start=time.perf_counter()
            try:r=solve(task['question'],arm,transport)
            except Exception as e:r={'arm':arm,'answer':None,'status':'unexpected_error','errors':[f'{type(e).__name__}: {e}'],'calls':[],'usage':{},'seconds':time.perf_counter()-start}
            r['id']=task['id'];out.append(r);Path(outp).write_text(json.dumps(out,ensure_ascii=False,allow_nan=False))
            print('ATTEMPT '+json.dumps({'id':r['id'],'arm':arm,'status':r['status'],'seconds':r['seconds'],'usage':r['usage']}),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=12);a=p.parse_args()
    if a.worker:return worker(a.input,a.output)
    root,core=unpack()
    if a.prepare:
        data,_,sources=acquire(0,1)
        print('PREPARED '+json.dumps({'core':core,'count':len(data),'cohort':[{'id':r['id'],'question_sha256':sha(r['question'].encode())} for r in data],'sources':sources}),flush=True)
        return 0
    proc=log=None;result={'status':'failed','shard':a.shard,'shards':a.shards,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'core':core};start=time.perf_counter()
    try:
        allrows,rows,sources=acquire(a.shard,a.shards)
        result.update(sources=sources,selected_ids=[r['id'] for r in allrows],cohort_sha256=sha(json.dumps([{'id':r['id'],'q':sha(r['question'].encode())} for r in allrows],sort_keys=True).encode()))
        temp=Path('/tmp/adaptive-language-eval');temp.mkdir(exist_ok=True);inp=temp/'questions.json';out=temp/'outputs.json'
        inp.write_text(json.dumps([{k:r[k] for k in ('id','question')} for r in rows]))
        proc,log,info=boot.setup(PIN);result['runtime']=info
        env={k:v for k,v in os.environ.items() if 'TOKEN' not in k.upper() and 'SECRET' not in k.upper() and 'KEY' not in k.upper()}
        try:
            child=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=23*60);exitcode=child.returncode
        except subprocess.TimeoutExpired:exitcode='worker_timeout'
        predictions=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in predictions}
        for t in rows:
            for arm in ('direct','reasoning','program'):
                if (t['id'],arm) not in seen:predictions.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'usage':{},'seconds':None})
        lookup={r['id']:r for r in rows}
        for r in predictions:
            t=lookup[r['id']];r.update(suite=t['suite'],family=t['family'],question=t['question'],reference=t['reference'],correct=score(r['answer'],t['reference'],t['suite']))
        result.update(status='completed',worker_exit=exitcode,rows=predictions);result['summary']={}
        for arm in ('direct','reasoning','program'):
            rs=[r for r in predictions if r['arm']==arm]
            result['summary'][arm]={'correct':sum(r['correct'] for r in rs),'n':len(rs),'answered':sum(r['answer'] is not None for r in rs),'seconds':sum(r.get('seconds') or 0 for r in rs),'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in rs),'statuses':dict(collections.Counter(r['status'] for r in rs))}
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        if proc:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if log:log.close()
        result['wall_seconds']=time.perf_counter()-start
        result['protocol']={'arms':{'direct':768,'reasoning':768,'program':[512,256]},'temperature':0,'seed':1707,'native_thinking':False,'memory':False,'training':False,'labels_solver_visible':False,'selection_salt':SALT,'gpu':False,'paid_inference':False,'energy_measured':False,'official_leaderboard':False,'frontier_reference_evaluated':False}
        print('SUMMARY '+json.dumps({k:v for k,v in result.items() if k not in ('rows','sources','selected_ids')}),flush=True)
        print(boot.publish(f'language-shard-{a.shard:02d}.json',result),flush=True)
    return 0 if result['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
