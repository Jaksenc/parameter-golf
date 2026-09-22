"""Public-only two-stage evaluation; old trained specialists execute locally between stages."""
from __future__ import annotations
import argparse,base64,collections,gzip,hashlib,json,os,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO='Jaksenc/parameter-golf'
BRANCH='research/adaptive-shared-core-20260922'
JEV='51a8d73fa798aa337bb1b26abd10995c0ab847e9'
PIN={'model_revision':'e87f176479d0855a907a41277aca2f8ee7a09523','weights':{'sha256':'00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'},'runtime':{'sha256':'9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'}}
SALT='adaptive-shared-core-20260922-fixed\0'
sys.path.insert(0,str(HERE.parent/'adaptive_language'))
import bootstrap as boot
import preflight_entry

def sha(raw):return hashlib.sha256(raw).hexdigest()
def setup_source():
    obj=json.loads(HERE.joinpath('source_bundle.json').read_text());data=obj['data']
    # Correct transport transcription only. A fixed digest must match before import or inference.
    repairs={'EXi44d3fXiXlg':'EXi44d3XiXlg','Talq3A5ZFyy':'Talq3AupZFyy','Pp3umuzPJ':'Pp3muzPJ'}
    for bad,good in repairs.items():data=data.replace(bad,good)
    raw=gzip.decompress(base64.b64decode(data,validate=True))
    if sha(raw)!=obj['sha256']:raise ValueError('source_integrity')
    root=Path('/tmp/adaptive-shared-core');root.mkdir(exist_ok=True)
    for name,text in json.loads(raw).items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='shared_system' or p.suffix!='.py':raise ValueError('source_path')
        dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    sys.path.insert(0,str(root));return root,obj['sha256']

def stop(proc,log):
    if proc:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    if log:log.close()

def save(name,obj,publish=True):
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False);Path(name).write_text(text)
    packed=base64.b64encode(gzip.compress(text.encode(),mtime=0)).decode()
    print('BACKUP '+json.dumps({'name':name,'sha256':sha(text.encode()),'gzip_base64':packed}),flush=True)
    token=os.environ.get('GITHUB_TOKEN')
    if not token or not publish:return
    url=f'https://api.github.com/repos/{REPO}/contents/research/shared_decisions/results/{name}'
    body={'message':'Record frozen public-only shared-core experiment','branch':BRANCH,'content':base64.b64encode(text.encode()).decode()}
    headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','User-Agent':'adaptive-shared-core'}
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers,method='PUT')
            result=json.loads(urllib.request.urlopen(req,timeout=40).read());print('PUBLISHED '+result['commit']['sha'],flush=True);return
        except Exception:
            if attempt==7:print('PUBLICATION_FAILED_BACKUP_RETAINED',flush=True);return
            time.sleep(attempt+1)

def authored():
    def task(id,state,labels,criteria,instructions,expected,family):
        return {'id':'authored-'+id,'tier':'authored','family':family,'state':state,'labels':labels,'question':{'type':'choice','instructions':instructions,'criteria':dict(zip(labels,criteria))},'expected':expected,'provenance':{'source':'Authored integration examples, not independent external benchmark','exclude_reason':None}}
    return [
      task('order-1','There are five cards in one strict left-to-right order: Birch, Cedar, Elm, Fir, Pine. Elm is before Fir. Birch is before Cedar. Fir is before Pine. Cedar is before Elm.',['Pine','Elm','Birch'],['Pine is the leftmost.','Elm is the leftmost.','Birch is the leftmost.'],'Which card must be leftmost?','Birch','order'),
      task('order-2','There are four panels in one strict left-to-right order: Amber, Blue, Coral, Dawn. Amber is before Coral. Coral is before Dawn. Blue is the second from the left.',['Dawn','Blue','Amber'],['Dawn is the rightmost.','Blue is the rightmost.','Amber is the rightmost.'],'Which panel must be rightmost?','Dawn','order'),
      task('math-1','An inventory has 37 crates containing 29 identical parts each. Then 118 parts are removed. The remaining parts are divided equally among 5 teams.',['191','189','193'],['Each team receives 191 parts.','Each team receives 189 parts.','Each team receives 193 parts.'],'How many parts does each team receive?','191','arithmetic'),
      task('math-2','One completed batch took 18 minutes and contained 3 parts. A second batch took 42 minutes and contained 9 parts. The requested quantity is the arithmetic mean of the TWO batch durations, not time per part.',['5','30','36'],['5 minutes','30 minutes','36 minutes'],'What is the arithmetic mean batch duration?','30','arithmetic')]

def acquire():
    sources=Path('sources');sources.mkdir(exist_ok=True);allrows=[];manifest=[]
    for tier in ('easy','original','hard'):
        url=f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/datasets/public/{tier}.jsonl'
        raw=boot.get(url);sources.joinpath(tier+'.jsonl').write_bytes(raw);rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
        for row in rows:row['tier']=tier
        allrows+=rows;manifest.append({'tier':tier,'url':url,'sha256':sha(raw),'n':len(rows)})
    groups=collections.defaultdict(list)
    from shared_system.contracts import canonical
    for row in allrows:groups[(row['tier'],row['family'])].append(row)
    selected=[]
    for key,rows in sorted(groups.items()):selected.append(min(rows,key=lambda r:sha((SALT+canonical({k:r[k] for k in ('state','question','labels')})).encode())))
    selected+=authored();selected.sort(key=lambda r:r['id'])
    for name in ('scoring.py','metrics.py','tasks.py'):
        raw=boot.get(f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/jevbench/{name}');sources.joinpath(name).write_bytes(raw);manifest.append({'path':name,'sha256':sha(raw)})
    raw=boot.get(f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/LICENSE');sources.joinpath('LICENSE').write_bytes(raw)
    return selected,{'upstream_commit':JEV,'source_files':manifest,'total_public':len(allrows),'selection':'one per tier/family, lowest fixed salted INPUT-ONLY hash; four authored integration examples appended','ids':[r['id'] for r in selected]}

def worker(inp,out):
    setup_source()
    from shared_system.contracts import Task
    from shared_system.core import LocalCore,first_pass
    core=LocalCore(model_identity='Qwen3.5-4B-Q4_K_M:'+PIN['weights']['sha256']);rows=[]
    for row in json.loads(Path(inp).read_text()):
        if set(row)!={'id','task'}:raise ValueError('worker_input_labels')
        task=Task.create(row['task']);arms=[False,True] if int(sha(row['id'].encode())[:8],16)%2 else [True,False]
        for planned in arms:
            item=first_pass(task,core,planned=planned);item['id']=row['id'];rows.append(item)
            Path(out).write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
            print('ATTEMPT '+json.dumps({'id':row['id'],'arm':item['source'],'issues':item['issues'],'seconds':item['call']['seconds']}),flush=True)

def preflight():
    root,identity=setup_source();proc=log=None;obj={'status':'failed','source_bundle_sha256':identity,'rows':[]}
    try:
        from shared_system.contracts import Task,selected
        from shared_system.core import LocalCore,first_pass
        proc,log,info=boot.setup(PIN);obj['runtime']=info
        cases=[({'state':'Nineteen plus twenty-three.','question':{'type':'choice','instructions':'Which value is the sum?','criteria':{'42':'42','43':'43'}},'labels':['42','43']},'42'),({'state':'Please cancel order 27.','question':{'type':'choice','instructions':'Which action is requested?','criteria':{'track':'Locate a shipment','cancel':'Cancel the order'}},'labels':['track','cancel']},'cancel')]
        core=LocalCore(model_identity='Qwen3.5-4B-Q4_K_M:'+PIN['weights']['sha256'])
        for value,expected in cases:
            task=Task.create(value)
            for planned in (False,True):
                row=first_pass(task,core,planned=planned);row.update(task=value,expected=expected,correct=selected(row['probabilities'],task.labels)==expected and not row['issues']);obj['rows'].append(row)
        obj['ready']=all(r['correct'] for r in obj['rows']);obj['status']='completed'
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('ready='+str(obj['ready']).lower()+'\n')
    except Exception as error:obj['error']=f'{type(error).__name__}: {error}'
    finally:
        stop(proc,log);save('preflight-shared.json',obj)
        with zipfile.ZipFile('shared-source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in root.rglob('*.py'):z.write(p,str(p.relative_to(root)))
            for name in ('source_bundle.json','remote_stage1.py'):z.write(HERE/name,'research/'+name)
    if not obj.get('ready'):raise SystemExit(1)

def run(shard):
    root,identity=setup_source();proc=log=None;started=time.perf_counter()
    obj={'status':'failed','shard':shard,'source_bundle_sha256':identity,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA')}
    try:
        selected,manifest=acquire();rows=[r for i,r in enumerate(selected) if i%6==shard]
        obj['manifest']=manifest;obj['protocol']={'direct_completion_cap':1280,'plan_completion_cap':768,'reserved_final_cap':512,'max_calls_per_deployed_route':2,'temperature':0,'native_thinking':False,'seed':1707,'persistent_memory_scored':False,'weight_training':False,'paid_model_api':False,'frontier_comparator':False,'same_initial_plan_shared_across_followup_arms':True,'public_pilot_not_official_composite':True,'probabilities':'verbalized, uncalibrated'}
        print('PROTOCOL_LOCK '+json.dumps(obj),flush=True)
        tmp=Path('/tmp/shared-pilot');tmp.mkdir(exist_ok=True);inp=tmp/'input.json';out=tmp/'output.json'
        inp.write_text(json.dumps([{'id':r['id'],'task':{k:r[k] for k in ('state','question','labels')}} for r in rows]))
        proc,log,info=boot.setup(PIN);obj['runtime']=info
        env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
        try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=14*60).returncode
        except subprocess.TimeoutExpired:exitcode='timeout'
        answers=json.loads(out.read_text()) if out.exists() else []
        obj.update(status='completed',worker_exit=exitcode,answers=answers,cases=rows)
    except Exception as error:obj['error']=f'{type(error).__name__}: {error}'
    finally:
        stop(proc,log);obj['wall_seconds']=time.perf_counter()-started;save(f'stage1-{shard:02d}.json',obj)
        with zipfile.ZipFile(f'sources-{shard:02d}.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in Path('sources').glob('*'):z.write(p,'sources/'+p.name)
    if obj['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.preflight:preflight()
    elif a.worker:worker(a.input,a.output)
    else:
        if not 0<=a.shard<6:raise ValueError('shard')
        run(a.shard)
