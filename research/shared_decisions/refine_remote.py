"""Staged intervention comparison over frozen first-pass outputs.

Only the small observation manifest is new: operations are executed by the
unchanged specialist locally, and checked against source/task/operation hashes.
First-pass outputs are never regenerated. Reference labels never reach workers.
This study pays the full logical cost for a reused identical model readout.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,importlib.util,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import remote_stage1 as original
import entry_v2
CORE='946de0d3aa1ae65a547fe0e6efa700ac267d9ffc3636d38bf739d39b6006e3c9'
PILOT='da56ebdc0868872d931f3409c68db006bd1de5b2'
SHARDS=6
HERE=Path(__file__).parent

def source():
    root,identity=original.setup_source()
    if original.sha((root/'shared_system/core.py').read_bytes())!=CORE:
        raise ValueError('evaluated_core_changed')
    if original.sha((HERE/'comparison.py').read_bytes())!='ef6aa78fb91820f058852fdccecbf3aa1b9e4316d7f70dd5d7c55fbd9deb3b18':
        raise ValueError('comparison_source_changed')
    spec=importlib.util.spec_from_file_location('shared_system.comparison',HERE/'comparison.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return root,identity,module

def load_objects(readouts):
    objects={}
    def add(name,raw):
        if name in objects:raise ValueError('duplicate_record:'+name)
        objects[name]=(json.loads(raw),hashlib.sha256(raw).hexdigest())
    for p in sorted(readouts.glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.startswith('full-stage1-') and n.endswith('.json') and '/' not in n:
                    add(n,z.read(n))
    for p in sorted(readouts.glob('full-stage1-*.json')):add(p.name,p.read_bytes())
    for i in range(6):
        name=f'v2-stage1-{i:02d}.json'
        raw=original.boot.get(f'https://raw.githubusercontent.com/Jaksenc/parameter-golf/{PILOT}/research/shared_decisions/results/{name}')
        add(name,raw)
    return objects

def prepare(readouts,bundle):
    _,_,_=source()
    from shared_system.contracts import Task,digest
    from shared_system.core import from_first
    raw=gzip.decompress(base64.b64decode(bundle['gzip_base64'],validate=True))
    if hashlib.sha256(raw).hexdigest()!=bundle['decoded_sha256']:raise ValueError('observation_manifest_integrity')
    entries=json.loads(raw)
    if not isinstance(entries,list) or len(entries)>235:raise ValueError('observation_entry_count')
    records=load_objects(readouts);packets=[];ids=set()
    for e in entries:
        if set(e)!={'id','observations','source_receipt'}:raise ValueError('entry_schema')
        key=e['id']
        if key in ids:raise ValueError('duplicate_observation')
        ids.add(key)
        d,h=records[e['source_receipt']['record']]
        if h!=e['source_receipt']['sha256']:raise ValueError('source_record_changed')
        cases=[c for c in d['cases'] if c['id']==key]
        answers=[a for a in d['answers'] if a['id']==key and a['source']=='planned']
        if len(cases)!=1 or len(answers)!=1:raise ValueError('source_ambiguity')
        task=Task.create({k:cases[0][k] for k in ('state','question','labels')})
        first=answers[0]
        for mode in ('rules','shared'):
            w=from_first(task,first)
            if w.candidate is None or w.operation is None:raise ValueError('no_qualified_proposal')
            w.integrate(e['observations'][mode])
        packets.append({'id':key,'task':task.value,'first':first,'observations':e['observations'],'source_receipt':e['source_receipt']})
    return sorted(packets,key=lambda r:r['id'])

def worker(inp,out):
    _,_,comparison=source()
    from shared_system.contracts import Task,canonical
    from shared_system.core import LocalCore
    core=LocalCore(model_identity='Qwen3.5-4B-Q4_K_M:'+original.PIN['weights']['sha256'])
    rows=[]
    for p in json.loads(Path(inp).read_text()):
        if set(p)!={'id','task','first','observations','source_receipt'}:raise ValueError('worker_input_schema')
        task=Task.create(p['task'])
        result=comparison.compare(task,p['first'],core,p['observations'],case_id=p['id'])
        result.update(id=p['id'],input_sha256=hashlib.sha256(canonical(p).encode()).hexdigest(),source_receipt=p['source_receipt'])
        rows.append(result)
        Path(out).write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
        print('REFINEMENT '+json.dumps({'id':p['id'],'new_calls':len(result['actual_calls']),'arms':{a:r['result']['status'] for a,r in result['arms'].items()}}),flush=True)

def run(shard,readouts):
    proc=log=None;started=time.perf_counter();rows=[]
    obj={'status':'failed','shard':shard,'shards':SHARDS,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        bundle=json.loads((HERE/'observation_bundle.json').read_text())
        packets=prepare(readouts,bundle);chosen=[r for i,r in enumerate(packets) if i%SHARDS==shard]
        obj['protocol']={'selected_ids':[r['id'] for r in packets],'manifest_sha256':bundle['decoded_sha256'],'core_sha256':CORE,'comparison_sha256':hashlib.sha256((HERE/'comparison.py').read_bytes()).hexdigest(),'max_final_tokens':512,'memory':False,'weights_updated':False,'reference_answers_solver_visible':False,'arms':['review_if_operation','rules','shared'],'identical_evidence_reuse':'one actual readout, full logical cost retained in each arm','cost_usd':None,'energy_joules':None,'formal_check_is_not_semantic_proof':True,'stage1_not_regenerated':True}
        print('LOCK '+json.dumps(obj),flush=True)
        if not chosen:
            obj.update(status='completed',worker_exit=0,rows=[],planned_ids=[])
        else:
            tmp=Path('/tmp/shared-refinement');tmp.mkdir(exist_ok=True);inp=tmp/'input.json';out=tmp/'output.json'
            inp.write_text(json.dumps(chosen,ensure_ascii=False,allow_nan=False));obj['planned_ids']=[r['id'] for r in chosen]
            proc,log,info=original.boot.setup(original.PIN);obj['runtime']=info
            env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
            try:code=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=1200).returncode
            except subprocess.TimeoutExpired:code='timeout'
            rows=json.loads(out.read_text()) if out.exists() else []
            obj.update(status='completed',worker_exit=code,rows=rows)
    except Exception as e:obj['error']=f'{type(e).__name__}: {e}'
    finally:
        original.stop(proc,log);obj['wall_seconds']=time.perf_counter()-started
        raw=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False).encode()
        name=f'refinement-{shard:02d}.json';Path(name).write_bytes(raw)
        encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
        print('RECEIPT '+json.dumps({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}),flush=True)
        for i in range(0,len(encoded),3000):print(f'BACKUP_{i//3000:03d} '+encoded[i:i+3000],flush=True)
    if obj['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);p.add_argument('--readouts',type=Path,default=Path('downloaded_readouts'));a=p.parse_args()
    if a.worker:worker(a.input,a.output)
    elif not 0<=a.shard<SHARDS:raise ValueError('shard')
    else:run(a.shard,a.readouts)
