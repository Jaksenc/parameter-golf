"""Complete the frozen bounded-operation comparison from saved readouts.
No first pass is regenerated. Only public inputs reach model workers.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,os,re,subprocess,sys,time,zipfile
from pathlib import Path
import remote_stage1 as original
import entry_v2
import refine_remote as frozen
import full_stage1
HERE=Path(__file__).parent
SHARDS=8

def sha(b): return hashlib.sha256(b).hexdigest()

def objects(root):
    result={}
    def add(name,raw):
        if not re.fullmatch(r'(?:full-stage1|recovered-readouts)-\d\d\.json',name):return
        if name in result and result[name][1]!=sha(raw):raise ValueError('conflicting_source:'+name)
        result[name]=(json.loads(raw),sha(raw))
    def archive(data,depth=0):
        if depth>2:raise ValueError('archive_depth')
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist())>80_000_000:raise ValueError('archive_size')
            for name in z.namelist():
                base=Path(name).name
                if base.startswith('Adaptive-JevBench-') and base.endswith('.zip'):archive(z.read(name),depth+1)
                elif re.fullmatch(r'(?:full-stage1|recovered-readouts)-\d\d\.json',base):add(base,z.read(name))
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        if path.suffix=='.json':add(path.name,path.read_bytes())
        elif path.name.startswith('Adaptive-JevBench-') and path.suffix=='.zip':archive(path.read_bytes())
    for i in range(6):
        name=f'v2-stage1-{i:02d}.json'
        raw=original.boot.get(f'https://raw.githubusercontent.com/Jaksenc/parameter-golf/{frozen.PILOT}/research/shared_decisions/results/{name}')
        result[name]=(json.loads(raw),sha(raw))
    return result

def prepare(root):
    frozen.source()
    bundle=json.loads((HERE/'final_observations.json').read_text())
    raw=gzip.decompress(base64.b64decode(bundle['data'],validate=True))
    if sha(raw)!=bundle['decoded_sha256']:raise ValueError('observation_manifest_hash')
    entries=json.loads(raw)
    if not isinstance(entries,list) or not 1<=len(entries)<=235:raise ValueError('entry_count')
    full_stage1.acquire()
    tasks={}
    for tier,h in full_stage1.SHA.items():
        rawdata=Path('sources',tier+'.jsonl').read_bytes()
        if sha(rawdata)!=h:raise ValueError('task_source_hash')
        for line in rawdata.splitlines():
            r=json.loads(line);tasks[r['id']]={k:r[k] for k in ('state','question','labels')}
    from shared_system.contracts import Task,digest
    from shared_system.core import from_first
    records=objects(root);packets=[];seen=set()
    for e in entries:
        if set(e)!={'id','observations','source_receipt'} or e['id'] in seen:raise ValueError('observation_schema')
        seen.add(e['id']);ref=e['source_receipt'];record,h=records[Path(ref['path']).name]
        if h!=ref['sha256']:raise ValueError('readout_record_hash')
        matches=[r for r in record.get('answers',[]) if r['id']==e['id'] and r['source']=='planned']
        if len(matches)!=1:raise ValueError('readout_unique')
        task=Task.create(tasks[e['id']]);first=matches[0]
        for mode in ('rules','shared'):
            w=from_first(task,first)
            if w.candidate is None or w.operation is None:raise ValueError('unqualified_proposal')
            o=e['observations'][mode]
            if o['operation']!=w.operation.value:raise ValueError('operation_mismatch')
            w.integrate(o)
        packets.append({'id':e['id'],'task':task.value,'first':first,'observations':e['observations'],'source_receipt':ref})
    return sorted(packets,key=lambda p:p['id']),bundle

def run(shard,root):
    proc=log=None;start=time.perf_counter();result={'status':'failed','shard':shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'driver_sha256':sha(Path(__file__).read_bytes())}
    try:
        packets,bundle=prepare(root);chosen=[p for i,p in enumerate(packets) if i%SHARDS==shard]
        result['protocol']={'all_selected_ids':[p['id'] for p in packets],'manifest_sha256':bundle['decoded_sha256'],'core_sha256':frozen.CORE,'comparison_sha256':sha((HERE/'comparison.py').read_bytes()),'final_token_cap':512,'model_pin':original.PIN,'memory':False,'new_weight_training':False,'original_candidates_regenerated':False,'scorer_targets_in_worker':False,'same_operation_evidence_can_reuse_readout':'Full logical readout cost retained for both equivalent arms; not two independent draws','no_paid_api':True,'official_composite':False}
        result['packets']=chosen
        print('PROTOCOL_LOCK '+json.dumps({k:v for k,v in result.items() if k!='packets'}),flush=True)
        if chosen:
            tmp=Path('/tmp/system1-final');tmp.mkdir(exist_ok=True);inp=tmp/'packets.json';out=tmp/'answers.json';inp.write_text(json.dumps(chosen,ensure_ascii=False,allow_nan=False))
            proc,log,info=original.boot.setup(original.PIN);result['runtime']=info
            env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
            try:code=subprocess.run([sys.executable,str(HERE/'refine_remote.py'),'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=1000).returncode
            except subprocess.TimeoutExpired:code='worker_timeout'
            result.update(status='completed',worker_exit=code,rows=json.loads(out.read_text()) if out.exists() else [])
        else:result.update(status='completed',worker_exit=0,rows=[])
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        original.stop(proc,log);result['wall_seconds']=time.perf_counter()-start
        full_stage1.emit(f'final-comparison-{shard:02d}.json',result)
        if shard==0:
            with zipfile.ZipFile('final-comparison-source.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in HERE.glob('*'):
                    if p.suffix in ('.py','.json') and p.is_file():z.write(p,'research/shared_decisions/'+p.name)
                for p in Path('/tmp/adaptive-shared-core').rglob('*.py'):z.write(p,str(p.relative_to('/tmp/adaptive-shared-core')))
                for p in Path('sources').glob('*'):
                    if p.is_file():z.write(p,'upstream/'+p.name)
            # Published competitor results are evidence only, fetched after solver termination.
            try:
                data=original.boot.get(f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{original.JEV}/results/v1.2/jevbench-v1.2-per-task.json')
                if len(data)>40_000_000:raise ValueError('comparison_size')
                Path('published-comparison.json.gz').write_bytes(gzip.compress(data,mtime=0))
                Path('published-comparison-receipt.json').write_text(json.dumps({'sha256':sha(data),'bytes':len(data),'upstream_commit':original.JEV,'file':'results/v1.2/jevbench-v1.2-per-task.json','new_inference':False}))
            except Exception as e:Path('published-comparison-receipt.json').write_text(json.dumps({'error':str(e)}))
    if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--readouts',type=Path,default=Path('readouts'));a=p.parse_args()
    if not 0<=a.shard<SHARDS:raise ValueError('shard')
    run(a.shard,a.readouts)
