"""Availability-only recovery of original worker 19. Scientific inference unchanged."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import probability_v13 as p
SOURCE='147e7610dec39eef889ef8938c4818e32cc008c45200b142b4b4b0b6aa7f091d'
SHARD=19
WORKERS=6

def assigned(root,worker):
    m=json.loads((root/'probability-prepared/manifest.json').read_text())
    jobs=json.loads((root/'probability-prepared/jobs.json').read_text())
    if p.filehash(p.__file__)!=SOURCE or m['source_sha256']!=SOURCE or p.digest(jobs)!=m['jobs_hash']:
        raise ValueError('Frozen scientific source or jobs changed')
    if worker not in range(WORKERS) or len(jobs[SHARD])!=18:
        raise ValueError('Recovery population changed')
    return jobs[SHARD][worker::WORKERS]

def run(root,out,worker):
    from reconstruct_v1 import Runtime
    root,out=Path(root),Path(out)
    plan=assigned(root,worker)
    missing=root/'failed-worker/records.jsonl'
    if missing.exists() and missing.read_text().strip():
        raise ValueError('Original worker has records: refuse duplicate recovery')
    rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
    anchor=json.loads((root/'probability-prepared/anchors.json').read_text())[SHARD]
    obs,_=rt.score(anchor['input'])
    error=max(abs(a-b) for a,b in zip(obs['logits'],anchor['native']['logits']))
    out.mkdir(parents=True,exist_ok=False)
    # Persist observations even if this check fails again; never relax its tolerance.
    p.write(out/'preflight.json',{'runtime':rt.receipt,'actually_generative':True,
        'fixture':fixture,'anchor_id':anchor['input']['id'],'anchor_error':error,
        'anchor_observation':obs,'anchor_reference':anchor['native'],
        'source_sha256':SOURCE,'original_shard':SHARD,'recovery_worker':worker})
    if error>1e-4 or obs['prompt_hash']!=anchor['native']['prompt_hash']:
        raise ValueError('Unchanged native anchor check failed again')
    path=out/'records.jsonl';path.write_text('');ids=[]
    for job in plan:
        row=p.solve(rt,job['input'],job['trace'])
        row['shard']=SHARD;row['recovery_worker']=worker
        with path.open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        ids.append(row['id']);print(json.dumps({'worker':worker,'done':len(ids),'total':len(plan)}),flush=True)
    p.write(out/'complete.json',{'ids':ids,'count':len(ids),'source_sha256':SOURCE,
        'records_sha256':p.filehash(path),'jobs_hash':p.digest(plan),
        'original_shard':SHARD,'recovery_worker':worker})

def aggregate(root):
    root=Path(root)
    jobs=json.loads((root/'probability-prepared/jobs.json').read_text())
    rows=[];seen=set();original_count=0
    failure=root/'records/probability-v13-shard-19/records.jsonl'
    if failure.exists() and failure.read_text().strip():
        raise ValueError('Cannot replace complete original observations')
    sources=[]
    for shard,plan in enumerate(jobs):
        if shard==SHARD:continue
        sources.append((root/'records'/f'probability-v13-shard-{shard}',plan,'original'))
    for worker in range(WORKERS):
        sources.append((root/'recovered-records'/f'probability-v13-recovered-{worker}',assigned(root,worker),'recovered'))
    for folder,plan,origin in sources:
        c=json.loads((folder/'complete.json').read_text());path=folder/'records.jsonl'
        rr=[json.loads(s) for s in path.read_text().splitlines()]
        wanted={j['input']['id']:j for j in plan}
        if c['source_sha256']!=SOURCE or c['records_sha256']!=p.filehash(path) or c['jobs_hash']!=p.digest(plan):
            raise ValueError('Record provenance')
        if len(rr)!=len(plan) or c['count']!=len(rr) or set(c['ids'])!=set(wanted):
            raise ValueError('Missing original or recovered records')
        for r in rr:
            if r['id'] in seen or r['id'] not in wanted or r['input_hash']!=p.digest(wanted[r['id']]['input']):
                raise ValueError('Unexpected/changed input')
            if set(r['readouts'])!=set(p.READOUTS):raise ValueError('Missing probability branch')
            seen.add(r['id']);rows.append(r)
        if origin=='original':original_count+=len(rr)
    if len(rows)!=441 or original_count!=423:raise ValueError('Incomplete population')
    p.write(root/'all_records.json',sorted(rows,key=lambda r:r['id']))
    p.write(root/'completion.json',{'complete':True,'inputs':441,'original_preserved':423,
        'recovered_inputs':18,'new_generations':sum(not r['trace_reused'] for r in rows),
        'new_readouts':sum(len(r['readouts']) for r in rows),'source_sha256':SOURCE})
    p.write(root/'RECOVERY.json',{'failed_original_shard':19,'original_records_in_failed_shard':0,
        'recovery_workers':6,'preserved':423,'recovered':18,
        'selection':'Missing worker inputs only; no labels, scores or uncertainty used.',
        'original_check':'Native anchor mismatch before any records. Failing values were not saved by original code.',
        'diagnostic':'Separate run 35780530571 repeated that anchor three times with identical archived logits and prompt hash.',
        'scientific_source_changed':False,'tolerance_changed':False,
        'interpretation':'The initial mismatch did not reproduce; its cause and size are unknown.'})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['run','aggregate'])
    a.add_argument('--root',default='.');a.add_argument('--out',default='recovered');a.add_argument('--worker',type=int,default=0)
    x=a.parse_args()
    if x.mode=='run':run(Path(x.root),Path(x.out),x.worker)
    else:aggregate(Path(x.root))
