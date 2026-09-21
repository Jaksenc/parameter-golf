"""Availability-only recovery; never edits the frozen scientific solver or answers.
Use only after the original run is terminal and all obtainable artifacts are saved.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import countercase_v8 as c

SOURCE='e50085c81381f226a632e06efbb7d740f276acd013a3931e0c18343d15a55ffd'
RECOVERY_SHARDS=32


def prepare(root: Path) -> dict:
    prepared=root/'countercase-prepared'
    m=json.loads((prepared/'manifest.json').read_text())
    plans=json.loads((prepared/'jobs.json').read_text())
    if m['source_sha256']!=SOURCE or c.prior.h5.filehash(c.__file__)!=SOURCE or c.prior.h5.digest(plans)!=m['jobs_hash']:
        raise ValueError('Frozen method or input changed')
    found={};audits=[]
    wanted={j['input']['id']:j for group in plans for j in group}
    for shard,plan in enumerate(plans):
        d=root/'original-records'/f'countercase-v8-shard-{shard}';p=d/'records.jsonl'
        expected={j['input']['id']:j for j in plan};ids=[];tail=0
        if p.exists():
            receipt=json.loads((d/'preflight.json').read_text())
            if receipt['source_sha256']!=SOURCE:raise ValueError('Source mismatch')
            for line in p.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):
                    tail+=len(line);continue
                r=json.loads(line);rid=r['id']
                if rid in found or rid not in expected or r['input_sha256']!=c.prior.h5.digest(expected[rid]['input']):raise ValueError('Duplicate/unknown/modified input')
                if set(r['outputs'])!=set(expected[rid]['arms']):raise ValueError('Incomplete matched record')
                for arm,obs in r['outputs'].items():
                    if obs['arm']!=arm or obs['answer'] not in expected[rid]['input']['labels'] or obs['input_sha256']!=r['input_sha256']:raise ValueError('Bad arm record')
                found[rid]=r;ids.append(rid)
        comp=d/'complete.json'
        if comp.exists():
            x=json.loads(comp.read_text())
            if set(ids)!=set(expected) or x['count']!=len(plan) or x['source_sha256']!=SOURCE or x['records_sha256']!=c.prior.h5.filehash(p):raise ValueError('Completion mismatch')
        audits.append({'shard':shard,'planned':len(plan),'complete_rows':len(ids),'completion_receipt':comp.exists(),
                       'trailing_uncommitted_bytes':tail,'record_sha256':c.prior.h5.filehash(p) if p.exists() else None})
    missing=[wanted[x] for x in sorted(set(wanted)-set(found))]
    if len(wanted)!=295 or len(missing)>295 or not found:raise ValueError('Unexpected recovery population or no surviving records')
    bins=[[] for _ in range(RECOVERY_SHARDS)];loads=[0]*RECOVERY_SHARDS
    for job in sorted(missing,key=lambda j:(-(len(json.dumps(j['input']))+3000)*len(j['arms']),j['input']['id'])):
        i=min(range(RECOVERY_SHARDS),key=lambda j:(loads[j],j))
        bins[i].append(job);loads[i]+=(len(json.dumps(job['input']))+3000)*len(job['arms'])
    out=root/'recovery-prepared';out.mkdir(exist_ok=False)
    original=[found[x] for x in sorted(found)]
    c.prior.h5.write(out/'original.json',original);c.prior.h5.write(out/'jobs.json',bins)
    receipt={'scientific_source_sha256':SOURCE,'original_count':len(original),'missing_count':len(missing),
             'original_hash':c.prior.h5.digest(original),'jobs_hash':c.prior.h5.digest(bins),'shard_audits':audits,
             'selection':'Only missing complete matched records; no correctness-based selection. Run after upstream termination.',
             'operational_note':'The initial 128-missing guard was widened to the already authorized 295-input population after the first wave timed out. Inference source, prompt, token cap, arms and selection remain unchanged.'}
    c.prior.h5.write(out/'receipt.json',receipt)
    print(json.dumps({'preserved':len(original),'missing':len(missing),'counts':list(map(len,bins))}),flush=True)
    return receipt


def run(root: Path,out: Path,shard:int) -> None:
    if shard not in range(RECOVERY_SHARDS):raise ValueError('Shard out of range')
    jobs=json.loads((root/'recovery-prepared/jobs.json').read_text())
    receipt=json.loads((root/'recovery-prepared/receipt.json').read_text())
    if c.prior.h5.filehash(c.__file__)!=SOURCE or c.prior.h5.digest(jobs)!=receipt['jobs_hash']:raise ValueError('Source/jobs mismatch')
    out.mkdir(parents=True,exist_ok=False);path=out/'records.jsonl';path.write_text('')
    ids=[]
    if jobs[shard]:
        from reconstruct_v1 import Runtime
        rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
        anchor=json.loads((root/'countercase-prepared/anchors.json').read_text())[shard]
        measured,_=rt.score(anchor['input'])
        err=max(abs(a-b) for a,b in zip(measured['logits'],anchor['native']['logits']))
        if err>1e-4 or measured['prompt_hash']!=anchor['native']['prompt_hash']:raise ValueError('Native anchor mismatch')
        c.prior.h5.write(out/'preflight.json',{'runtime':rt.receipt,'fixture':fixture,'source_sha256':SOURCE,
                    'explicitly_generative':True,'anchor_id':anchor['input']['id'],'anchor_error':err})
        for job in jobs[shard]:
            arms=list(job['arms']);c.random.Random(int(c.prior.h5.digest(job['input']['id'])[:8],16)).shuffle(arms)
            outputs={arm:c.solve(rt,job['input'],arm) for arm in arms}
            r={'id':job['input']['id'],'input_sha256':c.prior.h5.digest(job['input']),'outputs':outputs,
               'execution_order':arms,'shard':shard,'recovered':True}
            with path.open('a') as stream:
                stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush();c.os.fsync(stream.fileno())
            ids.append(r['id']);print(json.dumps({'shard':shard,'recovered':len(ids),'planned':len(jobs[shard])}),flush=True)
    c.prior.h5.write(out/'complete.json',{'ids':ids,'count':len(ids),'shard':shard,'source_sha256':SOURCE,
                'jobs_hash':c.prior.h5.digest(jobs[shard]),'records_sha256':c.prior.h5.filehash(path)})


def aggregate(root:Path) -> None:
    receipt=json.loads((root/'recovery-prepared/receipt.json').read_text())
    original=json.loads((root/'recovery-prepared/original.json').read_text())
    jobs=json.loads((root/'recovery-prepared/jobs.json').read_text())
    if c.prior.h5.digest(original)!=receipt['original_hash'] or c.prior.h5.digest(jobs)!=receipt['jobs_hash']:raise ValueError('Recovery data changed')
    allplans=json.loads((root/'countercase-prepared/jobs.json').read_text())
    wanted={j['input']['id']:j for group in allplans for j in group}
    rows=list(original);seen={r['id'] for r in rows}
    for i,plan in enumerate(jobs):
        d=root/'recovered-records'/f'countercase-v8-recovered-{i}';p=d/'records.jsonl'
        comp=json.loads((d/'complete.json').read_text())
        if comp['source_sha256']!=SOURCE or comp['jobs_hash']!=c.prior.h5.digest(plan) or comp['records_sha256']!=c.prior.h5.filehash(p):raise ValueError('Recovery receipt mismatch')
        rr=[json.loads(line) for line in p.read_text().splitlines()];expected={j['input']['id'] for j in plan}
        if len(rr)!=len(expected) or {r['id'] for r in rr}!=expected or expected&seen or comp['count']!=len(rr):raise ValueError('Missing/duplicated recovery')
        for r in rr:
            job=wanted[r['id']]
            if r['input_sha256']!=c.prior.h5.digest(job['input']) or set(r['outputs'])!=set(job['arms']):raise ValueError('Wrong recovered record')
        seen|=expected;rows+=rr
    if len(rows)!=295 or seen!=set(wanted):raise ValueError('Incomplete final population')
    c.prior.h5.write(root/'all_records.json',sorted(rows,key=lambda r:r['id']))
    c.prior.h5.write(root/'completion.json',{'complete':True,'unique_inputs':len(rows),'preserved':len(original),
         'recovered':len(rows)-len(original),'source_sha256':SOURCE,'recovery':True,
         'generated_responses':sum(len(r['outputs']) for r in rows),'original_rows_unchanged':all(r in rows for r in original)})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--root',default='.')
    p.add_argument('--out',default='recovered');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    root=Path(a.root)
    if a.mode=='prepare':prepare(root)
    elif a.mode=='run':run(root,Path(a.out),a.shard)
    else:aggregate(root)
if __name__=='__main__':main()
