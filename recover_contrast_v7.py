"""Availability-only recovery. Frozen scientific inference code is not modified."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import contrast_v7 as c

SOURCE = '52771fdb5d8c7e5315c6f49ac87b725ba1244c0fe5ba4f5a0674ec4de3778d0b'
RECOVERY_SHARDS = 16


def validated_rows(path: Path, wanted: dict, source: str):
    rows = []
    tail_bytes = 0
    if not path.exists():
        return rows, tail_bytes
    pre = json.loads((path.parent / 'preflight.json').read_text())
    if pre['source_sha256'] != source:
        raise ValueError('Different scientific source')
    raw = path.read_bytes()
    for i, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b'\n'):
            tail_bytes += len(line)
            continue
        row = json.loads(line)
        rid = row['id']
        if rid not in wanted or row['input_sha256'] != c.h5.digest(wanted[rid]['input']):
            raise ValueError('Unknown or modified inference input')
        if row['partition'] != wanted[rid]['partition']:
            raise ValueError('Partition mismatch')
        if set(row['predictions']) != set(c.ARMS):
            raise ValueError('Incomplete arm population')
        if any(x not in wanted[rid]['input']['labels'] for x in row['predictions'].values()):
            raise ValueError('Illegal output label')
        rows.append(row)
    return rows, tail_bytes


def prepare(root: Path):
    m = json.loads((root/'contrast-prepared/manifest.json').read_text())
    jobs = json.loads((root/'contrast-prepared/jobs.json').read_text())
    if c.h5.filehash(c.__file__) != SOURCE or m['source_sha256'] != SOURCE or c.h5.digest(jobs) != m['jobs_hash']:
        raise ValueError('Frozen source or jobs changed')
    wanted = {j['input']['id']: j for group in jobs for j in group}
    if len(wanted) != 295:
        raise ValueError('Wrong population')
    found = {}; audit = []
    for shard, plan in enumerate(jobs):
        directory = root/'original-records'/f'contrast-v7-shard-{shard}'
        path = directory/'records.jsonl'
        rows, tail = validated_rows(path, wanted, SOURCE)
        expected = {x['input']['id'] for x in plan}
        ids = {r['id'] for r in rows}
        if len(ids) != len(rows) or not ids <= expected or ids & found.keys():
            raise ValueError('Duplicate or wrong-shard records')
        completion = directory/'complete.json'
        if completion.exists():
            receipt = json.loads(completion.read_text())
            if receipt['count'] != len(plan) or ids != expected or receipt['records_sha256'] != c.h5.filehash(path):
                raise ValueError('Completion receipt mismatch')
        for row in rows:
            found[row['id']] = row
        audit.append({'shard':shard,'rows':len(rows),'planned':len(plan),'trailing_incomplete_bytes':tail,
                      'has_completion':completion.exists(),'records_sha256':c.h5.filehash(path) if path.exists() else None})
    missing = [wanted[k] for k in sorted(set(wanted)-set(found))]
    if len(missing) > 32:
        raise ValueError('Unexpectedly large recovery; stop for inspection')
    bins = [[] for _ in range(RECOVERY_SHARDS)]
    for i, row in enumerate(sorted(missing,key=lambda j:(-len(json.dumps(j['input'])),j['input']['id']))):
        bins[i % RECOVERY_SHARDS].append(row)
    target = root/'recovery-prepared';target.mkdir(exist_ok=True)
    c.h5.write(target/'original.json', [found[k] for k in sorted(found)])
    c.h5.write(target/'jobs.json',bins)
    c.h5.write(target/'receipt.json',{'source_sha256':SOURCE,'original_count':len(found),'missing_count':len(missing),
                'original_predictions_hash':c.h5.digest([found[k] for k in sorted(found)]),'jobs_hash':c.h5.digest(bins),
                'selection':'Missing complete records only; no labels or correctness inspected', 'shard_audit':audit})
    print(json.dumps({'preserved':len(found),'missing':len(missing),'recovery_shards':list(map(len,bins))}),flush=True)


def run(root: Path, out: Path, shard: int):
    from reconstruct_v1 import Runtime
    if not 0 <= shard < RECOVERY_SHARDS:
        raise ValueError('Invalid recovery shard')
    if c.h5.filehash(c.__file__) != SOURCE:
        raise ValueError('Scientific source changed')
    jobs = json.loads((root/'recovery-prepared/jobs.json').read_text())
    receipt = json.loads((root/'recovery-prepared/receipt.json').read_text())
    if c.h5.digest(jobs) != receipt['jobs_hash']:
        raise ValueError('Recovery jobs changed')
    out.mkdir(parents=True,exist_ok=False)
    path = out/'records.jsonl';path.write_text('')
    if jobs[shard]:
        rt = Runtime(root/'reconstruction-inputs'); fixture=rt.check()
        anchor=json.loads((root/'contrast-prepared/anchors.json').read_text())[shard]
        a,_=rt.score(anchor['input']);err=max(abs(x-y) for x,y in zip(a['logits'],anchor['native']['logits']))
        if err>1e-4 or a['prompt_hash']!=anchor['native']['prompt_hash']:
            raise ValueError('Anchor changed')
        c.h5.write(out/'preflight.json',{'source_sha256':SOURCE,'runtime':rt.receipt,'fixture':fixture,
                    'anchor_id':anchor['input']['id'],'anchor_error':err,'explicitly_generative':True})
        for job in jobs[shard]:
            row=c.solve(rt,job['input']);row['partition']=job['partition'];row['shard']=shard;row['recovered']=True
            with path.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            print(json.dumps({'recovered':row['id'],'shard':shard}),flush=True)
    c.h5.write(out/'complete.json',{'count':len(jobs[shard]),'ids':[x['input']['id'] for x in jobs[shard]],
                'records_sha256':c.h5.filehash(path),'source_sha256':SOURCE,'jobs_hash':c.h5.digest(jobs[shard])})


def aggregate(root: Path):
    jobs=json.loads((root/'recovery-prepared/jobs.json').read_text())
    original=json.loads((root/'recovery-prepared/original.json').read_text())
    receipt=json.loads((root/'recovery-prepared/receipt.json').read_text())
    if c.h5.digest(original)!=receipt['original_predictions_hash']:
        raise ValueError('Original rows changed')
    rows=list(original);seen={r['id'] for r in rows}
    for i,plan in enumerate(jobs):
        p=root/'recovered-records'/f'contrast-v7-recovered-{i}'
        comp=json.loads((p/'complete.json').read_text());path=p/'records.jsonl'
        if comp['source_sha256']!=SOURCE or comp['records_sha256']!=c.h5.filehash(path) or comp['jobs_hash']!=c.h5.digest(plan):
            raise ValueError('Recovery provenance failure')
        rr=[json.loads(s) for s in path.read_text().splitlines()]
        want={j['input']['id'] for j in plan}
        if len(rr)!=len(plan) or {r['id'] for r in rr}!=want or want&seen:
            raise ValueError('Missing or duplicated recovered rows')
        seen|=want;rows+=rr
    expected={j['input']['id']:j for group in json.loads((root/'contrast-prepared/jobs.json').read_text()) for j in group}
    if len(rows)!=295 or set(expected)!=seen:
        raise ValueError('Incomplete final experiment')
    for r in rows:
        if r['input_sha256']!=c.h5.digest(expected[r['id']]['input']):raise ValueError('Final input mismatch')
    c.h5.write(root/'all_contrast_records.json',sorted(rows,key=lambda r:r['id']))
    c.h5.write(root/'completion.json',{'complete':True,'total':len(rows),'preserved':len(original),
                'recovered':len(rows)-len(original),'source_sha256':SOURCE,
                'original_rows_unchanged':all(r in rows for r in original)})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--root',default='.')
    p.add_argument('--out',default='recovery-results');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    root=Path(a.root)
    if a.mode=='prepare':prepare(root)
    elif a.mode=='aggregate':aggregate(root)
    else:run(root,Path(a.out),a.shard)
if __name__=='__main__':main()
