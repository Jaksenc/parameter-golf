"""Resume frozen Evidence v4 on unrecorded inputs only; never select by outcome.
This is an execution repair, not a changed hypothesis or model configuration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

OLD_RUN = 35631351186
EFFECTIVE_SHA = 'db7e73bcea3b497213bfe252ad4f154d8f110f321bee23f9fc17f82518b89851'
SHARDS = 16


def read_json(path):
    return json.loads(Path(path).read_text())


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_experiment():
    import run_evidence_v4
    e = run_evidence_v4.experiment
    if e.sha(e.__file__) != EFFECTIVE_SHA:
        raise RuntimeError('Frozen experiment hash mismatch')
    if e.CAP != 160 or e.SEED != 927431:
        raise RuntimeError('Frozen configuration mismatch')
    return e


def validate_record(e, row, expected):
    rid = row['id']
    if rid not in expected or row['input_sha256'] != e.digest(e.inp(expected[rid])):
        raise RuntimeError('Input provenance mismatch: ' + rid)
    if row['partition'] != expected[rid]['partition']:
        raise RuntimeError('Partition mismatch: ' + rid)
    if set(row['predictions']) != {'native', 'compiler_claim', 'reasoning', 'evidence_exact'}:
        raise RuntimeError('Incomplete arms: ' + rid)
    if any(label not in expected[rid]['labels'] for label in row['predictions'].values()):
        raise RuntimeError('Invalid typed output: ' + rid)
    if row['execution'] != e.execute(expected[rid], row['compile']['text']):
        raise RuntimeError('Execution replay mismatch: ' + rid)


def prepare(root, old, out):
    e = load_experiment()
    root, old, out = Path(root), Path(old), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    expected = {r['id']: r for r in e.population(root)}
    if len(expected) != 295:
        raise RuntimeError('Incorrect frozen population')
    saved = {}
    original = []
    for shard in range(SHARDS):
        d = old / ('evidence-fixed-shard-' + str(shard))
        pre = read_json(d/'preflight.json')
        if pre['source'] != EFFECTIVE_SHA:
            raise RuntimeError('Wrong source in original shard')
        raw = (d/'records.jsonl').read_text()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if (d/'complete.json').exists():
            complete = read_json(d/'complete.json')
            if complete['count'] != len(rows) or complete['records_sha256'] != filehash(d/'records.jsonl'):
                raise RuntimeError('Corrupt original completion receipt')
        for row in rows:
            validate_record(e, row, expected)
            if row['id'] in saved:
                raise RuntimeError('Duplicate original input')
            saved[row['id']] = row
        original.append({'shard': shard, 'count': len(rows), 'completed_receipt': (d/'complete.json').exists(),
                         'records_sha256': filehash(d/'records.jsonl')})
    missing = [r for rid, r in expected.items() if rid not in saved]
    bins = [[] for _ in range(SHARDS)]
    loads = [0.0]*SHARDS
    for row in sorted(missing, key=lambda r: (-len(json.dumps(r['state'])), r['id'])):
        j = min(range(SHARDS), key=lambda j: (loads[j], j))
        bins[j].append(row)
        loads[j] += len(json.dumps(row['state'])) + 1800
    anchor_id = min(saved, key=lambda rid: (len(json.dumps(expected[rid]['state'])), rid))
    e.write(out/'saved_records.json', list(saved.values()))
    e.write(out/'assignments.json', bins)
    e.write(out/'anchor.json', {'input':expected[anchor_id], 'record':saved[anchor_id]})
    plan = {'old_run':OLD_RUN, 'effective_source_sha256':EFFECTIVE_SHA,
            'original':original, 'saved':len(saved), 'missing':len(missing),
            'missing_ids':[r['id'] for r in missing], 'population':len(expected),
            'population_sha256':e.digest(list(expected.values())),
            'assignments_sha256':e.digest(bins), 'saved_records_sha256':filehash(out/'saved_records.json'),
            'selection':'All and only unrecorded input IDs; no outcome-dependent reruns',
            'recovery_source_sha256':filehash(__file__)}
    e.write(out/'plan.json', plan)
    matrix = {'include':[{'shard':i} for i,rows in enumerate(bins) if rows]}
    print(json.dumps({'saved':len(saved),'missing':len(missing),'counts':[len(x) for x in bins]}), flush=True)
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write('matrix='+json.dumps(matrix)+'\n')
            f.write('missing='+str(len(missing))+'\n')


def run(root, plan, out, shard):
    import reconstruct_v1 as v1
    e = load_experiment()
    plan, out = Path(plan), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = read_json(plan/'plan.json')
    bins = read_json(plan/'assignments.json')
    if e.digest(bins) != receipt['assignments_sha256'] or filehash(__file__) != receipt['recovery_source_sha256']:
        raise RuntimeError('Recovery plan/source changed')
    rows = bins[shard]
    rt = v1.Runtime(root)
    fixture = rt.check()
    anchor = read_json(plan/'anchor.json')
    actual, _ = rt.score(anchor['input'])
    ref = anchor['record']['native']
    err = max(abs(a-b) for a,b in zip(actual['logits'], ref['logits']))
    if err > 2e-4 or actual['prompt_hash'] != ref['prompt_hash']:
        raise RuntimeError('Cross-run native anchor mismatch')
    e.write(out/'preflight.json', {'runtime':rt.receipt,'fixture':fixture,'source':EFFECTIVE_SHA,
                                 'native_anchor':{'id':anchor['input']['id'],'max_logit_error':err},
                                 'generation_enabled':True,'cap':e.CAP})
    records = []
    for i,r in enumerate(rows):
        native,_ = rt.score(r)
        baseline = r['labels'][max(range(len(r['labels'])), key=lambda j:native['logits'][j])]
        compiled = e.generate(rt,r,'compile')
        reason = e.generate(rt,r,'reason')
        executed = e.execute(r,compiled['text'])
        claim = e.claimed(r,compiled['text'],'CLAIM')
        reasoning = e.claimed(r,reason['text'],'FINAL')
        row = {'id':r['id'],'partition':r['partition'],'input_sha256':e.digest(e.inp(r)),
               'native':native,'compile':compiled,'reason':reason,'execution':executed,
               'predictions':{'native':baseline,'compiler_claim':claim or baseline,
                              'reasoning':reasoning or baseline,'evidence_exact':executed.get('label',baseline)},
               'valid_claim':claim is not None,'valid_reasoning':reasoning is not None}
        validate_record(e,row,{r['id']:r})
        records.append(row)
        with (out/'records.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps({'shard':shard,'done':i+1,'total':len(rows),'id':r['id']}),flush=True)
    e.write(out/'complete.json', {'shard':shard,'count':len(rows), 'ids':[r['id'] for r in rows],
                                'source_sha256':EFFECTIVE_SHA,'recovery_source_sha256':filehash(__file__),
                                'records_sha256':filehash(out/'records.jsonl')})


def aggregate(root, plan, new, out):
    e = load_experiment()
    plan,new,out = Path(plan),Path(new),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    expected = {r['id']:r for r in e.population(root)}
    saved = read_json(plan/'saved_records.json')
    receipt = read_json(plan/'plan.json')
    if filehash(plan/'saved_records.json') != receipt['saved_records_sha256']:
        raise RuntimeError('Saved records changed')
    bins = read_json(plan/'assignments.json')
    rows = list(saved)
    for i,items in enumerate(bins):
        if not items:continue
        d = new/f'evidence-recovered-{i}'
        comp = read_json(d/'complete.json')
        if comp['source_sha256'] != EFFECTIVE_SHA or comp['ids'] != [r['id'] for r in items]:
            raise RuntimeError('Wrong recovered population')
        if comp['records_sha256'] != filehash(d/'records.jsonl'):
            raise RuntimeError('Recovered bytes changed')
        part = [json.loads(line) for line in (d/'records.jsonl').read_text().splitlines()]
        if comp['count'] != len(part):raise RuntimeError('Incomplete recovered shard')
        rows.extend(part)
    ids = [r['id'] for r in rows]
    if len(ids) != 295 or len(set(ids)) != 295 or set(ids) != set(expected):
        raise RuntimeError('Evaluation population incomplete')
    for row in rows:validate_record(e,row,expected)
    rows.sort(key=lambda r:r['id'])
    e.write(out/'all_records.json',rows)
    e.write(out/'completion.json',{'complete':True,'inputs':295,'generations':590,'newly_recovered':receipt['missing'],
                                  'preserved_original':receipt['saved'],'effective_source_sha256':EFFECTIVE_SHA,
                                  'records_sha256':filehash(out/'all_records.json'),'recovery_source_sha256':filehash(__file__)})
    print(json.dumps({'complete':True,'inputs':295,'preserved':receipt['saved'],'recovered':receipt['missing']}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run','aggregate'])
    p.add_argument('--root',default='reconstruction-inputs')
    p.add_argument('--old',default='original-records')
    p.add_argument('--plan',default='recovery-plan')
    p.add_argument('--new',default='recovered-records')
    p.add_argument('--out',default='completed')
    p.add_argument('--shard',type=int,default=0)
    a = p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.old,a.plan)
    elif a.mode=='run':run(a.root,a.plan,a.out,a.shard)
    else:aggregate(a.root,a.plan,a.new,a.out)

if __name__=='__main__':main()
