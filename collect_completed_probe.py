"""Audit completed checkpoints only. The incomplete 4B run is not imputed."""
from __future__ import annotations
import base64,hashlib,json,math,statistics,struct,zlib
from collections import Counter
from audit_probe_v2 import api,parse,prob,choose,evaluate,emit
from decision_probe_v2 import synthetic,digest,public_input
JOBS={'gliclass-instruct':106099011148,'qwen-08b':106099011005}

def main():
    from benchmark import public_data
    rows=synthetic();public,sources=public_data(16)
    for r in public:r.update(split='regression-'+r['family'],base_id=None)
    rows+=public;idx={r['id']:r for r in rows};allps={};native={};metas={};maxdiff=0.
    for name,jid in JOBS.items():
        j=api(f'/actions/jobs/{jid}')
        if j['conclusion']!='success':raise RuntimeError('Job incomplete')
        text=api(f'/actions/jobs/{jid}/logs',True);parsed=parse(text)
        protocols=[p for p in parsed if p['kind']=='protocol'];summaries=[p for p in parsed if p['kind']=='probe_summary']
        if len(protocols)!=1 or len(summaries)!=1:raise ValueError('Missing protocol or final summary')
        if protocols[0]['data_hash']!=digest(rows) or summaries[0]['data_hash']!=digest(rows):raise ValueError('Data drift; cannot combine checkpoints')
        preds=[p for p in parsed if p['kind']=='probe_prediction']
        if len(preds)!=400 or len({p['id'] for p in preds})!=400 or {p['id'] for p in preds}!=set(idx):raise ValueError('Incomplete/duplicated predictions')
        for p in preds:
            r=idx[p['id']]
            if p['input_hash']!=digest(public_input(r)):raise ValueError('Input identity mismatch')
            if p['status'] not in ('ok','tie'):raise ValueError('Error record; refuse to drop it')
            ps=prob(p['logits']);d=max(abs(x-y) for x,y in zip(ps,p['probabilities'],strict=True));maxdiff=max(maxdiff,d)
            if d>1e-10 or choose(ps)!=p['prediction']:raise ValueError('Incorrect model arithmetic')
        native[name]={p['id']:p['logits'] for p in preds};allps[name]={p['id']:p['probabilities'] for p in preds}
        summary=summaries[0];metas[name]=summary
        times=sorted(p['seconds'] for p in preds)
        emit('completed_checkpoint_audit',model=name,job_id=jid,data_hash=digest(rows),log_sha256=hashlib.sha256(text.encode()).hexdigest(),
          checked_predictions=len(preds),recomputed=evaluate(rows,allps[name]),reported_summary=summary,
          elapsed_seconds={'p50':statistics.median(times),'p95_nearest_rank':times[math.ceil(.95*len(times))-1]},
          assets=[p for p in parsed if p['kind']=='assets'],loading=[p for p in parsed if p['kind']=='loading'],
          warmup=[p for p in parsed if p['kind']=='warmup'],fp32_readout=[p for p in parsed if p['kind']=='fp32_readout_check'])
    for split in ('test','regression'):
        rr=[r for r in rows if r['split']=='test'] if split=='test' else [r for r in rows if r['split'].startswith('regression-')]
        a,b=JOBS;errors=[r for r in rr if choose(allps[a][r['id']])!=r['target']]
        bothagree=[r for r in rr if choose(allps[a][r['id']])==choose(allps[b][r['id']])]
        emit('two_checkpoint_complementarity',population=split,n=len(rr),
          oracle_union=sum(any(choose(allps[n][r['id']])==r['target'] for n in JOBS) for r in rr),
          gliclass_errors=len(errors),qwen_08b_rescues=sum(choose(allps[b][r['id']])==r['target'] for r in errors),
          agreed=len(bothagree),agreed_correct=sum(choose(allps[a][r['id']])==r['target'] for r in bothagree))
    emit('completed_checkpoint_receipt',status='complete for two models only',verified_predictions=800,maximum_probability_discrepancy=maxdiff,
         data_hash=digest(rows),source_run=35518735165,excluded={'qwen-4b':'25-minute timeout; only 27 calibration cases; no test conclusion'},
         sources=sources,limitations=['six synthetic blocks per cohort','not private Duplex development set','public regression subsets not fresh benchmarks','no new encoder/decoder weights'])
    synth=synthetic()
    for name in JOBS:
        vals=[x for r in synth for x in native[name][r['id']]]
        data=struct.pack('<'+'f'*len(vals),*vals)
        emit('portable_logits',model=name,encoding='zlib-base64-little-endian-float32',shape=[len(synth),6],row_order_hash=digest([r['id'] for r in synth]),
          uncompressed_sha256=hashlib.sha256(data).hexdigest(),revision=metas[name]['revision'],checkpoint=metas[name]['checkpoint'],
          payload=base64.b64encode(zlib.compress(data,9)).decode())
if __name__=='__main__':main()
