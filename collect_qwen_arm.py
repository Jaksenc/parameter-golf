"""Read-only audit of the native ARM recovery. No inference or model selection."""
import base64,hashlib,math,os,statistics,struct,time,zlib
import requests
from audit_probe_v2 import parse,prob,choose,evaluate,emit
from decision_probe_v2 import synthetic,digest,public_input
JOBS={0:106103489901,1:106103489713,2:106103489947,3:106103489833}

def api(path,text=False):
    u='https://api.github.com/repos/Jaksenc/parameter-golf'+path
    if text:u+='?audit='+str(time.time_ns())
    r=requests.get(u,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'},timeout=90)
    r.raise_for_status();return r.content.decode('utf-8-sig') if text else r.json()
def main():
    statuses={i:api(f'/actions/jobs/{j}')['conclusion'] for i,j in JOBS.items()}
    if any(v!='success' for v in statuses.values()):
        emit('qwen_arm_incomplete',job_conclusions=statuses,status='not a complete model benchmark')
        raise RuntimeError('Require all four successful shards; no imputation')
    from benchmark import public_data
    rows=synthetic();public,sources=public_data(16)
    for r in public:r.update(split='regression-'+r['family'],base_id=None)
    rows+=public;idx={r['id']:r for r in rows};preds=[];receipts=[];assets=[];maxdiff=0.
    for shard,jid in JOBS.items():
        text=api(f'/actions/jobs/{jid}/logs',True);parsed=parse(text)
        protocol=[r for r in parsed if r['kind']=='protocol'];execution=[r for r in parsed if r['kind']=='shard_execution'];summary=[r for r in parsed if r['kind']=='shard_summary']
        if len(protocol)!=1 or protocol[0]['data_hash']!=digest(rows) or len(summary)!=1 or summary[0]['data_hash']!=digest(rows):raise ValueError('Protocol mismatch')
        if len(execution)!=1 or execution[0]['shard_index']!=shard or not execution[0]['native_score_functions_ast_unchanged']:raise ValueError('Invalid shard')
        p=[r for r in parsed if r['kind']=='probe_prediction']
        expected={r['id'] for i,r in enumerate(rows) if i%4==shard}
        if len(p)!=100 or {r['id'] for r in p}!=expected or len({r['id'] for r in p})!=100:raise ValueError('Incomplete/duplicate shard')
        for r in p:
            case=idx[r['id']]
            if r['status'] not in ('ok','tie') or r['input_hash']!=digest(public_input(case)):raise ValueError('Failure or changed input')
            ps=prob(r['logits']);d=max(abs(a-b) for a,b in zip(ps,r['probabilities'],strict=True));maxdiff=max(maxdiff,d)
            if d>1e-10 or choose(ps)!=r['prediction']:raise ValueError('Wrong score arithmetic')
        preds+=p
        receipts.append({'job_id':jid,'shard':shard,'log_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'execution':execution[0],'summary':summary[0], 'fp32_readout':[r for r in parsed if r['kind']=='fp32_readout_check'],
          'loading':[r for r in parsed if r['kind']=='loading'],'warmup':[r for r in parsed if r['kind']=='warmup']})
        assets+=[r for r in parsed if r['kind']=='assets']
    if len({p['id'] for p in preds})!=400:raise ValueError('Duplicate across shards')
    if len({r['revision'] for r in assets})!=1 or len({digest(r['files']) for r in assets})!=1:raise ValueError('Different weight files across shards')
    ps={r['id']:r['probabilities'] for r in preds};logits={r['id']:r['logits'] for r in preds};times=sorted(r['seconds'] for r in preds)
    emit('qwen_arm_verified',status='all 400 cases complete',data_hash=digest(rows),run_id=35520447897,checked_predictions=400,
      model='qwen-4b',checkpoint=assets[0]['model'],revision=assets[0]['revision'],maximum_probability_discrepancy=maxdiff,
      metrics=evaluate(rows,ps),receipts=receipts,assets=assets[0],sources=sources,
      elapsed_seconds={'p50':statistics.median(times),'p95_nearest_rank':times[math.ceil(.95*len(times))-1]},
      limitations=['Four separate standard ARM VMs; not a single-server end-to-end latency claim','Native score functions unchanged; CPU architecture differs from first job','No training or private Duplex evaluation'])
    for name,jid in [('gliclass-instruct',106099011148),('qwen-08b',106099011005)]:
        other={p['id']:p['probabilities'] for p in parse(api(f'/actions/jobs/{jid}/logs',True)) if p['kind']=='probe_prediction'}
        if set(other)!=set(idx):raise ValueError('Other model incomplete')
        for pop,rr in [('test',[r for r in rows if r['split']=='test']),('regression',[r for r in rows if r['split'].startswith('regression-')])]:
            errors=[r for r in rr if choose(ps[r['id']])!=r['target']]
            emit('qwen_arm_complementarity',other=name,population=pop,n=len(rr),qwen_4b_errors=len(errors),
              other_rescues=sum(choose(other[r['id']])==r['target'] for r in errors),
              oracle_union=sum(choose(ps[r['id']])==r['target'] or choose(other[r['id']])==r['target'] for r in rr))
    synth=synthetic();vals=[v for r in synth for v in logits[r['id']]];data=struct.pack('<'+'f'*len(vals),*vals)
    emit('portable_logits',model='qwen-4b',checkpoint=assets[0]['model'],revision=assets[0]['revision'],
      encoding='zlib-base64-little-endian-float32',shape=[336,6],row_order_hash=digest([r['id'] for r in synth]),
      uncompressed_sha256=hashlib.sha256(data).hexdigest(),payload=base64.b64encode(zlib.compress(data,9)).decode())
if __name__=='__main__':main()
