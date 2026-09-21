"""Read-only collection of completed pilot outputs; never invokes a model or writes GitHub."""
from __future__ import annotations
import base64,collections,gzip,hashlib,importlib.util,json,os,statistics,urllib.request
from pathlib import Path
from bbeh_run import BBEH_REF,SCORER_BLOB,download,sha
REPO='Jaksenc/parameter-golf'
BRANCH='research/adaptive-external-benchmarks-20260920'
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'adaptive-readonly-receipts','Accept':'application/vnd.github+json'})
    return urllib.request.urlopen(req,timeout=60).read()
def main():
    branch=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/ref/heads/{BRANCH}'))
    commit=branch['object']['sha'];files=[];missing=[];shards=[]
    for i in range(6):
        path=f'research/adaptive_external/results/bbeh-shard-{i:02d}.json'
        url=f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}'
        try:
            raw=fetch(url);obj=json.loads(raw)
            if obj['shard']!=i or obj['run_id']!='35546669639':raise ValueError('Unexpected receipt identity')
            files.append({'path':path,'sha256':sha(raw),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'bytes':len(raw),'url':url})
            shards.append(obj)
        except Exception as exc:missing.append({'shard':i,'error':f'{type(exc).__name__}: {exc}'})
    scorer=download('bbeh/evaluate.py',SCORER_BLOB);sp=Path('/tmp/bbeh-evaluator.py');sp.write_bytes(scorer)
    spec=importlib.util.spec_from_file_location('official_bbeh',sp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=[r for s in shards for r in s['rows']]
    keys=[(r['id'],r['arm']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate case/arm receipt')
    for r in rows:
        expected=r['status'] in ('ok','budget_limited') and mod.evaluate_correctness(r['response'],r['reference'])
        if bool(expected)!=r['correct']:raise ValueError('Score replay mismatch')
    if shards:
        selected=shards[0]['selected_ids']
        for s in shards:
            if s['selected_ids']!=selected:raise ValueError('Inconsistent selected cohort')
        if not missing and set(keys)!={(i,a) for i in selected for a in ('direct','reasoning')}:raise ValueError('Incomplete planned predictions')
    summary={};bytask={}
    for arm in ('direct','reasoning'):
        rr=[r for r in rows if r['arm']==arm];times=[r['generation_seconds'] for r in rr if r.get('generation_seconds') is not None]
        summary[arm]={'correct':sum(r['correct'] for r in rr),'received_cases':len(rr),'planned_cases':46,'statuses':dict(collections.Counter(r['status'] for r in rr)),'generated_tokens':sum(r.get('output_tokens') or 0 for r in rr),'input_tokens':sum(r.get('input_tokens') or 0 for r in rr),'total_generation_seconds':sum(times),'median_generation_seconds':statistics.median(times) if times else None,'time_limited':sum(bool(r.get('seconds_soft_limit_reached')) for r in rr),'eos_reached':sum(bool(r.get('eos_reached')) for r in rr)}
    for task in sorted({r['task'] for r in rows}):
        bytask[task]={arm:{'correct':sum(r['correct'] for r in rows if r['task']==task and r['arm']==arm),'total':sum(r['task']==task and r['arm']==arm for r in rows)} for arm in ('direct','reasoning')}
    direct={r['id']:r for r in rows if r['arm']=='direct'};reason={r['id']:r for r in rows if r['arm']=='reasoning'};common=set(direct)&set(reason)
    paired={'cases':len(common),'repaired':sum(not direct[i]['correct'] and reason[i]['correct'] for i in common),'broken':sum(direct[i]['correct'] and not reason[i]['correct'] for i in common),'both_correct':sum(direct[i]['correct'] and reason[i]['correct'] for i in common),'both_wrong':sum(not direct[i]['correct'] and not reason[i]['correct'] for i in common)}
    report={'kind':'BBEH Mini 46-case frozen-model pilot','results_commit':commit,'run_id':'35546669639','missing_shards':missing,'all_six_received':not missing,'official_score_replay_pass':True,'summary':summary,'paired':paired,'by_task':bytask,'source_files':files,'scorer_sha256':sha(scorer),'total_model_load_seconds':sum(s.get('manifest',{}).get('load_seconds',0) for s in shards),'shard_request_wall_seconds':[s['request_wall_seconds'] for s in shards],'not_full_adaptive_system':True,'not_full_benchmark':True,'energy_measured':False}
    bundle={'aggregate':report,'shards':shards,'official_scorer_utf8':scorer.decode()}
    raw=json.dumps(bundle,separators=(',',':')).encode();packed=gzip.compress(raw,mtime=0)
    print('RESULT_SUMMARY '+json.dumps(report),flush=True)
    print('RESULT_PACK_MANIFEST '+json.dumps({'compressed_sha256':sha(packed),'decoded_sha256':sha(raw),'bytes':len(packed)}),flush=True)
    payload=base64.b64encode(packed).decode()
    for i in range(0,len(payload),4000):print('RESULT_PACK '+str(i//4000)+' '+payload[i:i+4000],flush=True)
    if missing:raise SystemExit('Receipt collection incomplete; no complete aggregate claimed')
if __name__=='__main__':main()
