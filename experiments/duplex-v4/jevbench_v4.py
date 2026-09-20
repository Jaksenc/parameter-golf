"""External public-subset evaluation of development-locked v4 checkpoints.
No training, temperature fitting, prompt search, or test-based model selection.
All checkpoints share one frozen prefix per task; replay checks full forwards.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jevbench_public_v1 as bench
from jevbench_transport_v1 import safe_get, test_redirect
from duplex_v4.io import DATA_HASH, REVISION, filehash, save_json

TRAINING_RUN=35545041177
SHARDS=8
ORIGINAL_SOURCE='14cd69ec52ee1d75bebea28de354101bfe61faed5c9b4543f8203d60a5d17242'

def emit(kind,**kw):
    print(json.dumps(dict(kind=kind,**kw),sort_keys=True,allow_nan=False),flush=True)

def initialize_benchmark():
    if filehash(bench.__file__)!=ORIGINAL_SOURCE:raise RuntimeError('Pinned external adapter changed')
    test_redirect();bench.self_test();bench.get=safe_get;bench.SHARDS=SHARDS

def get_training(root):
    items=json.loads(safe_get(f'https://api.github.com/repos/{bench.REPO}/actions/runs/{TRAINING_RUN}/artifacts?per_page=100',True))['artifacts']
    matches=[a for a in items if a['name']=='v4-training' and not a['expired']]
    if len(matches)!=1:raise RuntimeError('Completed training artifact required')
    item=matches[0];blob=safe_get(item['archive_download_url'],True)
    if item.get('digest')!='sha256:'+bench.sha(blob):raise RuntimeError('Training archive identity changed')
    bench.unzip(blob,root)
    selection=json.loads((root/'selection.json').read_text())
    if not selection.get('locked_before_transfer') or selection['data_hash']!=DATA_HASH or selection['model_revision']!=REVISION:raise RuntimeError('Invalid development selection')
    expected={(a,s) for a in ['direct','anchored','structured'] for s in [41,73,101]}
    if len(selection['results'])!=9 or {(s['arm'],s['seed']) for s in selection['results']}!=expected:raise RuntimeError('Incomplete candidate set')
    for s in selection['results']:
        p=root/f"{s['arm']}-{s['seed']}"/'selected.npz'
        if not s['complete'] or s['steps']!=400 or filehash(p)!=s['selected_sha256']:raise RuntimeError('Unfinished or modified checkpoint')
    receipt={'source_run':TRAINING_RUN,'artifact_id':item['id'],'archive_sha256':bench.sha(blob),'selection_sha256':filehash(root/'selection.json'),'preselected_method':selection['winner']}
    return selection,receipt

def run(shard):
    if shard not in range(SHARDS):raise ValueError('Invalid shard')
    initialize_benchmark()
    root=Path(f'jevbench-v4-{shard}');root.mkdir(exist_ok=False)
    selection,receipt=get_training(root/'training')
    rows,manifest=bench.prepare(root/'assets',adapters=False)
    idx={r['id']:r for r in rows};selected=manifest['partitions'][shard]
    # No gold label is consulted to allocate work or choose a replay input.
    replay_id=min(selected,key=lambda rid:(len(json.dumps(bench.public_input(idx[rid]))),rid))
    import torch
    import torch.nn.functional as F
    from duplex_v4.cache import load_model,terminal_modules
    from duplex_v4.terminal import qwen_rms
    from duplex_v4.train import load_adapter
    from semif_phase1.direct import encode_prompt
    model,tokenizer,hashes=load_model()
    name,mod,layer,norm=terminal_modules(model);head=model.get_output_embeddings()
    candidates={f"{s['arm']}-{s['seed']}":load_adapter(root/'training'/f"{s['arm']}-{s['seed']}"/'selected.npz',mod.in_features,mod.out_features) for s in selection['results']}
    for a in candidates.values():a.eval()
    configs=['baseline']+list(candidates);capture={};handles=[]
    def pre(key):
        def fn(module,args):capture[key]=args[0][:,-1,:].detach().clone()
        return fn
    def after(module,args,y):capture['y']=y[:,-1,:].detach().clone()
    handles=[layer.post_attention_layernorm.register_forward_pre_hook(pre('r')),mod.register_forward_pre_hook(pre('x')),mod.register_forward_hook(after),norm.register_forward_pre_hook(pre('norm_in')),head.register_forward_pre_hook(pre('h'))]
    records=[];replay=[];errors=0;physical=0;max_norm=0.
    save_json(root/'protocol.json',{'source_run':TRAINING_RUN,'receipt':receipt,'benchmark_revision':bench.BENCH_REV,'semif_revision':bench.SEMIF_REV,'model_revision':REVISION,'weights':hashes,'data_hash':manifest['data_hash'],'source_sha256':filehash(__file__),'shard':shard,'shards':SHARDS,'task_ids':selected,'configs':configs,'training':False,'max_tokens':8192,'scoring':'Unmodified SemIf prompt with FP32 restricted-answer projection; all candidates retain original labels/order','replay_id':replay_id,'no_new_model_selection':True})
    try:
        with torch.no_grad():
            for index,rid in enumerate(selected):
                raw=idx[rid];inp=bench.public_input(raw);_,semi=bench.convert(inp)
                start=time.perf_counter();item=[]
                try:
                    ids,slots,prompt_hash=encode_prompt(tokenizer,semi,max_tokens=8192)
                    tensors={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)}
                    capture.clear();model(**tensors,use_cache=False,logits_to_keep=1);physical+=1
                    if not torch.equal(capture['r']+capture['y'],capture['norm_in']):raise RuntimeError('Terminal residual mismatch')
                    h=qwen_rms(capture['norm_in'],norm.weight,norm.variance_epsilon)
                    e=float((h.float()-capture['h'].float()).abs().max());max_norm=max(max_norm,e)
                    if e>2e-5:raise RuntimeError('Normalization mismatch')
                    weights=head.weight[slots].float();bias=head.bias[slots].float() if head.bias is not None else None
                    base=F.linear(capture['h'].float(),weights,bias)[0]
                    cache={k:capture[k] for k in ('x','r','y')}
                    cache.update(norm_weight=norm.weight,norm_epsilon=norm.variance_epsilon,answer_weight=weights)
                    if bias is not None:cache['answer_bias']=bias
                    backbone_seconds=time.perf_counter()-start
                    zz={'baseline':base}
                    for tag,adapter in candidates.items():zz[tag]=adapter(cache)[0]
                    if any(not torch.isfinite(z).all() for z in zz.values()):raise RuntimeError('Invalid model logits')
                    for tag,z in zz.items():
                        scores=z.float().tolist();item.append({'id':rid,'config':tag,'input_hash':bench.digest(inp),'prompt_sha256':prompt_hash,'labels':inp['labels'],'logits':scores,'probabilities':bench.softmax(scores),'tokens':len(ids),'status':'ok','shared_backbone_seconds':backbone_seconds})
                    if rid==replay_id:
                        for tag,adapter in candidates.items():
                            def apply(module,args,y):return y+adapter.delta(args[0])
                            active=mod.register_forward_hook(apply)
                            try:
                                capture.clear();model(**tensors,use_cache=False,logits_to_keep=1);physical+=1
                                actual=F.linear(capture['h'].float(),weights,bias)[0]
                                delta=float((actual-zz[tag]).abs().max())
                                if delta>2e-3 or actual.argmax()!=zz[tag].argmax():raise RuntimeError('Full model replay disagrees: '+tag)
                                replay.append({'id':rid,'config':tag,'max_logit_difference':delta})
                            finally:active.remove()
                        capture.clear();model(**tensors,use_cache=False,logits_to_keep=1);physical+=1
                        restore=F.linear(capture['h'].float(),weights,bias)[0]
                        if float((restore-base).abs().max())>2e-5:raise RuntimeError('Baseline restoration failed')
                except Exception as exc:
                    errors+=1
                    save_json(root/'failure.json',{'id':rid,'type':type(exc).__name__,'message':str(exc),'complete':False})
                    raise
                for p in item:
                    records.append(p)
                    with (root/'records.jsonl').open('a') as f:f.write(json.dumps(p,sort_keys=True,allow_nan=False)+'\n')
                emit('external_progress',shard=shard,completed=index+1,population=len(selected))
    finally:
        for h in handles:h.remove()
    if len(records)!=len(selected)*10 or len(replay)!=9:raise RuntimeError('Incomplete benchmark population')
    save_json(root/'complete.json',{'complete':True,'task_ids':selected,'score_vectors':len(records),'physical_model_forwards':physical,'full_network_replay':replay,'max_norm_error':max_norm,'errors':errors,'receipt':receipt,'data_hash':manifest['data_hash'],'configs':configs,'shard':shard,'shards':SHARDS,'records_sha256':filehash(root/'records.jsonl')})
    emit('external_shard_complete',shard=shard,cases=len(selected),vectors=len(records),physical_forwards=physical)

def audit(run_id):
    initialize_benchmark();root=Path('jevbench-v4-audit');root.mkdir(exist_ok=False)
    rows,manifest=bench.prepare(root/'assets',adapters=False);idx={r['id']:r for r in rows}
    from jevbench.tasks import Task
    from jevbench.scoring import score_task
    listing=json.loads(safe_get(f'https://api.github.com/repos/{bench.REPO}/actions/runs/{run_id}/artifacts?per_page=100',True))['artifacts']
    found={};receipts=[];selection_hash=None;configs=None;total_forwards=0;replays=0
    for shard in range(SHARDS):
        matches=[a for a in listing if a['name']==f'v4-jevbench-{shard}' and not a['expired']]
        if len(matches)!=1:raise RuntimeError('Missing or ambiguous shard')
        item=matches[0];blob=safe_get(item['archive_download_url'],True)
        if item.get('digest')!='sha256:'+bench.sha(blob):raise RuntimeError('Artifact integrity mismatch')
        dest=root/'shards'/str(shard);bench.unzip(blob,dest)
        c=json.loads((dest/'complete.json').read_text())
        if not c['complete'] or c['errors'] or c['data_hash']!=manifest['data_hash'] or c['task_ids']!=manifest['partitions'][shard] or filehash(dest/'records.jsonl')!=c['records_sha256']:raise RuntimeError('Bad completion record')
        h=c['receipt']['selection_sha256']
        if selection_hash is None:selection_hash=h;configs=c['configs']
        if h!=selection_hash or configs!=c['configs']:raise RuntimeError('Checkpoint selection changed across shards')
        pp=[json.loads(s) for s in (dest/'records.jsonl').read_text().splitlines()]
        if len(pp)!=c['score_vectors']:raise RuntimeError('Missing records')
        for p in pp:
            key=(p['id'],p['config']);raw=idx.get(p['id'])
            if key in found or raw is None or p['id'] not in c['task_ids'] or p['config'] not in configs or p['status']!='ok':raise RuntimeError('Unexpected record')
            if p['input_hash']!=bench.digest(bench.public_input(raw)) or p['labels']!=raw['labels']:raise RuntimeError('Changed input')
            calc=bench.softmax(p['logits'])
            if max(abs(a-b) for a,b in zip(calc,p['probabilities'],strict=True))>1e-8:raise RuntimeError('Probability mismatch')
            found[key]=p
        receipts.append({'artifact_id':item['id'],'sha256':bench.sha(blob),'completion':c})
        total_forwards+=c['physical_model_forwards'];replays+=len(c['full_network_replay'])
    if len(found)!=231*10:raise RuntimeError('Incomplete total population')
    scored={name:{} for name in configs}
    for name in configs:
        for r in rows:scored[name][r['id']]=score_task(dict(zip(r['labels'],found[r['id'],name]['probabilities'])),Task.from_dict(r))
    def metrics(rr,name):
        ss=[scored[name][r['id']] for r in rr];valid=[(r,p) for r,p in zip(rr,ss) if p['valid']]
        ece=0.
        for j in range(10):
            b=[p for r,p in valid if min(int(max(p['probs'].values())*10),9)==j]
            if b:ece+=len(b)/len(valid)*abs(statistics.mean(max(p['probs'].values()) for p in b)-statistics.mean(p['correct'] for p in b))
        pr=[(r,p) for r,p in valid if r.get('provenance',{}).get('gold_probs')]
        return {'n':len(rr),'correct':sum(bool(p['correct']) for p in ss),'valid':len(valid),'nll':statistics.mean(-math.log(max(p['probs'][str(r['expected'])],1e-30)) for r,p in valid),'brier':statistics.mean(sum((v-int(k==str(r['expected'])))**2 for k,v in p['probs'].items()) for r,p in valid),'ece10':ece,'wrong_above_95':sum(not p['correct'] and max(p['probs'].values())>.95 for r,p in valid),'gold_probability_n':len(pr),'probability_tvd':statistics.mean(.5*sum(abs(p['probs'][k]-r['provenance']['gold_probs'][k]) for k in r['labels']) for r,p in pr) if pr else None}
    groups={'all':rows,**{t:[r for r in rows if r['_tier']==t] for t in ['easy','standard','hard']}}
    summary={group:{name:metrics(rr,name) for name in configs} for group,rr in groups.items()}
    pairs={name:{g:{'repairs':sum(not scored['baseline'][r['id']]['correct'] and scored[name][r['id']]['correct'] for r in rr),'harms':sum(scored['baseline'][r['id']]['correct'] and not scored[name][r['id']]['correct'] for r in rr)} for g,rr in groups.items()} for name in configs if name!='baseline'}
    result={'complete':True,'run':run_id,'training_run':TRAINING_RUN,'public_tasks':231,'score_vectors':len(found),'physical_model_forwards':total_forwards,'full_network_replay_checks':replays,'selection_sha256':selection_hash,'preselected_method':receipts[0]['completion']['receipt']['preselected_method'],'model_revision':REVISION,'benchmark_revision':bench.BENCH_REV,'semif_revision':bench.SEMIF_REV,'summary':summary,'pairs':pairs,'receipts':receipts,'production_promoted':False,'limitations':['JevBench public subset only, not full leaderboard or rank','No benchmark training, calibration, prompt search or candidate selection','All nine development-locked candidates reported, not nine independent pretrained runs','Scores use cached exact terminal computation plus 72 full-forward spot checks; 231 shared prefixes','One original public corpus, already exposed in prior work; not fresh private held-out benchmark','No CPU-to-GPU/Mac latency comparison or invented zero-cost serving price']}
    save_json(root/'results.json',result);save_json(root/'predictions.json',[found[k] for k in sorted(found)]);save_json(root/'scored.json',scored)
    lines=['# Duplex v4: JevBench public-subset results','',f"Development-selected method: {result['preselected_method']}. No checkpoint selected using these outcomes.",'','| Configuration | Easy /48 | Standard /72 | Hard /111 | All /231 | NLL |','|---|---:|---:|---:|---:|---:|']
    for name in configs:lines.append(f"| {name} | {summary['easy'][name]['correct']} | {summary['standard'][name]['correct']} | {summary['hard'][name]['correct']} | {summary['all'][name]['correct']} | {summary['all'][name]['nll']:.4f} |")
    lines+=['','231 shared-prefix forwards plus replay checks. This is not an official full-suite score, independent replication, or serving benchmark.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');emit('external_audit_complete',summary=summary,pairs=pairs,preselected=result['preselected_method'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int);ap.add_argument('--audit',type=int);args=ap.parse_args()
    if args.shard is not None:run(args.shard)
    elif args.audit is not None:audit(args.audit)
    else:ap.error('Provide --shard or --audit')
