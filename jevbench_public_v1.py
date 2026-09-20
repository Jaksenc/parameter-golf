"""Duplex / JevBench public-only comparison. Frozen models, no training.
Four primary configurations; upstream SemIf direct.score on a shared CPU loader.
The SemIf published CUDA/MLX deployment is NOT reproduced by this CPU study.
All 231 public tasks are included; official tie/scoring rules are preserved.
"""
from __future__ import annotations
import argparse,copy,hashlib,io,json,math,os,platform,resource,statistics,sys,time,urllib.request,zipfile
from pathlib import Path
from collections import Counter,defaultdict
REPO='Jaksenc/parameter-golf'
BENCH_REV='7128f5cf445ca41e4da1bdc7c84f97926250724d'
SEMIF_REV='ca3ba65f142967030ecb453346e94d6f476a69df'
MODEL_REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAX_TOKENS=8192
SHARDS=16
CONFIGS=['base','direct_adapter','fact_adapter','semif_native']
DATA_FILES={'easy.jsonl':('1d5a81a6fc8c5e2dd3e243202802f0f3a04c7f35',48),'original.jsonl':('c65c0f7b40856f85e0e441d4dcbafa4a0be14fd9',72),'hard.jsonl':('8bf8451fb84fd3cd50514e8c0e0c55b0c38e3679',111)}
ASSETS={'direct_adapter':(10612070095,'d89c92118e3b12ca9f191a8e69ef6a9836021955d9bf6e0443887bed73b7cc5a','be9919dce3953466ac155d533c67458db52eec310252824954e6a174c1f8dc4e'),'fact_adapter':(10611114575,'7d5eecaae6f85ecf3096596100f95e4299883e6a9201dcf9e2629893da277e07','d2aca4c357b4b862602e4552c6a9cc0c78d0c7417406194247bbdbd765b7be9f')}
CODE_BLOBS={'semantic_data_v1.py':'9a2e7846940cfc9ca405855afcf2524af82b0fe3','semantic_train_v1.py':'651f53916739407e6a5b7c81be300673f37b6562'}

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def gitsha(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def emit(kind,**kw):print(json.dumps({'kind':kind,**kw},sort_keys=True,allow_nan=False),flush=True)
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False));temp.replace(p)
def get(url,auth=False):
    headers={'User-Agent':'Duplex-public-benchmark/1.0'}
    if auth:headers['Authorization']='Bearer '+os.environ['GH_TOKEN'];headers['Accept']='application/vnd.github+json'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as r:
        data=r.read(25_000_001)
    if len(data)>25_000_000:raise ValueError('Download exceeds bounded resource size')
    return data
def softmax(z):
    if not z or not all(math.isfinite(x) for x in z):raise ValueError('Invalid logits')
    m=max(z);v=[math.exp(x-m) for x in z];s=sum(v);return [x/s for x in v]
def unzip(blob,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if sum(v.file_size for v in z.infolist())>20_000_000:raise ValueError('Oversized archive')
        for v in z.infolist():
            p=Path(v.filename)
            if p.is_absolute() or '..' in p.parts or ((v.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe archive member')
        z.extractall(dest)

def public_input(row):
    # Metadata, expected, provenance, rationales and gold probabilities are NEVER forwarded.
    return {k:copy.deepcopy(row[k]) for k in ['id','state','question','labels']}
def convert(inp):
    labels=inp['labels'];q=inp['question'];criteria=q.get('criteria')
    if len(labels)!=len(set(labels)) or not 2<=len(labels)<=16:raise ValueError('Unsupported candidate count or duplicates')
    if q.get('type') not in ['choice','noul','score']:raise ValueError('Unsupported question type')
    defs=[]
    for i,label in enumerate(labels):
        if isinstance(criteria,list):
            if q['type']!='score' or not str(label).isdigit() or int(label)>=len(criteria):raise ValueError('Invalid ordinal criteria')
            detail=criteria[int(label)]
        elif isinstance(criteria,dict):
            key=({'yes':'true','no':'false'}[label] if q['type']=='noul' and label in ['yes','no'] and label not in criteria else label)
            if key not in criteria:raise ValueError('Incomplete option definition: '+label)
            detail=criteria[key]
        elif criteria is None:detail=label
        else:raise ValueError('Invalid rubric')
        defs.append({'id':label,'description':label+': '+(detail if isinstance(detail,str) else json.dumps(detail,ensure_ascii=False))})
    criterion=json.dumps(q,ensure_ascii=False)
    semi={'id':inp['id'],'state':inp['state'],'question':criterion,'options':defs}
    evidence=inp['state'] if isinstance(inp['state'],str) else json.dumps(inp['state'],ensure_ascii=False)
    ours={'content':'Evidence:\n'+evidence+'\nQuestion and complete rubric:\n'+criterion,'options':[o['description'] for o in defs]}
    return ours,semi

def prepare(root,adapters=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);vendor=root/'vendor';vendor.mkdir(exist_ok=True);manifest={'benchmark_revision':BENCH_REV,'semif_revision':SEMIF_REV,'files':{}};rows=[]
    def fetch(repo,rev,path,dest,expected=None):
        b=get('https://raw.githubusercontent.com/'+repo+'/'+rev+'/'+path)
        if expected and gitsha(b)!=expected:raise ValueError('Pinned source mismatch: '+path)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);manifest['files'][repo+':'+path]={'sha256':sha(b),'git_blob':gitsha(b),'bytes':len(b)};return b
    for name,(h,n) in DATA_FILES.items():
        b=fetch('fstandhartinger/jevbench',BENCH_REV,'datasets/public/'+name,root/'datasets'/name,h)
        rr=[json.loads(line) for line in b.decode().splitlines() if line.strip()]
        if len(rr)!=n:raise ValueError('Public population changed: '+name)
        for r in rr:r['_tier']={'original.jsonl':'standard','easy.jsonl':'easy','hard.jsonl':'hard'}[name]
        rows+=rr
    for f in ['__init__.py','tasks.py','scoring.py','metrics.py']:
        fetch('fstandhartinger/jevbench',BENCH_REV,'jevbench/'+f,vendor/'jevbench'/f)
    for f,h in [('__init__.py',None),('core.py','6e93b16dcd4ab56c7a859c9046c48c6731d7180c'),('direct.py','943f34728d1966bf2325d6e6d34d2a7043cd6a56')]:
        fetch('TheoLeeCJ/SemIf',SEMIF_REV,'src/semif_phase1/'+f,vendor/'semif_phase1'/f,h)
    for repo,rev,label in [('fstandhartinger/jevbench',BENCH_REV,'jevbench'),('TheoLeeCJ/SemIf',SEMIF_REV,'semif')]:
        fetch(repo,rev,'LICENSE',root/(label+'-LICENSE.txt'))
    sys.path.insert(0,str(vendor.resolve()));from jevbench.tasks import Task
    if len(rows)!=231 or len({r['id'] for r in rows})!=231:raise ValueError('Missing/duplicated public cases')
    for r in rows:
        t=Task.from_dict(r)
        if t.split!='public':raise ValueError('Nonpublic item')
        convert(public_input(r))
    # Length-balanced execution only. No scores, labels, expected answers or family are used.
    groups=defaultdict(list)
    for r in rows:groups[r.get('group') or r['id']].append(r)
    bins=[[] for _ in range(SHARDS)];loads=[0.]*SHARDS
    sortedgroups=sorted(groups.items(),key=lambda kv:(-sum(len(json.dumps(public_input(r)))**1.3 for r in kv[1]),kv[0]))
    for group,rr in sortedgroups:
        k=min(range(SHARDS),key=lambda j:(loads[j],j));bins[k]+=[r['id'] for r in rr];loads[k]+=sum(len(json.dumps(public_input(r)))**1.3 for r in rr)
    manifest.update(cases=231,data_hash=digest(rows),partitions=bins,max_context=MAX_TOKENS,readout='restricted next-token probabilities; no generated confidence; uncalibrated',training=False,cost_usd_per_1000=None,complete_leaderboard=False)
    write(root/'tasks.json',rows);write(root/'inputs.json',[public_input(r) for r in rows]);write(root/'manifest.json',manifest)
    if adapters:
        for arm,(ident,zh,ah) in ASSETS.items():
            b=get(f'https://api.github.com/repos/{REPO}/actions/artifacts/{ident}/zip',auth=True)
            if sha(b)!=zh:raise ValueError('Checkpoint archive changed')
            dest=root/'adapters'/arm;unzip(b,dest)
            if sha((dest/'adapter.safetensors').read_bytes())!=ah:raise ValueError('Adapter bytes changed')
            cfg=json.loads((dest/'adapter_config.json').read_text())
            if cfg['sha256']!=ah or cfg['revision']!=MODEL_REV:raise ValueError('Wrong model revision')
            for f,h in CODE_BLOBS.items():
                if gitsha((dest/f).read_bytes())!=h:raise ValueError('Archived runtime changed')
    return rows,manifest

def self_test():
    q={'id':'fixture','state':{'name':'x','value':5},'question':{'type':'choice','instructions':'Select its color.','criteria':{'red':'red','blue':'blue'}},'labels':['red','blue'],'expected':'red','provenance':{'rationale':'SECRET'},'family':'toy','group':'g'}
    a=convert(public_input(q));q['expected']='blue';q['provenance']={'gold_probs':{'red':0,'blue':1},'rationale':'OTHER_SECRET'}
    assert convert(public_input(q))==a and 'SECRET' not in json.dumps(a)
    t=public_input(q);t['question']={'type':'noul','instructions':'Is it blue?','criteria':{'false':'Not blue','true':'Blue'}};t['labels']=['no','yes'];assert convert(t)[1]['options'][0]['description']=='no: Not blue'
    t['question']={'type':'score','instructions':'Rate','criteria':['Low','Medium','High']};t['labels']=['0','1','2'];assert convert(t)[1]['options'][2]['description']=='2: High'
    assert abs(sum(softmax([1000,1001]))-1)<1e-12
    try:convert({**t,'labels':['0','0']});raise AssertionError('duplicates accepted')
    except ValueError:pass
    emit('self_tests',passed=True,tests=['gold_isolation','binary_mapping','ordinal_mapping','stable_softmax','duplicate_rejection'])

def run(shard):
    if shard not in range(SHARDS):raise ValueError('Bad shard')
    root=Path(f'jevbench-shard-{shard}');root.mkdir(exist_ok=False)
    rows,manifest=prepare(root/'assets',adapters=True);inputs={r['id']:public_input(r) for r in rows};selected=manifest['partitions'][shard]
    emit('protocol',shard=shard,planned_tasks=len(selected),planned_calls=4*len(selected),configs=CONFIGS,**{k:manifest[k] for k in ['benchmark_revision','semif_revision','data_hash','max_context']})
    source=root/'assets/adapters/direct_adapter';sys.path.insert(0,str(source.resolve()))
    import semantic_train_v1 as original
    # Only remove the pilot's artificial 384-token ceiling. All requests are unpadded.
    original.PAD=MAX_TOKENS
    rt=original.Runtime();torch=rt.torch
    from semif_phase1.direct import score as semif_score,encode_prompt
    from safetensors.torch import load_file
    fixture={'id':'integration-not-benchmark','state':'The only color mentioned is blue.','question':{'type':'choice','instructions':'Which color is stated?','criteria':{'red':'The color red','blue':'The color blue'}},'labels':['red','blue']}
    def encode_ours(inp):return rt.encode(convert(inp)[0],pad=False)
    def ours(inp):
        start=time.perf_counter();enc=encode_ours(inp)
        with torch.inference_mode():z=rt.forward(enc).tolist()
        return {'logits':z,'probabilities':softmax(z),'tokens':enc[2],'seconds':time.perf_counter()-start,'encoded_input_hash':digest({'input_ids':enc[0]['input_ids'].tolist(),'codes':enc[1]})}
    first=ours(fixture);rt.attach_lora();states={name:load_file(str(root/'assets/adapters'/name/'adapter.safetensors')) for name in ASSETS}
    expected=rt.adapter_state()
    if any(state.keys()!=expected.keys() or any(state[k].shape!=expected[k].shape for k in state) for state in states.values()):raise ValueError('Adapter layout mismatch')
    def switch(name):
        for path,mod in rt.layers.items():
            parent,leaf=path.rsplit('.',1);setattr(rt.model.get_submodule(parent),leaf,mod if name in ASSETS else mod.base)
        if name in ASSETS:rt.load_adapter(states[name])
        rt.model.eval()
    for name in CONFIGS:
        switch(name)
        if name=='semif_native':
            sf=semif_score(rt.model,rt.tokenizer,convert(fixture)[1],rt.receipt,MAX_TOKENS)
            assert len(sf['probabilities'])==2 and all(math.isfinite(x) for x in sf['option_logits'])
        else:ours(fixture)
    switch('base');again=ours(fixture);err=max(abs(x-y) for x,y in zip(first['logits'],again['logits'],strict=True))
    if err>2e-5:raise RuntimeError('Adapter switching changed frozen base')
    # Verify float32 permitted-row projection versus independent full vocabulary matmul.
    with torch.inference_mode():
        enc=encode_ours(fixture);selectedz=rt.forward(enc);h=rt.capture['hidden'].float();full=torch.nn.functional.linear(h,rt.head.weight.float(),rt.head.bias.float() if rt.head.bias is not None else None)[0]
        delta=float((full[enc[1]]-selectedz).abs().max());del full,h
    if delta>1e-4:raise RuntimeError('FP32 readout reference failed')
    # Tokenize all planned prompts first; long cases remain explicit failures, never dropped.
    lengths={};length_errors={}
    for rid in selected:
        lengths[rid]={}
        for label in ['duplex','semif']:
            try:lengths[rid][label]=(encode_ours(inputs[rid])[2] if label=='duplex' else len(encode_prompt(rt.tokenizer,convert(inputs[rid])[1],MAX_TOKENS)[0]))
            except Exception as e:length_errors[rid+':'+label]=str(e)
    pre={'baseline_restore_error':err,'fp32_reference_error':delta,'runtime':rt.receipt,'python':platform.python_version(),'machine':platform.machine(),'lengths':lengths,'length_errors':length_errors,'config_order':'cyclic by task index; all four on same VM','semif_scope':'Unmodified upstream encode_prompt and direct.score; common native CPU ConditionalGeneration loader replaces CUDA-only model loading. BF16 output readout retained; FP32 same-forward shadow also recorded.'}
    write(root/'preflight.json',pre);emit('preflight_complete',shard=shard,max_input_tokens=max([v for d in lengths.values() for v in d.values()],default=0),length_errors=length_errors)
    records=[];errors=0
    for i,rid in enumerate(selected):
        order=CONFIGS[(i+shard)%4:]+CONFIGS[:(i+shard)%4]
        for name in order:
            switch(name);start=time.perf_counter();row={'id':rid,'config':name,'shard':shard,'input_hash':digest(inputs[rid]),'labels':inputs[rid]['labels']}
            try:
                if name=='semif_native':
                    sem=convert(inputs[rid])[1];out=semif_score(rt.model,rt.tokenizer,sem,rt.receipt,MAX_TOKENS)
                    row.update(logits=out['option_logits'],probabilities=out['probabilities'],tokens=out['input_tokens'],seconds=out['total_seconds'],prompt_sha256=out['prompt_sha256'])
                    codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False)[0] for j in range(len(row['labels']))]
                    with torch.inference_mode():
                        z=torch.nn.functional.linear(rt.capture['hidden'].float(),rt.head.weight[codes].float(),rt.head.bias[codes].float() if rt.head.bias is not None else None)[0].tolist()
                    row['fp32_shadow_logits']=z;row['fp32_shadow_probabilities']=softmax(z)
                else:row.update(ours(inputs[rid]))
                if len(row['probabilities'])!=len(row['labels']) or not all(math.isfinite(x) and 0<=x<=1 for x in row['probabilities']) or abs(sum(row['probabilities'])-1)>1e-6:raise ValueError('Invalid distribution')
                row['status']='ok'
            except Exception as e:
                errors+=1;row.update(status='error',error_type=type(e).__name__,error=str(e)[:400],seconds=time.perf_counter()-start,probabilities=None,logits=None)
            records.append(row)
            with (root/'records.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        emit('progress',shard=shard,completed_tasks=i+1,total_tasks=len(selected))
    switch('base')
    last=ours(fixture)
    if max(abs(x-y) for x,y in zip(first['logits'],last['logits'],strict=True))>2e-5:raise RuntimeError('Base model drift')
    complete={'complete':len(records)==4*len(selected),'shard':shard,'task_ids':selected,'records':len(records),'errors':errors,'data_hash':manifest['data_hash'],'model_revision':MODEL_REV,'source_sha256':sha(Path(__file__).read_bytes()),'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_training_steps':0}
    write(root/'complete.json',complete);emit('shard_complete',**complete)

def audit(run_id):
    root=Path('jevbench-audit');root.mkdir(exist_ok=False);rows,manifest=prepare(root/'assets');idx={r['id']:r for r in rows}
    listing=json.loads(get(f'https://api.github.com/repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100',True))['artifacts'];found={};receipts=[]
    for shard in range(SHARDS):
        items=[v for v in listing if v['name']==f'jevbench-public-shard-{shard}' and not v['expired']]
        if len(items)!=1:raise RuntimeError('Missing or ambiguous result shard '+str(shard))
        item=items[0];b=get(f'https://api.github.com/repos/{REPO}/actions/artifacts/{item["id"]}/zip',True)
        if item.get('digest') and item['digest']!='sha256:'+sha(b):raise ValueError('Result archive checksum mismatch')
        dest=root/'shards'/str(shard);unzip(b,dest);complete=json.loads((dest/'complete.json').read_text())
        if not complete['complete'] or complete['data_hash']!=manifest['data_hash'] or complete['task_ids']!=manifest['partitions'][shard]:raise ValueError('Incomplete/changed population')
        records=[json.loads(line) for line in (dest/'records.jsonl').read_text().splitlines() if line.strip()]
        if len(records)!=complete['records']:raise ValueError('Record count mismatch')
        for p in records:
            key=(p['id'],p['config'])
            if key in found or p['id'] not in complete['task_ids'] or p['config'] not in CONFIGS:raise ValueError('Duplicate or unplanned prediction')
            if p['input_hash']!=digest(public_input(idx[p['id']])) or p['labels']!=idx[p['id']]['labels']:raise ValueError('Input/candidate mismatch')
            if p['status']=='ok':
                ps=softmax(p['logits'])
                if max(abs(a-b) for a,b in zip(ps,p['probabilities'],strict=True))>1e-7:raise ValueError('Probability arithmetic mismatch')
            found[key]=p
        receipts.append({'shard':shard,'artifact_id':item['id'],'sha256':sha(b),'complete':complete})
    if len(found)!=924:raise ValueError('Not all four configurations have all 231 decisions')
    from jevbench.tasks import Task
    from jevbench.scoring import score_task
    eligible=[r for r in rows if r.get('expected') is not None and not r.get('provenance',{}).get('exclude_reason')]
    allconfigs=CONFIGS+['semif_fp32_shadow'];scored={}
    for name in allconfigs:
        scored[name]={}
        for r in rows:
            p=found[(r['id'],'semif_native' if name=='semif_fp32_shadow' else name)]
            probs=p.get('fp32_shadow_probabilities') if name=='semif_fp32_shadow' else p['probabilities']
            probs=dict(zip(r['labels'],probs,strict=True)) if probs is not None else None
            result=score_task(probs,Task.from_dict(r));result['seconds']=p['seconds'];result['tokens']=p.get('tokens');scored[name][r['id']]=result
    def metrics(rr,name):
        ss=[scored[name][r['id']] for r in rr];valid=[(r,p) for r,p in zip(rr,ss) if p['valid']];n=len(rr)
        ece=0.
        for j in range(10):
            group=[(r,p) for r,p in valid if min(int(max(p['probs'].values())*10),9)==j]
            if group:ece+=len(group)/len(valid)*abs(statistics.mean(max(p['probs'].values()) for r,p in group)-statistics.mean(p['correct'] for r,p in group))
        probrows=[(r,p) for r,p in valid if r.get('provenance',{}).get('gold_probs')]
        return {'n':n,'correct':sum(bool(p['correct']) for p in ss),'accuracy':sum(bool(p['correct']) for p in ss)/n if n else None,'valid':len(valid),'invalid_or_error':n-len(valid),'nll_valid':statistics.mean(-math.log(max(p['probs'][str(r['expected'])],1e-30)) for r,p in valid) if valid else None,'brier_valid':statistics.mean(sum((v-int(k==str(r['expected'])))**2 for k,v in p['probs'].items()) for r,p in valid) if valid else None,'ece10_valid':ece if valid else None,'probability_fidelity_n':len(probrows),'probability_tv_valid':statistics.mean(.5*sum(abs(p['probs'][k]-r['provenance']['gold_probs'][k]) for k in r['labels']) for r,p in probrows) if probrows else None,'ties':sum(sum(abs(v-max(p['probs'].values()))<=1e-10 for v in p['probs'].values())>1 for r,p in valid),'median_call_seconds':statistics.median(p['seconds'] for p in ss) if ss else None,'p95_call_seconds':sorted(p['seconds'] for p in ss)[min(n-1,math.ceil(.95*n)-1)] if n else None}
    groups={'all':eligible,**{tier:[r for r in eligible if r['_tier']==tier] for tier in ['easy','standard','hard']}}
    summary={group:{name:metrics(rr,name) for name in allconfigs} for group,rr in groups.items()}
    families={f:{name:metrics([r for r in eligible if r['family']==f],name) for name in allconfigs} for f in sorted({r['family'] for r in eligible})}
    paired={}
    for name in allconfigs[1:]:
        paired[name]={}
        for group,rr in groups.items():
            repairs=sum(not scored['base'][r['id']]['correct'] and scored[name][r['id']]['correct'] for r in rr);harms=sum(scored['base'][r['id']]['correct'] and not scored[name][r['id']]['correct'] for r in rr)
            paired[name][group]={'repairs':repairs,'harms':harms,'net':repairs-harms,'changed_answers':sum(scored['base'][r['id']]['predicted']!=scored[name][r['id']]['predicted'] for r in rr)}
    report={'complete':True,'source_run':run_id,'model_revision':MODEL_REV,'benchmark_revision':BENCH_REV,'semif_revision':SEMIF_REV,'public_tasks':231,'verified_primary_predictions':924,'eligible':len(eligible),'excluded':[r['id'] for r in rows if r not in eligible],'summary':summary,'families':families,'paired_vs_base':paired,'receipt':receipts,'limitations':['Public subset only, not the official 534-decision ranking; no composite score computed','No model/adapter training, calibration, prompt search, or test-based checkpoint choice','SemIf original prompt/readout executed with common CPU loader, not its published CUDA or MLX deployment','SemIf native BF16-head output differs from Duplex FP32 answer-row projection; same-forward FP32 shadow reported separately','Latency is serial within each of 16 separate ARM VMs and includes prompt/tokenization/readout, excludes loading and HTTP; not local Mac or production latency','No price assigned to self-hosted compute; not zero-cost pricing','Public dataset authorship/review and pretraining-contamination limitations remain; all tasks now exposed regression evidence']}
    write(root/'REPORT.json',report);write(root/'scored.json',scored);write(root/'predictions.json',[found[k] for k in sorted(found)]);emit('audit_complete',summary=summary,paired=paired,verified=924)
    # Preserve experiment source without weights or credentials.
    (root/'jevbench_public_v1.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--shard',type=int);ap.add_argument('--audit',type=int);a=ap.parse_args()
    self_test()
    if a.prepare:
        rows,m=prepare('jevbench-prepared');emit('prepared',cases=len(rows),data_hash=m['data_hash'],partition_sizes=[len(x) for x in m['partitions']])
    elif a.shard is not None:run(a.shard)
    elif a.audit is not None:audit(a.audit)
    elif not a.self_test:ap.error('Choose --prepare, --shard, --audit or --self-test')
