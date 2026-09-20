"""Read-only audit of public pretrained logs; no model inference or private inputs.
Fits temperatures and small comparison pools, never backbone parameters.
Frozen protocol: calibration (48) -> fitting (48) -> reporting; no test tuning.
The confidence-feature pool is an independent implementation, not private source.
"""
from __future__ import annotations
import hashlib, json, math, os, statistics
from pathlib import Path
NAMES = ['gliclass-instruct', 'qwen-4b', 'qwen-08b']
JOBS = {'gliclass-instruct':[106099011148], 'qwen-4b':[106103489901,106103489713,106103489947,106103489833], 'qwen-08b':[106099011005]}
REVISIONS = {'gliclass-instruct':'825e5478c1bf4bffbf297690517097ccbdb2e006', 'qwen-4b':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'qwen-08b':'2fc06364715b967f1860aea9cf38778875588b17'}
DATA_HASH = '4309b5304f50e5f740a334f3bd8d7f8ac6a4cd9761a8822caddb843575131f0d'
CODE_HASH = 'b7267efe6fb8431b303490bafba0986bf899bb21c3b4677f90d36ff1de4695ef'
REPO = 'Jaksenc/parameter-golf'
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
def emit(kind, **data):
    print(json.dumps({'kind':kind,**data},sort_keys=True,allow_nan=False),flush=True)
def softmax(z):
    m=max(z); v=[math.exp(x-m) for x in z]; s=sum(v)
    return [x/s for x in v]
def choose(p):
    m=max(p); tied=[i for i,x in enumerate(p) if abs(x-m)<=1e-10]
    return tied[0] if len(tied)==1 else None
def nll(rows, probs):
    return statistics.mean(-math.log(max(probs[r['id']][r['target']],1e-30)) for r in rows)
def pairs(rows):
    lookup={r['id']:r for r in rows}; result=[]
    for r in rows:
        if r.get('base_id'):
            if r['base_id'] in lookup: result.append((r['split'],r['base_id'],r['id']))
            continue
        if r['split'] not in ('calibration','fit','test'): continue
        for a,b in [('p','not_p'),('and','nand'),('xor','xnor')]:
            other=f"{r['block']}-{int(r['p'])}{int(r['q'])}-{b}"
            if r['rule']==a and other in lookup: result.append(('rule',r['id'],other))
        for axis in ('p','q'):
            if r[axis]: continue
            p=True if axis=='p' else r['p']; q=True if axis=='q' else r['q']
            other=f"{r['block']}-{int(p)}{int(q)}-{r['rule']}"
            if other in lookup and r['target']!=lookup[other]['target']: result.append(('evidence',r['id'],other))
    return result
def selection(rows, probs):
    lookup={r['id']:r for r in rows}; pp=pairs(rows)
    acc=statistics.mean(choose(probs[r['id']])==r['target'] for r in rows)
    rates=[]
    for kind in ('rule','evidence'):
        matched=[(a,b) for k,a,b in pp if k==kind]
        rates.append(statistics.mean(choose(probs[a])==lookup[a]['target'] and choose(probs[b])==lookup[b]['target'] for a,b in matched))
    return (statistics.mean(rates),acc,-nll(rows,probs))
def evaluate(rows,probs):
    lookup={r['id']:r for r in rows}; decision={rid:choose(p) for rid,p in probs.items()}; out={}
    for split in sorted({r['split'] for r in rows}):
        sub=[r for r in rows if r['split']==split]
        out[split]={'n':len(sub),'correct':sum(decision[r['id']]==r['target'] for r in sub),'ties':sum(decision[r['id']] is None for r in sub),'nll':nll(sub,probs),'brier':statistics.mean(sum((v-int(i==r['target']))**2 for i,v in enumerate(probs[r['id']])) for r in sub)}
    out['pairs']={}
    for kind in ('rule','evidence','irrelevant','permutation'):
        matched=[(a,b) for k,a,b in pairs(rows) if k==kind and lookup[a]['split']=='test']
        both=sum(decision[a]==lookup[a]['target'] and decision[b]==lookup[b]['target'] for a,b in matched)
        out['pairs'][kind]={'n':len(matched),'both_correct':both}
        if kind in ('irrelevant','permutation'):
            valid=[(a,b) for a,b in matched if decision[a] is not None and decision[b] is not None]
            out['pairs'][kind].update(valid=len(valid),changed=sum(lookup[a]['order'][decision[a]]!=lookup[b]['order'][decision[b]] for a,b in valid))
    out['rule_slices']={}
    for group,rules in {'atomic':('p','not_p'),'and':('and','nand'),'xor':('xor','xnor')}.items():
        sub=[r for r in rows if r['split']=='test' and r['rule'] in rules]
        out['rule_slices'][group]={'n':len(sub),'correct':sum(decision[r['id']]==r['target'] for r in sub)}
    sub=[r for r in rows if r['split'].startswith('regression-')]
    out['public_regression']={'n':len(sub),'correct':sum(decision[r['id']]==r['target'] for r in sub),'nll':nll(sub,probs)}
    out['test_blocks']=[sum(decision[r['id']]==r['target'] for r in rows if r['split']=='test' and r['block']==f'test-{i}') for i in range(6)]
    return out
def parse(text):
    found=[]
    for line in text.splitlines():
        p=line.find('{')
        if p<0: continue
        try: row=json.loads(line[p:])
        except json.JSONDecodeError: continue
        if isinstance(row,dict) and 'kind' in row: found.append(row)
    return found
def api(path,text=False):
    import requests
    r=requests.get('https://api.github.com/repos/'+REPO+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},timeout=90)
    r.raise_for_status()
    return r.content.decode('utf-8-sig') if text else r.json()
def validate_records(rows,records):
    ids={r['id'] for r in rows}
    if len(records)!=len(rows) or {p['id'] for p in records}!=ids or len({p['id'] for p in records})!=len(records): raise ValueError('Missing, duplicate or unexpected model records')
    byid={r['id']:r for r in rows}; max_difference=0.
    for p in records:
        r=byid[p['id']]; inp={k:r[k] for k in ('evidence','question','options')}
        if p['input_hash']!=digest(inp) or p['status'] not in ('ok','tie'): raise ValueError('Invalid input or model error; do not drop cases')
        z=p['logits']; ps=p['probabilities']
        if len(z)!=len(r['options']) or len(ps)!=len(z) or not all(math.isfinite(x) for x in z+ps): raise ValueError('Invalid scores')
        calc=softmax(z); diff=max(abs(x-y) for x,y in zip(calc,ps)); max_difference=max(max_difference,diff)
        if diff>1e-10 or choose(calc)!=p['prediction']: raise ValueError('Probability/decision mismatch')
    return max_difference
def load_records(rows):
    all_records={}; receipts=[]
    for name,jobs in JOBS.items():
        records=[]
        for shard,job in enumerate(jobs):
            meta=api(f'/actions/jobs/{job}')
            if meta['status']!='completed' or meta['conclusion']!='success': raise ValueError('Incomplete source job')
            text=api(f'/actions/jobs/{job}/logs?audit=duplex-final-v1',True); parsed=parse(text)
            proto=[p for p in parsed if p['kind']=='protocol']
            if len(proto)!=1 or proto[0]['data_hash']!=DATA_HASH or proto[0]['code_sha256']!=CODE_HASH: raise ValueError('Changed source protocol')
            asset=[p for p in parsed if p['kind']=='assets']; load=[p for p in parsed if p['kind']=='loading']
            if len(asset)!=1 or asset[0]['revision']!=REVISIONS[name] or len(load)!=1 or load[0]['issues']: raise ValueError('Checkpoint identity or loading issue')
            pp=[p for p in parsed if p['kind']=='probe_prediction']
            expected=list(range(shard,len(rows),4)) if len(jobs)==4 else list(range(len(rows)))
            if [p['index'] for p in pp]!=expected or any(p['id']!=rows[p['index']]['id'] or p['model']!=name for p in pp): raise ValueError('Partition or row identity mismatch')
            terminal=[p for p in parsed if p['kind'] in ('probe_summary','shard_summary')]
            if len(terminal)!=1 or terminal[0]['data_hash']!=DATA_HASH: raise ValueError('Missing completion receipt')
            if len(jobs)==4:
                ex=[p for p in parsed if p['kind']=='shard_execution']
                if len(ex)!=1 or ex[0]['shard_index']!=shard or ex[0]['shard_count']!=4 or not ex[0]['native_score_functions_ast_unchanged']: raise ValueError('Invalid shard receipt')
            receipts.append({'model':name,'job':job,'run':meta['run_id'],'log_sha256':hashlib.sha256(text.encode()).hexdigest(),'asset_revision':asset[0]['revision'],'weights_sha256':{k:v['sha256'] for k,v in asset[0]['files'].items() if k.endswith('.safetensors')},'rss_kib':terminal[0]['peak_process_rss_kib'],'fp32_checks':[p for p in parsed if p['kind']=='fp32_readout_check']})
            records.extend(pp)
        validate_records(rows,records); all_records[name]={p['id']:p for p in records}
    return all_records,receipts
def pooled(probs,weights,rows):
    return {r['id']:[sum(w*probs[n][r['id']][j] for n,w in zip(NAMES,weights)) for j in range(len(r['options']))] for r in rows}
def feature_pool(calibrated,fit,rows,adaptive):
    """Independent standard softmax-mixture baseline: entropy, margin and KL.
    Fixed before this run: 400 Adam steps, lr=.03, L2=.03; no seed search.
    """
    import numpy as np
    def features(p):
        p=np.asarray(p,dtype=float); entropy=-np.sum(p*np.log(np.clip(p,1e-12,1)),axis=1)/math.log(p.shape[1])
        order=np.sort(p,axis=1); margin=order[:,-1]-order[:,-2]
        divergence=np.sum(p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(np.mean(p,axis=0),1e-12,1))),axis=1)
        return np.concatenate((entropy,margin,divergence))
    train_prob=[np.array([calibrated[n][r['id']] for n in NAMES]) for r in fit]
    f=np.array([features(p) for p in train_prob]); mu=f.mean(axis=0); scale=np.maximum(f.std(axis=0),.05)
    normalized=np.clip((f-mu)/scale,-8,8) if adaptive else np.zeros_like(f)
    x=np.column_stack((normalized,np.ones(len(fit)))); gold=np.array([p[:,r['target']] for p,r in zip(train_prob,fit)])
    w=np.zeros((10,3)); first=np.zeros_like(w); second=np.zeros_like(w)
    for step in range(1,401):
        score=x@w; score-=score.max(axis=1,keepdims=True); weight=np.exp(score); weight/=weight.sum(axis=1,keepdims=True)
        p_gold=np.maximum(np.sum(weight*gold,axis=1),1e-12)
        dz=weight*(1-gold/p_gold[:,None])/len(fit)
        reg=w.copy(); reg[-1,:]=0
        gradient=x.T@dz+.03*reg
        first=.9*first+.1*gradient; second=.999*second+.001*gradient**2
        w-=.03*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
    output={}; outweights={}
    for r in rows:
        p=np.array([calibrated[n][r['id']] for n in NAMES]); f=features(p)
        f=np.clip((f-mu)/scale,-8,8) if adaptive else np.zeros_like(f)
        weight=np.array(softmax(np.append(f,1.)@w)); output[r['id']]=(weight@p).tolist();outweights[r['id']]=weight.tolist()
    return output, {'adaptive':adaptive,'steps':400,'lr':.03,'l2':.03,'matrix':w.tolist(),'mean':mu.tolist(),'scale':scale.tolist(),'fit_groups':sorted({r['block'] for r in fit}),'test_mean_weights':np.mean([outweights[r['id']] for r in rows if r['split']=='test'],axis=0).tolist()}
def compare(rows,records):
    raw={n:{rid:p['probabilities'] for rid,p in pp.items()} for n,pp in records.items()}
    cal=[r for r in rows if r['split']=='calibration']; fit=[r for r in rows if r['split']=='fit']
    if {r['block'] for r in cal}&{r['block'] for r in fit}: raise ValueError('Split overlap')
    temps={}; calibrated={}
    for n in NAMES:
        grid=[.25*32**(i/60) for i in range(61)]
        t=min(grid,key=lambda t:nll(cal,{r['id']:softmax([v/t for v in records[n][r['id']]['logits']]) for r in cal}))
        temps[n]=t; calibrated[n]={rid:softmax([v/t for v in p['logits']]) for rid,p in records[n].items()}
    candidates=[]
    for i in range(11):
        for j in range(11-i):
            w=[i/10,j/10,(10-i-j)/10]
            candidates.append((selection(fit,pooled(calibrated,w,fit)),w))
    objective,weights=max(candidates,key=lambda z:z[0])
    configs={**{'raw-'+n:raw[n] for n in NAMES},**{'temperature-'+n:calibrated[n] for n in NAMES},'uniform-raw':pooled(raw,[1/3]*3,rows),'uniform-temperature':pooled(calibrated,[1/3]*3,rows),'fit-grid-pool':pooled(calibrated,weights,rows)}
    checkpoints={}
    for adaptive in (False,True):
        key='feature-gate' if adaptive else 'fit-continuous-pool'
        configs[key],checkpoints[key]=feature_pool(calibrated,fit,rows,adaptive)
    baseline=selection(fit,calibrated['qwen-4b']); eligible=[]
    for threshold in (.6,.7,.8,.9,.95,.99,1.01):
        ps={r['id']:calibrated['qwen-08b' if max(calibrated['qwen-08b'][r['id']])>=threshold else 'qwen-4b'][r['id']] for r in rows}
        val=selection(fit,ps); calls=sum(max(calibrated['qwen-08b'][r['id']])<threshold for r in fit)
        if val[0]>=baseline[0] and val[1]>=baseline[1]: eligible.append((calls,-val[0],-val[1],threshold,ps))
    cascade=min(eligible,key=lambda x:x[:4]); configs['fit-cascade']=cascade[4]
    test=[r for r in rows if r['split']=='test']; union={}
    for label,sub in [('test',test),('regression',[r for r in rows if r['split'].startswith('regression-')])]:
        wrong=[r for r in sub if choose(raw['qwen-4b'][r['id']])!=r['target']]
        union[label]={'n':len(sub),'oracle_top1_union':sum(any(choose(raw[n][r['id']])==r['target'] for n in NAMES) for r in sub),'qwen_errors':len(wrong),'gliclass_rescues':sum(choose(raw['gliclass-instruct'][r['id']])==r['target'] for r in wrong),'small_qwen_rescues':sum(choose(raw['qwen-08b'][r['id']])==r['target'] for r in wrong)}
    result={'data_hash':digest(rows),'temperatures':temps,'grid_weights':dict(zip(NAMES,weights)),'grid_fit_objective':objective,'fit_selected_single':max(NAMES,key=lambda n:selection(fit,calibrated[n])),'cascade_threshold':cascade[3],'cascade_fit_large_calls':cascade[0],'cascade_test_large_calls':sum(max(calibrated['qwen-08b'][r['id']])<cascade[3] for r in test),'results':{k:evaluate(rows,p) for k,p in configs.items()},'oracle':union,'checkpoints':checkpoints}
    decisions={k:''.join('-' if choose(p[r['id']]) is None else str(choose(p[r['id']])) for r in rows) for k,p in configs.items()}
    return result,decisions
def main():
    from decision_probe_v2 import synthetic
    from benchmark import public_data
    if hashlib.sha256(Path('decision_probe_v2.py').read_bytes()).hexdigest()!=CODE_HASH: raise ValueError('Changed inference source')
    rows=synthetic(); public,sources=public_data(16)
    for r in public: r.update(split='regression-'+r['family'],base_id=None)
    rows+=public
    if digest(rows)!=DATA_HASH: raise ValueError('Dataset drift; do not use new rows')
    records,receipts=load_records(rows)
    emit('verification',predictions=1200,maximum_probability_difference=max(validate_records(rows,list(p.values())) for p in records.values()),data_hash=digest(rows),receipts=receipts)
    results,decisions=compare(rows,records)
    emit('fusion_audit',**results)
    emit('decision_vectors',row_ids=[r['id'] for r in rows],targets=''.join(str(r['target']) for r in rows),vectors=decisions)
    if os.environ.get('AUDIT_LOCAL_OUTPUT'):
        out=Path(os.environ['AUDIT_LOCAL_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
        for name,data in [('rows',rows),('records',records),('results',results),('receipts',receipts),('decisions',decisions)]:
            (out/(name+'.json')).write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False))
if __name__=='__main__': main()
