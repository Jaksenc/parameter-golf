"""Recompute logged model results; fit only on declared calibration/fit cases."""
from __future__ import annotations
import argparse, base64, hashlib, json, math, os, statistics, struct, urllib.request, zlib
from collections import defaultdict
from pathlib import Path
from decision_probe_v2 import synthetic, relationships, public_input, digest
NAMES=['gliclass-instruct','qwen-4b','qwen-08b']
REPO='Jaksenc/parameter-golf'
def emit(kind,**kw):print(json.dumps(dict(kind=kind,**kw),sort_keys=True,allow_nan=False),flush=True)
def api(path, text=False):
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(req,timeout=90) as f:data=f.read().decode('utf-8-sig')
    return data if text else json.loads(data)
def parse(text):
    result=[]
    for line in text.splitlines():
        pos=line.find('{')
        if pos<0:continue
        try:r=json.loads(line[pos:])
        except json.JSONDecodeError:continue
        if isinstance(r,dict) and r.get('kind'):result.append(r)
    return result

def prob(logits,t=1.):
    x=[v/t for v in logits];m=max(x);v=[math.exp(y-m) for y in x];s=sum(v);return [y/s for y in v]
def choose(p):
    if p is None:return None
    m=max(p);r=[i for i,v in enumerate(p) if abs(v-m)<=1e-10]
    return r[0] if len(r)==1 else None
def nll(rows,ps):return sum(-math.log(max(ps[r['id']][r['target']],1e-30)) for r in rows)/len(rows)
def evaluate(rows,ps):
    out={};index={r['id']:r for r in rows}
    for split in sorted({r['split'] for r in rows}):
        rr=[r for r in rows if r['split']==split];valid=[r for r in rr if ps.get(r['id']) is not None]
        out[split]={'n':len(rr),'correct':sum(choose(ps.get(r['id']))==r['target'] for r in rr),
          'valid_distributions':len(valid),'nll_valid':nll(valid,ps) if valid else None,
          'brier_valid':sum(sum((v-(k==r['target']))**2 for k,v in enumerate(ps[r['id']])) for r in valid)/len(valid) if valid else None}
    pairs=defaultdict(list)
    for kind,a,b in relationships(rows):
        if index[a]['split']=='test':pairs[kind].append((a,b))
    out['paired']={kind:{'n':len(v),'both_correct':sum(choose(ps.get(a))==index[a]['target'] and choose(ps.get(b))==index[b]['target'] for a,b in v)} for kind,v in pairs.items()}
    for kind in ('irrelevant','permutation'):
        v=pairs.get(kind,[]);changed=0;diff=[]
        for a,b in v:
            pa,pb=choose(ps.get(a)),choose(ps.get(b))
            if pa is None or pb is None:continue
            changed+=index[a]['order'][pa]!=index[b]['order'][pb]
            qa={k:v for k,v in zip(index[a]['order'],ps[a])};qb={k:v for k,v in zip(index[b]['order'],ps[b])}
            diff.append(max(abs(qa[k]-qb[k]) for k in qa))
        out['paired'][kind].update(changed_semantic_choice=changed,valid_pair_distributions=len(diff),mean_max_probability_change=statistics.mean(diff) if diff else None)
    out['test_rule_slices']={k:{'n':sum(r['split']=='test' and r['rule'] in fam for r in rows),
        'correct':sum(choose(ps.get(r['id']))==r['target'] for r in rows if r['split']=='test' and r['rule'] in fam)}
        for k,fam in [('atomic',('p','not_p')),('and',('and','nand')),('xor',('xor','xnor'))]}
    return out

def selection(rows,ps):
    idx={r['id']:r for r in rows};pairs=relationships(rows);counts=[]
    for kind in ('rule','evidence'):
        pair=[(a,b) for k,a,b in pairs if k==kind]
        counts.append(sum(choose(ps.get(a))==idx[a]['target'] and choose(ps.get(b))==idx[b]['target'] for a,b in pair)/len(pair))
    accuracy=sum(choose(ps.get(r['id']))==r['target'] for r in rows)/len(rows)
    return (sum(counts)/2,accuracy,-nll(rows,ps))
def mix(allps,weights,ids):
    return {rid:[sum(w*allps[n][rid][k] for n,w in zip(NAMES,weights)) for k in range(len(allps[NAMES[0]][rid]))] for rid in ids}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=int,required=True);a=ap.parse_args()
    jobs=api(f'/actions/runs/{a.run}/jobs?per_page=30')['jobs'];selected={}
    for name in NAMES:
        matches=[j for j in jobs if j['name']==f'probe ({name})']
        if len(matches)!=1 or matches[0]['conclusion']!='success':raise RuntimeError('Incomplete/failed model job: '+name)
        selected[name]=matches[0]
    from benchmark import public_data
    rows=synthetic();public,sources=public_data(16)
    for r in public:r.update(split='regression-'+r['family'],base_id=None)
    rows+=public;idx={r['id']:r for r in rows};ids=list(idx);raw={};logits={};records={};maxdiff=0.;summaries={}
    for name,job in selected.items():
        text=api(f"/actions/jobs/{job['id']}/logs",True);parsed=parse(text)
        protocol=[p for p in parsed if p['kind']=='protocol'];last=[p for p in parsed if p['kind']=='probe_summary']
        if len(protocol)!=1 or len(last)!=1 or protocol[0]['data_hash']!=digest(rows) or last[0]['data_hash']!=digest(rows):raise ValueError('Protocol/data mismatch')
        pred=[p for p in parsed if p['kind']=='probe_prediction']
        if len(pred)!=len(rows) or len({p['id'] for p in pred})!=len(rows) or {p['id'] for p in pred}!=set(idx):raise ValueError('Incomplete or duplicated records')
        for p in pred:
            r=idx[p['id']]
            if p['input_hash']!=digest(public_input(r)):raise ValueError('Input mismatch')
            if p['status'] in ('ok','tie'):
                ps=prob(p['logits']);d=max(abs(x-y) for x,y in zip(ps,p['probabilities'],strict=True));maxdiff=max(maxdiff,d)
                if d>1e-10 or choose(ps)!=p['prediction']:raise ValueError('Arithmetic/prediction mismatch')
            elif p.get('probabilities') is not None or p.get('prediction') is not None:raise ValueError('Invalid failure record')
        raw[name]={p['id']:p['probabilities'] for p in pred};logits[name]={p['id']:p.get('logits') for p in pred};records[name]=pred;summaries[name]=last[0]
        emit('verified_model',model=name,job_id=job['id'],log_sha256=hashlib.sha256(text.encode()).hexdigest(),summary=last[0],
            recomputed=evaluate(rows,raw[name]),assets=[p for p in parsed if p['kind']=='assets'],fp32_readout=[p for p in parsed if p['kind']=='fp32_readout_check'])
    cal=[r for r in rows if r['split']=='calibration'];fit=[r for r in rows if r['split']=='fit']
    if any(raw[n][r['id']] is None for n in NAMES for r in rows):raise RuntimeError('Model error present; do not fit by dropping cases')
    temps={};calibrated={}
    for name in NAMES:
        grid=[.25*(32**(i/60)) for i in range(61)]
        best=min(grid,key=lambda t:nll(cal,{r['id']:prob(logits[name][r['id']],t) for r in cal}))
        temps[name]=best;calibrated[name]={rid:prob(logits[name][rid],best) for rid in ids}
    candidates=[]
    for i in range(11):
      for j in range(11-i):
        weights=[i/10,j/10,(10-i-j)/10];ps=mix(calibrated,weights,[r['id'] for r in fit])
        candidates.append((selection(fit,ps),weights))
    fit_score,weights=max(candidates,key=lambda v:v[0]);static=mix(calibrated,weights,ids)
    chosen_single=max(NAMES,key=lambda n:selection(fit,calibrated[n]))
    configs={**{'raw-'+n:raw[n] for n in NAMES},**{'temperature-'+n:calibrated[n] for n in NAMES},
       'uniform_raw':mix(raw,[1/3]*3,ids),'uniform_calibrated':mix(calibrated,[1/3]*3,ids),'fit_static':static}
    target=selection(fit,calibrated['qwen-4b'])[:2];cascades=[]
    for threshold in (0.6,0.7,0.8,0.9,0.95,0.99,1.01):
        ps={rid:(calibrated['qwen-08b'][rid] if max(calibrated['qwen-08b'][rid])>=threshold else calibrated['qwen-4b'][rid]) for rid in ids}
        score=selection(fit,ps);calls=sum(max(calibrated['qwen-08b'][r['id']])<threshold for r in fit)
        if score[0]>=target[0] and score[1]>=target[1]:cascades.append((calls,-score[0],-score[1],threshold,ps))
    cascade=min(cascades,key=lambda x:x[:4]);configs['fit_cascade']=cascade[4]
    test=[r for r in rows if r['split']=='test'];union={};conditional={}
    for pop,rr in [('test',test),('regression',[r for r in rows if r['split'].startswith('regression-')])]:
        union[pop]={'n':len(rr),'oracle_top_choice_union':sum(any(choose(raw[n][r['id']])==r['target'] for n in NAMES) for r in rr)}
        qwrong=[r for r in rr if choose(raw['qwen-4b'][r['id']])!=r['target']]
        conditional[pop]={'qwen_4b_errors':len(qwrong),**{n+'_rescues':sum(choose(raw[n][r['id']])==r['target'] for r in qwrong) for n in ('gliclass-instruct','qwen-08b')}}
    accepted=[r for r in test if choose(raw['qwen-4b'][r['id']]) is not None and choose(raw['qwen-4b'][r['id']])==choose(raw['qwen-08b'][r['id']])]
    emit('audit_result',status='complete',data_hash=digest(rows),verified_predictions=3*len(rows),maximum_probability_discrepancy=maxdiff,
      temperatures=temps,static_weights=dict(zip(NAMES,weights)),static_fit_selection=fit_score,fit_selected_single=chosen_single,
      cascade_threshold=cascade[3],cascade_fit_large_calls=cascade[0],cascade_test_large_calls=sum(max(calibrated['qwen-08b'][r['id']])<cascade[3] for r in test),
      results={k:evaluate(rows,v) for k,v in configs.items()},oracle=union,conditional_recovery=conditional,
      two_qwen_consensus={'population':144,'accepted':len(accepted),'correct':sum(choose(raw['qwen-4b'][r['id']])==r['target'] for r in accepted)},
      limitations=['6 correlated source blocks per synthetic cohort','no independent human adjudication','synthetic calibration does not guarantee real-text calibration','parallel VMs are not a matched hardware speed benchmark','this fits a static comparison, not private Duplex code'])
    synth=synthetic()
    for name in NAMES:
        values=[x for r in synth for x in logits[name][r['id']]]
        data=struct.pack('<'+'f'*len(values),*values);payload=base64.b64encode(zlib.compress(data,9)).decode()
        emit('portable_logits',model=name,encoding='zlib-base64-little-endian-float32',shape=[len(synth),6],row_order_hash=digest([r['id'] for r in synth]),uncompressed_sha256=hashlib.sha256(data).hexdigest(),revision=summaries[name]['revision'],checkpoint=summaries[name]['checkpoint'],payload=payload)
if __name__=='__main__':main()
