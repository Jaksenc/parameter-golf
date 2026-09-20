"""Analysis frozen before the first intervention run. No model fitting or training.
Select one correction policy on development; certify on separate source groups.
Certification is scoped to the diagnostic's population, never arbitrary products.
"""
from __future__ import annotations
import base64,hashlib,json,math,os,statistics,zlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,beta
from intervention_probe_v1 import data,digest,REV,VIEW_NAMES

def emit(kind,**kw):print(json.dumps(dict(kind=kind,**kw),sort_keys=True,allow_nan=False),flush=True)
def softmax(x):
    x=np.asarray(x,float);v=np.exp(x-x.max());return v/v.sum()
def kl(p,q):return float(np.sum(p*(np.log(np.maximum(p,1e-30))-np.log(np.maximum(q,1e-30)))))
def project(base,proposal,budget):
    if kl(proposal,base)<=budget:return proposal,1.
    lo,hi=0.,1.
    for _ in range(60):
        t=(lo+hi)/2;p=softmax((1-t)*np.log(base)+t*np.log(proposal))
        if kl(p,base)<=budget:lo=t
        else:hi=t
    return softmax((1-lo)*np.log(base)+lo*np.log(proposal)),lo

def alternatives(z):
    p=softmax(z[0]);group=softmax(np.mean(z[:4],axis=0))
    d={'baseline':p,'complement2':softmax((z[0]+z[2])/2),'order2':softmax((z[0]+z[1])/2),'orbit4':group,'order4':softmax(np.mean(z[[0,1,4,5]],axis=0)),'order4_arithmetic':np.mean([softmax(v) for v in z[[0,1,4,5]]],axis=0)}
    # Correct only when two separately paired complementary views favor the same meaning.
    a=int(np.argmax(z[0]+z[2]));b=int(np.argmax(z[1]+z[3]))
    sorted_scores=np.sort(np.mean(z[:4],axis=0));margin=float(sorted_scores[-1]-sorted_scores[-2])
    for radius in (.05,.2,.5):
        q,_=project(p,group,radius)
        for threshold in (0.,.25,.75):
            key=f'trust_{radius:g}_{threshold:g}'
            d[key]=q if a==b==int(np.argmax(q)) and margin>=threshold else p.copy()
    return d

def score(rows,ps,base):
    gold=np.array([r['target'] for r in rows]);p=np.asarray(ps);b=np.asarray(base)
    pred=p.argmax(axis=1);bp=b.argmax(axis=1);correct=pred==gold;bc=bp==gold
    # Ties are deliberately failures, not silently first-choice decisions.
    ties=np.sum(np.isclose(p,p.max(axis=1,keepdims=True),atol=1e-12,rtol=0),axis=1)>1
    bties=np.sum(np.isclose(b,b.max(axis=1,keepdims=True),atol=1e-12,rtol=0),axis=1)>1
    correct&=~ties;bc&=~bties
    return {'n':len(rows),'correct':int(correct.sum()),'repairs':int(np.sum(correct&~bc)),'harms':int(np.sum(~correct&bc)),'changes':int(np.sum(pred!=bp)),'ties':int(ties.sum()),'nll':float(np.mean(-np.log(np.maximum(p[np.arange(len(rows)),gold],1e-30)))),'brier':float(np.mean(np.sum((p-np.eye(3)[gold])**2,axis=1)))}

def compare(rows,z,seconds):
    pp=[alternatives(x) for x in z];configs={k:np.array([p[k] for p in pp]) for k in pp[0]}
    base=configs['baseline'];dev=np.array([i for i,r in enumerate(rows) if r['split']=='development']);cert=np.array([i for i,r in enumerate(rows) if r['split']=='certification']);test=np.array([i for i,r in enumerate(rows) if r['split']=='evaluation'])
    devrows=[rows[i] for i in dev];candidates=['baseline']+[k for k in configs if k.startswith('trust_')]
    def objective(k):
        s=score(devrows,configs[k][dev],base[dev]);return (s['correct'],-s['harms'],k=='baseline',-s['nll'])
    chosen=max(candidates,key=objective)
    cr=[rows[i] for i in cert];cm=score(cr,configs[chosen][cert],base[cert]);wins,losses=cm['repairs'],cm['harms'];discordant=wins+losses
    # Exact one-sided paired sign test; chosen policy was selected without these labels.
    pv=float(binomtest(wins,discordant,.5,alternative='greater').pvalue) if discordant else 1.
    upper_harm=float(beta.ppf(.95,losses+1,len(cert)-losses)) if losses<len(cert) else 1.
    # Gate additionally requires no measured probability-loss regression on certification.
    bcert=score(cr,base[cert],base[cert]);passed=chosen!='baseline' and pv<=.05 and cm['nll']<=bcert['nll']
    deploy=configs[chosen].copy() if passed else base.copy();configs['certified_policy']=deploy
    groups={**{name:[i for i,r in enumerate(rows) if r['split']==name] for name in ['development','certification','evaluation']},'all':list(range(len(rows)))}
    report={}
    for group,ii in groups.items():
        rr=[rows[i] for i in ii];report[group]={k:score(rr,p[ii],base[ii]) for k,p in configs.items()}
    families={f:{k:score([r for r in rows if r['family']==f and r['split']=='evaluation'],p[[i for i,r in enumerate(rows) if r['family']==f and r['split']=='evaluation']],base[[i for i,r in enumerate(rows) if r['family']==f and r['split']=='evaluation']]) for k,p in configs.items() if k in ['baseline','complement2','order2','orbit4','order4','certified_policy']} for f in ['boolq','paws']}
    groups_by_method={'baseline':[0],'complement2':[0,2],'order2':[0,1],'orbit4':[0,1,2,3],'order4':[0,1,4,5]}
    timing={k:{'calls':len(ii),'p50_seconds':float(np.median(seconds[:,ii].sum(axis=1))),'p95_seconds':float(np.quantile(seconds[:,ii].sum(axis=1),.95))} for k,ii in groups_by_method.items()}
    return {'selected_on_development':chosen,'certification':{'passed_for_diagnostic_population_only':bool(passed),'production_promoted':False,'n':len(cert),'repairs':wins,'harms':losses,'paired_pvalue':pv,'unconditional_harm_rate_upper95':upper_harm,'nll_guard_passed':cm['nll']<=bcert['nll'],'scope':'fixed BoolQ/PAWS derived decision diagnostic; independent-source/iid assumption not proven'},'metrics':report,'evaluation_families':families,'timing':timing,'timing_qualification':'Sum of measured sequential forward/tokenization call durations on one shard per case; excludes loading, proof checks and transport. Counts match; token lengths need not match. Not Mac or service latency.'}

def main():
    import requests
    from urllib.parse import urlparse
    repo='Jaksenc/parameter-golf';run=int(os.environ['SOURCE_RUN_ID']);token=os.environ['GH_TOKEN']
    def get(path,text=False):
        r=requests.get('https://api.github.com/repos/'+repo+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'},timeout=90)
        r.raise_for_status();return r.content.decode('utf-8-sig') if text else r.json()
    jobs=get(f'/actions/runs/{run}/jobs?per_page=30')['jobs'];jobs=[j for j in jobs if j['name'].startswith('inference (')]
    if len(jobs)!=6 or any(j['status']!='completed' or j['conclusion']!='success' for j in jobs):raise RuntimeError('Six successful inference shards required')
    rows,sources=data();byid={r['id']:i for i,r in enumerate(rows)};records={};receipts=[]
    for job in jobs:
        text=get(f"/actions/jobs/{job['id']}/logs?audit=intervention-v1",True);parsed=[]
        for line in text.splitlines():
            p=line.find('{')
            if p<0:continue
            try:v=json.loads(line[p:])
            except json.JSONDecodeError:continue
            if isinstance(v,dict) and 'kind' in v:parsed.append(v)
        prot=[p for p in parsed if p['kind']=='protocol'];complete=[p for p in parsed if p['kind']=='complete']
        if len(prot)!=1 or len(complete)!=1 or prot[0]['data_hash']!=digest(rows) or complete[0]['data_hash']!=digest(rows) or complete[0]['revision']!=REV:raise ValueError('Changed data/incomplete model')
        shard=prot[0]['shard'];batch=[p for p in parsed if p['kind']=='orbit_case']
        expected=[i for i in range(128) if i%6==shard]
        if [p['index'] for p in batch]!=expected or len(batch)*6!=complete[0]['calls']:raise ValueError('Incomplete or reordered partition')
        for p in batch:
            i=byid[p['id']];r=rows[i]
            if p['id'] in records or p['index']!=i or p['target']!=r['target'] or p['group']!=r['group']:raise ValueError('Identity mismatch')
            if [v['view'] for v in p['views']]!=VIEW_NAMES:raise ValueError('Wrong views')
            for v,source in zip(p['views'],r['views'],strict=True):
                if v['status']!='ok' or v['input_hash']!=digest(source['request']):raise ValueError('Failed or mismatched view')
            records[p['id']]=p
        receipts.append({'job':job['id'],'log_sha256':hashlib.sha256(text.encode()).hexdigest(),'protocol':prot[0],'completion':complete[0]})
    if len(records)!=128:raise ValueError('Missing source cases')
    z=np.array([[v['logits'] for v in records[r['id']]['views']] for r in rows]);seconds=np.array([[v['seconds'] for v in records[r['id']]['views']] for r in rows]);out=compare(rows,z,seconds)
    emit('intervention_result',run=run,model_revision=REV,data_hash=digest(rows),verified_calls=int(z.shape[0]*z.shape[1]),results=out,receipts=receipts)
    raw=z.astype('<f4').tobytes();emit('portable_logits',shape=list(z.shape),dtype='little-endian-float32',sha256=hashlib.sha256(raw).hexdigest(),payload=base64.b64encode(zlib.compress(raw,9)).decode(),row_metadata=[{k:r[k] for k in ['id','family','group','split','target']} for r in rows])
if __name__=='__main__':main()
