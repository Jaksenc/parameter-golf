"""Bridge v15: locate evidence/calculation/readout failures before adaptation.
No weight updates, Jev calls, oracle repair, or answer-based routing.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random
from fractions import Fraction
from pathlib import Path
import measure_v14 as parent

SEED=1509202622
SHARDS=16
FAMILIES=('counts','conditional','mixture','bayes')
SUPPORTS=('source_only','model_draft','selected_evidence','oracle')
CHANNELS=('legacy_codes','numeric_mass')
INPUT_KEYS=('id','state','question','labels')

def digest(x):return parent.digest(x)
def write(p,x):parent.write(p,x)
def sha(p):return parent.filehash(p)

def distribution(w):
    a=w['a']; total=sum(a)
    if w['family'] in ('counts','conditional'):v=[Fraction(x,total) for x in a]
    elif w['family']=='mixture':
        v=[Fraction(w['weight'],10)*Fraction(x,total)+Fraction(10-w['weight'],10)*Fraction(y,sum(w['b'])) for x,y in zip(a,w['b'])]
    else:
        weights=[x*y for x,y in zip(a,w['rates'])];v=[Fraction(x,sum(weights)) for x in weights]
    if sum(v)!=1 or min(v)<0:raise ValueError('Invalid law')
    return v

def render(w):
    labels=w['labels']; items=lambda a:', '.join(f'{s}={n}' for s,n in zip(labels,a))
    family=w['family']
    if family=='counts':
        state='The working urn contains '+items(w['a'])+'. A sealed reserve urn contains '+items(w['noise'])+'. Select one ticket uniformly from the working urn ONLY; never use the reserve urn. Counts are exact and exhaustive.'
        evidence='Selected source facts only: working urn counts '+items(w['a'])+'. The reserve is excluded. Normalize these counts over their sum.'
    elif family=='conditional':
        state='Colored tokens are divided into ACCEPTED and REJECTED groups. ACCEPTED counts: '+items(w['a'])+'. REJECTED counts: '+items(w['noise'])+'. Select uniformly among ACCEPTED tokens only. Counts are exact and exhaustive.'
        evidence='Relevant selected stratum: ACCEPTED counts '+items(w['a'])+'. REJECTED tokens are not eligible. Normalize selected counts only.'
    elif family=='mixture':
        state=f'Select urn L with exact probability {w["weight"]}/10, otherwise select urn R. Then sample uniformly inside the chosen urn. L counts: '+items(w['a'])+'. R counts: '+items(w['b'])+'. An unrelated unopened urn has '+items(w['noise'])+'. All counts are exact. The selected urn and ticket have not been observed.'
        evidence=f'Relevant raw inputs: L selection weight {w["weight"]}/10; R selection weight {10-w["weight"]}/10. L counts '+items(w['a'])+'. R counts '+items(w['b'])+'. Normalize within each urn before combining with the selection weights.'
    else:
        state='Initially select uniformly from a finite population with class counts '+items(w['a'])+'. Conditional on its class, an item passes an assay with these exact probabilities: '+', '.join(f'{s}={n}/10' for s,n in zip(labels,w['rates']))+'. The selected item is known to have PASSED; its class is unknown. A separate untested population has '+items(w['noise'])+'. All counts and rates are exact. Condition on this item passing, not on selection from the unrelated population.'
        evidence='Selected raw inputs: initial class counts '+items(w['a'])+'; pass likelihood numerators out of ten '+items(w['rates'])+'. Observation: PASSED. Multiply each prior count by its class likelihood then normalize the products. No probabilities have been calculated in this note.'
    return state,evidence

def worlds(seed=SEED):
    rng=random.Random(seed);out=[]
    for family in FAMILIES:
        for j,k in enumerate((2,3,4,5,3,4)):
            for attempt in range(1000):
                w={'world_id':f'{family}-{j:02d}','family':family,
                   'labels':rng.sample(['amber','jade','violet','cobalt','silver','ochre','ivory'],k),
                   'a':[rng.randint(1,35) for _ in range(k)],'b':[rng.randint(1,29) for _ in range(k)],
                   'noise':[rng.randint(1,80) for _ in range(k)],'rates':[rng.randint(1,9) for _ in range(k)],'weight':rng.randint(1,9)}
                if j%3==0:w['a'][0]=0
                q=distribution(w)
                if q.count(max(q))==1:break
            else:raise ValueError('No unique mode')
            state,evidence=render(w)
            w.update(state=state,evidence=evidence,event_target=[str(x) for x in q])
            out.append(w)
    if len({w['state'] for w in out})!=24:raise ValueError('Duplicate world')
    return out

def task(w,mode):
    q=distribution(w);modal=max(range(len(q)),key=lambda i:q[i])
    instruction=('Report the full distribution of the COLOR OF THE RANDOM SELECTED ITEM under the stated conditioning. Its color is unobserved. Each label is a possible event, not your confidence about which color is most likely.' if mode=='event' else
                 'Which color has the UNIQUE HIGHEST conditional selection probability under the exact mechanism? Report a distribution over the CORRECT ANSWER to this most-likely-color question, not the distribution of a random draw. Exact parameters and a unique mode are given.')
    row={'id':f'bridge-{w["world_id"]}-{mode}','state':w['state'],'question':{'type':'choice','instructions':instruction,'criteria':{s:f'The color is {s}.' for s in w['labels']}},'labels':w['labels']}
    row=parent.request(row)
    oracle='Reference calculation, supplied intentionally: '+ '; '.join(f'P(color={s})={x.numerator}/{x.denominator} (approximately {float(x):.10f})' for s,x in zip(w['labels'],q))+f'. The unique most probable color is {w["labels"][modal]}. Event probabilities and the identity of their mode are different quantities.'
    target=q if mode=='event' else [Fraction(int(i==modal)) for i in range(len(q))]
    return {'input':row,'world_id':w['world_id'],'family':w['family'],'mode':mode,'target':[str(x) for x in target],'evidence':w['evidence'],'oracle':oracle}

def prepare(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ws=worlds();rows=[task(w,m) for w in ws for m in ('event','mode')]
    old=json.loads((root/'measure-prepared/evaluation.json').read_text())
    old_keys={digest({k:r[k] for k in ('state','question','labels')}) for r in old}
    if any(digest({k:r['input'][k] for k in ('state','question','labels')}) in old_keys for r in rows):raise ValueError('Old overlap')
    jobs=[[] for _ in range(SHARDS)]
    for i,r in enumerate(rows):jobs[i%SHARDS].append({k:r[k] for k in ('input','evidence','oracle')})
    anchors=json.loads((root/'measure-prepared/anchors.json').read_text())[:SHARDS]
    manifest={'version':'bridge-v15.0','seed':SEED,'source_sha256':sha(__file__),'worlds':24,'tasks':48,'views':192,
              'ordinary_generations':48,'numeric_generations':192,'code_readouts':192,
              'jobs_sha256':digest(jobs),'evaluation_sha256':digest(rows),'worlds_sha256':digest(ws),
              'primary':'model_draft numeric_mass versus model_draft legacy_codes on event TVD and squared-vector loss',
              'parent_hashes':{n:sha(root/n) for n in ('measure_v14.py','contrast_v7.py','handoff_v5.py','reconstruct_v1.py','evidence_v4.py','jevbench_public_v1.py')}}
    for n,x in [('jobs',jobs),('evaluation',rows),('worlds',ws),('anchors',anchors),('manifest',manifest)]:write(out/(n+'.json'),x)
    print(json.dumps(manifest),flush=True)

def view(rt,row,note):
    order=list(CHANNELS);random.Random(int(digest({'input':row,'note':note})[:8],16)).shuffle(order)
    obs={}
    for name in order:obs[name]=parent.numerical_readout(rt,row,note) if name=='numeric_mass' else parent.code_readout(rt,row,note,name)
    numerical=obs['numeric_mass'];legacy=obs['legacy_codes']
    if numerical['parse_valid']:
        numerical['probabilities']=numerical['parsed']['probabilities'];numerical['label']=numerical['parsed']['label'];numerical['fallback_used']=False
        numerical['policy_seconds']=numerical['trace']['seconds']
    else:
        numerical['probabilities']=legacy['probabilities_uncalibrated'];numerical['label']=legacy['label'];numerical['fallback_used']=True
        numerical['policy_seconds']=numerical['trace']['seconds']+legacy['seconds']
    return {'outputs':obs,'execution_order':order,'note_sha256':hashlib.sha256(note.encode()).hexdigest()}

def run(root,prepared,out,shard):
    from reconstruct_v1 import Runtime
    root,prepared,out=map(Path,(root,prepared,out))
    if shard not in range(SHARDS):raise ValueError('Invalid shard')
    jobs=json.loads((prepared/'jobs.json').read_text());m=json.loads((prepared/'manifest.json').read_text())
    if sha(__file__)!=m['source_sha256'] or digest(jobs)!=m['jobs_sha256']:raise ValueError('Freeze mismatch')
    for n,h in m['parent_hashes'].items():
        if sha(root/n)!=h:raise ValueError('Parent mismatch')
    out.mkdir(parents=True,exist_ok=False);rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
    a=json.loads((prepared/'anchors.json').read_text())[shard];v,_=rt.score(a['input'])
    error=max(abs(x-y) for x,y in zip(v['logits'],a['native']['logits']))
    if error>1e-4 or v['prompt_hash']!=a['native']['prompt_hash']:raise ValueError('Native anchor mismatch')
    write(out/'preflight.json',{'source_sha256':m['source_sha256'],'runtime':rt.receipt,'actually_generative':True,'fixture':fixture,'anchor_error':error})
    path=out/'records.jsonl';path.write_text('');ids=[]
    for job in jobs[shard]:
        row=job['input']
        draft=parent.base.generate(rt,parent.base.e4.messages(row,'reason'),480)
        notes={'source_only':'','model_draft':draft['text'],'selected_evidence':job['evidence'],'oracle':job['oracle']}
        order=list(SUPPORTS);random.Random(int(digest(row['id'])[:8],16)).shuffle(order)
        views={s:view(rt,row,notes[s]) for s in order}
        result={'id':row['id'],'input_sha256':digest(row),'draft':draft,'support_order':order,'views':views,'shard':shard}
        with path.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        ids.append(row['id']);print(json.dumps({'shard':shard,'done':len(ids),'planned':len(jobs[shard])}),flush=True)
    write(out/'complete.json',{'count':len(ids),'ids':ids,'source_sha256':m['source_sha256'],'records_sha256':sha(path),'jobs_sha256':digest(jobs[shard])})

def aggregate(prepared,records,out):
    prepared,records,out=map(Path,(prepared,records,out));jobs=json.loads((prepared/'jobs.json').read_text());m=json.loads((prepared/'manifest.json').read_text());allrows=[];seen=set()
    for i,plan in enumerate(jobs):
        d=records/f'bridge-v15-shard-{i}';c=json.loads((d/'complete.json').read_text());p=d/'records.jsonl'
        if c['source_sha256']!=m['source_sha256'] or c['records_sha256']!=sha(p) or c['jobs_sha256']!=digest(plan):raise ValueError('Provenance')
        rows=[json.loads(s) for s in p.read_text().splitlines()];expected={j['input']['id']:j for j in plan}
        if len(rows)!=len(plan) or set(c['ids'])!=set(expected):raise ValueError('Incomplete shard')
        for r in rows:
            if r['id'] in seen or r['id'] not in expected or r['input_sha256']!=digest(expected[r['id']]['input']):raise ValueError('Input mismatch')
            if set(r['views'])!=set(SUPPORTS) or any(set(v['outputs'])!=set(CHANNELS) for v in r['views'].values()):raise ValueError('Missing arm')
            seen.add(r['id']);allrows.append(r)
    if len(allrows)!=48:raise ValueError('Incomplete experiment')
    write(out/'all_records.json',sorted(allrows,key=lambda r:r['id']));write(out/'completion.json',{'complete':True,'tasks':48,'worlds':24,'views':192,'source_sha256':m['source_sha256']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--root',default='.');p.add_argument('--prepared',default='bridge-prepared');p.add_argument('--records',default='records');p.add_argument('--out',default='.');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.prepared)
    elif a.mode=='run':run(a.root,a.prepared,a.out,a.shard)
    else:aggregate(a.prepared,a.records,a.out)
