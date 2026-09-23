"""Frozen, target-verified capability pilot sampled from Bridge's prepared worlds.
No benchmark examples or teacher answers are used. Both objectives see one
identical shuffled pass over the same 64 paired training edges per seed.
"""
from __future__ import annotations
import copy, hashlib, json, re
from fractions import Fraction as F
from pathlib import Path

DATA_SEED=1609202622
FIT_SEEDS=(615031,615032,615033,615034)
REPLICATION_SEEDS=(915151,915152,915153,915154)
FAMILIES=('counts','conditional','mixture','bayes')

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))

def law(state,labels):
    """Independent text-to-ticket arithmetic; no production probability function."""
    def numbers(s):
        pairs=re.findall(r'([a-z]+)=(\d+)',s)
        if len(pairs)!=len(labels) or {k for k,v in pairs}!=set(labels):raise ValueError('Count labels')
        d={k:int(v) for k,v in pairs};return [d[k] for k in labels]
    if state.startswith('The working urn'):
        mass=numbers(state.split('contains ',1)[1].split('. A sealed',1)[0])
    elif state.startswith('Colored tokens'):
        mass=numbers(state.split('ACCEPTED counts: ',1)[1].split('. REJECTED',1)[0])
    elif state.startswith('Select urn L'):
        weight=int(re.search(r'probability (\d+)/10',state).group(1))
        a=numbers(state.split('L counts: ',1)[1].split('. R counts:',1)[0])
        b=numbers(state.split('R counts: ',1)[1].split('. An unrelated',1)[0]);mass=[0]*len(a)
        for branch in range(10):
            chosen,other=(a,b) if branch<weight else (b,a)
            for i,n in enumerate(chosen):
                for _ in range(n*sum(other)):mass[i]+=1
    elif state.startswith('Initially select'):
        a=numbers(state.split('class counts ',1)[1].split('. Conditional',1)[0])
        rates=numbers(state.split('probabilities: ',1)[1].split('. The selected',1)[0]);mass=[0]*len(a)
        for i,n in enumerate(a):
            for _ in range(n):
                for assay in range(10):
                    if assay<rates[i]:mass[i]+=1
    else:raise ValueError('Unsupported independent grammar')
    return [F(n,sum(mass)) for n in mass]

def make_data(parent_root):
    import bridge_v15 as b
    parent_root=Path(parent_root)
    old=json.loads((parent_root/'bridge-prepared/evaluation.json').read_text())
    denied={digest({k:r['input'][k] for k in ('state','question','labels')}) for r in old}
    choose=lambda rows,n: sorted(rows,key=lambda x:digest({'seed':DATA_SEED,'group':x[0]}))[:n]
    pool=[]
    for seed in FIT_SEEDS:
        pool += [(f'{seed}-{w["world_id"]}',w) for w in b.worlds(seed)]
    fit=[]
    for f in FAMILIES:fit+=choose([x for x in pool if x[1]['family']==f],2)
    check_pool=[(f'715031-{w["world_id"]}',w) for w in b.worlds(715031)]
    check=[]
    for f in FAMILIES:check+=choose([x for x in check_pool if x[1]['family']==f],1)
    reps=[(f'replication-{s}-{w["world_id"]}',w) for s in REPLICATION_SEEDS for w in b.worlds(s) if w['family']=='mixture']
    fit_reps=choose(reps,8)
    check_reps=[(f'replication-915159-{w["world_id"]}',w) for w in b.worlds(915159) if w['family']=='mixture']
    records=[];relations=[];seen=set()
    for split,groups,replication in [('fit',fit,False),('check',check,False),('fit',fit_reps,True),('check',check_reps,True)]:
        for group,original in groups:
            variants=('base','left_urn_replicated') if replication else ('base','distractor','evidence','renamed')
            ids={}
            for variant in variants:
                w=copy.deepcopy(original)
                if variant=='distractor':w['noise']=[n*7+19 for n in w['noise']]
                elif variant=='evidence':w['a'][0]+=3
                elif variant=='renamed':
                    for key in ('labels','a','b','noise','rates'):w[key]=list(reversed(w[key]))
                elif variant=='left_urn_replicated':w['a']=[n*4 for n in w['a']]
                w['state'],w['evidence']=b.render(w)
                ref=law(w['state'],w['labels'])
                for mode in ('event','mode'):
                    if ref.count(max(ref))!=1:raise ValueError('Tied modal target')
                    t=b.task(w,mode);t['input']['id']=f'{group}-{variant}-{mode}'
                    q=ref if mode=='event' else [F(int(p==max(ref))) for p in ref]
                    if list(map(F,t['target']))!=q:raise ValueError('Independent target disagreement')
                    canonical=digest({k:t['input'][k] for k in ('state','question','labels')})
                    if canonical in denied or canonical in seen:raise ValueError('Input overlap')
                    seen.add(canonical);ids[(variant,mode)]=len(records)
                    records.append({'input':t['input'],'target':[str(x) for x in q], 'group':group,
                        'family':w['family'],'split':split,'mode':mode,'variant':variant,'replication':replication})
            k=len(original['labels'])
            for mode in ('event','mode'):
                for variant in variants[1:]:
                    i,j=ids[('base',mode)],ids[(variant,mode)]
                    A=[[int(z==(k-1-y if variant=='renamed' else y)) for z in range(k)] for y in range(k)]
                    qa=list(map(F,records[i]['target']));qb=list(map(F,records[j]['target']))
                    d=[qb[y]-sum(A[y][z]*qa[z] for z in range(k)) for y in range(k)]
                    kind='evidence_delta' if variant=='evidence' else 'permutation' if variant=='renamed' else 'irrelevant'
                    if kind!='evidence_delta' and any(d):raise ValueError('False relation')
                    relations.append({'left':i,'right':j,'mapping':A,'delta':[str(x) for x in d],
                        'kind':kind,'operation':variant,'group':group,'mode':mode,'split':split})
    # Fixed training probes: both base semantic questions from all eight ordinary fit worlds.
    probe=[i for i,r in enumerate(records) if r['split']=='fit' and not r['replication'] and r['variant']=='base']
    evaluate=probe+[i for i,r in enumerate(records) if r['split']=='check']
    train_edges=[i for i,e in enumerate(relations) if e['split']=='fit']
    if len(train_edges)!=64 or len(records)!=152 or len(probe)!=16 or len(evaluate)!=72:raise ValueError('Population drift')
    fit_groups={r['group'] for r in records if r['split']=='fit'};check_groups={r['group'] for r in records if r['split']=='check'}
    if fit_groups&check_groups:raise ValueError('World leakage')
    manifest={'records':len(records),'fit_examples':96,'check_examples':56,'fit_worlds':len(fit_groups),
              'check_worlds':len(check_groups),'fit_edges':64,'check_edges':len(relations)-64,
              'eval_indices':evaluate,'probe_indices':probe,'train_edges':train_edges,
              'records_sha256':digest(records),'relations_sha256':digest(relations),
              'selection_seed':DATA_SEED,'new_neural_training_targets':True,'provided_drafts':False,
              'scope':'Small same-grammar learning-capability pilot. One epoch over64 paired edges; not a generalization benchmark.'}
    return records,relations,manifest

def prepare(root,out):
    r,e,m=make_data(root);out=Path(out)
    for name,value in [('records',r),('relations',e),('manifest',m)]:save(out/(name+'.json'),value)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',default='.');p.add_argument('--out',default='learn-prepared');a=p.parse_args()
    print(json.dumps(prepare(a.parent,a.out),sort_keys=True))
