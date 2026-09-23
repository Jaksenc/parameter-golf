"""Independent reduction of a complete repaired factorial experiment.

No pretrained inference, trained-checkpoint selection, or test-fitted calibration.
Rebuild source targets without importing the production data generator.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import numpy as np

SEEDS=(17101,17102,17103)
ARMS=('event_high','mixed_high','event_low','mixed_low')
REPAIRED='c46f27f29b44f83426830d2c4a3ff265f6c6211952c98167d24d3248f44c7cad'
DATA='26ce5f59a61db89f66582935ffd3461c0e386a87487682819a5c23a7a9d4228e'

def read(p):
    p=Path(p)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))

def softmax(logits,temperature=1.):
    if not (0<temperature<math.inf) or not logits or not all(math.isfinite(z) for z in logits):raise ValueError('Invalid logits or temperature')
    with localcontext() as c:
        c.prec=60
        z=[Decimal(str(v)) for v in logits];t=Decimal(str(temperature));m=max(z)
        w=[((a-m)/t).exp() for a in z];s=sum(w)
        return [float(a/s) for a in w]

def table(text,labels):
    found=re.findall(r'\b([a-z]+)=(\d+)',text)
    if len(found)!=len(labels) or {a for a,b in found}!=set(labels):raise ValueError('Bad source table')
    m=dict(found);return [int(m[k]) for k in labels]

def section(text,begin,end):
    if text.count(begin)!=1:raise ValueError('Source grammar')
    return text.split(begin,1)[1].split(end,1)[0]

def source_law(row):
    s=row['input']['state'];labels=row['input']['labels'];counts=None
    if s.startswith('ACTIVE bag:'):
        counts=table(section(s,'ACTIVE bag:','. SPARE bag:'),labels)
    elif s.startswith('Draw uniformly from the in-use container'):
        counts=table(section(s,'In-use inventory:','. Reserve inventory:'),labels)
    elif s.startswith('FLAGGED counts:'):
        counts=table(section(s,'FLAGGED counts:','. CLEAR counts:'),labels)
    elif s.startswith('Only certified items are eligible'):
        counts=table(section(s,'Certified inventory:','. Uncertified inventory:'),labels)
    elif s.startswith('Choose L with probability') or s.startswith('A two-stage draw'):
        weight=int(re.search(r'probability (\d+)/10',s).group(1))
        if s.startswith('Choose L'):
            a=table(section(s,'L counts:','. R counts:'),labels);b=table(section(s,'R counts:','. An UNUSED'),labels)
        else:
            a=table(section(s,'X inventory:','. Y inventory:'),labels);b=table(section(s,'Y inventory:','. Ignored inventory'),labels)
        # Finite equiprobable tickets: each component is repeated to a common denominator.
        counts=[weight*x*sum(b)+(10-weight)*y*sum(a) for x,y in zip(a,b)]
    elif s.startswith('Initial class counts:'):
        a=table(section(s,'Initial class counts:','. Pass probabilities'),labels)
        rates=table(section(s,'Pass probabilities have numerators',', each over 10.'),labels)
        counts=[sum(1 for item in range(x) for outcome in range(10) if outcome<rate) for x,rate in zip(a,rates)]
    elif s.startswith('Choose uniformly from inventory'):
        a=table(section(s,'Choose uniformly from inventory','.'),labels)
        rates=table(section(s,'Acceptance likelihood numerators out of 10 by color:','.'),labels)
        counts=[sum(1 for item in range(x) for outcome in range(10) if outcome<rate) for x,rate in zip(a,rates)]
    else:raise ValueError('Unrecognized source syntax')
    if min(counts)<0 or sum(counts)<=0:raise ValueError('Source domain')
    q=[F(v,sum(counts)) for v in counts]
    if row['question_type']=='mode':
        if q.count(max(q))!=1:raise ValueError('Nonunique mode')
        q=[F(int(v==max(q))) for v in q]
    return q

def target(row):return [float(F(x)) for x in row['target']]

def metrics(logits,q,temperature=1.):
    p=np.asarray(softmax(logits,temperature));q=np.asarray(q,float)
    if len(p)!=len(q) or abs(q.sum()-1)>1e-12 or min(q)<0:raise ValueError('Bad target')
    z=np.asarray(logits,float)/temperature;m=z.max();lp=z-m-np.log(np.exp(z-m).sum())
    return {'ce':float(-np.dot(q,lp)),'squared':float(((p-q)**2).sum()),'tvd':float(np.abs(p-q).sum()/2),
            'accuracy':float(p.argmax()==q.argmax()),'entropy':float(-np.dot(p,lp)),
            'max_probability':float(p.max()),'probabilities':p.tolist()}

def means(mm):
    if not mm:raise ValueError('Empty cohort')
    return {'n':len(mm),**{k:float(np.mean([m[k] for m in mm])) for k in ('ce','squared','tvd','accuracy','entropy','max_probability')}}

def fit_temperature(observations,records,quantity):
    ids=[i for i,r in enumerate(records) if r['split']=='calibration' and r['question_type']==quantity]
    if len(ids)!=32:raise ValueError('Calibration cohort changed')
    temps=[2**(-2+j/20) for j in range(121)]
    losses=[float(np.mean([metrics(observations[i]['logits'],target(records[i]),t)['ce'] for i in ids])) for t in temps]
    j=int(np.argmin(losses))
    return {'temperature':temps[j],'n':32,'quantity':quantity,'calibration_loss':losses[j],
            'grid_min':.25,'grid_max':16.,'grid_points':121,'on_boundary':j in (0,120),'ids':ids}

def bootstrap(delta):
    """Cross paired seeds and entire worlds. Repeated observations are not iid."""
    x=np.asarray(delta,float)
    if x.ndim!=2:raise ValueError('Expected seeds by worlds')
    rng=np.random.default_rng(170923262);n,w=x.shape
    ss=rng.integers(n,size=(10000,n));ww=rng.integers(w,size=(10000,w))
    draws=x[ss[:,:,None],ww[:,None,:]].mean(axis=(1,2))
    return {'delta':float(x.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'paired_seeds':n,'worlds':w,'draws':10000,
            'scope':'Descriptive crossed bootstrap; three seeds, one authored grammar, no multiplicity correction.'}

def validate_source(records,m):
    if digest(records)!=DATA or m['record_hash']!=DATA or len(records)!=512:raise ValueError('Dataset digest')
    keys=[];groups=defaultdict(set)
    for r in records:
        if list(map(F,r['target']))!=source_law(r):raise ValueError('Target not implied by source '+r['input']['id'])
        if set(r['input'])!={'id','state','question','labels'}:raise ValueError('Unexpected request metadata')
        keys.append(digest({k:r['input'][k] for k in ('state','question','labels')}));groups[r['group']].add(r['split'])
    if len(keys)!=len(set(keys)) or any(len(s)!=1 for s in groups.values()):raise ValueError('Duplicate/split leak')
    for seed,sched in m['schedules'].items():
        if len(sched)!=32 or len({records[e]['group'] for e,d in sched})!=32:raise ValueError('Training coverage')
        for e,d in sched:
            if records[e]['split']!='fit' or records[d]['split']!='fit' or records[e]['group']!=records[d]['group']:raise ValueError('Leaked pair')
            if (records[e]['question_type'],records[d]['question_type'])!=('event','mode'):raise ValueError('Wrong pairing')
    return {'source_targets_checked':512,'worlds':len(groups),'fit_worlds':128,'calibration_worlds':32,'test_worlds':32,'canonical_unique':True,
            'same_grammar_only':True,'labels_do_not_enter_requests':True}

def load_observations(path,records,expected):
    data=read(path);idx={r['record_index']:r for r in data}
    if len(idx)!=len(data) or set(idx)!=set(expected):raise ValueError('Missing observations')
    err=0.
    for i,r in idx.items():
        if r['id']!=records[i]['input']['id'] or r['input_sha256']!=digest(records[i]['input']):raise ValueError('Observation binding')
        p=softmax(r['logits']);err=max(err,max(abs(a-b) for a,b in zip(p,r['probabilities'])))
        if len(p)!=len(records[i]['target']) or err>2e-7:raise ValueError('Probability mismatch')
    return idx,err

def run(root):
    root=Path(root);out=root/'results';records=read(root/'causal-prepared/records.json');manifest=read(root/'causal-prepared/manifest.json')
    source=validate_source(records,manifest);indices=manifest['evaluation_indices']
    baseline,error=load_observations(root/'baseline/predictions.jsonl',records,indices)
    if read(root/'baseline/complete.json')['record_hash']!=DATA:raise ValueError('Wrong baseline')
    by={('baseline',0):baseline};receipts={};logs={};probes={};snapshot_bindings={};total_calls=Counter()
    modelroot=root/'models'
    for seed in SEEDS:
        for arm in ARMS:
            d=modelroot/f'causal-v17-clean-model-{seed}-{arm}'
            receipt=read(d/'segment-32.json');pre=read(d/'preflight.json');half=read(d/'segment-16.json')
            if receipt['through_step']!=32 or receipt['from_step']!=16 or not receipt['complete'] or half['from_step']!=0 or not half['complete']:raise ValueError('Incomplete training')
            if pre['bindings']['trainer']!=REPAIRED or receipt['bindings']['trainer']!=REPAIRED:raise ValueError('Invalid old attempt mixed in')
            if pre['bindings']['data']!=DATA or receipt['roundtrip_logit_error']>1e-4 or receipt['base_restoration_error']>1e-4:raise ValueError('Restoration')
            if (seed,arm)!=(pre['seed'],pre['arm']):raise ValueError('Arm attribution')
            rows=read(d/'training.jsonl');sched=manifest['schedules'][str(seed)]
            if [r['step'] for r in rows]!=list(range(1,33)) or [r['pair'] for r in rows]!=sched:raise ValueError('Schedule changed')
            for row in rows:
                if row['optimizer_step']!=row['step']:raise ValueError('Optimizer step mismatch')
                if row['step'] in (1,16,32) and not row['counterfactual_state_hash_preserved']:raise ValueError('Contaminated diagnostics')
                wt=1 if arm.startswith('mixed') else 0
                expected_loss=.5*(row['event_ce']+wt*row['mode_ce'])
                if abs(expected_loss-row['weighted_loss'])>1e-7:raise ValueError('Objective mismatch')
                for i,z,loss in zip(row['pair'],row['observed_logits'],(row['event_ce'],row['mode_ce'])):
                    if abs(metrics(z,target(records[i]))['ce']-loss)>2e-5:raise ValueError('Logged CE mismatch')
            obs,err=load_observations(d/'final-predictions.json',records,indices);error=max(error,err)
            if any(obs[i]['prompt_sha256']!=baseline[i]['prompt_sha256'] for i in indices):raise ValueError('Input formatting drift')
            by[(arm,seed)]=obs;receipts[(arm,seed)]=receipt;logs[(arm,seed)]=rows
            snapshot_bindings[(arm,seed)]=pre['initial_sha256']
            total_calls.update(half['calls']);total_calls.update(receipt['calls'])
            probes[(arm,seed)]={step:load_observations(d/f'probe-step-{step}.json',records,manifest['probe_indices'])[0] for step in (0,1,8,16,32)}
    for seed in SEEDS:
        if len({snapshot_bindings[(a,seed)] for a in ARMS})!=1:raise ValueError('Unmatched initialization')
    ids_by_q={q:[i for i in indices if records[i]['split']=='test' and records[i]['variant']=='base' and records[i]['question_type']==q] for q in ('event','mode')}
    if any(len(v)!=32 for v in ids_by_q.values()):raise ValueError('Wrong test domain')
    calibrators={};performance={};per_example={};predictions=[]
    for (arm,seed),obs in by.items():
        tag=f'{arm}/{seed}';calibrators[tag]={q:fit_temperature(obs,records,q) for q in ('event','mode')}
        performance[tag]={};per_example[tag]={}
        for q,ids in ids_by_q.items():
            performance[tag][q]={};per_example[tag][q]={}
            for method,temp in [('raw',1.),('calibrated',calibrators[tag][q]['temperature'])]:
                mm=[metrics(obs[i]['logits'],target(records[i]),temp) for i in ids]
                performance[tag][q][method]=means(mm);per_example[tag][q][method]=mm
        for i in indices:
            q=records[i]['question_type'];temp=calibrators[tag][q]['temperature']
            predictions.append({'arm':arm,'seed':seed,'id':records[i]['input']['id'],'record_index':i,'group':records[i]['group'],
                                'split':records[i]['split'],'quantity':q,'variant':records[i]['variant'],'family':records[i]['family'],
                                'raw':metrics(obs[i]['logits'],target(records[i])),'calibrated':metrics(obs[i]['logits'],target(records[i]),temp)})
    overall={}
    for arm in ('baseline',)+ARMS:
        tags=[f'{arm}/{s}' for s in ((0,) if arm=='baseline' else SEEDS)]
        overall[arm]={q:{mode:{k:float(np.mean([performance[t][q][mode][k] for t in tags])) for k in ('ce','squared','tvd','accuracy','entropy','max_probability')} for mode in ('raw','calibrated')} for q in ('event','mode')}
    contrasts={}
    comparisons=[('mixed_high','event_high'),('mixed_low','event_low'),('event_low','event_high'),('mixed_low','mixed_high')]+[(a,'baseline') for a in ARMS]
    for a,b in comparisons:
        contrasts[a+' minus '+b]={}
        for q in ('event','mode'):
            contrasts[a+' minus '+b][q]={}
            for mode in ('raw','calibrated'):
                contrasts[a+' minus '+b][q][mode]={}
                for metric in ('ce','squared','tvd','accuracy'):
                    values=[]
                    for seed in SEEDS:
                        at=f'{a}/{seed}';bt=f'{b}/{0 if b=="baseline" else seed}'
                        values.append([x[metric]-y[metric] for x,y in zip(per_example[at][q][mode],per_example[bt][q][mode])])
                    contrasts[a+' minus '+b][q][mode][metric]=bootstrap(values)
    transformed={};curve={};counterfactual={}
    for (arm,seed),obs in by.items():
        tag=f'{arm}/{seed}';transformed[tag]={}
        lookup={(records[i]['group'],records[i]['variant']):i for i in indices if records[i]['split']=='test' and records[i]['question_type']=='event'}
        for v in ('replicate','distractor','relevant_edit','paraphrase'):
            qerrs=[];pd=[]
            for group in sorted({g for g,vv in lookup}):
                bi=lookup[(group,'base')];vi=lookup[(group,v)]
                p0=np.array(softmax(obs[bi]['logits']));p1=np.array(softmax(obs[vi]['logits']))
                q0=np.array(target(records[bi]));q1=np.array(target(records[vi]))
                pd.append(float(np.sum(((p1-p0)-(q1-q0))**2)))
                qerrs.append(metrics(obs[vi]['logits'],q1))
            transformed[tag][v]={'prediction_delta_squared':float(np.mean(pd)),**means(qerrs)}
        if arm=='baseline':continue
        curve[tag]={}
        for step,ob in probes[(arm,seed)].items():
            curve[tag][step]={q:means([metrics(v['logits'],target(records[i])) for i,v in ob.items() if records[i]['question_type']==q]) for q in ('event','mode')}
        rows=logs[(arm,seed)];cf=[]
        for r in rows:
            if r['counterfactuals']:
                for branch,o in r['counterfactuals'].items():
                    cf.append({'step':r['step'],'world':r['world'],'branch':branch,'event_loss_change':o['losses_after'][0]-o['losses_before'][0],
                               'modal_loss_change':o['losses_after'][1]-o['losses_before'][1], 'gradient_cosine':r['gradient_cosine'],
                               'scope':'Local finite optimizer intervention on the current training pair, not held-out causal evidence.'})
        counterfactual[tag]={'local_probes':cf,'negative_gradient_cosine_steps':sum(r['gradient_cosine'] is not None and r['gradient_cosine']<0 for r in rows),
                             'steps':32,'mean_gradient_cosine':float(np.mean([r['gradient_cosine'] for r in rows if r['gradient_cosine'] is not None])),
                             'clipped_steps':sum(r['combined_preclip_norm']>1 for r in rows),'training_phase_seconds':sum(r['seconds'] for r in rows)}
    uniform={q:means([metrics([0.]*len(target(records[i])),target(records[i])) for i in ids]) for q,ids in ids_by_q.items()}
    summary={'status':'complete_clean_replay','steps':32,'arms':4,'paired_seeds':3,'test_base_worlds':32,'calibration_worlds':32,
             'metrics':overall,'uniform':uniform,'all_models_retained':True,'new_optimizer_updates':384,'no_jevbench_evaluation':True,
             'limits':['One authored grammar; 32 selected fit worlds per seed; final test is shortcut-discriminating challenge data.',
                       'Parameter and optimizer interventions controlled; global model superiority not established.',
                       'Calibration temperatures selected on separate calibration worlds, raw scores remain primary.',
                       'Old invalid optimizer-aliasing attempt excluded completely.']}
    audit={'source':source,'models':12,'final_vector_count':len(predictions),'probability_max_error':error,'forward_categories':dict(total_calls),
           'paired_initialization_checked':True,'gradients_both_tasks_computed':True,'true_optimizer_steps':384,'discarded_diagnostic_steps':72,
           'schedules_verified':True,'source_and_output_binding_verified':True,'baseline_vectors':256,'trained_vectors':3072,
           'calibration_ids_disjoint_from_test':True,'offline_replay_only':True}
    for name,data in [('summary',summary),('per_model',performance),('calibration',calibrators),('factorial_contrasts',contrasts),
                      ('predictions',predictions),('transformations',transformed),('learning_curves',curve),('counterfactuals',counterfactual),('audit',audit)]:write(out/(name+'.json'),data)
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
