"""Independent reduction of the frozen Learn v16 paired-adapter experiment.

Replays recorded outputs and source laws, not the model's optimization. Stored
float32 softmax vectors remain the reported estimates; higher-precision softmax
is used only for an implementation-error check, never to improve a score.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, random, re
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

SEEDS=(16101,16102,16103)
ARMS=('supervised','relational')
SOURCE='88d3ce9ebae554adbb1728b45254bfe7a87043a26d1264fe0fad55077054b587'
RECORDS='069c87039a67b933922ef9dafcd2fc8fd47b2fe675114b55d6d9d7eb171ecf14'
RELATIONS='9d29f2a0a47d27ee4c291a7a1e35cb6475532e816992774cf62a51520c43a639'


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):return json.loads(Path(p).read_text())

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))


def source_target(row):
    """Text-derived reference, independently expressed as rational probability laws."""
    s=row['input']['state'];labels=row['input']['labels']
    def arr(text):
        fields=dict(re.findall(r'\b([a-z]+)=(\d+)(?:/10)?\b',text))
        if set(fields)!=set(labels):raise ValueError('Reference cannot identify all labels')
        return [F(fields[k]) for k in labels]
    def norm(a):
        if sum(a)<=0:raise ValueError('Empty source population')
        return [x/sum(a) for x in a]
    if s.startswith('The working urn'):
        q=norm(arr(s.partition('contains ')[2].partition('. A sealed')[0]))
    elif s.startswith('Colored tokens'):
        q=norm(arr(s.partition('ACCEPTED counts: ')[2].partition('. REJECTED')[0]))
    elif s.startswith('Select urn L'):
        weight=F(re.search(r'probability (\d+)/10',s).group(1))/10
        left=norm(arr(s.partition('L counts: ')[2].partition('. R counts:')[0]))
        right=norm(arr(s.partition('R counts: ')[2].partition('. An unrelated')[0]))
        q=[weight*x+(1-weight)*y for x,y in zip(left,right)]
    elif s.startswith('Initially select'):
        counts=arr(s.partition('class counts ')[2].partition('. Conditional')[0])
        likelihood=arr(s.partition('probabilities: ')[2].partition('. The selected')[0])
        q=norm([x*y for x,y in zip(counts,likelihood)])
    else:raise ValueError('Unsupported independent reference')
    if q.count(max(q))!=1:raise ValueError('Tied reference mode')
    if row['mode']=='mode':q=[F(int(x==max(q))) for x in q]
    if q!=list(map(F,row['target'])):raise ValueError('Target/source mismatch')
    return [float(x) for x in q]


def loss_values(p,q):
    if len(p)!=len(q) or not p or any(not math.isfinite(x) or x<0 or x>1 for x in p):
        raise ValueError('Invalid prediction')
    if abs(sum(p)-1)>3e-7:raise ValueError('Probability normalization error')
    false_zero=any(y>0 and x==0 for x,y in zip(p,q))
    ce=None if false_zero else -sum(y*math.log(x) for x,y in zip(p,q) if y>0)
    entropy=-sum(y*math.log(y) for y in q if y>0)
    return {'cross_entropy':ce,'excess_log_loss':None if ce is None else ce-entropy,
            'squared_vector_error':sum((x-y)**2 for x,y in zip(p,q)),
            'tvd':sum(abs(x-y) for x,y in zip(p,q))/2,
            'false_zero':false_zero,'impossible_mass':sum(x for x,y in zip(p,q) if y==0),
            'modal_correct':max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__),
            'max_confidence':max(p),'entropy':-sum(x*math.log(x) for x in p if x>0)}


def summarized(rows):
    if not rows:raise ValueError('Empty metric population')
    n=len(rows);keys=('cross_entropy','excess_log_loss','squared_vector_error','tvd','impossible_mass','max_confidence','entropy')
    result={'n':n,'worlds':len({x['group'] for x in rows}),'correct_modal_labels':sum(x['modal_correct'] for x in rows),
            'modal_accuracy':sum(x['modal_correct'] for x in rows)/n,'false_zero_cases':sum(x['false_zero'] for x in rows)}
    for k in keys:
        result[k]=sum(x[k] for x in rows)/n if all(x[k] is not None for x in rows) else None
    result['strict_log_loss_status']='infinite' if result['cross_entropy'] is None else 'finite'
    return result


def partition_metrics(rows):
    result={}
    for split in ('fit_probe','check'):
        part=[x for x in rows if x['population']==split]
        result[split]={mode:summarized([x for x in part if mode=='all' or x['mode']==mode]) for mode in ('all','event','mode')}
        result[split]['by_family']={f:summarized([x for x in part if x['family']==f]) for f in sorted({x['family'] for x in part})}
    return result


def describe(values):
    a=np.array(values,dtype=float)
    return {'n':len(values),'sum':float(a.sum()),'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max())}


def verify_observation(o,row,index):
    if o['record_index']!=index or o['id']!=row['input']['id'] or o['input_sha256']!=digest(row['input']):raise ValueError('Inference provenance')
    if not 1<=o['tokens']<=768 or not math.isfinite(o['seconds']) or o['seconds']<=0:raise ValueError('Invalid execution record')
    z=np.array(o['logits'],dtype=np.longdouble)
    if len(z)!=len(row['target']) or not np.isfinite(z).all():raise ValueError('Invalid logits')
    expected=np.exp(z-z.max());expected/=expected.sum()
    error=float(np.max(np.abs(expected-np.array(o['probabilities'],dtype=np.longdouble))))
    # The model explicitly records FP32 torch.softmax, not a float64 distribution.
    if error>2e-7:raise ValueError('Softmax discrepancy exceeds FP32 allowance')
    if int(np.argmax(z))!=int(np.argmax(o['probabilities'])):raise ValueError('Argmax inconsistency')
    return error


def relation_metrics(predictions,records,edges):
    result=[]
    for edge in edges:
        if edge['split']!='check':continue
        i,j=edge['left'],edge['right'];a=np.asarray(predictions[i],dtype=float);b=np.asarray(predictions[j],dtype=float)
        mat=np.asarray(edge['mapping'],dtype=float);delta=np.array([float(F(v)) for v in edge['delta']])
        residual=b-mat@a-delta
        result.append({'left':i,'right':j,'mode':edge['mode'],'group':edge['group'],'operation':edge['operation'],
                       'squared_relation_residual':float(residual@residual),'l1_relation_residual':float(np.abs(residual).sum())})
    by={}
    for op in sorted({r['operation'] for r in result}):
        rs=[r for r in result if r['operation']==op]
        by[op]={'n':len(rs),'squared_residual':sum(x['squared_relation_residual'] for x in rs)/len(rs),
                'l1_residual':sum(x['l1_relation_residual'] for x in rs)/len(rs)}
    return {'n':len(result),'squared_residual':sum(x['squared_relation_residual'] for x in result)/len(result),'by_operation':by,'records':result}


def paired_interval(delta_by_seed,meta,metric,mode,draws=10000):
    ids=[i for i,x in meta.items() if x['population']=='check' and (mode=='all' or x['mode']==mode)]
    groups=sorted({meta[i]['group'] for i in ids});group_ids={g:[i for i in ids if meta[i]['group']==g] for g in groups}
    if any(delta_by_seed[s][i][metric] is None for s in SEEDS for i in ids):
        return {'status':'infinite-loss observation; no finite bootstrap','metric':metric,'mode':mode}
    sums=np.array([[sum(delta_by_seed[s][i][metric] for i in group_ids[g]) for g in groups] for s in SEEDS])
    counts=np.array([len(group_ids[g]) for g in groups])
    rng=np.random.default_rng(161091);out=[]
    # Independently resample model seeds and shared worlds, maintaining paired arms.
    for _ in range(draws):
        selected_seed=rng.integers(3,size=3);selected_group=rng.integers(len(groups),size=len(groups))
        out.append(float(sums[np.ix_(selected_seed,selected_group)].sum()/(3*counts[selected_group].sum())))
    return {'status':'descriptive','metric':metric,'mode':mode,'mean_delta':float(sums.sum()/(3*counts.sum())),
            'interval95':list(map(float,np.quantile(out,[.025,.975]))),'draws':draws,'seeds':3,'worlds':len(groups),
            'limits':'Paired empirical seed/world bootstrap, only3 seeds/10 same-grammar worlds; no broad-language, multiplicity-adjusted or formal significance guarantee.'}


def analyze(root,out):
    root=Path(root).resolve();out=Path(out).resolve();prep=root/'learn-prepared'
    records,edges,m=load(prep/'records.json'),load(prep/'relations.json'),load(prep/'manifest.json')
    freeze=load(prep/'freeze.json')
    if digest(records)!=RECORDS or digest(edges)!=RELATIONS or freeze['sources']['learn_v16.py']!=SOURCE:raise ValueError('Scientific source/data changed')
    for name,h in freeze['sources'].items():
        if filehash(root/name)!=h:raise ValueError('Frozen dependency changed: '+name)
    refs=[source_target(r) for r in records]
    if {r['group'] for r in records if r['split']=='fit'} & {r['group'] for r in records if r['split']=='check'}:raise ValueError('Shared fit/check world')
    keys=[digest({k:r['input'][k] for k in ('state','question','labels')}) for r in records]
    if len(set(keys))!=len(records):raise ValueError('Duplicate canonical inputs')
    for e in edges:
        i,j=e['left'],e['right'];a=list(map(F,records[i]['target']));b=list(map(F,records[j]['target']))
        if records[i]['split']!=records[j]['split'] or records[i]['group']!=records[j]['group']:raise ValueError('Relation leakage')
        if b!=[sum(F(e['mapping'][y][z])*a[z] for z in range(len(a)))+F(e['delta'][y]) for y in range(len(b))]:raise ValueError('False relation')
    base_reference=None;probability_error=0.;paired_initial={};paired_schedules={};outputs={};receipts={};traces={};all_metric=[];states_info={};signatures={}
    meta={i:{'id':r['input']['id'],'group':r['group'],'family':r['family'],'mode':r['mode'],
             'population':'check' if r['split']=='check' else 'fit_probe','replication':r['replication'],'variant':r['variant']}
          for i,r in enumerate(records) if i in m['eval_indices']}
    for seed in SEEDS:
        for arm in ARMS:
            key=f'{seed}-{arm}';d=root/'trained'/f'learn-v16-model-{key}'
            receipt=load(d/'complete.json');pre=load(d/'preflight.json');schedule=load(d/'schedule.json')
            if receipt['status']!='complete' or receipt['steps']!=64 or receipt['seed']!=seed or receipt['arm']!=arm or receipt['source_sha256']!=SOURCE:raise ValueError('Incomplete model')
            if receipt['records_sha256']!=RECORDS or pre['records_sha256']!=RECORDS or pre['steps']!=64 or pre['rank']!=4:raise ValueError('Protocol mismatch')
            if receipt['updated_layers']!=32 or receipt['base_parameters_updated']!=0:raise ValueError('Weight scope mismatch')
            if receipt['replay_error']>2e-4 or receipt['reload_error']>1e-4 or receipt['base_restoration_error']>1e-4 or pre['zero_adapter_error']>1e-4:raise ValueError('Runtime integrity')
            for arr in ('first_B_gradient_norms','max_B_changes'):
                if len(receipt[arr])!=32 or any(not math.isfinite(v) or v<=0 for v in receipt[arr]):raise ValueError('Inactive depth')
            state=load_file(str(d/'adapter.safetensors'));h=hashlib.sha256()
            for name,a in sorted(state.items()):
                if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Corrupt adapter')
                h.update(name.encode());h.update(a.tobytes())
            if h.hexdigest()!=receipt['final_tensor_sha256'] or filehash(d/'adapter.safetensors')!=receipt['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
            if sum(a.size for a in state.values())!=receipt['trained_adapter_parameters'] or len(state)!=64:raise ValueError('Parameter count')
            if any(np.max(np.abs(state[f'{i}.b']))<=0 for i in range(32)):raise ValueError('Missing updated block')
            order=list(m['train_edges']);random.Random(seed).shuffle(order)
            if schedule['edge_indices']!=order or schedule['sha256']!=digest(order):raise ValueError('Changed order')
            if seed in paired_initial and paired_initial[seed]!=pre['initial_adapter_sha256']:raise ValueError('Unpaired initial weights')
            paired_initial[seed]=pre['initial_adapter_sha256'];paired_schedules[seed]=digest(order)
            t=[json.loads(x) for x in (d/'training.jsonl').read_text().splitlines()]
            if len(t)!=64:raise ValueError('Missing optimizer steps')
            seen=set();weight=.25 if arm=='relational' else 0.
            for step,(x,ei) in enumerate(zip(t,order),1):
                e=edges[ei];indices=[e['left'],e['right']]
                if x['step']!=step or x['edge_index']!=ei or x['input_ids']!=[records[i]['input']['id'] for i in indices]:raise ValueError('Wrong training examples')
                if any(records[i]['split']!='fit' for i in indices):raise ValueError('Check target used for training')
                seen.update(indices)
                if any(not math.isfinite(x[k]) for k in ('loss','cross_entropy','relation_loss','gradient_norm','seconds')):raise ValueError('Nonfinite step')
                if abs(x['loss']-(x['cross_entropy']+weight*x['relation_loss']))>2e-6:raise ValueError('Loss identity')
            if len(seen)!=96 or receipt['examples_presented']!=128 or receipt['unique_fit_examples']!=96:raise ValueError('Exposure mismatch')
            if receipt['calls']!={'baseline':72,'no_grad_training':128,'backward_replay':128,'after':72,'integrity':3}:raise ValueError('Call accounting')
            outputs[key]={};signatures[key]={}
            for phase in ('baseline','after'):
                obs=load(d/(phase+'.json'))
                if len(obs)!=72 or [x['record_index'] for x in obs]!=m['eval_indices']:raise ValueError('Incomplete evaluation')
                ps={};metric_rows=[];signature=[]
                for o in obs:
                    i=o['record_index'];probability_error=max(probability_error,verify_observation(o,records[i],i))
                    ps[i]=o['probabilities'];vals=loss_values(o['probabilities'],refs[i])
                    metric_rows.append({**meta[i],**vals,'seed':seed,'arm':arm,'phase':phase,'record_index':i})
                    signature.append({k:o[k] for k in ('id','logits','probabilities','prompt_sha256','input_sha256','tokens')})
                outputs[key][phase]={'predictions':ps,'metrics':partition_metrics(metric_rows),
                    'relations':relation_metrics(ps,records,edges),'timing':describe([x['seconds'] for x in obs]),'rows':metric_rows}
                signatures[key][phase]=digest(signature)
                if phase=='baseline':
                    if base_reference is not None and base_reference!=signature:raise ValueError('Baseline neural outputs differ across workers')
                    base_reference=signature
                all_metric+=metric_rows
            receipts[key]=receipt;traces[key]=t;states_info[key]={k:list(v.shape) for k,v in state.items()}
    # Simple untrained uniform control is known before inspecting model outcomes.
    uniform_rows=[{**meta[i],**loss_values([1/len(refs[i])]*len(refs[i]),refs[i]),'record_index':i} for i in m['eval_indices']]
    summary={'status':'complete','primary':'relational versus matched supervised, fixed step64',
             'seeds':list(SEEDS),'population':m,'base':outputs[f'{SEEDS[0]}-supervised']['baseline']['metrics'],
             'uniform':partition_metrics(uniform_rows),'models':{},'paired_intervals':{},'mean_across_seeds':{},
             'limits':['Only16 training worlds,10 held-out worlds and3 training seeds.','One pass over64 paired edges; same grammar, no independently authored transfer.','No JevBench evaluation, teacher distillation, generated inference, price estimate or full score.','No best-seed or intermediate-checkpoint selection. Probabilities are uncalibrated FP32 code preferences.']}
    for key,x in outputs.items():
        summary['models'][key]={'after':x['after']['metrics'],'before_relations':x['baseline']['relations']['squared_residual'],
                               'after_relations':x['after']['relations']['squared_residual'],'cost':{
                                'baseline':x['baseline']['timing'],'adapted':x['after']['timing'],
                                'train_seconds':receipts[key]['training_seconds'],'peak_rss_mib':receipts[key]['peak_rss_mib']}}
    for arm in ARMS:
        summary['mean_across_seeds'][arm]={}
        for population in ('check','fit_probe'):
            summary['mean_across_seeds'][arm][population]={}
            for mode in ('all','event','mode'):
                entries=[outputs[f'{s}-{arm}']['after']['metrics'][population][mode] for s in SEEDS]
                summary['mean_across_seeds'][arm][population][mode]={k:sum(x[k] for x in entries)/3 if all(x[k] is not None for x in entries) else None
                  for k in ('cross_entropy','excess_log_loss','squared_vector_error','tvd','correct_modal_labels','modal_accuracy','impossible_mass')}
    metric_index={(x['seed'],x['arm'],x['phase'],x['record_index']):x for x in all_metric}
    for a,b in [('relational','supervised'),('supervised','base'),('relational','base')]:
        differences={s:{} for s in SEEDS}
        for s in SEEDS:
            for i in m['eval_indices']:
                x=metric_index[(s,a,'after',i)]
                y=metric_index[(s,'supervised','baseline',i)] if b=='base' else metric_index[(s,b,'after',i)]
                differences[s][i]={k:x[k]-y[k] if x[k] is not None and y[k] is not None else None
                                   for k in ('cross_entropy','squared_vector_error','tvd')}
        summary['paired_intervals'][a+'_minus_'+b]={mode:{k:paired_interval(differences,meta,k,mode)
             for k in ('cross_entropy','squared_vector_error','tvd')} for mode in ('event','mode','all')}
    audit={'source_targets_reconstructed':len(records),'verified_relations':len(edges),'models':6,'new_optimizer_updates':384,
           'stored_evaluation_vectors':864,'unique_evaluation_inputs':72,'held_out_inputs':56,'fit_probe_inputs':16,
           'fp32_softmax_max_abs_error':probability_error,'paired_initial_hashes':paired_initial,'paired_schedule_hashes':paired_schedules,
           'baseline_all_six_workers_equal':True,'checkpoint_shapes':states_info,'evaluation_signatures':signatures,
           'base_weights_scope':'Runtime freezes/rejects original gradients and restores one base probe; not an independent full base-weight hash after training.',
           'training_replay_scope':'Recorded gradients, losses, schedules and final tensors audited; full neural optimization not rerun here.'}
    relationships={key:{phase:x[phase]['relations'] for phase in ('baseline','after')} for key,x in outputs.items()}
    write(out/'summary.json',summary);write(out/'all_metrics.json',all_metric);write(out/'audit.json',audit)
    write(out/'relation_metrics.json',relationships);write(out/'model_receipts.json',receipts)
    print(json.dumps({'means':summary['mean_across_seeds'],'base':summary['base'],'audit':{k:v for k,v in audit.items() if k not in ('checkpoint_shapes','evaluation_signatures')}},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data');p.add_argument('--out',default='results');a=p.parse_args();analyze(a.root,a.out)
