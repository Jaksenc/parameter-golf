"""Independent recorded-data scoring. Does not invoke the production parser.
No model execution, strategy selection, or calibration takes place in this file.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,re,sys
from pathlib import Path
import numpy as np
SOURCE='e50085c81381f226a632e06efbb7d740f276acd013a3931e0c18343d15a55ffd'
ARMS=('standard480','countercase480')
SEED=820260921


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))
def inputs(t):return {k:t[k] for k in ('id','state','question','labels')}


def final(text,labels,cut):
 found=[];lines=text.split('\n')
 for n,line in enumerate(lines):
  if line.startswith('FINAL:'):
   value=line[6:].strip()
   if value in labels and (n<len(lines)-1 or not cut):found.append(value)
 return found[0] if len(found)==1 else None


def replay(output,task,archived=False):
 labels=task['labels'];t=output['trace'];ro=output['readout']
 ids=t['token_ids'];count=t['output_tokens']
 if count!=len(ids) or not 0<count<=480 or not all(type(x) is int for x in ids):raise ValueError('Token record')
 ended=ids[-1] in t['eos_ids'];cut=count==480 and not ended
 if cut!=t['hit_cap']:raise ValueError('Stop-state mismatch')
 parsed=final(t['text'],labels,cut)
 error=0.
 if parsed is None:
  if ro is None:raise ValueError('Missing fallback')
  z=np.asarray(ro['logits'],dtype=np.longdouble)
  if z.shape!=(len(labels),) or not np.isfinite(z).all():raise ValueError('Invalid logits')
  p=np.exp(z-z.max());p/=p.sum()
  error=float(np.max(np.abs(p-np.asarray(ro['probabilities_uncalibrated'],dtype=np.longdouble))))
  if error>1e-12:raise ValueError('Probability mismatch')
  parsed=labels[int(np.argmax(z))]
  if parsed!=ro['label']:raise ValueError('Fallback argmax')
  if ro['draft_sha256']!=hashlib.sha256(t['text'].encode()).hexdigest():raise ValueError('Draft provenance')
  if len(ro['code_token_ids'])!=len(set(ro['code_token_ids'])) or len(ro['code_token_ids'])!=len(labels):raise ValueError('Code count')
 else:
  if ro is not None:raise ValueError('Unneeded fallback')
 if output['answer']!=parsed:raise ValueError('Independent label replay mismatch')
 if not archived and output['input_sha256']!=digest(inputs(task)):raise ValueError('Output input mismatch')
 seconds=t['seconds']+(ro['seconds'] if ro else 0.)
 if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid time')
 return parsed,error,int(ro is not None),seconds


def paired(rows,tasks,a,b,draws=10000):
 ds=np.asarray([int(r['predictions'][a]==str(tasks[r['id']]['expected']))-int(r['predictions'][b]==str(tasks[r['id']]['expected'])) for r in rows])
 rng=np.random.default_rng(SEED)
 if all(tasks[r['id']]['partition']=='jevbench' for r in rows):
  groups=collections.defaultdict(list)
  for i,r in enumerate(rows):groups[tasks[r['id']].get('group') or r['id']].append(i)
  sums=np.array([ds[g].sum() for g in groups.values()]);sizes=np.array([len(g) for g in groups.values()]);k=len(sizes)
  idx=rng.integers(k,size=(draws,k));boot=sums[idx].sum(1)/sizes[idx].sum(1)
  method='source-group bootstrap'
 else:
  groups=collections.defaultdict(list)
  for i,r in enumerate(rows):groups[tasks[r['id']]['family']].append(i)
  boot=np.zeros(draws)
  for name in sorted(groups):
   ids=groups[name];arr=ds[ids];boot+=arr[rng.integers(len(ids),size=(draws,len(ids)))].sum(1)
  boot/=len(rows);k=len(groups);method='within-family paired item bootstrap'
 w=int((ds>0).sum());l=int((ds<0).sum());d=w+l
 return {'a':a,'b':b,'n':len(rows),'repairs':w,'regressions':l,'delta_percentage_points':float(ds.mean()*100),
         'bootstrap95_percentage_points':list(map(float,np.quantile(boot,[.025,.975])*100)),
         'discordant_pair_p':min(1.,2*sum(math.comb(d,i) for i in range(min(w,l)+1))/2**d) if d else 1.,
         'resampling':method,'groups_or_strata':k,'draws':draws,
         'repair_ids':[r['id'] for r,v in zip(rows,ds) if v>0],'regression_ids':[r['id'] for r,v in zip(rows,ds) if v<0],
         'limits':'Exploratory, finite cohorts; no adjustment for multiplicity or repeated benchmark use.'}


def summarize(values):
 a=np.asarray(values,dtype=float)
 if not len(a):return {'n':0,'sum':0.,'mean':None,'median':None,'p95':None}
 return {'n':len(a),'sum':float(a.sum()),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95))}


def source_selection(root,evaluation,m):
 # Independent recreation of the exclusion, shuffling, option and target mapping.
 old=json.loads((root/'countercase-prepared/overlap_inputs.json').read_text())
 def strings(x):
  if isinstance(x,str):yield x
  elif isinstance(x,dict):
   for value in x.values():yield from strings(value)
  elif isinstance(x,list):
   for value in x:yield from strings(value)
 def norm(x):return ' '.join(re.findall(r'\w+',x.lower()))
 prior_texts=[norm(s) for x in old for s in strings({'state':x['state'],'question':x['question']}) if len(s)>=45]
 big=' '.join(prior_texts);matched=0;seen=[]
 for receipt in m['source_receipts']:
  family=receipt['family'];raw=(root/'contrast-prepared/sources'/f'{family}.json').read_bytes()
  if filehash(root/'contrast-prepared/sources'/f'{family}.json')!=receipt['sha256']:raise ValueError('Source digest')
  if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=receipt['git_blob']:raise ValueError('Source git blob')
  all_examples=json.loads(raw)['examples'];order=list(range(len(all_examples)));chosen=[];excluded=[]
  random.Random(SEED+int(hashlib.sha256(family.encode()).hexdigest()[:8],16)).shuffle(order)
  for ix in order:
   example=all_examples[ix];stem=norm(example['input'].split('\nOptions:')[0])
   if stem in big or any(len(t)>80 and t in stem for t in prior_texts) or norm(example['input']) in seen:
    excluded.append(ix);continue
   chosen.append(ix);seen.append(norm(example['input']))
   if len(chosen)==8:break
  if chosen!=receipt['selected_indices'] or excluded!=receipt['excluded_indices']:raise ValueError('Selection changed')
  cohort=[r for r in evaluation if r['partition']=='fresh_bbh' and r['family']==family]
  if len(cohort)!=8 or {r['provenance']['row_index'] for r in cohort}!=set(chosen):raise ValueError('Missing source cases')
  for r in cohort:
   ref=all_examples[r['provenance']['row_index']]
   if r['state']!=ref['input'] or str(r['expected']).lower()!=str(ref['target']).lower():raise ValueError('Source row/key altered')
   found=[]
   for line in r['state'].splitlines():
    if len(line)>4 and line[0]=='(' and line[2]==')' and line[1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and line[3].isspace():found.append((line[:3],line[3:].strip()))
   if found:
    if r['labels']!=[x[0] for x in found] or r['question']['criteria']!=dict(found):raise ValueError('Choice conversion')
   matched+=1
 keys=[digest({k:t[k] for k in ('state','question','labels')}) for t in evaluation]
 if len(keys)!=len(set(keys)):raise ValueError('Canonical duplicate')
 return {'checked_source_rows_and_keys':matched,'source_files':len(m['source_receipts']),'prior_reference_rows':len(old),
         'unique_test_inputs':len(keys),'selection_recreated':True,'limits':'Text/key and selection fidelity, not independent source-key truth or training-contamination exclusion.'}


def run(root):
 root=Path(root).resolve();out=root/'results';prepared=root/'countercase-prepared'
 m=json.loads((prepared/'manifest.json').read_text());evaluation=json.loads((prepared/'evaluation.json').read_text());arch=json.loads((prepared/'archived.json').read_text())
 if m['source_sha256']!=SOURCE or filehash(root/'countercase_v8.py')!=SOURCE:raise ValueError('Scientific source')
 if digest(evaluation)!=m['evaluation_hash'] or digest(arch)!=m['archived_hash']:raise ValueError('Evaluation provenance')
 jobs=json.loads((prepared/'jobs.json').read_text())
 if digest(jobs)!=m['jobs_hash']:raise ValueError('Jobs provenance')
 wanted={j['input']['id']:j for group in jobs for j in group};tasks={t['id']:t for t in evaluation}
 rows=json.loads((root/'all_records.json').read_text())
 if len(rows)!=len(tasks) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(tasks):raise ValueError('Incomplete cohort')
 error=0.;nv=0;new_readouts=0;ng=0;nt=0;scored=[];phase=[];fallbacks=collections.Counter();cuts=collections.Counter()
 for r in sorted(rows,key=lambda r:r['id']):
  t=tasks[r['id']];expected=wanted[r['id']]
  if r['input_sha256']!=digest(inputs(t)) or set(r['outputs'])!=set(expected['arms']) or sorted(r['execution_order'])!=sorted(expected['arms']):raise ValueError('Incomplete arm/input')
  preds={};measures={}
  for arm in ARMS:
   archived=(arm=='standard480' and t['partition']=='jevbench')
   obs=arch[r['id']] if archived else r['outputs'][arm]
   if not archived and obs['arm']!=arm:raise ValueError('Misnamed arm')
   label,e,v,s=replay(obs,t,archived);error=max(error,e);nv+=v
   if not archived:ng+=1;nt+=obs['trace']['output_tokens'];new_readouts+=v
   preds[arm]=label;measures[arm]={'seconds':s,'generated_tokens':obs['trace']['output_tokens'],'fallback':bool(v),'archive':archived}
   fallbacks[t['partition']+'/'+arm]+=v;cuts[t['partition']+'/'+arm]+=int(obs['trace']['hit_cap'])
  scored.append({'id':r['id'],'partition':t['partition'],'predictions':preds})
  phase.append({'id':r['id'],'partition':t['partition'],'metrics':measures})
 sys.path.insert(0,str(root/'reconstruction-inputs/benchmark/vendor'))
 from jevbench.scoring import score_label
 from jevbench.tasks import Task
 official=0
 for r in scored:
  if r['partition']=='jevbench':
   t=tasks[r['id']]
   for arm in ARMS:
    check=score_label(r['predictions'][arm],Task.from_dict(t))
    if bool(check['correct'])!=(r['predictions'][arm]==str(t['expected'])):raise ValueError('Official scoring mismatch')
    official+=1
 def metric(rr,arm):
  n=len(rr);k=sum(r['predictions'][arm]==str(tasks[r['id']]['expected']) for r in rr)
  return {'n':n,'correct':k,'accuracy':k/n if n else None}
 summary={'primary':'countercase480','version':m['version'],'new_generations':ng,'new_generated_tokens':nt,'groups':{},'comparisons':{},'countercase_vs_standard':{},'limits':m['limits'],'manifest':m}
 for cohort in ('jevbench','fresh_bbh'):
  rr=[r for r in scored if r['partition']==cohort]
  ps=[r for r in phase if r['partition']==cohort]
  summary['groups'][cohort]={arm:metric(rr,arm) for arm in ARMS}
  summary['comparisons'][cohort]=paired(rr,tasks,'countercase480','standard480')
  summary['countercase_vs_standard'][cohort]={arm:{'time':summarize([r['metrics'][arm]['seconds'] for r in ps]),'tokens':summarize([r['metrics'][arm]['generated_tokens'] for r in ps]),'fallbacks':fallbacks[cohort+'/'+arm],'hit_cap':cuts[cohort+'/'+arm]} for arm in ARMS}
 summary['by_family']={f:{arm:metric([r for r in scored if tasks[r['id']].get('family')==f and r['partition']=='fresh_bbh'],arm) for arm in ARMS} for f in sorted({t['family'] for t in evaluation if t['partition']=='fresh_bbh'})}
 summary['by_tier']={f:{arm:metric([r for r in scored if tasks[r['id']].get('_tier')==f],arm) for arm in ARMS} for f in ('easy','standard','hard')}
 public=[r for r in scored if r['partition']=='jevbench'];prior_none=[];prior_found=[];new_correct=[];after_none=[]
 for r in public:
  y=str(tasks[r['id']]['expected']);a=arch[r['id']]['all_v7_answers'];p=r['predictions']['countercase480']
  if a['long320']!=y and a['blind160']!=y:
   prior_none.append(r['id'])
   if p==y:new_correct.append(r['id'])
   else:after_none.append(r['id'])
  else:prior_found.append(r['id'])
 novelty={'v7_neither_correct':len(prior_none),'newly_supplied_correct_candidate':len(new_correct),'new_candidate_ids':new_correct,
          'still_no_correct_candidate':len(after_none),'two_prior_candidate_oracle_ceiling':len(prior_found),
          'three_candidate_oracle_ceiling':len(prior_found)+len(new_correct),
          'not_an_achieved_ensemble':True,'no_switch_policy_fitted':True}
 provenance=source_selection(root,evaluation,m)
 audit={'all_labels_replayed':len(rows)*2,'official_public_outcomes':official,'stored_vectors_checked':nv,'max_probability_error':error,
        'new_generations':ng,'new_generated_tokens':nt,'new_categorical_readouts':new_readouts,'old_standard_traces':len(arch),'complete':True,'scientific_source_sha256':SOURCE}
 if ng!=359 or official!=462:raise ValueError('Incorrect execution accounting')
 write(out/'all_predictions.json',scored);write(out/'summary.json',summary);write(out/'phase_sums.json',phase)
 write(out/'candidate_audit.json',novelty);write(out/'independent_audit.json',audit);write(out/'source_audit.json',provenance)
 print(json.dumps({'metrics':summary['groups'],'comparisons':summary['comparisons'],'candidate_audit':novelty,'audit':audit},indent=2))
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='data');a=p.parse_args();run(a.root)
