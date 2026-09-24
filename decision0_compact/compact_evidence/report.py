"""Read-only result analysis; never executes or repairs model output."""
from __future__ import annotations
import json,math,statistics
from pathlib import Path
from .cases import digest
from .prompts import MODES

def q(xs,q):
    if not xs:return None
    a=sorted(xs);pos=(len(a)-1)*q;i=int(pos);j=min(i+1,len(a)-1)
    return a[i]+(a[j]-a[i])*(pos-i)

def analyze(records,reference_rows,allow_fixture=False):
    refs={r['id']:r for r in reference_rows};index={}
    for r in records:
      key=(r['id'],r['mode'])
      if key in index:raise ValueError('Duplicate case/mode')
      if r.get('backend_kind')!='qwen3.5-4b-pinned-native' and not allow_fixture:raise ValueError('Fixture outputs are not model results')
      if r['id'] not in refs or r['mode'] not in MODES:raise ValueError('Unknown output')
      if r['public_hash']!=digest(refs[r['id']]['public']):raise ValueError('Input fingerprint mismatch')
      if r['status']=='ok':
        d=r['decision'];labels=[o['id'] for o in refs[r['id']]['public']['options']]
        if d['labels']!=labels or len(d['probabilities'])!=len(labels) or len(d['logits'])!=len(labels):raise ValueError('Outcome alignment mismatch')
        if any(not math.isfinite(v) for v in d['logits']+d['probabilities']):raise ValueError('Nonfinite outputs')
        weights=[math.exp(z-max(d['logits'])) for z in d['logits']];probs=[v/sum(weights) for v in weights]
        if max(abs(a-b) for a,b in zip(probs,d['probabilities']))>1e-10:raise ValueError('Probability/logit mismatch')
        predicted=min(zip(labels,probs),key=lambda pair:(-pair[1],pair[0]))[0]
        if predicted!=d['predicted']:raise ValueError('Argmax mismatch')
      index[key]=r
    wanted={(i,m) for i in refs for m in MODES}
    if set(index)!=wanted:raise ValueError(f'Incomplete coverage: {len(index)} of {len(wanted)} case/modes; no full-score report')
    def correct(r):return r['status']=='ok' and r['decision']['predicted']==refs[r['id']]['reference']['answer']
    summary={}
    for mode in MODES:
      rr=[index[(i,mode)] for i in refs];success=[r for r in rr if r['status']=='ok']
      groups={};family={}
      for r in rr:
        ref=refs[r['id']];groups.setdefault(ref['source'],{})[ref['edit']]=r
        family.setdefault(ref['family'],[]).append(r)
      support_scores=[]
      for r in success:
        if mode in ('evidence','length_matched_control','compact_state'):
          ids=set(r['selection']['ids']);valid=r['selection']['valid'];sets=refs[r['id']]['reference']['sufficient_support']
          support_scores.append(valid and any(set(s)<=ids for s in sets))
      summary[mode]={'n':len(rr),'correct':sum(correct(r) for r in rr),'model_errors':len(rr)-len(success),
       'family_correct':{f:{'n':len(xs),'correct':sum(correct(x) for x in xs)} for f,xs in family.items()},
       'complete_three_view_groups':sum(all(correct(x) for x in g.values()) for g in groups.values()),
       'relevant_pairs_both_correct':sum(correct(g['original']) and correct(g['relevant']) for g in groups.values()),
       'irrelevant_prediction_stability':sum(g['original']['status']=='ok' and g['irrelevant']['status']=='ok' and g['original']['decision']['predicted']==g['irrelevant']['decision']['predicted'] for g in groups.values()),
       'support_sufficient':sum(support_scores) if support_scores else None,
       'structured_notes_valid':sum(r['selection']['valid'] is True for r in success) if mode in ('evidence','length_matched_control','compact_state') else None,
       'generation_cap_hits':sum(bool(r['generation'] and r['generation']['reached_cap']) for r in success),
       'generation_incomplete_at_cap':sum(bool(r['generation'] and r['generation']['reached_cap'] and not r['generation']['format_complete']) for r in success),
       'route_input_tokens':sum(r['route_input_tokens'] for r in success),
       'route_output_tokens':sum(r['route_output_tokens'] for r in success),
       'route_p50_seconds_successes_only':q([r['route_seconds'] for r in success],.5),
       'route_p95_seconds_successes_only':q([r['route_seconds'] for r in success],.95),
       'repairs_vs_baseline':sum(correct(r) and not correct(index[(r['id'],'baseline')]) for r in rr),
       'regressions_vs_baseline':sum(not correct(r) and correct(index[(r['id'],'baseline')]) for r in rr)}
    return {'status':'complete_saved_record_audit','fixture_only':allow_fixture,'cases':len(refs),
      'source_groups':len({r['source'] for r in refs.values()}),'records':len(index),'summary':summary,
      'model_promoted':False,'official_score':None,'notes':['Source cases are clustered and share finite task specifications.',
      'Source coverage does not certify model-generated inference text.', 'Random length-matched records can contain relevant evidence; coverage is measured, not assumed.',
      'Runtime values are measurements only after actual model execution; per-route costs count shared generation in each route.']}
