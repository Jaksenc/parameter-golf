"""Independently recompute decisions and metrics from public workflow logs."""
from __future__ import annotations
import hashlib,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path

def audit(text):
    entries=[]
    for line in text.splitlines():
        start=line.find('{"kind":')
        if start<0:continue
        try:obj=json.loads(line[start:])
        except json.JSONDecodeError:continue
        if isinstance(obj,dict):entries.append(obj)
    protocol=next(e for e in entries if e.get('kind')=='protocol')
    summary=next(e for e in entries if e.get('kind')=='summary')
    loaded=next(e for e in entries if e.get('kind')=='model_loaded')
    preds=[e for e in entries if e.get('kind')=='prediction']
    assert len(preds)==protocol['cases']==128
    assert len({e['id'] for e in preds})==128
    assert [e['index'] for e in preds]==list(range(128))
    assert summary['data_hash']==protocol['data_hash']
    max_delta=0.;groups=defaultdict(list)
    for row in preds:
        assert row['status'] in ('ok','tie'),row
        logits=row['logits'];peak=max(logits);exps=[math.exp(x-peak) for x in logits];probs=[v/sum(exps) for v in exps]
        max_delta=max(max_delta,max(abs(a-b) for a,b in zip(probs,row['probabilities'],strict=True)))
        near=[i for i,p in enumerate(probs) if abs(p-max(probs))<=1e-10]
        expected=near[0] if len(near)==1 else None
        assert row['prediction']==expected
        assert row['status']==('ok' if expected is not None else 'tie')
        groups[row['id'].split('-')[0]].append(row)
    assert max_delta<1e-12
    metrics={}
    for family,rows in groups.items():
        result={'n':len(rows),'correct':sum(r['prediction']==r['target'] for r in rows),'ties':sum(r['status']=='tie' for r in rows),'nll':sum(-math.log(max(r['probabilities'][r['target']],1e-30)) for r in rows)/len(rows)}
        original=summary['metrics'][family]
        assert result['n']==original['n'] and result['correct']==original['correct']
        assert abs(result['nll']-original['nll_on_successes'])<1e-12
        metrics[family]=result
    byid={p['id']:p for p in preds};rules=[];evidence=[]
    for p in groups['counterfactual']:
        _,block,bits,rule=p['id'].split('-',3);correct=p['prediction']==p['target']
        if rule in ('amber','dotted'):
            other=byid[f'counterfactual-{block}-{bits}-not_{rule}'];rules.append(correct and other['prediction']==other['target'])
        axis=0 if 'amber' in rule else 1
        if bits[axis]=='0':
            changed=list(bits);changed[axis]='1';other=byid[f'counterfactual-{block}-{"".join(changed)}-{rule}'];evidence.append(correct and other['prediction']==other['target'])
    assert len(rules)==len(evidence)==32
    pair_metrics={'rule_both_correct':sum(rules),'rule_pairs':32,'evidence_both_correct':sum(evidence),'evidence_pairs':32}
    assert pair_metrics==summary['metrics']['counterfactual_pairs']
    return {'model':summary['model'],'revision':summary['revision'],'data_hash':summary['data_hash'],'parameters':loaded['parameters'],'dtype':loaded['dtype'],'metrics':metrics,'pair_metrics':pair_metrics,'total_correct':sum(v['correct'] for v in metrics.values()),'total_cases':128,'score_arithmetic_max_difference':max_delta,'prediction_index_target_selected':[[p['index'],p['target'],p['prediction']] for p in preds],'summary_wall_seconds':summary['wall_seconds'],'request_time_median_seconds':statistics.median(p['seconds'] for p in preds),'request_time_range_seconds':[min(p['seconds'] for p in preds),max(p['seconds'] for p in preds)],'dependencies':summary['dependencies'],'sources':protocol['sources'],'job_log_sha256':hashlib.sha256(text.encode()).hexdigest()}

if __name__=='__main__':
    for filename in sys.argv[1:]:print(json.dumps(audit(Path(filename).read_text(encoding='utf-8-sig')),sort_keys=True,allow_nan=False))
