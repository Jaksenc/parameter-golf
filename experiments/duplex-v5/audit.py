"""Read completed records only. No inference, policy fitting, or silent case dropping."""
from __future__ import annotations
import argparse,json,math,statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
from study import digest,save,ARMS,program_value,final_number

def sign_test(w,l):
 n=w+l
 return sum(math.comb(n,k) for k in range(w,n+1))/2**n if n else 1.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--records',default='shards');ap.add_argument('--out',default='results');a=ap.parse_args()
 d=Path(a.data);out=Path(a.out);manifest=json.loads((d/'manifest.json').read_text());inputs=json.loads((d/'requests.json').read_text());labels=json.loads((d/'labels.json').read_text());ret=json.loads((d/'retention.json').read_text())
 assert digest(inputs)==manifest['target_hash'] and digest(labels)==manifest['labels_hash'] and digest(ret)==manifest['retention_hash']
 records=[];completions=[]
 for p in Path(a.records).rglob('records.jsonl'):records += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 for p in Path(a.records).rglob('complete.json'):completions.append(json.loads(p.read_text()))
 if len(completions)!=16 or {c['shard'] for c in completions}!=set(range(16)) or not all(c['complete'] for c in completions):raise RuntimeError('incomplete_shards')
 targets={};rpred={};idx={r['id']:r for r in inputs};ridx={r['id']:r for r in ret}
 for r in records:
  if r['kind']=='target':
   key=(r['id'],r['arm'])
   if key in targets or r['input_hash']!=digest(idx[r['id']]):raise ValueError('duplicate_or_changed_target')
   if r['status']=='ok':
    val=program_value(r['raw'],idx[r['id']]['bank'])[0] if r['arm']=='program' else final_number(r['raw'])
    if str(val)!=r['value'] or r['cap_hit']:raise ValueError('invalid_numeric_record')
   targets[key]=r
  elif r['kind']=='retention':
   if r['id'] in rpred:raise ValueError('duplicate_retention')
   raw=ridx[r['id']];inp={k:raw[k] for k in ('state','question','options')}
   if r['input_hash']!=digest(inp):raise ValueError('changed_retention_input')
   z=r['logits'];v=[math.exp(x-max(z)) for x in z];p=[x/sum(v) for x in v]
   if max(abs(x-y) for x,y in zip(p,r['probabilities'],strict=True))>1e-9:raise ValueError('probability_mismatch')
   order=sorted(range(len(p)),key=p.__getitem__,reverse=True);choice=order[0] if p[order[0]]-p[order[1]]>1e-12 else None
   if choice!=r['choice']:raise ValueError('choice_mismatch')
   rpred[r['id']]=r
  else:raise ValueError('unknown_record')
 expected={(r['id'],arm) for r in inputs for arm in ARMS}
 if set(targets)!=expected or set(rpred)!=set(ridx):raise RuntimeError('missing_or_extra_case')
 def correct(r):return r['status']=='ok' and round(float(Fraction(r['value'])),5)==round(float(Fraction(labels[r['id']]['value'])),5)
 def lenient(r):
  if r['status']!='ok':return False
  y=float(Fraction(labels[r['id']]['value']));v=float(Fraction(r['value']))
  return abs(v-y)<=max(1e-5,abs(y)*.005)
 metrics={};paired={}
 for split in ('development','evaluation','all'):
  ids=[r['id'] for r in inputs if split=='all' or r['split']==split];metrics[split]={}
  for arm in ARMS:
   rr=[targets[(rid,arm)] for rid in ids]
   metrics[split][arm]={'n':len(ids),'correct_round5':sum(correct(r) for r in rr),'within_half_percent':sum(lenient(r) for r in rr),'valid':sum(r['status']=='ok' for r in rr),'cap_hits':sum(r['cap_hit'] for r in rr),'median_seconds':statistics.median(r['seconds'] for r in rr),'median_output_tokens':statistics.median(r['output_tokens'] for r in rr),'failure_reasons':dict(Counter(r.get('error','') for r in rr if r['status']!='ok'))}
  paired[split]={}
  for other in ('direct','reasoning'):
   wins=sum(correct(targets[(i,'program')]) and not correct(targets[(i,other)]) for i in ids);loss=sum(not correct(targets[(i,'program')]) and correct(targets[(i,other)]) for i in ids)
   paired[split]['program_vs_'+other]={'repairs':wins,'harms':loss,'net':wins-loss,'paired_one_sided_p':sign_test(wins,loss)}
 retention={}
 for family in ('nli','paraphrase','qa'):
  retention[family]={}
  for split in ('replay','sentinel'):
   sub=[r for r in ret if r['family']==family and r['split']==split]
   retention[family][split]={'n':len(sub),'correct':sum(rpred[r['id']]['choice']==r['target'] for r in sub),'nll':statistics.mean(-math.log(max(rpred[r['id']]['probabilities'][r['target']],1e-30)) for r in sub),'targets':dict(Counter(r['target'] for r in sub))}
 passed=all(paired['evaluation']['program_vs_'+arm]['paired_one_sided_p']<=.025 and paired['evaluation']['program_vs_'+arm]['net']>0 for arm in ('direct','reasoning')) and metrics['evaluation']['program']['valid']/24>=.9
 result={'complete':True,'target_predictions':len(targets),'retention_predictions':len(rpred),'metrics':metrics,'paired':paired,'retention':retention,'qualification_rule':'program must improve both controls on evaluation with Bonferroni one-sided p<=.025 and >=90% valid programs; narrow diagnostic only','qualified_for_followup_not_deployment':passed,'training_steps':0,'production_promoted':False,'data_manifest':manifest,'limits':['Native numerical generation diagnostic, not the bounded JevBench classifier or full FinQA leaderboard','Same evidence and quantity bank across numerical arms; token caps differ for cheap direct answer','Exact source provenance plus arithmetic does not prove correct semantic translation','No probabilities for numerical generation; no confidence invented','24 evaluation cases and selection filters limit power and generality']}
 save(out/'results.json',result);save(out/'records.json',records);save(out/'completions.json',completions)
 lines=['# Duplex v5 teacher qualification','',f'Completed {len(targets)} numerical predictions and {len(rpred)} retention predictions. No parameter updates.','', '| Arm | Development /8 | Evaluation /24 | Evaluation valid | Evaluation median seconds |','|---|---:|---:|---:|---:|']
 for arm in ARMS:
  s=metrics['evaluation'][arm];lines.append(f'| {arm} | {metrics["development"][arm]["correct_round5"]} | {s["correct_round5"]} | {s["valid"]}/24 | {s["median_seconds"]:.2f} |')
 lines+=['',f'Qualified for a follow-up: **{passed}**. No automatic model training or deployment.','',json.dumps(paired,indent=2),'','See results.json for relaxed-precision sensitivity, failures, retention references, provenance and limitations.']
 out.mkdir(exist_ok=True,parents=True);(out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(result,allow_nan=False),flush=True)
if __name__=='__main__':main()
