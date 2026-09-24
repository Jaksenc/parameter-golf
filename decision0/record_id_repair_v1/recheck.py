"""Read-only secondary evidence-ID syntax check; primary results stay unchanged.

Only rows whose complete ID list fails JSON solely through leading zeros are
rescored. Selection is syntax-based, never based on whether the answer was right.
No new note generation or weight update occurs. Every other primary record is
retained for the secondary all-case summary. This is an adaptive engineering fix.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'compact_evidence_v1'))
import compact as c
from record_ids import resolve

P={'id':'decision0-leading-zero-recheck-v1','primary_run':36012861468,'primary_source':'52265ba4c256018dc87ed28576401067086e913c','selection':'strict-JSON failure plus complete valid decimal-ID-list syntax with leading zeros; no correctness filtering','new_note_generations':0,'weight_updates':0,'type':'post-hoc parser repair; not a new trained model','shards':4}

def candidates(primary):
 rows={r['id']:r for r in c.lines(primary/'records.jsonl') if r['mode']=='evidence_ids'}
 cases={r['id']:r for r in json.loads((primary/'cases.json').read_text())}
 gens={r['id']:r for r in c.lines(primary/'generations.jsonl') if r['mode']=='evidence_ids'}
 out=[]
 for rid in sorted(rows):
  raw=gens[rid]['text'];ids,_,error=c.select_ids(raw,cases[rid])
  if ids is not None:continue
  try:parsed=resolve(raw,c.parsed_records(cases[rid]))
  except ValueError:continue
  if not parsed['leading_zero_normalized']:continue
  out.append({'id':rid,'parsed':parsed})
 return out,rows,cases,gens

def run(args):
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
 from flow_study import Runtime
 p=Path(args.out);p.mkdir(parents=True,exist_ok=False);primary=Path(args.primary)
 repairs,old,cases,gens=candidates(primary)
 c.save(p/'protocol.json',P);c.save(p/'selection.json',repairs)
 subset=repairs[args.shard::P['shards']]
 if not subset:
  (p/'records.jsonl').write_text('');c.save(p/'receipt.json',{'records':0,'sha256':c.sha(p/'records.jsonl'),'no_model_loaded':True,'protocol_hash':c.digest(P)});return
 rt=Runtime();c.save(p/'runtime.json',rt.meta)
 # An arbitrary direct baseline, fixed by ID ordering, verifies the new worker.
 base=sorted((r for r in c.lines(primary/'records.jsonl') if r['mode']=='direct'),key=lambda r:r['id'])[0]
 replay=rt.score(c.public(cases[base['id']]),'baseline');error=max(abs(a-b) for a,b in zip(replay['logits'],base['logits']))
 if error>0.001 or replay['predicted']!=base['predicted']:raise RuntimeError('Baseline replay failed')
 c.save(p/'baseline_replay.json',{'case_id':base['id'],'maximum_logit_error':error,'predicted_same':True,'scope':'one baseline example on each active recovery worker, not the complete test set'})
 written=[]
 with (p/'records.jsonl').open('w') as f:
  for repair in subset:
   rid=repair['id'];case=cases[rid];gen=gens[rid];note=repair['parsed']['evidence']
   rec=rt.score(c.with_note(case,note),'baseline')
   rec.update({'mode':'evidence_ids_syntax_repaired','source':case['source'],'family':case['family'],'edit':case['edit'],'expected':case['expected'],'correct':rec['predicted']==case['expected'],'generation_input_tokens':gen['input_tokens'],'generation_output_tokens':gen['generated_tokens'],'accounted_seconds':rec['request_seconds']+gen['seconds'],'generation_reused_for_control':False,'selected_ids':repair['parsed']['ids'],'selection_error':None,'note':note,'note_input_target':None,'previous_predicted':old[rid]['predicted'],'previous_correct':old[rid]['correct'],'syntax_repaired':True})
   written.append(rec);f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush()
   print(json.dumps({'id':rid,'records':len(written),'subset':len(subset)}),flush=True)
 c.save(p/'receipt.json',{'records':len(written),'sha256':c.sha(p/'records.jsonl'),'protocol_hash':c.digest(P),'new_note_generation':False,'weight_updates':0})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--primary',type=Path,required=True);a.add_argument('--out',required=True);a.add_argument('--shard',type=int,choices=range(4),required=True);args=a.parse_args();run(args)
