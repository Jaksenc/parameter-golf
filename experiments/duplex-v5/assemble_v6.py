"""Assembly only: no result-based selection or repeat of recorded failures."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from study import digest,save
ARMS=('direct','reasoning','program','binding')
def load_raw(root):
 out=[]
 for f in sorted(Path(root).rglob('records.jsonl')):
  out.extend(json.loads(x) for x in f.read_text().splitlines() if x.strip())
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','assemble']);a=ap.parse_args()
 rows=json.loads(Path('data/requests.json').read_text());manifest=json.loads(Path('data/manifest.json').read_text())
 if digest(rows)!=manifest['hash']:raise ValueError('data mismatch')
 byid={r['id']:r for r in rows};expected={(r['id'],m) for r in rows for m in ARMS};old=load_raw('original')
 oldkeys={(p['id'],p['arm']) for p in old}
 if len(oldkeys)!=len(old) or not oldkeys<=expected:raise ValueError('duplicate/unexpected original output')
 for p in old:
  if p['input_hash']!=digest({k:byid[p['id']][k] for k in ('evidence','question','bank')}):raise ValueError('changed original input')
 missing=expected-oldkeys;shards=sorted({i%16 for i,r in enumerate(rows) if any((r['id'],m) in missing for m in ARMS)})
 if not set(shards)<={2,11,13,14}:raise ValueError('Unexpected missing partition')
 if a.mode=='prepare':
  save('resume-existing.json',old)
  receipt={'original_records':len(old),'missing_pairs':[list(k) for k in sorted(missing)],'missing_shards':shards,'source_run':35553557091,'selection_uses_output_correctness':False,'recorded_invalid_answers_retained':True}
  save('resume-plan.json',receipt)
  with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('shards='+json.dumps(shards)+'\n')
  print(json.dumps(receipt),flush=True);return
 new=load_raw('resumed');newkeys={(p['id'],p['arm']) for p in new}
 if len(newkeys)!=len(new) or newkeys!=missing:raise ValueError('Resumed outputs must match missing pairs exactly; never overwrite a response')
 for p in new:
  if p['input_hash']!=digest({k:byid[p['id']][k] for k in ('evidence','question','bank')}):raise ValueError('changed resumed input')
 root=Path('evidence');root.mkdir(exist_ok=False)
 records=old+new
 if {(p['id'],p['arm']) for p in records}!=expected:raise ValueError('Incomplete final population')
 for s in range(16):
  ids={r['id'] for i,r in enumerate(rows) if i%16==s};pp=[p for p in records if p['id'] in ids];out=root/'assembled'/str(s);out.mkdir(parents=True)
  if len(pp)!=8:raise ValueError('Incorrect assembled partition')
  (out/'records.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in pp))
  save(out/'complete.json',{'complete':True,'hash':manifest['hash'],'shard':s,'records':8,'kind':'verified_union_of_original_and_missing_response_recovery','original_records':sum(p['id'] in ids for p in old),'recovered_records':sum(p['id'] in ids for p in new)})
 from binding_v6 import audit
 audit(root)
 # Histories copied after scoring so originals are not counted twice by recursive readers.
 shutil.copytree('original',root/'execution-history'/'original')
 shutil.copytree('resumed',root/'execution-history'/'resumed')
 shutil.copytree('data',root/'data')
 for f in ('binding_v6.py','study.py','numeric_boundary_fix.py','v6_direct_contract.py','resume_missing_v6.py','assemble_v6.py','V6-PREINFERENCE-REVIEW.json'):
  shutil.copy(f,root/f)
 save(root/'recovery-receipt.json',{'original_run':35553557091,'recovery_run':os.environ['GITHUB_RUN_ID'],'original_completed_responses':len(old),'new_completed_responses':len(new),'original_cancelled_shards':[2,11,13,14],'all_original_outputs_preserved':True,'repeated_invalid_or_capped_answers':0,'missing_pairs':[list(k) for k in sorted(missing)],'generation_function_unchanged':True,'time_qualification':'Four original workers exceeded the24-minute job budget. Aborted decoder attempts are not included in per-completed-response timings. Recovered cases may use different same-class ARM VMs; no matched end-to-end serving-cost claim.'})
 print(json.dumps({'complete':True,'records':len(records),'new_completions':len(new)}),flush=True)
if __name__=='__main__':main()
