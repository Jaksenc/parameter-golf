"""Read-only evaluation of the last saved microfit checkpoint, not missing step 32."""
from __future__ import annotations
import argparse,copy,json,sys,traceback
from pathlib import Path
import microfit
p=microfit.parent
EXPECTED='3c2c71faa4baeba8356711535c6ab797c06af121a59ab2ee0151d294a1bd93ed'
PROTOCOL={'id':'decision0-microfit-recovery-v1','source_run':35771406144,
 'checkpoint_sha256':EXPECTED,'saved_checkpoint_update':24,'recorded_training_updates':32,
 'checkpoint_note':'Step 32 was recorded, but its evaluation and checkpoint save did not finish. The last recovery save occurred at step 24.',
 'weight_updates_in_this_run':0,'selection':'only recoverable checkpoint; not selected by accuracy','benchmark_calls':0}

def run(args):
 from safetensors.torch import load_file
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 checkpoint=Path(args.checkpoint)
 assert p.filehash(checkpoint)==EXPECTED
 p.save(out/'protocol.json',PROTOCOL)
 rt=p.Runtime();ad=p.Adapter(rt,microfit.PROTOCOL['seed']);p.save(out/'runtime.json',rt.meta)
 state=load_file(str(checkpoint));assert len(state)==64
 ad.factors.load_state_dict(state)
 rows=microfit.examples('seen')+microfit.examples('unseen');records=[]
 for variant in ('base','recovery24'):
  ad.enabled=variant!='base'
  for row in rows:
   for order in ('original','reversed'):
    v=copy.deepcopy(row)
    if order=='reversed':v['options'].reverse()
    rec=rt.score(v,'baseline');rec.update({'variant':variant,'split':'seen' if row['id'].startswith('seen-') else 'unseen','order':order,'expected':row['expected'],'correct':rec['predicted']==row['expected']})
    records.append(rec)
    p.emit(phase='recovery24',variant=variant,done=len(records),total=24)
 ad.enabled=False;probe=rt.score(rows[0],'baseline')['logits'];old=next(r['logits'] for r in records if r['variant']=='base' and r['id']==rows[0]['id'] and r['order']=='original')
 err=max(abs(a-b) for a,b in zip(probe,old));assert err<=1e-4
 (out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
 p.save(out/'receipt.json',{'records':24,'records_sha256':p.filehash(out/'records.jsonl'),'checkpoint_sha256':EXPECTED,'base_restore_error':err,'weight_updates':0,'step32_performance_unknown':True})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True);args=a.parse_args()
 try:run(args)
 except Exception as e:
  out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'FAILED.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()}));raise
