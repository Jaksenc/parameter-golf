"""Corrected primary fit; no changes chosen from benchmark outcomes.
Removes duplicate canonical inputs within/across training/dev/calibration.
The uncorrected automatic run is diagnostic only, not the final primary result.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
import reconstruct_v1 as experiment

def dedup_protocol(rows):
 seen={};keep=[];drop=[];counts=Counter()
 if any(r['partition'] not in ('train','development','calibration') for r in rows):raise ValueError('Invalid fitting partition')
 for part in ('train','development','calibration'):
  for r in rows:
   if r['partition']!=part:continue
   x=experiment.input_only(r);x.pop('id');key=experiment.digest(x);y=experiment.target(r)
   if key in seen:
    old_id,old_y=seen[key]
    if old_y!=y:raise ValueError('Conflicting labels for duplicate input')
    drop.append({'id':r['id'],'duplicate_of':old_id,'partition':part,'input_sha256':key})
   else:seen[key]=(r['id'],y);keep.append(r['id']);counts[part]+=1
 return {'version':'dedup-before-outcomes-v1','selection_rule':'First canonical input; train, development, calibration precedence','uses_model_predictions':False,'uses_benchmark_outcomes':False,'retained_ids':keep,'retained_source_examples':dict(counts),'dropped':drop,'training_input_sha256':experiment.digest(rows),'extraction_source_sha256':experiment.filehash(experiment.__file__)}

def run(root,features,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=json.loads((root/'training.json').read_text());protocol=dedup_protocol(rows)
 if protocol['retained_source_examples']!={'train':301,'development':73,'calibration':73}:raise ValueError('Unexpected corrected population')
 experiment.write(out/'split_correction.json',protocol);builder=experiment.build_training_batch;fitter=experiment.fit_head;keep=set(protocol['retained_ids'])
 def filtered(a,b):
  fit,bench=builder(a,b);fit=[r for r in fit if r['id'] in keep]
  if len(fit)!=2*len(keep):raise ValueError('Incomplete corrected features')
  return fit,bench
 def receipt(fit,path):
  delta,cfg=fitter(fit,path);cfg['split_correction']={k:v for k,v in protocol.items() if k not in ('retained_ids','dropped')};cfg['source_examples_removed']=len(protocol['dropped']);experiment.write(Path(path)/'head_config.json',cfg);return delta,cfg
 try:
  experiment.build_training_batch=filtered;experiment.fit_head=receipt;experiment.finalize(root,features,out)
 finally:experiment.build_training_batch=builder;experiment.fit_head=fitter
 p=out/'results.json';r=json.loads(p.read_text());r['protocol_correction']=protocol;r['limitations']=['Qwen3.5 public weights and independently trained head, not recovered Jev weights.','231 public JevBench decisions; 303 nonpublic unavailable. No official composite or submission.','301 unique training, 73 development, 73 calibration inputs; two orderings per input are augmentations.','Public benchmark previously inspected and evaluated; not a fresh sealed holdout.','Exact canonical split deduplication, not a semantic or pretraining contamination guarantee.','One seed, frozen backbone, small sample; no recovered RLCD.','SciQ CC-BY-NC-3.0 supervision: research-only.','Questions evaluated separately; no verified shared-prefix speedup.'];experiment.write(p,r)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='reconstruction-inputs');p.add_argument('--features',default='features');p.add_argument('--out',default='results');p.add_argument('--protocol-only',action='store_true');a=p.parse_args()
 if a.protocol_only:experiment.write(Path(a.out)/'split_correction.json',dedup_protocol(json.loads((Path(a.root)/'training.json').read_text())))
 else:run(a.root,a.features,a.out)
if __name__=='__main__':main()
