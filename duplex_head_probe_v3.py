"""Actual-model equivalence/speed diagnostic, not a leaderboard rerun."""
import json,hashlib,time,statistics
from pathlib import Path
from duplex_native_head_v3 import scoped_head
from duplex_learning_v3 import encode,prompt
from semantic_train_v1 import Runtime
from semantic_data_v1 import dataset,digest

def main():
    root=Path('head-v3');root.mkdir(exist_ok=False);rt=Runtime();torch=rt.torch
    rows=[r for r in dataset() if r['split']=='train'][::9]
    def measured(row,lean):
        start=time.perf_counter();enc=encode(rt,row)
        with torch.inference_mode():
            if lean:
                with scoped_head(rt.model,enc[1]):z=rt.model(**enc[0],logits_to_keep=1,use_cache=False).logits[0,-1].float()
            else:z=rt.forward(enc)
            result=z.cpu().tolist()
        return result,time.perf_counter()-start
    measured(rows[0],False);measured(rows[0],True)
    records=[]
    for repeat in range(3):
        for i,row in enumerate(rows):
            pair={}
            for lean in ((False,True) if (repeat+i)%2==0 else (True,False)):
                z,seconds=measured(row,lean);pair['lean' if lean else 'full']={'logits':z,'seconds':seconds}
            error=max(abs(a-b) for a,b in zip(pair['full']['logits'],pair['lean']['logits'],strict=True))
            if error>2e-5:raise RuntimeError('Restricted/full projection mismatch '+str(error))
            record={'id':row['id'],'repeat':repeat,'input_hash':digest(prompt(row)),'max_logit_difference':error,**pair};records.append(record)
            with (root/'pairs.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
    full=[r['full']['seconds'] for r in records];lean=[r['lean']['seconds'] for r in records]
    result={'complete':True,'pairs':len(records),'cases':len(rows),'benchmark_calls':len(records)*2,'warmup_calls':2,'maximum_logit_difference':max(r['max_logit_difference'] for r in records),'median_full_seconds':statistics.median(full),'median_lean_seconds':statistics.median(lean),'ratio_of_medians':statistics.median(full)/statistics.median(lean),'median_paired_ratio':statistics.median(a/b for a,b in zip(full,lean)),'model':rt.receipt,'no_training':True,'qualification':'6 synthetic development inputs, 3 repeats, alternating order, one shared ARM CPU model. Small serving microbenchmark, not full JevBench/Mac/production throughput.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True));print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
