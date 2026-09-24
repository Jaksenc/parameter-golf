"""Inspect one recorded decision. This command never invokes a language model."""
import argparse,json,math
from pathlib import Path

def inspect(root,index,model='unchanged',calibrated=False):
 root=Path(root);items=json.loads((root/'format-prepared/records.json').read_text());rows=json.loads((root/'all_records.json').read_text())
 matches=[x for x in rows if x['index']==index]
 if len(matches)!=1:raise ValueError('Index is not an evaluated request')
 row=matches[0];item=items[index]
 if model not in row['outputs']:raise ValueError('Unknown recorded model')
 t=1.
 if calibrated:
  if item['split']=='retention':raise ValueError('Event calibration is not authorized for unrelated retention tasks')
  t=json.loads((root/'results/calibration_grid.json').read_text())['temperatures'][model]
 z=row['outputs'][model]['logits'];mx=max(z);w=[math.exp((v-mx)/t) for v in z];p=[v/sum(w) for v in w];labels=item['input']['labels']
 return {'id':row['id'],'model':model,'answer':labels[max(range(len(p)),key=lambda i:p[i])],
         'probabilities':dict(zip(labels,p)),'temperature':t,'mode':'recorded_replay','new_neural_calls':0,
         'input_sha256':row['input_sha256'],'general_correctness_verified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--index',type=int,required=True);p.add_argument('--model',default='unchanged');p.add_argument('--calibrated',action='store_true');a=p.parse_args()
 print(json.dumps(inspect(a.root,a.index,a.model,a.calibrated),indent=2,allow_nan=False))
