"""Independent tensor/optimizer and Torch-float64 checks of the completed run."""
from pathlib import Path
import argparse,json,hashlib,math
from fractions import Fraction
import numpy as np
import torch
from safetensors.torch import load_file

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,sort_keys=True,indent=2,allow_nan=False))
def audit(root):
 root=Path(root);raw=read(root/'all_records.json');items=read(root/'format-prepared/records.json');metrics=read(root/'results/all_metrics.json')
 fits=read(root/'results/calibration_grid.json');models=[];checks=[]
 for folder in sorted((root/'models').glob('format-v19-model-*')):
  c=read(folder/'complete-32.json');last=read(folder/'full-state/latest.json');file=folder/'full-state'/last['file']
  if sha(file)!=last['sha256'] or file.stat().st_size!=last['bytes']:raise ValueError('Corrupt full state')
  state=torch.load(file,map_location='cpu',weights_only=True)
  if state['completed_steps']!=32:raise ValueError('Wrong endpoint')
  counters={int(v['step']) for v in state['optimizer']['state'].values()}
  if counters!={32} or len(state['optimizer']['state'])!=64:raise ValueError('Optimizer mismatch')
  factors=load_file(str(folder/'adapter-32.safetensors'))
  if set(factors)!=set(state['model']) or any(not torch.equal(v,state['model'][k]) for k,v in factors.items()):raise ValueError('Factors/state mismatch')
  if any(not torch.isfinite(v).all() for v in factors.values()):raise ValueError('Nonfinite factors')
  if sum(v.numel() for v in factors.values())!=1507328:raise ValueError('Layout changed')
  prior=load_file(str(folder/'adapter-8.safetensors'))
  changed=sum(not torch.equal(factors[k],prior[k]) for k in factors)
  if changed!=64:raise ValueError('Unchanged factor between8 and32')
  models.append({'name':folder.name,'optimizer_step':32,'optimizer_parameter_states':64,'changed_factor_tensors':changed,'full_state_sha256':last['sha256'],'adapter_sha256':sha(folder/'adapter-32.safetensors')})
 metricerr=0.;griderr=0.;count=0
 for row in raw:
  i=row['index'];q=torch.tensor([float(Fraction(x)) for x in items[i]['target']],dtype=torch.float64)
  for key,o in row['outputs'].items():
   for scale in ('raw','calibrated'):
    t=fits['temperatures'][key] if scale=='calibrated' and items[i]['split']!='retention' else 1.
    z=torch.tensor(o['logits'],dtype=torch.float64)/t;l=torch.log_softmax(z,0);p=l.exp()
    vals={'ce':float(-(q*l).sum()),'squared':float(((p-q)**2).sum()),'tvd':float(torch.abs(p-q).sum()/2)}
    for k,v in vals.items():metricerr=max(metricerr,abs(v-metrics[scale][str(i)][key][k]))
    count+=1
 cal=[row for row in raw if items[row['index']]['split']=='calibration']
 for key in fits['temperatures']:
  losses=[]
  for t in fits['grid']:
   values=[]
   for row in cal:
    q=torch.tensor([float(Fraction(x)) for x in items[row['index']]['target']],dtype=torch.float64)
    l=torch.log_softmax(torch.tensor(row['outputs'][key]['logits'],dtype=torch.float64)/t,0)
    values.append(float(-(q*l).sum()))
   losses.append(sum(values)/len(values))
  griderr=max(griderr,max(abs(x-y) for x,y in zip(losses,fits['losses'][key])))
  if fits['grid'][int(np.argmin(losses))]!=fits['temperatures'][key]:raise ValueError('Different calibration optimum')
 if metricerr>1e-12 or griderr>1e-12 or len(models)!=6:raise ValueError('Arithmetic discrepancy')
 result={'complete':True,'models':models,'metric_vectors_checked':count,'maximum_metric_difference':metricerr,
         'calibration_models':len(fits['temperatures']),'calibration_grid_points':len(fits['grid']),'maximum_grid_loss_difference':griderr,
         'scope':'Independent recorded-data tensor/arithmetic checks, not another model inference or optimizer run.'}
 save(root/'results/independent_state_and_metrics.json',result);print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();audit(a.root)
