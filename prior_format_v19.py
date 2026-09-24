"""Pre-outcome-declared input-blind option-count prior, fitted only on training targets."""
import argparse,json,math
from fractions import Fraction as F
from pathlib import Path
import numpy as np

def fit(records,indices):
 groups={}
 for i in indices:
  r=records[i]
  if r['split']!='fit':raise ValueError('Non-fit data in prior estimation')
  q=list(map(F,r['target']));k=len(q)
  if sum(q)!=1 or min(q)<0:raise ValueError('Invalid fit law')
  groups.setdefault(k,[]).append(q)
 return {k:[sum(q[j] for q in group)/len(group) for j in range(k)] for k,group in groups.items()}

def metrics(p,q):
 if len(p)!=len(q) or sum(p)!=1 or min(p)<0:raise ValueError('Invalid prior')
 # Current fit means are positive; fail rather than silently clip an unsupported zero.
 if any(a==0 and b>0 for a,b in zip(p,q)):raise ValueError('Infinite prior log loss')
 return {'ce':-sum(float(b)*math.log(float(a)) for a,b in zip(p,q) if b),
         'squared':sum(float(a-b)**2 for a,b in zip(p,q)),
         'tvd':sum(float(abs(a-b)) for a,b in zip(p,q))/2,
         'correct':int(max(range(len(p)),key=lambda i:p[i])==max(range(len(q)),key=lambda i:q[i]))}

def run(root):
 root=Path(root);r=json.loads((root/'format-prepared/records.json').read_text());m=json.loads((root/'format-prepared/manifest.json').read_text())
 output={'fitting':'Only target vectors of actual scheduled fit examples; no source facts or test/calibration targets enter fit.','by_seed':{},'aggregate':{},'scope':'Pre-outcome-declared secondary null; not a model policy, selector, or replacement primary.'}
 for seed,arms in m['schedules'].items():
  p=fit(r,arms['single']);other=fit(r,arms['varied'])
  if p!=other:raise ValueError('Unmatched target priors')
  cells={}
  for fmt in ('prose','table'):
   for noise in (False,True):
    ids=[i for i in m['evaluation_indices'] if r[i]['split']=='test' and r[i]['format']==fmt and r[i]['noise']==noise and r[i]['variant']=='base']
    vals=[metrics(p[len(r[i]['target'])],list(map(F,r[i]['target']))) for i in ids]
    if len(vals)!=32:raise ValueError('Wrong base-world population')
    cells[f'{fmt}_noise{int(noise)}']={k:sum(v[k] for v in vals)/len(vals) for k in vals[0]}
  output['by_seed'][seed]={'exact_priors':{str(k):list(map(str,v)) for k,v in p.items()},'fit_presentations':len(arms['single']),'metrics':cells}
 for cell in next(iter(output['by_seed'].values()))['metrics']:
  vs=[x['metrics'][cell] for x in output['by_seed'].values()];output['aggregate'][cell]={k:sum(v[k] for v in vs)/len(vs) for k in vs[0]}
 (root/'results').mkdir(exist_ok=True)
 (root/'results/prior_control.json').write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False))
 print(json.dumps(output['aggregate'],indent=2));return output
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();run(a.root)
