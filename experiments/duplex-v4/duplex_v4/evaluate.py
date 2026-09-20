"""One-way transfer evaluation after development selection has been locked.
Computes paired source-family resampling without treating eight views as IID.
No model, hyperparameter, seed or checkpoint is selected using transfer labels.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .io import DATA_HASH,REVISION,filehash,save_json,load_rows,merge_caches
from .train import load_adapter,metric,SEEDS

TRANSFER=('transfer_wording','transfer_domain','transfer_composition')

def paired_interval(rows,z,base,repetitions=3000):
    # Conditional sampling interval, not multiple-comparison correction or new-domain assurance.
    gold=np.array([r['target'] for r in rows]);pred=z.argmax(-1).numpy();bp=base.argmax(-1).numpy()
    p=z.float().softmax(-1);b=base.float().softmax(-1)
    ties=((p-p.max(-1,keepdim=True).values).abs()<=1e-12).sum(-1).numpy()>1
    bties=((b-b.max(-1,keepdim=True).values).abs()<=1e-12).sum(-1).numpy()>1
    ok=(pred==gold)&~ties;bok=(bp==gold)&~bties
    delta=ok.astype(float)-bok.astype(float)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['group']].append(i)
    vals=np.array([delta[v].mean() for _,v in sorted(groups.items())])
    rng=np.random.default_rng(47021)
    draws=vals[rng.integers(0,len(vals),size=(repetitions,len(vals)))].mean(axis=1)
    return {'source_families':len(vals),'mean_accuracy_difference':float(vals.mean()),
            'conditional_95_interval':np.quantile(draws,[.025,.975]).tolist(),
            'repairs':int((ok&~bok).sum()),
            'harms':int((~ok&bok).sum()),
            'qualification':'Paired source-family bootstrap within this generated split; not independent-source or simultaneous inference'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--cache',default='cache')
    ap.add_argument('--training',default='runs/training');ap.add_argument('--out',default='runs/transfer')
    args=ap.parse_args();torch.set_num_threads(4)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    training=Path(args.training);selection_path=training/'selection.json'
    selection=json.loads(selection_path.read_text())
    if not selection.get('locked_before_transfer') or selection['data_hash']!=DATA_HASH or selection['model_revision']!=REVISION:
        raise ValueError('Missing/incompatible locked selection')
    selection_hash=filehash(selection_path)
    rows=sum((load_rows(args.data,s) for s in TRANSFER),[])
    cache,receipts=merge_caches(sorted((Path(args.cache)/'transfer').glob('*.npz')),rows)
    if any(r['phase']!='transfer' for r in receipts):raise ValueError('Wrong cache phase')
    base=cache['z0'];results={};prediction_records=[]
    for split in TRANSFER:
        ii=[i for i,r in enumerate(rows) if r['split']==split];rr=[rows[i] for i in ii]
        results[split]={'baseline':metric(base[ii],rr),'candidates':{}}
    for spec in selection['results']:
        tag=f"{spec['arm']}-{spec['seed']}";path=training/tag/'selected.npz'
        if filehash(path)!=spec['selected_sha256']:raise ValueError('Selected checkpoint mutated')
        model=load_adapter(path,cache['x'].shape[-1],cache['r'].shape[-1])
        with torch.no_grad():z=model(cache)
        for split in TRANSFER:
            ii=[i for i,r in enumerate(rows) if r['split']==split];rr=[rows[i] for i in ii]
            results[split]['candidates'][tag]={**metric(z[ii],rr),**paired_interval(rr,z[ii],base[ii]),
                 'was_development_selected_method':spec['arm']==selection['winner']}
        for i,row in enumerate(rows):
            prediction_records.append({'model':tag,'id':row['id'],'group':row['group'],'split':row['split'],
                'input_hash':row['input_hash'],'target':row['target'],
                'baseline_logits':base[i].tolist(),'trained_logits':z[i].tolist()})
    if filehash(selection_path)!=selection_hash:raise RuntimeError('Selection changed during evaluation')
    save_json(root/'results.json',{'complete':True,'selection_sha256':selection_hash,
        'preselected_method':selection['winner'],'metrics':results,'model_promoted':False,
        'live_checkpoint_replay_completed':False,'data_hash':DATA_HASH,
        'limitations':['Controlled fresh generated splits, not external human-authored language',
          'Only 12 source families per transfer condition; 8 correlated rows each',
          'All candidate results reported; no transfer-based winner selection',
          'Full-network replay is a separate required validation step',
          'No JevBench score or official ranking computed']})
    with (root/'predictions.jsonl').open('w') as f:
        for p in prediction_records:f.write(json.dumps(p,sort_keys=True,allow_nan=False)+'\n')
    report=['# Duplex v4 transfer experiment','',f"Development-selected method: **{selection['winner']}**.",
       'No production model is promoted. This report contains completed cached-model evaluation, not yet live replay.',
       '', '| Split | Configuration | Correct | NLL | High-confidence errors |','|---|---|---:|---:|---:|']
    for split,values in results.items():
        for name,m in [('baseline',values['baseline'])]+list(values['candidates'].items()):
            report.append(f"| {split} | {name} | {m['correct']}/{m['n']} | {m['nll']:.4f} | {m['wrong_above_95']} |")
    report+=['','Every split has 12 correlated source families, not 96 independent situations. Candidate selection was locked on development. All methods and seeds remain visible.','',
       'Fresh external-language validation and the maintainer-held-out JevBench suite are still separate requirements.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))

if __name__=='__main__':main()
