"""Fixed-exposure v4 study. Training consumes ONLY train+development caches.

No new source/answer selection, temperature fitting, test-set tuning, or implicit
model promotion. Baseline competes with every fixed arm on development NLL.
"""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .io import (DATA_HASH,REVISION,save_json,load_rows,merge_caches,filehash)
from .terminal import TerminalAdapter,baseline_logits
from .objectives import ARMS,semantic_losses
from .semantics import digest

SEEDS=(41,73,101)
STEPS=400
INTERVAL=20
LR=0.0005

def index_families(rows):
    group=defaultdict(list)
    for i,r in enumerate(rows):group[r['group']].append((r['view'],i))
    if any(sorted(v for v,_ in group[g])!=list(range(8)) for g in group):raise ValueError('Incomplete source family')
    return torch.tensor([[i for _,i in sorted(group[g])] for g in sorted(group)])

def metric(z,rows):
    gold=torch.tensor([r['target'] for r in rows]);p=z.float().softmax(-1)
    ties=((p-p.max(-1,keepdim=True).values).abs()<=1e-12).sum(-1)>1
    ok=(p.argmax(-1)==gold)&~ties
    confidence=p.max(-1).values
    g=index_families(rows)
    return {'n':len(rows),'source_families':len(g),'correct':int(ok.sum()),'ties':int(ties.sum()),
        'nll':float(F.cross_entropy(z.float(),gold)),
        'brier':float((p-F.one_hot(gold,6)).square().sum(-1).mean()),
        'wrong_above_95':int(((~ok)&(confidence>.95)).sum()),
        'complete_families_correct':int(ok[g].all(-1).sum()),
        'invariant_pairs_both_correct':int((ok[g[:,0,None]]&ok[g[:,1:6]]).sum()),
        'invariant_pair_n':len(g)*5,
        'change_pairs_both_correct':int((ok[g[:,0,None]]&ok[g[:,6:]]).sum()),
        'change_pair_n':len(g)*2}

def slice_cache(cache,indices):
    return {k:(v[indices] if k in ('x','r','y','z0') else v) for k,v in cache.items()}

def save_adapter(path,adapter):
    # Tensor-only npz, no Python-pickle loading.
    with Path(path).open('wb') as f:
        np.savez_compressed(f,a=adapter.a.detach().cpu().numpy(),b=adapter.b.detach().cpu().numpy())

def load_adapter(path,input_size,hidden_size):
    model=TerminalAdapter(input_size,hidden_size)
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'a','b'}:raise ValueError('Unexpected adapter tensors')
        for k in ('a','b'):
            tensor=torch.from_numpy(z[k].copy())
            if tensor.shape!=getattr(model,k).shape or not torch.isfinite(tensor).all():raise ValueError('Bad adapter shape/data')
            with torch.no_grad():getattr(model,k).copy_(tensor)
    return model

def fit_arm(train_cache,train_rows,dev_cache,dev_rows,arm,seed,out):
    torch.manual_seed(seed)
    adapter=TerminalAdapter(train_cache['x'].shape[-1],train_cache['r'].shape[-1])
    opt=torch.optim.AdamW(adapter.parameters(),lr=LR,weight_decay=0)
    labels=torch.tensor([r['target'] for r in train_rows])
    order=torch.tensor([r['semantic_order'] for r in train_rows])
    families=index_families(train_rows)
    spec=ARMS[arm];start=time.perf_counter();history=[]
    # Zero adapter is a checkpoint candidate, preserving a genuine unchanged fallback.
    with torch.no_grad():
        initial_train=adapter(train_cache);initial_dev=adapter(dev_cache)
    if float((initial_train-train_cache['z0']).abs().max())>2e-5:raise RuntimeError('Cache baseline mismatch')
    best=metric(initial_dev,dev_rows)['nll'];best_step=0
    checkpoint=out/'selected.npz';save_adapter(checkpoint,adapter)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        z=adapter(train_cache)
        loss,parts=semantic_losses(z,train_cache['z0'],order,labels,families,spec)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
        if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradients')
        opt.step()
        # Persist EVERY update, unlike the old timeout-lost trace.
        line={'step':step,'loss':float(loss.detach()),'gradient_norm':float(gn),
              **{k:float(v.detach()) for k,v in parts.items()}}
        if step%INTERVAL==0:
            with torch.no_grad():
                tm=metric(adapter(train_cache),train_rows);dm=metric(adapter(dev_cache),dev_rows)
            line.update(train=tm,development=dm)
            if dm['nll']<best-1e-8:
                best=dm['nll'];best_step=step;save_adapter(checkpoint,adapter)
            save_adapter(out/'last.npz',adapter)
            save_json(out/'progress.json',{'step':step,'selected_step':best_step,'selected_development_nll':best})
        history.append(line)
        with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(line,sort_keys=True,allow_nan=False)+'\n')
    chosen=load_adapter(checkpoint,train_cache['x'].shape[-1],train_cache['r'].shape[-1])
    with torch.no_grad():tz=chosen(train_cache);dz=chosen(dev_cache)
    result={'complete':True,'arm':arm,'seed':seed,'steps':STEPS,'selected_step':best_step,
        'train':metric(tz,train_rows),'development':metric(dz,dev_rows),
        'selected_sha256':filehash(checkpoint),'optimizer_seconds':time.perf_counter()-start,
        'base_parameters_updated':False,'direct_answer_presentations':STEPS*len(train_rows),
        'cache_forward_population_per_step':len(train_rows),'additional_model_calls_for_structure':0,
        'production_promoted':False}
    save_json(out/'result.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--cache',default='cache')
    ap.add_argument('--out',default='runs/training');args=ap.parse_args()
    torch.set_num_threads(4)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    # This entrypoint has no path to any transfer file.
    train=load_rows(args.data,'train');dev=load_rows(args.data,'development')
    if {r['group'] for r in train}&{r['group'] for r in dev}:raise ValueError('Source-family leakage')
    learn=train+dev
    cache,metadata=merge_caches(sorted((Path(args.cache)/'learn').glob('*.npz')),learn)
    if any(m.get('phase')!='learn' for m in metadata):raise ValueError('Transfer cache supplied to training')
    tc=slice_cache(cache,list(range(len(train))));dc=slice_cache(cache,list(range(len(train),len(learn))))
    protocol={'data_hash':DATA_HASH,'model_revision':REVISION,'seeds':SEEDS,'arms':list(ARMS),
        'steps':STEPS,'learning_rate':LR,'checkpoint_interval':INTERVAL,'selection_metric':'development source-balanced NLL',
        'rank':8,'scale':2.0,'train_groups':48,'development_groups':12,'no_test_file_access':True,
        'same_direct_examples_and_updates':True,'loss_specs':{k:vars(v) for k,v in ARMS.items()},
        'cache_receipt_hashes':[digest(m) for m in metadata]}
    save_json(root/'protocol.json',protocol)
    results=[]
    with torch.no_grad():base_dev=metric(dc['z0'],dev);base_train=metric(tc['z0'],train)
    for seed in SEEDS:
        for arm in ARMS:
            out=root/f'{arm}-{seed}';out.mkdir()
            results.append(fit_arm(tc,train,dc,dev,arm,seed,out))
    # Choose a method by the mean across the SAME three seeds, not the best seed.
    means={arm:sum(r['development']['nll'] for r in results if r['arm']==arm)/len(SEEDS) for arm in ARMS}
    winner=min(['baseline']+list(ARMS),key=lambda k:base_dev['nll'] if k=='baseline' else means[k])
    selection={'locked_before_transfer':True,'data_hash':DATA_HASH,'model_revision':REVISION,
        'winner':winner,'baseline_train':base_train,'baseline_development':base_dev,
        'mean_development_nll':means,'seeds':SEEDS,'results':results,
        'production_promoted':False,'selection_is_not_certification':True}
    save_json(root/'selection.json',selection)
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
