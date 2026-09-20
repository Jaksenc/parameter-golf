from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from .semantics import digest

SPLIT_SHA={'development': '53452b06416ffc9bc4421fbc747707373e97a787f22d3aad37432d18175948af', 'train': 'ee315dd56d4f050280a5c1983abe032b7adbaa6554325bbfc822d9004458113f', 'transfer_composition': 'ebac59063630bde4a70ff63e855ad0f36146c317050e14be6a89401d9bc04f0e', 'transfer_domain': 'f735f5a531b411d335c21b33c5595716be43e82ccc5dda7bcf07e1b637120e42', 'transfer_wording': 'e647d64f3d13ef54be7584ca7f0c30787c699e80d48ab4cb1211560d041c7412'}
MODEL='Qwen/Qwen3.5-4B'
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DATA_HASH='d1bf6cb3e034923e598c79ff857dd7029387b5aa9c03048ed37f2392854086f4'
MODEL_SHA={
 'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
 'model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,p)

def load_rows(root,split):
    p=Path(root)/(split+'.jsonl')
    if split not in SPLIT_SHA or filehash(p)!=SPLIT_SHA[split]:raise ValueError('Frozen split file changed')
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    if any(r['split']!=split for r in rows):raise ValueError('Incorrect split file')
    return rows

def write_cache(path,arrays,metadata):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() or p.with_suffix('.json').exists():raise FileExistsError(p)
    with p.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,**{k:(v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.array(v)) for k,v in arrays.items()})
    os.replace(p.with_suffix('.tmp'),p)
    save_json(p.with_suffix('.json'),{**metadata,'cache_sha256':filehash(p)})

def read_cache(path):
    p=Path(path);m=json.loads(p.with_suffix('.json').read_text())
    if m.get('kind')!='qwen3.5-4b-terminal-v4' or m.get('model_revision')!=REVISION:
        raise ValueError('Not an actual pinned-model v4 cache')
    if m.get('model_hashes')!=MODEL_SHA:raise ValueError('Base-weight provenance mismatch')
    if m.get('data_hash')!=DATA_HASH or m.get('cache_sha256')!=filehash(p):raise ValueError('Cache identity mismatch')
    with np.load(p,allow_pickle=False) as z:
        arrays={k:torch.from_numpy(z[k].copy()) for k in z.files}
    for k in ('x','r','y'):arrays[k]=arrays[k].to(torch.bfloat16)
    arrays['norm_epsilon']=float(arrays['norm_epsilon'])
    if not all(torch.isfinite(v).all() for v in arrays.values() if torch.is_tensor(v)):
        raise ValueError('Cache contains nonfinite tensors')
    return arrays,m

def merge_caches(paths,expected_rows):
    if not paths:raise ValueError('No cache partitions found')
    payloads=[read_cache(p) for p in paths]
    seen={};common=None
    rowkeys=('x','r','y','z0')
    for a,m in payloads:
        c={k:a[k] for k in ('answer_weight','norm_weight')}
        identity=(m['model_revision'],m['module'],m['norm_source_hash'],m['prompt_source_hash'],m['model_hashes'])
        if common is None:common=(c,identity,a['norm_epsilon'])
        elif identity!=common[1] or a['norm_epsilon']!=common[2] or any(not torch.equal(c[k],common[0][k]) for k in c):
            raise ValueError('Mismatched model / norm / prompt across cache shards')
        if len(m['rows'])!=len(a['x']):raise ValueError('Cache rows/tensors disagree')
        for i,r in enumerate(m['rows']):
            if r['id'] in seen:raise ValueError('Duplicate cache row')
            seen[r['id']]=(a,i,r)
    if set(seen)!={r['id'] for r in expected_rows}:raise ValueError('Incomplete or extra cache population')
    out={k:torch.stack([seen[r['id']][0][k][seen[r['id']][1]] for r in expected_rows]) for k in rowkeys}
    for r in expected_rows:
        if seen[r['id']][2]['input_hash']!=r['input_hash']:raise ValueError('Model-input hash mismatch')
    out.update(common[0]);out['norm_epsilon']=common[2]
    if 'answer_bias' in payloads[0][0]:
        bias=payloads[0][0]['answer_bias']
        if any(not torch.equal(a['answer_bias'],bias) for a,_ in payloads):raise ValueError('Bias mismatch')
        out['answer_bias']=bias
    return out,[m for _,m in payloads]
