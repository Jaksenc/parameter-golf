"""Atomic, hash-checked CPU training checkpoints for single-process experiments.

This is a new recovery utility, not recovered optimizer state from v16. Capture
immediately after an optimizer step, with explicit schedule/data/source bindings.
Not a distributed checkpoint system, authenticity mechanism, or GPU reproducibility
claim. Concurrent writers to the same directory are unsupported.
"""
from __future__ import annotations
import copy, hashlib, io, json, os, platform, random, re, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch

FORMAT='full-cpu-checkpoint-v1'

def canonical(x: Any) -> str:
    return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)

def cpu_only(model):
    if any(x.device.type!='cpu' for x in list(model.parameters())+list(model.buffers())):
        raise ValueError('Only a CPU model is supported by this utility')

def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        if hasattr(os,'O_DIRECTORY'):
            handle=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(handle)
            finally:os.close(handle)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def random_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'torch':torch.get_rng_state(),
            'numpy':{'algorithm':n[0],'keys':torch.tensor(n[1].astype(np.int64)),
                     'position':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])}}

def restore_random(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch'])
    n=state['numpy'];np.random.set_state((n['algorithm'],n['keys'].numpy().astype(np.uint32),
                                       n['position'],n['has_gauss'],n['cached_gaussian']))

def environment():
    return {'torch':str(torch.__version__),'numpy':np.__version__,
            'machine':platform.machine(),'threads':torch.get_num_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}

def save(directory, model, optimizer, *, completed_steps: int, bindings: dict,
         schedule_state: dict | None = None) -> dict:
    cpu_only(model)
    if type(completed_steps) is not int or completed_steps<0:raise ValueError('Invalid completed step')
    if not isinstance(bindings,dict) or not bindings:raise ValueError('Source/data bindings required')
    b=canonical(bindings);schedule=canonical(schedule_state or {})
    payload={'format':FORMAT,'completed_steps':completed_steps,'bindings_json':b,
             'schedule_json':schedule,'environment':environment(),
             'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
             'module_training':{n:m.training for n,m in model.named_modules()},'rng':random_state()}
    buf=io.BytesIO();torch.save(payload,buf);raw=buf.getvalue();h=hashlib.sha256(raw).hexdigest()
    directory=Path(directory);name=f'step-{completed_steps:08d}-{h}.pt'
    atomic(directory/name,raw)
    receipt={'format':FORMAT,'file':name,'sha256':h,'bytes':len(raw),'completed_steps':completed_steps}
    atomic(directory/'latest.json',canonical(receipt).encode())
    return receipt

def load(directory,model,optimizer,*,bindings:dict) -> dict:
    cpu_only(model);directory=Path(directory)
    receipt=json.loads((directory/'latest.json').read_text())
    if receipt.get('format')!=FORMAT:raise ValueError('Unsupported checkpoint')
    name=receipt.get('file','')
    if not re.fullmatch(r'step-\d{8}-[0-9a-f]{64}\.pt',name):raise ValueError('Invalid path')
    path=directory/name
    if path.stat().st_size!=receipt.get('bytes') or path.stat().st_size>2**31:raise ValueError('Checkpoint size mismatch')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt['sha256']:raise ValueError('Checkpoint checksum mismatch')
    obj=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if obj['format']!=FORMAT or obj['bindings_json']!=canonical(bindings):raise ValueError('Source/data bindings differ')
    if obj['completed_steps']!=receipt['completed_steps']:raise ValueError('Progress mismatch')
    if obj['environment']!=environment():raise ValueError('Runtime changed; exact-resume claim not supported')
    current=model.state_dict()
    if current.keys()!=obj['model'].keys():raise ValueError('Model keys changed')
    for k,v in current.items():
        other=obj['model'][k]
        if v.shape!=other.shape or v.dtype!=other.dtype:raise ValueError('Model layout changed')
        if not torch.isfinite(other).all():raise ValueError('Nonfinite model tensor')
    existing=optimizer.state_dict()
    if len(existing['param_groups'])!=len(obj['optimizer']['param_groups']):raise ValueError('Optimizer group count changed')
    for a,b in zip(existing['param_groups'],obj['optimizer']['param_groups']):
        if len(a['params'])!=len(b['params']):raise ValueError('Optimizer parameter count changed')
    backup=(copy.deepcopy(current),copy.deepcopy(existing),random_state(),
            {n:m.training for n,m in model.named_modules()})
    try:
        model.load_state_dict(obj['model'],strict=True);optimizer.load_state_dict(obj['optimizer'])
        for name,module in model.named_modules():module.training=obj['module_training'][name]
        restore_random(obj['rng'])
    except Exception:
        model.load_state_dict(backup[0]);optimizer.load_state_dict(backup[1]);restore_random(backup[2])
        for name,module in model.named_modules():module.training=backup[3][name]
        raise
    return {'completed_steps':obj['completed_steps'],'schedule_state':json.loads(obj['schedule_json']),
            'checkpoint_sha256':receipt['sha256']}
