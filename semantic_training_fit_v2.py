"""Post-hoc training-fit diagnostic. All 54 training cases, no parameter updates.
Separates fitting from transfer; training scores are NOT held-out capability.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,sys,zipfile
from pathlib import Path
from semantic_saved_eval_v2 import ASSETS,BLOBS,DATA_HASH,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['baseline','direct','auxiliary'],required=True);args=ap.parse_args()
    root=Path('fit-'+args.arm);root.mkdir(exist_ok=False)
    import requests
    spec=ASSETS['direct' if args.arm=='baseline' else args.arm]
    r=requests.get(f"https://api.github.com/repos/Jaksenc/parameter-golf/actions/artifacts/{spec['artifact']}/zip",headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'},timeout=90);r.raise_for_status();blob=r.content
    if hashlib.sha256(blob).hexdigest()!=spec['zip_hash']:raise RuntimeError('Archive changed')
    src=root/'source';src.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if sum(v.file_size for v in z.infolist())>10_000_000:raise ValueError('Oversized archive')
        for v in z.infolist():
            if Path(v.filename).is_absolute() or '..' in Path(v.filename).parts:raise ValueError('Unsafe path')
        z.extractall(src)
    for name,h in BLOBS.items():
        b=(src/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=h:raise RuntimeError('Source changed')
    sys.path.insert(0,str(src.resolve()))
    from semantic_train_v1 import Runtime,stats
    from semantic_data_v1 import dataset,tests
    if tests()['data_hash']!=DATA_HASH:raise RuntimeError('Data changed')
    rt=Runtime();cfg=None
    if args.arm!='baseline':
        cfg=json.loads((src/'adapter_config.json').read_text())
        if hashlib.sha256((src/'adapter.safetensors').read_bytes()).hexdigest()!=spec['adapter_hash']:raise RuntimeError('Checkpoint changed')
        from safetensors.torch import load_file
        rt.attach_lora();state=load_file(str(src/'adapter.safetensors'));expected=rt.adapter_state()
        if state.keys()!=expected.keys() or any(state[k].shape!=expected[k].shape for k in state):raise RuntimeError('Adapter shape mismatch')
        rt.load_adapter(state)
    rows=[r for r in dataset() if r['split']=='train'];records=[]
    for i,row in enumerate(rows):
        p=rt.predict(row);records.append(p)
        with (root/'records.jsonl').open('a') as f:f.write(json.dumps(p,sort_keys=True,allow_nan=False)+'\n')
        if i%18==17:print(json.dumps({'kind':'progress','arm':args.arm,'done':i+1,'population':54}),flush=True)
    out={'complete':True,'arm':args.arm,'data_hash':DATA_HASH,'checkpoint':cfg,'model':rt.receipt,'predictions':records,'metrics':stats(rows,records),'new_optimizer_steps':0,'source_artifact':spec['artifact'],'run':os.environ['GITHUB_RUN_ID'],'qualification':'Post-hoc fit diagnostic on all 54 original training examples; no tuning, model promotion or test selection.'}
    write(root/'fit.json',out);print(json.dumps({'kind':'fit_complete','arm':args.arm,'metrics':out['metrics']}),flush=True)
if __name__=='__main__':main()
