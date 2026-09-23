"""Secondary simple control fixed before Bridge output inspection."""
import argparse,hashlib,json,os
from pathlib import Path
import bridge_v15 as b

def run(root,out,shard):
    from reconstruct_v1 import Runtime
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=False)
    if shard not in range(8):raise ValueError('Invalid shard')
    m=json.loads((root/'bridge-prepared/manifest.json').read_text())
    if b.sha(root/'bridge_v15.py')!=m['source_sha256']:raise ValueError('Bridge changed')
    for n,h in m['parent_hashes'].items():
        if b.sha(root/n)!=h:raise ValueError('Parent changed')
    inputs={x['input']['id']:x for group in json.loads((root/'bridge-prepared/jobs.json').read_text()) for x in group}
    records=json.loads((root/'all_records.json').read_text());records=sorted(records,key=lambda x:x['id'])
    if len(records)!=48:raise ValueError('Incomplete main population')
    rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
    anchor=json.loads((root/'bridge-prepared/anchors.json').read_text())[shard]
    native,_=rt.score(anchor['input']);error=max(abs(a-c) for a,c in zip(native['logits'],anchor['native']['logits']))
    if error>1e-4 or native['prompt_hash']!=anchor['native']['prompt_hash']:raise ValueError('Anchor')
    b.write(out/'preflight.json',{'runtime':rt.receipt,'fixture':fixture,'anchor_error':error,'source_sha256':b.sha(__file__)})
    rows=[];path=out/'records.jsonl';path.write_text('')
    for i,r in enumerate(records):
        if i%8!=shard:continue
        job=inputs[r['id']];row=job['input']
        if r['input_sha256']!=b.digest(row):raise ValueError('Input mismatch')
        notes={'source_only':'','model_draft':r['draft']['text'],'selected_evidence':job['evidence'],'oracle':job['oracle']}
        for support,note in notes.items():
            if hashlib.sha256(note.encode()).hexdigest()!=r['views'][support]['note_sha256']:raise ValueError('Different note')
            obs=b.parent.code_readout(rt,row,note,'semantic_codes')
            result={'id':r['id'],'support':support,'input_sha256':r['input_sha256'],'observation':obs,'shard':shard}
            with path.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(result)
        print(json.dumps({'shard':shard,'records':len(rows)}),flush=True)
    b.write(out/'complete.json',{'count':len(rows),'source_sha256':b.sha(__file__),'records_sha256':b.sha(path)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',default='semantic');p.add_argument('--shard',type=int,default=0);a=p.parse_args();run(a.root,a.out,a.shard)
