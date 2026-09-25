"""Acquisition-only repair after the original public BoolQ endpoint returned 403.
The source, curriculum selection, model, and optimization are otherwise unchanged.
This is a publicly distributed dataset copy, not an authenticated access workaround.
"""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
from . import train, data
REV='8ab2d3e9ff5449d3957950129cea176fe214d884'
FILES={
 'train':('train-00000-of-00001.parquet','4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623'),
 'dev':('validation-00000-of-00001.parquet','52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8')}

def install(out):
    original_download=train.download
    original_boolq=train.boolq
    records={}
    def downloaded(url):
        prefix='https://storage.googleapis.com/boolq/'
        if not url.startswith(prefix):return original_download(url)
        split=url.removeprefix(prefix).removesuffix('.jsonl')
        if split not in FILES:raise ValueError('unexpected BoolQ object')
        import pyarrow.parquet as pq
        name,expected=FILES[split]
        public=f'https://huggingface.co/datasets/google/boolq/resolve/{REV}/data/{name}'
        raw=original_download(public)
        if data.sha(raw)!=expected:raise ValueError('public BoolQ parquet digest mismatch')
        rows=pq.read_table(io.BytesIO(raw)).to_pylist()
        if any(set(r)!={'question','answer','passage'} for r in rows):raise ValueError('BoolQ schema differs')
        converted=('\n'.join(data.canonical(r) for r in rows)+'\n').encode()
        Path(out,name).write_bytes(raw)
        records[url]={'url':public,'repository':'google/boolq','revision':REV,
          'parquet_sha256':expected,'converted_jsonl_sha256':data.sha(converted),'rows':len(rows),
          'failed_original_endpoint':url,'conversion':'same question/passage/answer columns, row order retained; canonical JSONL'}
        train.save(Path(out,'boolq-acquisition.json'),records)
        return converted
    def boolq(out):
        examples,receipts=original_boolq(out)
        for receipt in receipts:
            r=records[receipt['url']]
            receipt.update(r)
            receipt['sha256_basis']='converted canonical JSONL'
        return examples,receipts
    train.download=downloaded
    train.boolq=boolq

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['grouped','shuffled'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();install(a.out)
    try:
        if a.prepare_only:
            Path(a.out).mkdir(parents=True,exist_ok=False)
            d=train.prepare(Path(a.out))
            print(json.dumps({'counts':{k:len(v) for k,v in d.items()},'sha256':train.filehash(Path(a.out,'dataset.json'))}))
        else:train.train(a)
    except BaseException as e:
        if Path(a.out).exists():
            import traceback
            train.save(Path(a.out,'FAILED.json'),{'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':main()
