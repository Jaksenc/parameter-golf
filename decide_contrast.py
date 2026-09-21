"""Actual 320+160-token policy, without running the longer research control.
It returns a typed label, not a calibrated probability or a correctness proof.
"""
from __future__ import annotations
import argparse,contextlib,json,sys,time
from pathlib import Path
import contrast_v7 as c


def decide(runtime,payload):
    row=c.canonical(payload);start=time.perf_counter()
    base=c.generate(runtime,c.e4.messages(row,'reason'),c.BASE_CAP)
    base_label=c.h5.parse_final(base['text'],row['labels'],cut=base['hit_cap'])
    base_readout=None
    if base_label is None:
        base_readout=c.h5.readout(runtime,row,base['text']);base_label=base_readout['label']
    check=c.generate(runtime,c.checker_messages(row),c.CHECK_CAP)
    check_label=c.h5.parse_final(check['text'],row['labels'],cut=check['hit_cap'])
    check_readout=None
    if check_label is None:
        check_readout=c.h5.readout(runtime,row,check['text']);check_label=check_readout['label']
    judgments={}
    if base_label!=check_label:
        proposals=[{'label':base_label,'draft':base['text']},{'label':check_label,'draft':check['text']}]
        for reversed_ in (False,True):judgments[str(int(reversed_))]=c.judge(runtime,row,proposals,reversed_)
    js=[judgments[str(i)]['label'] for i in range(2)] if judgments else []
    answer=c.verdict(base_label,check_label,js)
    assert answer in row['labels']
    return {'id':row['id'],'answer':answer,'primary':'contrast_consensus','base_answer':base_label,'challenger_answer':check_label,
            'changed':answer!=base_label,'disagreed':base_label!=check_label,'base_tokens':base['output_tokens'],'checker_tokens':check['output_tokens'],
            'seconds':time.perf_counter()-start,'calibrated_confidence':None,
            'trace_sha256':c.h5.digest({'base':base['text'],'checker':check['text']}),
            'judgments':judgments,'readouts':{'base':base_readout,'challenger':check_readout},
            'limits':'Same-model correlated verification; no semantic correctness guarantee.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--runtime-root',default='data');a=p.parse_args()
    payload=json.loads(Path(a.input).read_text());c.canonical(payload)
    root=Path(a.runtime_root).resolve();sys.path.insert(0,str(root))
    from reconstruct_v1 import Runtime
    with contextlib.redirect_stdout(sys.stderr):
        rt=Runtime(root/'reconstruction-inputs');result=decide(rt,payload)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
