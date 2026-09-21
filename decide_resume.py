"""Validated JSON CLI for the frozen completion-gated policy.
First use downloads pinned Qwen3.5-4B weights. CPU runtime is research-grade.
"""
from __future__ import annotations
import argparse, contextlib, json, sys
from pathlib import Path
import resume_v6 as r


def validate(payload):
    if not isinstance(payload,dict):raise ValueError('Input must be a JSON object')
    allowed={'id','state','question','labels'}
    if set(payload)-allowed:raise ValueError('Unknown fields, including expected/oracle, are forbidden')
    if not {'state','question','labels'}<=set(payload):raise ValueError('Missing state, question or labels')
    labels=payload['labels'];q=payload['question']
    if not isinstance(labels,list) or not 2<=len(labels)<=16 or any(not isinstance(x,str) or not x or len(x)>200 or '\n' in x or '\r' in x for x in labels):raise ValueError('Expected 2-16 nonempty single-line string labels')
    if len(set(labels))!=len(labels):raise ValueError('Labels must be unique')
    if not isinstance(q,dict) or q.get('type') not in ('choice','noul','score') or not isinstance(q.get('instructions'),str):raise ValueError('Invalid question type or instructions')
    if not isinstance(q.get('criteria'),(dict,list)):raise ValueError('Question requires criteria')
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False)
    if len(encoded)>60000:raise ValueError('Input exceeds bounded research interface')
    return {'id':str(payload.get('id','request')), 'state':payload['state'],'question':q,'labels':labels}


def decide(runtime,payload):
    row=validate(payload)
    base=r.h5.generate_fresh(runtime,row)
    final=r.h5.parse_final(base['text'],row['labels'],cut=base.get('hit_cap',False))
    continuation=None;readout=None
    if final is None:
        continuation=r.continue_trace(runtime,row,base['text'])
        final=continuation['final']
        if final is None:
            readout=r.h5.readout(runtime,row,continuation['text'])
            final=readout['label']
    assert final in row['labels']
    return {'id':row['id'],'answer':final,'method':'resume_complete','extended':continuation is not None,
            'base_tokens':base['output_tokens'],'added_tokens':continuation['added_tokens'] if continuation else 0,
            'probabilities_uncalibrated':readout['probabilities_uncalibrated'] if readout else None,
            'semantic_correctness_verified':False,'model_weights_trained_here':False,
            'trace_sha256':r.h5.digest({'base':base['text'],'continued':continuation['text'] if continuation else None})}


def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--runtime-root',default='data');a=p.parse_args()
    root=Path(a.runtime_root).resolve();sys.path.insert(0,str(root));payload=json.loads(Path(a.input).read_text());validate(payload)
    from reconstruct_v1 import Runtime
    with contextlib.redirect_stdout(sys.stderr):
        rt=Runtime(root/'reconstruction-inputs');result=decide(rt,payload)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
