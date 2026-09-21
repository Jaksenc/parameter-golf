"""Run either frozen v8 strategy on a supplied task, without benchmark labels."""
from __future__ import annotations
import argparse,contextlib,json,sys
from pathlib import Path
import countercase_v8 as c


def decide(runtime,payload,arm=c.PRIMARY):
 row=c.canonical(payload)
 result=c.solve(runtime,row,arm)
 return {'id':row['id'],'answer':result['answer'],'method':arm,
         'generated_tokens':result['trace']['output_tokens'],'used_categorical_fallback':result['readout'] is not None,
         'raw_uncalibrated_probabilities':result['readout']['probabilities_uncalibrated'] if result['readout'] else None,
         'semantic_correctness_verified':False,'weights_trained_here':False,
         'record_sha256':c.prior.h5.digest({'input':row,'text':result['trace']['text'],'answer':result['answer']})}


def main():
 p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--arm',choices=c.ARMS,default=c.PRIMARY)
 p.add_argument('--data',default='data');a=p.parse_args();row=json.loads(Path(a.input).read_text());c.canonical(row)
 from reconstruct_v1 import Runtime
 with contextlib.redirect_stdout(sys.stderr):
  runtime=Runtime(Path(a.data)/'reconstruction-inputs');answer=decide(runtime,row,a.arm)
 print(json.dumps(answer,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
