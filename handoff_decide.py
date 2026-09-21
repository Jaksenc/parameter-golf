"""Handoff v5 reference 64-token CLI. Cold model loading is not an inference metric."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import handoff_v5 as h

def validate_request(value):
    if not isinstance(value,dict):raise ValueError('Request must be an object')
    required={'state','question','labels'}
    if not required<=set(value) or set(value)-required-{'id'}:raise ValueError('Unexpected request fields')
    labels=value['labels'];q=value['question']
    if not isinstance(labels,list) or not 2<=len(labels)<=16 or not all(isinstance(s,str) and s for s in labels):raise ValueError('Require 2-16 nonempty labels')
    if len(set(labels))!=len(labels):raise ValueError('Duplicate labels')
    if not isinstance(q,dict) or q.get('type') not in ('noul','choice','score') or not isinstance(q.get('instructions'),str):raise ValueError('Invalid typed question')
    if len(json.dumps(value,allow_nan=False))>64000:raise ValueError('Request size limit')
    return {'id':str(value.get('id','request')),**{k:value[k] for k in required}}

def bounded_generate(runtime,row,budget=64):
    import evidence_v4 as e4
    if type(budget) is not int or budget!=64:raise ValueError('Frozen budget=64')
    torch=runtime.torch
    prompt=runtime.tokenizer.apply_chat_template(e4.messages(row,'reason'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=runtime.tokenizer.encode(prompt,add_special_tokens=False)
    if len(ids)>10000:raise ValueError('Input too long')
    start=time.perf_counter();runtime.model.set_output_embeddings(runtime.original_head)
    try:
        with torch.inference_mode():
            out=runtime.model.generate(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=runtime.tokenizer.eos_token_id)
    finally:runtime.model.set_output_embeddings(runtime.head)
    tokens=out[0,len(ids):].tolist();eos=runtime.model.generation_config.eos_token_id
    stop=set(eos if isinstance(eos,list) else [eos]);ended=bool(tokens) and tokens[-1] in stop
    return {'text':runtime.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'output_tokens':len(tokens),'input_tokens':len(ids),'seconds':time.perf_counter()-start,'truncated':len(tokens)==64 and not ended}

class Engine:
    def __init__(self,runtime,generator=None,reader=None):self.runtime=runtime;self.generator=generator or bounded_generate;self.reader=reader or h.readout
    def decide(self,request):
        row=validate_request(request);start=time.perf_counter();trace=self.generator(self.runtime,row)
        if trace['output_tokens']>64:raise ValueError('Token budget exceeded')
        final=h.parse_final(trace['text'],row['labels'],cut=trace['truncated']);result=None
        if final is None:result=self.reader(self.runtime,row,trace['text']);final=result['label']
        if final not in row['labels']:raise ValueError('Invalid readout label')
        return {'answer':final,'readout_invoked':result is not None,'reasoning_tokens':trace['output_tokens'],'probabilities':None if result is None else dict(zip(row['labels'],result['probabilities_uncalibrated'])),'calibrated':False,'reasoning_seconds':trace['seconds'],'readout_seconds':None if result is None else result['seconds'],'total_seconds':time.perf_counter()-start,'policy':'handoff64','source_of_answer':'completed_generation' if result is None else 'restricted_readout'}

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','decide']);p.add_argument('--request');p.add_argument('--out',default='handoff-smoke');a=p.parse_args()
    import reconstruct_v1 as v1
    rt=v1.Runtime('reconstruction-inputs')
    if a.mode=='smoke':
        wanted={'handoff-new-deadline-00-0','handoff-new-exception-00-0','handoff-new-receipt-00-0'};rows=[r for r in h.fresh_tasks() if r['id'] in wanted];results=[]
        for row in rows:
            capture={}
            def generate(runtime,x):
                capture['trace']=bounded_generate(runtime,x);return capture['trace']
            answer=Engine(rt,generator=generate).decide(h.input_only(row));results.append({'id':row['id'],'input_sha256':h.digest(h.input_only(row)),'trace':capture['trace'],'response':answer})
        h.write(Path(a.out)/'smoke.json',{'runtime':rt.receipt,'rows':results,'selected_before_outcomes':True,'selection':'fixed first pair of deadline, exception, receipt; no quality selection','source_sha256':h.filehash(__file__)})
    else:
        if not a.request:raise ValueError('Missing --request')
        print(json.dumps(Engine(rt).decide(json.loads(Path(a.request).read_text())),indent=2))
if __name__=='__main__':main()
