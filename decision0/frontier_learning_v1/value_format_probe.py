"""Unchanged real Qwen3.5-4B: answer-blind exact-value generation diagnostic.
No optimizer, no supplied intermediate result, no benchmark, no test selection.
Six simple arithmetic cases, two fixed output formats; tiny correlated probe.
"""
from __future__ import annotations
import argparse,json,sys,time,traceback,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent

def main(out):
    import torch
    p=Path(out);p.mkdir(parents=True,exist_ok=False)
    protocol={'id':'decision0-value-format-probe-v1','model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','cases':6,'formats':['scalar','compact_states'],'max_new_tokens':48,'do_sample':False,'use_cache':True,'optimizer_updates':0,'teacher_forcing':False,'benchmark_calls':0,'scope':'small paired interface diagnostic, not generalization or leaderboard evidence'}
    parent.save(p/'protocol.json',protocol);rt=parent.Runtime();parent.save(p/'runtime.json',rt.meta)
    cases=[]
    for split,(n,price,fee,credit) in [('seen_values',(4,121,151,27)),('new_values',(7,143,127,35))]:
        for delta in (-1,0,1):
            f=fee+delta;expected={'product':n*price,'subtotal':n*price+f,'total':n*price+f-credit}
            cases.append({'id':split+str(delta),'split':split,'input':f'Quantity is {n}. Unit price is {price} cents. The handling fee is {f} cents. The credit is {credit} cents. Product is quantity times unit price. Subtotal is product plus handling fee. Total is subtotal minus credit. Use exact integer cents.', 'expected':expected})
    parent.save(p/'cases.json',cases);jobs=[(i,fmt) for i in range(len(cases)) for fmt in protocol['formats']];random.Random(97103).shuffle(jobs)
    records=[]
    with (p/'records.jsonl').open('w') as stream:
        for index,fmt in jobs:
            case=cases[index]
            instruction='Return exactly one JSON object with only the integer field total.' if fmt=='scalar' else 'Return exactly one JSON object with the integer fields product, subtotal, and total, in that order.'
            conv=[{'role':'system','content':instruction+' Do not include commentary, code fences, or explanations.'},{'role':'user','content':case['input']}]
            prompt=rt.tokenizer.apply_chat_template(conv,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)>512:raise RuntimeError('Unexpected input length')
            x=torch.tensor([ids]);tick=time.perf_counter()
            with torch.inference_mode():generated=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=48,use_cache=True)
            suffix=generated[0,len(ids):].tolist();text=rt.tokenizer.decode(suffix,skip_special_tokens=True).strip();seconds=time.perf_counter()-tick
            expected=case['expected'] if fmt=='compact_states' else {'total':case['expected']['total']}
            parsed=None;parse_error=None
            try:
                parsed=json.loads(text)
                valid=isinstance(parsed,dict) and set(parsed)==set(expected) and all(type(v) is int for v in parsed.values())
                correct=bool(valid and parsed==expected)
            except Exception as exc:parse_error=str(exc);valid=False;correct=False
            rec={'id':case['id'],'split':case['split'],'format':fmt,'prompt_sha256':__import__('hashlib').sha256(prompt.encode()).hexdigest(),'prompt_tokens':len(ids),'generated_tokens':len(suffix),'reached_token_limit':len(suffix)==48,'raw_output':text,'parsed':parsed,'expected':expected,'schema_valid':valid,'correct':correct,'parse_error':parse_error,'seconds':seconds,'intermediate_values_supplied_in_prompt':False}
            stream.write(json.dumps(rec,allow_nan=False)+'\n');stream.flush();records.append(rec);parent.emit(phase='value_probe',done=len(records),total=len(jobs),**rec)
    parent.save(p/'receipt.json',{'records':len(records),'record_sha256':parent.filehash(p/'records.jsonl'),'summary':{fmt:{'n':sum(r['format']==fmt for r in records),'exact':sum(r['correct'] for r in records if r['format']==fmt),'token_limits':sum(r['reached_token_limit'] for r in records if r['format']==fmt)} for fmt in protocol['formats']},'training_updates':0,'benchmark_calls':0,'official_score':None})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    try:main(a.out)
    except Exception as exc:
        p=Path(a.out);p.mkdir(parents=True,exist_ok=True);parent.save(p/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc(),'complete_result':False});raise
