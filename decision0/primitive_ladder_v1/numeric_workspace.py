"""Bounded model-generated numerical workspace; no solver or oracle at inference.

Controls: ordinary decision, own unverified scalar estimate, and a deliberately
swapped estimate from a sibling counterfactual. Only evaluation reads references.
"""
from __future__ import annotations
import argparse, copy, json, re, sys, time, traceback
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from primitive_ladder import corpus, digest, visible

PROTOCOL={'id':'decision0-numeric-workspace-v1','sources':16,'endpoints':48,
 'seed':87119,'max_generated_tokens':48,'generation':'greedy nonthinking; one sample',
 'final_variants':['direct','own_estimate','swapped_estimate'],
 'final_outcome_orders':['forward','reverse'],'shards':8,
 'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'new_weights':False,'no_test_time_oracle':True,'benchmark_calls':0,
 'swapped_control':'Use the next endpoint of the same counterfactual source; it may accidentally produce the same estimate.',
 'scope':'An extra-compute diagnostic, not a single-forward-pass model improvement.'}

SYSTEM=('Compute the numerical quantity needed to apply the supplied decision criterion to the evidence. '
        'Do not choose an outcome or explain your reasoning. Return only a JSON object with '
        'keys value and unit, where value is the computed scalar and unit is USD or minutes. '
        'For an order, compute its exact total after charges and credits. For a timing question, '
        'compute elapsed minutes respecting UTC offsets. Never copy a limit as the computed value.')

def generate(rt,row):
    torch=rt.torch
    payload={'evidence':row['state'],'criterion':row['question']}
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    text=rt.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=rt.tokenizer.encode(text,add_special_tokens=False)
    if len(ids)>8192:raise ValueError('Input too long; no truncation')
    x=torch.tensor([ids]);start=time.perf_counter();torch.manual_seed(PROTOCOL['seed'])
    with torch.inference_mode():
        out=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=PROTOCOL['max_generated_tokens'],do_sample=False,use_cache=True)
    generated=out[0,len(ids):].tolist();answer=rt.tokenizer.decode(generated,skip_special_tokens=True).strip()
    parsed=None
    try:
        obj=json.loads(answer)
        if isinstance(obj,dict) and set(obj)=={'value','unit'} and obj['unit'] in ('USD','minutes') and not isinstance(obj['value'],bool):
            value=Decimal(str(obj['value']))
            if value.is_finite():parsed={'value':str(value),'unit':obj['unit']}
    except (ValueError,TypeError,InvalidOperation):pass
    return {'id':row['id'],'source':row['source'],'endpoint':row['endpoint'],'text':answer,'parsed':parsed,
      'generated_token_ids':generated,'input_tokens':len(ids),'generated_tokens':len(generated),
      'at_token_limit':len(generated)==PROTOCOL['max_generated_tokens'],'seconds':time.perf_counter()-start,
      'prompt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}

def run(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
    from flow_study import Runtime,save,filehash
    from prompt_variants import encode_checked
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in corpus() if r['stage']=='original_end_to_end'];sources=sorted({r['source'] for r in rows});selected=set(sources[args.shard::8]);rows=[r for r in rows if r['source'] in selected]
    save(out/'protocol.json',PROTOCOL);save(out/'corpus_check.json',{'full_original_hash':digest([r for r in corpus() if r['stage']=='original_end_to_end']),'selected_ids':[r['id'] for r in rows]})
    rt=Runtime();save(out/'runtime.json',rt.meta)
    estimates={}
    with (out/'estimates.jsonl').open('w') as f:
      for r in rows:
        est=generate(rt,visible(r)|{k:r[k] for k in ('source','endpoint')});est['family']=r['family']
        # Grading follows generation and is never appended to a model input.
        expected_value=Decimal(r['oracle']['result_cents'])/100 if r['family']=='amount' else Decimal(r['oracle']['elapsed_minutes'])
        expected_unit='USD' if r['family']=='amount' else 'minutes'
        est['exact_correct']=est['parsed'] is not None and est['parsed']['unit']==expected_unit and Decimal(est['parsed']['value'])==expected_value
        estimates[r['id']]=est;f.write(json.dumps(est,allow_nan=False)+'\n');f.flush()
        print(json.dumps({'phase':'numeric_workspace','shard':args.shard,'generated':len(estimates),'total':len(rows)}),flush=True)
    sibling={(r['source'],r['endpoint']):r['id'] for r in rows};count=0
    with (out/'records.jsonl').open('w') as f:
      for r in rows:
        for variant in PROTOCOL['final_variants']:
          v=copy.deepcopy(r);producer=r['id'] if variant=='own_estimate' else sibling[(r['source'],(r['endpoint']+1)%3)]
          if variant!='direct':v['state']+='\nNumerical workspace from a fallible model (unverified; check against the evidence): '+estimates[producer]['text']
          for order in PROTOCOL['final_outcome_orders']:
            q=copy.deepcopy(v)
            if order=='reverse':q['options'].reverse()
            rec=rt.score(q,'baseline');rec.update({k:r[k] for k in ('source','family','endpoint','expected')});rec.update({'variant':variant,'outcome_order':order,'estimate_from':producer if variant!='direct' else None,'correct':rec['predicted']==r['expected']})
            f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
            print(json.dumps({'phase':'workspace_decide','shard':args.shard,'done':count,'total':len(rows)*6}),flush=True)
    save(out/'receipt.json',{'records':count,'estimates':len(estimates),'records_sha256':filehash(out/'records.jsonl'),'estimates_sha256':filehash(out/'estimates.jsonl'),'protocol_sha256':digest(PROTOCOL),'no_weight_updates':True})

def aggregate(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
    from flow_study import save,filehash
    rows=[r for r in corpus() if r['stage']=='original_end_to_end'];expected={r['id']:r for r in rows};records=[];estimates=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
      receipt=json.loads((f.parent/'receipt.json').read_text());e=f.parent/'estimates.jsonl'
      assert receipt['protocol_sha256']==digest(PROTOCOL) and receipt['records_sha256']==filehash(f) and receipt['estimates_sha256']==filehash(e)
      estimates.extend(json.loads(l) for l in e.read_text().splitlines())
      for line in f.read_text().splitlines():
        r=json.loads(line);key=(r['id'],r['variant'],r['outcome_order']);assert key not in seen;seen.add(key);assert r['expected']==expected[r['id']]['expected'];records.append(r)
    assert seen=={(r['id'],v,o) for r in rows for v in PROTOCOL['final_variants'] for o in PROTOCOL['final_outcome_orders']}
    assert len(estimates)==48 and {r['id'] for r in estimates}==set(expected)
    summary={}
    for family in ('amount','elapsed'):
      es=[r for r in estimates if r['family']==family];item={'estimates':{'n':len(es),'valid_json':sum(r['parsed'] is not None for r in es),'exact_correct':sum(r['exact_correct'] for r in es),'token_limit':sum(r['at_token_limit'] for r in es)}}
      for variant in PROTOCOL['final_variants']:
        item[variant]={}
        for order in PROTOCOL['final_outcome_orders']:
          rs=[r for r in records if r['family']==family and r['variant']==variant and r['outcome_order']==order]
          item[variant][order]={'n':len(rs),'correct':sum(r['correct'] for r in rs),'complete_triplets':sum(all(r['correct'] for r in rs if r['source']==s) for s in {r['source'] for r in rs})}
      summary[family]=item
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'results.json',{'summary':summary,'protocol':PROTOCOL,'decision_records':len(records),'generated_estimates':len(estimates),'new_weights':False,'official_score':None})
    for name,data in [('records',records),('estimates',estimates)]:
      (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in data))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('run');a.add_argument('--shard',type=int,choices=range(8),required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',required=True);a.add_argument('--out',required=True)
    args=p.parse_args()
    try:{'run':run,'aggregate':aggregate}[args.mode](args)
    except Exception as exc:
      out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'FAILED.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}));raise
