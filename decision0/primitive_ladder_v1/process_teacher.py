"""Post-hoc bounded process-teacher diagnostic, no training or benchmark selection."""
from __future__ import annotations
import argparse,json,re,sys,time,traceback
from decimal import Decimal
from pathlib import Path
from primitive_ladder import corpus,digest,visible

PROTOCOL={'id':'decision0-process-teacher-v1','source_selection':'two lexicographically first source hashes per family; all three endpoints',
 'sources':4,'calls':12,'max_new_tokens':192,'generation':'one greedy nonthinking sample with compact explicit calculation',
 'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','weight_updates':0,
 'scope':'post-hoc same-diagnostic-corpus teacher feasibility; not a new holdout, model gain or speed claim.'}
SYSTEM=('Compute the numerical quantity required by the evidence and criterion. '
 'Write a compact calculation showing the necessary intermediate arithmetic or UTC conversion, then finish with '
 'one JSON object with exactly keys value and unit. Unit must be USD or minutes. '
 'The JSON value must be the computed order total or elapsed minutes, NOT the boundary or deadline. '
 'Do not choose an outcome. Do not discuss alternatives. Keep the calculation short.')

def selected():
 rows=[r for r in corpus() if r['stage']=='original_end_to_end'];sources=[]
 for family in ('amount','elapsed'):sources+=sorted({r['source'] for r in rows if r['family']==family})[:2]
 return [r for r in rows if r['source'] in sources]

def parse(text):
 # Accept exactly one valid requested object, not choose one using the answer key.
 candidates=[]
 for m in re.finditer(r'\{[^{}]*\}',text):
  try:
   obj=json.loads(m.group())
   if isinstance(obj,dict) and set(obj)=={'value','unit'} and obj['unit'] in ('USD','minutes') and not isinstance(obj['value'],bool):
    v=Decimal(str(obj['value']))
    if v.is_finite():candidates.append({'value':str(v),'unit':obj['unit']})
  except (ValueError,TypeError,ArithmeticError):pass
 return candidates[0] if len(candidates)==1 else None

def run(args):
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
 from flow_study import Runtime,save,filehash
 rows=selected();assert len(rows)==12;sources=sorted({r['source'] for r in rows});subset=[r for r in rows if r['source']==sources[args.shard]]
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',PROTOCOL);save(out/'selected_inputs.json',rows)
 rt=Runtime();save(out/'runtime.json',rt.meta);records=[]
 with (out/'records.jsonl').open('w') as f:
  for r in subset:
   msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'evidence':r['state'],'criterion':r['question']},ensure_ascii=False)}]
   text=rt.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=rt.tokenizer.encode(text,add_special_tokens=False)
   assert len(ids)<8192;x=rt.torch.tensor([ids]);start=time.perf_counter()
   with rt.torch.inference_mode():z=rt.model.generate(input_ids=x,attention_mask=rt.torch.ones_like(x),max_new_tokens=192,do_sample=False,use_cache=True)
   token_ids=z[0,len(ids):].tolist();answer=rt.tokenizer.decode(token_ids,skip_special_tokens=True).strip();value=parse(answer)
   ref=Decimal(r['oracle']['result_cents'])/100 if r['family']=='amount' else Decimal(r['oracle']['elapsed_minutes']);unit='USD' if r['family']=='amount' else 'minutes'
   rec={'id':r['id'],'source':r['source'],'family':r['family'],'endpoint':r['endpoint'],'text':answer,'parsed':value,'exact_correct':value is not None and value['unit']==unit and Decimal(value['value'])==ref,
    'reference_value':str(ref),'reference_unit':unit,'prompt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),'input_tokens':len(ids),'generated_token_ids':token_ids,'generated_tokens':len(token_ids),'at_token_limit':len(token_ids)==192,'seconds':time.perf_counter()-start}
   f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();records.append(rec)
   print(json.dumps({'phase':'process_teacher','done':len(records),'total':len(subset),'shard':args.shard}),flush=True)
 save(out/'receipt.json',{'records':len(records),'sha256':filehash(out/'records.jsonl'),'protocol_sha256':digest(PROTOCOL),'selection_sha256':digest(rows),'new_weights':False})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,choices=range(4),required=True);a.add_argument('--out',required=True);args=a.parse_args()
 try:run(args)
 except Exception as e:
  p=Path(args.out);p.mkdir(parents=True,exist_ok=True);(p/'FAILED.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()}));raise
