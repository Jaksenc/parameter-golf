"""Fixed-checkpoint stress test; no training, benchmark calls, or test selection.
24 source situations, two fee endpoints, four base output interfaces and two
matched trained checkpoints. Gold computations are used only after generation.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
from decimal import Decimal

PROTOCOL = {
 'id':'decision0-state-stress-v1','seed':823114,'source_run':35795105921,
 'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'sources':24,'cases':48,'shards':8,'max_new_tokens':64,'max_input_tokens':1024,
 'variants':['base_scalar','base_repeat_total','base_process','base_reverse_process','trained_repeat_total','trained_process'],
 'selection':'none: all fixed variants reported; no weights updated',
 'scope':'one arithmetic expression under four stress strata; not JevBench or unrestricted arithmetic',
 'checkpoint_hashes':{'repeat_total':'34681e493d3aec151c8ac2fae7752cf065aee9fe84eb8cb6c98d374ed00f33bf','process':'708d2a6993f031a67bfba69e2c2be67a012d7a35ed5d3523222fe9a45405732a'},
 'benchmark_calls':0,'optimizer_updates':0,'official_score':None,
}
def digest(obj):
 return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
def cases():
 out=[]
 for stratum in ('ordinary','multi_digit','signed_zero','distractors'):
  for i in range(6):
   rng=random.Random(int(digest([PROTOCOL['id'],PROTOCOL['seed'],stratum,i])[:16],16))
   if stratum=='multi_digit':q,p,f,c=rng.randint(11,49),rng.randint(1001,9000),rng.randint(99,2501),rng.randint(9,700)
   elif stratum=='signed_zero':q,p,f,c=(0 if i==0 else rng.randint(1,7)),rng.randint(0,300),rng.randint(0,20),rng.randint(100,2500)
   else:q,p,f,c=rng.randint(2,9),rng.randint(37,199),rng.randint(20,160),rng.randint(5,90)
   source=digest([stratum,i])[:16]
   other=[rng.randint(10,999) for _ in range(4)]
   for edit in (0,1):
    out.append({'id':digest([source,edit])[:24],'source':source,'stratum':stratum,'edit':edit,'quantity':q,'price':p,'fee':f+edit,'credit':c,'other':other})
 return out

def gold(row,mode):
 a=row['quantity']*row['price']; b=a+row['fee']; c=b-row['credit']
 if mode=='scalar':return {'c':c}
 if mode=='repeat_total':return {'a':c,'b':c,'c':c}
 return {'a':a,'b':b,'c':c}

def conversation(row,mode):
 if mode not in ('scalar','repeat_total','process','reverse_process'):raise ValueError(mode)
 if mode=='scalar':schema='Return only one JSON object with integer field c. c is total.'
 elif mode=='repeat_total':schema='Return only one JSON object with integer fields a, b, c in that order; a is total, b is total, and c is total.'
 elif mode=='process':schema='Return only one JSON object with integer fields a, b, c in that order; a is product, b is subtotal, and c is total.'
 else:schema='Return only one JSON object with integer fields c, b, a in that order; a is product, b is subtotal, and c is total.'
 system='Calculate using exact integer cents. Product is quantity times unit price. Subtotal is product plus handling fee. Total is subtotal minus credit. Signed totals are allowed. '+schema
 text=f"Quantity {row['quantity']}. Unit price {row['price']} cents. Handling fee {row['fee']} cents. Credit {row['credit']} cents."
 if row['stratum']=='distractors':
  a,b,c,d=row['other']
  text=f'Calculate only order KESTREL. Order MOSS has quantity {a}, unit price {b} cents, fee {c} cents, and credit {d} cents. Order KESTREL has: '+text+' A separate order LARCH has quantity 8, unit price 271 cents, fee 29 cents, credit 13 cents. Do not combine orders.'
 return [{'role':'system','content':system},{'role':'user','content':text}]

def parse(text,mode):
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate field')
   d[k]=v
  return d
 try:
  x=json.loads(text,object_pairs_hook=pairs)
  keys={'c'} if mode=='scalar' else {'a','b','c'}
  if not isinstance(x,dict) or set(x)!=keys or any(type(v) is not int for v in x.values()):raise ValueError('wrong schema')
  return x,None
 except (ValueError,TypeError) as e:return None,str(e)

def selftest():
 rows=cases();assert len(rows)==48 and len({r['source'] for r in rows})==24
 seen=set()
 old={(3,137),(6,89),(5,113),(9,127),(4,121),(7,143)}
 for r in rows:
  assert (r['quantity'],r['price']) not in old
  for mode in ('scalar','repeat_total','process','reverse_process'):
   msg=conversation(r,mode);assert r['id'] not in json.dumps(msg)
   g=gold(r,mode);exact=int(Decimal(r['quantity'])*Decimal(r['price'])+Decimal(r['fee'])-Decimal(r['credit']))
   assert g['c']==exact
   assert parse(json.dumps(g),mode)[0]==g
   k=digest(msg);assert k not in seen;seen.add(k)
 for j in range(0,len(rows),2):
  a,b=rows[j:j+2]; assert a['source']==b['source']
  assert [k for k in a if a[k]!=b[k]]==['id','edit','fee']
  assert gold(b,'process')['c']-gold(a,'process')['c']==1
  assert gold(b,'process')['a']==gold(a,'process')['a']
 assert parse('{"c":1,"c":2}','scalar')[0] is None
 assert parse('{"c":true}','scalar')[0] is None
 assert parse('{"c":1.0}','scalar')[0] is None
 return {'cases':48,'sources':24,'data_hash':digest(rows),'protocol_hash':digest(PROTOCOL),'tested_conversations':len(seen)}

def run(args):
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
 import study as parent
 from safetensors.torch import load_file
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',PROTOCOL);save(out/'preflight.json',selftest());rows=cases();save(out/'cases.json',rows)
 states={}
 for arm,expected in PROTOCOL['checkpoint_hashes'].items():
  paths=list(Path(args.checkpoints).glob(f'state-learning-{arm}/adapter.safetensors'))
  if len(paths)!=1 or filehash(paths[0])!=expected:raise RuntimeError('checkpoint identity mismatch '+arm)
  states[arm]=load_file(str(paths[0]))
 rt=parent.Runtime();rt.hook.remove();ad=parent.Adapter(rt,108731);ad.enabled=False;rt.model.eval();save(out/'runtime.json',rt.meta)
 count=0
 with (out/'records.jsonl').open('w') as stream:
  for i,row in enumerate(rows):
   if i%PROTOCOL['shards']!=args.shard:continue
   variants=PROTOCOL['variants'];rot=i%len(variants);order=variants[rot:]+variants[:rot]
   for variant in order:
    mode=variant.split('_',1)[1];trained=variant.startswith('trained_');ad.enabled=trained
    if trained:ad.factors.load_state_dict(states[mode])
    msg=conversation(row,mode)
    text=rt.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=rt.tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>PROTOCOL['max_input_tokens']:raise RuntimeError('input length violation')
    x=torch.tensor([ids]);tic=time.perf_counter()
    with torch.no_grad():
     y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=PROTOCOL['max_new_tokens'],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
    tok=y[0,len(ids):].tolist();raw=rt.tokenizer.decode(tok,skip_special_tokens=True);obj,error=parse(raw,mode)
    expected=gold(row,mode)
    desired_order=['c'] if mode=='scalar' else ['c','b','a'] if mode=='reverse_process' else ['a','b','c']
    rec={'id':row['id'],'source':row['source'],'stratum':row['stratum'],'edit':row['edit'],'variant':variant,'raw_output':raw,'generated_token_ids':tok,'parsed':obj,'error':error,'expected':expected,
         'total_correct':obj is not None and obj['c']==expected['c'],'all_fields_correct':obj==expected,'requested_order_obeyed':obj is not None and list(obj)==desired_order,
         'prompt_hash':hashlib.sha256(text.encode()).hexdigest(),'input_tokens':len(ids),'output_tokens':len(tok),'token_limit':len(tok)>=PROTOCOL['max_new_tokens'],'seconds':time.perf_counter()-tic}
    stream.write(json.dumps(rec,allow_nan=False)+'\n');stream.flush();count+=1
    print(json.dumps({'phase':'stress','shard':args.shard,'done':count,'id':row['id'],'variant':variant}),flush=True)
 save(out/'receipt.json',{'records':count,'records_sha256':filehash(out/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'checkpoint_hashes':PROTOCOL['checkpoint_hashes'],'optimizer_updates':0})

def aggregate(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rows=cases();seen=set();records=[]
 for p in Path(args.root).rglob('records.jsonl'):
  r=json.loads((p.parent/'receipt.json').read_text())
  if r['protocol_hash']!=digest(PROTOCOL) or r['records_sha256']!=filehash(p):raise RuntimeError('bad shard')
  lines=[json.loads(x) for x in p.read_text().splitlines()]
  if len(lines)!=r['records']:raise RuntimeError('bad count')
  for x in lines:
   key=(x['id'],x['variant'])
   if key in seen:raise RuntimeError('duplicate record')
   seen.add(key);records.append(x)
 expected={(r['id'],v) for r in rows for v in PROTOCOL['variants']}
 if seen!=expected:raise RuntimeError('missing or extra records')
 summary={}
 for v in PROTOCOL['variants']:
  rs=[r for r in records if r['variant']==v]
  summary[v]={'n':len(rs),'exact_total':sum(x['total_correct'] for x in rs),'exact_all_fields':sum(x['all_fields_correct'] for x in rs),'schema_valid':sum(x['parsed'] is not None for x in rs),'token_limits':sum(x['token_limit'] for x in rs),'both_endpoints':sum(all(x['total_correct'] for x in rs if x['source']==s) for s in {x['source'] for x in rs}),'strata':{k:{'n':sum(x['stratum']==k for x in rs),'correct':sum(x['total_correct'] for x in rs if x['stratum']==k)} for k in sorted({x['stratum'] for x in rs})}}
 records.sort(key=lambda x:(x['id'],x['variant']))
 (out/'records.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in records));save(out/'results.json',{'summary':summary,'protocol':PROTOCOL});save(out/'cases.json',rows);save(out/'receipt.json',{'records_sha256':filehash(out/'records.jsonl'),'n':len(records),'protocol_hash':digest(PROTOCOL)})
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();s=ap.add_subparsers(dest='command',required=True)
 s.add_parser('selftest')
 p=s.add_parser('run');p.add_argument('--shard',type=int,choices=range(8),required=True);p.add_argument('--checkpoints',required=True);p.add_argument('--out',required=True)
 p=s.add_parser('aggregate');p.add_argument('--root',required=True);p.add_argument('--out',required=True)
 a=ap.parse_args()
 if a.command=='selftest':print(json.dumps(selftest(),indent=2))
 elif a.command=='run':run(a)
 else:aggregate(a)
