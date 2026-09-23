"""Fixed 4B state-to-decision bridge. No training or benchmark answers.
All six interfaces are reported. Field order is a measured manipulation check.
Only conversation() supplies model inputs; gold() is called after generation.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, sys, time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

MODES=('decision_only','opaque_scalar','named_scalar','forward_state','repeat_scalar','decision_first')
P={'id':'decision0-state-bridge-v1','seed':22092731,'model':'Qwen/Qwen3.5-4B',
 'revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','modes':MODES,
 'sources_per_family':6,'families':['money','elapsed'],'endpoints':3,'cases':36,
 'shards':12,'max_new_tokens':112,'max_input_tokens':1536,'training_updates':0,
 'selection':'none, all fixed variants reported','benchmark_calls':0,'official_score':None,
 'scope':'synthetic arithmetic-to-decision diagnostic, not a broad model evaluation'}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))
def money(x):return f'${x/100:.2f}'
def cases():
 out=[]
 for family in P['families']:
  for j in range(P['sources_per_family']):
   rng=random.Random(int(digest([P['id'],P['seed'],family,j])[:16],16))
   key=''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(7))
   source=digest([family,j])[:16];labels=['below','equal','above'];rng.shuffle(labels)
   options=[{'code':chr(65+i),'meaning':lab} for i,lab in enumerate(labels)]
   if family=='money':
    q=rng.randint(2,13);u=rng.randint(37,1100);f=rng.randint(50,140);c=rng.randint(0,80)
    if j==4:c=q*u+f+50
    bound=q*u+f-c
    lines=[f'Order {key}: quantity {q}, unit price {money(u)}.',
           f'Order {key}: handling fee VALUE.',
           f'Order {key}: credit {money(c)}, comparison amount {money(bound)}.',
           f'Order MOSS: quantity 9, unit price $23.13, handling fee $2.40, credit $3.01; comparison amount $220.00.']
    rng.shuffle(lines)
    for edit,d in enumerate((-1,0,1)):
     evidence='\n'.join(s.replace('VALUE',money(f+d)) for s in lines)
     question=f'For order {key}, multiply quantity by unit price, add its handling fee and subtract its credit. Compare the exact total with its comparison amount. Use only this order. Signed totals are allowed. Classify as below, equal or above.'
     out.append({'id':digest([source,edit])[:24],'source':source,'family':family,'edit':edit,
                 'evidence':evidence,'question':question,'options':options,
                 'reference':{'quantity':q,'unit_cents':u,'fee_cents':f+d,'credit_cents':c,'boundary':bound}})
   else:
    z0=timezone(timedelta(minutes=rng.choice([-300,0,120,330])))
    z1=timezone(timedelta(minutes=rng.choice([-240,60,345])))
    start=datetime(2028,3+j,12,20,37,tzinfo=z0);bound=rng.choice([95,175,495,785,1505])
    lines=[f'Case {key}: started {start.isoformat()}.',f'Case {key}: submitted VALUE.',
           f'Case {key}: comparison interval {bound} elapsed minutes.',
           'Case WILLOW: started 2028-01-03T08:00:00+00:00, submitted 2028-01-04T10:00:00+00:00. This is a separate case.']
    rng.shuffle(lines)
    for edit,d in enumerate((-1,0,1)):
     end=(start+timedelta(minutes=bound+d)).astimezone(z1)
     evidence='\n'.join(s.replace('VALUE',end.isoformat()) for s in lines)
     question=f'For case {key}, calculate elapsed minutes from its start to submission. Compare with its comparison interval. Respect explicit UTC offsets; do not count business hours. Classify as below, equal or above.'
     out.append({'id':digest([source,edit])[:24],'source':source,'family':family,'edit':edit,
                 'evidence':evidence,'question':question,'options':options,
                 'reference':{'start':start.isoformat(),'end':end.isoformat(),'boundary':bound}})
 return out

def schema(family,mode):
 numeric='total_cents' if family=='money' else 'elapsed_minutes'
 base_fields=['product_cents','subtotal_cents','total_cents'] if family=='money' else ['start_utc','submission_utc','elapsed_minutes']
 if mode=='decision_only':return ['decision']
 if mode=='opaque_scalar':return ['c','decision']
 if mode=='named_scalar':return [numeric,'decision']
 if mode=='forward_state':return base_fields+['decision']
 if mode=='repeat_scalar':return ['a','b','c','decision']
 if mode=='decision_first':return ['decision']+list(reversed(base_fields))
 raise ValueError(mode)

def conversation(row,mode):
 fields=schema(row['family'],mode)
 if row['family']=='money':
  meanings='total_cents is quantity times unit price plus handling fee minus credit, in signed integer cents. product_cents is quantity times unit price in cents. subtotal_cents is product_cents plus handling fee in cents.'
  scalar='the final total in signed integer cents'
 else:
  meanings='start_utc and submission_utc are the two timestamps converted to UTC, as ISO 8601 strings ending +00:00. elapsed_minutes is the signed integer number of minutes from start to submission.'
  scalar='the elapsed interval in signed integer minutes'
 if mode=='opaque_scalar':meanings=f'c is {scalar}.'
 if mode=='repeat_scalar':meanings=f'a, b, and c must each be {scalar}; all three represent the same final value.'
 if mode=='decision_only':meanings=''
 system=('Apply the supplied calculation and comparison to the relevant subject. Return exactly one JSON object and no prose. '
         +'Write keys in exactly this order: '+', '.join(fields)+'. '+meanings+
         ' decision is the uppercase code A, B, or C whose meaning is correct. Numeric fields must be JSON integers, not strings. No other keys.')
 payload={'evidence':row['evidence'],'criterion':row['question'],'outcomes':row['options']}
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]

def gold(row):
 r=row['reference']
 if row['family']=='money':
  a=int(Decimal(r['quantity'])*Decimal(r['unit_cents']));b=a+r['fee_cents'];v=b-r['credit_cents']
  fields={'product_cents':a,'subtotal_cents':b,'total_cents':v}
 else:
  a=datetime.fromisoformat(r['start']).astimezone(timezone.utc);b=datetime.fromisoformat(r['end']).astimezone(timezone.utc)
  seconds=(b-a).total_seconds();assert seconds%60==0;v=int(seconds//60)
  fields={'start_utc':a.isoformat(),'submission_utc':b.isoformat(),'elapsed_minutes':v}
 relation='below' if v<r['boundary'] else 'above' if v>r['boundary'] else 'equal'
 fields['decision']=next(o['code'] for o in row['options'] if o['meaning']==relation)
 return v,fields

def parse(raw,row,mode):
 def pairs(xs):
  out={}
  for k,v in xs:
   if k in out:raise ValueError('Duplicate key')
   out[k]=v
  return out
 try:
  x=json.loads(raw,object_pairs_hook=pairs)
  if not isinstance(x,dict) or set(x)!=set(schema(row['family'],mode)):raise ValueError('Wrong fields')
  if x['decision'] not in ['A','B','C']:raise ValueError('Invalid decision')
  for k,v in x.items():
   if k in ['decision','start_utc','submission_utc']:
    if not isinstance(v,str):raise ValueError('Invalid text type')
   elif type(v) is not int:raise ValueError('Invalid integer type')
  return x,None
 except (ValueError,TypeError) as e:return None,str(e)

def grade(raw,row,mode):
 x,err=parse(raw,row,mode);v,g=gold(row);required=schema(row['family'],mode)
 expected={k:(v if k in ['a','b','c'] else g[k]) for k in required}
 scalar_key=next((k for k in ['total_cents','elapsed_minutes','c'] if x is not None and k in x),None)
 return {'parsed':x,'error':err,'expected':expected,'decision_correct':x is not None and x['decision']==g['decision'],
         'scalar_correct':None if mode=='decision_only' else x is not None and x[scalar_key]==v,
         'all_fields_correct':x==expected,'order_obeyed':x is not None and list(x)==required}

def selftest():
 rows=cases();assert len(rows)==P['cases'];assert len({r['source'] for r in rows})==12
 fingerprints=set()
 for row in rows:
  value,g=gold(row);assert g['decision'] in ['A','B','C']
  for mode in MODES:
   msgs=conversation(row,mode);assert row['id'] not in json.dumps(msgs);assert 'reference' not in json.loads(msgs[1]['content'])
   fp=digest(msgs);assert fp not in fingerprints;fingerprints.add(fp)
   expected={k:(value if k in ['a','b','c'] else g[k]) for k in schema(row['family'],mode)}
   test=grade(json.dumps(expected),row,mode);assert test['all_fields_correct'] and test['decision_correct'] and test['order_obeyed']
 for i in range(0,len(rows),3):
  rs=rows[i:i+3];assert len({gold(r)[1]['decision'] for r in rs})==3
  assert rs[0]['question']==rs[1]['question']==rs[2]['question'];assert rs[0]['options']==rs[1]['options']==rs[2]['options']
  assert sum(a!=b for a,b in zip(rs[0]['evidence'].splitlines(),rs[1]['evidence'].splitlines()))==1
  assert [gold(r)[0]-r['reference']['boundary'] for r in rs]==[-1,0,1]
 assert parse('{"decision":"A","decision":"B"}',rows[0],'decision_only')[0] is None
 return {'cases':len(rows),'sources':12,'model_inputs_checked':len(fingerprints),'corpus_sha256':digest(rows),'protocol_sha256':digest(P)}

def run(args):
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
 from flow_study import Runtime
 p=Path(args.out);p.mkdir(parents=True,exist_ok=False);save(p/'protocol.json',P);save(p/'preflight.json',selftest());rows=cases();save(p/'cases.json',rows)
 rt=Runtime();rt.hook.remove();rt.model.eval();save(p/'runtime.json',rt.meta);count=0
 with (p/'records.jsonl').open('w') as f:
  for i,r in enumerate(rows):
   if i%P['shards']!=args.shard:continue
   order=MODES[i%len(MODES):]+MODES[:i%len(MODES)]
   for mode in order:
    text=rt.tokenizer.apply_chat_template(conversation(r,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=rt.tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>P['max_input_tokens']:raise ValueError('Input exceeds frozen budget')
    x=torch.tensor([ids]);start=time.perf_counter()
    with torch.no_grad():y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=P['max_new_tokens'],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
    seconds=time.perf_counter()-start;tokens=y[0,len(ids):].tolist();raw=rt.tokenizer.decode(tokens,skip_special_tokens=True)
    rec={'id':r['id'],'source':r['source'],'family':r['family'],'edit':r['edit'],'mode':mode,'raw_output':raw,
         'output_token_ids':tokens,'input_tokens':len(ids),'output_tokens':len(tokens),'seconds':seconds,
         'token_limit':len(tokens)>=P['max_new_tokens'],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),**grade(raw,r,mode)}
    f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
    print(json.dumps({'event':'bridge_scored','done':count,'id':r['id'],'mode':mode}),flush=True)
 save(p/'receipt.json',{'n':count,'records_sha256':sha(p/'records.jsonl'),'protocol_sha256':digest(P),'weights_updated':False})

def aggregate(args):
 p=Path(args.out);p.mkdir(parents=True,exist_ok=False);seen=set();records=[];rows=cases()
 for path in Path(args.root).rglob('records.jsonl'):
  receipt=json.loads((path.parent/'receipt.json').read_text());assert receipt['records_sha256']==sha(path) and receipt['protocol_sha256']==digest(P)
  part=[json.loads(l) for l in path.read_text().splitlines()];assert len(part)==receipt['n']
  for r in part:
   key=(r['id'],r['mode']);assert key not in seen;seen.add(key);records.append(r)
 assert seen=={(r['id'],m) for r in rows for m in MODES}
 summary={}
 for m in MODES:
  rs=[r for r in records if r['mode']==m];fams={}
  for fam in P['families']:
   fr=[r for r in rs if r['family']==fam];src={r['source'] for r in fr}
   fams[fam]={'n':len(fr),'decision_correct':sum(r['decision_correct'] for r in fr),'scalar_correct':None if m=='decision_only' else sum(r['scalar_correct'] for r in fr),
              'complete_triplets':sum(all(r['decision_correct'] for r in fr if r['source']==s) for s in src),'sources':len(src)}
  summary[m]={'n':len(rs),'decision_correct':sum(r['decision_correct'] for r in rs),'scalar_correct':None if m=='decision_only' else sum(r['scalar_correct'] for r in rs),
              'all_fields_correct':sum(r['all_fields_correct'] for r in rs),'order_obeyed':sum(r['order_obeyed'] for r in rs),'schema_valid':sum(r['parsed'] is not None for r in rs),
              'token_limits':sum(r['token_limit'] for r in rs),'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rs),'families':fams}
 records.sort(key=lambda r:(r['id'],r['mode']));(p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));save(p/'results.json',{'summary':summary,'protocol':P});save(p/'cases.json',rows);save(p/'receipt.json',{'records':len(records),'records_sha256':sha(p/'records.jsonl'),'protocol_sha256':digest(P)})
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);sub.add_parser('selftest')
 r=sub.add_parser('run');r.add_argument('--shard',type=int,choices=range(P['shards']),required=True);r.add_argument('--out',required=True)
 a=sub.add_parser('aggregate');a.add_argument('--root',required=True);a.add_argument('--out',required=True)
 args=ap.parse_args()
 if args.command=='selftest':print(json.dumps(selftest(),indent=2))
 elif args.command=='run':run(args)
 else:aggregate(args)
