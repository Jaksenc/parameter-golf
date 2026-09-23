"""Fixed-format diagnostic; no training, oracle inputs, or benchmark decisions."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path

MODES=('state','sham','boundary','joint','named_joint','decision')
P={'id':'decision0-query-interference-v1','seed':23092261,'sources':8,'endpoints':3,'cases':24,
   'modes':MODES,'shards':8,'max_new_tokens':88,'max_input_tokens':1024,'model':'Qwen/Qwen3.5-4B',
   'revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','training_updates':0,'benchmark_calls':0,
   'selection':'None. Fixed checkpoint, cases and modes; report every outcome.',
   'scope':'Paired integer-cent arithmetic with task-cue/output-schema interventions; not broad language evaluation.'}
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))
def cases():
    rows=[]
    for i in range(P['sources']):
        r=random.Random(int(digest([P['id'],i,P['seed']])[:16],16))
        q=r.randint(2,18);u=r.randint(31,1499);f=r.randint(20,180);c=r.randint(0,150)
        if i in (6,7): c=q*u+f+r.randint(10,90)
        boundary=q*u+f-c;labels=['below','equal','above'];r.shuffle(labels)
        opts=[{'code':chr(65+j),'meaning':v} for j,v in enumerate(labels)]
        source=digest(['source',i])[:16]
        for d in (-1,0,1):
            rows.append({'id':digest([source,d])[:24],'source':source,'edit':d,
                'quantity':q,'unit_cents':u,'fee_cents':f+d,'credit_cents':c,'boundary':boundary,'options':opts})
    return rows

def expected(row):
    a=row['quantity']*row['unit_cents'];b=a+row['fee_cents'];c=b-row['credit_cents']
    relation='below' if c<row['boundary'] else 'above' if c>row['boundary'] else 'equal'
    return {'a':a,'b':b,'c':c,'d':next(o['code'] for o in row['options'] if o['meaning']==relation)}
def keys(mode):
    if mode in ('state','sham','boundary'): return ['a','b','c']
    if mode=='joint': return ['a','b','c','d']
    if mode=='named_joint': return ['product_cents','subtotal_cents','total_cents','decision']
    if mode=='decision': return ['d']
    raise ValueError(mode)
def conversation(row,mode):
    system='Calculate using exact integer cents. Product is quantity times unit price. Subtotal is product plus handling fee. Total is subtotal minus credit. Signed totals are allowed. '
    if mode=='named_joint':
        system+='Return only one JSON object with integer fields product_cents, subtotal_cents, total_cents and string decision, in that order. Each numeric field is its named quantity. '
    elif mode=='decision': system+='Return only one JSON object with string field d. '
    else:
        system+='Return only one JSON object with integer fields a, b, c'+(' and string d' if mode=='joint' else '')+', in that order; a is product, b is subtotal, and c is total. '
    if mode in ('joint','named_joint','decision'):
        system+='The final string is the uppercase code matching whether total is below, equal to, or above the comparison amount. Do not include the comparison amount in the arithmetic.'
    text=f"Quantity {row['quantity']}. Unit price {row['unit_cents']} cents. Handling fee {row['fee_cents']} cents. Credit {row['credit_cents']} cents."
    if mode=='sham': text+=f" Audit reference number {row['boundary']}; this is not an arithmetic operand."
    if mode in ('boundary','joint','named_joint','decision'):
        text+=f" Comparison amount {row['boundary']} cents; this is not an arithmetic operand. Outcome codes: "+json.dumps(row['options'],separators=(',',':'))
    return [{'role':'system','content':system},{'role':'user','content':text}]

def grade(raw,row,mode):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate key')
            out[k]=v
        return out
    g=expected(row);obj=None;err=None
    try:
        obj=json.loads(raw,object_pairs_hook=pairs)
        if not isinstance(obj,dict) or set(obj)!=set(keys(mode)): raise ValueError('wrong fields')
        for k,v in obj.items():
            if k in ('d','decision'):
                if type(v) is not str or v not in ('A','B','C'):raise ValueError('wrong decision')
            elif type(v) is not int:raise ValueError('wrong integer')
    except (ValueError,TypeError) as e:obj=None;err=str(e)
    mapped=None
    if obj is not None:
        mapped={({'product_cents':'a','subtotal_cents':'b','total_cents':'c','decision':'d'}.get(k,k)):v for k,v in obj.items()}
    return {'parsed':obj,'error':err,'order_obeyed':obj is not None and list(obj)==keys(mode),
        'total_correct':None if mode=='decision' else mapped is not None and mapped['c']==g['c'],
        'states_correct':None if mode=='decision' else mapped is not None and all(mapped[k]==g[k] for k in ('a','b','c')),
        'decision_correct':None if mode in ('state','sham','boundary') else mapped is not None and mapped['d']==g['d'],
        'expected':g}
def selftest():
    from decimal import Decimal
    rows=cases();assert len(rows)==24;seen=set()
    for r in rows:
        v=Decimal(r['quantity'])*Decimal(r['unit_cents'])+Decimal(r['fee_cents'])-Decimal(r['credit_cents']);assert v==expected(r)['c']
        for m in MODES:
            text=json.dumps(conversation(r,m));assert r['id'] not in text and r['source'] not in text
            g=expected(r);names={'product_cents':'a','subtotal_cents':'b','total_cents':'c','decision':'d'}
            raw=json.dumps({k:g[names.get(k,k)] for k in keys(m)})
            check=grade(raw,r,m);assert check['order_obeyed'] and check['total_correct'] is not False and check['decision_correct'] is not False
            seen.add(digest(conversation(r,m)))
    assert len(seen)==144
    for i in range(0,24,3):
        group=rows[i:i+3];assert len({expected(r)['d'] for r in group})==3
        assert len({r['boundary'] for r in group})==1
        assert len({json.dumps(r['options']) for r in group})==1
    assert grade('{"a":1,"a":2}',rows[0],'state')['parsed'] is None
    return {'cases':24,'source_groups':8,'input_hashes':144,'corpus_sha256':digest(rows),'protocol_sha256':digest(P)}
def run(args):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
    from flow_study import Runtime
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False);save(p/'protocol.json',P);save(p/'preflight.json',selftest());rows=cases();save(p/'cases.json',rows)
    rt=Runtime();rt.hook.remove();rt.model.eval();save(p/'runtime.json',rt.meta);n=0
    with (p/'records.jsonl').open('w') as f:
        for i,r in enumerate(rows):
            if i%P['shards']!=args.shard:continue
            order=MODES[i%6:]+MODES[:i%6]
            for mode in order:
                text=rt.tokenizer.apply_chat_template(conversation(r,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=rt.tokenizer.encode(text,add_special_tokens=False);assert 0<len(ids)<=P['max_input_tokens']
                x=torch.tensor([ids]);t=time.perf_counter()
                with torch.no_grad():y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=P['max_new_tokens'],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
                ts=y[0,len(ids):].tolist();raw=rt.tokenizer.decode(ts,skip_special_tokens=True)
                rec={'id':r['id'],'source':r['source'],'edit':r['edit'],'mode':mode,'raw_output':raw,'generated_token_ids':ts,'input_tokens':len(ids),'output_tokens':len(ts),'seconds':time.perf_counter()-t,'token_limit':len(ts)>=P['max_new_tokens'],'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),**grade(raw,r,mode)}
                f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();n+=1
                print(json.dumps({'scored':n,'shard':args.shard,'mode':mode}),flush=True)
    save(p/'receipt.json',{'records':n,'records_sha256':sha(p/'records.jsonl'),'protocol_sha256':digest(P),'weights_updated':False})
def aggregate(args):
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False);rs=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
        receipt=json.loads((f.parent/'receipt.json').read_text());assert receipt['records_sha256']==sha(f) and receipt['protocol_sha256']==digest(P)
        part=[json.loads(l) for l in f.read_text().splitlines()];assert len(part)==receipt['records']
        for r in part:
            k=(r['id'],r['mode']);assert k not in seen;seen.add(k);rs.append(r)
    assert seen=={(r['id'],m) for r in cases() for m in MODES}
    summary={}
    for m in MODES:
        a=[r for r in rs if r['mode']==m];s={'n':len(a),'schema_valid':sum(r['parsed'] is not None for r in a),'order_obeyed':sum(r['order_obeyed'] for r in a),'token_limits':sum(r['token_limit'] for r in a)}
        for metric in ('total_correct','states_correct','decision_correct'):
            s[metric]=None if a[0][metric] is None else sum(r[metric] for r in a)
            if a[0][metric] is not None:s[metric+'_triplets']=sum(all(r[metric] for r in a if r['source']==src) for src in {r['source'] for r in a})
        summary[m]=s
    rs.sort(key=lambda r:(r['id'],r['mode']));(p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs))
    save(p/'results.json',{'protocol':P,'summary':summary});save(p/'cases.json',cases());save(p/'receipt.json',{'records':len(rs),'records_sha256':sha(p/'records.jsonl'),'protocol_sha256':digest(P)})
if __name__=='__main__':
    a=argparse.ArgumentParser();s=a.add_subparsers(dest='cmd',required=True);s.add_parser('selftest')
    r=s.add_parser('run');r.add_argument('--shard',type=int,choices=range(8),required=True);r.add_argument('--out',required=True)
    t=s.add_parser('aggregate');t.add_argument('--root',required=True);t.add_argument('--out',required=True)
    x=a.parse_args()
    if x.cmd=='selftest':print(json.dumps(selftest(),indent=2))
    elif x.cmd=='run':run(x)
    else:aggregate(x)
