"""A fixed semantic/coded-output ablation, not a leaderboard submission.
No training, oracle inputs, external teacher or test-time arithmetic solver.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path

MODES=("direct_code","state_code","state_relation","relation_then_code","code_then_relation","relation_no_codes")
P={"id":"decision0-semantic-bridge-v2","seed":230923173,"sources":8,"endpoints":3,"cases":24,
   "modes":MODES,"shards":8,"max_new_tokens":112,"max_input_tokens":1024,
   "model":"Qwen/Qwen3.5-4B","revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
   "training_updates":0,"benchmark_calls":0,"selection":"None; all predefined modes reported.",
   "scope":"Integer-cent comparison with jointly generated intermediates and output-label interventions; one controlled family."}
NUMERIC=("product_cents","subtotal_cents","total_cents")
def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))
def cases():
    out=[]
    for i in range(P["sources"]):
        rng=random.Random(int(digest([P["id"],P["seed"],i])[:16],16))
        q=rng.randint(2,17);u=rng.randint(41,1749);fee=rng.randint(51,241);credit=rng.randint(0,199)
        if i in (6,7):credit=q*u+fee+rng.randint(50,180)
        boundary=q*u+fee-credit
        names=["below","equal","above"];rng.shuffle(names)
        options=[{"code":chr(65+j),"meaning":name} for j,name in enumerate(names)]
        source=digest(["source",P["id"],i])[:20]
        # Labels, records and boundary are fixed inside the counterfactual family.
        for edit in (-1,0,1):
            out.append({"id":digest([source,edit])[:24],"source":source,"edit":edit,
                        "quantity":q,"unit_cents":u,"fee_cents":fee+edit,
                        "credit_cents":credit,"boundary_cents":boundary,"options":options})
    return out

def gold(r):
    a=r["quantity"]*r["unit_cents"];b=a+r["fee_cents"];c=b-r["credit_cents"]
    relation="below" if c<r["boundary_cents"] else "above" if c>r["boundary_cents"] else "equal"
    code=next(o["code"] for o in r["options"] if o["meaning"]==relation)
    return dict(zip(NUMERIC,(a,b,c)))|{"relation":relation,"decision_code":code}

def fields(mode):
    if mode=="direct_code":return ("decision_code",)
    if mode=="state_code":return NUMERIC+("decision_code",)
    if mode in ("state_relation","relation_no_codes"):return NUMERIC+("relation",)
    if mode=="relation_then_code":return NUMERIC+("relation","decision_code")
    if mode=="code_then_relation":return NUMERIC+("decision_code","relation")
    raise ValueError(mode)

def messages(r,mode):
    ks=fields(mode)
    system=("Calculate using exact integer cents. Product is quantity times unit price. "
            "Subtotal is product plus handling fee. Total is subtotal minus credit. Signed totals are allowed. "
            "Compare total with the comparison amount; do not include that comparison amount as an arithmetic operand. "
            "Return only one JSON object, with the following fields in exactly this order: "+", ".join(ks)+". ")
    if mode!="direct_code":
        system+="product_cents, subtotal_cents and total_cents must be integers with their named meanings. "
    if "relation" in ks:
        system+='relation must be the string "below", "equal" or "above" describing total relative to comparison amount. '
    if "decision_code" in ks:
        system+="decision_code must be the uppercase code in the supplied mapping for that comparison. "
    text=(f"Quantity {r['quantity']}. Unit price {r['unit_cents']} cents. "
          f"Handling fee {r['fee_cents']} cents. Credit {r['credit_cents']} cents. "
          f"Comparison amount {r['boundary_cents']} cents; this is not an arithmetic operand.")
    if mode!="relation_no_codes":
        text+=" Outcome codes: "+json.dumps(r["options"],separators=(",",":"))
    return [{"role":"system","content":system},{"role":"user","content":text}]

def parse(raw,mode):
    def unique_pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError("duplicate key")
            out[k]=v
        return out
    obj=json.loads(raw,object_pairs_hook=unique_pairs)
    if not isinstance(obj,dict) or set(obj)!=set(fields(mode)):raise ValueError("wrong fields")
    for k,v in obj.items():
        if k in NUMERIC:
            if type(v) is not int:raise ValueError("not an integer")
        elif k=="relation":
            if v not in ("below","equal","above"):raise ValueError("wrong relation")
        elif k=="decision_code":
            if v not in ("A","B","C"):raise ValueError("wrong code")
    return obj

def grade(raw,r,mode):
    expected=gold(r);obj=None;error=None
    try:obj=parse(raw,mode)
    except (ValueError,TypeError) as e:error=str(e)
    has_state=mode!="direct_code";has_relation="relation" in fields(mode);has_code="decision_code" in fields(mode)
    state_correct=None if not has_state else obj is not None and all(obj[k]==expected[k] for k in NUMERIC)
    relation_correct=None if not has_relation else obj is not None and obj["relation"]==expected["relation"]
    code_correct=None if not has_code else obj is not None and obj["decision_code"]==expected["decision_code"]
    # This is a name-to-code serialization lookup, not evaluation-key substitution.
    mapped=None
    if obj is not None and has_relation:
        mapped=next(o["code"] for o in r["options"] if o["meaning"]==obj["relation"])
    decision_correct=bool(relation_correct if has_relation and not has_code else code_correct)
    consistency=None
    if obj is not None and has_relation and has_code:consistency=mapped==obj["decision_code"]
    return {"parsed":obj,"parse_error":error,"schema_valid":obj is not None,
            "order_obeyed":obj is not None and tuple(obj)==fields(mode),
            "state_correct":state_correct,"relation_correct":relation_correct,"code_correct":code_correct,
            "decision_correct":decision_correct,"serialized_code":mapped,"code_relation_consistent":consistency,
            "strict_success":obj is not None and all(obj[k]==expected[k] for k in fields(mode))}

def selftest():
    from decimal import Decimal
    import itertools
    rows=cases();seen=set();checks=0
    assert len(rows)==24
    for r in rows:
        independent=Decimal(r["quantity"])*Decimal(r["unit_cents"])+Decimal(r["fee_cents"])-Decimal(r["credit_cents"])
        assert int(independent)==gold(r)["total_cents"]
        for mode in MODES:
            m=messages(r,mode);s=json.dumps(m)
            assert r["id"] not in s and r["source"] not in s
            assert s not in seen;seen.add(s)
            expected=gold(r);raw=json.dumps({k:expected[k] for k in fields(mode)})
            assert grade(raw,r,mode)["strict_success"]
            if mode!="direct_code":
                assert not grade(raw.replace('"product_cents":','"product_cents":true,"product_cents":',1),r,mode)["strict_success"]
            checks+=1
        for perm in itertools.permutations(("below","equal","above")):
            test=dict(r,options=[{"code":chr(65+i),"meaning":n} for i,n in enumerate(perm)])
            assert gold(test)["relation"]==gold(r)["relation"]
            assert [o["meaning"] for o in test["options"] if o["code"]==gold(test)["decision_code"]]==[gold(r)["relation"]]
            checks+=1
    for group in (rows[i:i+3] for i in range(0,24,3)):
        assert len({r["boundary_cents"] for r in group})==1
        assert len({json.dumps(r["options"]) for r in group})==1
        assert {gold(r)["relation"] for r in group}=={"below","equal","above"}
        assert len({r["quantity"] for r in group})==1
        assert len({r["unit_cents"] for r in group})==1
    for bad in ('{"decision_code":true}','{"decision_code":"D"}','{"decision_code":"A","decision_code":"B"}','null'):
        assert not grade(bad,rows[0],"direct_code")["schema_valid"];checks+=1
    return {"checks":checks,"cases":len(rows),"source_groups":8,"prompt_hashes":len(seen),
            "corpus_sha256":digest(rows),"protocol_sha256":digest(P)}

def run(args):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"input_flow_v1"))
    from flow_study import Runtime
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False)
    save(p/"protocol.json",P);save(p/"preflight.json",selftest());rows=cases();save(p/"cases.json",rows)
    rt=Runtime();rt.hook.remove();rt.model.eval();save(p/"runtime.json",rt.meta)
    n=0
    with (p/"records.jsonl").open("w") as f:
        for i,r in enumerate(rows):
            if i%P["shards"]!=args.shard:continue
            modes=MODES[i%len(MODES):]+MODES[:i%len(MODES)]
            for mode in modes:
                text=rt.tokenizer.apply_chat_template(messages(r,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=rt.tokenizer.encode(text,add_special_tokens=False)
                if not 0<len(ids)<=P["max_input_tokens"]:raise ValueError("input limit; no truncation")
                x=torch.tensor([ids]);tick=time.perf_counter()
                with torch.no_grad():
                    y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,
                        max_new_tokens=P["max_new_tokens"],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
                ts=y[0,len(ids):].tolist();raw=rt.tokenizer.decode(ts,skip_special_tokens=True)
                rec={"id":r["id"],"source":r["source"],"edit":r["edit"],"mode":mode,"raw_output":raw,
                     "generated_token_ids":ts,"input_tokens":len(ids),"output_tokens":len(ts),
                     "seconds":time.perf_counter()-tick,"token_limit":len(ts)>=P["max_new_tokens"],
                     "prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),**grade(raw,r,mode)}
                f.write(json.dumps(rec,allow_nan=False)+"\n");f.flush();n+=1
                print(json.dumps({"phase":"semantic_bridge","shard":args.shard,"completed":n,"mode":mode}),flush=True)
    save(p/"receipt.json",{"records":n,"records_sha256":sha(p/"records.jsonl"),"protocol_sha256":digest(P),"weights_updated":False})

def aggregate(args):
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False);records=[];seen=set()
    for f in Path(args.root).rglob("records.jsonl"):
        r=json.loads((f.parent/"receipt.json").read_text())
        assert sha(f)==r["records_sha256"] and r["protocol_sha256"]==digest(P)
        rows=[json.loads(s) for s in f.read_text().splitlines()];assert len(rows)==r["records"]
        for rec in rows:
            key=(rec["id"],rec["mode"]);assert key not in seen;seen.add(key);records.append(rec)
    assert seen=={(r["id"],m) for r in cases() for m in MODES},"Incomplete evaluation"
    summary={}
    for mode in MODES:
        rs=[r for r in records if r["mode"]==mode]
        s={"n":len(rs),"schema_valid":sum(r["schema_valid"] for r in rs),
           "token_limits":sum(r["token_limit"] for r in rs),"output_tokens":sum(r["output_tokens"] for r in rs)}
        for metric in ("state_correct","relation_correct","code_correct","decision_correct","strict_success","code_relation_consistent"):
            xs=[r[metric] for r in rs if r[metric] is not None];s[metric]=sum(xs) if xs else None
            if xs:s[metric+"_complete_sources"]=sum(all(r[metric] for r in rs if r["source"]==src) for src in {r["source"] for r in rs})
        summary[mode]=s
    records.sort(key=lambda r:(r["id"],r["mode"]))
    (p/"records.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records))
    save(p/"results.json",{"protocol":P,"summary":summary})
    save(p/"cases.json",cases());save(p/"receipt.json",{"records":len(records),"records_sha256":sha(p/"records.jsonl"),"protocol_sha256":digest(P)})

if __name__=="__main__":
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="cmd",required=True);sub.add_parser("selftest")
    a=sub.add_parser("run");a.add_argument("--shard",type=int,choices=range(8),required=True);a.add_argument("--out",required=True)
    a=sub.add_parser("aggregate");a.add_argument("--root",required=True);a.add_argument("--out",required=True)
    args=parser.parse_args()
    if args.cmd=="selftest":print(json.dumps(selftest(),indent=2))
    elif args.cmd=="run":run(args)
    else:aggregate(args)
