"""Bounded native-readout LoRA pilot; no automatic promotion or hosted model API."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,resource,sys,time,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
from data import canonical,curriculum,digest,checks

MODEL="Qwen/Qwen3.5-4B"
REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HASHES={"model.safetensors-00001-of-00002.safetensors":"26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
"model.safetensors-00002-of-00002.safetensors":"cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"}
JEV="51a8d73fa798aa337bb1b26abd10995c0ab847e9"
DATA_HASH={"easy":"231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
"original":"5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"}
SALT="adaptive-native-training-v03-regression\0"
STEPS=48
SEED=1703

def filehash(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def save(p,obj):
    p=Path(p);tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)

def fetch(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()

def external(out):
    """Two input-hash-selected examples per easy/original tier-family; never selects using target."""
    groups={};sources=out/"sources";sources.mkdir(exist_ok=True)
    for tier,h in DATA_HASH.items():
        raw=fetch(f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/datasets/public/{tier}.jsonl")
        if digest(raw)!=h:raise ValueError("external data identity changed")
        (sources/(tier+".jsonl")).write_bytes(raw)
        for line in raw.splitlines():
            if line:
                r=json.loads(line)
                groups.setdefault((tier,r["family"]),[]).append(r)
    selected=[]
    for group,rs in sorted(groups.items()):
        rs.sort(key=lambda r:digest(SALT+canonical({k:r[k] for k in ("state","question","labels")})))
        for r in rs[:2]:
            selected.append({"id":r["id"],"task":{k:r[k] for k in ("state","question","labels")},
                             "target":str(r["expected"]),"split":"jev_public_regression",
                             "family":r["family"],"group":r.get("group")})
    if len(selected)!=20:raise ValueError(f"Expected 20 external smoke cases, found {len(selected)}")
    raw=fetch(f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV}/jevbench/scoring.py")
    (sources/"scoring.py").write_bytes(raw)
    return selected

def main(args):
    import torch,transformers
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    # Frozen renderer is copied from Native v0.2; this experiment does not alter its prompts.
    import readout
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    begun=time.perf_counter()
    metadata={"status":"started","scope":args.scope,"steps_planned":STEPS,"seed":SEED,
              "model":MODEL,"revision":REV,"base_dtype":"bfloat16","adapter_dtype":"float32",
              "rank":4,"alpha":8,"learning_rate":1e-4,"accumulation":2,
              "training_order":"single seeded shuffle, 96 unique decisions, one pass",
              "selection":"highest development accuracy then lowest NLL; checkpoints 24/48",
              "previous_quantized_runtime_not_reproduced":True,
              "prompt_source_sha256":filehash(readout.__file__),"python":sys.version,
              "torch":torch.__version__,"transformers":transformers.__version__,
              "run_id":os.environ.get("GITHUB_RUN_ID"),"source_commit":os.environ.get("GITHUB_SHA"),
              "pid":os.getpid(),"gpu_used":False,"paid_api":False,"memory_used":False,
              "calibration_fitted":False,"official_jevbench_score":None,
              "promotion":False,"general_reasoning_retention_measured":False}
    save(out/"metadata.json",metadata)
    data=curriculum();save(out/"authored_data.json",data)
    save(out/"protocol.json",{"data":checks(),"metadata":metadata,
          "source_hashes":{p.name:filehash(p) for p in Path(__file__).parent.glob("*.py")}})
    try:
        local=snapshot_download(MODEL,revision=REV,allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja","LICENSE*"],max_workers=2)
        actual={p.name:filehash(p) for p in Path(local).glob("*.safetensors")}
        if actual!=HASHES:raise ValueError("base weight checksum mismatch")
        tok=transformers.AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        model,loadinfo=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            local,dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True,
            trust_remote_code=False,output_loading_info=True)
        bad={k:v for k,v in loadinfo.items() if k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs") and v}
        if bad:raise ValueError("incomplete pretrained model load: "+str(bad))
        model.requires_grad_(False);model.eval()
        data["jev_public_regression"]=external(out)
        save(out/"evaluation_data.json",{k:v for k,v in data.items() if k!="train"})
        cache={}
        for split,rs in data.items():
            for r in rs:
                messages,ledger=readout.messages(r["task"],"canonical")
                text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(text,add_special_tokens=False)
                if len(ids)>512:raise ValueError(f"No truncation permitted: {r['id']} has {len(ids)} tokens")
                slots=[tok.encode(row["marker"],add_special_tokens=False) for row in ledger]
                if any(len(x)!=1 for x in slots):raise ValueError("marker not single token")
                for row,slot in zip(ledger,slots):
                    if tok.encode(text+row["marker"],add_special_tokens=False)!=ids+slot:
                        raise ValueError("marker token boundary")
                labels=[row["label"] for row in ledger]
                cache[r["id"]]={"inputs":{"input_ids":torch.tensor([ids]),
                  "attention_mask":torch.ones(1,len(ids),dtype=torch.long),"use_cache":False,
                  "return_dict":True,"logits_to_keep":1},"slots":[s[0] for s in slots],
                  "labels":labels,"prompt_sha256":digest(text),"tokens":len(ids),
                  "target":labels.index(r["target"])}
        save(out/"token_manifest.json",{k:{kk:vv for kk,vv in v.items() if kk!="inputs"} for k,v in cache.items()})
        def logits(key):
            c=cache[key]
            return model(**c["inputs"]).logits[0,-1,c["slots"]].float()
        def evaluate(tag,splits):
            model.eval();records=[];scores={}
            with torch.inference_mode(), (out/(tag+".jsonl")).open("x") as stream:
                for split in splits:
                    subset=[]
                    for r in data[split]:
                        c=cache[r["id"]];t=time.perf_counter();z=logits(r["id"])
                        p=z.softmax(-1).tolist();pred=sorted(zip(c["labels"],p),key=lambda x:(-x[1],x[0]))[0][0]
                        row={"id":r["id"],"split":split,"probs":dict(zip(c["labels"],p)),
                             "logits":z.tolist(),"correct":pred==r["target"],"target":r["target"],
                             "predicted":pred,"nll":-math.log(max(1e-30,p[c["target"]])),
                             "seconds":time.perf_counter()-t,"prompt_sha256":c["prompt_sha256"]}
                        stream.write(canonical(row)+"\n");stream.flush();os.fsync(stream.fileno())
                        subset.append(row);records.append(row)
                    scores[split]={"correct":sum(r["correct"] for r in subset),"n":len(subset),
                      "nll":sum(r["nll"] for r in subset)/len(subset)}
            save(out/(tag+"-summary.json"),scores)
            print("EVALUATION "+canonical({"tag":tag,"scores":scores}),flush=True)
            return scores
        eval_splits=["development","transfer","composition","jev_public_regression"]
        baseline=evaluate("baseline",eval_splits)
        class Factors(torch.nn.Module):
            def __init__(self,down):
                super().__init__()
                self.a=torch.nn.Parameter(torch.empty(4,down.in_features))
                self.b=torch.nn.Parameter(torch.zeros(down.out_features,4))
                torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
            def forward(self,x):
                return (2*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
        targets=[(n,m) for n,m in model.named_modules() if n.startswith("model.language_model.layers.") and n.endswith(".mlp.down_proj") and isinstance(m,torch.nn.Linear)]
        if len(targets)!=32:raise ValueError(f"expected 32 block projections; got {len(targets)}")
        if args.scope=="terminal":targets=targets[-1:]
        factors=torch.nn.ModuleList([Factors(m) for _,m in targets])
        hooks=[]
        for (_,module),f in zip(targets,factors):
            hooks.append(module.register_forward_hook(lambda module,inputs,result,f=f:result+f(inputs[0])))
        metadata.update(adapter_parameters=sum(p.numel() for p in factors.parameters()),
                        targets=[n for n,m in targets],weight_hashes=actual,
                        setup_seconds=time.perf_counter()-begun)
        with torch.inference_mode():
            reference=json.loads((out/"baseline.jsonl").read_text().splitlines()[0])
            zero=logits(reference["id"]).tolist()
            err=max(abs(a-b) for a,b in zip(zero,reference["logits"]))
        if err>1e-4:raise ValueError("zero adapter changed baseline")
        metadata["zero_adapter_max_difference"]=err
        save(out/"metadata.json",metadata)
        optimizer=torch.optim.AdamW(factors.parameters(),lr=1e-4,weight_decay=0)
        sequence=list(data["train"]);random.Random(SEED).shuffle(sequence)
        assert len(sequence)==STEPS*2
        best=None;best_state=None
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        with (out/"training.jsonl").open("x") as journal:
            for step in range(1,STEPS+1):
                if time.perf_counter()-begun>24*60:raise TimeoutError("bounded experiment wall budget")
                model.train();optimizer.zero_grad(set_to_none=True);loss_total=0.;tick=time.perf_counter()
                rows=sequence[(step-1)*2:step*2]
                for r in rows:
                    z=logits(r["id"])
                    loss=torch.nn.functional.cross_entropy(z[None,:],torch.tensor([cache[r["id"]]["target"]]))/2
                    if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
                    loss.backward();loss_total+=float(loss.detach())
                gradnorms={n:float(f.b.grad.float().norm()) if f.b.grad is not None else None for (n,m),f in zip(targets,factors)}
                if any(v is None or not math.isfinite(v) for v in gradnorms.values()):raise ValueError("missing/nonfinite block gradients")
                if any(p.grad is not None for p in model.parameters()):raise ValueError("original parameter received gradient")
                if step==1 and any(v<=0 for v in gradnorms.values()):raise ValueError("zero initial block gradient")
                torch.nn.utils.clip_grad_norm_(factors.parameters(),1.0);optimizer.step()
                rec={"step":step,"loss":loss_total,"ids":[r["id"] for r in rows],
                     "seconds":time.perf_counter()-tick,"block_B_gradient_norms":gradnorms}
                journal.write(canonical(rec)+"\n");journal.flush();os.fsync(journal.fileno())
                if step%4==0:print("TRAIN "+canonical({k:v for k,v in rec.items() if k!="block_B_gradient_norms"}),flush=True)
                if step in (24,48):
                    dev=evaluate(f"development-step{step}",["development"])["development"]
                    value=(dev["correct"],-dev["nll"])
                    state={k:v.detach().contiguous().cpu().clone() for k,v in factors.state_dict().items()}
                    save_file(state,str(out/f"step-{step}.safetensors"))
                    if best is None or value>best:
                        best=value;best_state=state;metadata["selected_step"]=step
        factors.load_state_dict(best_state)
        model.eval();model.gradient_checkpointing_disable();model.disable_input_require_grads()
        save_file(best_state,str(out/"adapter.safetensors"))
        final=evaluate("adapted",eval_splits)
        metadata.update(status="completed",completed_updates=STEPS,baseline=baseline,adapted=final,
                        adapter_sha256=filehash(out/"adapter.safetensors"),
                        changed_projection_count=sum(bool(f.b.detach().abs().max()>0) for f in factors),
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        for h in hooks:h.remove()
        with torch.inference_mode():restored=logits(reference["id"]).tolist()
        restoration=max(abs(a-b) for a,b in zip(restored,reference["logits"]))
        metadata["base_restoration_max_difference"]=restoration
        if restoration>1e-4:raise ValueError("base restore failed")
        metadata["base_weight_files_unchanged"]=all(filehash(Path(local)/n)==h for n,h in HASHES.items())
        if not metadata["base_weight_files_unchanged"]:raise ValueError("base files changed")
    except Exception as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        raise
    finally:
        metadata["wall_seconds"]=time.perf_counter()-begun
        save(out/"metadata.json",metadata)
        print("RESULT "+canonical(metadata),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scope",choices=["terminal","distributed"],required=True)
    p.add_argument("--out",required=True);a=p.parse_args();main(a)
