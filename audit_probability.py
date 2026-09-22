"""Independent recorded-data audit; no production probability/target/parser calls."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, random, re
from pathlib import Path
from decimal import Decimal, localcontext

def h(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def fh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False))

def decimal_vector(z,t):
    with localcontext() as c:
        c.prec=50
        zz=[Decimal(str(v)) for v in z]
        tt=Decimal(str(t))
        mx=max(zz)
        v=[(x-mx)/tt for x in zz]
        norm=sum(x.exp() for x in v)
        lognorm=norm.ln()
        return [float(x.exp()/norm) for x in v],[float(x-lognorm) for x in v]

def target(t):
    if t.get("target_probs") is not None:
        return list(t["target_probs"])
    gp=t.get("provenance",{}).get("gold_probs")
    if gp:
        return [gp.get(l,0.) for l in t["labels"]]
    return [1. if str(t["expected"])==l else 0. for l in t["labels"]]

def audit(root):
    root=Path(root);tasks=json.load(open(root/"probability-prepared/evaluation.json"))
    ti={t["id"]:t for t in tasks}
    records=json.load(open(root/"all_records.json"));ri={r["id"]:r for r in records}
    pred=json.load(open(root/"results/predictions.json"))
    report=json.load(open(root/"results/summary.json"))
    cal=json.load(open(root/"CALIBRATION.json"))
    manifest=json.load(open(root/"probability-prepared/manifest.json"))
    if len(ti)!=len(tasks) or len(ri)!=len(records) or len(tasks)!=441:
        raise ValueError("Population incomplete")
    originals=json.load(open(root/"probability-prepared/archive.json"))
    keys=[];fit=set(cal["fit_ids"]);fit_records=0
    source_targets=0
    for t in tasks:
        y=target(t)
        if y!=t["target"]:raise ValueError("Target reconstruction mismatch")
        source_targets+=1
        canonical={k:t[k] for k in ("state","question","labels")}
        keys.append(h(canonical))
        if t["partition"]=="calibration_fit":
            if t["id"] not in fit:raise ValueError("Missing fitting ID")
            fit_records+=1
        elif t["id"] in fit:raise ValueError("Evaluation contaminated fitting")
    if len(set(keys))!=441 or fit_records!=73:raise ValueError("Overlap")
    vector_error=loss_error=metric_error=0.
    n=0;fresh_readout_error=0.;native_checked=0
    reduced={}
    for arm,rr in pred.items():
        path="masked_final" if arm.startswith("masked_final") else arm.split("_")[0]
        calibrated=arm.endswith("_calibrated")
        temperature=cal["paths"][path]["temperature"] if calibrated else 1.
        cohort=collections.defaultdict(list)
        for i,result in rr.items():
            t=ti[i];r=ri[i]
            z=r["native"]["logits"] if path=="native" else r["readouts"][path]["logits"]
            pp,lp=decimal_vector(z,temperature)
            actual=[result["probabilities"][l] for l in t["labels"]]
            vector_error=max(vector_error,max(abs(a-b) for a,b in zip(pp,actual)))
            j=min(range(len(pp)),key=lambda k:(-pp[k],t["labels"][k]))
            if result["answer"]!=t["labels"][j]:raise ValueError("Label differs from probability maximum")
            yy=target(t)
            nll=-math.fsum(y*logp for y,logp in zip(yy,lp))
            bs=math.fsum((x-y)**2 for x,y in zip(pp,yy))
            loss_error=max(loss_error,abs(nll-result["cross_entropy"]),abs(bs-result["brier"]))
            correct=result["answer"]==str(t["expected"])
            if correct!=result["correct"]:raise ValueError("Answer scoring mismatch")
            cohort[t["partition"]].append((correct,nll,bs))
            n+=1
        for co,stats in cohort.items():
            expected=report["cohorts"][co][arm]
            if expected["n"]!=len(stats) or expected["correct"]!=sum(s[0] for s in stats):
                raise ValueError("Count mismatch")
            metric_error=max(metric_error,abs(math.fsum(s[1] for s in stats)/len(stats)-expected["target_cross_entropy"]),
                              abs(math.fsum(s[2] for s in stats)/len(stats)-expected["target_brier"]))
    for i,r in ri.items():
        old=originals[i]["native"]
        if r["native"]["logits"]!=old["logits"] or r["native"]["prompt_hash"]!=old["prompt_hash"]:
            raise ValueError("Native anchor drift")
        native_checked+=1
        for path in ("full","masked_final"):
            obs=r["readouts"][path]
            pp,_=decimal_vector(obs["logits"],1)
            fresh_readout_error=max(fresh_readout_error,max(abs(a-b) for a,b in zip(pp,obs["probabilities_uncalibrated"])))
    if n!=2646 or max(vector_error,loss_error,metric_error,fresh_readout_error)>1e-11:
        raise ValueError("Independent numerical audit failed")
    # Source-only temperature selection reproduced with high-precision arithmetic.
    fit_validation={}
    for path in ("native","full","masked_final"):
        losses=[]
        for temperature in manifest["temperature_grid"]:
            total=0.
            for i in sorted(fit):
                r=ri[i];z=r["native"]["logits"] if path=="native" else r["readouts"][path]["logits"]
                # Faster separate scalar logsumexp for the grid; high precision tested above.
                v=[(float(x)-max(z))/temperature for x in z]
                ln=math.log(math.fsum(math.exp(x) for x in v))
                total+=-math.fsum(y*(x-ln) for x,y in zip(v,target(ti[i])))
            losses.append(total/73)
        best=min(range(len(losses)),key=lambda k:(losses[k],abs(math.log(manifest["temperature_grid"][k]))))
        if manifest["temperature_grid"][best]!=cal["paths"][path]["temperature"]:raise ValueError("Fitted parameter mismatch")
        fit_validation[path]={"temperature":manifest["temperature_grid"][best],"fit_count":73}
    out={"vectors_checked":n,"targets_reconstructed":source_targets,"native_records_compared":native_checked,
         "max_probability_error_decimal50":vector_error,"max_individual_loss_error":loss_error,
         "max_aggregate_metric_error":metric_error,"max_raw_readout_probability_error":fresh_readout_error,
         "fit_test_overlap":0,"temperature_selection_reproduced":fit_validation,
         "scope":"Independent recorded-data arithmetic. Not a complete neural rerun or semantic calibration proof."}
    put(root/"results/independent_audit.json",out)
    print(json.dumps(out,indent=2))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default="data")
    a=p.parse_args();audit(a.root)
