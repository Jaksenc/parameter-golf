"""Frozen fitting and paired evaluation for measured Probability v13 outputs."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import probability_v13 as exp
import probability_boundary as boundary

PATHS=("native","full","masked_final")
SEED=13260922

def load(root):
    root=Path(root)
    prepared=root/"probability-prepared"
    m=json.loads((prepared/"manifest.json").read_text())
    tasks=json.loads((prepared/"evaluation.json").read_text())
    rows=json.loads((root/"all_records.json").read_text())
    archive=json.loads((prepared/"archive.json").read_text())
    if exp.digest(tasks)!=m["evaluation_hash"] or exp.digest(archive)!=m["archive_hash"]:
        raise ValueError("Input/target archive changed")
    if len(rows)!=441 or len({r["id"] for r in rows})!=441 or {r["id"] for r in rows}!={t["id"] for t in tasks}:
        raise ValueError("Incomplete population")
    ti={t["id"]:t for t in tasks}
    for r in rows:
        if r["input_hash"]!=exp.digest(exp.request(ti[r["id"]])) or set(r["readouts"])!=set(exp.READOUTS):
            raise ValueError("Input mismatch or incomplete readout")
    return m,ti,{r["id"]:r for r in rows},archive

def logits(r,path):
    return r["native"]["logits"] if path=="native" else r["readouts"][path]["logits"]

def fit(root):
    root=Path(root);m,tasks,rows,archive=load(root)
    selected=sorted((t for t in tasks.values() if t["partition"]=="calibration_fit"),key=lambda t:t["id"])
    if len(selected)!=73:raise ValueError("Calibration population")
    result={"model_revision":m["model_revision"],"scientific_source_sha256":m["source_sha256"],
            "fit_ids":[t["id"] for t in selected],"method":"one fixed grid temperature per path",
            "evaluation_used_to_fit":False,"paths":{}}
    for path in PATHS:
        examples=[{"partition":"calibration_fit","id":t["id"],"logits":logits(rows[t["id"]],path),"target":t["target"]} for t in selected]
        fitted=boundary.fit_temperature(examples,m["temperature_grid"])
        fitted["fit_record_hash"]=exp.digest(examples)
        result["paths"][path]=fitted
    exp.write(root/"CALIBRATION.json",result)
    print(json.dumps({p:{k:v for k,v in r.items() if k!="loss_curve"} for p,r in result["paths"].items()},indent=2))
    return result

def vector(z,t):
    z=np.asarray(z,dtype=np.float64)
    v=(z-z.max())/t
    lp=v-np.log(np.exp(v).sum())
    return lp,np.exp(lp)

def pred_index(p,labels):
    # Official scorer resolves exact ties lexicographically.
    return min(range(len(p)),key=lambda i:(-float(p[i]),labels[i]))

def summary(rr):
    if not rr:return {"n":0}
    def mean(name):return float(np.mean([r[name] for r in rr]))
    def ece(rows,field):
        if not rows:return None
        total=0.
        for i in range(10):
            items=[r for r in rows if min(int(r["confidence"]*10),9)==i]
            if items:
                total+=len(items)/len(rows)*abs(np.mean([r["confidence"] for r in items])-np.mean([r[field] for r in items]))
        return float(total)
    hard=[r for r in rr if not r["soft_target"]]
    soft=[r for r in rr if r["soft_target"]]
    return {"n":len(rr),"correct":sum(r["correct"] for r in rr),"accuracy":mean("correct"),
            "target_cross_entropy":mean("cross_entropy"),"target_brier":mean("brier"),
            "ece_modal_all":ece(rr,"correct"),"ece_deterministic_only":ece(hard,"correct"),
            "ece_expected_target_mass":ece(rr,"target_mass"),"mean_max_probability":mean("confidence"),
            "soft_target_count":len(soft),"soft_target_mean_tvd":float(np.mean([r["tvd"] for r in soft])) if soft else None,
            "overconfident_errors_95":sum(not r["correct"] and r["confidence"]>=.95 for r in rr)}

def paired(values,tasks,ids,draws=10000):
    d=np.asarray(values,dtype=float)
    rng=np.random.default_rng(SEED)
    groups=collections.defaultdict(list)
    public=all(tasks[i]["partition"]=="jevbench" for i in ids)
    for n,i in enumerate(ids):
        t=tasks[i]
        group=(t.get("group") or i) if public else (t.get("family") or t.get("source") or "source")
        groups[group].append(n)
    if public:
        sums=np.array([d[v].sum() for v in groups.values()])
        sizes=np.array([len(v) for v in groups.values()])
        choices=rng.integers(len(sizes),size=(draws,len(sizes)))
        boot=sums[choices].sum(1)/sizes[choices].sum(1)
        method="paired source-group bootstrap"
    else:
        boot=np.zeros(draws)
        for k in sorted(groups):
            values=d[groups[k]]
            ix=rng.integers(len(values),size=(draws,len(values)))
            boot+=values[ix].sum(1)
        boot/=len(d)
        method="paired item bootstrap within source/family"
    return {"mean_difference":float(d.mean()),"interval95":list(map(float,np.quantile(boot,[.025,.975]))),
            "groups_or_strata":len(groups),"draws":draws,"method":method,
            "scope":"Historical finite samples; unadjusted for repeated use, multiple comparisons and pretraining exposure."}

def parse_final(text,labels,cut):
    # Independent parser rather than importing the production final parser.
    found=[]
    lines=text.split("\n")
    for i,line in enumerate(lines):
        if line.startswith("FINAL:"):
            x=line[6:].strip()
            if x in labels and (i<len(lines)-1 or not cut):found.append(x)
    return found[0] if len(found)==1 else None

def moments(xs):
    a=np.asarray(xs,dtype=float)
    return {"n":len(xs),"sum":float(a.sum()),"mean":float(a.mean()),
            "median":float(np.median(a)),"p95":float(np.quantile(a,.95))}

def score(root):
    root=Path(root);m,tasks,rows,archive=load(root)
    cal=json.loads((root/"CALIBRATION.json").read_text())
    # Reproduce fitted values from fit records; no test values can enter.
    fit_ids=sorted(i for i,t in tasks.items() if t["partition"]=="calibration_fit")
    if fit_ids!=cal["fit_ids"] or cal["scientific_source_sha256"]!=m["source_sha256"]:
        raise ValueError("Calibration identity mismatch")
    for path in PATHS:
        examples=[{"partition":"calibration_fit","id":i,"logits":logits(rows[i],path),"target":tasks[i]["target"]} for i in fit_ids]
        if exp.digest(examples)!=cal["paths"][path]["fit_record_hash"]:raise ValueError("Calibration observations changed")
        assert boundary.fit_temperature(examples,m["temperature_grid"])["temperature"]==cal["paths"][path]["temperature"]
    distributions={f"{p}_{scale}":{} for p in PATHS for scale in ("raw","calibrated")}
    ordinary={};quality=[];direct_matches=0;prior_fallbacks=0;max_prior_error=0.
    times={};masked_changes=0;readout_aliases=0
    for i in sorted(tasks):
        task,r=tasks[i],rows[i];labels=task["labels"];target=np.asarray(task["target"],dtype=float)
        reference=archive[i]["native"]
        if reference["logits"]!=r["native"]["logits"] or reference["prompt_hash"]!=r["native"]["prompt_hash"]:
            raise ValueError("Recomputed native inference changed")
        direct_matches+=1
        if r["trace_reused"]:
            if r["trace"]!=archive[i]["trace"]:raise ValueError("Archived trace changed")
            ordinary[i]=archive[i]["ordinary_answer"]
            prev=archive[i]["old_fallback"]
            if prev:
                prior_fallbacks+=1
                if prev["prompt_sha256"]!=r["readouts"]["full"]["prompt_sha256"]:raise ValueError("Fallback prompt changed")
                e=max(abs(a-b) for a,b in zip(prev["logits"],r["readouts"]["full"]["logits"]))
                max_prior_error=max(max_prior_error,e)
                if e>1e-4:raise ValueError("Previously measured readout changed")
        else:
            answer=parse_final(r["trace"]["text"],labels,r["trace"]["hit_cap"])
            ordinary[i]=answer or r["readouts"]["full"]["label"]
        mh=hashlib.sha256(re.sub(r"(?m)^[ \t]*FINAL:[^\r\n]*(?:\r?\n|$)","",r["trace"]["text"]).encode()).hexdigest()
        if mh!=r["masked_text_hash"]:raise ValueError("Mask hash mismatch")
        masked_changes+=mh!=hashlib.sha256(r["trace"]["text"].encode()).hexdigest()
        for mode in exp.READOUTS:
            expected_hash=hashlib.sha256(r["trace"]["text"].encode()).hexdigest() if mode=="full" else mh
            if r["readouts"][mode]["draft_sha256"]!=expected_hash:raise ValueError("Draft provenance")
        readout_aliases+=r["readouts"]["full"]["logits"]==r["readouts"]["masked_final"]["logits"]
        for path in PATHS:
            z=logits(r,path)
            for scale in ("raw","calibrated"):
                temperature=1. if scale=="raw" else cal["paths"][path]["temperature"]
                lp,p=vector(z,temperature);j=pred_index(p,labels);answer=labels[j]
                row={"id":i,"answer":answer,"probabilities":dict(zip(labels,map(float,p))),
                    "correct":answer==str(task["expected"]),"cross_entropy":float(-target@lp),
                    "brier":float(np.square(p-target).sum()),"tvd":float(abs(p-target).sum()/2),
                    "soft_target":sum(v>0 for v in target)>1,"confidence":float(p[j]),"target_mass":float(target[j]),
                    "temperature":temperature}
                distributions[f"{path}_{scale}"][i]=row
        trace_seconds=r["trace"]["seconds"]
        native_seconds=r["native"]["seconds"]
        times[i]={"partition":task["partition"],"trace_reused":r["trace_reused"],
            "native":native_seconds,"full":trace_seconds+r["readouts"]["full"]["seconds"],
            "masked_final":trace_seconds+r["readouts"]["masked_final"]["seconds"],
            "ordinary":trace_seconds+(archive[i]["old_fallback"]["seconds"] if r["trace_reused"] and archive[i]["old_fallback"] else (r["readouts"]["full"]["seconds"] if not r["trace_reused"] and parse_final(r["trace"]["text"],labels,r["trace"]["hit_cap"]) is None else 0)),
            "generation":trace_seconds,"full_readout":r["readouts"]["full"]["seconds"],
            "masked_readout":r["readouts"]["masked_final"]["seconds"]}
    report={"version":m["version"],"primary":"full_calibrated","calibration":cal,"cohorts":{},"comparisons":{},
        "ordinary_text_accuracy":{},"phase_times":{},"by_tier":{},
        "full_jevbench_composite":None,"price":None,
        "limits":["All evaluation datasets historical.","295 archived generation phases combined with new readout measurements.",
                  "No calibrated-probability guarantee under domain shift.","Source fit has 73 examples; one scalar per path.",
                  "No hidden 303 decisions or matched production price/latency."]}
    cohorts=sorted(set(t["partition"] for t in tasks.values()))
    for cohort in cohorts:
        ids=sorted(i for i,t in tasks.items() if t["partition"]==cohort)
        report["cohorts"][cohort]={arm:summary([v[i] for i in ids]) for arm,v in distributions.items()}
        report["ordinary_text_accuracy"][cohort]={"n":len(ids),"correct":sum(ordinary[i]==str(tasks[i]["expected"]) for i in ids)}
        report["phase_times"][cohort]={path:moments([times[i][path] for i in ids]) for path in ("native","ordinary","full","masked_final","generation","full_readout")}
        if cohort=="calibration_fit":continue
        cp={}
        for left,right in (("full_calibrated","native_calibrated"),("full_calibrated","full_raw"),
                           ("masked_final_calibrated","full_calibrated")):
            cp[left+"__"+right]={metric:paired([float(distributions[left][i][metric])-float(distributions[right][i][metric]) for i in ids],tasks,ids)
                                 for metric in ("correct","cross_entropy","brier")}
        d=[int(distributions["full_calibrated"][i]["correct"])-int(ordinary[i]==str(tasks[i]["expected"])) for i in ids]
        cp["full_vs_ordinary"]={**paired(d,tasks,ids),"repairs":sum(v==1 for v in d),"regressions":sum(v==-1 for v in d),
            "repair_ids":[i for i,v in zip(ids,d) if v==1],"regression_ids":[i for i,v in zip(ids,d) if v==-1]}
        report["comparisons"][cohort]=cp
    for tier in ("easy","standard","hard"):
        ids=[i for i,t in tasks.items() if t.get("_tier")==tier]
        report["by_tier"][tier]={arm:summary([v[i] for i in ids]) for arm,v in distributions.items()}
    # Check native API and official distribution scores. Every vector is genuine.
    sys.path.insert(0,str(root/"reconstruction-inputs/benchmark/vendor"))
    from jevbench.scoring import score_task
    from jevbench.tasks import Task
    official=0
    for arm,rr in distributions.items():
        for i,r in rr.items():
            if tasks[i]["partition"]=="jevbench":
                result=score_task(r["probabilities"],Task.from_dict(tasks[i]))
                if not result["strict_valid"] or result["predicted"]!=r["answer"] or bool(result["correct"])!=r["correct"]:
                    raise ValueError("Official score mismatch")
                official+=1
    audit={"native_rows_exactly_matched":direct_matches,"prior_fallback_readouts_compared":prior_fallbacks,
        "prior_fallback_max_logit_error":max_prior_error,"official_distribution_scores_checked":official,
        "emitted_vectors":sum(len(x) for x in distributions.values()),"new_native_forwards":len(rows),
        "new_readout_forwards":2*len(rows),"new_generations":sum(not r["trace_reused"] for r in rows.values()),
        "new_generated_tokens":sum(r["trace"]["output_tokens"] for r in rows.values() if not r["trace_reused"]),
        "drafts_changed_by_mask":masked_changes,"full_masked_identical_logit_vectors":readout_aliases}
    if official!=1386:raise ValueError("Public score count")
    out=root/"results"
    exp.write(out/"summary.json",report);exp.write(out/"predictions.json",distributions)
    exp.write(out/"ordinary_labels.json",ordinary);exp.write(out/"phase_records.json",times);exp.write(out/"audit.json",audit)
    print(json.dumps({"audit":audit,"cohorts":report["cohorts"],"text":report["ordinary_text_accuracy"]},indent=2))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("fit","score"));p.add_argument("--root",default="data")
    a=p.parse_args()
    if a.mode=="fit":fit(a.root)
    else:score(a.root)

if __name__=="__main__":main()
