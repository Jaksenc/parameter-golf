"""Probability-carrying boundary. No invented one-hot outputs or hidden fallbacks."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    fit_split_hash: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")

def probabilities(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    Calibration(temperature)
    if len(logits) < 2 or any(not math.isfinite(z) for z in logits):
        raise ValueError("At least two finite logits required")
    top=max(logits)
    weights=[math.exp((z-top)/temperature) for z in logits]
    s=sum(weights)
    return [w/s for w in weights]

def log_probabilities(logits, temperature=1.0):
    Calibration(temperature)
    if len(logits)<2 or any(not math.isfinite(z) for z in logits):
        raise ValueError("Finite vector required")
    top=max(logits)
    zs=[(z-top)/temperature for z in logits]
    if any(not math.isfinite(z) for z in zs):
        raise ValueError("Logit range exceeds finite log-probability arithmetic")
    a=math.log(sum(math.exp(z) for z in zs))
    return [z-a for z in zs]

def validate(payload):
    if not isinstance(payload,dict) or set(payload)-{"id","state","question","labels"}:
        raise ValueError("Only request fields; targets are not allowed")
    if not {"state","question","labels"} <= set(payload):
        raise ValueError("Missing request fields")
    labels=payload["labels"]; q=payload["question"]
    if not isinstance(labels,list) or not 2<=len(labels)<=16 or any(
        not isinstance(l,str) or not l or "\n" in l or "\r" in l for l in labels):
        raise ValueError("Require 2–16 unique single-line string labels")
    if len(set(labels))!=len(labels):
        raise ValueError("Duplicate labels")
    if not isinstance(q,dict) or q.get("type") not in ("choice","noul","score"):
        raise ValueError("Unsupported primitive")
    if not isinstance(q.get("instructions"),str) or not isinstance(q.get("criteria"),(dict,list)):
        raise ValueError("Missing instructions/criteria")
    if q["type"]=="score" and (len(labels)>10 or labels!=[str(i) for i in range(len(labels))]):
        raise ValueError("Score levels must be 0 through K-1, K<=10")
    if q["type"]=="noul" and {l.lower() for l in labels} not in ({"no","yes"},{"false","true"}):
        raise ValueError("Noul requires an explicit positive/negative pair")
    return payload

def render(payload, logits, calibration=Calibration(), *, observation_hash=None):
    r=validate(payload); labels=r["labels"]
    if len(logits)!=len(labels):
        raise ValueError("Incomplete vector")
    p=probabilities(logits,calibration.temperature)
    j=min(range(len(p)),key=lambda i:(-p[i],labels[i]))
    kind=r["question"]["type"]
    out={"answer":labels[j],"probabilities":dict(zip(labels,p)),
         "temperature":calibration.temperature,"calibration_fit_split_hash":calibration.fit_split_hash,
         "observation_hash":observation_hash,"semantic_correctness_verified":False,
         "distribution_status":"measured model logits; calibration is not a per-answer guarantee"}
    if kind=="score":
        out["expected_level"]=sum(i*v for i,v in enumerate(p))
    elif kind=="noul":
        pos=next(i for i,l in enumerate(labels) if l.lower() in ("yes","true"))
        out["value"]=p[pos]
    else:
        out["distribution_concentration"]=(len(p)*max(p)-1)/(len(p)-1)
    return out

def fit_temperature(rows, temperatures):
    """Only explicit calibration records supplied by the caller enter fitting."""
    if not rows:
        raise ValueError("No calibration records")
    if any(r.get("partition")!="calibration_fit" for r in rows):
        raise ValueError("Evaluation data rejected from calibrator")
    losses=[]
    for t in temperatures:
        terms=[]
        for r in rows:
            y=r["target"]; z=r["logits"]
            if len(y)!=len(z) or any(v<0 or not math.isfinite(v) for v in y) or abs(sum(y)-1)>1e-8:
                raise ValueError("Invalid calibration distribution")
            terms.append(-sum(a*b for a,b in zip(y,log_probabilities(z,t))))
        losses.append(sum(terms)/len(terms))
    index=min(range(len(losses)),key=lambda i:(losses[i],abs(math.log(temperatures[i]))))
    return {"temperature":temperatures[index],"fit_log_loss":losses[index],
            "identity_log_loss":losses[temperatures.index(1.0)] if 1.0 in temperatures else None,
            "on_grid_boundary":index in (0,len(losses)-1),"loss_curve":losses}
