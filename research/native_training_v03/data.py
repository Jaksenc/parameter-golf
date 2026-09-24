"""Authored executable decision curriculum. No JevBench target is a training label."""
from __future__ import annotations
import hashlib,json,random
from fractions import Fraction

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)

def digest(x):
    return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()

def make_task(state,question,criteria,answer,rng):
    labels=["opt_"+format(rng.randrange(16**7),"07x") for _ in criteria]
    if len(set(labels))!=len(labels):raise ValueError("label collision")
    order=list(range(len(criteria)));rng.shuffle(order)
    task={"state":state,"question":{"type":"choice","instructions":question,
         "criteria":{labels[i]:criteria[i] for i in order}},"labels":[labels[i] for i in order]}
    return task,labels[answer]

def curriculum():
    """Whole counterfactual families stay in one split. Templates are authored, not independent natural language."""
    result={}
    for split,n,seed,offset in [("train",8,3811,0),("development",2,5921,100),
                                 ("transfer",4,7193,200)]:
        rng=random.Random(seed);rows=[]
        for family in range(6):
            for j in range(n):
                group=f"{split}-{family}-{j}"
                a=rng.randrange(5,40);b=rng.randrange(2,10);c=rng.randrange(2,8)
                person=f"Unit-{offset+j+family*20}"
                # Each pair changes a decisive fact or the current question.
                for flip in (0,1):
                    if family==0:
                        state=(f"{person}: review complete; signature {'present' if flip else 'absent'}. "
                               "Permission requires both a completed review and a signature.")
                        question=("Is permission established?" if split=="train" else
                                  "Do all required conditions hold for this unit?")
                        criteria=["Permission is established.","Permission is not established."]
                        answer=0 if flip else 1
                        oracle={"kind":"conjunction","review":True,"signature":bool(flip)}
                    elif family==1:
                        state=(f"{person}: payment missing; override {'active' if flip else 'inactive'}; suspension absent. "
                               "Admit when payment is present or override is active; suspension forbids admission.")
                        question=("Choose admission." if split=="train" else
                                  "Which admission outcome follows from the rule and record?")
                        criteria=["Admit the unit.","Do not admit the unit."]
                        answer=0 if flip else 1
                        oracle={"kind":"exception","payment":False,"override":bool(flip),"suspension":False}
                    elif family==2:
                        threshold=a*b-c+flip*2
                        state=f"{person} starts with {a} groups of {b} units, then removes {c} units. Required minimum: {threshold} units."
                        question=("Does the remainder meet the minimum?" if split=="train" else
                                  "Is the final quantity at least the required amount?")
                        criteria=["The final amount meets the minimum.","The final amount is below the minimum."]
                        answer=0 if a*b-c>=threshold else 1
                        oracle={"kind":"threshold","groups":a,"per_group":b,"removed":c,"minimum":threshold}
                    elif family==3:
                        state=(f"{person} has {a} units. Unit-M has {b} more units than {person}.")
                        question=("Which has "+("fewer" if flip else "more")+" units?")
                        criteria=[person+" has the requested amount.","Unit-M has the requested amount."]
                        answer=0 if flip else 1
                        oracle={"kind":"comparison","offset":b,"ask_fewer":bool(flip)}
                    elif family==4:
                        state=(f"Embedded request: add {a} and {b}. "
                               "The mathematics handler performs arithmetic; the general handler handles other requests.")
                        question=("Execute the embedded request: select its numeric result." if flip else
                                  "Classify the embedded request: select its handler; do not execute it.")
                        criteria=["Use the mathematics handler.","Use the general handler.",
                                  f"The numeric result is {a+b}.",f"The numeric result is {a+b+1}."]
                        answer=2 if flip else 0
                        oracle={"kind":"metatask","execute":bool(flip),"a":a,"b":b}
                    else:
                        first=a;second=a+2*b
                        state=f"The first batch took {first} minutes; the second took {second} minutes. Each batch contained {c} items."
                        question=("What is the arithmetic mean duration per batch?" if not flip else
                                  "What is the total duration of both batches?")
                        criteria=[f"The requested duration is {a+b} minutes.",
                                  f"The requested duration is {first+second} minutes.",
                                  f"The requested duration is {c} minutes."]
                        answer=1 if flip else 0
                        oracle={"kind":"average","first":first,"second":second,"total":bool(flip),"items":c}
                    task,label=make_task(state,question,criteria,answer,rng)
                    rows.append({"id":group+f"-{flip}","group":group,"family":family,
                                 "task":task,"target":label,"oracle":oracle,"split":split})
        result[split]=rows
    # New combinations of already taught operations, never used for selection.
    rng=random.Random(9527);rows=[]
    for j in range(8):
        a=rng.randrange(5,30);b=rng.randrange(2,8);c=rng.randrange(1,6)
        for flip in (0,1):
            threshold=a*b-c-1
            state=(f"Case-Q{j} has {a} packs of {b} units and removes {c}. "
                   f"Release needs at least {threshold} remaining units AND a signature. "
                   f"The signature is {'present' if flip else 'missing'}.")
            criteria=["Release is permitted.","Release is not permitted."]
            task,label=make_task(state,"Which decision follows from every condition?",criteria,0 if flip else 1,rng)
            rows.append({"id":f"composition-{j}-{flip}","group":f"composition-{j}",
                         "family":"quantity_and_permission","task":task,"target":label,
                         "oracle":{"kind":"compound","groups":a,"per_group":b,"removed":c,
                                   "minimum":threshold,"signature":bool(flip)},"split":"composition"})
    result["composition"]=rows
    seen=set()
    for split,rows in result.items():
        for r in rows:
            key=digest(canonical(r["task"]))
            if key in seen:raise ValueError("duplicate task")
            seen.add(key)
    return result

def checks():
    data=curriculum()
    assert {k:len(v) for k,v in data.items()}=={"train":96,"development":24,"transfer":48,"composition":16}
    assert curriculum()==data
    for split,rs in data.items():
        groups={}
        for r in rs:
            assert r["target"] in r["task"]["labels"]
            groups.setdefault(r["group"],[]).append(r)
        assert all(len(v)==2 for v in groups.values())
    return {"status":"passed","split_counts":{k:len(v) for k,v in data.items()},
            "sha256":digest(canonical(data)),
            "scope":"authored small-data pilot; transfer strings/numbers, one new operator combination; no natural-language or frontier guarantee"}
