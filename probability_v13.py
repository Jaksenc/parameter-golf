"""Probability v13: real post-reasoning distributions, without weight training.
The primary is the unchanged v12 full-draft readout. Masked FINAL is diagnostic.
No answer keys are passed to neural inference. Each completed item is fsynced.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re
from pathlib import Path
import contrast_v7 as old

VERSION = "probability-v13.0"
SHARDS = 24
CAP = 480
SEED = 13260922
PRIMARY = "full"
READOUTS = ("full", "masked_final")
TEMPERATURES = [2.0 ** (i / 32) for i in range(-64, 97)]

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def filehash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(x, sort_keys=True, indent=2, allow_nan=False))
    temporary.replace(p)

def request(x):
    return old.canonical({k: x[k] for k in ("id", "state", "question", "labels")})

def masked(text):
    """Only explicit FINAL lines are removed; earlier answer cues remain."""
    return re.sub(r"(?m)^[ \t]*FINAL:[^\r\n]*(?:\r?\n|$)", "", text)

def target(x):
    labels = x["labels"]
    p = x.get("target_probs")
    if p is None:
        g = x.get("provenance", {}).get("gold_probs")
        p = [g.get(k, 0.0) for k in labels] if g else [float(k == str(x["expected"])) for k in labels]
    if len(p) != len(labels) or any(not math.isfinite(v) or v < 0 for v in p):
        raise ValueError("Invalid target")
    if abs(sum(p)-1) > 1e-8:
        raise ValueError("Target must sum to one")
    return list(p)

def prepare(v7, v1, out):
    v7, v1, out = map(Path, (v7, v1, out))
    prior = json.loads((v7/"all_contrast_records.json").read_text())
    public = json.loads((v7/"contrast-prepared/evaluation.json").read_text())
    idx = {r["id"]: r for r in prior}
    source_rows = json.loads((v1/"reconstruction-inputs/training.json").read_text())
    keep = set(json.loads((v1/"results/split_correction.json").read_text())["retained_ids"])
    source_rows = [r for r in source_rows if r["id"] in keep and r["partition"] in ("development", "calibration")]
    if collections.Counter(r["partition"] for r in source_rows) != {"development":73, "calibration":73}:
        raise ValueError("Source split changed")
    if len(public) != len(idx) or len(idx) != 295:
        raise ValueError("Prior population incomplete")
    source_native = {}
    for p in (v1/"features").glob("shard-*/records.json"):
        for r in json.loads(p.read_text()):
            if not r["reverse"]:
                source_native[r["id"]] = r
    tasks, jobs, archive = [], [], {}
    for raw in public + source_rows:
        inp = request(raw)
        is_old = raw["id"] in idx
        cohort = raw["partition"] if is_old else (
            "calibration_fit" if raw["partition"] == "development" else "source_check")
        t = {**raw, "partition":cohort, "target":target(raw), "input_hash":digest(inp)}
        tasks.append(t)
        if is_old:
            previous = idx[raw["id"]]
            if previous["input_sha256"] != digest(inp):
                raise ValueError("Archived input differs")
            trace = previous["long_trace"]
            native = previous["native"]
            archive[raw["id"]] = {
                "native":native, "ordinary_answer":previous["predictions"]["long480"],
                "old_fallback":previous["readouts"]["long"], "trace":trace}
        else:
            trace = None
            native = source_native[raw["id"]]
            if native["input_hash"] != digest(inp):
                raise ValueError("Source native input differs")
            archive[raw["id"]] = {"native":native}
        jobs.append({"input":inp, "trace":trace})
    keys = [digest({k:r[k] for k in ("state","question","labels")}) for r in tasks]
    if len(keys) != 441 or len(set(keys)) != 441:
        raise ValueError("Canonical overlap or wrong population")
    bins = [[] for _ in range(SHARDS)]
    loads = [0.0]*SHARDS
    def cost(j):
        # A priori work estimate, never based on correctness.
        return len(json.dumps(j["input"]))**1.25 + (22000 if j["trace"] is None else 2500)
    for job in sorted(jobs, key=lambda j:(-cost(j),j["input"]["id"])):
        n = min(range(SHARDS), key=lambda n:(loads[n],n))
        bins[n].append(job)
        loads[n] += cost(job)
    anchors = [{"input":request(t), "native":archive[t["id"]]["native"]}
               for t in sorted(tasks, key=lambda r:r["id"])[:SHARDS]]
    out.mkdir(parents=True, exist_ok=False)
    write(out/"evaluation.json", tasks)
    write(out/"jobs.json", bins)
    write(out/"archive.json", archive)
    write(out/"anchors.json", anchors)
    manifest = {"version":VERSION, "primary":PRIMARY, "readouts":READOUTS,
        "source_sha256":filehash(__file__), "jobs_hash":digest(bins),
        "evaluation_hash":digest(tasks), "archive_hash":digest(archive),
        "counts":dict(collections.Counter(t["partition"] for t in tasks)),
        "shards":list(map(len,bins)), "new_generations":146,
        "new_native_forwards":441, "planned_readout_forwards":882,
        "temperature_grid":TEMPERATURES,
        "calibration":"Each path fits one temperature on 73 old development inputs only.",
        "scope":"Historical data, new logits. All-source check and target sets excluded from fitting.",
        "training":"No pretrained-weight updates. No public outcome selects a method.",
        "controls":"Masked FINAL is diagnostic, not a replacement for full primary.",
        "model":old.__dict__.get("MODEL", "Qwen/Qwen3.5-4B"),
        "model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
    write(out/"manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest

def solve(rt, payload, trace=None):
    inp = request(payload)
    native, _ = rt.score(inp)
    archived = trace is not None
    if trace is None:
        trace = old.generate(rt, old.e4.messages(inp, "reason"), CAP)
    if not isinstance(trace["text"], str) or trace["output_tokens"] > CAP:
        raise ValueError("Invalid draft")
    results = {}
    order = list(READOUTS)
    random.Random(int(digest(inp["id"])[:8],16)).shuffle(order)
    for mode in order:
        text = trace["text"] if mode == "full" else masked(trace["text"])
        # Deliberately unconditional; FINAL never bypasses the probability readout.
        observation = old.h5.readout(rt, inp, text)
        if len(observation["logits"]) != len(inp["labels"]):
            raise ValueError("Missing probability output")
        if any(not math.isfinite(v) for v in observation["logits"]):
            raise ValueError("Nonfinite logit")
        results[mode] = observation
    return {"id":inp["id"], "input_hash":digest(inp), "native":native,
        "trace":trace, "trace_reused":archived, "readouts":results,
        "readout_order":order, "masked_text_hash":hashlib.sha256(masked(trace["text"]).encode()).hexdigest()}

def run(root, prepared, out, shard):
    from reconstruct_v1 import Runtime
    root, prepared, out = map(Path, (root, prepared, out))
    m = json.loads((prepared/"manifest.json").read_text())
    jobs = json.loads((prepared/"jobs.json").read_text())
    if shard not in range(SHARDS) or filehash(__file__) != m["source_sha256"] or digest(jobs) != m["jobs_hash"]:
        raise ValueError("Frozen work changed")
    out.mkdir(parents=True, exist_ok=False)
    rt = Runtime(root/"reconstruction-inputs")
    fixture = rt.check()
    anchor = json.loads((prepared/"anchors.json").read_text())[shard]
    obs, _ = rt.score(anchor["input"])
    ref = anchor["native"]
    err = max(abs(a-b) for a,b in zip(obs["logits"],ref["logits"]))
    if err > 1e-4 or obs["prompt_hash"] != ref["prompt_hash"]:
        raise ValueError("Native anchor mismatch")
    write(out/"preflight.json", {"runtime":rt.receipt, "actually_generative":True,
        "fixture":fixture, "anchor_id":anchor["input"]["id"], "anchor_error":err,
        "source_sha256":m["source_sha256"]})
    path = out/"records.jsonl"
    path.write_text("")
    ids = []
    for job in jobs[shard]:
        row = solve(rt, job["input"], job["trace"])
        row["shard"] = shard
        with path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
            os.fsync(f.fileno())
        ids.append(row["id"])
        print(json.dumps({"shard":shard,"done":len(ids),"planned":len(jobs[shard])}), flush=True)
    write(out/"complete.json", {"ids":ids, "count":len(ids),
        "source_sha256":m["source_sha256"], "records_sha256":filehash(path),
        "jobs_hash":digest(jobs[shard])})

def aggregate(prepared, records, out):
    prepared, records, out = map(Path, (prepared, records, out))
    m = json.loads((prepared/"manifest.json").read_text())
    jobs = json.loads((prepared/"jobs.json").read_text())
    rows, seen = [], set()
    for i, plan in enumerate(jobs):
        d = records/f"probability-v13-shard-{i}"
        c = json.loads((d/"complete.json").read_text())
        p = d/"records.jsonl"
        rr = [json.loads(s) for s in p.read_text().splitlines()]
        wanted = {j["input"]["id"]:j for j in plan}
        if c["source_sha256"] != m["source_sha256"] or c["records_sha256"] != filehash(p) or c["jobs_hash"] != digest(plan):
            raise ValueError("Shard provenance")
        if len(rr) != len(wanted) or c["count"] != len(rr) or set(c["ids"]) != set(wanted):
            raise ValueError("Incomplete shard")
        for r in rr:
            if r["id"] in seen or r["id"] not in wanted or r["input_hash"] != digest(wanted[r["id"]]["input"]):
                raise ValueError("Unexpected row")
            if set(r["readouts"]) != set(READOUTS):
                raise ValueError("Incomplete paired readout")
            seen.add(r["id"])
            rows.append(r)
    if len(rows) != 441:
        raise ValueError("Incomplete experiment")
    write(out/"all_records.json", sorted(rows,key=lambda r:r["id"]))
    write(out/"completion.json", {"complete":True, "inputs":len(rows),
        "new_generations":sum(not r["trace_reused"] for r in rows),
        "new_readouts":sum(len(r["readouts"]) for r in rows),
        "source_sha256":m["source_sha256"]})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare","run","aggregate"))
    p.add_argument("--v7", default="parent-v7")
    p.add_argument("--v1", default="parent-v1")
    p.add_argument("--root", default=".")
    p.add_argument("--prepared", default="probability-prepared")
    p.add_argument("--records", default="records")
    p.add_argument("--out", default=".")
    p.add_argument("--shard", type=int, default=0)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.v7,a.v1,a.prepared)
    elif a.mode == "run":
        run(a.root,a.prepared,a.out,a.shard)
    else:
        aggregate(a.prepared,a.records,a.out)

if __name__ == "__main__":
    main()
