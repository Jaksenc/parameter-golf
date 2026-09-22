"""Research CLI: a measured logit vector on every successful answer path."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, resource, sys, time
from pathlib import Path
import probability_v13 as exp
from probability_boundary import Calibration, render, validate

def decide(rt,payload,path="full",calibration=None):
    validate(payload)
    row=exp.request(payload)
    if path not in ("native","full","masked_final"):raise ValueError("Unknown path")
    fitted=None
    if calibration is not None:
        if calibration["scientific_source_sha256"]!=exp.filehash(exp.__file__):
            raise ValueError("Calibrator belongs to a different inference source")
        if calibration["model_revision"]!="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
            raise ValueError("Calibrator model mismatch")
        fitted=calibration["paths"][path]
    c=Calibration(fitted["temperature"],fitted["fit_record_hash"]) if fitted else Calibration()
    start=time.perf_counter()
    trace=None
    if path=="native":
        obs,_=rt.score(row)
        observation_hash=obs["prompt_hash"]
    else:
        trace=exp.old.generate(rt,exp.old.e4.messages(row,"reason"),exp.CAP)
        text=trace["text"] if path=="full" else exp.masked(trace["text"])
        obs=exp.old.h5.readout(rt,row,text)
        observation_hash=obs["prompt_sha256"]
    answer=render(row,obs["logits"],c,observation_hash=observation_hash)
    answer.update({"id":row["id"],"path":path,"calibration_fitted":fitted is not None,
        "new_weight_training":False,"model":"Qwen/Qwen3.5-4B",
        "model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "elapsed_seconds":time.perf_counter()-start,
        "generated_tokens":trace["output_tokens"] if trace else 0,
        "trace_sha256":hashlib.sha256(trace["text"].encode()).hexdigest() if trace else None})
    # Return raw receipt for independent fixture validation, not a fabricated confidence.
    return {"answer":answer,"raw_observation":obs,"trace":trace}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input");p.add_argument("--path",choices=("native","full","masked_final"),default="full")
    p.add_argument("--data",default="data");p.add_argument("--uncalibrated",action="store_true")
    a=p.parse_args();payload=json.loads(Path(a.input).read_text());validate(payload)
    data=Path(a.data)
    cal=None if a.uncalibrated else json.loads((data/"CALIBRATION.json").read_text())
    from reconstruct_v1 import Runtime
    with contextlib.redirect_stdout(sys.stderr):
        rt=Runtime(data/"reconstruction-inputs")
        result=decide(rt,payload,a.path,cal)
    print(json.dumps(result["answer"],ensure_ascii=False,allow_nan=False))
if __name__=="__main__":main()
