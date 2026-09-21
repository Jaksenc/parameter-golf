"""Canonical request serialization matching the frozen benchmark input format.
A release-interface correction, not a change to the benchmark experiment.
"""
from __future__ import annotations
import argparse,contextlib,json,sys
from pathlib import Path
import decide_resume as original


def canonical_payload(payload):
    validated=original.validate(payload)
    return json.loads(json.dumps(validated,sort_keys=True,ensure_ascii=False,allow_nan=False))


def decide(runtime,payload):
    return original.decide(runtime,canonical_payload(payload))


def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--runtime-root',default='data');a=p.parse_args()
    root=Path(a.runtime_root).resolve();sys.path.insert(0,str(root));payload=json.loads(Path(a.input).read_text());canonical_payload(payload)
    from reconstruct_v1 import Runtime
    with contextlib.redirect_stdout(sys.stderr):
        rt=Runtime(root/'reconstruction-inputs');result=decide(rt,payload)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
