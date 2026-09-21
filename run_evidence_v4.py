"""Pre-outcome static-audit repair for Evidence v4.
1. Recognize numbers followed by sentence punctuation.
2. State generated access rules as sufficient AND necessary (iff).
No model output or benchmark outcome was inspected to choose these repairs.
"""
import re
from pathlib import Path
import evidence_v4 as experiment

original_sha = experiment.sha
original_fresh = experiment.fresh
experiment.NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w)')
experiment.VERSION = 'evidence-v4-1.1-static-audit'

def fresh(seed=experiment.SEED,count=64):
    rows=original_fresh(seed,count)
    for r in rows:
        if r['family']=='threshold':
            r['state']=r['state'].replace('Access is allowed only when', 'Access is allowed if and only if')
    return rows

def code_sha(path):
    if Path(path).resolve() == Path(experiment.__file__).resolve():
        return experiment.digest({'base_sha256':original_sha(path),'repair_sha256':original_sha(__file__)})
    return original_sha(path)

experiment.fresh=fresh
experiment.sha=code_sha
if __name__=='__main__':experiment.main()
