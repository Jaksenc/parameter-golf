"""Execution-only patch for the frozen external v4 evaluator.
No task, prompt, logits, checkpoint selection, or scoring-rule change.
Retains original source and records exact applied code hash before execution.
"""
from __future__ import annotations
import hashlib,json,runpy
from pathlib import Path

p=Path(__file__).with_name('jevbench_v4.py')
text=p.read_text()
original='2cc200a6f1b04446954997624478020d821a858a391aa3d759624a1bc77c5833'
if hashlib.sha256(text.encode()).hexdigest()!=original:raise RuntimeError('Frozen external evaluator changed')
if text.count('norm.variance_epsilon')!=2 or text.count('TRAINING_RUN=35545041177')!=1:raise RuntimeError('Unexpected patch boundary')
text=text.replace('norm.variance_epsilon','norm.eps').replace('TRAINING_RUN=35545041177','TRAINING_RUN=35545335396')
out=p.with_name('_jevbench_v4_applied.py')
if out.exists() and out.read_text()!=text:raise RuntimeError('Conflicting applied evaluator')
out.write_text(text)
print(json.dumps({'kind':'execution_patch','original_sha256':original,'applied_sha256':hashlib.sha256(text.encode()).hexdigest(),'changes':['Use upstream Qwen3_5RMSNorm.eps rather than nonexistent variance_epsilon','Read the corrected training run 35545335396, not failed preflight run 35545041177'],'data_or_objective_change':False}),flush=True)
runpy.run_path(str(out),run_name='__main__')
