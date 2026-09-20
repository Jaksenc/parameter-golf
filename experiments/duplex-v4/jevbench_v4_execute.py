"""Execution-only binding for the frozen public benchmark experiment.
Use the upstream Qwen RMSNorm attribute and the current completed training run.
No task, prompt, loss, scoring, or selected checkpoint content changes.
"""
from __future__ import annotations
import hashlib,json,os,runpy
from pathlib import Path

source=Path(__file__).with_name('jevbench_v4.py')
text=source.read_text()
original='2cc200a6f1b04446954997624478020d821a858a391aa3d759624a1bc77c5833'
if hashlib.sha256(text.encode()).hexdigest()!=original:raise RuntimeError('Frozen benchmark source changed')
run_id=int(os.environ['V4_TRAINING_RUN'])
if run_id<=0 or text.count('norm.variance_epsilon')!=2 or text.count('TRAINING_RUN=35545041177')!=1:raise RuntimeError('Invalid execution binding')
text=text.replace('norm.variance_epsilon','norm.eps').replace('TRAINING_RUN=35545041177',f'TRAINING_RUN={run_id}')
path=source.with_name('_jevbench_v4_bound.py')
if path.exists() and path.read_text()!=text:raise RuntimeError('Conflicting bound script')
path.write_text(text)
print(json.dumps({'kind':'execution_binding','source_sha256':original,'applied_sha256':hashlib.sha256(text.encode()).hexdigest(),'training_run':run_id,'epsilon_attribute':'Qwen3_5RMSNorm.eps','benchmark_tuning':False}),flush=True)
runpy.run_path(str(path),run_name='__main__')
