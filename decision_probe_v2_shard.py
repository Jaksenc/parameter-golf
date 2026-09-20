"""Execution-only partition of the frozen 400-case probe; no prompt/model tuning.
The original AMD job timed out after 27 calibration cases. Do not use those
partial cases as an accuracy benchmark. A full four-shard union is required.
"""
import ast, hashlib, json, os, sys
from pathlib import Path

index=int(os.environ['PROBE_SHARD_INDEX']); count=4
if index not in range(count):raise ValueError('Invalid shard')
source=Path('decision_probe_v2.py').read_text()
expected='b7267efe6fb8431b303490bafba0986bf899bb21c3b4677f90d36ff1de4695ef'
if hashlib.sha256(source.encode()).hexdigest()!=expected:raise RuntimeError('Frozen inference source changed')
needle='      for i,r in enumerate(rows):\n'
if source.count(needle)!=1:raise RuntimeError('Unexpected inference loop')
patched=source.replace(needle,needle+f'        if i % {count} != {index}: continue\n')
patched=patched.replace("emit('probe_summary',model=", "emit('shard_summary',model=")
# Prove that both nested native model score functions are unchanged.
def scorers(text):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='score']
if scorers(source)!=scorers(patched):raise RuntimeError('Scoring implementation changed')
print(json.dumps({'kind':'shard_execution','shard_index':index,'shard_count':count,
 'original_code_sha256':expected,'patched_code_sha256':hashlib.sha256(patched.encode()).hexdigest(),
 'partition':'original row index modulo four; no label-dependent selection',
 'native_score_functions_ast_unchanged':True,'original_failed_job':106099011100}),flush=True)
sys.argv=['decision_probe_v2.py','--model','qwen-4b']
exec(compile(patched,'decision_probe_v2.py','exec'),{'__name__':'__main__','__file__':str(Path('decision_probe_v2.py').resolve())})
