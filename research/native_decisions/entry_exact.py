"""Exact native-logit entrypoint. First readiness attempt produced zero benchmark outputs.
Only the probability-access backend changes; data, prompts, selection, and scoring stay fixed.
"""
import argparse,json,os,sys,zipfile
from pathlib import Path
import readout,exact_backend
import experiment as exp
readout.Backend=exact_backend.RawBackend
original_setup=exp.boot.setup
original_write=exp.write
HERE=Path(__file__).resolve().parent

def setup(pin):
 proc=log=None
 try:
  proc,log,info=original_setup(pin);info['native_build']=exact_backend.build(exp.boot)
  return proc,log,info
 except Exception:
  exp.stop(proc,log);raise

def write(path,obj):
 if isinstance(obj,dict):
  obj['backend_revision']='direct C API logits; no probability sampling or truncation'
  obj['source_files']={p.name:readout.digest(p.read_bytes()) for p in HERE.iterdir() if p.suffix in ('.py','.cpp')}
  if 'protocol' in obj:
   obj['protocol'].pop('native_forward_tokens',None)
   obj['protocol'].pop('no_token_filtering_except_option_grammar',None)
   obj['protocol'].update(native_generated_tokens=0,native_computation='prompt prefill; read all requested raw logits; subset softmax in float64',native_sampler=None)
 return original_write(path,obj)

def bundle():
 with zipfile.ZipFile('native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(HERE.iterdir()):
   if p.suffix in ('.py','.cpp'):z.write(p,'native_decisions/'+p.name)
  for name in ('bootstrap.py','preflight_entry.py'):
   p=HERE.parent/'adaptive_language'/name;z.write(p,'bootstrap/'+name)
  for p in sorted((exp.boot.ROOT/'native-headers').glob('*')):z.write(p,'native-build/include/'+p.name)
  if (exp.boot.ROOT/'native-build.json').exists():z.write(exp.boot.ROOT/'native-build.json','native-build/BUILD.json')

exp.boot.setup=setup;exp.write=write;exp.bundle=bundle;exp.__file__=str(Path(__file__).resolve())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--journal');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.preflight:exp.preflight()
 elif a.worker:exp.worker(a.input,a.journal)
 else:
  if not 0<=a.shard<exp.SHARDS:raise ValueError('shard')
  exp.run(a.shard)
