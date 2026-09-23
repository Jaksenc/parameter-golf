"""Same-protocol recovery of missing records only. No metric-based reselection."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import transfer as t

INITIAL_HASHES={3: '6d49fab0a6529e9ce05bcbdb25f4db21e5ac27fc1c6f729ca10b868b6d986fbf', 2: '11e26edf58030f4b30be86fb12b640d9188dbb0f8477b8915f6c93409b7d6843', 1: 'f765e0fc12de49ec75d4582f393e2f13c1529e35fc6e51931a492083e53e6872', 0: '157fb70237593f63f88b75bca78a385f5f36777f8ef2fb39bee6afa495164c99', 5: '11b41647b49fde6ea679b2eae2450c68e075c431fc380aa7e30ee4177f15f0a2', 4: 'e3e5476e81d9b4d34b17c1f8b0ade67dec91d23b90491183f32d696df4fc9857', 7: '0c85892287867a824e8a90b182f6dca37f07933530783d853039949ab428d5d2', 6: 'cf52a08f7a56aa3db1d44910a0f9796fce0fdb1e458748ade6dc2bfe05ccc1e0'}

def initial_checks(root):
 paths=list(Path(root).rglob("records.jsonl"))
 if len(paths)!=8:raise ValueError("Expected all eight original shards")
 runtime=None
 for f in paths:
  a=json.loads((f.parent/"assignment.json").read_text()); i=a["shard"]
  if t.filehash(f)!=INITIAL_HASHES[i]:raise ValueError("Initial record bytes changed")
  meta=json.loads((f.parent/"runtime.json").read_text())
  relevant={k:meta[k] for k in ("model","revision","hashes","machine","threads","torch","transformers","parameters","training")}
  if runtime is None:runtime=relevant
  elif runtime!=relevant:raise ValueError("Original runtime mismatch")
 return runtime

def prior(root):
 found={};sources=[]
 expected={(r['id'],v) for r in t.corpus() for v in t.MODES}
 for p in sorted(Path(root).rglob('records.jsonl')):
  proto=json.loads((p.parent/'protocol.json').read_text());assignment=json.loads((p.parent/'assignment.json').read_text())
  if t.digest(proto)!=t.digest(t.PROTOCOL) or assignment['corpus_hash']!=t.digest(t.corpus()):raise ValueError('Cannot mix protocols')
  lines=p.read_text().splitlines()
  for i,l in enumerate(lines):
   try:r=json.loads(l)
   except json.JSONDecodeError:
    if i!=len(lines)-1:raise
    sources.append({'file':str(p),'ignored_incomplete_final_line':True});continue
   key=(r['id'],r['mode'])
   if key not in expected or key in found:raise ValueError('Duplicate or foreign record')
   if r.get('ok'):found[key]=r
  sources.append({'file':str(p),'sha256':t.filehash(p)})
 return found,sources

def run(args):
 expected_runtime=initial_checks(args.prior_root)
 found,sources=prior(args.prior_root);rows=t.corpus();rowmap={r['id']:r for r in rows}
 # Whole case grouped whenever any of its modes is missing. Order is fixed by input size.
 missing_rows=[r for r in rows if any((r['id'],v) not in found for v in t.MODES)]
 ids=t.assignment(missing_rows,args.nshards)[args.shard]
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);t.save(out/'protocol.json',t.PROTOCOL);t.save(out/'prior_sources.json',sources)
 t.save(out/'assignment.json',{'ids':[missing_rows[i]['id'] for i in ids],'corpus_hash':t.digest(rows),'phase':'dev','modes':list(t.MODES),'nshards':args.nshards,'shard':args.shard,'recovery':True})
 rt=t.Runtime();t.save(out/'runtime.json',rt.meta)
 if {k:rt.meta[k] for k in expected_runtime}!=expected_runtime:raise ValueError("Recovery runtime mismatch")
 # Check a saved deterministic reference before accepting mixed-run outputs.
 reference=next((r for r in found.values() if r['mode']=='baseline'),None)
 if reference:
  replay=rt.score(rowmap[reference['id']],'baseline');error=max(abs(x-y) for x,y in zip(replay['logits'],reference['logits']))
  t.save(out/'baseline_replay.json',{'id':reference['id'],'max_logit_difference':error,'fresh_record':replay})
  if error>1e-4:raise ValueError('Runtime drift in recovery')
 n=0
 with (out/'records.jsonl').open('w') as f:
  for i in ids:
   r=missing_rows[i]
   for v in t.MODES:
    if (r['id'],v) in found:continue
    rec=t.score(rt,r,v);rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit','length_stratum')});rec['correct']=rec['predicted']==r['expected'];rec['phase']='dev';rec['recovery_run']=True
    f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();n+=1
    print(json.dumps({'phase':'same-protocol recovery','shard':args.shard,'done':n,'id':r['id'],'mode':v}),flush=True)
 t.save(out/'receipt.json',{'records':n,'errors':0,'records_sha256':t.filehash(out/'records.jsonl'),'protocol_hash':t.digest(t.PROTOCOL),'weights_updated':False,'recovery':True})

def combine(args):
 found,sources=prior(args.prior_root);new,nsources=prior(args.recovery_root)
 if set(found)&set(new):raise ValueError('Duplicate old/new inference')
 found.update(new);sources+=nsources;rows=t.corpus()
 if set(found)!={(r['id'],m) for r in rows for m in t.MODES}:raise ValueError('Recovery incomplete')
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 records=sorted(found.values(),key=lambda r:(r['id'],r['mode']))
 (out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
 t.save(out/'protocol.json',t.PROTOCOL);t.save(out/'assignment.json',{'phase':'dev','corpus_hash':t.digest(rows),'reconstructed_from_partial_and_recovery':True});t.save(out/'receipt.json',{'records':len(records),'errors':0,'records_sha256':t.filehash(out/'records.jsonl'),'protocol_hash':t.digest(t.PROTOCOL)});t.save(out/'sources.json',sources)
if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('run');a.add_argument('--shard',type=int,required=True);a.add_argument('--nshards',type=int,default=8);a.add_argument('--prior-root',required=True);a.add_argument('--out',required=True)
 a=sub.add_parser('combine');a.add_argument('--prior-root',required=True);a.add_argument('--recovery-root',required=True);a.add_argument('--out',required=True)
 a=ap.parse_args();run(a) if a.cmd=='run' else combine(a)
