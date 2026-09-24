from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from . import cases,prompts
from .protocol import PROTOCOL
from .runtime import doctor,Runtime
from .ledger import Ledger,verify_saved_call
from .engine import score_case,assign
from .report import analyze

ROOT=Path(__file__).resolve().parents[1]
def dump(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text);tmp.replace(p)

def code_hash():
    return cases.digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})

def validate():
    from .oracles import reference
    seen=set();counts={};allrows=[]
    for split in ('development','replication'):
      rows=cases.corpus(split)
      for r in rows:
        public=cases.public(r);prompts.validate_public(public)
        derived=reference(public)
        if derived['answer']!=r['reference']['answer'] or {frozenset(s) for s in derived['sufficient_support']}!={frozenset(s) for s in r['reference']['sufficient_support']}:raise ValueError('Reference mismatch')
        key=cases.digest(public)
        if key in seen:raise ValueError('Exact visible overlap')
        seen.add(key);allrows.append(r)
      for i in range(0,len(rows),3):
        a,b,c=rows[i:i+3]
        if a['source']!=b['source'] or a['source']!=c['source']:raise ValueError('Bad source grouping')
        if a['public']['criterion']!=b['public']['criterion'] or a['public']['criterion']!=c['public']['criterion'] or a['public']['options']!=b['public']['options'] or a['public']['options']!=c['public']['options']:raise ValueError('Changed criterion/options')
        if any(sum(x!=y for x,y in zip(a['public']['records'],r['public']['records']))!=1 for r in (b,c)):raise ValueError('Not a one-record intervention')
        if a['reference']['answer']==b['reference']['answer'] or a['reference']['answer']!=c['reference']['answer']:raise ValueError('Wrong intervention effect')
      counts[split]={'cases':len(rows),'sources':len({r['source'] for r in rows}),'public_corpus_sha256':cases.digest([{'id':r['id'],'public':r['public']} for r in rows])}
    return {'status':'passed','splits':counts,'independently_parsed_references':len(allrows),'exact_visible_overlap':0,'model_calls':0,'scope':'Finite generator specifications; not independently authored language data.'}

def prepare(path):
    result=validate();manifest=cases.export(path)
    dump(Path(path)/'protocol.json',PROTOCOL);dump(Path(path)/'validation.json',result)
    return result|{'files':manifest}

def frozen_check():
    current=validate();saved=json.loads((ROOT/'data/validation.json').read_text())
    if current!=saved or json.loads((ROOT/'data/protocol.json').read_text())!=PROTOCOL:raise ValueError('Frozen dataset or protocol changed')
    import hashlib
    for name,meta in json.loads((ROOT/'data/manifest.json').read_text()).items():
      if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Frozen file hash changed')
    return current

def run(args):
    pre=doctor()
    if pre['status']=='blocked':
      print(json.dumps(pre,indent=2));return 2
    frozen_check();rows=cases.corpus(args.split);assigned=assign(rows,args.nshards)[args.shard]
    backend=Runtime();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    identity={'protocol':PROTOCOL,'source_hash':code_hash(),'public_hash':cases.digest([r['public'] for r in rows]),'backend':backend.meta,'shard':args.shard,'nshards':args.nshards,'split':args.split}
    identity_file=out/'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text())!=identity:raise ValueError('Output belongs to a different run')
    dump(identity_file,identity)
    warmup_path=out/'warmup.json'
    if not warmup_path.exists():
      smoke={'records':[{'id':'RSMOKE','text':'The only recorded parcel color is teal.'}],
             'criterion':'Choose the recorded parcel color.',
             'options':[{'id':'teal','description':'The color is teal.'},{'id':'orange','description':'The color is orange.'}]}
      prompt=prompts.baseline(smoke);labels=[o['id'] for o in smoke['options']]
      first=backend.score(prompt,labels);second=backend.score(prompt,labels)
      error=max(abs(a-b) for a,b in zip(first['logits'],second['logits']))
      if error>1e-4:raise RuntimeError('Same-prompt repeatability preflight failed')
      dump(warmup_path,{'first':first,'second':second,'max_logit_difference':error,'extra_unscored_model_calls':2})
    ledger=Ledger(out/'calls.sqlite',identity)
    record_directory=out/'records';record_directory.mkdir(exist_ok=True)
    errors=0
    try:
      for i,r in enumerate(assigned):
        modes=list(prompts.MODES);modes=modes[i%len(modes):]+modes[:i%len(modes)]
        for mode in modes:
          record_path=record_directory/f'{r["id"]}-{mode}.json'
          if record_path.exists():
            old=json.loads(record_path.read_text())
            if old['status']=='ok' or not args.retry_errors:continue
          try:
            record=score_case(backend,ledger,cases.public(r),mode,args.retry_errors)
          except Exception as e:
            errors+=1
            record={'status':'error','backend_kind':backend.kind,'mode':mode,'public_hash':cases.digest(r['public']),'error':{'type':type(e).__name__,'message':str(e)}}
          record['id']=r['id']
          # Prior attempts remain in the call ledger; recovered mode-level state is explicit.
          if record_path.exists():
            history=out/'record_history';history.mkdir(exist_ok=True)
            record_path.rename(history/f'{record_path.stem}-{time.time_ns()}.json')
          dump(record_path,record)
          print(json.dumps({'case':i+1,'cases':len(assigned),'mode':mode,'status':record['status']}),flush=True)
      dump(out/'receipt.json',{'identity_sha256':cases.digest(identity),'records':len(list(record_directory.glob('*.json'))),'new_errors':errors,'physical_calls':ledger.receipt(),'extra_unscored_warmup_calls':2,'weights_updated':False,'official_score':None})
    finally:ledger.close();backend.close()
    return 1 if errors else 0

def report(args):
    import sqlite3
    frozen_check();rows=cases.corpus(args.split);records=[];source_identity=None
    for directory in sorted(Path(args.root).glob('shard-*')):
      identity=json.loads((directory/'identity.json').read_text())
      if identity['split']!=args.split or identity['protocol']!=PROTOCOL or identity['source_hash']!=code_hash():raise ValueError('Wrong experiment shard or source code changed')
      signature={k:v for k,v in identity.items() if k!='shard'}
      if source_identity is None:source_identity=signature
      elif signature!=source_identity:raise ValueError('Shards differ in sources/model/runtime')
      connection=sqlite3.connect((directory/'calls.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
      try:
        for f in sorted((directory/'records').glob('*.json')):
          r=json.loads(f.read_text())
          if r['status']=='ok':
            verify_saved_call(connection,r['decision_key'],r['decision'])
            if r['generation_key']:verify_saved_call(connection,r['generation_key'],r['generation'])
          records.append(r)
      finally:connection.close()
    result=analyze(records,rows);dump(args.out,result)
    print(json.dumps(result,indent=2));return 0

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate');sub.add_parser('doctor')
    a=sub.add_parser('prepare');a.add_argument('--out',required=True)
    a=sub.add_parser('run');a.add_argument('--split',choices=['development','replication'],required=True);a.add_argument('--shard',type=int,required=True);a.add_argument('--nshards',type=int,default=8);a.add_argument('--out',required=True);a.add_argument('--retry-errors',action='store_true')
    a=sub.add_parser('report');a.add_argument('--split',choices=['development','replication'],required=True);a.add_argument('--root',required=True);a.add_argument('--out',required=True)
    args=p.parse_args()
    if args.command=='doctor':print(json.dumps(doctor(),indent=2));return 0
    if args.command=='validate':print(json.dumps(validate(),indent=2));return 0
    if args.command=='prepare':print(json.dumps(prepare(args.out),indent=2));return 0
    if args.command=='run':
      if not 0<=args.shard<args.nshards:p.error('shard must be within the shard count')
      return run(args)
    return report(args)
if __name__=='__main__':sys.exit(main())
