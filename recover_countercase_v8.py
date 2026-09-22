"""Availability-only v8 recovery; the scientific model and prompts are unchanged."""
from __future__ import annotations
import argparse,json,os,random
from pathlib import Path
import countercase_v8 as c
SOURCE='e50085c81381f226a632e06efbb7d740f276acd013a3931e0c18343d15a55ffd'

def read_json(p):return json.loads(Path(p).read_text())
def check_row(row,job):
 if row['id']!=job['input']['id'] or row['input_sha256']!=c.prior.h5.digest(job['input']):raise ValueError('Input changed')
 if set(row['outputs'])!=set(job['arms']) or sorted(row['execution_order'])!=sorted(job['arms']):raise ValueError('Missing arm')
 for arm,out in row['outputs'].items():
  if out['arm']!=arm or out['input_sha256']!=row['input_sha256'] or out['answer'] not in job['input']['labels']:raise ValueError('Invalid branch')

def prepare(root):
 m=read_json(root/'countercase-prepared/manifest.json');old=read_json(root/'countercase-prepared/jobs.json')
 if c.prior.h5.filehash(c.__file__)!=SOURCE or m['source_sha256']!=SOURCE or c.prior.h5.digest(old)!=m['jobs_hash']:raise ValueError('Frozen source changed')
 available={};audit=[]
 for i,plan in enumerate(old):
  directory=root/'original-records'/f'countercase-v8-shard-{i}';p=directory/'records.jsonl';wanted={j['input']['id']:j for j in plan};count=0;trailing=0
  if p.exists():
   if read_json(directory/'preflight.json')['source_sha256']!=SOURCE:raise ValueError('Original source mismatch')
   for line in p.read_bytes().splitlines(keepends=True):
    if not line.endswith(b'\n'):trailing+=len(line);continue
    row=json.loads(line);rid=row['id']
    if rid not in wanted or rid in available:raise ValueError('Unexpected or duplicated original row')
    check_row(row,wanted[rid]);available[rid]=row;count+=1
   if (directory/'complete.json').exists():
    complete=read_json(directory/'complete.json')
    if complete['count']!=len(plan) or count!=len(plan) or complete['records_sha256']!=c.prior.h5.filehash(p):raise ValueError('Bad original completion receipt')
  audit.append({'shard':i,'completed_rows':count,'planned':len(plan),'trailing_incomplete_bytes':trailing,'file_sha256':c.prior.h5.filehash(p) if p.exists() else None})
 wanted={j['input']['id']:j for group in old for j in group}
 missing=[wanted[k] for k in sorted(set(wanted)-set(available))]
 # At most two inputs per recovery worker. All original successes are immutable.
 groups=[missing[i:i+2] for i in range(0,len(missing),2)] or [[]]
 saved=[available[k] for k in sorted(available)]
 out=root/'recovery-prepared';out.mkdir(exist_ok=True)
 c.prior.h5.write(out/'original.json',saved);c.prior.h5.write(out/'jobs.json',groups)
 receipt={'source_sha256':SOURCE,'original_count':len(saved),'missing_count':len(missing),'original_sha256':c.prior.h5.digest(saved),'jobs_hash':c.prior.h5.digest(groups),'matrix':list(range(len(groups))),'selection':'Missing complete records only; no correctness-based selection','original_shards':audit}
 c.prior.h5.write(out/'receipt.json',receipt)
 if os.environ.get('GITHUB_OUTPUT'):
  with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('matrix='+json.dumps(receipt['matrix'])+'\n')
 print(json.dumps({'preserved':len(saved),'missing':len(missing),'workers':len(groups)}),flush=True)

def run(root,out,shard):
 groups=read_json(root/'recovery-prepared/jobs.json');receipt=read_json(root/'recovery-prepared/receipt.json')
 if c.prior.h5.filehash(c.__file__)!=SOURCE or c.prior.h5.digest(groups)!=receipt['jobs_hash']:raise ValueError('Recovery input changed')
 if shard not in range(len(groups)):raise ValueError('Unknown worker')
 out.mkdir(parents=True,exist_ok=False);p=out/'records.jsonl';p.write_text('');ids=[]
 if groups[shard]:
  from reconstruct_v1 import Runtime
  rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
  anchor=read_json(root/'countercase-prepared/anchors.json')[shard%32];measured,_=rt.score(anchor['input']);reference=anchor['native']
  err=max(abs(a-b) for a,b in zip(measured['logits'],reference['logits']))
  if err>1e-4 or measured['prompt_hash']!=reference['prompt_hash']:raise ValueError('Native anchor changed')
  c.prior.h5.write(out/'preflight.json',{'runtime':rt.receipt,'explicitly_generative':True,'fixture':fixture,'anchor_id':anchor['input']['id'],'anchor_error':err,'source_sha256':SOURCE})
  for job in groups[shard]:
   arms=list(job['arms']);random.Random(int(c.prior.h5.digest(job['input']['id'])[:8],16)).shuffle(arms)
   results={arm:c.solve(rt,job['input'],arm) for arm in arms}
   row={'id':job['input']['id'],'input_sha256':c.prior.h5.digest(job['input']),'outputs':results,'execution_order':arms,'shard':shard,'recovered':True}
   check_row(row,job)
   with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   ids.append(row['id']);print(json.dumps({'shard':shard,'completed':len(ids),'planned':len(groups[shard])}),flush=True)
 c.prior.h5.write(out/'complete.json',{'ids':ids,'count':len(ids),'source_sha256':SOURCE,'records_sha256':c.prior.h5.filehash(p),'jobs_hash':c.prior.h5.digest(groups[shard])})

def aggregate(root):
 groups=read_json(root/'recovery-prepared/jobs.json');receipt=read_json(root/'recovery-prepared/receipt.json');original=read_json(root/'recovery-prepared/original.json')
 if c.prior.h5.digest(original)!=receipt['original_sha256']:raise ValueError('Original data changed')
 rows=list(original);seen={r['id'] for r in rows}
 for i,plan in enumerate(groups):
  directory=root/'recovered-records'/f'countercase-v8-recovered-{i}';p=directory/'records.jsonl';comp=read_json(directory/'complete.json');rr=[json.loads(x) for x in p.read_text().splitlines()]
  if comp['source_sha256']!=SOURCE or comp['records_sha256']!=c.prior.h5.filehash(p) or comp['jobs_hash']!=c.prior.h5.digest(plan):raise ValueError('Recovery provenance mismatch')
  wanted={j['input']['id']:j for j in plan}
  if comp['count']!=len(plan) or len(rr)!=len(plan) or {r['id'] for r in rr}!=set(wanted):raise ValueError('Incomplete recovery')
  for row in rr:
   if row['id'] in seen:raise ValueError('Duplicate recovery')
   check_row(row,wanted[row['id']]);rows.append(row);seen.add(row['id'])
 expected={j['input']['id'] for group in read_json(root/'countercase-prepared/jobs.json') for j in group}
 if seen!=expected or len(rows)!=295:raise ValueError('Experiment incomplete')
 c.prior.h5.write(root/'all_records.json',sorted(rows,key=lambda r:r['id']))
 c.prior.h5.write(root/'completion.json',{'complete':True,'inputs':len(rows),'preserved_original':len(original),'recovered':len(rows)-len(original),'source_sha256':SOURCE,'generated_responses':sum(len(r['outputs']) for r in rows)})

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--root',default='.');p.add_argument('--out',default='recovery-output');p.add_argument('--shard',type=int,default=0);a=p.parse_args();root=Path(a.root)
 if a.mode=='prepare':prepare(root)
 elif a.mode=='aggregate':aggregate(root)
 else:run(root,Path(a.out),a.shard)
if __name__=='__main__':main()
