"""Witness v9 matched control. No symbolic verdict or target enters the model."""
from pathlib import Path
import argparse,hashlib,json,random,re,os
import countercase_v8 as previous
SEED=920260922
SHARDS=16
SAMPLE='d76b02feb42cb8df551ef119c9f724836118978d35a769861893cb02836b8afa'
POOL='720c5f967eb85c29cd2bb6ba4b7d9561fb82d34f90ba7014004ded16fa0b515c'

def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False))
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 source=json.load(open('contrast-prepared/sources/formal_fallacies.json'))
 old=json.load(open('countercase-prepared/evaluation.json'))+json.load(open('countercase-prepared/overlap_inputs.json'))
 def texts(v):
  if isinstance(v,str):yield v
  elif isinstance(v,dict):
   for x in v.values():yield from texts(x)
  elif isinstance(v,list):
   for x in v:yield from texts(x)
 old_texts=[]
 for row in old:old_texts.extend(norm(s) for s in texts({'state':row['state'],'question':row['question']}) if len(s)>45)
 union=' '.join(old_texts);order=list(range(len(source['examples'])));random.Random(SEED).shuffle(order);pool=[]
 for i in order:
  ex=source['examples'][i];stem=norm(ex['input'].split('\nOptions:')[0])
  if stem in union or any(len(x)>80 and x in stem for x in old_texts):continue
  pool.append({'id':f'bbh-v9-formal-{i:03d}','state':ex['input'],'question':{'type':'choice','instructions':'Answer the question or assess the claim in the state. Use the exact allowed answer label.','criteria':{'invalid':'invalid','valid':'valid'}},'labels':['invalid','valid'],'expected':ex['target'],'source_index':i,'family':'formal_fallacies','partition':'unseen_logic'})
 sample=pool[:64]
 assert len(pool)==234 and len(sample)==64 and digest(sample)==SAMPLE and digest(pool)==POOL
 jobs=[[{k:r[k] for k in ('id','state','question','labels')} for r in sample[i::SHARDS]] for i in range(SHARDS)]
 write('witness-prepared/jobs.json',jobs);write('witness-prepared/sample.json',sample)
 write('witness-prepared/manifest.json',{'sample_hash':SAMPLE,'pool_hash':POOL,'jobs_hash':digest(jobs),'source_hash':filehash(__file__),'new_calls':64,'base':'ordinary480, unchanged v8 source','targets_in_jobs':False})
 print(json.dumps({'sample':len(sample),'shards':list(map(len,jobs)),'jobs_hash':digest(jobs)}),flush=True)

def run(shard):
 from reconstruct_v1 import Runtime
 m=json.load(open('witness-prepared/manifest.json'));jobs=json.load(open('witness-prepared/jobs.json'))
 assert filehash(__file__)==m['source_hash'] and digest(jobs)==m['jobs_hash'] and shard in range(SHARDS)
 out=Path(f'witness-shard-{shard}');out.mkdir(exist_ok=False);rt=Runtime(Path('reconstruction-inputs'));fixture=rt.check()
 anchor=json.load(open('countercase-prepared/anchors.json'))[shard];score,_=rt.score(anchor['input']);a=anchor['native']
 error=max(abs(x-y) for x,y in zip(score['logits'],a['logits']))
 assert error<1e-4 and score['prompt_hash']==a['prompt_hash']
 write(out/'preflight.json',{'runtime':rt.receipt,'fixture':fixture,'native_anchor_error':error,'source_hash':m['source_hash'],'explicitly_generative':True})
 path=out/'records.jsonl';path.write_text('');ids=[]
 for row in jobs[shard]:
  result=previous.solve(rt,row,'standard480');r={'id':row['id'],'output':result,'input_hash':digest(row)}
  with path.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  ids.append(row['id']);print(json.dumps({'shard':shard,'completed':len(ids),'planned':len(jobs[shard])}),flush=True)
 write(out/'complete.json',{'ids':ids,'count':len(ids),'source_hash':m['source_hash'],'records_hash':filehash(path)})

def aggregate():
 jobs=json.load(open('witness-prepared/jobs.json'));m=json.load(open('witness-prepared/manifest.json'));rows=[];seen=set()
 for i,plan in enumerate(jobs):
  d=Path('records')/f'witness-v9-shard-{i}';c=json.load(open(d/'complete.json'));p=d/'records.jsonl'
  assert c['source_hash']==m['source_hash'] and c['records_hash']==filehash(p) and c['count']==len(plan)
  rr=[json.loads(l) for l in p.read_text().splitlines()];wanted={r['id']:r for r in plan}
  assert len(rr)==len(wanted) and set(c['ids'])==set(wanted)
  for r in rr:
   assert r['id'] not in seen and r['input_hash']==digest(wanted[r['id']]);seen.add(r['id']);rows.append(r)
 assert len(rows)==64
 write('all_baseline_records.json',rows);write('completion.json',{'complete':True,'inputs':len(rows),'source_hash':m['source_hash']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','aggregate']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.mode=='prepare':prepare()
 elif a.mode=='run':run(a.shard)
 else:aggregate()
