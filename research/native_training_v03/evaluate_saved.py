"""Evaluate immutable step-24 adapters from an interrupted run. No training.

The original 48-step experiment did not finish. This is a separately declared
checkpoint diagnostic, not a reconstruction of its planned final checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, subprocess, sys, time, traceback
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-4B'
REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BASE_HASHES = {
 'model.safetensors-00001-of-00002.safetensors': '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
 'model.safetensors-00002-of-00002.safetensors': 'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188',
}
ADAPTER_HASHES = {
 'terminal': '8f674d751835e93af574cc7b6d2401dd1f3e209efcf397a39bc744c7c5491c99',
 'distributed': 'a41fad093376ca6faa339ba94460b7148b897bed7c977cacbfeba69aeb436c01',
}
SOURCE_HASHES = {
 'train.py': '69ff23e08bc5f0c4210fd685000853172beb21437618060e6cc3c0647f0fe844',
 'data.py': 'ebf0b65010abab3c56aaf83b529f8ce3c08f5caf24d0d56da4cf52ebed0415a6',
 'readout.py': '0e12ec6914bb29c14222bb432e7bd9e901f35ee7c62325ca12b938dcfaf5b186',
}
SEED = 1703

def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
 return h.hexdigest()

def digest(data): return hashlib.sha256(data).hexdigest()

def canonical(obj):
 return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def save(path, obj):
 path=Path(path); tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def validate_source(source, scope):
 source=Path(source)
 for name, h in SOURCE_HASHES.items():
  if sha(source/'source'/name)!=h: raise ValueError('changed source: '+name)
 if sha(source/'step-24.safetensors') != ADAPTER_HASHES[scope]: raise ValueError('changed adapter')
 meta=json.loads((source/'metadata.json').read_text())
 if meta['scope']!=scope or meta['selected_step']!=24 or meta['status']!='failed':
  raise ValueError('unexpected original run identity')
 if meta['error']!='TimeoutError: bounded experiment wall budget': raise ValueError('unexpected stop')
 training=[json.loads(x) for x in (source/'training.jsonl').read_text().splitlines()]
 if [r['step'] for r in training] != list(range(1,26)): raise ValueError('training journal changed')
 return meta

def summarize(rows, labels):
 stats={}
 for (split, arm) in sorted({(r['split'],r['arm']) for r in rows}):
  subset=[r for r in rows if (r['split'],r['arm'])==(split,arm)]
  valid=[r for r in subset if r.get('probs') is not None]
  stats.setdefault(split,{})[arm]={
   'n':len(subset), 'valid':len(valid),
   'correct':sum(r['correct'] for r in subset),
   'nll_valid_only':sum(r['nll'] for r in valid)/len(valid) if valid else None,
   'brier_valid_only':sum(sum((p-float(label==labels[r['id']]))**2 for label,p in r['probs'].items()) for r in valid)/len(valid) if valid else None,
   'forward_seconds_known':sum(r.get('seconds') or 0 for r in subset),
  }
 return stats

def worker(args):
 import torch, transformers
 from huggingface_hub import snapshot_download
 from safetensors.torch import load_file
 source=Path(args.source).resolve(); out=Path(args.out).resolve()
 original=validate_source(source,args.scope)
 sys.path.insert(0,str(source/'source'));import readout
 packets=json.loads(Path(args.input).read_text())
 for p in packets:
  if set(p)!={'id','task'}:raise ValueError('solver packet contains non-input fields')
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
 start=time.perf_counter()
 meta={'status':'started','scope':args.scope,'checkpoint_step':24,'new_training':False,
       'model':MODEL,'revision':REV,'adapter_sha256':ADAPTER_HASHES[args.scope],
       'base_dtype':'bfloat16','adapter_dtype':'float32','torch':torch.__version__,
       'transformers':transformers.__version__,'python':sys.version,'platform':platform.platform(),
       'threads':4,'device':'cpu','seed':SEED,'run_id':os.environ.get('GITHUB_RUN_ID'),
       'source_commit':os.environ.get('GITHUB_SHA'),'original_run_id':original['run_id'],
       'generator_and_data_unchanged':True,'energy_measured':False,'cost_usd':None,
       'mode_order':'input-id hash parity alternates base/adapted per item',
       'generation_tokens':0,'memory_used':False,'semantic_verification':False}
 save(out/'worker-metadata.json',meta)
 try:
  local=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2)
  hashes={p.name:sha(p) for p in Path(local).glob('*.safetensors')}
  if hashes!=BASE_HASHES:raise ValueError('base weight identity mismatch')
  tok=transformers.AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
  model,info=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(local,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
  bad={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
  if bad:raise ValueError('model load mismatch: '+str(bad))
  model.requires_grad_(False);model.eval()
  original_tokens=json.loads((source/'token_manifest.json').read_text()); prepared={};manifest={}
  for packet in packets:
   messages,ledger=readout.messages(packet['task'],'canonical')
   text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=tok.encode(text,add_special_tokens=False)
   if not 1<=len(ids)<=512:raise ValueError('context length, no truncation')
   slots=[]
   for row in ledger:
    token=tok.encode(row['marker'],add_special_tokens=False)
    if len(token)!=1 or tok.encode(text+row['marker'],add_special_tokens=False)!=ids+token:raise ValueError('token boundary')
    slots.append(token[0])
   labels=[r['label'] for r in ledger]
   record={'labels':labels,'slots':slots,'prompt_sha256':digest(text.encode()),'tokens':len(ids)}
   old=original_tokens[packet['id']]
   if any(record[k]!=old[k] for k in record):raise ValueError('original prompt differs: '+packet['id'])
   manifest[packet['id']]=record
   prepared[packet['id']]={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long),'use_cache':False,'return_dict':True,'logits_to_keep':1}
  save(out/'reconstructed-token-manifest.json',manifest)
  targets=[(n,m) for n,m in model.named_modules() if n.startswith('model.language_model.layers.') and n.endswith('.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
  if len(targets)!=32:raise ValueError('model projection layout changed')
  if args.scope=='terminal':targets=targets[-1:]
  if [n for n,_ in targets]!=original['targets']:raise ValueError('adapter binding changed')
  class Factors(torch.nn.Module):
   def __init__(self, projection):
    super().__init__();self.a=torch.nn.Parameter(torch.empty((4,projection.in_features)))
    self.b=torch.nn.Parameter(torch.empty((projection.out_features,4)))
   def forward(self,x):
    return (2*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
  factors=torch.nn.ModuleList([Factors(m) for _,m in targets]);weights=load_file(str(source/'step-24.safetensors'))
  if any(not torch.isfinite(v).all() for v in weights.values()):raise ValueError('nonfinite adapter')
  factors.load_state_dict(weights,strict=True);factors.eval();factors.requires_grad_(False)
  if sum(p.numel() for p in factors.parameters())!=original['adapter_parameters']:raise ValueError('parameter count')
  enabled=False;hooks=[]
  def make_hook(f):
   def hook(module,inputs,result):return result+f(inputs[0]) if enabled else result
   return hook
  for (_,m),f in zip(targets,factors):hooks.append(m.register_forward_hook(make_hook(f)))
  meta.update(status='evaluating',weight_hashes=hashes,targets=[n for n,_ in targets],adapter_parameters=original['adapter_parameters'],setup_seconds=time.perf_counter()-start)
  save(out/'worker-metadata.json',meta)
  def forward(key):return model(**prepared[key]).logits[0,-1,manifest[key]['slots']].float()
  with torch.inference_mode(),(out/'predictions.jsonl').open('x') as journal:
   first=packets[0]['id'];enabled=False;reference=forward(first).tolist()
   for i,packet in enumerate(packets):
    key=packet['id'];order=['base','adapted']
    if int(digest(key.encode())[:8],16)%2:order.reverse()
    for arm in order:
     enabled=arm=='adapted';t=time.perf_counter();z=forward(key)
     if not torch.isfinite(z).all():raise ValueError('nonfinite logits')
     p=z.softmax(-1).tolist();labels=manifest[key]['labels']
     rec={'id':key,'arm':arm,'logits':z.tolist(),'probs':dict(zip(labels,p)),
          'predicted':sorted(zip(labels,p),key=lambda x:(-x[1],x[0]))[0][0],
          'seconds':time.perf_counter()-t,'prompt_sha256':manifest[key]['prompt_sha256'],
          'status':'ok'}
     journal.write(canonical(rec)+'\n');journal.flush();os.fsync(journal.fileno())
    if i%12==0:print('EVALUATED '+str(i+1)+'/'+str(len(packets)),flush=True)
   enabled=False;restored=forward(first).tolist()
  for h in hooks:h.remove()
  meta['base_toggle_restoration_max_difference']=max(abs(a-b) for a,b in zip(reference,restored))
  if meta['base_toggle_restoration_max_difference']>1e-4:raise ValueError('base toggling changed output')
  meta['adapter_unchanged']=sha(source/'step-24.safetensors')==ADAPTER_HASHES[args.scope]
  meta['base_files_unchanged']=all(sha(Path(local)/n)==h for n,h in BASE_HASHES.items())
  if not meta['adapter_unchanged'] or not meta['base_files_unchanged']:raise ValueError('weights modified')
  meta.update(status='completed',completed_attempts=len(packets)*2)
 except Exception as e:
  meta.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
 finally:
  meta['wall_seconds']=time.perf_counter()-start;save(out/'worker-metadata.json',meta)

def main(args):
 source=Path(args.source).resolve();out=Path(args.out).resolve()
 if args.worker:return worker(args)
 if args.validate_only:
  print(canonical(validate_source(source,args.scope)));return
 out.mkdir(parents=True,exist_ok=False);original=validate_source(source,args.scope)
 data=json.loads((source/'evaluation_data.json').read_text())
 if {k:len(v) for k,v in data.items()}!={'development':24,'transfer':48,'composition':16,'jev_public_regression':20}:raise ValueError('changed cohort')
 flat=[r for rows in data.values() for r in rows]
 inputs=[{'id':r['id'],'task':r['task']} for r in flat];save(out/'input-packets.json',inputs)
 protocol={'scope':'post-interruption, fixed step-24 diagnostic; not completed step-48 experiment',
  'original_run_id':original['run_id'],'checkpoint_step':24,'new_training':False,
  'source_sha256':sha(__file__),'selected_by':'only complete saved checkpoint; both scopes evaluated',
  'all_original_evaluation_ids':[r['id'] for r in flat],'targets_worker_visible':False,
  'base_revision':REV,'adapter_sha256':ADAPTER_HASHES[args.scope],
  'public_jevbench':'20 exposed easy/original items only; no hard or sealed evaluation',
  'timeout_seconds':1500,'promotion':False}
 save(out/'protocol.json',protocol);print('PROTOCOL_LOCK '+canonical(protocol),flush=True)
 env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))}
 command=[sys.executable,__file__,'--worker','--scope',args.scope,'--source',str(source),'--out',str(out),'--input',str(out/'input-packets.json')]
 try:code=subprocess.run(command,env=env,timeout=1500).returncode
 except subprocess.TimeoutExpired:code='timeout'
 p=out/'predictions.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
 lookup={r['id']:r for r in flat};seen=set()
 for row in rows:
  if (row['id'],row['arm']) in seen:raise ValueError('duplicate attempt')
  seen.add((row['id'],row['arm']));r=lookup[row['id']]
  row.update(split=r['split'],target=r['target'],correct=row['predicted']==r['target'],
             nll=-math.log(max(1e-30,row['probs'][r['target']])))
 for r in flat:
  for arm in ('base','adapted'):
   if (r['id'],arm) not in seen:rows.append({'id':r['id'],'arm':arm,'split':r['split'],'target':r['target'],'probs':None,'correct':False,'status':'not_completed'})
 summary=summarize(rows,{r['id']:r['target'] for r in flat})
 save(out/'scored.json',rows);save(out/'summary.json',{'worker_exit':code,'summary':summary,'protocol':protocol})
 print('FINAL_SUMMARY '+canonical({'worker_exit':code,'summary':summary}),flush=True)
 if code!=0:raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['terminal','distributed'],required=True)
 p.add_argument('--source',required=True);p.add_argument('--out',default='evaluation')
 p.add_argument('--input');p.add_argument('--worker',action='store_true');p.add_argument('--validate-only',action='store_true')
 main(p.parse_args())
