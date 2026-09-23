"""Availability-defined step32 evaluation of all six interrupted Learn-v16 runs.
No optimizer, no generation, no calibration. Same model, inputs and code readout.
"""
from __future__ import annotations
import argparse, hashlib, json, os, resource, time
from pathlib import Path
import learn_v16 as original

SOURCE='88d3ce9ebae554adbb1728b45254bfe7a87043a26d1264fe0fad55077054b587'
STEP=32
EXPECTED={
 '16101-supervised':'b7b2a1da8015abb2d205bed06878d335de7111e55ec212b6e00899a9742e6b4f',
 '16101-relational':'4178b503143f4c0e8c64e30717021d2733a9b1f98071998205d45c48d71a452c',
 '16102-supervised':'2fb43e01029bc540e72ffc6ec7896b10e515c2448177b3db656f3d0f90ea39ab',
 '16102-relational':'766a05cd00e2a34f33a46d72a941f309535730a87a2c88adb39bb6b3facd228c',
 '16103-supervised':'4f2f731ab7a715f5a70b4b1602cb93b3c675ccdefbfac73859ca767516e37368',
 '16103-relational':'a7491cd19c1818a6fccfd7aa74a61105c35b1496f872d46d92d4e5d412c7574e'}

def run(root:Path,out:Path,seed:int,arm:str):
 import torch
 from safetensors.torch import load_file
 from reconstruct_v1 import Runtime
 import measure_v14
 key=f'{seed}-{arm}'
 if key not in EXPECTED:raise ValueError('Unregistered model')
 if original.filehash(original.__file__)!=SOURCE:raise ValueError('Original inference code changed')
 r,e,m=original.load_data(root/'learn-prepared')
 parent=root/'trained'/f'learn-v16-model-{key}'
 checkpoint=parent/f'adapter-step-{STEP}.safetensors'
 if original.filehash(checkpoint)!=EXPECTED[key]:raise ValueError('Checkpoint changed')
 baseline=json.loads((parent/'baseline.json').read_text())
 prior=json.loads((parent/'preflight.json').read_text())
 trace=[json.loads(s) for s in (parent/'training.jsonl').read_text().splitlines()]
 schedule=json.loads((parent/'schedule.json').read_text())
 if [a['step'] for a in trace]!=list(range(1,len(trace)+1)) or len(trace)<STEP:raise ValueError('Training ledger incomplete')
 if [a['edge_index'] for a in trace[:STEP]]!=schedule['edge_indices'][:STEP]:raise ValueError('Training schedule mismatch')
 if prior['source_sha256']!=SOURCE:raise ValueError('Wrong parent source')
 out.mkdir(parents=True,exist_ok=False)
 rt=Runtime(root/'reconstruction-inputs');model=rt.model;fixture=rt.check()
 model.requires_grad_(False);model.eval();cache={}
 for i in m['eval_indices']:
  row=r[i];messages=measure_v14.arm_messages(row['input'],'','semantic_codes')
  prompt=rt.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
  codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(row['target']))]
  if len(ids)>original.MAX_TOKENS or any(len(c)!=1 for c in codes) or len({c[0] for c in codes})!=len(codes):raise ValueError('Token contract')
  for j,c in enumerate(codes):
   if rt.tokenizer.encode(prompt+chr(65+j),add_special_tokens=False)!=ids+c:raise ValueError('Code boundary')
  cache[i]={'ids':torch.tensor([ids]),'mask':torch.ones((1,len(ids)),dtype=torch.long),'codes':[c[0] for c in codes],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'tokens':len(ids)}
 index={b['record_index']:b for b in baseline}
 if set(index)!=set(m['eval_indices']):raise ValueError('Baseline population')
 for i in m['eval_indices']:
  if index[i]['prompt_sha256']!=cache[i]['prompt_sha256'] or index[i]['input_sha256']!=original.data.digest(r[i]['input']):raise ValueError('Prompt/input changed')
 calls=0
 def score(i):
  nonlocal calls
  c=cache[i];rt.head.codes=c['codes'];calls+=1
  with torch.inference_mode():
   z=model(input_ids=c['ids'],attention_mask=c['mask'],logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
  if z.numel()!=len(r[i]['target']) or not torch.isfinite(z).all():raise ValueError('Invalid logits')
  return z
 first,last=m['eval_indices'][0],m['eval_indices'][-1]
 anchors={i:float((score(i)-torch.tensor(index[i]['logits'])).abs().max()) for i in (first,last)}
 if max(anchors.values())>1e-4:raise ValueError('Native baseline anchor mismatch')
 targets,factors,hooks=original.make_factors(torch,model,seed)
 initial_hash=original.tensors_digest(factors.state_dict())
 if initial_hash!=prior['initial_adapter_sha256']:raise ValueError('Initialization mismatch')
 zero_error=float((score(first)-torch.tensor(index[first]['logits'])).abs().max())
 if zero_error>1e-4:raise ValueError('Zero factors alter baseline')
 state=load_file(str(checkpoint));factors.load_state_dict(state,strict=True);factors.requires_grad_(False)
 loaded_hash=original.tensors_digest(factors.state_dict())
 if loaded_hash!=original.tensors_digest(state) or loaded_hash==initial_hash:raise ValueError('Weights not loaded')
 changes={k:float(v.abs().max()) for k,v in state.items() if k.endswith('.b')}
 if len(changes)!=32 or min(changes.values())<=0:raise ValueError('Unchanged layer')
 receipt={'parent_run':35808253851,'seed':seed,'arm':arm,'evaluated_step':STEP,'planned_step':64,
          'checkpoint_sha256':EXPECTED[key],'loaded_tensor_sha256':loaded_hash,'parent_source_sha256':SOURCE,
          'evaluation_source_sha256':original.filehash(__file__),'runtime':rt.receipt,'fixture':fixture,
          'native_anchor_errors':anchors,'zero_adapter_error':zero_error,'max_B_by_layer':changes,
          'initial_adapter_sha256':initial_hash,'trained_adapter_parameters':sum(v.numel() for v in state.values()),
          'new_optimizer_updates':0,'autoregressive_generation':False,'heldout_worlds':m['check_worlds'],
          'original_experiment_status':'64-update primary incomplete; this is the fixed common saved checkpoint audit'}
 original.save(out/'preflight.json',receipt)
 path=out/'predictions.jsonl';path.write_text('');answers=[];start=time.perf_counter()
 for n,i in enumerate(m['eval_indices']):
  tick=time.perf_counter();z=score(i);p=torch.softmax(z,-1)
  row={'id':r[i]['input']['id'],'record_index':i,'input_sha256':original.data.digest(r[i]['input']),
       'prompt_sha256':cache[i]['prompt_sha256'],'tokens':cache[i]['tokens'],'code_token_ids':cache[i]['codes'],
       'logits':z.tolist(),'probabilities':p.tolist(),'seconds':time.perf_counter()-tick,
       'label':r[i]['input']['labels'][int(z.argmax())],'seed':seed,'arm':arm,'checkpoint_step':STEP}
  with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  answers.append(row)
  if n%12==0:original.emit(stage='checkpoint-eval',seed=seed,arm=arm,done=n+1,total=len(m['eval_indices']))
 # Independently load from disk again and verify same probe, then disable adapters.
 factors.load_state_dict(load_file(str(checkpoint)),strict=True)
 reload_error=float((score(first)-torch.tensor(answers[0]['logits'])).abs().max())
 for f in factors:f.enabled=False
 restoration_error=float((score(first)-torch.tensor(index[first]['logits'])).abs().max())
 for h in hooks:h.remove()
 if reload_error>1e-4 or restoration_error>1e-4:raise ValueError('Reload/restoration mismatch')
 if original.filehash(checkpoint)!=EXPECTED[key]:raise ValueError('Checkpoint was modified')
 unique_seen=sorted({i for eid in schedule['edge_indices'][:STEP] for i in (e[eid]['left'],e[eid]['right'])})
 original.save(out/'complete.json',{**receipt,'status':'saved-checkpoint-evaluation-complete','outputs':len(answers),
              'records_sha256':original.filehash(path),'eval_indices':m['eval_indices'],'forward_calls':calls,
              'reload_error':reload_error,'base_restoration_error':restoration_error,
              'evaluation_seconds_excluding_model_load':time.perf_counter()-start,
              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'unique_fit_inputs_by_step32':len(unique_seen),'fit_record_indices_seen':unique_seen})
 original.emit(stage='saved-checkpoint-evaluation-complete',seed=seed,arm=arm,records=len(answers))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=original.ARMS,required=True);a=p.parse_args()
 run(Path(a.root),Path(a.out),a.seed,a.arm)
