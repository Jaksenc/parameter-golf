"""No label file is opened by inference; unchanged Qwen weights throughout."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,time
from pathlib import Path
from study import MODEL,MODEL_REV,ARMS,CAPS,digest,save,emit,task_messages,program_value,final_number,number_bank
EXPECTED={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 if a.shard not in range(16):raise ValueError('invalid_shard')
 import torch
 import torch.nn.functional as F
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from huggingface_hub import snapshot_download
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260920)
 root=Path('out')/('preflight' if a.preflight else f'shard-{a.shard}');root.mkdir(parents=True,exist_ok=False)
 snap=snapshot_download(MODEL,revision=MODEL_REV,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4)
 hashes={}
 for p in Path(snap).glob('*.safetensors'):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  hashes[p.name]=h.hexdigest()
 if hashes!=EXPECTED:raise RuntimeError('model_bytes_changed')
 tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
 t=time.perf_counter()
 model,info=Qwen3_5ForConditionalGeneration.from_pretrained(snap,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
 issues={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
 if issues:raise RuntimeError(str(issues))
 model.eval();head=model.get_output_embeddings();cap={}
 def hook(module,args):cap['h']=args[0][:,-1,:].detach()
 handle=head.register_forward_pre_hook(hook)
 save(root/'runtime.json',{'model':MODEL,'revision':MODEL_REV,'weights':hashes,'torch':torch.__version__,'python':platform.python_version(),'machine':platform.machine(),'load_seconds':time.perf_counter()-t,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_run':os.environ.get('GITHUB_RUN_ID'),'optimizer_steps':0})
 def encode(msg):
  text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  x=tok(text,return_tensors='pt',add_special_tokens=False,truncation=False);n=x['input_ids'].shape[1]
  if n>8192:raise ValueError('context_limit_no_truncation')
  return x,n,hashlib.sha256(text.encode()).hexdigest()
 def generate(inp,arm):
  started=time.perf_counter();x,n,ph=encode(task_messages(inp,arm));limit=CAPS[arm]
  with torch.inference_mode():
   out=model.generate(**x,do_sample=False,max_new_tokens=limit,use_cache=True,return_dict_in_generate=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
  new=out.sequences[0,n:];text=tok.decode(new,skip_special_tokens=True)
  eos=model.generation_config.eos_token_id
  eos=set(eos if isinstance(eos,list) else [eos]);finished=bool(len(new) and int(new[-1]) in eos)
  record={'kind':'target','id':inp['id'],'arm':arm,'input_hash':digest(inp),'prompt_hash':ph,'input_tokens':n,'output_tokens':len(new),'cap_hit':len(new)>=limit and not finished,'raw':text,'seconds':time.perf_counter()-started,'status':'ok'}
  try:
   if record['cap_hit']:raise ValueError('generation_cap_without_EOS')
   if arm=='program':v,used=program_value(text,inp['bank']);record['source_references']=used
   else:v=final_number(text)
   record['value']=str(v)
  except (ValueError,SyntaxError,ZeroDivisionError,TypeError,OverflowError) as e:record.update(status='invalid',error=str(e)[:200],value=None)
  return record
 def score_retention(raw):
  inp={k:raw[k] for k in ('state','question','options')};payload={'evidence':inp['state'],'criterion':inp['question'],'options':[{'letter':chr(65+i),'description':d} for i,d in enumerate(inp['options'])]}
  msg=[{'role':'system','content':'Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.'},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
  t=time.perf_counter();x,n,ph=encode(msg);ids=[]
  for i in range(len(inp['options'])):
   letter=chr(65+i);code=tok.encode(letter,add_special_tokens=False)
   if len(code)!=1:raise ValueError('non_single_token_slot')
   text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   if tok.encode(text+letter,add_special_tokens=False)!=x['input_ids'][0].tolist()+code:raise ValueError('slot_boundary')
   ids.append(code[0])
  cap.clear()
  with torch.inference_mode():
   model(**x,logits_to_keep=1,use_cache=False)
   logits=F.linear(cap['h'].float(),head.weight[ids].float(),head.bias[ids].float() if head.bias is not None else None)[0]
   p=torch.softmax(logits.double(),0).tolist()
  order=sorted(range(len(p)),key=lambda i:p[i],reverse=True);choice=order[0] if p[order[0]]-p[order[1]]>1e-12 else None
  return {'kind':'retention','id':raw['id'],'input_hash':digest(inp),'prompt_hash':ph,'logits':logits.tolist(),'probabilities':p,'choice':choice,'seconds':time.perf_counter()-t,'tokens':n,'status':'ok'}
 def retain(record):
  with (root/'records.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
  emit('record',id=record['id'],arm=record.get('arm'),status=record['status'],seconds=record['seconds'])
 warm={'id':'owned-integration','family':'owned','split':'integration','evidence':'The basket contains 9 green tokens and 3 white tokens.','question':'How many tokens are in the basket?'};warm['bank']=number_bank(warm['evidence'],warm['question'])
 if a.preflight:
  records=[generate(warm,arm) for arm in ARMS]
  for record in records:retain(record)
  # Runtime/contract validation is separate from benchmark scoring.
  if any(r['status']!='ok' for r in records):raise RuntimeError('Owned generation contract failed; no benchmark fanout')
  retain(score_retention({'id':'owned-retention','state':'The token is green.','question':'What color is the token?','options':['Green.','White.']}))
  save(root/'complete.json',{'complete':True,'owned_fixture_only':True,'records':4,'no_target_data':True});return
 manifest=json.loads(Path('data/manifest.json').read_text());requests=json.loads(Path('data/requests.json').read_text());retention=json.loads(Path('data/retention.json').read_text())
 if digest(requests)!=manifest['target_hash'] or digest(retention)!=manifest['retention_hash']:raise RuntimeError('population_drift')
 subset=[(i,r) for i,r in enumerate(requests) if i%16==a.shard];count=0
 for i,inp in subset:
  if number_bank(inp['evidence'],inp['question'])!=inp['bank']:raise RuntimeError('changed_quantity_binding')
  order=list(ARMS);shift=i%3;order=order[shift:]+order[:shift]
  for arm in order:retain(generate(inp,arm));count+=1
 for i,row in enumerate(retention):
  if i%16==a.shard:retain(score_retention(row));count+=1
 expected=len(subset)*3+sum(i%16==a.shard for i in range(len(retention)))
 if count!=expected:raise RuntimeError('incomplete_partition')
 save(root/'complete.json',{'complete':True,'shard':a.shard,'records':count,'target_hash':manifest['target_hash'],'retention_hash':manifest['retention_hash'],'model_revision':MODEL_REV,'training_steps':0})
 handle.remove()
if __name__=='__main__':main()
