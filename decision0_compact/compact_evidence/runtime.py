"""Pinned full-model runtime. No fallback, quantization or substituted checkpoint.

Only public prompt messages and semantic labels reach this class. Memory and
platform guards run before imports/downloads. Real equivalence remains a model
execution preflight, not something unit tests can establish on a toy backend.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,platform,sys,time
from pathlib import Path
from .prompts import json_complete
from .protocol import PROTOCOL

WEIGHT_HASHES={
 'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
 'model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
      for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def doctor():
    limits=[]
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
      p=Path(path)
      if p.exists():
        value=p.read_text().strip()
        if value.isdigit():limits.append(int(value))
    try:limits.append(os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError,ValueError):pass
    memory=min(limits) if limits else None
    problems=[]
    if platform.machine() not in ('aarch64','arm64'):problems.append('Pinned comparison requires ARM; this environment is '+platform.machine())
    if memory is None or memory<12*(1<<30):problems.append('Conservative full-model guard requires at least 12 GiB available physical/cgroup allowance')
    if sys.version_info[:2]!=(3,11):problems.append('Pinned study requires Python 3.11')
    modules={n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','safetensors','huggingface_hub')}
    for name,present in modules.items():
      if not present:problems.append(name+' is not installed')
    return {'status':'blocked' if problems else 'environment_preflight_passed','machine':platform.machine(),
       'memory_limit_bytes':memory,'python':sys.version,'modules':modules,'problems':problems,
       'minimum_memory_policy_bytes':12*(1<<30),'memory_policy_is_capacity_guard_not_benchmark':True,
       'model_downloaded':False,'model_loaded':False,'inference_calls':0}

def selected_projection(hidden,head,slot_ids):
    """Same final normalized hidden state, FP32 selected native output rows."""
    import torch
    weights=head.weight[slot_ids].float();bias=head.bias[slot_ids].float() if head.bias is not None else None
    return torch.nn.functional.linear(hidden.float(),weights,bias)

class Runtime:
    kind='qwen3.5-4b-pinned-native'
    def __init__(self):
      pre=doctor()
      if pre['status']=='blocked':raise RuntimeError(json.dumps(pre,indent=2))
      import torch,transformers
      from huggingface_hub import snapshot_download
      if torch.__version__.split('+')[0]!='2.10.0' or transformers.__version__!='5.17.0':raise RuntimeError('Pinned torch/transformers versions differ; no silent replacement')
      self.torch=torch
      torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(441109)
      directory=snapshot_download(PROTOCOL['model'],revision=PROTOCOL['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2)
      hashes={p.name:fingerprint(p) for p in Path(directory).glob('*.safetensors')}
      if hashes!=WEIGHT_HASHES:raise RuntimeError('Base weight hashes differ')
      self.tokenizer=transformers.AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
      self.model,info=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(directory,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
      if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('Model loading mismatch: '+str(info))
      self.model.eval();self.model.requires_grad_(False)
      self.head=self.model.get_output_embeddings();self.hidden=None
      def capture(module,args):self.hidden=args[0][:,-1,:].detach()
      self.hook=self.head.register_forward_pre_hook(capture)
      self.meta={'kind':self.kind,'model':PROTOCOL['model'],'revision':PROTOCOL['revision'],'weight_hashes':hashes,
        'torch':torch.__version__,'transformers':transformers.__version__,'threads':4,'machine':platform.machine(),
        'python':sys.version,'parameter_count':sum(p.numel() for p in self.model.parameters()),'weights_updated':False}
    def encode(self,messages):
      text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
      ids=self.tokenizer.encode(text,add_special_tokens=False)
      if not ids or len(ids)>PROTOCOL['input_cap']:raise ValueError('Input outside limit; no truncation')
      return text,ids
    def generate(self,messages,kind,cap):
      from transformers import StoppingCriteria,StoppingCriteriaList
      start=time.perf_counter();torch=self.torch;text,ids=self.encode(messages);tokenizer=self.tokenizer;prefix_len=len(ids)
      class CompleteJSON(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
          generated=tokenizer.decode(input_ids[0,prefix_len:].tolist(),skip_special_tokens=True)
          return json_complete(generated,kind)
      kwargs={}
      if kind!='full_workspace':kwargs['stopping_criteria']=StoppingCriteriaList([CompleteJSON()])
      start=time.perf_counter();x=torch.tensor([ids])
      with torch.inference_mode():
        y=self.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,use_cache=True,
              max_new_tokens=cap,pad_token_id=tokenizer.eos_token_id,**kwargs)
      output=y[0,len(ids):].tolist();raw=tokenizer.decode(output,skip_special_tokens=True)
      elapsed=time.perf_counter()-start;self.hidden=None
      # A JSON result closing exactly at cap is completed, not silently a failure.
      complete=json_complete(raw,kind) if kind!='full_workspace' else (bool(output) and output[-1]==tokenizer.eos_token_id)
      return {'kind':'generation','text':raw,'token_ids':output,'input_tokens':len(ids),'output_tokens':len(output),
        'reached_cap':len(output)>=cap,'format_complete':complete,'seconds':elapsed,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
    def score(self,messages,labels):
      torch=self.torch;start=time.perf_counter();text,ids=self.encode(messages);slots=[]
      for i in range(len(labels)):
        token=self.tokenizer.encode(chr(65+i),add_special_tokens=False)
        if len(token)!=1 or self.tokenizer.encode(text+chr(65+i),add_special_tokens=False)!=ids+token:raise RuntimeError('Answer-token boundary changed')
        slots.append(token[0])
      if len(set(slots))!=len(slots):raise RuntimeError('Answer tokens collide')
      x=torch.tensor([ids]);self.hidden=None
      with torch.inference_mode():
        result=self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
        if self.hidden is None:raise RuntimeError('Missing final normalized hidden state')
        logits=selected_projection(self.hidden,self.head,slots)[0].tolist()
        native=result.logits[0,-1,slots].float().tolist()
      if any(not math.isfinite(v) for v in logits):raise RuntimeError('Nonfinite scores')
      exponentials=[math.exp(v-max(logits)) for v in logits];probabilities=[v/sum(exponentials) for v in exponentials]
      predicted=min(zip(labels,probabilities),key=lambda x:(-x[1],x[0]))[0]
      self.hidden=None
      return {'kind':'decision','labels':list(labels),'logits':logits,'native_logits':native,'probabilities':probabilities,
              'predicted':predicted,'input_tokens':len(ids),'output_tokens':0,'seconds':time.perf_counter()-start,
              'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
    def close(self):self.hook.remove()
