"""One real all-depth adapter update on a separate synthetic smoke request.

This is an execution/memory test, not a capability or benchmark experiment.
No public benchmark or prompt-selection example is used for its update.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, sys, time, traceback
from pathlib import Path

MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
HASHES={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save(out,obj):
    p=Path(out);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False,sort_keys=True));temp.replace(p)

def memory():
    return {'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'mem_available_line':next((l.strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')),'unknown')}

def run(out):
    import torch, transformers
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(73019)
    cpuinfo=Path('/proc/cpuinfo').read_text()
    meta={'scope':'single synthetic request, one optimizer update, architecture/memory preflight only','model':MODEL,'revision':REV,'platform':platform.machine(),'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'git_sha':os.environ.get('GITHUB_SHA'),'cpu_bf16_flag':('bf16' in cpuinfo),'training_seed':73019,'public_benchmark_used':False,'prompt_selection_data_used':False,'official_score':None}
    save(out/'metadata.json',meta)
    start=time.perf_counter()
    local=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2)
    hashes={p.name:digest_file(p) for p in Path(local).glob('*.safetensors')}
    if hashes!=HASHES:raise RuntimeError('Base checkpoint digest mismatch')
    tokenizer=transformers.AutoTokenizer.from_pretrained(local,trust_remote_code=False,local_files_only=True)
    model,info=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(local,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    bad={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
    if bad:raise RuntimeError(str(bad))
    model.requires_grad_(False);model.eval();head=model.get_output_embeddings()
    # Independent toy state; archive clauses are not taken from any evaluation set.
    records=['Current inventory for transfer ZULM: 17 sealed packs, each containing 9 copper chips. An additional tray contains 6 copper chips.',
             'Rule: dispatch is allowed only when at least 160 copper chips are present. Sealed packs count toward inventory.']
    records += [f'Unrelated archive position {i}: record LARK-{i} concerns discarded packaging. It changes no copper-chip count or dispatch rule.' for i in range(20)]
    records.insert(12,'Revision: after the count, remove exactly 4 damaged copper chips. No other deductions apply.')
    question='Compute usable copper chips for transfer ZULM and apply the dispatch threshold. A means dispatch is allowed; B means dispatch is not allowed. Return only A or B.'
    messages=[{'role':'system','content':'Use the supplied evidence and rule. Choose the correct answer letter.'},{'role':'user','content':'\n'.join(records)+'\n'+question}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tokenizer.encode(text,add_special_tokens=False)
    if not 512<=len(ids)<=2048:raise RuntimeError(f'Unexpected preflight length {len(ids)}')
    slots=[tokenizer.encode(x,add_special_tokens=False) for x in ['A','B']]
    if any(len(s)!=1 for s in slots):raise RuntimeError('Answer code is not single-token')
    for ch,s in zip(['A','B'],slots):
        if tokenizer.encode(text+ch,add_special_tokens=False)!=ids+s:raise RuntimeError('Answer token boundary mismatch')
    slots=[s[0] for s in slots];inputs={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long),'use_cache':False,'return_dict':True,'logits_to_keep':1}
    def score():return model(**inputs).logits[0,-1,slots].float()
    with torch.inference_mode():base=score().clone()
    class Factors(torch.nn.Module):
        def __init__(self,down):
            super().__init__();self.a=torch.nn.Parameter(torch.empty(8,down.in_features));self.b=torch.nn.Parameter(torch.zeros(down.out_features,8));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):return (2*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
    targets=[(n,m) for n,m in model.named_modules() if n.startswith('model.language_model.layers.') and n.endswith('.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
    if len(targets)!=32:raise RuntimeError(f'Expected exactly 32 feed-forward targets, found {len(targets)}')
    factors=torch.nn.ModuleList([Factors(m) for _,m in targets]);hooks=[]
    for (_,module),factor in zip(targets,factors):
        def hook(m,args,result,factor=factor):return result+factor(args[0])
        hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():zero=score().clone()
    err=float((zero-base).abs().max())
    if err>1e-4:raise RuntimeError(f'Zero-adapter equivalence failed: {err}')
    save(out/'loaded.json',{'hashes':hashes,'tokens':len(ids),'targets':[n for n,_ in targets],'adapter_parameters':sum(p.numel() for p in factors.parameters()),'zero_adapter_max_logit_difference':err,'base_logits':base.tolist(),'load_and_initial_forward_seconds':time.perf_counter()-start,**memory()})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train();optimizer=torch.optim.AdamW(factors.parameters(),lr=1e-4,weight_decay=0)
    before=[m.b.detach().clone() for m in factors];optimizer.zero_grad(set_to_none=True);tick=time.perf_counter()
    logits=score();loss=torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([1]))
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();backward_seconds=time.perf_counter()-tick
    norms={n:float(f.b.grad.float().norm()) if f.b.grad is not None else None for (n,_),f in zip(targets,factors)}
    if any(v is None or not math.isfinite(v) or v<=0 for v in norms.values()):raise RuntimeError('Missing, zero or nonfinite gradient on a layer: '+str(norms))
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('An original base parameter acquired a gradient')
    torch.nn.utils.clip_grad_norm_(factors.parameters(),1.0);optimizer.step()
    changes={n:float((f.b.detach()-b).abs().max()) for (n,_),f,b in zip(targets,factors,before)}
    if not all(v>0 and math.isfinite(v) for v in changes.values()):raise RuntimeError('An earlier layer did not update')
    model.eval();model.gradient_checkpointing_disable();model.disable_input_require_grads()
    with torch.inference_mode():after=score().clone()
    for h in hooks:h.remove()
    with torch.inference_mode():restored=score().clone()
    restore_err=float((restored-base).abs().max())
    if restore_err>1e-4:raise RuntimeError(f'Base restoration failed: {restore_err}')
    result={**meta,'status':'passed','tokens':len(ids),'optimizer_updates':1,'updated_layers':len(changes),'adapter_parameters':sum(p.numel() for p in factors.parameters()),'rank_per_projection':8,'trainable_base_parameters':0,'bf16_base':True,'adapter_dtype':'float32','gradient_checkpointing':'non-reentrant','training_loss':float(loss.detach()),'forward_backward_seconds':backward_seconds,'elapsed_seconds':time.perf_counter()-start,'base_logits':base.tolist(),'after_update_logits':after.tolist(),'zero_adapter_max_difference':err,'restored_base_max_difference':restore_err,'gradient_norms':norms,'maximum_B_parameter_changes':changes,**memory(),'capability_improvement_established':False,'new_model_checkpoint_promoted':False}
    save(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:run(out)
    except Exception as e:
        save(out/'FAILED.json',{'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'scope':'execution preflight, no capability or benchmark result',**memory()});raise
