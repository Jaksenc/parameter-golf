"""Model-only cache extraction. Labels and semantic proofs never enter prompts.
CPU execution is deliberately pinned; no paid service, hidden fallback, or truncation.
"""
from __future__ import annotations
import argparse,hashlib,inspect,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import messages,audit
from .io import (MODEL,REVISION,MODEL_SHA,DATA_HASH,filehash,save_json,
                 load_rows,write_cache)
from .semantics import digest
from .terminal import qwen_rms

PHASES={'learn':('train','development'),
        'transfer':('transfer_wording','transfer_domain','transfer_composition')}

def load_model(offline=False):
    # Import here so data generation and numerical tests do not need model packages.
    from huggingface_hub import snapshot_download
    import transformers
    if transformers.__version__!='5.17.0':raise RuntimeError('Expected transformers==5.17.0')
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    torch.set_num_threads(4)
    snapshot=snapshot_download(MODEL,revision=REVISION,local_files_only=offline,
        allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4)
    hashes={p.name:filehash(p) for p in Path(snapshot).glob('*.safetensors')}
    if hashes!=MODEL_SHA:raise RuntimeError('Pinned checkpoint bytes do not match')
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model,info=Qwen3_5ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    issues={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
    if issues:raise RuntimeError('Checkpoint load mismatch: '+str(issues))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,tokenizer,hashes

def encode(tokenizer,row,max_tokens=2048):
    text=tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>max_tokens:raise ValueError(f'Input length {len(ids)} outside tested bound; never truncate')
    slots=[]
    for letter in 'ABCDEF':
        z=tokenizer.encode(letter,add_special_tokens=False)
        if len(z)!=1 or tokenizer.encode(text+letter,add_special_tokens=False)!=ids+z:
            raise ValueError('Answer boundary or slot identity is invalid')
        slots.append(z[0])
    return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)},slots,len(ids),digest(text)

def terminal_modules(model):
    names=[n for n,m in model.named_modules() if n.endswith('.layers.31.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
    if len(names)!=1:raise RuntimeError('Unexpected terminal architecture')
    name=names[0];mod=model.get_submodule(name)
    layer=model.get_submodule(name.rsplit('.mlp.',1)[0]);body=model.get_submodule(name.split('.layers.')[0])
    return name,mod,layer,body.norm

def extract(rows,model,tokenizer,hashes,path,phase,shard,shards):
    name,mod,layer,norm=terminal_modules(model);head=model.get_output_embeddings()
    capture={};handles=[]
    def pre(key):
        def hook(m,args):capture[key]=args[0][:,-1,:].detach().clone()
        return hook
    def after(m,args,result):capture['y']=result[:,-1,:].detach().clone()
    handles.extend([layer.post_attention_layernorm.register_forward_pre_hook(pre('r')),
        mod.register_forward_pre_hook(pre('x')),mod.register_forward_hook(after),
        norm.register_forward_pre_hook(pre('norm_in')),head.register_forward_pre_hook(pre('h'))])
    arrays={k:[] for k in ('x','r','y','z0')};records=[];slots_ref=None;max_norm_error=0.;start=time.perf_counter()
    try:
        with torch.no_grad():
            for index,row in enumerate(rows):
                encoded,slots,nt,prompt_hash=encode(tokenizer,row)
                if slots_ref is not None and slots_ref!=slots:raise ValueError('Changed answer slots')
                slots_ref=slots;capture.clear();call=time.perf_counter()
                model(**encoded,use_cache=False,logits_to_keep=1)
                if not torch.equal(capture['r']+capture['y'],capture['norm_in']):raise RuntimeError('Terminal residual reconstruction failed')
                h=qwen_rms(capture['norm_in'],norm.weight,norm.eps)
                error=float((h.float()-capture['h'].float()).abs().max());max_norm_error=max(max_norm_error,error)
                if error>2e-5:raise RuntimeError('Cached normalization differs from native model')
                w=head.weight[slots].float();bias=head.bias[slots].float() if head.bias is not None else None
                z=F.linear(capture['h'].float(),w,bias)[0]
                for k in ('x','r','y'):arrays[k].append(capture[k][0])
                arrays['z0'].append(z)
                records.append({'id':row['id'],'split':row['split'],'group':row['group'],
                    'input_hash':row['input_hash'],'serialized_prompt_hash':prompt_hash,'tokens':nt,
                    'seconds':time.perf_counter()-call})
                print(json.dumps({'kind':'cache_progress','phase':phase,'shard':shard,'done':index+1,'total':len(rows)}),flush=True)
    finally:
        for h in handles:h.remove()
    arrays={k:torch.stack(v) for k,v in arrays.items()}
    arrays['answer_weight']=head.weight[slots_ref].detach().float()
    if head.bias is not None:arrays['answer_bias']=head.bias[slots_ref].detach().float()
    arrays['norm_weight']=norm.weight.detach().float();arrays['norm_epsilon']=norm.eps
    norm_source=inspect.getsource(type(norm))
    write_cache(path,arrays,{'kind':'qwen3.5-4b-terminal-v4','model_revision':REVISION,'model_hashes':hashes,
        'data_hash':DATA_HASH,'module':name,'norm_source_hash':hashlib.sha256(norm_source.encode()).hexdigest(),
        'prompt_source_hash':hashlib.sha256(inspect.getsource(messages).encode()).hexdigest(),
        'rows':records,'phase':phase,'shard':shard,'shards':shards,'native_norm_max_error':max_norm_error,
        'torch_version':torch.__version__,'build_seconds':time.perf_counter()-start,'truncation':False,
        'label_or_proof_forwarded':False})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',default='cache')
    ap.add_argument('--phase',choices=PHASES,required=True);ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1);ap.add_argument('--offline',action='store_true')
    args=ap.parse_args()
    if not 0<=args.shard<args.shards<=32:ap.error('Invalid partition')
    audit_file=json.loads((Path(args.data)/'audit.json').read_text())
    if audit_file['data_hash']!=DATA_HASH:raise RuntimeError('Dataset contract changed')
    rows=sum((load_rows(args.data,s) for s in PHASES[args.phase]),[])
    # Whole families remain in one shard; label independent and deterministic.
    families=sorted({r['group'] for r in rows})
    selected={g for i,g in enumerate(families) if i%args.shards==args.shard}
    rows=[r for r in rows if r['group'] in selected]
    root=Path(args.out)/args.phase;root.mkdir(parents=True,exist_ok=True)
    path=root/f'shard-{args.shard:02d}.npz'
    if path.exists():raise FileExistsError('Do not overwrite cache evidence')
    if not rows:raise RuntimeError('Empty partition: reduce shard count')
    try:
        model,tokenizer,hashes=load_model(args.offline)
        extract(rows,model,tokenizer,hashes,path,args.phase,args.shard,args.shards)
    except Exception as e:
        save_json(root/f'failure-{args.shard:02d}.json',{'complete':False,'phase':args.phase,
            'exception':type(e).__name__,'message':str(e)[:1000],'model_results_available':False})
        raise

if __name__=='__main__':main()
