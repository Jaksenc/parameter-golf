"""All-depth LoRA pilot: matched soft-target versus relational supervision.
The original checkpoint is frozen; new low-rank weights alter all32 MLP outputs.
Two-pass output-gradient replay avoids retaining two full backward graphs. Every
recomputed training logit is checked against the detached loss observation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, time, traceback
from fractions import Fraction
from pathlib import Path
import learn_data as data

SEEDS=(16101,16102,16103)
RANK=4
SCALE=2.0
LR=1e-4
STEPS=64
RELATION_WEIGHT=0.25
MAX_TOKENS=768
ARMS=('supervised','relational')


def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):data.save(p,v)
def emit(**kw):print(json.dumps(kw,sort_keys=True,allow_nan=False),flush=True)
def tensors_digest(values):
    h=hashlib.sha256()
    for key,tensor in sorted(values.items()):
        h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_data(root):
    r=json.loads((root/'records.json').read_text());e=json.loads((root/'relations.json').read_text());m=json.loads((root/'manifest.json').read_text())
    if data.digest(r)!=m['records_sha256'] or data.digest(e)!=m['relations_sha256']:raise ValueError('Data provenance')
    for edge in e:
        i,j=edge['left'],edge['right']
        if r[i]['split']!=edge['split'] or r[j]['split']!=edge['split'] or r[i]['group']!=r[j]['group']:raise ValueError('Invalid relation scope')
        qa=list(map(Fraction,r[i]['target']));qb=list(map(Fraction,r[j]['target']))
        if any(qb[y]!=sum(Fraction(edge['mapping'][y][z])*qa[z] for z in range(len(qa)))+Fraction(edge['delta'][y]) for y in range(len(qb))):raise ValueError('Invalid target relation')
    return r,e,m

def make_factors(torch,model,seed):
    torch.manual_seed(seed)
    class Factor(torch.nn.Module):
        def __init__(self,module):
            super().__init__();self.enabled=True
            self.a=torch.nn.Parameter(torch.empty(RANK,module.in_features,dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(module.out_features,RANK,dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):
            return (SCALE*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
    targets=[(n,m) for n,m in model.named_modules() if n.startswith('model.language_model.layers.') and n.endswith('.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
    if len(targets)!=32:raise RuntimeError('Expected32 MLP targets, got '+str(len(targets)))
    factors=torch.nn.ModuleList([Factor(m) for _,m in targets]);hooks=[]
    for (_,module),factor in zip(targets,factors):
        def hook(_m,args,out,factor=factor):return out+factor(args[0]) if factor.enabled else out
        hooks.append(module.register_forward_hook(hook))
    return targets,factors,hooks


def loss_grads(torch,z,targets,edge,weight):
    from relational_objective import Relation,compute_loss
    detached=[v.detach().clone().requires_grad_(True) for v in z]
    rel=Relation(0,1,torch.tensor(edge['mapping'],dtype=torch.float32),torch.tensor([float(Fraction(v)) for v in edge['delta']],dtype=torch.float32),edge['kind'])
    result=compute_loss(detached,targets,[rel],weight)
    grads=torch.autograd.grad(result.total,detached)
    return grads,{'loss':float(result.total.detach()),'cross_entropy':float(result.supervised.detach()),'relation_loss':float(result.relational.detach())}


def run(root,out,seed,arm):
    import torch
    from safetensors.torch import save_file,load_file
    from reconstruct_v1 import Runtime
    import measure_v14
    if seed not in SEEDS or arm not in ARMS:raise ValueError('Unregistered run')
    out.mkdir(parents=True,exist_ok=False);r,e,m=load_data(root/'learn-prepared')
    rt=Runtime(root/'reconstruction-inputs');fixture=rt.check();model=rt.model
    original_parameters=list(model.parameters());model.requires_grad_(False);model.eval()
    if any(isinstance(x,torch.nn.Dropout) and x.p>0 for x in model.modules()):raise RuntimeError('Replay requires deterministic dropout-free model')
    cache={}
    for i,row in enumerate(r):
        request=row['input'];messages=measure_v14.arm_messages(request,'','semantic_codes')
        prompt=rt.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
        codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(request['labels']))]
        if len(ids)>MAX_TOKENS or any(len(c)!=1 for c in codes) or len({c[0] for c in codes})!=len(codes):raise ValueError('Token contract')
        # Explicit continuation boundaries must preserve code-token interpretation.
        for j,c in enumerate(codes):
            if rt.tokenizer.encode(prompt+chr(65+j),add_special_tokens=False)!=ids+c:raise ValueError('Code boundary')
        cache[i]={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long),'codes':[c[0] for c in codes], 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'tokens':len(ids)}
    calls={'baseline':0,'no_grad_training':0,'backward_replay':0,'after':0,'integrity':0}
    def score(index,phase):
        c=cache[index];rt.head.codes=c['codes'];calls[phase]+=1
        z=model(input_ids=c['input_ids'],attention_mask=c['attention_mask'],logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
        if z.numel()!=len(r[index]['target']) or not torch.isfinite(z).all():raise RuntimeError('Nonfinite logits')
        return z
    def evaluate(phase):
        model.eval();ans=[]
        with torch.inference_mode():
            for n,i in enumerate(m['eval_indices']):
                tick=time.perf_counter();z=score(i,phase);p=torch.softmax(z,-1)
                ans.append({'id':r[i]['input']['id'],'record_index':i,'logits':z.tolist(),'probabilities':p.tolist(),
                  'seconds':time.perf_counter()-tick,'prompt_sha256':cache[i]['prompt_sha256'],'input_sha256':data.digest(r[i]['input']),'tokens':cache[i]['tokens']})
                if n%18==0:emit(stage=phase,seed=seed,arm=arm,completed=n+1,total=len(m['eval_indices']))
        return ans
    start=time.perf_counter();baseline=evaluate('baseline');save(out/'baseline.json',baseline)
    target_modules,factors,hooks=make_factors(torch,model,seed)
    initial_hash=tensors_digest(factors.state_dict());initial=[x.b.detach().clone() for x in factors]
    torch.manual_seed(seed)
    with torch.inference_mode():zero=score(m['eval_indices'][0],'integrity')
    zeroerr=float((zero-torch.tensor(baseline[0]['logits'])).abs().max())
    if zeroerr>1e-4:raise RuntimeError('Zero adapter changes baseline')
    count=sum(x.numel() for x in factors.parameters())
    optimizer=torch.optim.AdamW(factors.parameters(),lr=LR,weight_decay=0.01)
    if {id(p) for p in factors.parameters()} & {id(p) for p in original_parameters}:raise RuntimeError('Optimizer reaches base parameters')
    save(out/'preflight.json',{'runtime':rt.receipt,'scope':'Actual all-depth LoRA training; no generation', 'fixture':fixture,
      'seed':seed,'arm':arm,'trainable_parameters':count,'target_modules':[n for n,_ in target_modules],
      'initial_adapter_sha256':initial_hash,'zero_adapter_error':zeroerr,'rank':RANK,'scale':SCALE,
      'learning_rate':LR,'steps':STEPS,'relation_weight':RELATION_WEIGHT if arm=='relational' else 0.,
      'records_sha256':m['records_sha256'],'source_sha256':filehash(__file__),'max_prompt_tokens':max(x['tokens'] for x in cache.values())})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    order=list(m['train_edges']);random.Random(seed).shuffle(order)
    if len(order)!=STEPS:raise RuntimeError('Incorrect budget')
    save(out/'schedule.json',{'edge_indices':order,'sha256':data.digest(order)})
    train_start=time.perf_counter();first_grad=None;max_replay_error=0.;trace=[]
    for step,edge_id in enumerate(order,1):
        edge=e[edge_id];indices=[edge['left'],edge['right']]
        if any(r[i]['split']!='fit' for i in indices):raise RuntimeError('Test item in optimizer')
        q=[torch.tensor([float(Fraction(v)) for v in r[i]['target']],dtype=torch.float32) for i in indices]
        tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():observed=[score(i,'no_grad_training') for i in indices]
        grads,losses=loss_grads(torch,observed,q,edge,RELATION_WEIGHT if arm=='relational' else 0.)
        for i,expected,g in zip(indices,observed,grads):
            z=score(i,'backward_replay');error=float((z.detach()-expected).abs().max());max_replay_error=max(error,max_replay_error)
            if error>2e-4:raise RuntimeError('Output-gradient replay changed logits: '+str(error))
            torch.autograd.backward(z,g)
        if any(p.grad is not None for p in original_parameters):raise RuntimeError('Frozen base has gradients')
        norms=[float(f.b.grad.norm()) if f.b.grad is not None else None for f in factors]
        if any(v is None or not math.isfinite(v) for v in norms):raise RuntimeError('Invalid adapter gradient')
        if step==1:
            if any(v<=0 for v in norms):raise RuntimeError('Some depth receives no initial gradient')
            first_grad=norms
        norm=float(torch.nn.utils.clip_grad_norm_(factors.parameters(),1.0));optimizer.step()
        if any(not torch.isfinite(p).all() for p in factors.parameters()):raise RuntimeError('Nonfinite update')
        row={'step':step,'edge_index':edge_id,'input_ids':[r[i]['input']['id'] for i in indices],
             **losses,'gradient_norm':norm,'seconds':time.perf_counter()-tick,'replay_error':max_replay_error}
        trace.append(row)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        if step==1 or step%8==0:emit(stage='train',seed=seed,arm=arm,step=step,total=STEPS,**losses,seconds=row['seconds'])
        if step in (1,16,32,64):
            state={k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()}
            save_file(state,str(out/f'adapter-step-{step}.safetensors'))
    train_seconds=time.perf_counter()-train_start
    # Fix the final checkpoint before opening the held-out result comparison.
    state={k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()}
    final_hash=tensors_digest(state);save_file(state,str(out/'adapter.safetensors'))
    if final_hash==initial_hash:raise RuntimeError('No learned parameters')
    bchanges=[float((f.b.detach()-old).abs().max()) for f,old in zip(factors,initial)]
    if any(v<=0 for v in bchanges):raise RuntimeError('Some adapter never changed')
    model.gradient_checkpointing_disable();model.disable_input_require_grads();model.eval()
    after=evaluate('after');save(out/'after.json',after)
    # Actual serialized weight reload and identical test-probe logits.
    reloaded=load_file(str(out/'adapter.safetensors'));factors.load_state_dict(reloaded,strict=True)
    with torch.inference_mode():repeat=score(m['eval_indices'][0],'integrity')
    reload_error=float((repeat-torch.tensor(after[0]['logits'])).abs().max())
    for f in factors:f.enabled=False
    with torch.inference_mode():restored=score(m['eval_indices'][0],'integrity')
    restore_error=float((restored-torch.tensor(baseline[0]['logits'])).abs().max())
    if reload_error>1e-4 or restore_error>1e-4:raise RuntimeError('Checkpoint reload/base restoration failed')
    for h in hooks:h.remove()
    receipt={'status':'complete','seed':seed,'arm':arm,'steps':STEPS,'examples_presented':STEPS*2,
       'unique_fit_examples':len({i for ei in order for i in (e[ei]['left'],e[ei]['right'])}),
       'trained_adapter_parameters':count,'updated_layers':32,'base_parameters_updated':0,
       'initial_adapter_sha256':initial_hash,'final_tensor_sha256':final_hash,'checkpoint_sha256':filehash(out/'adapter.safetensors'),
       'first_B_gradient_norms':first_grad,'max_B_changes':bchanges,'replay_error':max_replay_error,
       'reload_error':reload_error,'base_restoration_error':restore_error,
       'training_seconds':train_seconds,'experiment_seconds_excluding_load':time.perf_counter()-start,
       'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       'calls':calls,'source_sha256':filehash(__file__),'records_sha256':m['records_sha256'],
       'output_contract':'Measured code-softmax at temperature1; answer is argmax; not post-hoc calibrated',
       'limits':'One small same-grammar epoch; no JevBench, no new backbone pretraining, no generated reasoning'}
    save(out/'complete.json',receipt);emit(stage='complete',seed=seed,arm=arm,checkpoint=receipt['checkpoint_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    try:run(Path(a.root),Path(a.out),a.seed,a.arm)
    except Exception as exc:
        Path(a.out).mkdir(parents=True,exist_ok=True)
        save(Path(a.out)/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'source_sha256':filehash(__file__)})
        raise
