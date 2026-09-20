"""Exact terminal-FFN cache experiment. Development/training data only.
A frozen prefix is evaluated once; updates touch only the final FFN down projection.
Cached and full-network logits are compared before/after training. No JevBench data.
This is a learning diagnostic, not a promoted general model or novel algorithm.
"""
from __future__ import annotations
import hashlib,json,math,random,time
from pathlib import Path

SEED=41
RANK=8
SCALE=2.0
TRAIN_STEPS=800
MICRO_STEPS=400
EXPECTED_DATA='2daff7f09aec40b92b5edc76241d6c3561dd0597d278e10f46bc4f9714590501'
SYSTEM=('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
        'Respond with only its uppercase letter, with no explanation or reasoning.')


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False));tmp.replace(path)
def emit(kind,**v):print(json.dumps({'kind':kind,**v},sort_keys=True,allow_nan=False),flush=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prompt(row):
    from semantic_data_v1 import OUTCOMES,POLICY
    payload={'evidence':row['evidence'],'criterion':row['rule']+'\n'+POLICY,
       'options':[{'letter':chr(65+i),'description':OUTCOMES[j]} for i,j in enumerate(row['order'])]}
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
def encode(rt,row):
    t=rt.tokenizer;text=t.apply_chat_template(prompt(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=t.encode(text,add_special_tokens=False);codes=[]
    if not ids or len(ids)>768:raise ValueError('Input outside tested bound, never truncate')
    for i in range(6):
        letter=chr(65+i);token=t.encode(letter,add_special_tokens=False)
        if len(token)!=1 or t.encode(text+letter,add_special_tokens=False)!=ids+token:raise ValueError('Answer boundary')
        codes.append(token[0])
    return ({'input_ids':rt.torch.tensor([ids]),'attention_mask':rt.torch.ones((1,len(ids)),dtype=rt.torch.long)},codes,len(ids))

def scores(z,gold):
    import torch
    p=z.float().softmax(-1);target=torch.tensor(gold)
    return {'n':len(gold),'correct':int((p.argmax(-1)==target).sum()),'nll':float(torch.nn.functional.cross_entropy(z.float(),target)),
       'brier':float(((p-torch.nn.functional.one_hot(target,z.shape[-1]))**2).sum(-1).mean())}

def main():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file,load_file
    from semantic_train_v1 import Runtime,MODEL,REV
    from semantic_data_v1 import dataset,tests,digest
    out=Path('learning-v3');out.mkdir(exist_ok=False)
    integrity=tests()
    if integrity['data_hash']!=EXPECTED_DATA:raise RuntimeError('Original developmental source changed')
    train=[r for r in dataset() if r['split']=='train']
    test=[r for r in dataset() if r['split'] in ('wording','composition')]
    micro=[]
    for meaning in range(3):micro += sorted([i for i,r in enumerate(train) if r['action']==meaning],key=lambda i:digest({'micro':train[i]['id'],'seed':SEED}))[:4]
    micro=sorted(micro)
    protocol={'name':'terminal-cache-learning-v3','seed':SEED,'rank':RANK,'scale':SCALE,'full_steps':TRAIN_STEPS,'micro_steps':MICRO_STEPS,
      'learning_rate':.0005,'batch':'full selected population','weight_decay':0,'gradient_clip':1.,'data':integrity,'model':MODEL,'revision':REV,
      'train_n':len(train),'micro_ids':[train[i]['id'] for i in micro],'regression_n':len(test),'regression_exposed':True,
      'no_jevbench_inputs':True,'source_sha256':sha(__file__),
      'limitations':['Last FFN only; not multi-layer or full finetuning','Controlled synthetic templates, not independent human data','Previously exposed wording/composition rows are regression only','Changed interface, placement and optimization; not single-factor causal attribution to padding']}
    save(out/'protocol.json',protocol);emit('protocol',**protocol)
    rt=Runtime();torch=rt.torch
    for p in rt.model.parameters():p.requires_grad_(False)
    names=[n for n,m in rt.model.named_modules() if n.endswith('.layers.31.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
    if len(names)!=1:raise RuntimeError('Unexpected architecture')
    name=names[0];module=rt.model.get_submodule(name);layer=rt.model.get_submodule(name.rsplit('.mlp.',1)[0]);body=rt.model.get_submodule(name.split('.layers.')[0]);norm=body.norm
    captured={}
    def pre(key):
        def f(m,args):captured[key]=args[0][:,-1,:].detach().clone()
        return f
    def post(m,args,y):captured['y']=y[:,-1,:].detach().clone()
    hooks=[layer.post_attention_layernorm.register_forward_pre_hook(pre('r')),module.register_forward_pre_hook(pre('x')),module.register_forward_hook(post),norm.register_forward_pre_hook(pre('norm_in'))]
    entries=[];rows=train+test;start=time.perf_counter();codes=None
    with torch.no_grad():
        for i,row in enumerate(rows):
            captured.clear();enc=encode(rt,row);z=rt.forward(enc).detach().clone()
            if codes is not None and codes!=enc[1]:raise RuntimeError('Answer slots changed')
            codes=enc[1]
            if not torch.equal(captured['r']+captured['y'],captured['norm_in']):raise RuntimeError('Terminal residual reconstruction fails')
            rec={'id':row['id'],'input_sha256':digest(prompt(row)),'tokens':enc[2],'baseline':z.tolist()}
            entries.append({**captured,'z':z,'record':rec})
            if (i+1)%18==0:emit('cache_progress',completed=i+1,total=len(rows),seconds=time.perf_counter()-start)
    cache_seconds=time.perf_counter()-start
    for h in hooks:h.remove()
    x=torch.cat([e['x'] for e in entries]);r=torch.cat([e['r'] for e in entries]);y=torch.cat([e['y'] for e in entries]);z0=torch.stack([e['z'] for e in entries])
    w=rt.head.weight[codes].float().detach();bias=rt.head.bias[codes].float().detach() if rt.head.bias is not None else None
    class TerminalLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Parameter(torch.empty(RANK,module.in_features));self.b=torch.nn.Parameter(torch.zeros(module.out_features,RANK))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def delta(self,v):return (F.linear(F.linear(v.float(),self.a),self.b)*SCALE).to(v.dtype)
        def forward(self,ii):return F.linear(norm(r[ii]+(y[ii]+self.delta(x[ii]))).float(),w,bias)
    def run_training(tag,ii,steps):
        torch.manual_seed(SEED);adapter=TerminalLoRA();target=torch.tensor([rows[i]['answer'] for i in ii]);opt=torch.optim.AdamW(adapter.parameters(),lr=.0005,weight_decay=0)
        with torch.no_grad():before=adapter(ii);err=float((before-z0[ii]).abs().max())
        if err>2e-5:raise RuntimeError('Cached baseline mismatch '+str(err))
        history=[];timer=time.perf_counter();passing=0;used=0
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True);logits=adapter(ii);loss=F.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            opt.step();used=step
            if step==1 or step%10==0:
                with torch.no_grad():m=scores(adapter(ii),target.tolist())
                entry={'step':step,'gradient_norm':float(grad),**m};history.append(entry)
                with (out/(tag+'-trace.jsonl')).open('a') as f:f.write(json.dumps(entry,sort_keys=True)+'\n')
                passing=passing+1 if m['correct']==len(ii) and m['nll']<.1 else 0
                if step%100==0:emit('fit_progress',arm=tag,**entry)
                if passing>=3:break
        elapsed=time.perf_counter()-timer
        save_file({'a':adapter.a.detach(),'b':adapter.b.detach()},str(out/(tag+'.safetensors')))
        save(out/(tag+'-config.json'),{'base_model':MODEL,'revision':REV,'module':name,'rank':RANK,'alpha':16,'scale':SCALE,'prompt_system':SYSTEM,'readout':'fp32 allowed rows','untrained_prefix_frozen':True,'optimization_steps':used,'adapter_sha256':sha(out/(tag+'.safetensors')),'production_promoted':False})
        with torch.no_grad():allz=adapter(list(range(len(rows))));fit=scores(allz[ii],[rows[i]['answer'] for i in ii])
        saved=load_file(str(out/(tag+'.safetensors')))
        if not torch.equal(saved['a'],adapter.a) or not torch.equal(saved['b'],adapter.b):raise RuntimeError('Save corruption')
        def apply(m,args,result):return result+adapter.delta(args[0])
        handle=module.register_forward_hook(apply);checks=[]
        try:
            for i in [ii[0],ii[-1],len(train),len(rows)-1]:
                with torch.no_grad():actual=rt.forward(encode(rt,rows[i]));error=float((actual-allz[i]).abs().max())
                if error>2e-3:raise RuntimeError('Cached/full trained mismatch '+str(error))
                checks.append({'id':rows[i]['id'],'max_logit_difference':error,'same_choice':bool(actual.argmax()==allz[i].argmax())})
        finally:handle.remove()
        result={'arm':tag,'train_metrics':fit,'baseline_train':scores(z0[ii],[rows[i]['answer'] for i in ii]),'steps':used,'optimization_seconds':elapsed,'cache_baseline_error':err,'full_replay_checks':checks,'history':history,'fit_passed':fit['correct']==len(ii) and fit['nll']<.1,'regression':{}}
        for split in ['wording','composition']:
            idx=[i for i,v in enumerate(rows) if v['split']==split];gold=[rows[i]['answer'] for i in idx]
            result['regression'][split]={'baseline':scores(z0[idx],gold),'trained':scores(allz[idx],gold)}
        save(out/(tag+'-results.json'),result)
        save(out/(tag+'-predictions.json'),[{'id':row['id'],'split':row['split'],'target':row['answer'],'baseline_logits':z0[i].tolist(),'trained_logits':allz[i].tolist(),'input_sha256':entries[i]['record']['input_sha256']} for i,row in enumerate(rows)])
        emit('learning_complete',arm=tag,fit_passed=result['fit_passed'],train_metrics=fit,steps=used,optimization_seconds=elapsed,regression=result['regression'],full_replay_checks=checks)
        return result
    micro_result=run_training('micro12',micro,MICRO_STEPS)
    full_result=run_training('full54',list(range(len(train))),TRAIN_STEPS)
    with torch.no_grad():restore=rt.forward(encode(rt,rows[0]));restore_error=float((restore-z0[0]).abs().max())
    if restore_error>2e-5:raise RuntimeError('Frozen baseline changed')
    save(out/'summary.json',{'complete':True,'protocol':protocol,'cache_seconds':cache_seconds,'micro':micro_result,'full':full_result,'restore_error':restore_error,'production_promoted':False,'note':'Learning fit and exposed synthetic regression only; not JevBench or new held-out evidence.'})
    emit('complete',micro_passed=micro_result['fit_passed'],full_passed=full_result['fit_passed'],cache_seconds=cache_seconds,restore_error=restore_error)
if __name__=='__main__':main()
