"""Bounded same-checkpoint pilot: diagnostic -> matched direct/auxiliary LoRA.
Only public newly generated controlled cases are used. No Jev, API inference,
private source data, or baseline parameter replacement. Results are not production claims.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, time, resource
from pathlib import Path
from semantic_data_v1 import dataset, tests, digest, prompt_task, evaluate, ASTS, TRUTH, OPS

MODEL='Qwen/Qwen3.5-4B';REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PAD=384;SEED=17;RANK=8;STEPS=108
EXPECTED={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def emit(kind,**kw):print(json.dumps({'kind':kind,**kw},sort_keys=True,allow_nan=False),flush=True)
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(rows,records):
    idx={r['id']:r for r in rows};out={}
    for split in sorted({r['split'] for r in rows}):
        ss=[p for p in records if idx[p['id']]['split']==split];rr=[r for r in rows if r['split']==split]
        assert len(ss)==len(rr)
        out[split]={'n':len(rr),'correct':sum(p['choice']==idx[p['id']]['answer'] for p in ss),'ties':sum(p['choice'] is None for p in ss),'nll':sum(-math.log(max(p['probabilities'][idx[p['id']]['answer']],1e-30)) for p in ss)/len(rr),'wrong_object_actions':sum(p['choice'] is not None and idx[p['id']]['order'][p['choice']]>=3 for p in ss)}
    return out

class Runtime:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        from huggingface_hub import snapshot_download
        self.torch=torch;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
        self.snapshot=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4)
        hashes={p.name:filehash(p) for p in Path(self.snapshot).glob('*.safetensors')}
        if hashes!=EXPECTED:raise RuntimeError('Checkpoint file mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(self.snapshot,trust_remote_code=False,local_files_only=True)
        t=time.perf_counter()
        self.model,info=Qwen3_5ForConditionalGeneration.from_pretrained(self.snapshot,dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False,local_files_only=True,output_loading_info=True)
        issues={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
        if issues:raise RuntimeError(str(issues))
        self.model.eval();self.head=self.model.get_output_embeddings();self.capture={};self.layers={}
        def hook(module,args):self.capture['hidden']=args[0][:,-1,:]
        self.head.register_forward_pre_hook(hook)
        self.receipt={'model':MODEL,'revision':REV,'hashes':hashes,'load_seconds':time.perf_counter()-t,'parameters':sum(p.numel() for p in self.model.parameters()),'torch':torch.__version__}
        emit('loaded',**self.receipt)

    def encode(self,task,pad=False):
        content=task['content']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(task['options']))+'\nAnswer with exactly one option letter, no explanation.'
        text=self.tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)>PAD:raise ValueError(f'Overlength {len(ids)}>{PAD}; never truncate')
        codes=[]
        for i in range(len(task['options'])):
            letter=chr(65+i);code=self.tokenizer.encode(letter,add_special_tokens=False)
            if len(code)!=1 or self.tokenizer.encode(text+letter,add_special_tokens=False)!=ids+code:raise ValueError('Invalid answer boundary')
            codes.append(code[0])
        n=len(ids);padding=PAD-n if pad else 0;padid=self.tokenizer.pad_token_id
        if padid is None:padid=self.tokenizer.eos_token_id
        tensors={'input_ids':self.torch.tensor([[padid]*padding+ids]),'attention_mask':self.torch.tensor([[0]*padding+[1]*n])}
        return tensors,codes,n

    def forward(self,encoded):
        import torch.nn.functional as F
        inputs,codes,n=encoded;self.capture.clear()
        out=self.model(**inputs,logits_to_keep=1,use_cache=False)
        h=self.capture['hidden'].float();bias=self.head.bias[codes].float() if self.head.bias is not None else None
        z=F.linear(h,self.head.weight[codes].float(),bias)[0]
        return z

    def predict(self,row,mode='direct'):
        task,gold=prompt_task(row,mode);t=time.perf_counter();enc=self.encode(task)
        with self.torch.no_grad():z=self.forward(enc).float();p=self.torch.softmax(z,dim=0).tolist()
        best=max(p);ties=[i for i,v in enumerate(p) if abs(v-best)<=1e-10];choice=ties[0] if len(ties)==1 else None
        return {'id':row['id'],'mode':mode,'choice':choice,'logits':z.tolist(),'probabilities':p,'seconds':time.perf_counter()-t,'tokens':enc[2],'input_hash':digest(task)}

    def reasoning(self,row):
        task,gold=prompt_task(row,'direct')
        content=task['content']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(task['options']))+'\nReason briefly about the target facts and condition. End with ANSWER: followed by one option letter.'
        text=self.tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        inputs=self.tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False)
        if inputs['input_ids'].shape[1]>PAD:raise ValueError('Reasoning input overlength')
        t=time.perf_counter()
        with self.torch.no_grad():out=self.model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True)
        generated=out[0,inputs['input_ids'].shape[1]:];decoded=self.tokenizer.decode(generated,skip_special_tokens=True)
        matches=re.findall(r'ANSWER:\s*([A-F])\b',decoded)
        eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
        finished=int(generated[-1]) in eos if eos else len(generated)<256
        choice=ord(matches[-1])-65 if matches and finished else None
        return {'id':row['id'],'choice':choice,'complete':bool(finished),'tokens_generated':len(generated),'seconds':time.perf_counter()-t,'response_sha256':hashlib.sha256(decoded.encode()).hexdigest(),'valid_answer':choice is not None}

    def attach_lora(self):
        torch=self.torch
        class LowRank(torch.nn.Module):
            def __init__(self,base):
                super().__init__();self.base=base
                self.a=torch.nn.Parameter(torch.empty(RANK,base.in_features,dtype=torch.float32));self.b=torch.nn.Parameter(torch.zeros(base.out_features,RANK,dtype=torch.float32))
                torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
            def forward(self,x):
                import torch.nn.functional as F
                return self.base(x)+(F.linear(F.linear(x.float(),self.a),self.b)*2).to(x.dtype)
        for p in self.model.parameters():p.requires_grad_(False)
        torch.manual_seed(SEED)
        names=[name for name,m in self.model.named_modules() if re.search(r'\.layers\.(28|29|30|31)\.mlp\.down_proj$',name) and isinstance(m,torch.nn.Linear)]
        if len(names)!=4:raise RuntimeError('Expected four final language FFN projections; found '+str(names))
        for name in names:
            parent,leaf=name.rsplit('.',1);obj=self.model.get_submodule(parent);mod=LowRank(getattr(obj,leaf));setattr(obj,leaf,mod);self.layers[name]=mod
        return {'modules':names,'rank':RANK,'alpha':16,'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'frozen_prefix_layers':28}

    def adapter_state(self):return {name+'.'+part:getattr(mod,part).detach().cpu().contiguous() for name,mod in self.layers.items() for part in ('a','b')}
    def load_adapter(self,state):
        with self.torch.no_grad():
            for name,mod in self.layers.items():
                for part in ('a','b'):getattr(mod,part).copy_(state[name+'.'+part])

def diagnose(out):
    rows=[r for r in dataset() if r['split']=='development'];rt=Runtime();records=[];reason=[]
    for i,row in enumerate(rows):
        ps={mode:rt.predict(row,mode) for mode in ['direct','fact_p','fact_q','rule']}
        pred_facts={a:TRUTH[ps['fact_'+a]['choice']] if ps['fact_'+a]['choice'] is not None else -1 for a in ['p','q']}
        ro=ps['rule']['choice'];pred_ast=ASTS[ro] if ro is not None else None
        truths={'C_gold':row['truth'],'D_facts':evaluate(row['ast'],pred_facts),'E_rule':evaluate(pred_ast,row['facts']) if pred_ast else None,'F_pipeline':evaluate(pred_ast,pred_facts) if pred_ast else None}
        rec={'id':row['id'],'operator':row['operator'],'gold_truth':row['truth'],'gold_facts':row['facts'],'gold_rule':OPS.index(row['operator']),'predictions':ps,'truths':truths,'A_correct':ps['direct']['choice']==row['answer']}
        records.append(rec);emit('diagnostic_case',**rec)
        if i%3==1:
            b=rt.reasoning(row);b['correct']=b['choice']==row['answer'];reason.append(b);emit('reasoning_case',**b)
    metrics={'n':len(rows),'A_direct':sum(r['A_correct'] for r in records),'B_reasoning_correct':sum(r['correct'] for r in reason),'B_reasoning_n':len(reason),'B_direct_same_subset':sum(r['A_correct'] for i,r in enumerate(records) if i%3==1),'fact_joint':sum(all(TRUTH[r['predictions']['fact_'+a]['choice']]==r['gold_facts'][a] for a in ['p','q']) for r in records),'rule_exact':sum(r['predictions']['rule']['choice']==r['gold_rule'] for r in records)}
    for key in ['C_gold','D_facts','E_rule','F_pipeline']:metrics[key]=sum(r['truths'][key]==r['gold_truth'] for r in records)
    stage='fact' if metrics['D_facts']<metrics['E_rule'] else ('rule' if metrics['E_rule']<metrics['D_facts'] else ('execute' if metrics['A_direct']<metrics['D_facts'] else 'rule'))
    result={'mode':'diagnostic','metrics':metrics,'selected_stage':stage,'cases':records,'reasoning':reason,'runtime':rt.receipt,'privileged_arms':['C_gold','D_facts','E_rule'],'data':tests()}
    save(out/'diagnostic.json',result);emit('diagnostic_complete',metrics=metrics,selected_stage=stage)

def train(arm,stage,out):
    import torch
    from safetensors.torch import save_file,load_file
    rows=dataset();trainrows=[r for r in rows if r['split']=='train'];testrows=[r for r in rows if r['split'] in ('wording','composition')]
    rt=Runtime()
    for r in rows:
        for mode in ['direct','rule','fact_p','fact_q','execute']:rt.encode(prompt_task(r,mode)[0],pad=True)
    base=[rt.predict(r) for r in testrows];save(out/'baseline.json',base);emit('baseline_complete',metrics=stats(testrows,base))
    adapter=rt.attach_lora();initial={k:v.clone() for k,v in rt.adapter_state().items()}
    probe=rt.predict(testrows[0]);assert max(abs(a-b) for a,b in zip(probe['logits'],base[0]['logits']))<2e-5,'Zero-adapter altered model'
    sequence=list(trainrows);random.Random(SEED).shuffle(sequence);schedule=[]
    for i,r in enumerate(sequence):
        second='direct' if arm=='direct' else (('fact_p' if i%2==0 else 'fact_q') if stage=='fact' else stage)
        schedule.extend([(r,'direct'),(r,second)])
    prepared=[(rt.encode(prompt_task(r,m)[0],pad=True),prompt_task(r,m)[1],r['id'],m) for r,m in schedule]
    assert len(prepared)==STEPS
    opt=torch.optim.AdamW([p for p in rt.model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    trace=[];start=time.perf_counter()
    for step,(enc,target,rid,mode) in enumerate(prepared,1):
        opt.zero_grad(set_to_none=True);z=rt.forward(enc);loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([target]))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_([p for p in rt.model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradient')
        for group in opt.param_groups:group['lr']=2e-4*min(step/8,1.)
        opt.step();entry={'step':step,'source':rid,'task':mode,'loss':float(loss.detach()),'gradient_norm':float(gn),'tokens':enc[2]};trace.append(entry)
        if step==1 or step%12==0:emit('training',arm=arm,elapsed=time.perf_counter()-start,**entry)
    training_seconds=time.perf_counter()-start
    state=rt.adapter_state();changed={k:float((v-initial[k]).abs().max()) for k,v in state.items()}
    if not all(v>0 for k,v in changed.items() if k.endswith('.b')):raise RuntimeError('No learned updates in an adapter')
    save_file(state,str(out/'adapter.safetensors'));save(out/'adapter_config.json',{'base_model':MODEL,'revision':REV,**adapter,'seed':SEED,'arm':arm,'stage':stage,'format':'custom FFN down-projection LoRA, a/b tensor keys; load with semantic_train_v1.Runtime','sha256':filehash(out/'adapter.safetensors')})
    after=[rt.predict(r) for r in testrows]
    rt.load_adapter(initial);rt.load_adapter(load_file(str(out/'adapter.safetensors')))
    reload_error=max(max(abs(a-b) for a,b in zip(rt.predict(testrows[i])['logits'],after[i]['logits'])) for i in [0,35,71])
    if reload_error>2e-5:raise RuntimeError('Reload mismatch')
    training_fit=[rt.predict(r) for r in trainrows[:12]]
    result={'mode':'training','arm':arm,'stage':stage,'seed':SEED,'adapter':adapter,'runtime':rt.receipt,'data':tests(),'baseline':base,'predictions':after,'metrics':stats(testrows,after),'baseline_metrics':stats(testrows,base),'training_trace':trace,'training_fit_sample':stats(trainrows[:12],training_fit),'training_seconds':training_seconds,'updates':len(trace),'presentations':len(trace),'original_scenarios':54,'direct_presentations':sum(t['task']=='direct' for t in trace),'padded_positions':PAD*len(trace),'nonpadding_tokens':sum(t['tokens'] for t in trace),'fixed_forward_shape':[1,PAD],'max_parameter_changes':changed,'reload_max_logit_error':reload_error,'reload_qualification':'Three cases, saved adapter on same in-memory frozen backbone, not independent retraining','adapter_sha256':filehash(out/'adapter.safetensors'),'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['one seed, one 108-update pilot','only final four FFN down projections adapted; not full fine-tuning','synthetic 3-context diagnostic, no independently reviewed real language','same padded transformer shapes/updates, differing useful nonpadding tokens and candidate counts','no pretrained weights included or promoted']}
    save(out/'training.json',result);emit('training_complete',arm=arm,stage=stage,metrics=result['metrics'],baseline_metrics=result['baseline_metrics'],updates=STEPS,training_seconds=training_seconds,adapter_sha256=result['adapter_sha256'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['diagnostic','direct','auxiliary'],required=True);ap.add_argument('--stage',choices=['fact','rule','execute'],default='rule');ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',{'model':MODEL,'revision':REV,'mode':a.mode,'stage':a.stage,'seed':SEED,'updates':STEPS,'rank':RANK,'padding':PAD,'source_hash':filehash(__file__),'data':tests()});emit('protocol',**json.loads((out/'protocol.json').read_text()))
    try:
        if a.mode=='diagnostic':diagnose(out)
        else:train(a.mode,a.stage,out)
    except Exception as e:
        save(out/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
