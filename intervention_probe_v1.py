"""Bounded public-only Qwen4B intervention experiment. No training or paid API.
Six views: identity, reverse order, complement, complement+reverse, two rotations.
Natural evidence from new, previously unselected BoolQ/PAWS source groups.
All score functions receive only evidence/question/options, never gold labels.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, platform, resource, time
from pathlib import Path
from collections import Counter
MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SOURCES=[('google/boolq',None,'validation','35b264d03638db9f4ce671b711558bf7ff0f80d5','boolq'),('google-research-datasets/paws','labeled_final','test','161ece9501cf0a11f3e48bd356eaa82de46d6a09','paws')]
VIEW_NAMES=['identity','reverse','complement','complement_reverse','rotation1','rotation2']
LABELS=['True: the requested expression is true.','False: the requested expression is false.','Unknown: the evidence does not determine its truth.']
INV=[1,0,2]

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
def emit(kind,**data):print(json.dumps(dict(kind=kind,**data),sort_keys=True,allow_nan=False),flush=True)
def complement(value):return INV[value]
def make_views(evidence,proposition,order):
    out=[]
    orders=[order,order[::-1],order,order[::-1],order[1:]+order[:1],order[2:]+order[:2]]
    for i,(name,ordered) in enumerate(zip(VIEW_NAMES,orders,strict=True)):
        neg=i in (2,3)
        expr='NOT(P)' if neg else 'P'
        question=('Use only the evidence to judge the requested expression.\nProposition P: '+proposition+'\nRequested expression: '+expr+'.\nNOT swaps true and false; NOT of unknown remains unknown. Choose exactly one truth value.')
        out.append({'name':name,'request':{'evidence':evidence,'question':question,'options':[LABELS[j] for j in ordered]},'to_base':[complement(j) if neg else j for j in ordered]})
    # Exhaustively verify output transport for three-valued truth, not the answer.
    for v in range(3):
        for i,view in enumerate(out):
            actual=complement(v) if i in (2,3) else v
            if view['to_base'][orders[i].index(actual)]!=v:raise ValueError('Invalid intervention transport')
    return out

def data():
    from datasets import load_dataset
    rows=[];receipts=[]
    for repo,config,split,rev,family in SOURCES:
        ds=load_dataset(repo,config,split=split,revision=rev);candidates={}
        for raw in ds:
            if family=='boolq':
                ev=raw['passage'];oldq='Answer this question using the supplied passage: '+raw['question']
                proposition='The answer to this question is yes: '+raw['question']
                value=bool(raw['answer']);group=digest(ev.strip().casefold())
            else:
                ev=raw['sentence1'];oldq='Does this second sentence express the same meaning as the supplied sentence? Second sentence: '+raw['sentence2']
                proposition='This sentence has the same meaning as the evidence: '+raw['sentence2']
                value=bool(raw['label']);group=digest(sorted([ev.strip().casefold(),raw['sentence2'].strip().casefold()]))
            if len(ev.split())>160 or len(oldq.split())>100:continue
            rid=('boolean_qa' if family=='boolq' else 'paraphrase')+'-'+digest({'evidence':ev,'question':oldq})[:16]
            candidates[rid]={'id':rid,'family':family,'group':group,'evidence':ev,'proposition':proposition,'target':0 if value else 1}
        old=sorted(candidates.values(),key=lambda r:digest({'seed':20260920,'id':r['id']}))[:16]
        banned={r['group'] for r in old}
        eligible=sorted((r for r in candidates.values() if r['group'] not in banned),key=lambda r:digest({'seed':'duplex-orbit-v1-20260920','id':r['id']}))
        selected=[];seen=set()
        for r in eligible:
            if r['group'] in seen:continue
            seen.add(r['group']);selected.append(r)
            if len(selected)==64:break
        if len(selected)!=64:raise ValueError('Insufficient independent exact-source groups')
        for i,r in enumerate(selected):
            order=list(itertools.permutations(range(3)))[int(digest({'order':r['id']})[:8],16)%6]
            r['views']=make_views(r.pop('evidence'),r.pop('proposition'),list(order))
            r['split']='development' if i<12 else ('certification' if i<44 else 'evaluation')
            rows.append(r)
        receipts.append({'repo':repo,'revision':rev,'split':split,'family':family,'eligible_rows':len(candidates),'previously_exposed_groups_excluded':len(banned),'selected':64})
    if len(rows)!=128 or len({r['group'] for r in rows})!=128:raise ValueError('Duplicate groups')
    return rows,receipts

def self_test():
    for order in itertools.permutations(range(3)):
        vv=make_views('The value is not reported.','The value exceeds five.',list(order))
        assert len(vv)==6 and all(sorted(v['to_base'])==[0,1,2] for v in vv)
    assert all(complement(complement(i))==i for i in range(3))
    emit('intervention_tests',passed=True,permutations=6,truth_values=3,no_gold_required=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    self_test()
    if a.self_test:return
    if a.shard not in range(6):raise ValueError('Six bounded shards required')
    rows,sources=data()
    emit('protocol',name='duplex-orbit-v1',model=MODEL,revision=REV,rows=128,calls=768,data_hash=digest(rows),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=sources,split_counts=dict(Counter(r['split'] for r in rows)),shard=a.shard,shards=6,
         comparisons={'one_call':['identity'],'two_call_complement':['identity','complement'],'two_call_order':['identity','reverse'],'four_call_orbit':VIEW_NAMES[:4],'four_call_order':['identity','reverse','rotation1','rotation2']},
         limitations=['New to this project, not guaranteed absent from pretraining','128 exact-source groups; near-duplicate passages may remain','Existing external labels; no new independent human adjudication','Both source datasets have binary gold labels; unknown is an available but non-gold outcome','NOT operator transport is certified; model interpretation and dataset labels are not proven','Whole cases partitioned, no label-dependent shard selection','No new weights, no Jev, no paid API'])
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260920)
    snap=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4)
    checks={}
    for p in sorted(Path(snap).glob('*.safetensors')):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
        checks[p.name]=h.hexdigest()
    expected={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
    if checks!=expected:raise ValueError('Checkpoint bytes changed')
    tokenizer=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
    t=time.perf_counter()
    model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(snap,dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    issues={k:v for k,v in loading.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
    if issues:raise ValueError(str(issues))
    model.eval();head=model.get_output_embeddings();capture={}
    def hook(module,args):capture['last']=args[0][:,-1,:].detach()
    head.register_forward_pre_hook(hook)
    emit('loaded',revision=REV,weight_hashes=checks,load_seconds=time.perf_counter()-t,python=platform.python_version(),machine=platform.machine())
    def score(inp):
        letters=['A','B','C']
        content='Evidence:\n'+inp['evidence']+'\n\nQuestion:\n'+inp['question']+'\n\nOptions:\n'+'\n'.join(f'{l}. {s}' for l,s in zip(letters,inp['options'],strict=True))+'\n\nAnswer with exactly one option letter and nothing else.'
        text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False);nt=int(inputs['input_ids'].shape[1])
        if nt>640:raise ValueError('Overlength; no silent truncation')
        ids=[]
        for s in letters:
            z=tokenizer.encode(s,add_special_tokens=False)
            if len(z)!=1 or tokenizer.encode(text+s,add_special_tokens=False)!=inputs['input_ids'][0].tolist()+z:raise ValueError('Ambiguous answer boundary')
            ids.append(z[0])
        capture.clear();out=model(**inputs,logits_to_keep=1,use_cache=False)
        h=capture['last'].float();bias=head.bias[ids].float() if head.bias is not None else None
        z=F.linear(h,head.weight[ids].float(),bias)[0]
        return z.tolist(),nt
    with torch.inference_mode():
        t=time.perf_counter();score(make_views('The sample is blue.','The sample is blue.',[0,1,2])[0]['request']);warm=time.perf_counter()-t
        h=capture['last'].float();full=F.linear(h,head.weight.float(),head.bias.float() if head.bias is not None else None)[0]
        ids=[tokenizer.encode(s,add_special_tokens=False)[0] for s in ['A','B','C']]
        sub=F.linear(h,head.weight[ids].float(),head.bias[ids].float() if head.bias is not None else None)[0]
        error=float((full[ids]-sub).abs().max());del full,h,sub
        if error>1e-4:raise ValueError('Readout numerical mismatch')
        emit('readout_check',max_error=error,warm_seconds=warm)
        count=0;errors=0
        for i,r in enumerate(rows):
            if i%6!=a.shard:continue
            views=[]
            for view in r['views']:
                t=time.perf_counter()
                try:
                    z,nt=score(view['request'])
                    if len(z)!=3 or not all(math.isfinite(x) for x in z):raise ValueError('Nonfinite logits')
                    base=[0.,0.,0.]
                    for position,meaning in enumerate(view['to_base']):base[meaning]=z[position]
                    entry={'view':view['name'],'logits':base,'tokens':nt,'seconds':time.perf_counter()-t,'status':'ok','input_hash':digest(view['request'])}
                except Exception as e:
                    errors+=1;entry={'view':view['name'],'status':'error','error':str(e)[:200],'seconds':time.perf_counter()-t,'input_hash':digest(view['request'])}
                views.append(entry);count+=1
            emit('orbit_case',id=r['id'],index=i,group=r['group'],family=r['family'],split=r['split'],target=r['target'],views=views)
            if errors:raise RuntimeError('Model/adapter failure retained; stop rather than drop')
    expected_calls=sum(i%6==a.shard for i in range(128))*6
    if count!=expected_calls:raise RuntimeError('Incomplete partition')
    emit('complete',shard=a.shard,calls=count,errors=errors,data_hash=digest(rows),revision=REV,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if __name__=='__main__':main()
