"""Bounded public-only pretrained inference. No training or paid API calls."""
from __future__ import annotations
import argparse,hashlib,json,math,platform,random,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path

def emit(kind,**data):
    print(json.dumps({'kind':kind,**data},ensure_ascii=True,allow_nan=False),flush=True)
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
def diagnostics():
    rows=[]
    for block,noun in enumerate(('record','packet','card','entry')):
        for p in (False,True):
            for q in (False,True):
                evidence=f'The target {noun} is {"amber" if p else "violet"}. Its border is {"dotted" if q else "solid"}. The other {noun} must remain unchanged.'
                for rule in ('amber','not_amber','dotted','not_dotted'):
                    phrase={'amber':'the target is amber','not_amber':'the target is not amber','dotted':'the target has a dotted border','not_dotted':'the target does not have a dotted border'}[rule]
                    answer={'amber':p,'not_amber':not p,'dotted':q,'not_dotted':not q}[rule]
                    definitions=[f'Put the target {noun} in Copper.',f'Put the target {noun} in Indigo.',f'Put the other {noun} in Copper.',f'Put the other {noun} in Indigo.','Observe again without making a change.','Abstain without making a change.']
                    rid=f'counterfactual-{block}-{int(p)}{int(q)}-{rule}'
                    order=list(range(6));random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:16],16)).shuffle(order)
                    rows.append(dict(id=rid,family='counterfactual',evidence=evidence,question=f'Put the target {noun} in Copper if {phrase}; otherwise put it in Indigo. Choose exactly one action. Both actions are permitted. Do not change the other {noun}.',options=[definitions[i] for i in order],target=order.index(0 if answer else 1),semantic_order=order,block=block,p=p,q=q,rule=rule))
    for key in ('evidence','question'):
        groups=defaultdict(list)
        for r in rows:groups[r[key]].append(r['semantic_order'][r['target']])
        assert sum(max(Counter(v).values()) for v in groups.values())==len(rows)//2
    return rows

def public_data(n):
    import requests
    from datasets import load_dataset
    from huggingface_hub import HfApi
    rows=[];sources=[];api=HfApi()
    specs=[('stanfordnlp/snli',None,'test','nli'),('google-research-datasets/paws','labeled_final','test','paraphrase'),('google/boolq',None,'validation','boolean_qa')]
    for repo,config,split,family in specs:
        rev=api.dataset_info(repo).sha;ds=load_dataset(repo,config,split=split,revision=rev);candidates=[]
        for raw in ds:
            if family=='nli':
                if raw['label'] not in (0,1,2):continue
                ev=raw['premise'];question='Using only the supplied passage, classify this claim: '+raw['hypothesis'];options=['The passage supports the claim.','The passage neither establishes nor contradicts the claim.','The passage contradicts the claim.'];target=int(raw['label'])
            elif family=='paraphrase':
                ev=raw['sentence1'];question='Does this second sentence express the same meaning as the supplied sentence? Second sentence: '+raw['sentence2'];options=['The two sentences express different meanings.','The two sentences express the same meaning.'];target=int(raw['label'])
            else:
                ev=raw['passage'];question='Answer this question using the supplied passage: '+raw['question'];options=['No.','Yes.'];target=int(raw['answer'])
            if len(ev.split())>160 or len(question.split())>100:continue
            key=digest({'evidence':ev,'question':question});candidates.append(dict(id=family+'-'+key[:16],family=family,evidence=ev,question=question,options=options,target=target))
        unique={r['id']:r for r in candidates};selected=sorted(unique.values(),key=lambda r:digest({'seed':20260920,'id':r['id']}))[:n]
        if len(selected)!=n:raise RuntimeError('Insufficient eligible data: '+repo)
        rows.extend(selected);sources.append(dict(repo=repo,revision=rev,split=split,eligible=len(unique),selected=n))
    commit='828f8093932c8fe6ca7936c3d2e52903b1c523de';root=f'https://raw.githubusercontent.com/clinc/oos-eval/{commit}/data/'
    def get(name):
        r=requests.get(root+name,timeout=60);r.raise_for_status();return r.json(),hashlib.sha256(r.content).hexdigest()
    raw,sha=get('data_full.json');domains,dsha=get('domains.json');inverse={intent:domain for domain,intents in domains.items() for intent in intents};keys=sorted(domains)
    descriptions={'banking':'Banking, accounts, cards, or transactions.','credit_cards':'Credit card accounts, balances, limits, or transactions.','kitchen_and_dining':'Food, cooking, recipes, or dining.','home':'Household chores, shopping lists, or home devices.','auto_and_commute':'Vehicles, driving, traffic, or commuting.','travel':'Travel planning, flights, hotels, or trips.','utility':'Utility assistant functions such as timers, alarms, math, or conversion.','small_talk':'Social conversation or questions about the assistant.','work':'Work schedules, meetings, employment, or paid time off.','meta':'Control the assistant or manage its settings.'}
    options=[descriptions.get(d,d.replace('_',' ')+'.') for d in keys];candidates=[]
    for text,label in raw['test']:
        if label in inverse:candidates.append(dict(id='routing-'+digest(text)[:16],family='routing',evidence=text,question='Choose the domain that best describes the request.',options=options,target=keys.index(inverse[label])))
    rows.extend(sorted(candidates,key=lambda r:digest({'seed':20260920,'id':r['id']}))[:n]);sources.append(dict(repo='clinc/oos-eval',revision=commit,split='test',data_sha256=sha,domains_sha256=dsha,domain_order=keys,selected=n,task='derived domain routing, not standard CLINC150'))
    return rows,sources

def metrics(rows,predictions):
    groups=defaultdict(list)
    for row,out in zip(rows,predictions,strict=True):groups[row['family']].append((row,out))
    summary={}
    for family,pairs in groups.items():
        losses=[-math.log(max(p['probabilities'][r['target']],1e-30)) for r,p in pairs if p.get('probabilities')]
        summary[family]={'n':len(pairs),'correct':sum(p.get('prediction')==r['target'] for r,p in pairs),'failures':sum(p.get('status')!='ok' for r,p in pairs),'nll_on_successes':sum(losses)/len(losses) if losses else None,'majority_label_count':max(Counter(r['target'] for r,p in pairs).values())}
    d={r['id']:(r,p) for r,p in zip(rows,predictions,strict=True) if r['family']=='counterfactual'};rule_pairs=[];evidence_pairs=[]
    for r,pred in d.values():
        if r['rule'] in ('amber','dotted'):
            rr,pp=d[f"counterfactual-{r['block']}-{int(r['p'])}{int(r['q'])}-not_{r['rule']}"];rule_pairs.append(pred.get('prediction')==r['target'] and pp.get('prediction')==rr['target'])
        axis='p' if 'amber' in r['rule'] else 'q'
        if not r[axis]:
            pv=True if axis=='p' else r['p'];qv=True if axis=='q' else r['q'];rr,pp=d[f"counterfactual-{r['block']}-{int(pv)}{int(qv)}-{r['rule']}"];evidence_pairs.append(pred.get('prediction')==r['target'] and pp.get('prediction')==rr['target'])
    summary['counterfactual_pairs']={'rule_both_correct':sum(rule_pairs),'rule_pairs':len(rule_pairs),'evidence_both_correct':sum(evidence_pairs),'evidence_pairs':len(evidence_pairs)}
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['gliclass','qwen'],required=True);ap.add_argument('--real-per-family',type=int,default=16);a=ap.parse_args()
    import torch
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260920)
    rows=diagnostics();real,sources=public_data(a.real_per_family);rows+=real
    emit('protocol',test='public-pretrained-baseline-v1',cases=len(rows),data_hash=digest(rows),sources=sources,limitations=['small diagnostic, not full public benchmarks','fixed untuned prompt','synthetic fixtures assistant-authored','public benchmark pretraining overlap unknown','no model training','not comparable with private earlier test scores'],python=sys.version,torch=torch.__version__,cpu=platform.processor(),threads=4,hardware='standard public GitHub Actions CPU runner',paid_api_calls=0)
    model_id={'gliclass':'knowledgator/gliclass-large-v3.0','qwen':'Qwen/Qwen3.5-4B'}[a.model];rev=HfApi().model_info(model_id).sha
    snapshot=snapshot_download(model_id,revision=rev,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*','README.md'],max_workers=4)
    emit('model_assets',model=model_id,revision=rev,files={p.name:p.stat().st_size for p in Path(snapshot).iterdir() if p.is_file()})
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    if a.model=='gliclass':
        from gliclass import GLiClassModel,ZeroShotClassificationPipeline
        model,loading=GLiClassModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,output_loading_info=True)
        critical={k:v for k,v in loading.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v};emit('loading_info',details=critical)
        if critical:raise RuntimeError('Checkpoint loading mismatch; no random-head fallback')
        model.eval();pipeline=ZeroShotClassificationPipeline(model,tokenizer,classification_type='single-label',device='cpu',progress_bar=False,max_length=1024)
        def score(row):
            text=pipeline.prepare_input(row['evidence'],row['options'],None,row['question']+'\n');nt=len(tokenizer(text)['input_ids'])
            if nt>1024:raise ValueError('Over token budget: '+str(nt))
            out=pipeline.get_embeddings(texts=[row['evidence']],labels=row['options'],batch_size=1,prompt=row['question']+'\n')[0]
            return torch.as_tensor(out['logits'][:len(row['options'])]).float(),nt,None
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(snapshot,dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
        critical={k:v for k,v in loading.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v};emit('loading_info',details=critical)
        if critical:raise RuntimeError('Checkpoint loading mismatch')
        model.eval()
        def score(row):
            letters=[chr(65+i) for i in range(len(row['options']))]
            content='Evidence:\n'+row['evidence']+'\n\nQuestion:\n'+row['question']+'\n\nOptions:\n'+'\n'.join(f'{l}. {v}' for l,v in zip(letters,row['options'],strict=True))+'\n\nAnswer with exactly one option letter and nothing else.'
            text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False);nt=inputs['input_ids'].shape[1]
            if nt>1024:raise ValueError('Over token budget')
            code_ids=[]
            for letter in letters:
                ids=tokenizer.encode(letter,add_special_tokens=False);combined=tokenizer.encode(text+letter,add_special_tokens=False)
                if len(ids)!=1 or combined!=inputs['input_ids'][0].tolist()+ids:raise ValueError('Ambiguous answer token boundary')
                code_ids.append(ids[0])
            out=model(**inputs,logits_to_keep=1,use_cache=False);full=out.logits[0,-1].float();logits=full[code_ids]
            mass=torch.exp(torch.logsumexp(logits,0)-torch.logsumexp(full,0)).item()
            return logits,nt,mass
    emit('model_loaded',parameters=sum(p.numel() for p in model.parameters()),dtype=str(next(model.parameters()).dtype))
    predictions=[];start=time.perf_counter()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            t=time.perf_counter()
            try:
                logits,nt,mass=score(row)
                if len(logits)!=len(row['options']) or not torch.isfinite(logits).all():raise ValueError('Bad output scores')
                probs=torch.softmax(logits.double(),-1).tolist();mx=max(probs);tied=sum(abs(p-mx)<=1e-10 for p in probs)>1
                out=dict(id=row['id'],status='tie' if tied else 'ok',prediction=None if tied else probs.index(mx),target=row['target'],probabilities=probs,logits=logits.tolist(),tokens=nt,answer_protocol_mass=mass,seconds=time.perf_counter()-t)
            except Exception as e:out=dict(id=row['id'],status='error',prediction=None,target=row['target'],error=str(e)[:300],seconds=time.perf_counter()-t)
            predictions.append(out);emit('prediction',model=a.model,index=i,**out)
            if i==0 and out['status']=='error':raise RuntimeError('First-case integration failure: '+str(out))
    import importlib.metadata as metadata
    emit('summary',model=model_id,revision=rev,data_hash=digest(rows),metrics=metrics(rows,predictions),wall_seconds=time.perf_counter()-start,dependencies={name:metadata.version(name) for name in ('torch','transformers','datasets','huggingface-hub')})
if __name__=='__main__':
    try:main()
    except Exception as e:emit('fatal',error=str(e),traceback=traceback.format_exc());raise
