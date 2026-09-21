"""Pinned, no-training Qwen3.5-4B input-flow ablation; benchmark labels never enter prompts."""
from __future__ import annotations
import argparse, hashlib, inspect, json, math, os, platform, statistics, sys, time
from pathlib import Path
from urllib.request import urlopen
from flow_data import development, digest, visible, FAMILIES
from prompt_variants import messages, encode_checked, VARIANTS
MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BENCH='c6004e008ffba24aec091261ca1a5c02f7324702'
WEIGHTS={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
PROTOCOL={'id':'decision0-input-flow-v1','variants':list(VARIANTS),'model':MODEL,'revision':REV,'benchmark':BENCH,'development_hash':'ccd4534b425fd1f6f1f295086dc1d6f3e9ebc8c9a8d09d3230d339d9ad75387a','readout':'BF16 backbone; native normalized final hidden state; FP32 selected output projection','max_tokens':8192,'selection':'mean of five deterministic-family accuracies and one minus probability-family TVD; exact ties prefer fewer mean input tokens then declared variant order','benchmark_role':'repeatedly inspected public regression, not model selection or official score','training':False}

def save(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); temp=p.with_suffix(p.suffix+'.tmp'); temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False));temp.replace(p)

def filehash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def softmax(z):
    if not z or any(not math.isfinite(x) for x in z): raise ValueError('Nonfinite scores')
    m=max(z); w=[math.exp(x-m) for x in z]; return [x/sum(w) for x in w]

def prediction(labels,p):
    return max(sorted(labels),key=lambda k:p[labels.index(k)])

def corpus(phase):
    if phase=='dev':
        rows=development()
        if digest(rows)!=PROTOCOL['development_hash']: raise ValueError('Development hash mismatch')
        return rows
    if phase!='bench': raise ValueError(phase)
    rows=[]
    for tier,name in [('easy','easy.jsonl'),('standard','original.jsonl'),('hard','hard.jsonl')]:
        url=f'https://raw.githubusercontent.com/fstandhartinger/jevbench/{BENCH}/datasets/public/{name}'
        raw=urlopen(url,timeout=60).read().decode()
        for line in raw.splitlines():
            if not line.strip(): continue
            r=json.loads(line);q=r['question'];crit=q.get('criteria');kind=q['type']
            if kind=='noul': opts=[{'id':lab,'description':key+': '+(crit or {}).get(key,f'The proposition is {key}.')} for key,lab in [('true','yes'),('false','no')]]
            elif kind=='choice': opts=[{'id':k,'description':k+': '+(v or k)} for k,v in crit.items()]
            elif kind=='score': opts=[{'id':str(i),'description':str(i)+': '+str(v)} for i,v in enumerate(crit)]
            else: raise ValueError(kind)
            gold=r.get('provenance',{}).get('gold_probs')
            if gold and kind=='noul': gold={('yes' if k=='true' else 'no' if k=='false' else k):v for k,v in gold.items()}
            rows.append({'id':r['id'],'state':r['state'],'question':q['instructions'],'options':opts,'expected':str(r['expected']),'family':r.get('family','unknown'),'tier':tier,'gold_probs':gold,'source':r['id']})
    if len(rows)!=231 or len({r['id'] for r in rows})!=231: raise ValueError('Public corpus changed')
    return rows

def assigned_indices(rows,nshards):
    # Greedy assignment by serialized input size; all variants of a case stay on one VM.
    cost=[sum(len(messages(visible(r),v)[1]['content']) for v in VARIANTS) for r in rows]
    loads=[0]*nshards; buckets=[[] for _ in range(nshards)]
    for i in sorted(range(len(rows)),key=lambda i:(-cost[i],i)):
        b=min(range(nshards),key=lambda j:(loads[j],j)); buckets[b].append(i);loads[b]+=cost[i]
    return [sorted(x) for x in buckets]

class Runtime:
    def __init__(self):
        import torch, transformers
        from huggingface_hub import snapshot_download
        self.torch=torch;torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(7391)
        if platform.machine() not in ('aarch64','arm64'): raise RuntimeError('This study pins ARM execution')
        local=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2)
        actual={p.name:filehash(p) for p in Path(local).glob('*.safetensors')}
        if actual!=WEIGHTS: raise RuntimeError('Weight hash mismatch')
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        cls=transformers.Qwen3_5ForConditionalGeneration
        self.model,info=cls.from_pretrained(local,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
        bad={k:v for k,v in info.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
        if bad: raise RuntimeError(str(bad))
        self.model.eval(); self.model.requires_grad_(False);self.head=self.model.get_output_embeddings();self.hidden=None
        def capture(m,args):
            self.hidden=args[0][:,-1,:].detach() if args[0].ndim==3 else args[0].detach()
        self.hook=self.head.register_forward_pre_hook(capture)
        if 'logits_to_keep' not in inspect.signature(self.model.forward).parameters: raise RuntimeError('Missing last-position logit support')
        self.meta={'model':MODEL,'revision':REV,'hashes':actual,'torch':torch.__version__,'transformers':transformers.__version__,'machine':platform.machine(),'threads':4,'parameters':sum(p.numel() for p in self.model.parameters()),'training':False,'git_sha':os.getenv('GITHUB_SHA'),'python':sys.version}

    def score(self,row,variant):
        torch=self.torch;start=time.perf_counter();enc=encode_checked(self.tokenizer,visible(row),variant,PROTOCOL['max_tokens']);slots=enc['answer_token_ids'];ids=enc['input_ids']
        x=torch.tensor([ids],dtype=torch.long);mask=torch.ones_like(x);self.hidden=None;fw=time.perf_counter()
        with torch.inference_mode():
            out=self.model(input_ids=x,attention_mask=mask,use_cache=False,return_dict=True,logits_to_keep=1)
            if self.hidden is None: raise RuntimeError('No output hidden state captured')
            w=self.head.weight[slots].float();b=self.head.bias[slots].float() if self.head.bias is not None else None
            z=torch.nn.functional.linear(self.hidden.float(),w,b)[0].tolist();native=out.logits[0,-1,slots].float().tolist()
        forward=time.perf_counter()-fw;p=softmax(z);labels=[o['id'] for o in row['options']];pred=prediction(labels,p)
        result={'id':row['id'],'variant':variant,'ok':True,'labels':labels,'logits':z,'probabilities':p,'native_logits':native,'predicted':pred,'input_tokens':len(ids),'prompt_sha256':hashlib.sha256(self.tokenizer.apply_chat_template(messages(visible(row),variant),tokenize=False,add_generation_prompt=True,enable_thinking=False).encode()).hexdigest(),'forward_seconds':forward,'request_seconds':time.perf_counter()-start}
        del out;self.hidden=None
        return result

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',PROTOCOL)
    selection=None
    if args.phase=='bench':
        selection=json.loads(Path(args.selection).read_text())
        if selection['protocol_hash']!=digest(PROTOCOL) or selection['n']!=72 or selection['selected'] not in VARIANTS:raise ValueError('Invalid locked development selection')
        save(out/'selection.json',selection)
    rows=corpus(args.phase);ids=assigned_indices(rows,args.nshards)[args.shard]
    save(out/'assignment.json',{'phase':args.phase,'shard':args.shard,'nshards':args.nshards,'corpus_hash':digest(rows),'ids':[rows[i]['id'] for i in ids]})
    rt=Runtime();save(out/'runtime.json',rt.meta)
    # Check tokenizer/boundary for every assigned input before any scored forward.
    lengths={}
    for i in ids:
        for v in VARIANTS:lengths[f'{rows[i]["id"]}/{v}']=encode_checked(rt.tokenizer,visible(rows[i]),v,PROTOCOL['max_tokens'])['input_tokens']
    save(out/'token_preflight.json',lengths)
    smoke={'id':'unscored-smoke','state':'The only parcel is marked teal.','question':'Which color is marked?','options':[{'id':'teal','description':'Teal is marked.'},{'id':'orange','description':'Orange is marked.'}]}
    first=rt.score(smoke,'baseline');second=rt.score(smoke,'baseline');delta=max(abs(a-b) for a,b in zip(first['logits'],second['logits']))
    if delta>1e-4:raise RuntimeError(f'Unstable repeat preflight {delta}')
    save(out/'smoke.json',{'repeat_max_logit_difference':delta,'predicted':first['predicted'],'warmup_calls':2})
    failures=0;count=0
    with (out/'records.jsonl').open('w') as f:
        for i in ids:
            r=rows[i]; variants=VARIANTS[i%4:]+VARIANTS[:i%4]
            for v in variants:
                try: rec=rt.score(r,v)
                except Exception as e:
                    rec={'id':r['id'],'variant':v,'ok':False,'error':type(e).__name__+': '+str(e),'probabilities':None,'logits':None,'predicted':None,'input_tokens':lengths[f'{r["id"]}/{v}']};failures+=1
                rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit')});rec['correct']=bool(rec['ok'] and rec['predicted']==r['expected']);rec['phase']=args.phase
                f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
                print(json.dumps({'event':'scored','phase':args.phase,'shard':args.shard,'done':count,'total':len(ids)*4,'id':r['id'],'variant':v,'ok':rec['ok']}),flush=True)
    rt.hook.remove(); save(out/'receipt.json',{'n_records':count,'failed':failures,'records_hash':filehash(out/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'full_model_calls':count+2,'weights_changed':False})
    # Completed individual failures remain explicit data; workflow fails rather than masking them.
    if failures:raise RuntimeError(f'{failures} retained model failures')

def percentile(xs,q):
    xs=sorted(xs)
    if not xs:return None
    x=(len(xs)-1)*q;i=int(x);j=min(i+1,len(xs)-1);return xs[i]+(xs[j]-xs[i])*(x-i)

def tvd(row):
    target=row.get('gold_probs') or row.get('target_probs')
    if not target:return None
    if not row['ok']:return 1.0
    p=dict(zip(row['labels'],row['probabilities']));return sum(abs(p.get(k,0)-v) for k,v in target.items())/2

def summary(records,phase):
    out={}
    for variant in VARIANTS:
        rs=[r for r in records if r['variant']==variant];ok=[r for r in rs if r['ok']];total=len(rs)
        item={'n':total,'failures':total-len(ok),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/total,'mean_tokens':statistics.mean(r['input_tokens'] for r in rs),'p50_request_s':percentile([r['request_seconds'] for r in ok],.5),'p95_request_s':percentile([r['request_seconds'] for r in ok],.95),'families':{}}
        for fam in sorted({r['family'] for r in rs}):
            fr=[r for r in rs if r['family']==fam];item['families'][fam]={'n':len(fr),'correct':sum(r['correct'] for r in fr),'accuracy':sum(r['correct'] for r in fr)/len(fr)}
        item['nll']=sum(-math.log(max(dict(zip(r['labels'],r['probabilities'])).get(r['expected'],0),1e-15)) if r['ok'] else -math.log(1e-15) for r in rs)/total
        if phase=='dev':
            prob=[r for r in rs if r['family']=='probability'];prob_tvd=statistics.mean(tvd(r) for r in prob);item['probability_tvd']=prob_tvd
            item['quality']=(sum(item['families'][fam]['accuracy'] for fam in FAMILIES if fam!='probability')+1-prob_tvd)/len(FAMILIES)
        else:
            item['tiers']={t:{'n':sum(r['tier']==t for r in rs),'correct':sum(r['correct'] for r in rs if r['tier']==t)} for t in ('easy','standard','hard')}
            prs=[r for r in rs if r.get('gold_probs')];item['probability_n']=len(prs);item['probability_tvd']=statistics.mean(tvd(r) for r in prs) if prs else None
        out[variant]=item
    return out

def aggregate(args):
    paths=sorted(Path(args.root).rglob('records.jsonl'));rows=corpus(args.phase);allrows=[];keys=set();receipts=[]
    for p in paths:
        assignment=json.loads((p.parent/'assignment.json').read_text())
        if assignment['phase']!=args.phase:continue
        if assignment['corpus_hash']!=digest(rows):raise ValueError('Corpus hash mismatch')
        receipt=json.loads((p.parent/'receipt.json').read_text())
        if receipt['records_hash']!=filehash(p) or receipt['protocol_hash']!=digest(PROTOCOL):raise ValueError('Artifact mismatch')
        receipts.append({'path':str(p),'sha256':filehash(p)})
        for line in p.read_text().splitlines():
            r=json.loads(line);key=(r['id'],r['variant'])
            if key in keys:raise ValueError('Duplicate result')
            keys.add(key);allrows.append(r)
    expected={(r['id'],v) for r in rows for v in VARIANTS}
    if keys!=expected:raise ValueError(f'Coverage mismatch: got {len(keys)} expected {len(expected)}')
    table=summary(allrows,args.phase);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'summary.json',table);save(out/'protocol.json',PROTOCOL);save(out/'provenance.json',receipts)
    (out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in sorted(allrows,key=lambda r:(r['id'],r['variant']))))
    if args.phase=='dev':
        selected=max(VARIANTS,key=lambda v:(table[v]['quality'],-table[v]['mean_tokens'],-VARIANTS.index(v)))
        result={'selected':selected,'n':len(rows),'protocol_hash':digest(PROTOCOL),'summary':table,'source_blocks':len({r['source'] for r in rows}),'records_hash':filehash(out/'records.jsonl'),'frozen_before_benchmark':True}
        save(out/'selection.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    else:
        selection=json.loads(Path(args.selection).read_text());save(out/'selection.json',selection)
        base={r['id']:r for r in allrows if r['variant']=='baseline'}
        paired={v:{'repairs':sum(r['correct'] and not base[r['id']]['correct'] for r in allrows if r['variant']==v),'regressions':sum(not r['correct'] and base[r['id']]['correct'] for r in allrows if r['variant']==v)} for v in VARIANTS}
        result={'summary':table,'selected_on_development':selection['selected'],'paired':paired,'official_score':None,'weights_changed':False};save(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('run');a.add_argument('--phase',choices=['dev','bench'],required=True);a.add_argument('--shard',type=int,required=True);a.add_argument('--nshards',type=int,required=True);a.add_argument('--selection');a.add_argument('--out',required=True)
    b=sub.add_parser('aggregate');b.add_argument('--phase',choices=['dev','bench'],required=True);b.add_argument('--root',required=True);b.add_argument('--selection');b.add_argument('--out',required=True)
    args=ap.parse_args();run(args) if args.command=='run' else aggregate(args)
