"""Fixed paired-vs-shuffled curriculum experiment on a frozen pretrained backbone.
Standalone execution is explicit. Artifacts include optimizer and RNG state; no
score-driven retries, automatic deployment, or writes to remote model registries.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, resource, signal, sys, time, traceback, urllib.request
from pathlib import Path
from . import data, readout

MODEL='Qwen/Qwen3.5-4B'
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
HASHES={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
'model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
SEED=7041

def filehash(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def append(path,obj):
    with Path(path).open('a') as f:f.write(data.canonical(obj)+'\n');f.flush();os.fsync(f.fileno())

def download(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read(30_000_001)

def boolq(out):
    """Balanced, length-bounded public train/dev subsets with no shared passages."""
    seen=set();result={};receipts=[]
    for source,n,split in [('train',64,'train'),('dev',32,'retention')]:
        url=f'https://storage.googleapis.com/boolq/{source}.jsonl';raw=download(url)
        (out/f'boolq-{source}.jsonl').write_bytes(raw)
        parsed=[json.loads(x) for x in raw.splitlines() if x.strip()]
        expected=9427 if source=='train' else 3270
        if len(parsed)!=expected:raise ValueError('BoolQ source row count')
        eligible=[(i,r) for i,r in enumerate(parsed) if len(r['passage'])<=900 and len(r['question'])<=160]
        eligible.sort(key=lambda x:data.sha('quality-boolq-v04\0'+x[1]['passage']+'\0'+x[1]['question']))
        selected=[];count=collections.Counter()
        for i,r in eligible:
            p=data.sha(' '.join(r['passage'].lower().split()));y=r['answer']
            if type(y)!=bool:raise ValueError('BoolQ requires actual Boolean target')
            if p in seen or count[y]>=n//2:continue
            seen.add(p);count[y]+=1
            task,target=data.envelope(r['passage'],r['question'],['Yes.','No.'],0 if y else 1,f'boolq:{source}:{i}')
            selected.append({'id':f'boolq:{source}:{i}','group':f'boolq/{split}/{len(selected)//4}',
              'split':split,'family':'boolq_human','task':task,'target':target,'passage_sha256':p,
              'source_row':i,'source_split':source,'source_title':r.get('title'),'human_answer':y})
            if len(selected)==n:break
        if len(selected)!=n:raise ValueError('insufficient qualified BoolQ examples')
        result[split]=selected;receipts.append({'url':url,'sha256':data.sha(raw),'rows':len(parsed),'selected':n,
          'selection':'input-hash order, no repeated normalized passages; <=900 passage chars; balanced true/false; labels used only for declared class balance'})
    return result,receipts

def schedule(rows,mode):
    if mode not in ('grouped','shuffled'):raise ValueError('mode')
    rng=random.Random(SEED);groups=collections.defaultdict(list)
    for r in rows:groups[r['group']].append(r['id'])
    if any(len(g)!=4 for g in groups.values()):raise ValueError('every training minibatch group has four members')
    names=sorted(groups);rng.shuffle(names)
    sequence=[rid for name in names for rid in sorted(groups[name])]
    if mode=='shuffled':random.Random(SEED+1).shuffle(sequence)
    return [sequence[i:i+4] for i in range(0,len(sequence),4)]

def prepare(out):
    corp=data.build(groups_per_family=6) # 192 authored + 64 human = 256 training rows
    for rows in corp.values():data.verify(rows)
    bq,receipts=boolq(out);corp['train']+=bq['train'];corp['retention']=bq['retention']
    # A predeclared oracle-assisted diagnostic, not another independent test set.
    diagnostic=[]
    for family in data.FAMILIES:
        chosen=[r for r in corp['transfer'] if r['family']==family][:2]
        for r in chosen:
            q=dict(r);q['id']=r['id']+'/normalized';q['task']=data.normalized_task(r);q['split']='oracle_diagnostic'
            q['source_task_id']=r['id'];diagnostic.append(q)
    corp['oracle_diagnostic']=diagnostic
    # Old short public slice, retained strictly as exposed regression evidence.
    prior=Path('research/native_training_v03/train.py')
    if prior.is_file():
        import importlib.util
        here=Path.cwd();sys.path.insert(0,str(prior.parent.resolve()))
        spec=importlib.util.spec_from_file_location('v03_training_source',prior)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        public=module.external(out)
        for r in public:r['split']='jev_public_regression'
        corp['jev_public_regression']=public
        sys.path.pop(0)
    else:raise ValueError('required pinned prior JevBench selector unavailable')
    all_ids=[r['id'] for rows in corp.values() for r in rows]
    if len(all_ids)!=len(set(all_ids)):raise ValueError('duplicate id')
    fingerprints={}
    for split,rows in corp.items():
        for r in rows:
            h=data.sha(data.canonical(r['task']))
            if h in fingerprints:raise ValueError('duplicate task across splits')
            fingerprints[h]=split
    save(out/'dataset.json',corp)
    save(out/'sources.json',receipts)
    return corp

class StopRequested(Exception):pass

def train(args):
    import torch,transformers
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);random.seed(SEED)
    out=Path(args.out)
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():raise ValueError('resume needs existing output directory')
    started=time.perf_counter();meta={'status':'starting','mode':args.mode,'model':MODEL,'revision':REV,'seed':SEED,
      'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,
      'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),
      'base_dtype':'bfloat16','adapter_dtype':'float32','rank':4,'scale':2.,'lr':3e-5,
      'batch_size':4,'weight_updates':'all 32 language MLP down projections only',
      'native_thinking':False,'generated_answer_tokens':0,'paid_model_api':False,'automatic_promotion':False}
    save(out/'metadata.json',meta)
    corp=json.loads((out/'dataset.json').read_text()) if args.resume else prepare(out)
    batches=schedule(corp['train'],args.mode);steps=len(batches)
    source_hashes={p.name:filehash(p) for p in Path(__file__).parent.glob('*.py')}
    protocol={'dataset_sha256':filehash(out/'dataset.json'),'source_hashes':source_hashes,
      'model':MODEL,'revision':REV,'mode':args.mode,'seed':SEED,'steps':steps,'batches':batches,
      'checkpoint_policy':'final after one complete pass; no held-out checkpoint selection',
      'max_input_tokens':1024,'training_budget_seconds':5400,'evaluation_budget_seconds':2400,
      'scope':'256-row completed training pilot; not full 512/2048/8192 scaling study; no sealed JevBench'}
    if args.resume:
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('resume protocol mismatch')
    else:save(out/'protocol.json',protocol)
    print('PROTOCOL '+data.canonical({k:v for k,v in protocol.items() if k!='batches'}),flush=True)
    local=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2)
    if {p.name:filehash(p) for p in Path(local).glob('*.safetensors')}!=HASHES:raise ValueError('base shards differ')
    tok=transformers.AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    model,info=transformers.Qwen3_5ForConditionalGeneration.from_pretrained(local,dtype=torch.bfloat16,
      attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('model loading incomplete')
    model.requires_grad_(False);model.eval()
    packets={};truth={}
    for split,rows in corp.items():
        for r in rows:
            messages,ledger=readout.messages(r['task'],'canonical')
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(text,add_special_tokens=False)
            if not 1<=len(ids)<=1024:raise ValueError(f'no truncation: {r["id"]} has {len(ids)} tokens')
            slots=[]
            for row in ledger:
                s=tok.encode(row['marker'],add_special_tokens=False)
                if len(s)!=1 or tok.encode(text+row['marker'],add_special_tokens=False)!=ids+s:raise ValueError('option marker boundary')
                slots.append(s[0])
            labels=[row['label'] for row in ledger]
            packets[r['id']]={'input_ids':ids,'slots':slots,'labels':labels,'prompt_sha256':data.sha(text)}
            truth[r['id']]=labels.index(r['target'])
    save(out/'token_packets.json',packets)
    pad=tok.pad_token_id
    if pad is None:raise ValueError('explicit pad token required')
    def score(keys):
        length=max(len(packets[k]['input_ids']) for k in keys)
        ids=[[pad]*(length-len(packets[k]['input_ids']))+packets[k]['input_ids'] for k in keys]
        mask=[[0]*(length-len(packets[k]['input_ids']))+[1]*len(packets[k]['input_ids']) for k in keys]
        logits=model(input_ids=torch.tensor(ids),attention_mask=torch.tensor(mask),use_cache=False,
                     return_dict=True,logits_to_keep=1).logits[:,-1]
        return [logits[i,packets[k]['slots']].float() for i,k in enumerate(keys)]
    # Readiness is numerical/protocol-only; no test-accuracy gate chooses which models run.
    first=batches[0]
    with torch.inference_mode():
        singleton=[score([k])[0] for k in first];batched=score(first)
    batching_diff=max(float((a.softmax(-1)-b.softmax(-1)).abs().max()) for a,b in zip(singleton,batched))
    save(out/'batching_probe.json',{'ids':first,'max_probability_difference':batching_diff,
      'singleton_logits':[x.tolist() for x in singleton],'batched_logits':[x.tolist() for x in batched]})
    if batching_diff>0.05:raise ValueError('batched execution materially differs from singleton; stop before training')
    class Factors(torch.nn.Module):
        def __init__(self,m):
            super().__init__();self.a=torch.nn.Parameter(torch.empty(4,m.in_features));self.b=torch.nn.Parameter(torch.zeros(m.out_features,4));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):return (2*torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)).to(x.dtype)
    targets=[(n,m) for n,m in model.named_modules() if n.startswith('model.language_model.layers.') and n.endswith('.mlp.down_proj') and isinstance(m,torch.nn.Linear)]
    if len(targets)!=32:raise ValueError('expected 32 projections')
    factors=torch.nn.ModuleList([Factors(m) for n,m in targets]);enabled=[True];hooks=[]
    for (_,module),factor in zip(targets,factors):
        hooks.append(module.register_forward_hook(lambda m,a,y,f=factor:y+f(a[0]) if enabled[0] else y))
    with torch.inference_mode():zero=score(first)
    if max(float((a-b).abs().max()) for a,b in zip(zero,batched))>1e-4:raise ValueError('zero adapter mismatch')
    optimizer=torch.optim.AdamW(factors.parameters(),lr=3e-5,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/4)*(.2+.8*.5*(1+math.cos(math.pi*min(s,steps)/steps))))
    done=0
    def checkpoint(step):
        path=out/'resume.pt';temp=out/'resume.tmp'
        state={'step':step,'factors':factors.state_dict(),'optimizer':optimizer.state_dict(),
          'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
          'protocol_sha256':filehash(out/'protocol.json'),'dataset_sha256':filehash(out/'dataset.json')}
        torch.save(state,temp)
        with temp.open('rb') as f:os.fsync(f.fileno())
        temp.replace(path)
        save_file({k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()},str(out/'adapter-latest.safetensors'))
        save(out/'checkpoint.json',{'step':step,'resume_sha256':filehash(path),'adapter_sha256':filehash(out/'adapter-latest.safetensors')})
    if args.resume:
        c=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        if c['protocol_sha256']!=filehash(out/'protocol.json') or c['dataset_sha256']!=filehash(out/'dataset.json'):raise ValueError('checkpoint lineage')
        factors.load_state_dict(c['factors']);optimizer.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
        torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng']);done=c['step']
        existing=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()] if (out/'training.jsonl').exists() else []
        if len(existing)!=done:raise ValueError('journal/checkpoint mismatch; manual reconciliation required')
    if not args.resume:
        save_file({k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()},str(out/'adapter-initial.safetensors'))
    checkpoint(done)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    stop=[False]
    def request_stop(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    train_start=time.perf_counter()
    try:
        for i in range(done,steps):
            if stop[0] or time.perf_counter()-train_start>5400:raise StopRequested('training budget or termination, saved at whole-minibatch boundary')
            model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();keys=batches[i];zs=score(keys)
            loss=sum(torch.nn.functional.cross_entropy(z[None],torch.tensor([truth[k]])) for k,z in zip(keys,zs))/len(keys)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward()
            if any(p.grad is not None for p in model.parameters()):raise ValueError('base acquired gradient')
            norms=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in factors]
            if any(x is None or not math.isfinite(x) for x in norms):raise ValueError('missing or nonfinite gradients')
            if i==0 and any(x<=0 for x in norms):raise ValueError('zero initial block gradient')
            g=float(torch.nn.utils.clip_grad_norm_(factors.parameters(),1.0));optimizer.step();scheduler.step();done=i+1
            append(out/'training.jsonl',{'step':done,'ids':keys,'loss':float(loss.detach()),'gradient_norm':g,
              'B_gradient_norms':norms,'lr':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-tick})
            checkpoint(done)
            print('TRAIN '+data.canonical({'mode':args.mode,'step':done,'loss':float(loss.detach()),'seconds':time.perf_counter()-tick}),flush=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in factors.state_dict().items()},str(out/'adapter-final.safetensors'))
        save(out/'training-complete.json',{'steps':done,'examples':sum(len(x) for x in batches),'training_seconds':time.perf_counter()-train_start,'parameters':sum(p.numel() for p in factors.parameters())})
    except BaseException:
        checkpoint(done)
        raise
    finally:
        model.eval();model.gradient_checkpointing_disable();model.disable_input_require_grads()
    # Singletons isolate inference from training-batch padding; no checkpoint selection on these data.
    eval_start=time.perf_counter();count=0
    for split,rows in corp.items():
        if split=='train':continue
        for r in rows:
            if time.perf_counter()-eval_start>2400:raise TimeoutError('separate evaluation budget')
            arms=('base','adapted') if int(data.sha(r['id'])[:6],16)%2 else ('adapted','base')
            for arm in arms:
                enabled[0]=arm=='adapted';tick=time.perf_counter()
                with torch.inference_mode():z=score([r['id']])[0];p=z.softmax(-1).tolist()
                c=packets[r['id']];record={'id':r['id'],'split':split,'arm':arm,'labels':c['labels'],
                  'logits':z.tolist(),'probs':dict(zip(c['labels'],p)),'predicted':c['labels'][max(range(len(p)),key=p.__getitem__)],
                  'prompt_sha256':c['prompt_sha256'],'seconds':time.perf_counter()-tick}
                append(out/'predictions.jsonl',record);count+=1
    enabled[0]=False
    with torch.inference_mode():restored=score(first)
    restoration=max(float((a-b).abs().max()) for a,b in zip(restored,batched))
    if restoration>1e-4:raise ValueError('base restoration mismatch')
    meta.update(status='completed',completed_steps=done,training_examples=len(corp['train']),evaluation_forwards=count,
      adapter_parameters=sum(p.numel() for p in factors.parameters()),adapter_sha256=filehash(out/'adapter-final.safetensors'),
      base_restoration_max_difference=restoration,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      wall_seconds=time.perf_counter()-started,evaluation_seconds=time.perf_counter()-eval_start,targets=[n for n,m in targets])
    save(out/'metadata.json',meta)
    print('COMPLETED '+data.canonical(meta),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['grouped','shuffled'],required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:train(args)
    except BaseException as exc:
        out=Path(args.out)
        if out.exists():save(out/'FAILED.json',{'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
