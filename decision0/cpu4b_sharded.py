from __future__ import annotations
import argparse, glob, json, math, random
from pathlib import Path
from types import SimpleNamespace
import torch
import cpu4b_verified as c

NSHARDS=16
CACHE_VERSION='decision0-cache-v3'

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))
def datasets():
    return {
      'train': c.make_data('train',64,11),
      'dev': c.make_data('development',16,23),
      'calibration': c.make_data('calibration',16,31),
      'held': c.make_data('heldout',16,37),
      'bench': c.fetch_bench(),
    }
def visible(name, rows):
    return [c.sem_row_from_bench(r) for r in rows] if name=='bench' else [r['public'] for r in rows]

def cache_mode(args):
    rows=datasets()
    rt=c.RT()
    jobs=[]
    g=0
    for name in ('train','dev','calibration','held','bench'):
        for idx,(r,v) in enumerate(zip(rows[name],visible(name,rows[name]))):
            if g % args.nshards == args.shard:
                jobs.append((name,idx,r['id'],v))
            g+=1
    out=[]
    for k,(name,idx,rid,v) in enumerate(jobs,1):
        z=rt.cache(v)
        out.append({'set':name,'index':idx,'id':rid,'cache':z})
        if k%10==0 or k==len(jobs): c.emit('shard_cache',shard=args.shard,done=k,total=len(jobs))
    eps=float(getattr(rt.norm,'variance_epsilon',getattr(rt.norm,'eps',1e-6)))
    payload={
      'version':CACHE_VERSION,'shard':args.shard,'nshards':args.nshards,
      'records':out,'norm_weight':rt.norm.weight.detach().cpu().contiguous(),
      'norm_eps':eps,'runtime':rt.meta,
      'data_hashes':{k:c.jhash(v) for k,v in rows.items()},
    }
    Path(args.out).mkdir(parents=True,exist_ok=True)
    torch.save(payload,Path(args.out)/'cache.pt')
    dump(Path(args.out)/'receipt.json',{'version':CACHE_VERSION,'shard':args.shard,'records':len(out),'data_hashes':payload['data_hashes'],'runtime':rt.meta})
    rt.close()

class ProxyNorm(torch.nn.Module):
    def __init__(self,w,eps):
        super().__init__(); self.register_buffer('weight',w); self.eps=float(eps)
    def forward(self,x):
        typ=x.dtype; y=x.float(); y=y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+self.eps)
        return self.weight.to(typ)*y.to(typ)

def load_caches(root):
    files=sorted(Path(root).rglob('cache.pt'))
    if len(files)!=NSHARDS: raise RuntimeError(f'expected {NSHARDS} cache shards, got {len(files)}')
    parts=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
    if any(p['version']!=CACHE_VERSION for p in parts): raise RuntimeError('cache version mismatch')
    hashes=parts[0]['data_hashes']; runtime=parts[0]['runtime']; eps=parts[0]['norm_eps']; w=parts[0]['norm_weight']
    if any(p['data_hashes']!=hashes or p['runtime']!=runtime or p['norm_eps']!=eps or not torch.equal(p['norm_weight'],w) for p in parts[1:]): raise RuntimeError('shard metadata mismatch')
    rec={}
    for part in parts:
        for x in part['records']:
            key=(x['set'],x['index'])
            if key in rec: raise RuntimeError('duplicate '+str(key))
            rec[key]=x['cache']
    rows=datasets()
    expected=sum(len(v) for v in rows.values())
    if len(rec)!=expected: raise RuntimeError(f'cache coverage {len(rec)} != {expected}')
    ordered={name:[rec[(name,i)] for i in range(len(rows[name]))] for name in rows}
    return rows,ordered,w,eps,runtime

def make_adapter(caches,w,eps):
    x=caches[0]['x']; y=caches[0]['y']
    down=SimpleNamespace(in_features=x.shape[-1],out_features=y.shape[-1])
    hidden=y.shape[-1]
    return c.Adapter(torch,down,ProxyNorm(w,eps),hidden)

def nll(preds):
    s=0.
    for p in preds:
        t=p['target_probs']; y=max(range(len(t)),key=t.__getitem__); s-=math.log(max(p['probabilities'][y],1e-30))
    return s/len(preds)

def apply_alpha(state,alpha):
    return {k:(v*alpha if k.endswith('.b') else v.clone()) for k,v in state.items()}

def ranking_loss(z,target):
    y=max(range(len(target)),key=target.__getitem__)
    if target[y] < .999: return z.sum()*0
    mask=torch.ones(len(target),dtype=torch.bool); mask[y]=False
    return torch.nn.functional.softplus(z[mask].max()-z[y])

def train(rows,caches,w,eps,arm,out):
    torch.manual_seed(c.SEED); ad=make_adapter(caches['train'],w,eps)
    opt=torch.optim.AdamW(ad.m.parameters(),lr=5e-4,weight_decay=.01)
    rng=random.Random(c.SEED); best=None; best_key=None; history=[]
    for epoch in range(1,17):
        order=list(range(len(rows['train']))); rng.shuffle(order)
        for start in range(0,len(order),32):
            ids=order[start:start+32]; opt.zero_grad(set_to_none=True); loss=0
            for i in ids:
                r=rows['train'][i]; z,h=ad.logits_hidden(caches['train'][i]); t=r['target_probs']; hard=[0.]*len(t); hard[max(range(len(t)),key=t.__getitem__)]=1.
                if arm=='verified':
                    main=c.ce_soft(torch,z,t)
                    extra=c.aux_loss(torch,ad.m,h,r.get('annotations',{}))
                    rank=ranking_loss(z,t)
                    li=main+.2*extra+.15*rank
                else:
                    li=c.ce_soft(torch,z,hard)
                loss=loss+li/len(ids)
            loss.backward(); gn=torch.nn.utils.clip_grad_norm_(ad.m.parameters(),1.); opt.step()
        preds=c.eval_rows(torch,ad,rows['dev'],caches['dev']); st=c.stats(preds); ll=nll(preds); key=(st['accuracy'],-ll)
        history.append({'epoch':epoch,**st,'nll':ll,'gradient_norm':float(gn)})
        c.emit('epoch_v3',arm=arm,**history[-1])
        if best_key is None or key>best_key: best_key=key; best=ad.state()
    # Development-only adapter-strength selection.
    grid=[0.0,0.25,0.5,0.75,1.0]; alpha_rows=[]
    for alpha in grid:
        ad.load(apply_alpha(best,alpha)); ps=c.eval_rows(torch,ad,rows['dev'],caches['dev']); st=c.stats(ps); ll=nll(ps)
        alpha_rows.append({'alpha':alpha,**st,'nll':ll})
    chosen=max(alpha_rows,key=lambda x:(x['accuracy'],-x['nll']))['alpha']
    selected=apply_alpha(best,chosen); ad.load(selected)
    import safetensors.torch as st
    st.save_file(selected,str(Path(out)/f'{arm}.safetensors'))
    dump(Path(out)/f'{arm}-history.json',history); dump(Path(out)/f'{arm}-alpha.json',{'chosen':chosen,'grid':alpha_rows})
    return ad,chosen

def bench_eval(ad,rows,caches):
    return c.eval_rows(torch,ad,rows['bench'],caches['bench'],True)

def native_base(rows,caches):
    out=[]
    for r,cc in zip(rows['bench'],caches['bench']):
        p=torch.softmax(cc['native'].double(),-1).tolist(); ix=max(range(len(p)),key=p.__getitem__); opts=c.sem_row_from_bench(r)['options']; ids=[o['id'] for o in opts]
        out.append({'id':r['id'],'probabilities':p,'logits':cc['native'].tolist(),'pred_index':ix,'tier':r['_tier'],'labels':ids,'predicted':ids[ix],'expected':r['expected'],'correct':ids[ix]==r['expected'],'family':r.get('family')})
    return out

def fit_temperature(preds):
    # Golden-section minimization of CE on off-benchmark calibration targets.
    def loss(logt):
        T=math.exp(logt); s=0.
        for p in preds:
            z=[x/T for x in p['logits']]; m=max(z); ex=[math.exp(x-m) for x in z]; sm=sum(ex); q=[x/sm for x in ex]
            s-=sum(t*math.log(max(v,1e-30)) for t,v in zip(p['target_probs'],q))
        return s/len(preds)
    a,b=math.log(.25),math.log(8.); gr=(math.sqrt(5)-1)/2; x1=b-gr*(b-a);x2=a+gr*(b-a);f1=loss(x1);f2=loss(x2)
    for _ in range(80):
        if f1<f2: b,x2,f2=x2,x1,f1;x1=b-gr*(b-a);f1=loss(x1)
        else: a,x1,f1=x1,x2,f2;x2=a+gr*(b-a);f2=loss(x2)
    return math.exp((a+b)/2)

def scale_probs(records,T):
    out=[]
    for r in records:
        z=[x/T for x in r['logits']]; m=max(z); ex=[math.exp(x-m) for x in z]; sm=sum(ex); p=[x/sm for x in ex]; x=dict(r);x['probabilities']=p;x['pred_index']=max(range(len(p)),key=p.__getitem__)
        if 'labels' in x: x['predicted']=x['labels'][x['pred_index']];x['correct']=x['predicted']==x['expected']
        out.append(x)
    return out

def train_mode(args):
    Path(args.out).mkdir(parents=True,exist_ok=True)
    rows,caches,w,eps,runtime=load_caches(args.cache_root)
    # Reconstructed FP32 base.
    torch.manual_seed(c.SEED); base=make_adapter(caches['bench'],w,eps)
    with torch.no_grad(): base.m.b.zero_()
    fp32=c.eval_rows(torch,base,rows['bench'],caches['bench'],True); native=native_base(rows,caches)
    results={'base_fp32':c.stats(fp32,True),'base_native':c.stats(native,True)}
    dump(Path(args.out)/'base-fp32-benchmark.json',fp32);dump(Path(args.out)/'base-native-benchmark.json',native)
    for arm in ('hard','verified'):
        ad,alpha=train(rows,caches,w,eps,arm,args.out)
        held=c.eval_rows(torch,ad,rows['held'],caches['held']); cal=c.eval_rows(torch,ad,rows['calibration'],caches['calibration']); b=bench_eval(ad,rows,caches)
        T=fit_temperature(cal); scaled=scale_probs(b,T)
        results[arm]={'heldout':c.stats(held),'benchmark_raw':c.stats(b,True),'benchmark_scaled':c.stats(scaled,True),'alpha':alpha,'temperature':T}
        dump(Path(args.out)/f'{arm}-heldout.json',held);dump(Path(args.out)/f'{arm}-benchmark.json',b);dump(Path(args.out)/f'{arm}-benchmark-scaled.json',scaled)
    dump(Path(args.out)/'results.json',{'results':results,'runtime':runtime,'training_uses_jevbench':False,'benchmark_public_only':True,'official_score':None,'cache_version':CACHE_VERSION})
    c.emit('v3_complete',results=results)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True)
    x=sub.add_parser('cache'); x.add_argument('--shard',type=int,required=True);x.add_argument('--nshards',type=int,default=NSHARDS);x.add_argument('--out',required=True)
    t=sub.add_parser('train');t.add_argument('--cache-root',required=True);t.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.mode=='cache': cache_mode(a)
    else: train_mode(a)
if __name__=='__main__': main()
