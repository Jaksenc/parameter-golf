"""Complete the predeclared confidence-matched control; no new teacher calls.

The parent training implementation is byte-pinned and used unchanged except for
its target provider. Its protocol is retained as a parent, not misrepresented as
this experiment's protocol. All new checkpoint selection is fixed-final.
"""
from __future__ import annotations
import argparse, json, math, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'depth_distill_v1'))
import study as parent

PARENT_HASH = '30396857954c90eb990cb33697dd0824b76949700a592031ad368f6ae52aab1f'
PROTOCOL = {
 'id':'decision0-confidence-control-v1',
 'parent_protocol_sha256':PARENT_HASH,
 'source_run':35666799668,
 'seeds':[73019,73037], 'arms':['confidence_matched'],
 'updates':24, 'rows':96, 'lr':3e-5, 'accumulation':4,
 'target':'Preserve exact reference/teacher mixture mass on correct label; spread residual uniformly across incorrect labels.',
 'selection':'fixed final checkpoint, all seeds reported',
 'public_benchmark':'regression only; never used for training or selection',
 'new_teacher_calls':0, 'official_score':None,
}
original_target = parent.target_for

def confidence_target(r, t, arm):
    if arm != 'confidence_matched': raise ValueError('Wrong experimental arm')
    q, use, weight = original_target(r, t, 'repeat_distill')
    if weight:
        labels=[o['id'] for o in r['options']]
        y=labels.index(r['expected'])
        correct=q[y]
        q=[(1-correct)/(len(q)-1)]*len(q)
        q[y]=correct
    if abs(sum(q)-1)>1e-10 or any(not math.isfinite(x) or x<0 for x in q):
        raise ValueError('Invalid target')
    return q, use, weight

def check_parent():
    if parent.digest(parent.PROTOCOL) != PARENT_HASH:
        raise RuntimeError('Parent protocol is not the completed, repaired v2 study')

def train(args):
    check_parent()
    parent.target_for=confidence_target
    args.arm='confidence_matched'
    parent.train(args)
    p=Path(args.out)
    (p/'protocol.json').rename(p/'parent_protocol.json')
    parent.save(p/'protocol.json',PROTOCOL)
    r=json.loads((p/'receipt.json').read_text())
    r['parent_protocol_hash']=r.pop('protocol_hash')
    r['protocol_hash']=parent.digest(PROTOCOL)
    r['control']='teacher confidence preserved, wrong-class information removed'
    parent.save(p/'receipt.json',r)

def checkpoints(root):
    cps={}
    for p in Path(root).rglob('adapter.safetensors'):
        r=json.loads((p.parent/'receipt.json').read_text())
        if r['protocol_hash']!=parent.digest(PROTOCOL) or parent.filehash(p)!=r['checkpoint_sha256']:
            raise RuntimeError('Control checkpoint mismatch')
        key=f"confidence_matched-{r['seed']}"
        if key in cps: raise RuntimeError('Duplicate checkpoint')
        cps[key]=p
    if set(cps)!={f'confidence_matched-{s}' for s in PROTOCOL['seeds']}:
        raise RuntimeError('Missing control checkpoint')
    return cps

def evaluate(args):
    check_parent()
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False)
    parent.save(p/'protocol.json',PROTOCOL)
    rows=parent.eval_rows();ids=parent.assignment(rows,12)[args.shard]
    cps=checkpoints(args.checkpoint_root)
    rt=parent.Runtime();ad=parent.Adapter(rt,PROTOCOL['seeds'][0])
    parent.save(p/'runtime.json',rt.meta)
    from safetensors.torch import load_file
    states={k:load_file(str(v)) for k,v in cps.items()}
    variants=['base']+sorted(cps);count=0
    with (p/'records.jsonl').open('w') as f:
        for i in ids:
            r=rows[i]
            for variant in variants[i%len(variants):]+variants[:i%len(variants)]:
                ad.enabled=variant!='base'
                if ad.enabled: ad.factors.load_state_dict(states[variant])
                rec=rt.score(r,'baseline');rec['variant']=variant
                rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit')})
                rec['correct']=rec['predicted']==r['expected']
                f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
            parent.emit(phase='control_eval',shard=args.shard,done=count,total=len(ids)*len(variants))
    parent.save(p/'receipt.json',{'records':count,'sha256':parent.filehash(p/'records.jsonl'),'protocol_hash':parent.digest(PROTOCOL),'full_model_inference':True})

def aggregate(args):
    check_parent()
    p=Path(args.out);p.mkdir(parents=True,exist_ok=False)
    records=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
        receipt=json.loads((f.parent/'receipt.json').read_text())
        if receipt['protocol_hash']!=parent.digest(PROTOCOL) or receipt['sha256']!=parent.filehash(f):
            raise RuntimeError('Evaluation artifact mismatch')
        for l in f.read_text().splitlines():
            r=json.loads(l);key=(r['id'],r['variant'])
            if key in seen: raise RuntimeError('Duplicate prediction')
            seen.add(key); records.append(r)
    variants=['base']+[f'confidence_matched-{s}' for s in PROTOCOL['seeds']]
    if seen!={(r['id'],v) for r in parent.eval_rows() for v in variants}:
        raise RuntimeError('Missing prediction; do not compute partial score')
    summary={}
    base={r['id']:r for r in records if r['variant']=='base'}
    for v in variants:
        summary[v]={}
        for split in ('public','held'):
            rs=[r for r in records if r['variant']==v and (r['tier']=='held')==(split=='held')]
            m=parent.metrics(rs)
            m['families']={fam:parent.metrics([r for r in rs if r['family']==fam]) for fam in sorted({r['family'] for r in rs})}
            m['repairs']=sum(r['correct'] and not base[r['id']]['correct'] for r in rs)
            m['regressions']=sum(not r['correct'] and base[r['id']]['correct'] for r in rs)
            summary[v][split]=m
    records.sort(key=lambda r:(r['id'],r['variant']))
    (p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    parent.save(p/'results.json',{'summary':summary,'protocol':PROTOCOL,'records':len(records),'official_score':None})
    parent.save(p/'receipt.json',{'sha256':parent.filehash(p/'records.jsonl'),'protocol_hash':parent.digest(PROTOCOL)})
    parent.emit(phase='control_completed',summary=summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--seed',type=int,choices=PROTOCOL['seeds'],required=True);t.add_argument('--teacher-root',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--shard',type=int,required=True);e.add_argument('--checkpoint-root',required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',required=True)
    for parser in (t,e,a):parser.add_argument('--out',required=True)
    args=ap.parse_args()
    try: {'train':train,'evaluate':evaluate,'aggregate':aggregate}[args.mode](args)
    except Exception as exc:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True)
        parent.save(p/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise
