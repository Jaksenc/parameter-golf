"""Full-network reload verification of cached evaluations, with hooks restored.
Two prespecified transfer cases per trained checkpoint; not a full replication.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from .cache import load_model,encode,terminal_modules
from .io import filehash,load_rows,save_json
from .train import load_adapter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--training',default='runs/training')
    ap.add_argument('--evaluation',default='runs/transfer');ap.add_argument('--offline',action='store_true');args=ap.parse_args()
    train=Path(args.training);root=Path(args.evaluation)
    result=json.loads((root/'results.json').read_text());selection=json.loads((train/'selection.json').read_text())
    if result['selection_sha256']!=filehash(train/'selection.json'):raise ValueError('Selection mismatch')
    rows=load_rows(args.data,'transfer_wording')+load_rows(args.data,'transfer_composition')
    # Fixed positions, independent of whether earlier predictions were correct.
    chosen=[rows[0],rows[-1]]
    records=[json.loads(s) for s in (root/'predictions.jsonl').read_text().splitlines()]
    lookup={(r['model'],r['id']):r for r in records}
    model,tokenizer,hashes=load_model(args.offline)
    name,module,_,_=terminal_modules(model);head=model.get_output_embeddings();capture={}
    def pre(m,args):capture['h']=args[0][:,-1,:]
    hook=head.register_forward_pre_hook(pre);checks=[]
    def score(row):
        inp,slots,_,_=encode(tokenizer,row)
        with torch.no_grad():
            model(**inp,use_cache=False,logits_to_keep=1)
            return F.linear(capture['h'].float(),head.weight[slots].float(),
                            head.bias[slots].float() if head.bias is not None else None)[0]
    try:
        base={r['id']:score(r) for r in chosen}
        for spec in selection['results']:
            tag=f"{spec['arm']}-{spec['seed']}";path=train/tag/'selected.npz'
            if filehash(path)!=spec['selected_sha256']:raise ValueError('Checkpoint hash mismatch')
            adapter=load_adapter(path,module.in_features,module.out_features)
            def apply(m,args,y):return y+adapter.delta(args[0])
            active=module.register_forward_hook(apply)
            try:
                for row in chosen:
                    actual=score(row);expected=torch.tensor(lookup[tag,row['id']]['trained_logits'])
                    difference=float((actual-expected).abs().max())
                    if difference>2e-3 or actual.argmax()!=expected.argmax():raise RuntimeError('Full-network replay mismatch')
                    checks.append({'model':tag,'id':row['id'],'max_logit_error':difference})
            finally:active.remove()
        restore=max(float((score(r)-base[r['id']]).abs().max()) for r in chosen)
        if restore>2e-5:raise RuntimeError('Baseline not restored')
    finally:hook.remove()
    save_json(root/'live-replay.json',{'passed':True,'checked_calls':len(checks),'checks':checks,
        'restoration_error':restore,'qualification':'Two fixed examples per checkpoint; new process/base load, not retraining or full evaluation repeat'})
    result['live_checkpoint_replay_completed']=True;save_json(root/'results.json',result)
    print(json.dumps({'passed':True,'checks':len(checks),'restoration_error':restore}))

if __name__=='__main__':main()
