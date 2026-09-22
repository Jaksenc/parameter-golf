"""Read-only train-fit diagnosis. Never selects weights or modifies a checkpoint."""
from __future__ import annotations
import argparse, json, traceback
from pathlib import Path
import triplet_study as experiment

PROTOCOL={
    'id':'decision0-conditional-assignment-fit-v1',
    'parent_protocol_hash':experiment.digest(experiment.PROTOCOL),
    'nshards':4,'rows':96,'variants':['base','endpoint','conditional'],
    'purpose':'Diagnose fit on seen training inputs, not generalization or model selection.',
    'updates':0,'official_score':None,
}

def run(args):
    from safetensors.torch import load_file
    parent=experiment.parent
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    parent.save(out/'protocol.json',PROTOCOL)
    experiment.validate_data()
    rows=experiment.corpus('train')
    states={k:load_file(str(p)) for k,p in experiment.checkpoints(args.checkpoint_root).items()}
    runtime=parent.Runtime();ad=parent.Adapter(runtime,experiment.PROTOCOL['seed'])
    parent.save(out/'runtime.json',runtime.meta)
    indices=parent.assignment(rows,PROTOCOL['nshards'])[args.shard]
    count=0
    with (out/'records.jsonl').open('w') as f:
        for i in indices:
            row=rows[i];variants=PROTOCOL['variants']
            for variant in variants[i%3:]+variants[:i%3]:
                ad.enabled=variant!='base'
                if ad.enabled:ad.factors.load_state_dict(states[variant])
                rec=runtime.score(row,'baseline');rec['variant']=variant
                rec.update({k:row.get(k) for k in ('family','tier','source','expected','target_probs','edit','matched','long_context')})
                rec['correct']=rec['predicted']==row['expected']
                f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
            parent.emit(phase='training_fit_diagnostic',shard=args.shard,done=count,total=3*len(indices))
    parent.save(out/'receipt.json',{'records':count,'sha256':parent.filehash(out/'records.jsonl'),'protocol_hash':experiment.digest(PROTOCOL),'checkpoint_updates':0,'training_cases_not_holdout':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-root',required=True);parser.add_argument('--shard',type=int,choices=range(4),required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as exc:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        experiment.parent.save(out/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
