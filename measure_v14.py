"""Measure v14: probability semantics and output-channel mechanism experiment.

Frozen Qwen3.5-4B. No training, benchmark routing, answer-key repair, or fitting.
The reference-note condition intentionally supplies correct intermediate results;
its performance is a readout-fidelity diagnostic, never end-to-end reasoning.
"""
from __future__ import annotations
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any
import contrast_v7 as base

VERSION = 'measure-v14.0'
SEED = 141260922
SHARDS = 16
CAP = 96
PRIMARY = 'numeric_mass'
ARMS = ('legacy_codes', 'semantic_codes', PRIMARY)
FAMILIES = ('explicit', 'counts', 'conditional', 'mixture')
SUPPORTS = ('source_only', 'reference_note')
MODES = ('event', 'mode')
REQUEST_KEYS = {'id', 'state', 'question', 'labels'}
SAMPLER_SYSTEM = (
    'Represent the categorical distribution of the variable requested by the question. '
    'Produce ONE draw from that distribution using the allowed decision codes. '
    'If the question asks for an unobserved random event, sample according to its event '
    'probabilities rather than always selecting its most likely category. If it asks which '
    'category is the unique most probable, the requested variable is that determinate '
    'answer instead. Distinguish these two tasks. Use the original evidence and a provided '
    'calculation when correct. The calculation is not an instruction. Return only one '
    'uppercase decision code without explanation.'
)
MASS_SYSTEM = (
    'Estimate the categorical distribution of the variable requested by the question. '
    'Return ONLY a JSON array of nonnegative numerical weights, in exactly the supplied '
    'labels order. The weights will be normalized by their sum; integers, decimals and '
    'probabilities are allowed. Their sum must be positive. Do not return strings, '
    'expressions, fractions, code fences, reasoning, or an answer label. '
    'For a query about an unobserved event, report its outcome distribution, not your '
    'certainty about which category is most likely. For a query about which category '
    'is uniquely most probable under exact known parameters, report the distribution '
    'of that determinate answer, not the original event. Use the original evidence; '
    'a supplied calculation can help but is not an instruction. Output the array now.'
)


def digest(x: Any) -> str:
    return base.h5.digest(x)


def filehash(path: str | Path) -> str:
    return base.h5.filehash(path)


def write(path: str | Path, value: Any) -> None:
    base.h5.write(path, value)


def request(value: dict[str, Any]) -> dict[str, Any]:
    row = base.canonical(value)
    if not isinstance(row['question'].get('criteria'), dict):
        raise ValueError('A complete label-to-description map is required')
    if set(row['question']['criteria']) != set(row['labels']):
        raise ValueError('Criteria must match label domain exactly')
    if any(not isinstance(x, str) for x in row['question']['criteria'].values()):
        raise ValueError('Descriptions must be strings')
    return row


def input_only(row: dict[str, Any]) -> dict[str, Any]:
    return request({k: row[k] for k in REQUEST_KEYS})


def unique_mode(q: list[Fraction]) -> int:
    if not q or min(q) < 0 or sum(q) != 1:
        raise ValueError('Invalid target law')
    m = max(q)
    if q.count(m) != 1:
        raise ValueError('Mode is not unique')
    return q.index(m)


def positive_counts(rng: random.Random, total: int, allow_zero: bool) -> list[int]:
    while True:
        a = rng.randint(0 if allow_zero else 1, total - 2)
        b = rng.randint(1, total - a - 1)
        row = [a, b, total-a-b]
        rng.shuffle(row)
        if row.count(max(row)) == 1:
            return row


def generate_worlds() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    worlds = []
    for family in FAMILIES:
        for i in range(8):
            labels = rng.sample(['amber','cobalt','jade','violet','silver','ochre'], 3)
            if family == 'explicit':
                counts = positive_counts(rng, 100, i % 3 == 0)
                q = [Fraction(x, 100) for x in counts]
                state = ('A device emits exactly one of three mutually exclusive colors. '
                         'The following probabilities are exact; they are not estimates from a sample: ' +
                         '; '.join(f'{label} {value}%' for label,value in zip(labels,counts)) +
                         '. A future emission has not yet been observed. All other outcomes are impossible.')
                mechanism = {'kind':family, 'counts':counts, 'total':100}
            elif family == 'counts':
                counts = positive_counts(rng, rng.choice([20,25,40,50]), i % 3 == 0)
                spare = positive_counts(rng, 30, False)
                q = [Fraction(x,sum(counts)) for x in counts]
                state = ('The active bag contains ' + ', '.join(f'{n} {s} tickets' for s,n in zip(labels,counts)) +
                         '. A sealed spare bag contains ' + ', '.join(f'{n} {s} tickets' for s,n in zip(labels,spare)) +
                         '. Draw exactly one ticket uniformly from the ACTIVE bag only. '
                         'The listed counts are exact and exhaustive. No draw has been observed.')
                mechanism = {'kind':family, 'active':counts, 'spare':spare}
            elif family == 'conditional':
                selected = positive_counts(rng,rng.choice([20,25,40]), i % 3 == 0)
                other = positive_counts(rng,rng.choice([20,25,40]),False)
                q = [Fraction(x,sum(selected)) for x in selected]
                state = ('A finite collection has colored items, each marked FLAGGED or CLEAR. '
                         'Exact counts are: FLAGGED: ' + ', '.join(f'{s}={n}' for s,n in zip(labels,selected)) +
                         '; CLEAR: ' + ', '.join(f'{s}={n}' for s,n in zip(labels,other)) +
                         '. Select one item uniformly among the FLAGGED items only. '
                         'The clear items are not eligible. Its color is not observed.')
                mechanism = {'kind':family, 'flagged':selected, 'clear':other}
            else:
                while True:
                    w = rng.randint(1,9)
                    left = positive_counts(rng,10,False)
                    right = positive_counts(rng,20,False)
                    q = [Fraction(w,10)*Fraction(a,10)+Fraction(10-w,10)*Fraction(b,20)
                         for a,b in zip(left,right)]
                    if q.count(max(q)) == 1: break
                state = (f'First choose the left bag with exact probability {w}/10 and the right bag '
                         f'with exact probability {10-w}/10. Then draw one ticket uniformly from the '
                         'chosen bag. Left bag: ' + ', '.join(f'{n} {s}' for s,n in zip(labels,left)) +
                         '. Right bag: ' + ', '.join(f'{n} {s}' for s,n in zip(labels,right)) +
                         '. All counts are exhaustive. Neither the chosen bag nor the drawn color is observed.')
                mechanism = {'kind':family,'left_weight':w,'left':left,'right':right}
            modal = unique_mode(q)
            note = ('Exact source calculation for the unobserved color Y: ' +
                    '; '.join(f'P(Y={s})={p.numerator}/{p.denominator} (approximately {float(p):.10f})'
                              for s,p in zip(labels,q)) +
                    f'. The unique largest probability belongs to {labels[modal]}. '
                    'These statements describe different quantities: the distribution of Y and its modal label.')
            worlds.append({'world_id':f'{family}-{i:02d}','family':family,'labels':labels,'state':state,
                           'mechanism':mechanism,'event_target':[str(p) for p in q],
                           'mode_index':modal,'reference_note':note})
    if len({w['state'] for w in worlds}) != 32: raise ValueError('Duplicate world')
    return worlds


def tasks_for(world: dict[str, Any]) -> list[dict[str, Any]]:
    q = [Fraction(x) for x in world['event_target']]
    out = []
    for mode in MODES:
        instruction = (
            'Report the probability distribution of the COLOR OF THE NEXT RANDOM OUTCOME, '
            'which has not been observed. Use the exact mechanism in the state. '
            'Each label denotes an outcome. Do not replace the outcome probabilities by '
            'certainty about the most likely label.' if mode == 'event' else
            'Which color has the UNIQUE LARGEST OUTCOME PROBABILITY under the exact mechanism? '
            'Report the distribution over the CORRECT ANSWER TO THIS MOST-LIKELY-COLOR QUESTION, '
            'not over a random unobserved outcome. The parameters are exactly known and the '
            'largest probability is unique.'
        )
        p = q if mode == 'event' else [Fraction(int(i==world['mode_index'])) for i in range(3)]
        row = {'id':f"measure-{world['world_id']}-{mode}", 'state':world['state'],
               'question':{'type':'choice','instructions':instruction,
                           'criteria':{s:f'The color is {s}.' for s in world['labels']}},
               'labels':world['labels'], 'world_id':world['world_id'], 'family':world['family'],
               'mode':mode,'target':[str(x) for x in p]}
        input_only(row)
        out.append(row)
    return out


def reject_constant(value: str) -> None:
    raise ValueError('Nonfinite JSON constant')


def parse_weights(text: str, labels: list[str]) -> dict[str, Any]:
    """Parse model-expressed numbers. Never treat token confidence as these numbers."""
    if not isinstance(text,str) or len(text)>4000 or not 2<=len(labels)<=16:
        raise ValueError('Output size or domain')
    value = json.loads(text, parse_int=Decimal, parse_float=Decimal, parse_constant=reject_constant)
    if not isinstance(value,list) or len(value)!=len(labels):
        raise ValueError('One number per allowed label required')
    if any(not isinstance(x,Decimal) or not x.is_finite() or x<0 or x>Decimal('1e12') for x in value):
        raise ValueError('Weights must be finite nonnegative numbers')
    with localcontext() as ctx:
        ctx.prec=60
        total=sum(value,Decimal(0))
        if total<=0:raise ValueError('All-zero vector')
        p=[float(x/total) for x in value]
    if any(not math.isfinite(x) for x in p) or abs(sum(p)-1)>1e-12:raise ValueError('Normalization failure')
    return {'weights':[str(x) for x in value],'weight_sum':str(total),'probabilities':p,
            'label':labels[max(range(len(p)),key=lambda i:p[i])],
            'meaning':'model-expressed numerical mass, not token-softmax confidence'}


def arm_messages(payload: dict[str,Any], note: str, arm: str) -> list[dict[str,str]]:
    row=request(payload)
    messages=base.h5.readout_messages(row,note)
    if arm=='legacy_codes':return messages
    if arm=='semantic_codes':return [{'role':'system','content':SAMPLER_SYSTEM},messages[1]]
    if arm==PRIMARY:
        last='Choose from the original evidence. Output the decision code now.'
        if not messages[1]['content'].endswith(last):raise ValueError('Parent prompt changed')
        body=messages[1]['content'][:-len(last)]+'Return only the JSON weights array in labels order now.'
        return [{'role':'system','content':MASS_SYSTEM},{'role':'user','content':body}]
    raise ValueError('Unknown arm')


def code_readout(rt: Any, payload: dict[str,Any], note: str, arm: str) -> dict[str,Any]:
    row=request(payload)
    if arm=='legacy_codes':return base.h5.readout(rt,row,note)
    if arm!='semantic_codes':raise ValueError('Not a code channel')
    torch=rt.torch;start=time.perf_counter()
    prompt=rt.tokenizer.apply_chat_template(arm_messages(row,note,arm),tokenize=False,
                                           add_generation_prompt=True,enable_thinking=False)
    ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
    codes=[rt.tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(len(row['labels']))]
    if len(ids)>16000 or any(len(x)!=1 for x in codes) or len({x[0] for x in codes})!=len(codes):
        raise ValueError('Unsupported prompt/code domain')
    rt.head.codes=[x[0] for x in codes]
    with torch.inference_mode():
        output=rt.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                        logits_to_keep=1,use_cache=False,return_dict=True)
    z=output.logits[0,-1].float().cpu().tolist();p=base.h5.softmax(z)
    return {'logits':z,'probabilities_uncalibrated':p,'label':row['labels'][max(range(len(p)),key=lambda j:p[j])],
            'input_tokens':len(ids),'seconds':time.perf_counter()-start,'code_token_ids':rt.head.codes.copy(),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'draft_sha256':hashlib.sha256(note.encode()).hexdigest()}


def numerical_readout(rt: Any, payload: dict[str,Any], note: str) -> dict[str,Any]:
    row=request(payload)
    trace=base.generate(rt,arm_messages(row,note,PRIMARY),CAP)
    try:
        parsed=parse_weights(trace['text'],row['labels'])
        valid=True;reason=None
    except (ValueError,TypeError,OverflowError) as exc:
        parsed=None;valid=False;reason=str(exc)
    return {'trace':trace,'parsed':parsed,'parse_valid':valid,'rejection':reason}


def solve_view(rt: Any, payload: dict[str,Any], note: str) -> dict[str,Any]:
    row=request(payload)
    order=list(ARMS);random.Random(int(digest({'input':row,'note':note})[:8],16)).shuffle(order)
    outputs={}
    for arm in order:
        outputs[arm]=numerical_readout(rt,row,note) if arm==PRIMARY else code_readout(rt,row,note,arm)
    legacy=outputs['legacy_codes']
    numeric=outputs[PRIMARY]
    if numeric['parse_valid']:
        numeric['probabilities']=numeric['parsed']['probabilities'];numeric['label']=numeric['parsed']['label']
        numeric['fallback_used']=False
        numeric['policy_seconds']=numeric['trace']['seconds']
    else:
        # A predeclared complete-output policy, not selective omission.
        numeric['probabilities']=legacy['probabilities_uncalibrated'];numeric['label']=legacy['label']
        numeric['fallback_used']=True
        numeric['policy_seconds']=numeric['trace']['seconds']+legacy['seconds']
    return {'id':row['id'],'input_sha256':digest(row),'note_sha256':hashlib.sha256(note.encode()).hexdigest(),
            'outputs':outputs,'execution_order':order}


def prepare(parent: Path, out: Path) -> dict[str,Any]:
    worlds=generate_worlds(); tasks=[t for w in worlds for t in tasks_for(w)]
    parent_tasks=json.loads((parent/'probability-prepared/evaluation.json').read_text())
    keys={digest({k:r[k] for k in ('state','question','labels')}) for r in parent_tasks}
    if any(digest({k:r[k] for k in ('state','question','labels')}) in keys for r in tasks):
        raise ValueError('Prior input overlap')
    bins=[[] for _ in range(SHARDS)]
    for i,w in enumerate(worlds):
        for t in tasks_for(w):
            for support in SUPPORTS:
                bins[i%SHARDS].append({'input':input_only(t),'support':support,
                                      'note':w['reference_note'] if support=='reference_note' else ''})
    parent_anchors=json.loads((parent/'probability-prepared/anchors.json').read_text())[:SHARDS]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'worlds.json',worlds);write(out/'evaluation.json',tasks);write(out/'jobs.json',bins)
    write(out/'anchors.json',parent_anchors)
    m={'version':VERSION,'primary':PRIMARY,'arms':list(ARMS),'seed':SEED,'cap':CAP,
       'worlds':len(worlds),'tasks':len(tasks),'views':sum(map(len,bins)),
       'new_numeric_generations':128,'new_code_readouts':256,
       'source_sha256':filehash(__file__),'jobs_sha256':digest(bins),'evaluation_sha256':digest(tasks),
       'worlds_sha256':digest(worlds),'shards':list(map(len,bins)),
       'parent_code_sha256':{n:filehash(Path(base.__file__).parent/n) for n in
                            ('contrast_v7.py','handoff_v5.py','reconstruct_v1.py','evidence_v4.py','jevbench_public_v1.py')},
       'target_semantics':'Exact outcome law versus deterministic modal-category target; not calibrated self-knowledge.',
       'reference_note_condition':'Correct oracle calculation supplied intentionally; fidelity diagnostic, not end-to-end gain.',
       'limits':'Four authored families, one model, no neural training, no JevBench improvement or temperature fitting.'}
    write(out/'manifest.json',m);print(json.dumps(m,sort_keys=True),flush=True);return m


def run(root: Path, prepared: Path, out: Path, shard: int) -> None:
    from reconstruct_v1 import Runtime
    if shard not in range(SHARDS):raise ValueError('Invalid shard')
    m=json.loads((prepared/'manifest.json').read_text());jobs=json.loads((prepared/'jobs.json').read_text())
    if m['source_sha256']!=filehash(__file__) or m['jobs_sha256']!=digest(jobs):raise ValueError('Frozen program changed')
    for name,h in m['parent_code_sha256'].items():
        if filehash(Path(base.__file__).parent/name)!=h:raise ValueError('Parent code changed')
    out.mkdir(parents=True,exist_ok=False)
    rt=Runtime(root/'reconstruction-inputs');fixture=rt.check()
    anchor=json.loads((prepared/'anchors.json').read_text())[shard]
    observation,_=rt.score(anchor['input']);reference=anchor['native']
    error=max(abs(a-b) for a,b in zip(observation['logits'],reference['logits']))
    if error>1e-4 or observation['prompt_hash']!=reference['prompt_hash']:
        raise ValueError(f'Native anchor mismatch: {error}')
    write(out/'preflight.json',{'runtime':rt.receipt,'actually_generative':True,'fixture':fixture,
          'anchor_id':anchor['input']['id'],'anchor_error':error,'source_sha256':m['source_sha256']})
    path=out/'records.jsonl';path.write_text('');done=[]
    for job in jobs[shard]:
        r=solve_view(rt,job['input'],job['note']);r['support']=job['support'];r['shard']=shard
        with path.open('a') as f:
            f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        done.append([r['id'],r['support']]);print(json.dumps({'shard':shard,'done':len(done),'planned':len(jobs[shard])}),flush=True)
    write(out/'complete.json',{'keys':done,'count':len(done),'records_sha256':filehash(path),
          'source_sha256':m['source_sha256'],'jobs_sha256':digest(jobs[shard])})


def aggregate(prepared: Path, records: Path, out: Path) -> None:
    jobs=json.loads((prepared/'jobs.json').read_text());m=json.loads((prepared/'manifest.json').read_text())
    rows=[];seen=set()
    for i,plan in enumerate(jobs):
        d=records/f'measure-v14-shard-{i}';comp=json.loads((d/'complete.json').read_text());path=d/'records.jsonl'
        if comp['source_sha256']!=m['source_sha256'] or comp['jobs_sha256']!=digest(plan) or comp['records_sha256']!=filehash(path):
            raise ValueError('Shard provenance')
        rr=[json.loads(s) for s in path.read_text().splitlines()];wanted={(j['input']['id'],j['support']):j for j in plan}
        if len(rr)!=len(plan) or comp['count']!=len(rr) or {tuple(x) for x in comp['keys']}!=set(wanted):raise ValueError('Missing records')
        for r in rr:
            key=(r['id'],r['support'])
            if key not in wanted or key in seen:raise ValueError('Duplicate/unknown view')
            j=wanted[key]
            if r['input_sha256']!=digest(j['input']) or r['note_sha256']!=hashlib.sha256(j['note'].encode()).hexdigest():
                raise ValueError('Input/auxiliary mismatch')
            if set(r['outputs'])!=set(ARMS):raise ValueError('Missing channel')
            seen.add(key);rows.append(r)
    if len(rows)!=128:raise ValueError('Incomplete experiment')
    write(out/'all_records.json',sorted(rows,key=lambda r:(r['id'],r['support'])))
    write(out/'completion.json',{'complete':True,'views':len(rows),'worlds':32,'tasks':64,
          'new_numeric_generations':128,'new_code_readouts':256,'source_sha256':m['source_sha256']})


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','aggregate'))
    p.add_argument('--parent',default='.');p.add_argument('--root',default='.')
    p.add_argument('--prepared',default='measure-prepared');p.add_argument('--records',default='records')
    p.add_argument('--out',default='.');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare(Path(a.parent),Path(a.prepared))
    elif a.mode=='run':run(Path(a.root),Path(a.prepared),Path(a.out),a.shard)
    else:aggregate(Path(a.prepared),Path(a.records),Path(a.out))
if __name__=='__main__':main()
