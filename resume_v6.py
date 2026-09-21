"""Resume v6: terminal-completion allocation, with a causal rollback control.
No model training. No oracle labels in generation/readout. Archived context is
re-prefilled: this is not a claim of restored KV caches or exact token replay.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import random
import re
import time
from pathlib import Path
import handoff_v5 as h5

VERSION = 'resume-v6.0'
SEED = 6230927
SHARDS = 16
EXTRA = 160
BASE = 160
PRIMARY = 'resume_complete'
ARMS = ('native', 'reason160_fallback', 'handoff160', PRIMARY,
        'resume_native_fallback', 'rollback_complete', 'rollback_readout')


def canonical(r):
    return h5.digest({k: r[k] for k in ('state', 'question', 'labels')})


def fresh_tasks(seed=SEED, per_family=16):
    """English tasks with exact independent references; no supplied answer trace."""
    rng = random.Random(seed)
    rows = []
    for family in ('ledger', 'schedule', 'pointer', 'policy'):
        for i in range(per_family):
            name, other = rng.sample(['Mina', 'Taro', 'Neri', 'Sora', 'Ari', 'Ren', 'Yuki', 'Hana'], 2)
            if family == 'ledger':
                start = rng.randint(180, 550)
                active = [rng.choice((-1, 1)) * rng.randint(3, 59) for _ in range(8 + i % 5)]
                noise = [rng.choice((-1, 1)) * rng.randint(10, 90) for _ in range(3)]
                events = [(name, a) for a in active] + [(other, a) for a in noise]
                rng.shuffle(events)
                text = [f'{name} begins with {start} credits. {other} has a separate account.']
                for actor, delta in events:
                    text.append(f'{actor} ' + (f'receives {delta}' if delta > 0 else f'spends {-delta}') + ' credits.')
                answer = start + sum(active)
                balance = start
                for actor, delta in events:
                    if actor == name:
                        balance = balance + delta
                assert balance == answer
                state = ' '.join(text)
                question = f'After all events, how many credits does {name} have? Ignore the separate account.'
                candidates = [answer, answer + abs(active[0]), answer - abs(active[-1]), answer + sum(noise)]
                oracle = {'opening': start, 'target': name, 'events': events, 'answer': answer}
            elif family == 'schedule':
                start = rng.randint(6, 11) * 60 + rng.choice([5, 10, 15, 25, 35, 45, 55])
                stages = [rng.randint(8, 43) for _ in range(5 + i % 3)]
                gaps = [rng.randint(2, 17) for _ in range(len(stages)-1)]
                parts = [f'{name} starts at {start//60:02d}:{start%60:02d}. Stages run sequentially, not in parallel.']
                for j, duration in enumerate(stages):
                    parts.append(f'Stage {j+1} lasts {duration} minutes.')
                    if j < len(gaps):
                        parts.append(f'After stage {j+1}, pause {gaps[j]} minutes before starting the next stage.')
                parts.append(f'{other} has a separate appointment lasting {rng.randint(80,140)} minutes; it does not affect this job. There is no pause after the final stage.')
                answer = start + sum(stages) + sum(gaps)
                t = datetime.datetime(2000, 1, 1, start//60, start%60)
                for j, duration in enumerate(stages):
                    t += datetime.timedelta(minutes=duration)
                    if j < len(gaps):
                        t += datetime.timedelta(minutes=gaps[j])
                assert answer == t.hour * 60 + t.minute and answer < 1440
                state = ' '.join(parts)
                question = 'At how many minutes after midnight does the final stage finish? Return the numeric option, not a clock string.'
                candidates = [answer, answer - gaps[-1], answer - stages[-1], answer + gaps[0]]
                oracle = {'start_minute': start, 'stages': stages, 'gaps': gaps, 'answer': answer}
            elif family == 'pointer':
                nodes = ['amber', 'cobalt', 'jade', 'violet', 'silver', 'ochre', 'indigo', 'ivory']
                red = nodes.copy(); blue = nodes.copy()
                rng.shuffle(red); rng.shuffle(blue)
                maps = {'red': dict(zip(nodes, red)), 'blue': dict(zip(nodes, blue))}
                start = rng.choice(nodes)
                actions = [rng.choice(['red', 'blue']) for _ in range(9 + i % 6)]
                state = ('There are two directed successor maps. RED: ' + '; '.join(f'{k} goes to {v}' for k,v in maps['red'].items()) +
                         '. BLUE: ' + '; '.join(f'{k} goes to {v}' for k,v in maps['blue'].items()) +
                         f'. Begin at {start}. Apply these map names in order: ' + ', '.join(actions) +
                         '. Each name applies exactly one transition of that map. A map not named in the sequence is not used.')
                answer = start
                for action in actions:
                    answer = maps[action][answer]
                indices = {x:j for j,x in enumerate(nodes)}
                index = indices[start]
                arrays = {c: [indices[maps[c][x]] for x in nodes] for c in maps}
                for action in actions:
                    index = arrays[action][index]
                assert nodes[index] == answer
                question = 'Which node is reached after the entire ordered sequence?'
                candidates = [answer] + rng.sample([n for n in nodes if n != answer], 3)
                oracle = {'maps': maps, 'start': start, 'actions': actions, 'answer': answer}
            else:
                # Outcome balancing is fixed in the generator, never selected on model behavior.
                for attempt in range(10000):
                    paid, verified, sponsored, blocked, waiver = [bool(rng.getrandbits(1)) for _ in range(5)]
                    amount = rng.randint(20, 120); limit = rng.randint(30, 100)
                    req = rng.choice(['all', 'any'])
                    docs = [bool(rng.getrandbits(1)) for _ in range(3)]
                    truth = ((verified and paid) or sponsored) and ((all(docs) if req=='all' else any(docs)) or waiver) and amount <= limit and not blocked
                    if truth == (i % 2 == 0):
                        break
                else:
                    raise RuntimeError('Failed to generate balanced policy')
                state = (f'An application is approved if and only if all three numbered conditions hold: '
                         '(1) the applicant is verified AND paid, OR the applicant is sponsored; '
                         f'(2) {"all" if req=="all" else "at least one"} of the applicant\'s three documents is signed, OR the applicant has a document waiver; '
                         f'(3) the requested amount is at most {limit} credits AND the applicant is not blocked. '
                         'Sponsorship and document waivers never remove the amount limit or the blocked-account prohibition. '
                         f'{name} is '+('verified' if verified else 'not verified')+', '+('paid' if paid else 'not paid')+', '+('sponsored' if sponsored else 'not sponsored')+', and '+('blocked' if blocked else 'not blocked')+'. '+
                         f'{name} '+('has' if waiver else 'does not have')+' a document waiver. '+
                         f'{name}\'s documents are: '+', '.join(f'document {j+1} '+('signed' if d else 'not signed') for j,d in enumerate(docs))+'. '+
                         f'{name} requests {amount} credits. {other} is a different applicant, verified and paid, not blocked, with all documents signed.')
                flags = [(verified and paid) or sponsored, (all(docs) if req=='all' else any(docs)) or waiver, amount <= limit and not blocked]
                answer = all(flags)
                independent = (int(verified)*int(paid) + int(sponsored)>0) and ((sum(docs)==3 if req=='all' else sum(docs)>0) or waiver) and (amount<=limit) and (not blocked)
                assert independent == answer
                question = f'Is {name}\'s application approved under this complete policy?'
                labels = ['no', 'yes']; criteria = {'no':'No.', 'yes':'Yes.'}
                gold = 'yes' if answer else 'no'
                oracle = {'paid':paid, 'verified':verified, 'sponsored':sponsored, 'blocked':blocked, 'waiver':waiver, 'docs':docs, 'quantifier':req, 'amount':amount, 'limit':limit}
                candidates = None
            if candidates is not None:
                # Deterministic collision repair occurs before any model inference.
                seen = set(); values = []
                for value in candidates:
                    while value in seen:
                        value = value + 1 if isinstance(value, int) else value + '_unused'
                    seen.add(value); values.append(value)
                assert values[0] == answer
                rng.shuffle(values); labels = ['a','b','c','d']
                criteria = dict(zip(labels,map(str,values))); gold = labels[values.index(answer)]
            rows.append({'id':f'resume-{seed}-{family}-{i:02d}', 'state':state,
                         'question':{'type':'choice','instructions':question,'criteria':criteria},
                         'labels':labels,'expected':gold,'family':family,'partition':'fresh_v6','oracle':oracle})
    assert len({canonical(r) for r in rows}) == len(rows)
    return rows


def checkpoint_text(tokenizer, text, max_drop=32):
    """Causal, lexical rollback. Sentence punctuation is NOT semantic proof."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    boundaries = [m.end() for m in re.finditer(r'(?:[.!?](?=\s)|\n)',text)]
    for end in reversed(boundaries):
        prefix = text[:end]
        removed = len(ids) - len(tokenizer.encode(prefix,add_special_tokens=False))
        if 0 < removed <= max_drop:
            return prefix, {'changed':True,'removed_tokens':removed,'offset':end}
    return text, {'changed':False,'removed_tokens':0,'offset':len(text)}


def load_old(root):
    root=Path(root)
    rows=json.loads((root/'all_evaluation_inputs.json').read_text())
    records=json.loads((root/'all_handoff_records.json').read_text())
    index={r['id']:r for r in records}
    assert len(rows)==len(records)==len(index)==359
    unique=[]; seen=set()
    for r in rows:
        assert h5.digest(h5.input_only(r))==index[r['id']]['input_sha256']
        if canonical(r) not in seen:
            unique.append({**r, 'partition':index[r['id']]['partition']}); seen.add(canonical(r))
    assert len(unique)==357
    return unique,index


def prepare(root, out):
    old,index=load_old(root); fresh=fresh_tasks()
    assert not {canonical(r) for r in old} & {canonical(r) for r in fresh}
    items=[]; inherited=[]
    for r in old:
        previous=index[r['id']]
        final=h5.parse_final(previous['trace']['text'],r['labels'],cut=previous['trace'].get('hit_cap',False))
        assert final==previous['parsed_final160']
        # Only terminally incomplete drafts enter the new inference pool.
        if final is None:
            items.append({'input':h5.input_only(r),'partition':r['partition'],'previous':previous})
        else:
            inherited.append({'id':r['id'],'input_sha256':previous['input_sha256'],'predictions':{arm: previous['predictions']['native'] if arm=='native' else previous['predictions']['reason160_fallback'] if arm=='reason160_fallback' else previous['predictions']['handoff160'] for arm in ARMS},'preserved':True})
    items += [{'input':h5.input_only(r),'partition':r['partition'],'previous':None} for r in fresh]
    bins=[[] for _ in range(SHARDS)]; costs=[0.]*SHARDS
    for x in sorted(items,key=lambda x: (-(len(json.dumps(x['input']))+ (3500 if x['previous'] is None else 1500)), x['input']['id'])):
        j=min(range(SHARDS),key=lambda i:(costs[i],i));bins[j].append(x)
        costs[j]+=len(json.dumps(x['input']))+(3500 if x['previous'] is None else 1500)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    h5.write(out/'jobs.json',bins);h5.write(out/'inherited.json',inherited)
    h5.write(out/'evaluation.json',old+fresh)
    receipt={'version':VERSION,'primary':PRIMARY,'extra_token_cap':EXTRA,'base_token_cap':BASE,
             'source_sha256':h5.filehash(__file__),'arms':ARMS,'new_inference_inputs':len(items),
             'inherited':len(inherited),'old_unique':len(old),'fresh_unique':len(fresh),'jobs_hash':h5.digest(bins),
             'evaluation_hash':h5.digest(old+fresh),'shard_counts':list(map(len,bins)),
             'selection':'none; missing terminal FINAL only; not confidence or ground truth',
             'mode':'old archived context replay; fresh live base generation; all continuations re-prefill'}
    h5.write(out/'freeze.json',receipt);print(json.dumps(receipt),flush=True)
    return receipt


def continue_trace(rt,row,original_text,cap=EXTRA):
    import evidence_v4 as e4
    from transformers import StoppingCriteria,StoppingCriteriaList
    torch=rt.torch
    prompt=rt.tokenizer.apply_chat_template(e4.messages(row,'reason'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt_ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
    prefix_ids=rt.tokenizer.encode(original_text,add_special_tokens=False)
    decoded=rt.tokenizer.decode(prefix_ids,skip_special_tokens=True)
    if decoded!=original_text:
        raise ValueError('Archived prefix is not losslessly retokenizable')
    ids=prompt_ids+prefix_ids
    if len(ids)+cap>16000:raise ValueError('Context bound; no silent truncation')
    start=time.perf_counter();calls=0
    class Finish(StoppingCriteria):
        def __call__(self,input_ids,scores,**kw):
            nonlocal calls
            calls+=1
            # Ignore a possibly partial label unless followed by a newline.
            text=rt.tokenizer.decode(input_ids[0,len(prompt_ids):].tolist(),skip_special_tokens=True)
            return h5.parse_final(text,row['labels'],cut=True) is not None
    rt.model.set_output_embeddings(rt.original_head)
    try:
        with torch.inference_mode():
            out=rt.model.generate(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=rt.tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([Finish()]))
    finally:rt.model.set_output_embeddings(rt.head)
    added=out[0,len(ids):].tolist();complete_ids=prefix_ids+added
    text=rt.tokenizer.decode(complete_ids,skip_special_tokens=True)
    eos=rt.model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else (eos or [])
    ended=bool(added and added[-1] in eos)
    capped=len(added)==cap and not ended
    return {'text':text,'added_token_ids':added,'added_tokens':len(added),'prefix_tokens':len(prefix_ids),
            'input_tokens':len(ids),'seconds':time.perf_counter()-start,'hit_cap':capped,'eos':ended,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prefix_sha256':hashlib.sha256(original_text.encode()).hexdigest(),
            'final':h5.parse_final(text,row['labels'],cut=capped),'stopping_checks':calls,
            'cache_scope':'new cache inside continuation; original draft cache not reused'}


def run(root,prepared,out,shard):
    import reconstruct_v1 as v1
    if shard not in range(SHARDS):raise ValueError('Invalid shard')
    root,prepared,out=map(Path,(root,prepared,out));out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((prepared/'freeze.json').read_text());bins=json.loads((prepared/'jobs.json').read_text())
    assert manifest['source_sha256']==h5.filehash(__file__) and manifest['jobs_hash']==h5.digest(bins)
    rt=v1.Runtime(root/'reconstruction-inputs');fixture=rt.check();old,idx=load_old(root)
    anchor_row=sorted(old,key=lambda r:r['id'])[shard];anchor,_=rt.score(h5.input_only(anchor_row));ref=idx[anchor_row['id']]['native']
    err=max(abs(x-y) for x,y in zip(anchor['logits'],ref['logits']))
    assert err<1e-4 and anchor['prompt_hash']==ref['prompt_hash']
    h5.write(out/'preflight.json',{'runtime':rt.receipt,'explicitly_generative':True,'fixture':fixture,'anchor':anchor_row['id'],'anchor_error':err,'source_sha256':h5.filehash(__file__)})
    done=[]
    for job in bins[shard]:
        row=job['input'];prior=job['previous']
        if prior is None:
            native,_=rt.score(row);native_label=row['labels'][max(range(len(row['labels'])),key=lambda j:native['logits'][j])]
            trace=h5.generate_fresh(rt,row)
            final=h5.parse_final(trace['text'],row['labels'],cut=trace.get('hit_cap',False))
            base_readout=None if final else h5.readout(rt,row,trace['text'])
            baseline=final or base_readout['label']
            import evidence_v4 as e4
            legacy=e4.claimed(row,trace['text'],'FINAL') or native_label
        else:
            native=prior['native'];native_label=prior['predictions']['native'];trace=prior['trace'];base_readout=prior['readouts']['draft160'];baseline=prior['predictions']['handoff160'];legacy=prior['predictions']['reason160_fallback'];final=prior['parsed_final160']
        pred={'native':native_label,'reason160_fallback':legacy,'handoff160':baseline}
        continued={};reads={};checkpoint=None
        if final is not None:
            pred.update({arm:baseline for arm in ARMS if arm not in pred})
        else:
            clipped,checkpoint=checkpoint_text(rt.tokenizer,trace['text'])
            # Equal generation ceilings, randomized branch order; no labels inspected.
            paths=[('resume',trace['text']),('rollback',clipped)]
            random.Random(int(h5.digest(row['id'])[:8],16)).shuffle(paths)
            memo={}
            for name,text in paths:
                if text not in memo:
                    c=continue_trace(rt,row,text)
                    rd=None if c['final'] else h5.readout(rt,row,c['text'])
                    memo[text]=(c,rd)
                continued[name],reads[name]=memo[text]
            c=continued['resume'];r=continued['rollback']
            pred['resume_complete']=c['final'] or reads['resume']['label']
            pred['resume_native_fallback']=c['final'] or native_label
            pred['rollback_complete']=r['final'] or reads['rollback']['label']
            if clipped==trace['text']:
                reads['rollback_only']=base_readout
            else:
                reads['rollback_only']=h5.readout(rt,row,clipped)
            pred['rollback_readout']=reads['rollback_only']['label']
        result={'id':row['id'],'partition':job['partition'],'input_sha256':h5.digest(row),'native':native,'base_trace':trace,'base_readout':base_readout,'base_final':final,'predictions':pred,'continuations':continued,'readouts':reads,'checkpoint':checkpoint,'inherited_base':prior is not None,'shard':shard}
        assert set(pred)==set(ARMS) and all(v in row['labels'] for v in pred.values())
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
        done.append(row['id']);print(json.dumps({'shard':shard,'completed':len(done),'planned':len(bins[shard]),'id':row['id'],'extended':final is None}),flush=True)
    h5.write(out/'complete.json',{'ids':done,'count':len(done),'source_sha256':h5.filehash(__file__),'records_sha256':h5.filehash(out/'records.jsonl'),'jobs_hash':h5.digest(bins[shard]),'shard':shard})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',default='.');p.add_argument('--prepared',default='resume-prepared');p.add_argument('--out',default='resume-results');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.prepared)
    else:run(a.root,a.prepared,a.out,a.shard)
if __name__=='__main__':main()
