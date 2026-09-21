"""Orbit v2: counterfactual-order probes and new executable challenge set.
No benchmark labels enter inference. Uses the pinned v1 runtime without decoding.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import reconstruct_v1 as v1
from finalize_clean_v1 import dedup_protocol
SEED=902731
SHARDS=16
PROBE_VERSION='orbit-v2-cyclic-intervention-1'

def orders(k):
    if k < 2: raise ValueError('At least two options required')
    return {'native':list(range(k)), 'reverse':list(reversed(range(k))), 'cycle':list(range(1,k))+[0]}

def challenge():
    """Fresh code-derived tasks, never used for fitting or arm selection."""
    rng=random.Random(SEED); rows=[]
    for family in ('ledger','schedule','pointer','affine'):
        for i in range(32):
            k=2+i%4
            if family=='ledger':
                start=rng.randint(100,500); events=[rng.choice((-1,1))*rng.randint(3,45) for _ in range(5+i%6)]
                answer=start+sum(events)
                state={'opening_units':start,'events':[{'operation':'credit' if a>=0 else 'debit','units':abs(a)} for a in events]}
                instruction='Apply every credit and debit in order. What is the exact closing number of units?'
                check=start
                for event in state['events']:check+=event['units']*(1 if event['operation']=='credit' else -1)
                assert check==answer
            elif family=='schedule':
                start=rng.randint(300,1050); intervals=[rng.randint(5,60) for _ in range(3+i%5)]
                rests=[rng.randint(2,15) for _ in range(len(intervals)-1)]
                state={'start_clock':f'{start//60:02d}:{start%60:02d}','work_durations_minutes':intervals,'rest_after_each_nonfinal_task_minutes':rests}
                instruction='Tasks occur sequentially. Rest only between tasks. At what clock time does the final task finish, in 24-hour HH:MM format?'
                total=start+sum(intervals)+sum(rests); answer=total
                timeline=start
                for j,duration in enumerate(intervals):timeline+=duration+(rests[j] if j<len(rests) else 0)
                assert total==timeline and total<1440
            elif family=='pointer':
                names=[f'node_{j}' for j in range(9)]; shuffled=names.copy(); rng.shuffle(shuffled)
                mapping={shuffled[j]:shuffled[(j+1)%len(shuffled)] for j in range(len(shuffled))}
                start=rng.choice(names); steps=rng.randint(3,17)
                state={'start':start,'steps':steps,'successor':mapping}
                instruction='Starting at start, follow exactly steps successor links. Which node is reached? Count a transition as one step.'
                answer=shuffled[(shuffled.index(start)+steps)%len(shuffled)]
                check=start
                for _ in range(steps):check=mapping[check]
                assert check==answer
            else:
                modulus=rng.choice((17,19,23,29)); multiplier=rng.randint(2,6); offset=rng.randint(1,7); start=rng.randrange(modulus); steps=rng.randint(2,6)
                state={'initial_x':start,'steps':steps,'update_rule':f'x = ({multiplier} * x + {offset}) mod {modulus}', 'mod_definition':'Return the integer remainder from 0 through modulus minus 1.'}
                instruction='Apply the specified update exactly steps times. What is the final integer x?'
                answer=(multiplier**steps*start+offset*sum(multiplier**j for j in range(steps)))%modulus
                check=start
                for _ in range(steps):check=(multiplier*check+offset)%modulus
                assert check==answer
            if family=='pointer':
                wrong=rng.sample([x for x in names if x!=answer],k-1)
            elif family=='affine':
                wrong=rng.sample([x for x in range(modulus) if x!=answer],k-1)
            else:
                candidates=sorted({answer+d for d in (-31,-17,-11,-5,5,11,17,31) if 0<=answer+d<(1440 if family=='schedule' else 2000)})
                wrong=rng.sample(candidates,k-1)
            values=[answer]+wrong; rng.shuffle(values); labels=[f'opt_{j}' for j in range(k)]
            desc=lambda x: f'{x//60:02d}:{x%60:02d}' if family=='schedule' else str(x)
            r={'id':f'orbit-new-{family}-{i:03d}','state':state,'question':{'type':'choice','instructions':instruction,'criteria':dict(zip(labels,map(desc,values)))},'labels':labels,'expected':labels[values.index(answer)],'source':f'executable-{family}','partition':'fresh_test','family':family,'reference_value':answer}
            rows.append(r)
    keys=[v1.digest({k:r[k] for k in ('state','question','labels')}) for r in rows]
    assert len(set(keys))==len(rows)==128
    return rows

def inputs(root):
    root=Path(root)
    labeled=json.loads((root/'training.json').read_text()); keep=set(dedup_protocol(labeled)['retained_ids'])
    old=json.loads((root/'inputs.json').read_text())
    old=[r for r in old if r['id'] in keep or r['partition']=='benchmark']
    fresh=challenge(); oldkeys={v1.digest({k:r[k] for k in ('state','question','labels')}) for r in old}
    assert not oldkeys & {v1.digest({k:r[k] for k in ('state','question','labels')}) for r in fresh}
    return old+[{**v1.input_only(r),'partition':'fresh_test'} for r in fresh]

def plan(root):
    rows=inputs(root); jobs=[]
    for row in rows:
        variants=['cycle'] if row['partition']!='fresh_test' else ['native','reverse','cycle']
        done=set()
        for variant in variants:
            order=tuple(orders(len(row['labels']))[variant])
            if row['partition']!='fresh_test' and len(order)==2:continue  # identical to archived reverse
            if order in done:continue
            done.add(order); jobs.append({'input':row,'variant':variant,'order':list(order)})
    bins=[[] for _ in range(SHARDS)]; cost=[0.]*SHARDS
    for job in sorted(jobs,key=lambda j:(-len(json.dumps(j['input']))**1.4,j['input']['id'],j['variant'])):
        index=min(range(SHARDS),key=lambda n:(cost[n],n));bins[index].append(job);cost[index]+=len(json.dumps(job['input']))**1.4
    return bins

class OrbitRuntime(v1.Runtime):
    variant='native'
    def encode(self,inp,reverse=False):
        import jevbench_public_v1 as upstream
        from semif_phase1.direct import encode_prompt
        _,sem=upstream.convert(v1.input_only(inp))
        order=orders(len(sem['options']))[self.variant]
        sem['options']=[sem['options'][j] for j in order]
        return encode_prompt(self.tokenizer,sem,v1.MAX_TOKENS)

def run(root,feature_root,out,shard):
    if shard not in range(SHARDS):raise ValueError('Invalid shard')
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    jobs=plan(root)[shard]; rt=OrbitRuntime(root); fixture=rt.check()
    prior=json.loads((Path(feature_root)/f'shard-{shard}'/'records.json').read_text())
    anchor=next(x for x in prior if not x['reverse'])
    all_inputs={r['id']:r for r in json.loads((root/'inputs.json').read_text())}
    measured,_=rt.score(all_inputs[anchor['id']]); err=max(abs(a-b) for a,b in zip(measured['logits'],anchor['logits']))
    if err>2e-4 or measured['prompt_hash']!=anchor['prompt_hash']:raise RuntimeError('Archived native anchor mismatch: '+str(err))
    v1.write(out/'preflight.json',{'runtime':rt.receipt,'fixture':fixture,'native_anchor':{'id':anchor['id'],'max_logit_error':err}})
    records=[]
    for n,job in enumerate(jobs):
        row=job['input'];rt.variant=job['variant']; rec,_=rt.score(row)
        rec.update({'id':row['id'],'partition':row['partition'],'variant':job['variant'],'order':job['order'],'labels':row['labels'],'shard':shard})
        records.append(rec)
        if n%4==0:v1.emit('orbit_progress',shard=shard,done=n+1,total=len(jobs))
    v1.write(out/'records.json',records)
    v1.write(out/'complete.json',{'version':PROBE_VERSION,'shard':shard,'count':len(records),'records_sha256':v1.filehash(out/'records.json'),'planned_sha256':v1.digest(jobs),'source_sha256':v1.filehash(__file__),'fresh_inputs_sha256':v1.digest([v1.input_only(r) for r in challenge()])})
    v1.emit('orbit_complete',shard=shard,records=len(records))

def test():
    r=challenge();assert len(r)==128
    for k in range(2,17):
        for name,p in orders(k).items():assert sorted(p)==list(range(k))
    assert orders(2)['cycle']==orders(2)['reverse']
    assert all('expected' not in v1.input_only(x) for x in r)
    print('generator_and_permutation_tests_passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['test','plan','extract']);p.add_argument('--root',default='reconstruction-inputs');p.add_argument('--features',default='features');p.add_argument('--out',default='orbit-shard');p.add_argument('--shard',type=int,default=0);a=p.parse_args();test()
    if a.mode=='plan':
        v1.write(Path(a.out)/'plan.json',{'shards':[len(x) for x in plan(a.root)],'inputs_sha256':v1.digest(inputs(a.root)),'source_sha256':v1.filehash(__file__)})
        v1.write(Path(a.out)/'fresh_tasks.json',challenge())
    elif a.mode=='extract':run(a.root,a.features,a.out,a.shard)
if __name__=='__main__':main()
