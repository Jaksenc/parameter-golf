"""New off-benchmark development situations. Reference fields never enter prompts."""
from __future__ import annotations
import copy, hashlib, json, random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

FAMILIES=('amount','elapsed','lookup','policy','judge','probability')

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def development():
    rows=[]
    for family in FAMILIES:
        for block in range(6):
            seed=int(hashlib.sha256(f'flow-development-v1/{family}/{block}'.encode()).hexdigest()[:16],16)
            rng=random.Random(seed)
            entity='R-'+''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(7))
            block_id=hashlib.sha256(f'{seed}:source'.encode()).hexdigest()[:16]
            options=None
            for edit in range(2):
                # Restore shared random state so both endpoints share nuisance factors.
                rr=random.Random(seed+801)
                ref={}; dist=None
                if family=='amount':
                    count=rr.randint(7,24); price=rr.randint(137,967); fee=rr.randint(80,250); credit=rr.randint(20,190)
                    total=count*price+fee-credit; inclusive=bool(rr.getrandbits(1))
                    boundary=total+(0 if inclusive else 1) if edit==0 else total+(-1 if inclusive else 0)
                    money=lambda x:f'{x//100}.{x%100:02d}'
                    state=f'Invoice {entity}. Units supplied: {count}. Price per unit: ${money(price)}. Add one handling fee of ${money(fee)} for the whole invoice, not per unit. A single credit of ${money(credit)} reduces the final amount. The authorization ceiling is ${money(boundary)}. Another order has a $900.00 ceiling; it is not this invoice.'
                    question=f'Calculate the final amount for invoice {entity}. Does it fall {"at or below" if inclusive else "strictly below"} its authorization ceiling? Apply the stated handling fee and credit exactly once.'
                    opts=[('satisfies','The calculated invoice meets the stated comparison.'),('exceeds','The calculated invoice does not meet the stated comparison.')]
                    y='satisfies' if edit==0 else 'exceeds'
                    ref={'units':count,'cents':price,'fee':fee,'credit':credit,'total':total,'boundary':boundary,'inclusive':inclusive}
                    assert Decimal(count)*Decimal(money(price))+Decimal(money(fee))-Decimal(money(credit))==Decimal(money(total))
                elif family=='elapsed':
                    start=datetime(2027,rr.randint(1,10),rr.randint(2,24),rr.randint(4,18),rr.choice([5,20,45]),tzinfo=timezone.utc)
                    minutes=rr.randint(300,3600); inclusive=bool(rr.getrandbits(1)); event=start+timedelta(minutes=minutes)
                    limit=minutes+(0 if inclusive else 1) if edit==0 else minutes+(-1 if inclusive else 0)
                    zone=timezone(timedelta(hours=rr.choice([-5,-3,2,7])))
                    state=f'The sensor for {entity} opened its window at {start.isoformat()}. The event is recorded as {event.astimezone(zone).isoformat()}. These timestamps describe actual instants and their explicit offsets apply. A separate archived event happened in 2024.'
                    question=f'Was this event recorded {"within, including exactly at" if inclusive else "strictly before"} {limit} minutes after its window opened? Compare elapsed time, not the displayed local clock hour.'
                    opts=[('inside','The elapsed-time requirement is met.'),('outside','The elapsed-time requirement is not met.')]
                    y='inside' if edit==0 else 'outside'; ref={'minutes':minutes,'limit':limit,'inclusive':inclusive,'start':start.isoformat(),'event':event.astimezone(zone).isoformat()}
                    assert int((datetime.fromisoformat(ref['event'])-datetime.fromisoformat(ref['start'])).total_seconds())==minutes*60
                elif family=='lookup':
                    pools=['copper','indigo','jade','silver','amber','violet'][:4+2*(block%2)]
                    staff=[entity]+['P-'+''.join(rr.choice('ABCDEFGHJKLMN') for _ in range(5)) for j in range(4)]
                    teams=['T-'+''.join(rr.choice('PQRSTUVWXY') for _ in range(4)) for j in range(5)]
                    perm=list(range(len(pools))); rr.shuffle(perm)
                    first=perm[0]; second=perm[1]; selected=pools[first if edit==0 else second]
                    records=[f'Active membership: {staff[0]} -> {teams[0]}.', f'Current route: {teams[0]} -> {selected}.',f'Superseded route (ignore): {teams[0]} -> {pools[(first+2)%len(pools)]}.']
                    for j in range(1,5):
                        records += [f'Active membership: {staff[j]} -> {teams[j]}.',f'Current route: {teams[j]} -> {pools[perm[j%len(pools)]]}.']
                    rr.shuffle(records); state='\n'.join(records)
                    question=f'Follow the active membership and then the current route to select the pool for {entity}. A superseded route has no authority.'
                    opts=[(x,f'The current destination is pool {x}.') for x in pools]; y=selected
                    ref={'entity':entity,'team':teams[0],'destination':selected}
                elif family=='policy':
                    # Decisive policy/evidence are dispersed; archive notes are explicitly unrelated.
                    blocked=edit==1; permit=bool(rr.getrandbits(1)); count=18+block*7
                    notes=[f'Archive note {j+1}: equipment group Q{rr.randint(1000,9999)} retains its maintenance record. This note concerns that separate group, not the request being decided.' for j in range(count)]
                    early='General procedure: active requests may be released. Requests missing a required permit go to permit-review unless a higher-priority rule overrides this.'
                    mid='Current amendment: an active safety block requires a hold. No audit flag or permit waives a safety block. This amendment takes priority over the general procedure.'
                    late=f'Current record for {entity}: active=true; required permit present={str(permit).lower()}; safety block={str(blocked).lower()}; audit flag=true.'
                    notes.insert(1,early); notes.insert(len(notes)//2,mid); notes.insert(len(notes)-2,late)
                    state='\n\n'.join(notes)
                    question=f'Decide the disposition of request {entity} under the current procedure and amendment. Apply the higher-priority safety condition before the permit rule. Use only the current record for this request.'
                    opts=[('release','Active, no safety block, and required permit present.'),('permit-review','Active, no safety block, and required permit absent.'),('hold','An active safety block applies, regardless of permit or audit flag.'),('inactive','The request is not active and no active safety block applies.')]
                    y='hold' if blocked else ('release' if permit else 'permit-review'); ref={'blocked':blocked,'permit':permit,'active':True,'decisive_records':3}
                elif family=='judge':
                    rate=rr.randint(17,83); n=rr.randint(7,23); fee=rr.randint(11,71); total=rate*n+fee
                    errtype=block%3
                    if edit==0: candidate=f'The result is {total}. The one-time fee is {fee}.'; y='valid'
                    elif errtype==0: candidate=f'The result is {total+rr.choice([1,10,100])}. The one-time fee is {fee}.'; y='arithmetic'
                    elif errtype==1: candidate=f'The result is {total}. The one-time fee is {fee}. The customer has also confirmed delivery.'; y='unsupported'
                    else: candidate=f'The result is {total}.'; y='missing'
                    state=f'Source for {entity}: {n} units at {rate} credits each, plus one fee of {fee} credits. No delivery confirmation is supplied. Required answer format: state the total and explicitly name the fee.\nCandidate answer: {candidate}'
                    question='Identify the first applicable verdict in this priority: arithmetic error; unsupported factual assertion; omitted required fee; otherwise valid. Judge against the source, not writing style.'
                    opts=[('valid','Correct calculation, no unsupported fact, and required fee explicitly stated.'),('arithmetic','The stated total is arithmetically incorrect.'),('unsupported','The total is correct but an additional factual assertion lacks source support.'),('missing','The total is correct and no unsupported assertion occurs, but the required fee is not explicitly stated.')]
                    ref={'rate':rate,'n':n,'fee':fee,'total':total,'candidate':candidate,'verdict':y}
                else:
                    labels=['ash','birch','cedar','elm']; nums=[rr.randint(7,70) for _ in labels]
                    changed=rr.randrange(4); nums[changed]+=0 if edit==0 else rr.randint(35,90)
                    outside=[rr.randint(30,140) for _ in labels]; cohort=rr.choice(['north','south','east'])
                    records=[f'Complete {cohort} cohort: '+', '.join(f'{x}={v}' for x,v in zip(labels,nums))+'.', 'Separate excluded cohort: '+', '.join(f'{x}={v}' for x,v in zip(labels,outside))+'.']
                    rr.shuffle(records); state='\n'.join(records)
                    question=f'For one uniformly sampled record from the complete {cohort} cohort only, give probabilities over the mutually exclusive tree categories. The other cohort is excluded; use population proportions without smoothing.'
                    opts=[(x,f'The sampled record has category {x}.') for x in labels]
                    dist={x:float(Fraction(v,sum(nums))) for x,v in zip(labels,nums)}; y=max(sorted(dist),key=dist.get); ref={'counts':dict(zip(labels,nums)),'total':sum(nums),'excluded':dict(zip(labels,outside)),'cohort':cohort}
                if options is None:
                    random.Random(seed+909).shuffle(opts); options=copy.deepcopy(opts)
                else:
                    descriptions=dict(opts); opts=[(k,descriptions[k]) for k,_ in options]
                options=copy.deepcopy(opts)
                p=dist if dist is not None else {k:float(k==y) for k,_ in opts}
                row={'id':f'dev-{block_id}-{edit}','source':block_id,'family':family,'edit':edit,'state':state,'question':question,'options':[{'id':k,'description':d} for k,d in opts],'expected':y,'target_probs':p,'reference':ref}
                rows.append(row)
    assert len(rows)==72 and len({r['id'] for r in rows})==72
    assert len({digest({'state':r['state'],'question':r['question'],'options':r['options']}) for r in rows})==72
    return rows

def visible(row):
    return {k:copy.deepcopy(row[k]) for k in ('id','state','question','options')}

if __name__=='__main__':
    r=development(); print(json.dumps({'n':len(r),'source_blocks':len({x['source'] for x in r}),'sha256':digest(r),'families':{f:sum(x['family']==f for x in r) for f in FAMILIES}},indent=2))
