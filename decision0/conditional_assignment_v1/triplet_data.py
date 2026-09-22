"""Off-benchmark matched counterfactual triplets with independently checkable labels.

A triplet holds criterion, outcomes, identities, record layout and distractors
fixed, changing exactly one source field. Oracle metadata is never model input.
This is controlled synthetic data, not independently authored business evidence.
"""
from __future__ import annotations
import hashlib, json, random, re, sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import cases as replay

FAMILIES=('amount','elapsed','join','policy')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def visible(r):return {k:r[k] for k in ('id','state','question','options')}
def tag(rng):return ''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(7))
def amount_text(cents):return f'${cents//100}.{cents%100:02d}'

def make_group(split,family,index):
    rng=random.Random(int(digest(['matched-triplets-v1',split,family,index])[:16],16))
    key=tag(rng); group_id=digest([split,family,index])[:24]
    distractors=2 if index%2==0 else 12
    records=[];states=[];oracles=[]
    if family=='amount':
        count=rng.randint(2,13);unit=rng.randint(12,190);fee=rng.randint(45,120);credit=rng.randint(0,40)
        cap=count*unit+fee-credit;gap=rng.choice([1,5,17] if split=='train' else [1,7,23])
        records=[f'Order {key}: quantity {count}; price per unit {amount_text(unit)}.',
                 f'Order {key}: mandatory handling fee {{VALUE}}.',
                 f'Order {key}: credit {amount_text(credit)}; spending cap {amount_text(cap)}.']
        for _ in range(distractors):
            other=tag(rng);records.append(f'Order {other}: quantity {rng.randint(2,20)}; price per unit {amount_text(rng.randint(1,350))}; handling fee {amount_text(rng.randint(1,70))}; credit $0.00; spending cap $15.00.')
        values=[amount_text(fee+d) for d in (-gap,0,gap)]
        answers=['below','equal','above']
        opts=[('below','The final order total is strictly below its cap.'),('equal','The final order total is exactly equal to its cap.'),('above','The final order total is strictly above its cap.')]
        question=f'For order {key}, the total is quantity multiplied by price per unit, plus its mandatory handling fee, minus its credit. Compare the exact total with that order\'s cap. Other orders do not contribute. A dollar is 100 cents; do not round intermediates.'
        oracles=[{'key':key,'quantity':count,'unit_cents':unit,'fee_cents':fee+d,'credit_cents':credit,'cap_cents':cap,'signed_gap_cents':d} for d in (-gap,0,gap)]
    elif family=='elapsed':
        start=datetime(2027 if split=='train' else 2028,rng.randint(1,11),rng.randint(1,20),rng.randint(0,22),rng.randint(0,59),tzinfo=timezone(timedelta(minutes=rng.choice([-300,0,120]))))
        duration=rng.choice([95,170,480,755,1190,1515]);gap=rng.choice([1,5,17] if split=='train' else [1,7,23])
        display_zone=timezone(timedelta(minutes=rng.choice([-240,60,330] if split=='train' else [-210,90,345])))
        deadline=start+timedelta(minutes=duration)
        records=[f'Case {key}: start {start.isoformat()}; allowed elapsed interval {duration} minutes.',
                 f'Case {key}: submission {{VALUE}}.',
                 f'Case {key}: use elapsed time, not business time. Explicit UTC offsets control.']
        for _ in range(distractors):
            other=tag(rng);dt=start+timedelta(minutes=rng.randint(-200,2000));records.append(f'Case {other}: start {dt.isoformat()}; allowed elapsed interval {rng.randint(30,2000)} minutes; status pending. This case has a separate deadline.')
        values=[(deadline+timedelta(minutes=d)).astimezone(display_zone).isoformat() for d in (-gap,0,gap)]
        answers=['before','at','after'];opts=[('before','Submission is before the deadline.'),('at','Submission is exactly at the deadline.'),('after','Submission is after the deadline.')]
        question=f'Classify submission {key} relative to its start plus its allowed elapsed interval. Use only its own records and respect all explicit UTC offsets.'
        oracles=[{'key':key,'start':start.isoformat(),'duration_minutes':duration,'submission':v,'signed_gap_minutes':d} for d,v in zip((-gap,0,gap),values)]
    elif family=='join':
        k=6;contracts=[tag(rng) for _ in range(k)];teams=[tag(rng) for _ in range(k)]
        pools=rng.sample(['amber','cobalt','sage','coral','plum','silver','ivory','ochre','teal','umber','pearl','indigo'],k)
        rng.shuffle(teams);rng.shuffle(pools)
        records=[f'Person {key}: current contract {{VALUE}}.']
        records += [f'Contract {a}: service team {b}.' for a,b in zip(contracts,teams)]
        records += [f'Team {b}: destination {c}.' for b,c in zip(teams,pools)]
        for _ in range(distractors):records.append(f'Person {tag(rng)}: current contract {rng.choice(contracts)}. This is not person {key}.')
        selected=rng.sample(range(k),3);values=[contracts[i] for i in selected];answers=[pools[i] for i in selected]
        opts=[(p,f'The destination is {p}.') for p in pools]
        question=f'Route person {key}. Follow person to current contract, contract to service team, and team to destination. All mappings are current. Do not use another person\'s contract.'
        oracles=[{'key':key,'contract':contracts[i],'team':teams[i],'destination':pools[i]} for i in selected]
    elif family=='policy':
        driver=rng.choice(['active','clearance','hold'])
        fixed={'active':{'hold':'no','clearance':'no'},'clearance':{'active':'yes','hold':'yes'},'hold':{'active':'yes','clearance':'no'}}[driver]
        values=['yes','no','unrecorded'] if driver!='hold' else ['no','yes','unrecorded']
        fields=['active','hold','clearance'];records=[f'Account {key}: {name} '+('{VALUE}' if name==driver else fixed[name])+'.' for name in fields]
        records += [f'Account {key}: the current export rule is active AND (NOT hold OR clearance).',
                    'The current rule has three evidence states. Permit only when the rule is established. Reject when inactive is recorded, or when both hold=yes and clearance=no are recorded. Otherwise report insufficient evidence.',
                    'An obsolete rule rejected every held account, including cleared accounts. The obsolete rule must not be used.']
        for _ in range(distractors):records.append(f'Account {tag(rng)}: active={rng.choice(["yes","no","unrecorded"])}; hold={rng.choice(["yes","no"])}; clearance={rng.choice(["yes","no"])}. Each account is assessed separately.')
        answers=['allow','deny','unknown'];opts=[('allow','The current permission conditions are established.'),('deny','The current evidence establishes a disqualifying condition.'),('unknown','The evidence is insufficient to establish permission or a disqualifying condition.')]
        question=f'Assess whether account {key} may export under the current rule. Unrecorded does not mean no. Ignore the obsolete rule and facts about other accounts.'
        oracles=[{'key':key,**fixed,driver:v} for v in values]
    else:raise ValueError(family)
    assert sum('{VALUE}' in s for s in records)==1
    rng.shuffle(records);rng.shuffle(opts)
    options=[{'id':k,'description':d} for k,d in opts]
    # The same layout, criterion and outcome order are used at all three endpoints.
    variant_order=list(range(3));rng.shuffle(variant_order)
    out=[]
    for e in variant_order:
        state='\n'.join(f'Entry {i+1}. '+s.replace('{VALUE}',values[e]) for i,s in enumerate(records))
        if split=='held':state='Current records follow. Evaluate the requested subject only.\n'+state
        ans=answers[e]
        out.append({'id':digest([group_id,e])[:24],'source':group_id,'family':family,'tier':'new-held' if split=='held' else 'new-train','split':split,'edit':e,'state':state,'question':question,'options':options,'expected':ans,'target_probs':{o['id']:float(o['id']==ans) for o in options},'oracle':oracles[e],'matched':True,'long_context':distractors==12})
    return out

def groups(split):
    n=6 if split=='train' else 8
    out=[make_group(split,fam,i) for fam in FAMILIES for i in range(n)]
    language=[r for r in replay.corpus(split) if r['family']=='snli']
    language=sorted(language,key=lambda r:digest(['triplet-replay',r['id']]))[:24]
    for i in range(0,len(language),3):
        group=[]
        for r in language[i:i+3]:
            r=dict(r);r['matched']=False;r['tier']='new-held' if split=='held' else 'new-train';r['long_context']=False;group.append(r)
        out.append(group)
    return out

def corpus(split):return [r for g in groups(split) for r in g]

def reference_from_text(row):
    """Independent text-parsing checker; never imported by inference scoring."""
    state=row['state'];key=row['oracle']['key'];family=row['family']
    if family=='amount':
        n,p=re.search(rf'Order {key}: quantity (\d+); price per unit (\$\d+\.\d{{2}})',state).groups()
        fee=re.search(rf'Order {key}: mandatory handling fee (\$\d+\.\d{{2}})',state).group(1)
        credit,cap=re.search(rf'Order {key}: credit (\$\d+\.\d{{2}}); spending cap (\$\d+\.\d{{2}})',state).groups()
        total=int(n)*Decimal(p[1:])+Decimal(fee[1:])-Decimal(credit[1:]);cap=Decimal(cap[1:])
        return 'below' if total<cap else 'above' if total>cap else 'equal'
    if family=='elapsed':
        start,window=re.search(rf'Case {key}: start ([^;\s]+); allowed elapsed interval (\d+) minutes',state).groups()
        event=re.search(rf'Case {key}: submission ([^\s]+)\.',state).group(1)
        delta=(datetime.fromisoformat(event)-datetime.fromisoformat(start)).total_seconds()-int(window)*60
        return 'before' if delta<0 else 'after' if delta>0 else 'at'
    if family=='join':
        contract=re.search(rf'Person {key}: current contract ([A-Z]+)\.',state).group(1)
        team=re.search(rf'Contract {contract}: service team ([A-Z]+)\.',state).group(1)
        return re.search(rf'Team {team}: destination ([a-z]+)\.',state).group(1)
    if family=='policy':
        vals={k:re.search(rf'Account {key}: {k} (yes|no|unrecorded)\.',state).group(1) for k in ('active','hold','clearance')}
        if vals['active']=='no' or (vals['hold']=='yes' and vals['clearance']=='no'):return 'deny'
        if vals['active']=='yes' and (vals['hold']=='no' or vals['clearance']=='yes'):return 'allow'
        return 'unknown'
    raise ValueError(family)

def checks():
    seen=set();out={}
    for split in ('train','held'):
        gs=groups(split);rows=[r for g in gs for r in g]
        assert len(rows)==(96 if split=='train' else 120)
        for r in rows:
            key=digest(visible(r)|{'id':''})
            assert key not in seen,('Duplicate visible request',r['id']);seen.add(key)
            assert abs(sum(r['target_probs'].values())-1)<1e-12
            assert set(visible(r))=={'id','state','question','options'}
            if r['family']!='snli':assert reference_from_text(r)==r['expected'],r['id']
        for g in gs:
            if not g[0]['matched']:continue
            assert len(g)==3 and len({r['expected'] for r in g})==3
            assert g[0]['options']==g[1]['options']==g[2]['options']
            assert g[0]['question']==g[1]['question']==g[2]['question']
            for a,b in ((g[0],g[1]),(g[0],g[2])):
                assert sum(x!=y for x,y in zip(a['state'].splitlines(),b['state'].splitlines()))==1
        out[split]={'rows':len(rows),'matched_sources':sum(g[0]['matched'] for g in gs),'groups':len(gs),'sha256':digest(rows),'max_characters':max(len(r['state']+r['question']) for r in rows),'long_rows':sum(r['long_context'] for r in rows)}
    return out

if __name__=='__main__':print(json.dumps(checks(),indent=2))
