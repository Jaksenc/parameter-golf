"""Independent input construction. No benchmark metadata enters model prompts."""
import json,random,datetime as dt,hashlib
from fractions import Fraction
from pathlib import Path
FAMILIES=('cashflow','calendar','route','recurrence','sampling','threshold')

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def new_tasks(partition):
    rng=random.Random(930571 if partition=='calibration' else 2819043)
    count=8 if partition=='calibration' else 16;out=[]
    for family in FAMILIES:
        for i in range(count):
            k=2+i%4
            if family=='cashflow':
                start=rng.randint(120,650);changes=[rng.choice((-1,1))*rng.randint(4,37) for _ in range(3+i%7)]
                gold=start+sum(changes)
                state=f'A register initially holds {start} units. In chronological order, the adjustments are '+', '.join(('add ' if x>=0 else 'remove ')+str(abs(x)) for x in changes)+'. There are no other adjustments.'
                inst='What is the exact final register balance?'
                check=start
                for x in changes:check+=x
                assert gold==check
            elif family=='calendar':
                start=dt.date(2023+rng.randrange(4),rng.randint(1,12),rng.randint(1,20));n=rng.randint(8,80)
                end=start+dt.timedelta(days=n)
                state=f'The interval starts on {start.isoformat()} and ends on {end.isoformat()}. Count elapsed calendar days, not business days. Do not count the starting day twice.'
                inst='How many calendar days elapsed?';gold=n;assert gold==(end-start).days
            elif family=='route':
                names=[f'place_{j}' for j in range(7)];order=names[:];rng.shuffle(order);links={order[j]:order[(j+1)%7] for j in range(7)}
                start=rng.choice(names);n=rng.randint(2,5) if partition=='calibration' else rng.randint(6,20)
                state={'initial_place':start,'transitions_to_take':n,'next_place':links};inst='Follow next_place exactly transitions_to_take times from initial_place. Where do you finish?'
                gold=order[(order.index(start)+n)%7];x=start
                for _ in range(n):x=links[x]
                assert gold==x
            elif family=='recurrence':
                x,a,b,m=rng.randrange(19),rng.randint(2,5),rng.randint(1,7),rng.choice((19,23,29));n=rng.randint(2,4) if partition=='calibration' else rng.randint(5,10)
                state=f'Start with x={x}. Repeat this rule exactly {n} times: replace x by the remainder of ({a} times x + {b}) divided by {m}. Remainders are nonnegative.'
                inst='What is x after all updates?';gold=(a**n*x+b*sum(a**j for j in range(n)))%m
                xx=x
                for _ in range(n):xx=(a*xx+b)%m
                assert xx==gold
            elif family=='sampling':
                a,b,c=[rng.randint(2,15) for _ in range(3)];n=a+b+c
                state={'red':a,'blue':b,'green':c,'protocol':'Sample two objects uniformly without replacement. Each object has equal chance.'}
                inst='What is the exact probability that both sampled objects are red?';gold=Fraction(a*(a-1),n*(n-1));assert gold==Fraction(a,n)*Fraction(a-1,n-1)
            else:
                limit=rng.randint(30,90);age=rng.randint(1,120);verified=bool(rng.getrandbits(1));urgent=bool(rng.getrandbits(1))
                state={'age_days':age,'verified':verified,'urgent':urgent,'maximum_age_days':limit,'policy':'Eligible exactly when verified AND (age_days <= maximum_age_days OR urgent).'}
                inst='Is the case eligible under the policy?';gold=bool(verified and (age<=limit or urgent));k=2
                assert gold==(verified and not(age>limit and not urgent))
            if family=='route':wrong=rng.sample([v for v in names if v!=gold],k-1)
            elif family=='sampling':
                candidates={Fraction(a,n),Fraction(a*a,n*n),Fraction(a*a,n*(n-1)),Fraction(b*(b-1),n*(n-1)),Fraction(c,n),Fraction(1,n),Fraction(0),Fraction(1)}-{gold}
                wrong=rng.sample(sorted(candidates),k-1)
            elif family=='threshold':wrong=[not gold]
            else:wrong=rng.sample([gold+d for d in (-13,-7,-3,-1,1,3,7,13) if gold+d>=0],k-1)
            values=[gold]+wrong;rng.shuffle(values);labels=[f'v{j}' for j in range(k)]
            text=lambda x: str(x) if not isinstance(x,bool) else 'eligible' if x else 'not eligible'
            row={'id':f'witness-{partition}-{family}-{i:03d}','state':state,'question':{'type':'choice','instructions':inst,'criteria':dict(zip(labels,map(text,values)))},'labels':labels,'expected':labels[values.index(gold)],'family':family,'partition':partition,'reference_value':str(gold)}
            out.append(row)
    return out

def safe(r):return {k:r[k] for k in ('id','state','question','labels')}

def population(root):
    from orbit_probe import challenge
    benchmark=json.loads((Path(root)/'benchmark/tasks.json').read_text())
    rows=[{**safe(r),'partition':'jevbench'} for r in benchmark]
    rows += [{**safe(r),'partition':'prior_stress'} for r in challenge()]
    rows += [{**safe(r),'partition':r['partition']} for p in ('calibration','fresh') for r in new_tasks(p)]
    assert len(rows)==503 and len({r['id'] for r in rows})==503
    assert len({digest({k:r[k] for k in ('state','question','labels')}) for r in rows})==503
    return rows

def partition(root,shards=16):
    bins=[[] for _ in range(shards)];cost=[0.]*shards
    for r in sorted(population(root),key=lambda r:(-len(json.dumps(safe(r)))**1.3,r['id'])):
        j=min(range(shards),key=lambda i:(cost[i],i));bins[j].append(r);cost[j]+=1000+len(json.dumps(safe(r)))**1.3
    return bins
