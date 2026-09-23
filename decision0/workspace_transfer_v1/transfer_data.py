"""Independent, bounded transfer study. Synthetic sources; not population coverage.
Only visible(row) is allowed into prompts. References are construction/audit only.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import hashlib, json, random, re

FAMILIES=('amount','temporal','linkage','policy','judging','probability')
SEED=92318117

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def visible(r):
    return {k:r[k] for k in ('id','state','question','options')}

def tag(rng):return ''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(8))
def dollar(c):return ('-' if c<0 else '')+f'${abs(c)//100}.{abs(c)%100:02d}'
def relation(x,y):return 'below' if x<y else 'above' if x>y else 'equal'

def prob_reference(kind,f):
    if kind=='conditional':
        return [Fraction(f['target_counts'][j],sum(f['target_counts'])) for j in range(3)]
    if kind=='mixture':
        w=Fraction(f['mix_n'],f['mix_d'])
        return [w*Fraction(f['a'][j],sum(f['a']))+(1-w)*Fraction(f['b'][j],sum(f['b'])) for j in range(3)]
    if kind=='without_replacement':
        a,b,c=f['counts'];n=a+b+c
        return [Fraction(a*(a-1)+b*(b-1)+c*(c-1),n*(n-1)),Fraction(n*(n-1)-(a*(a-1)+b*(b-1)+c*(c-1)),n*(n-1))]
    raise ValueError(kind)

def amount_value(kind,f):
    raw=sum(q*p for q,p in f['items'])
    if kind=='credit':return raw+f['shipping']-f['credit']
    if kind=='discount':
        discounted=(Decimal(raw)*Decimal(100-f['discount_pct'])/100).quantize(Decimal('1'),rounding=ROUND_HALF_UP)
        return int(discounted)+f['shipping']-f['credit']
    if kind=='per_line':
        discounted=sum(int((Decimal(q*p)*Decimal(100-f['discount_pct'])/100).quantize(Decimal('1'),rounding=ROUND_HALF_UP)) for q,p in f['items'])
        return discounted+f['shipping']-f['credit']
    raise ValueError(kind)

def policy_value(kind,f):
    if kind=='override':
        if f['active']=='no':return 'deny'
        if f['active']=='unrecorded':return 'unknown'
        if f['hold']=='no' or f['clearance']=='yes':return 'allow'
        if f['hold']=='yes' and f['clearance']=='no':return 'deny'
        return 'unknown'
    if kind=='precedence':
        if f['withdrawn']=='yes':return 'deny'
        if f['withdrawn']=='unrecorded':return 'unknown'
        if f['exception']=='yes':return 'allow'
        if f['exception']=='unrecorded':return 'unknown'
        if f['region'] not in ('east','west'):return 'review'
        return 'allow' if f['certified']=='yes' else 'deny' if f['certified']=='no' else 'unknown'
    if kind=='scope':
        if f['class']=='restricted':return 'deny'
        if f['class']=='unrecorded':return 'unknown'
        if f['class']=='public':return 'allow'
        return 'allow' if f['consent']=='yes' else 'deny' if f['consent']=='no' else 'unknown'
    raise ValueError(kind)

def context(rng,records,long,domain):
    """Relevant records are dispersed among explicitly other-subject records."""
    others=[]
    for _ in range(30 if long else 3):
        name=tag(rng)
        templates=[
          f'Other subject {name}: the registered owner is office {tag(rng)}. A prior review remains archived. Its certified status is {rng.choice(["yes","no"])} and its region is {rng.choice(["east","west","north"])}. These facts apply to {name} only.',
          f'Other transaction {name}: {rng.randint(2,17)} units at {dollar(rng.randint(120,2300))} per unit; credit {dollar(rng.randint(20,510))}; handling fee {dollar(rng.randint(10,200))}. No quantity or value in this entry belongs to any differently named transaction.',
          f'Archive record {name}: created 2025-05-14T10:10:00+02:00 and reviewed 2025-05-15T18:40:00+02:00. Its local processing rule permits an interval of {rng.randint(2,80)} hours. This is a separate case, not a default rule for other subjects.',
          f'Routing registry {name}: team {tag(rng)} uses depot {tag(rng)}. The contact office is {rng.choice(["orchard","harbor","cedar","copper"])}. Legacy identifiers in that registry are preserved for archival lookup but are not aliases for other current records.'
        ]
        others.append(rng.choice(templates))
    rng.shuffle(records)
    positions=sorted(rng.sample(range(len(others)+len(records)),len(records)))
    full=[];a=iter(records);b=iter(others)
    for k in range(len(others)+len(records)):full.append(next(a) if k in positions else next(b))
    return '\n\n'.join(f'Document entry {k+1}. {s}' for k,s in enumerate(full))

def source_pair(split,family,i):
    rng=random.Random(int(digest([SEED,split,family,i])[:16],16))
    name=tag(rng);src=digest(['source',SEED,split,family,i])[:24]
    long=(i%2==1)
    forms=[];refs=[];choices=[];derived=[]
    if family=='amount':
        kind=('credit','discount','per_line')[i%3]
        f={'items':[(rng.randint(2,9),rng.randint(112,1999)) for _ in range(3)],'shipping':rng.randint(35,240),'credit':rng.randint(0,250),'discount_pct':rng.choice([7,13,19])}
        if i in (4,5):f['credit']=sum(q*p for q,p in f['items'])+1000
        boundary=amount_value(kind,f);ends=rng.sample([-1,0,1],2)
        records=[f'Purchase {name}: line items are '+json.dumps([{'quantity':q,'unit_cents':p} for q,p in f['items']])+'.',
                 f'Purchase {name}: shipping {{EDIT}} cents; credit {f["credit"]} cents.',
                 f'Purchase {name}: comparison budget {boundary} cents. Signed totals are permitted. This budget is not an arithmetic input.']
        rule={'credit':'Sum quantity times unit price for every line, add shipping, then subtract the credit. No discount applies.',
              'discount':f'Sum all line amounts first, then reduce that subtotal by {f["discount_pct"]} percent. Round that discounted subtotal once to the nearest cent, with a half cent rounded upward. Add shipping and subtract credit afterwards.',
              'per_line':f'Reduce each individual line amount by {f["discount_pct"]} percent, rounding each discounted line to the nearest cent with half upward. Then sum those rounded lines, add shipping and subtract the credit. Do not discount the combined subtotal.'}[kind]
        records.append(f'Calculation instruction applicable to {name}: {rule}')
        text=context(rng,records,long,family)
        question=f'For purchase {name}, compare the final signed total with its comparison budget using its own calculation instruction. A cent is the unit throughout; ignore other subjects.'
        for e in ends:
            ff=dict(f,shipping=f['shipping']+e);value=amount_value(kind,ff)
            forms.append(text.replace('{EDIT}',str(ff['shipping'])));refs.append(relation(value,boundary));derived.append({'kind':kind,'facts':ff,'boundary':boundary,'value':value})
        choices=[('below','The final total is strictly below the budget.'),('equal','The final total is exactly equal to the budget.'),('above','The final total is strictly above the budget.')]
    elif family=='temporal':
        kind=('elapsed','paused','deadline')[i%3]
        zone=timezone(timedelta(minutes=rng.choice([-300,60,330])))
        start=datetime(2027 if split=='development' else 2028,rng.randint(1,11),rng.randint(2,24),22,25,tzinfo=zone)
        allowed=rng.choice([95,365,750,1530]);pause=37 if kind=='paused' else 0
        threshold=start+timedelta(minutes=allowed+pause)
        zone2=timezone(timedelta(minutes=rng.choice([-210,90,345])))
        records=[f'Service case {name}: start {start.isoformat()}. Allowed counted time is {allowed} minutes.',f'Service case {name}: received timestamp {{EDIT}}.',
                 f'Case {name}: subtract {pause} minutes of approved pause from elapsed time. All explicit UTC offsets must be respected. Count ordinary elapsed minutes, not business days.']
        if kind=='deadline':
            records=[f'Case {name}: official deadline is {threshold.astimezone(timezone.utc).isoformat()}.',f'Case {name}: received timestamp {{EDIT}}.',f'An earlier notice for {name} stated a deadline two hours earlier; that notice has been superseded by the official deadline above.']
        text=context(rng,records,long,family)
        question=f'Is receipt for case {name} before, exactly at, or after its current allowed deadline? '+('Use its official deadline, not the superseded notice.' if kind=='deadline' else 'The deadline is the start plus allowed counted time plus the approved pause.')
        for e in rng.sample([-1,0,1],2):
            event=(threshold+timedelta(minutes=e)).astimezone(zone2)
            forms.append(text.replace('{EDIT}',event.isoformat()));refs.append('before' if e<0 else 'after' if e>0 else 'at');derived.append({'start':start.isoformat(),'allowed':allowed,'pause':pause,'event':event.isoformat(),'deadline':threshold.isoformat(),'gap_minutes':e})
        choices=[('before','Receipt precedes the current deadline.'),('at','Receipt is exactly at the current deadline.'),('after','Receipt is later than the current deadline.')]
    elif family=='linkage':
        contracts=[tag(rng) for _ in range(6)];nodes=[tag(rng) for _ in range(6)];depots=[tag(rng) for _ in range(6)]
        destinations=rng.sample(['amber','cedar','cobalt','coral','harbor','indigo','orchard','pearl','sage','silver'],6)
        records=[f'Customer {name}: current contract {{EDIT}}. Use that identifier, not a similar customer name.']
        records += [f'Contract {a}: current service node {b}.' for a,b in zip(contracts,nodes)]
        records += [f'Node {a}: currently handled by depot {b}.' for a,b in zip(nodes,depots)]
        records += [f'Depot {a}: its destination is {b}.' for a,b in zip(depots,destinations)]
        if i%3==1:records.append(f'An archived route for customer {name} listed {destinations[0]}. It is superseded; current mappings control.')
        if i%3==2:records += [f'Contract {contracts[k]}: predecessor node {nodes[(k+1)%6]} is obsolete.' for k in range(3)]
        text=context(rng,records,long,family);question=f'Which destination applies to customer {name}? Follow current customer → contract → service node → depot → destination links, not archived/predecessor assignments.'
        for k in rng.sample(range(6),2):
            forms.append(text.replace('{EDIT}',contracts[k]));refs.append(destinations[k]);derived.append({'contract':contracts[k],'node':nodes[k],'depot':depots[k],'destination':destinations[k]})
        choices=[(v,f'The current destination is {v}.') for v in destinations]
    elif family=='policy':
        kind=('override','precedence','scope')[i%3]
        if kind=='override':
            f={'active':'yes','hold':'yes','clearance':'no'};driver='clearance';values=rng.sample(['yes','no','unrecorded'],2)
            rules=['Current rule: an active subject is permitted when no hold exists or an audit clearance exists. An audit clearance overrides a hold; it does not override inactivity.',
                   'Reject when inactivity is recorded, or a hold is recorded and clearance is recorded as no. Otherwise, when relevant facts are unrecorded, report insufficient evidence.',
                   'Obsolete rule: all held subjects were rejected. The current rule supersedes that rule.']
        elif kind=='precedence':
            f={'withdrawn':'no','exception':'no','region':'north','certified':'yes'};driver='exception';values=['yes','no']
            rules=['Apply the following current rules in order. Recorded withdrawal requires rejection. If withdrawal is unrecorded, the outcome is insufficient evidence.',
                   'Otherwise a recorded special exception permits. If exception status is unrecorded, the outcome is insufficient evidence.',
                   'If no exception applies, subjects outside east or west require manual review, regardless of certification. Within east/west, certification=yes permits and certification=no rejects; missing certification is insufficient evidence.']
        else:
            f={'class':'internal','consent':'no'};driver='consent';values=rng.sample(['yes','no','unrecorded'],2)
            rules=['Use current classification, not historical classification. Restricted material is rejected. Public material is permitted. Internal material is permitted only with recorded consent=yes, rejected with consent=no, and otherwise has insufficient evidence.',
                   'Historical classification records do not override the current one. Unknown classification itself means insufficient evidence.']
        records=[f'Policy applicable to subject {name}: '+v for v in rules]
        records += [f'Current subject {name}: {key}='+('{EDIT}' if key==driver else v)+'.' for key,v in f.items()]
        records.append(f'Historical subject {name}: a prior reviewer approved a different request. That approval is not current permission.')
        text=context(rng,records,long,family);question=f'Assess current permission for subject {name}, applying the supplied policy and its explicit priorities. Unrecorded is not the same as no. Use only this subject’s facts.'
        for v in values:
            ff=dict(f,**{driver:v});forms.append(text.replace('{EDIT}',v));refs.append(policy_value(kind,ff));derived.append({'kind':kind,'facts':ff,'decision':policy_value(kind,ff)})
        choices=[('allow','Permission is established by the current rule.'),('deny','A current rejection condition is established.'),('review','The current rule requires manual review.'),('unknown','The evidence is insufficient under the current rule.')]
    elif family=='judging':
        kind=('calculation','qualification','entity')[i%3]
        if kind=='calculation':
            q=rng.randint(3,15);p=rng.randint(110,1850);fee=rng.randint(12,230);credit=rng.randint(2,140);correct=q*p+fee-credit
            records=[f'Reference invoice {name}: {q} units priced at {p} cents each, a handling fee of {fee} cents, and a credit of {credit} cents. Final amount is units times unit price plus fee minus credit.',
                     f'Question posed to the candidate: what is the final amount for invoice {name}?', 'Candidate response: {EDIT}']
            values=[f'The final amount is {correct} cents.',f'The final amount is {correct+rng.choice([1,10,100])} cents.']
            answers=['correct','numeric_error']
        elif kind=='qualification':
            records=[f'Current record for request {name}: review passed. Release still requires a countersignature; no countersignature has been recorded.',f'Question posed to the candidate: is request {name} authorized for immediate release?', 'Candidate response: {EDIT}']
            values=['Not yet. Passing review alone is insufficient; the required countersignature has not been recorded.','Yes. The review passed, so the request is authorized for immediate release.'];answers=['correct','rule_error']
        else:
            other=tag(rng);a,b=rng.sample(['archived','approved','pending','withdrawn'],2)
            records=[f'Current record: request {name} is {a}.',f'Current record: request {other} is {b}.',f'Question posed to the candidate: what is the current status of {name}?','Candidate response: {EDIT}']
            values=[f'Request {name} is {a}.',f'Request {name} is {b}.'];answers=['correct','wrong_evidence']
        text=context(rng,records,long,family);question='Judge the candidate response against the supplied question and reference evidence. Use correct only if its substantive conclusion is supported. Select the category of its decisive error, not its prose style.'
        for value,answer in zip(values,answers):forms.append(text.replace('{EDIT}',value));refs.append(answer);derived.append({'kind':kind,'candidate':value,'decision':answer})
        choices=[('correct','The response is substantively correct and supported.'),('numeric_error','The decisive error is numerical computation.'),('rule_error','The response ignores a required rule or qualification.'),('wrong_evidence','The response attributes another subject’s evidence or makes an unsupported factual claim.')]
    elif family=='probability':
        kind=('conditional','mixture','without_replacement')[i%3]
        if kind=='conditional':
            counts=[rng.randint(9,25) for _ in range(3)];other=[rng.randint(25,60) for _ in range(3)]
            records=[f'Population {name}, screened members: alpha={{EDIT}}, beta={counts[1]}, gamma={counts[2]}. These are complete, mutually exclusive counts.',
                     f'Population {name}, unscreened members: alpha={other[0]}, beta={other[1]}, gamma={other[2]}.',
                     'Sampling selects uniformly from screened members only. Unscreened members cannot be sampled.']
            question=f'For one uniform draw from screened members of population {name}, provide the probability distribution over alpha, beta, and gamma. Condition on screened membership; do not pool in unscreened members.'
            facts=[{'target_counts':[v,counts[1],counts[2]],'other':other} for v in (3,70)]
            values=['3','70'];labels=['alpha','beta','gamma']
        elif kind=='mixture':
            a=[rng.randint(4,22) for _ in range(3)];b=[rng.randint(5,25) for _ in range(3)]
            a[0]+=70;b[2]+=70
            records=[f'Station {name}-A has alpha={a[0]}, beta={a[1]}, gamma={a[2]}. Station {name}-B has alpha={b[0]}, beta={b[1]}, gamma={b[2]}. Each table is complete.',
                     f'Sampling protocol for {name}: choose station A with probability {{EDIT}} out of 10, otherwise station B. Within the selected station, choose one member uniformly.',
                     'Station choice follows the stated protocol, not the relative station populations.']
            question=f'What is the distribution of alpha, beta and gamma under sampling protocol {name}? Account for both the station-selection probabilities and the conditional frequencies.'
            facts=[{'mix_n':v,'mix_d':10,'a':a,'b':b} for v in (2,8)];values=['2','8'];labels=['alpha','beta','gamma']
        else:
            b,c=rng.randint(3,10),rng.randint(4,9)
            records=[f'Container {name} contains alpha={{EDIT}}, beta={b}, gamma={c}. Each item has exactly one category.',
                     'Draw two distinct items uniformly without replacement. The second draw cannot return the first item.']
            question=f'For two draws without replacement from container {name}, give the probabilities that their categories are the same or different. These outcomes are mutually exclusive and exhaustive.'
            facts=[{'counts':[v,b,c]} for v in (3,75)];values=['3','75'];labels=['same','different']
        text=context(rng,records,long,family)
        choices=[(k,f'The sampled outcome is {k}.' if kind!='without_replacement' else f'The two categories are {k}.') for k in labels]
        for value,f in zip(values,facts):
            q=prob_reference(kind,f);forms.append(text.replace('{EDIT}',value));refs.append(dict(zip(labels,map(float,q))));derived.append({'kind':kind,'facts':f,'fractions':{k:str(v) for k,v in zip(labels,q)}})
    else:raise ValueError(family)
    rng.shuffle(choices)
    options=[{'id':k,'description':v} for k,v in choices]
    result=[]
    for e,(state,ref,oracle) in enumerate(zip(forms,refs,derived)):
        target=ref if isinstance(ref,dict) else {k:float(k==ref) for k,_ in choices}
        expected=max(sorted(target),key=target.get)
        result.append({'id':digest([src,e])[:24],'source':src,'family':family,'tier':'independent','split':split,'edit':e,
                       'long_context':long,'state':state,'question':question,'options':options,'expected':expected,
                       'target_probs':target,'gold_probs':target if family=='probability' else None,'oracle':oracle})
    assert forms[0]!=forms[1]
    if family!='probability':assert refs[0]!=refs[1],(family,i,refs)
    return result

def corpus(split):
    if split not in ('development','held'):raise ValueError(split)
    return [r for f in FAMILIES for i in range(6) for r in source_pair(split,f,i)]

def validate():
    seen=set();result={}
    for split in ('development','held'):
        rows=corpus(split)
        assert len(rows)==72 and len({r['id'] for r in rows})==72
        for r in rows:
            key=digest({k:visible(r)[k] for k in ('state','question','options')})
            assert key not in seen;seen.add(key)
            assert r['expected'] in [o['id'] for o in r['options']]
            assert abs(sum(r['target_probs'].values())-1)<1e-12
            assert min(r['target_probs'].values())>=0
            assert r['id'] not in r['state'] and r['source'] not in r['state']
            assert '{EDIT}' not in r['state']
        for src in {r['source'] for r in rows}:
            a,b=[r for r in rows if r['source']==src]
            assert a['question']==b['question'] and a['options']==b['options']
            assert sum(x!=y for x,y in zip(a['state'].split('\n\n'),b['state'].split('\n\n')))==1
        result[split]={'rows':len(rows),'source_situations':len({r['source'] for r in rows}),'sha256':digest(rows),
                       'families':dict(Counter(r['family'] for r in rows)),'long_cases':sum(r['long_context'] for r in rows),
                       'max_characters':max(len(r['state']) for r in rows)}
    return result

if __name__=='__main__':print(json.dumps(validate(),indent=2))
