"""Executable counterfactual environments. All four members travel together.
No benchmark answer, teacher text, or normalized state is included in student inputs.
"""
from __future__ import annotations
import datetime as dt
import hashlib, json, random
from fractions import Fraction

FAMILIES=('conjunction','override_veto','quantity_permission','temporal_receipt',
          'authoritative_update','mean_total','classify_execute','rule_inference')

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def sha(x):
    return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()

def evaluate(expr,values):
    """Reference interpreter for a small typed expression language, not Python eval."""
    if isinstance(expr,dict):
        if set(expr)=={'var'}:return values[expr['var']]
        if set(expr)=={'const'}:return expr['const']
        op=expr['op'];args=[evaluate(a,values) for a in expr['args']]
        if op=='and':return all(args)
        if op=='or':return any(args)
        if op=='not':return not args[0]
        if op=='ge':return args[0]>=args[1]
        if op=='gt':return args[0]>args[1]
        if op=='le':return args[0]<=args[1]
        if op=='lt':return args[0]<args[1]
        if op=='add':return sum(args)
        if op=='sub':return args[0]-args[1]
        if op=='mul':
            p=1
            for a in args:p*=a
            return p
        if op=='div':return Fraction(args[0],args[1])
        if op=='if':return args[1] if args[0] else args[2]
        raise ValueError('unsupported oracle operation '+str(op))
    raise ValueError('invalid oracle expression')

def V(n):return {'var':n}
def C(x):return {'const':x}
def E(op,*args):return {'op':op,'args':list(args)}
def yn(v):return 'present' if v else 'absent'

def envelope(state,question,options,target,seed):
    rng=random.Random(seed)
    ids=['opt_'+sha(f'{seed}:{i}')[:10] for i in range(len(options))]
    order=list(range(len(options)));rng.shuffle(order)
    task={'state':state,'question':{'type':'choice','instructions':question,
          'criteria':{ids[i]:options[i] for i in order}},'labels':[ids[i] for i in order]}
    return task,ids[target]

def group(family,split,index,seed=92437):
    rng=random.Random(sha(f'{seed}:{split}:{family}:{index}'))
    key=f'{split}/{family}/{index}'
    name='Unit-'+sha(key)[:7]
    a=rng.randrange(8,55);b=rng.randrange(2,10);removed=rng.randrange(1,8)
    window=rng.randrange(7,40);unsigned_claim=bool(rng.randrange(2))
    terse=split in ('train','development')
    rows=[]
    for u,v in ((False,False),(False,True),(True,False),(True,True)):
        world={};expr=None;options=['The request is eligible.','The request is not eligible.']
        if family=='conjunction':
            op=('and','or')[index%2]
            world={'signature':u,'inspection':v}
            expr=E(op,V('signature'),V('inspection'));truth=(u and v) if op=='and' else (u or v)
            connector='both a signature and an inspection' if op=='and' else 'at least one of a signature or an inspection'
            state=f'{name}: signature {yn(u)}; inspection {yn(v)}. Eligibility requires {connector}.'
            question='Is the request eligible under this rule?'
            if not terse:state=f'For {name}, an inspection is {yn(v)} and a signature is {yn(u)}. A case qualifies if it has {connector}.'
        elif family=='override_veto':
            veto=index%2==0
            world={'permit':u,'blocked':v}
            expr=E('and',V('permit'),E('not',V('blocked'))) if veto else V('permit')
            truth=u and not v if veto else u
            rule=('An active restriction overrides a permit and prohibits access.' if veto else
                  'A permit overrides a restriction and allows access despite it.')
            state=f'{name}: permit {yn(u)}; restriction {yn(v)}. A permit is required for access. {rule}'
            question='Is access allowed?';options=['Access is allowed.','Access is forbidden.']
            if not terse:state=f'{rule} Without a permit, access is forbidden. Record for {name}: restriction {yn(v)}, permit {yn(u)}.'
        elif family=='quantity_permission':
            minimum=a*b-removed+(1 if not u else -1)
            world={'packs':a,'per_pack':b,'damaged':removed,'minimum':minimum,'signature':v}
            expr=E('and',E('ge',E('sub',E('mul',V('packs'),V('per_pack')),V('damaged')),V('minimum')),V('signature'))
            truth=(a*b-removed>=minimum) and v
            state=f'{name}: {a} packs of {b} pieces each; remove {removed} damaged pieces. Minimum usable count is {minimum}. Signature {yn(v)}. Release requires the minimum AND a signature.'
            question='May this shipment be released?';options=['Release the shipment.','Do not release the shipment.']
            if not terse:state=f'Release is permitted only with a signature and at least {minimum} usable pieces. {name} has a signature {yn(v)}. From {a} packs containing {b} pieces apiece, {removed} pieces are discarded.'
        elif family=='temporal_receipt':
            inclusive=index%2==0;days=(window if inclusive else window-1) if u else (window+1 if inclusive else window)
            now=dt.date(2030,1,1)+dt.timedelta(days=index*7);purchase=now-dt.timedelta(days=days)
            world={'days_since':days,'window':window,'receipt':v}
            expr=E('and',E('le' if inclusive else 'lt',V('days_since'),V('window')),V('receipt'))
            truth=(days<=window if inclusive else days<window) and v
            comp='at most' if inclusive else 'strictly fewer than'
            state=f'Today is {now.isoformat()}. {name} purchased on {purchase.isoformat()}. Receipt {yn(v)}. Refunds require a receipt and {comp} {window} elapsed calendar days since purchase.'
            question='Is a refund permitted?';options=['Permit a refund.','Deny a refund.']
            if not terse:state=f'The rule allows refunds when a receipt exists and elapsed calendar days are {comp} {window}. Purchase: {purchase.isoformat()}; assessment date: {now.isoformat()}; receipt {yn(v)}. Reference {name}.'
        elif family=='authoritative_update':
            later=index%2==0
            world={'earlier':u,'later':v,'use_later':later}
            expr=E('if',V('use_later'),V('later'),V('earlier'));truth=v if later else u
            which='latest' if later else 'earliest'
            state=f'For this audit, use the {which} signed record only. Signed 09:00: approval {yn(u)}. Signed 11:00: approval {yn(v)}. An unsigned note claims approval {yn(unsigned_claim)}. Case {name}.'
            question='Does the controlling record establish approval?';options=['Approval is established.','Approval is not established.']
            if not terse:state=f'Case {name} has two signed entries: 09:00 says approval {yn(u)}; 11:00 says approval {yn(v)}. A note lacking a signature says approval {yn(unsigned_claim)}. Only the {which} signed entry governs the audit.'
        elif family=='mean_total':
            second=a+2*b;factor=60 if v else 1
            world={'first':a,'second':second,'want_total':u,'factor':factor}
            expr=E('mul',E('if',V('want_total'),E('add',V('first'),V('second')),E('div',E('add',V('first'),V('second')),C(2))),V('factor'))
            value=(a+second if u else (a+second)/2)*factor
            candidates=[a+b,a+second,(a+b)*60,(a+second)*60]
            options=[f'The requested duration is {x}.' for x in candidates]
            target=candidates.index(value)
            measure='total duration' if u else 'arithmetic mean duration per batch'
            unit='seconds' if v else 'minutes'
            state=f'{name} had two batches lasting {a} and {second} minutes. Each batch contained {removed+2} items.'
            question=f'What is the {measure}, expressed in {unit}?'
            if not terse:state=f'Batch one lasted {a} minutes; batch two, {second} minutes. Both held {removed+2} objects. Record {name}.'
        elif family=='classify_execute':
            total=a+b if v else a-b;operator='add' if v else 'subtract'
            world={'execute':u,'add':v,'a':a,'b':b}
            expr=E('if',V('execute'),E('if',V('add'),E('add',V('a'),V('b')),E('sub',V('a'),V('b'))),C('math'))
            state=f'Embedded request for {name}: {operator} {b} '+(f'to {a}.' if v else f'from {a}.')+' Arithmetic belongs to the mathematics team; unrelated requests go to general support.'
            question='Execute the embedded request and select its numeric answer.' if u else 'Route the embedded request to a team, without executing it.'
            options=['Assign the mathematics team.','Assign general support.',f'The numeric answer is {total}.',f'The numeric answer is {total+3}.'];target=2 if u else 0
            if not terse:question='Which number answers the quoted arithmetic request?' if u else 'Which team should handle this request? Do not supply its calculation result.'
        elif family=='rule_inference':
            world={'a_true':u,'c_negative':v}
            expr=E('if',V('a_true'),E('if',V('c_negative'),C('both'),C('yes')),E('if',V('c_negative'),C('no'),C('unknown')))
            outcome='both' if u and v else 'yes' if u else 'no' if v else 'unknown'
            options=['C is supported and its negation is not.','Only the negation of C is supported.','Both C and its negation are supported.','Neither is supported.'];target=['yes','no','both','unknown'].index(outcome)
            state=f'For {name}, A implies B and B implies C. '+('A is stated true. ' if u else 'A is not stated. ')+('Not C is explicitly stated.' if v else 'No negation of C is stated.')+' Absence of a statement is not its negation.'
            question='What follows about C after applying all rules?'
            if not terse:state=f'Rules: every A entails B; every B entails C. Facts for {name}: '+('A. ' if u else 'No A fact. ')+('Not C. ' if v else 'No not-C fact. ')+'Use open-world reasoning; a missing fact alone proves nothing.'
        else:raise ValueError(family)
        if family not in ('mean_total','classify_execute','rule_inference'):target=0 if truth else 1
        task,label=envelope(state,question,options,target,sha(f'{key}:{u}:{v}'))
        row={'id':key+f'/{int(u)}{int(v)}','group':key,'family':family,'split':split,
             'task':task,'target':label,'target_index':target,'option_texts':options,
             'world':world,'program':expr,'intervention_bits':[int(u),int(v)]}
        rows.append(row)
    return rows

def compound_group(split,index):
    rows=[];rng=random.Random(sha(f'compound:{split}:{index}'));a=rng.randrange(6,40);b=rng.randrange(2,8);loss=3
    # Cross all 16 combinations; neither arithmetic nor signature alone suffices.
    for mask in range(16):
        within,signature,override,blocked=[bool(mask&(1<<i)) for i in range(4)]
        limit=a*b-loss+(-1 if within else 1)
        world={'packs':a,'per_pack':b,'loss':loss,'limit':limit,'signature':signature,'override':override,'blocked':blocked}
        expr=E('and',E('or',E('ge',E('sub',E('mul',V('packs'),V('per_pack')),V('loss')),V('limit')),V('override')),V('signature'),E('not',V('blocked')))
        truth=((a*b-loss>=limit) or override) and signature and not blocked
        state=f'Case F{index}: {a} packs of {b} usable items, then {loss} removed. Minimum is {limit}. Signature {yn(signature)}; quantity override {yn(override)}; suspension {yn(blocked)}. Release needs a signature AND either enough remaining items or a quantity override. A suspension prohibits release regardless.'
        options=['Release is permitted.','Release is prohibited.']
        task,target=envelope(state,'Which release decision follows?',options,0 if truth else 1,sha(f'compound{index}:{mask}'))
        rows.append({'id':f'{split}/compound/{index}/{mask}','group':f'{split}/compound/{index}',
                     'family':'quantity_override_signature_veto','split':split,'task':task,'target':target,
                     'target_index':0 if truth else 1,'option_texts':options,'world':world,'program':expr,
                     'intervention_bits':[int(within),int(signature),int(override),int(blocked)]})
    return rows

def build(groups_per_family=8):
    out={}
    for split,n in [('train',groups_per_family),('development',1),('transfer',2)]:
        out[split]=[r for fam in FAMILIES for i in range(n) for r in group(fam,split,i)]
    out['composition']=[r for i in range(2) for r in compound_group('composition',i)]
    return out

def normalized_task(row):
    # Privileged diagnostic only: no final answer/label is present in normalized state.
    task=json.loads(canonical(row['task']))
    task['state']='Verified normalized facts and governing expression (not an answer):\n'+canonical({'facts':row['world'],'expression':row['program']})
    return task

def verify(rows):
    for r in rows:
        value=evaluate(r['program'],r['world']);fam=r['family']
        if fam=='mean_total':expected=r['option_texts'].index(f'The requested duration is {int(value)}.')
        elif fam=='classify_execute':expected=0 if value=='math' else 2
        elif fam=='rule_inference':expected=['yes','no','both','unknown'].index(value)
        else:expected=0 if value else 1
        assert expected==r['target_index'],(r['id'],expected,r['target_index'])
        assert r['task']['question']['criteria'][r['target']]==r['option_texts'][expected]
    return len(rows)
