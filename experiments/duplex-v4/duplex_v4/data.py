"""Original controlled English sources with checked intervention families.
No benchmark rows, external teacher outputs, or model predictions are used.
Generated language is NOT independent human adjudication or real-world evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random
from .semantics import (ATOMS, STATES, digest, oracle, checked_proof,
                        function_signature, validate, bitset_reference)

SEED = 'duplex-v4-transfer-20260920-v1'
SEEN = [
    ['and','p','q'], ['or','p','q'], ['xor','p','q'],
    ['and','p',['not','q']], ['not',['or','p','q']],
    ['not',['xor','p','q']],
    ['or',['and','p','q'],'r'], ['and',['or','p','q'],'r'],
]
NEW_COMPOSITIONS = [
    ['and','p',['or','q',['not','r']]],
    ['xor',['and','p','q'],'r'],
    ['and',['xor','p','q'],['not','r']],
    ['or',['and','p',['not','q']],['and','q',['not','r']]],
]
DOMAINS = {
    'cards': ('card','align', ('selected','locked','featured')),
    'documents': ('document','publish', ('reviewed','restricted','signed')),
    'tickets': ('ticket','expedite', ('verified','flagged','urgent')),
    'parcels': ('parcel','release', ('inspected','damaged','priority-tagged')),
    'samples': ('sample','queue for assay', ('labelled','contaminated','sealed')),
    'bookings': ('booking','confirm', ('approved','cancelled','prepaid')),
}
PLAN = [('train',48,tuple(DOMAINS)[:4],(0,1),False),
        ('development',12,tuple(DOMAINS)[:4],(2,3),False),
        ('transfer_wording',12,tuple(DOMAINS)[:4],(4,5),False),
        ('transfer_domain',12,tuple(DOMAINS)[4:],(4,5),False),
        ('transfer_composition',12,tuple(DOMAINS)[:4],(4,5),True)]
POLICY = (
    'Use only the named target and the condition below. A confirmed observation fixes a fact as true; '
    'an explicit denial fixes it as false. Missing observations or equally authoritative conflicting '
    'observations leave that fact unresolved. Every occurrence of the same property refers to the same fact. '
    'Consider all true/false completions compatible with the observations. If the condition is true in '
    'every completion, perform the named operation on the target only. If false in every completion, '
    'leave it unchanged. Otherwise request clarification. Never change any other object.'
)
SYSTEM = ('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
          'Respond with only its uppercase letter, with no explanation or reasoning.')

def render_condition(tree, props, style):
    if isinstance(tree,str): return f'the target is {props[ATOMS.index(tree)]}'
    op=tree[0]; a=render_condition(tree[1],props,style)
    if op=='not':
        return [f'it is not the case that ({a})', f'({a}) is false',
                f'NOT ({a})', f'the statement ({a}) does not hold',
                f'the negation of ({a}) holds', f'it is false that ({a})'][style]
    b=render_condition(tree[2],props,style)
    templates={
      'and':[f'({a}) and ({b})',f'both ({a}) and ({b})',
             f'({a}) AND ({b})',f'({a}) holds together with ({b})',
             f'each of these holds: ({a}); ({b})',f'({a}) as well as ({b})'],
      'or':[f'({a}) or ({b}), including when both hold',f'at least one of ({a}) and ({b}) holds',
            f'({a}) OR ({b}), inclusively',f'either ({a}) or ({b}) or both',
            f'one or both of these holds: ({a}); ({b})',f'({a}) holds, or ({b}) holds, or both hold'],
      'xor':[f'exactly one of ({a}) and ({b}) holds',f'({a}) or ({b}), but not both',
             f'({a}) XOR ({b}); XOR means exactly one',f'only one of ({a}) and ({b}) is true',
             f'one, but only one, of these holds: ({a}); ({b})',f'({a}) and ({b}) have different truth values']}
    return templates[op][style]

def render_evidence(target, other, props, facts, style, order, conflict, irrelevant=False):
    lines=[]
    for atom in order:
        prop=props[ATOMS.index(atom)]; value=facts[atom]
        if value is None:
            text=(f'Equally authoritative records conflict: {target} is {prop}; {target} is not {prop}.'
                  if conflict else f'No observation of whether {target} is {prop} is available.')
        elif style % 3 == 0:
            text=f'{target} is '+('' if value else 'not ')+prop+'.'
        elif style % 3 == 1:
            text=f'Record for {target}: {prop} is '+('confirmed.' if value else 'explicitly denied.')
        else:
            text=f'The review '+('confirms' if value else 'rules out')+f' that {target} is {prop}.'
        lines.append(text)
    lines.insert(1, f'{other}, a different object, is {props[0]} and {props[1]}.')
    if irrelevant:
        lines.append(f'The archive label for {other} was printed on a Tuesday. This is metadata, not an observation about {target}.')
    return f'The named target is {target}. '+ ' '.join(lines)

def option_descriptions(noun,verb,target,other):
    return [f'{verb.capitalize()} {target}, and no other {noun}.',
            f'Leave {target} unchanged; do not perform the operation.',
            f'Request clarification before changing any {noun}.',
            f'{verb.capitalize()} {other} instead of {target}.',
            f'{verb.capitalize()} every {noun}, including {other}.',
            f'Delete {target}.']

def public_input(row):
    """Allowlist boundary: no targets, proof, AST, family names, IDs or split."""
    inp=row['input']
    return {'evidence':inp['evidence'],'criterion':inp['criterion'],
            'options':[{'letter':chr(65+i),'description':x} for i,x in enumerate(inp['options'])]}

def messages(row):
    return [{'role':'system','content':SYSTEM},
            {'role':'user','content':json.dumps(public_input(row),ensure_ascii=False)}]

def semantic_key(domain,tree,facts):
    return digest({'domain':domain,'truth_function':function_signature(tree),'facts':facts})

def fact_counterfactual(tree,facts,rng):
    changed=[]
    for atom in ATOMS:
        for val in STATES:
            if val is facts[atom]:continue
            f=dict(facts);f[atom]=val
            if oracle(tree,f)!=oracle(tree,facts):changed.append((f,atom))
    if not changed:return None
    return rng.choice(changed)

def make_group(split,serial,domain,tree,facts,styles,pool,rng):
    factcf=fact_counterfactual(tree,facts,rng)
    if factcf is None:return None
    alternate=[x for x in pool if oracle(x,facts)!=oracle(tree,facts)]
    if not alternate:return None
    rulecf=rng.choice(alternate)
    group='world-'+digest({'namespace':SEED,'split':split,'serial':serial,'domain':domain,'tree':tree,'facts':facts})[:20]
    noun,verb,props=DOMAINS[domain]
    target=f'{noun}-{digest([group,"target"])[:8]}'
    other=f'{noun}-{digest([group,"other"])[:8]}'
    base_order=list(range(6));rng.shuffle(base_order)
    evidence_order=list(ATOMS);rng.shuffle(evidence_order)
    conflict=bool(rng.randrange(2));options=option_descriptions(noun,verb,target,other)
    rows=[]
    for v in range(8):
        # Six meaning-preserving views rotate every outcome through every slot.
        order=base_order[v:]+base_order[:v] if v<6 else base_order[:]
        f=factcf[0] if v==6 else facts
        t=rulecf if v==7 else tree
        style=styles[v%2] if v<6 else styles[0]
        eo=evidence_order if v in (0,6,7) else (evidence_order[v%3:]+evidence_order[:v%3])
        ev=render_evidence(target,other,props,f,style,eo,conflict,irrelevant=(v==3))
        instruction=f'The operation is to {verb} the named target. Condition: '+render_condition(t,props,style)+'.\n'+POLICY
        proof=checked_proof(t,f);gold=proof['semantic_target']
        inp={'evidence':ev,'criterion':instruction,'options':[options[i] for i in order]}
        rows.append({'id':group+f'-v{v}','group':group,'split':split,'domain':domain,'view':v,
                     'relation':'invariant' if v<6 else 'fact_change' if v==6 else 'rule_change',
                     'input':inp,'semantic_order':order,'target':order.index(gold),'semantic_target':gold,
                     'ast':t,'facts':f,'proof':proof,'style':style,
                     'semantic_key':semantic_key(domain,t,f),
                     'input_hash':digest(public_input({'input':inp})),
                     'fact_changed_atom':factcf[1] if v==6 else None})
    return rows

def build():
    rng=random.Random(SEED);allrows=[];used_by_split=defaultdict(set)
    for split,count,domains,styles,compositions in PLAN:
        pool=NEW_COMPOSITIONS if compositions else SEEN
        serial=0;tries=0
        while serial<count:
            tries+=1
            if tries>100000:raise RuntimeError('Cannot satisfy preregistered balancing/split constraints')
            domain=domains[(serial//3)%len(domains)]
            truth=serial%3
            tree=rng.choice(pool);facts=dict(zip(ATOMS,rng.choice(list(itertools.product(STATES,repeat=3)))))
            if oracle(tree,facts)!=truth:continue
            group=make_group(split,serial,domain,tree,facts,styles,pool,rng)
            if group is None:continue
            keys={r['semantic_key'] for r in group}
            others=set().union(*(v for k,v in used_by_split.items() if k!=split))
            if keys & others:continue
            # Do not inflate population with a duplicate base world in its own split.
            if group[0]['semantic_key'] in used_by_split[split]:continue
            used_by_split[split].update(keys)
            allrows.extend(group);serial+=1
    audit(allrows)
    return allrows

def audit(rows):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate ID')
    if len({r['input_hash'] for r in rows})!=len(rows):raise ValueError('Duplicate model input')
    groups=defaultdict(list);splits=defaultdict(set)
    for r in rows:
        groups[r['group']].append(r);splits[r['split']].add(r['semantic_key'])
        if checked_proof(r['ast'],r['facts'])!=r['proof']:raise ValueError('Proof mismatch')
        if r['semantic_order'][r['target']]!=r['semantic_target']:raise ValueError('Target mapping')
        if digest(public_input(r))!=r['input_hash']:raise ValueError('Input binding')
    for a,b in itertools.combinations(splits,2):
        if splits[a]&splits[b]:raise ValueError('Cross-split latent-world alias')
    for gg in groups.values():
        gg.sort(key=lambda r:r['view'])
        if [r['view'] for r in gg]!=list(range(8)) or len({r['split'] for r in gg})!=1:raise ValueError('Broken intervention family')
        base=gg[0]
        if sorted(r['target'] for r in gg[:6])!=list(range(6)):raise ValueError('Unbalanced invariant slots')
        for r in gg[1:6]:
            if r['ast']!=base['ast'] or r['facts']!=base['facts'] or r['semantic_target']!=base['semantic_target']:raise ValueError('False invariant')
        if sum(gg[6]['facts'][a] is not base['facts'][a] for a in ATOMS)!=1:raise ValueError('Fact edit changes more than one atom')
        if gg[6]['ast']!=base['ast'] or gg[7]['facts']!=base['facts']:raise ValueError('Uncontrolled intervention')
        if any(r['semantic_target']==base['semantic_target'] for r in gg[6:]):raise ValueError('Nonchanging change edge')
    seen={function_signature(r['ast']) for r in rows if r['split']=='train'}
    comp={function_signature(r['ast']) for r in rows if r['split']=='transfer_composition'}
    if seen&comp:raise ValueError('Composition truth function was trained')
    out={'data_hash':digest(rows),'rows':len(rows),'source_families':len(groups),
         'split_rows':dict(Counter(r['split'] for r in rows)),
         'split_groups':{s:len({r['group'] for r in rows if r['split']==s}) for s in splits},
         'base_target_counts':{s:dict(Counter(str(r['semantic_target']) for r in rows if r['split']==s and r['view']==0)) for s in splits},
         'all_target_counts':{s:dict(Counter(str(r['semantic_target']) for r in rows if r['split']==s)) for s in splits},
         'cross_split_aliases':0,'invariant_edges':len(groups)*5,'change_edges':len(groups)*2,
         'independent_human_review':False,'pretrained_model_calls':0,
         'limitations':['Controlled generated English, not independently authored real-world examples',
           'Eight correlated views per source family; do not count them as independent samples',
           'Domain and wording holdouts share operators; composition holdout uses new truth functions',
           'Logical proof validates formal meanings, not an external human audit of wording',
           'Possible-world semantics differs from earlier strong-Kleene pilot and is explicit in every prompt']}
    return out

def write_bundle(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    rows=build()
    for split,*_ in PLAN:
        path=root/(split+'.jsonl')
        with path.open('w') as f:
            for r in rows:
                if r['split']==split:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n')
    (root/'audit.json').write_text(json.dumps(audit(rows),indent=2,sort_keys=True))
    return audit(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='data');args=ap.parse_args()
    print(json.dumps(write_bundle(args.out),sort_keys=True,indent=2))
