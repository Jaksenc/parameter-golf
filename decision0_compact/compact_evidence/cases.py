"""Fresh constructed decision situations; inference payloads omit all references.

Three finite task specifications and a shared renderer remain across splits.
Source-disjoint data is not a claim of independent human-authored generalization.
"""
from __future__ import annotations
import copy, hashlib, itertools, json, random, re
from pathlib import Path

FAMILIES=('lookup','judging','policy')
EDITS=('original','relevant','irrelevant')
VERSION='compact-evidence-v1'

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def public(case):
    p=case['public']
    if set(p)!={'records','criterion','options'}:raise ValueError('Unexpected public input fields')
    return copy.deepcopy(p)

def policy_result(a,h,c):
    if a=='no':return 'deny'
    if a=='unrecorded':return 'unknown'
    if h=='no' or c=='yes':return 'allow'
    if h=='yes' and c=='no':return 'hold'
    return 'unknown'

def policy_support(values,ids):
    """All minimal sufficient fact subsets, with the criterion always available."""
    domain=('yes','no','unrecorded');subsets=[];answer=policy_result(*values)
    for size in range(1,4):
      for subset in itertools.combinations(range(3),size):
        possibilities={policy_result(*candidate) for candidate in itertools.product(domain,repeat=3) if all(candidate[i]==values[i] for i in subset)}
        if possibilities=={answer} and not any(set(prior)<=set(subset) for prior in subsets):subsets.append(subset)
    return [[ids[i] for i in subset] for subset in subsets]

def one_source(split,family,index):
    rng=random.Random(int(digest([VERSION,split,family,index])[:16],16))
    # Identifiers and line order have a separate RNG stream, unaffected by answers.
    nr=random.Random(int(digest([VERSION,'nuisance',split,family,index])[:16],16))
    used=set()
    def tag(prefix=''):
      while True:
        x=prefix+''.join(nr.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(5))
        if x not in used:used.add(x);return x
    entity=tag('E');source=digest([VERSION,split,family,index,'group'])
    records=[]
    def add(text,kind='fact'):
      rid=tag('R');records.append({'id':rid,'text':text});return rid
    changed=None;alternate=None
    if family=='lookup':
      hops=3+index%3;branches=4+(index%2)
      nodes=[[entity]+[tag('E') for _ in range(branches-1)]]+[[tag('N') for _ in range(branches)] for _ in range(hops-1)]
      destinations=nr.sample(['amber','cobalt','coral','ivory','ochre','pearl','plum','sage','teal','umber'],branches)
      columns=nodes+[destinations]
      relation=('agreement','team','hub','desk','destination')[:hops-1]+('destination',)
      # The parenthesized chain is part of the task definition, never an answer trace.
      criterion=f'Route subject {entity} by following the current '+', then '.join(relation)+' links. Start at the named subject. At each stage match the entire subject identifier; obsolete records have no authority. Return the final destination.'
      paths=[[] for _ in range(branches)]
      for hop in range(hops):
        for branch in range(branches):
          text=f'CURRENT {relation[hop]}: {columns[hop][branch]} -> {columns[hop+1][branch]}.'
          rid=add(text);paths[branch].append(rid)
          if hop==0 and branch==0:
            changed=rid;alternate=f'CURRENT {relation[hop]}: {entity} -> {columns[hop+1][1]}.'
      add(f'OBSOLETE {relation[0]}: {entity} -> {columns[1][-1]}.')
      initial_answer=destinations[0];other_answer=destinations[1]
      initial_support=[paths[0]];other_support=[[paths[0][0]]+paths[1][1:]]
      options=[{'id':d,'description':f'The destination is {d}.'} for d in destinations]
    elif family=='judging':
      subtype=index%3
      criterion=f'Judge the proposed conclusion about subject {entity} using only the supplied current evidence. Return supported if entailed, contradicted if the evidence establishes its negation, otherwise not_established. Missing evidence is not a contradiction.'
      options=[{'id':'supported','description':'The proposed conclusion follows from the evidence.'},
               {'id':'contradicted','description':'The evidence establishes that the proposed conclusion is false.'},
               {'id':'not_established','description':'The evidence establishes neither the conclusion nor its negation.'}]
      if subtype==0:
        c=add(f'CLAIM {entity}: Every item in the batch passed inspection.')
        changed=add(f'COVERAGE {entity}: Every item in the batch was inspected.')
        result=add(f'RESULT {entity}: Every inspected item passed; none of the inspected items failed.')
        alternate=f'COVERAGE {entity}: Only a proper subset of the batch was inspected; no outcome is known for the rest.'
        initial_answer='supported';other_answer='not_established';initial_support=other_support=[[c,changed,result]]
      elif subtype==1:
        c=add(f'CLAIM {entity}: No inspected item failed.')
        changed=add(f'RESULT {entity}: At least one item was inspected and all inspected items passed; no inspected item failed.')
        alternate=f'RESULT {entity}: At least one inspected item failed.'
        initial_answer='supported';other_answer='contradicted';initial_support=other_support=[[c,changed]]
      else:
        c=add(f'CLAIM {entity}: The unit is certified.')
        rule=add('RULE: Certification entails inspection; inspection alone does not entail certification. No other certification rule is given.')
        changed=add(f'STATUS {entity}: The unit is certified.')
        alternate=f'STATUS {entity}: The unit was inspected; its certification status is unrecorded.'
        initial_answer='supported';other_answer='not_established'
        initial_support=[[c,changed]];other_support=[[c,changed,rule]]
    elif family=='policy':
      criterion=f'Apply the following current rule to subject {entity}. First, if active=no, deny. If active=unrecorded, return unknown. For active=yes, allow if hold=no or clearance=yes. Otherwise hold if hold=yes and clearance=no. In every remaining case return unknown. Each unrecorded value is neither yes nor no. Earlier rules in this paragraph take priority. Ignore obsolete notices.'
      states=list(itertools.product(('yes','no','unrecorded'),repeat=3))
      wanted=('allow','deny','hold','unknown')[index%4]
      initial=rng.choice([s for s in states if policy_result(*s)==wanted])
      neighbors=[s for s in states if sum(a!=b for a,b in zip(initial,s))==1 and policy_result(*s)!=wanted]
      other=rng.choice(neighbors);field=next(i for i in range(3) if initial[i]!=other[i])
      ids=[add(f'FACT {entity}: {key}={value}.') for key,value in zip(('active','hold','clearance'),initial)]
      changed=ids[field];alternate=f'FACT {entity}: {("active","hold","clearance")[field]}={other[field]}.'
      add(f'OBSOLETE NOTICE: Subject {entity} was previously suspended; this notice does not establish any current fact.')
      initial_answer=policy_result(*initial);other_answer=policy_result(*other)
      initial_support=policy_support(initial,ids);other_support=policy_support(other,ids)
      options=[{'id':x,'description':y} for x,y in [('allow','Allow this request.'),('deny','Deny for recorded inactivity.'),('hold','Hold for the known unmet clearance requirement.'),('unknown','The current facts do not determine one of the other decisions.')]]
    else:raise ValueError(family)
    # Irrelevant evidence is written once and held stable across relevant edits.
    noise_ids=[]
    for j in range((8,24,56)[index%3]):
      owner=tag('X');reserve=nr.randint(10,900)
      noise_ids.append(add(f'ARCHIVE {owner}: quantity {nr.randint(2,90)}; reserve {reserve}; team {tag("T")}. This record concerns only {owner}, not other subjects.'))
    unrelated=noise_ids[0]
    noise_alternate=next(x['text'] for x in records if x['id']==unrelated).replace('reserve ', 'reserve -')
    # Flip orientation independently of outcome code order, never through public IDs.
    if bool(rng.getrandbits(1)):
      for r in records:
        if r['id']==changed:r['text'],alternate=alternate,r['text']
      initial_answer,other_answer=other_answer,initial_answer
      initial_support,other_support=other_support,initial_support
    nr.shuffle(records);nr.shuffle(options)
    result=[]
    for edit in EDITS:
      rr=copy.deepcopy(records)
      if edit!='original':
        change_id=changed if edit=='relevant' else unrelated
        for record in rr:
          if record['id']==change_id:record['text']=alternate if edit=='relevant' else noise_alternate
      answer=other_answer if edit=='relevant' else initial_answer
      support=other_support if edit=='relevant' else initial_support
      result.append({'id':digest([source,edit]),'source':source,'split':split,'family':family,'edit':edit,
        'public':{'records':rr,'criterion':criterion,'options':options},
        'reference':{'answer':answer,'sufficient_support':support},
        'intervention_record':changed if edit=='relevant' else unrelated if edit=='irrelevant' else None})
    return result

def corpus(split='development',sources_per_family=8):
    if split not in ('development','replication'):raise ValueError('Unknown split')
    return [r for family in FAMILIES for i in range(sources_per_family) for r in one_source(split,family,i)]

def export(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=False);manifest={}
    for split in ('development','replication'):
      rows=corpus(split)
      inputs=[{'id':r['id'],'public':public(r)} for r in rows]
      refs=[{k:r[k] for k in ('id','source','split','family','edit','reference','intervention_record')} for r in rows]
      for suffix,data in [('inputs',inputs),('references',refs)]:
        text=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for x in data)
        file=d/f'{split}-{suffix}.jsonl';file.write_text(text)
        manifest[file.name]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'rows':len(data)}
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest
