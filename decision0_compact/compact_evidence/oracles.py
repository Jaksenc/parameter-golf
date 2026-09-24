"""Independent reference parser for these constructed languages only.

Not imported by runtime model-input code. Never used to choose source records at
inference. Parsing arbitrary business documents is explicitly out of scope.
"""
from __future__ import annotations
import itertools,re

def unique(items):
    if len(items)!=1:raise ValueError(f'Expected one matching record, got {len(items)}')
    return items[0]

def reference(public):
    records=public['records'];question=public['criterion']
    byid={r['id']:r['text'] for r in records}
    if question.startswith('Route subject '):
      entity=re.search(r'Route subject (\w+)',question)[1]
      chain=re.search(r'current (.+?) links\.',question)[1].split(', then ')
      used=[];node=entity
      for relation in chain:
        found=[]
        for rid,text in byid.items():
          m=re.fullmatch(r'CURRENT '+relation+': '+re.escape(node)+r' -> (\w+)\.',text)
          if m:found.append((rid,m[1]))
        rid,node=unique(found);used.append(rid)
      return {'answer':node,'sufficient_support':[used]}
    if question.startswith('Judge the proposed conclusion'):
      entity=re.search(r'about subject (\w+)',question)[1]
      def find(prefix):return unique([(k,v) for k,v in byid.items() if v.startswith(prefix)])
      claim_id,claim=find(f'CLAIM {entity}:')
      if 'Every item in the batch' in claim:
        coverage_id,coverage=find(f'COVERAGE {entity}:');result_id,result=find(f'RESULT {entity}:')
        if 'Every inspected item passed' not in result:raise ValueError('Unsupported renderer')
        answer='supported' if 'Every item in the batch was inspected.' in coverage else 'not_established'
        return {'answer':answer,'sufficient_support':[[claim_id,coverage_id,result_id]]}
      if 'No inspected item failed.' in claim:
        rid,text=find(f'RESULT {entity}:')
        answer='contradicted' if text.endswith('At least one inspected item failed.') else 'supported'
        return {'answer':answer,'sufficient_support':[[claim_id,rid]]}
      if 'The unit is certified.' in claim:
        rid,text=find(f'STATUS {entity}:')
        if text.endswith('The unit is certified.'):
          return {'answer':'supported','sufficient_support':[[claim_id,rid]]}
        rule_id,_=find('RULE: Certification entails inspection;')
        return {'answer':'not_established','sufficient_support':[[claim_id,rid,rule_id]]}
      raise ValueError('Unknown judging specification')
    if question.startswith('Apply the following current rule'):
      entity=re.search(r'to subject (\w+)\.',question)[1];values=[];ids=[]
      for field in ('active','hold','clearance'):
        matches=[]
        for rid,text in byid.items():
          m=re.fullmatch(r'FACT '+entity+': '+field+r'=(yes|no|unrecorded)\.',text)
          if m:matches.append((rid,m[1]))
        rid,value=unique(matches);values.append(value);ids.append(rid)
      # Priority dispatch independent of the generator's nested condition function.
      def classify(v):
        a,h,c=v
        rules=[(a=='no','deny'),(a=='unrecorded','unknown'),(a=='yes' and (h=='no' or c=='yes'),'allow'),(a=='yes' and h=='yes' and c=='no','hold'),(True,'unknown')]
        return next(label for condition,label in rules if condition)
      expected=classify(values);sufficient=[]
      for size in range(1,4):
        for subset in itertools.combinations(range(3),size):
          possible=[v for v in itertools.product(('yes','no','unrecorded'),repeat=3) if all(v[i]==values[i] for i in subset)]
          if all(classify(v)==expected for v in possible) and not any(set(s)<=set(subset) for s in sufficient):sufficient.append(subset)
      return {'answer':expected,'sufficient_support':[[ids[i] for i in s] for s in sufficient]}
    raise ValueError('Unrecognized constructed task')
