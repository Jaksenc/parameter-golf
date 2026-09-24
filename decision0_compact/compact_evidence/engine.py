"""Experiment orchestration; all backend requests contain only public fields."""
from __future__ import annotations
from .cases import digest
from .prompts import MODES,generation,final_messages
from .protocol import PROTOCOL

def score_case(backend,ledger,public,mode,retry_errors=False):
    if mode not in MODES:raise ValueError(mode)
    note=None;generation_key=None
    if mode!='baseline':
      kind='evidence' if mode=='length_matched_control' else mode
      prompt=generation(public,kind);cap=PROTOCOL['generation_caps'][kind]
      generation_key=digest({'model':backend.meta,'operation':'generate','kind':kind,'cap':cap,'prompt':prompt})
      note=ledger.call(generation_key,lambda:backend.generate(prompt,kind,cap),retry_errors)
    prompt,selection=final_messages(public,mode,note['text'] if note else None)
    labels=[r['id'] for r in public['options']]
    decision_key=digest({'model':backend.meta,'operation':'score','labels':labels,'prompt':prompt})
    decision=ledger.call(decision_key,lambda:backend.score(prompt,labels),retry_errors)
    return {'status':'ok','backend_kind':backend.kind,'mode':mode,'public_hash':digest(public),
       'decision':decision,'generation':note,'selection':selection,'generation_key':generation_key,'decision_key':decision_key,
       'route_input_tokens':decision['input_tokens']+(note['input_tokens'] if note else 0),
       'route_output_tokens':note['output_tokens'] if note else 0,
       'route_seconds':decision['seconds']+(note['seconds'] if note else 0),
       'cost_note':'Same selector generation is reused across paired evidence/control evaluations; charged in full to each hypothetical route.'}

def assign(rows,nshards):
    if type(nshards) is not int or not 1<=nshards<=32:raise ValueError('Invalid shard count')
    # All three source variants stay together; source assignment ignores references.
    grouped={}
    for r in rows:grouped.setdefault(r['source'],[]).append(r)
    loads=[0]*nshards;assignments=[[] for _ in range(nshards)]
    for source,group in sorted(grouped.items(),key=lambda pair:(-sum(len(str(r['public'])) for r in pair[1]),pair[0])):
      i=min(range(nshards),key=lambda j:(loads[j],j));assignments[i].extend(group);loads[i]+=sum(len(str(r['public'])) for r in group)
    return [sorted(a,key=lambda r:r['id']) for a in assignments]
