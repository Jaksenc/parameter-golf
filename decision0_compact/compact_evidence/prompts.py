"""Inference views with no access to labels or construction references."""
from __future__ import annotations
import json,random
from .cases import digest

MODES=('baseline','full_workspace','evidence','length_matched_control','compact_state')
SYSTEM=('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
        'Respond with only its uppercase letter, with no explanation or reasoning.')
FINAL=('Verify the candidate notes against the original evidence and criterion. '
       'Original evidence is authoritative. Return only the uppercase letter of one listed option.')
FULL=('Build a brief, evidence-grounded decision workspace, not an answer letter. '
      'Use three named sections: Relevant evidence; Computation or governing rule; Derived result. '
      'For policies resolve priorities; for linked records identify the complete chain; '
      'for judging identify the decisive supported claim or error. Do not invent missing facts. '
      'Be concise; do not repeat irrelevant input or output an option letter.')
EVIDENCE=('Select only source records necessary to apply the criterion. Return exactly one JSON '
          'array of at most six existing record IDs, such as ["RBCDFG","RHJKLM"]. '
          'No prose, no explanation, no verdict. IDs must be copied exactly from the evidence. '
          'Do not select a record just because its vocabulary resembles the question. '
          'An empty array is allowed when no record is relevant.')
STATE=('Return exactly one compact JSON object with keys e and s. e is an array of at most six '
       'existing record IDs. s is one short inference or check, at most 160 characters. '
       'Use the selected facts and criterion. Do not write an answer letter. '
       'Do not copy long evidence; do not invent facts. No prose outside the JSON object. '
       'Example shape only: {"e":["RBCDFG"],"s":"The applicable condition is explicitly recorded."}')

def validate_public(p):
    if set(p)!={'records','criterion','options'}:raise ValueError('Nonpublic or missing fields')
    if not isinstance(p['criterion'],str) or not p['criterion'].strip():raise ValueError('Missing criterion')
    if not isinstance(p['records'],list) or not p['records']:raise ValueError('Missing evidence')
    if not 2<=len(p['options'])<=16:raise ValueError('Outcome cardinality outside interface')
    if len({r['id'] for r in p['records']})!=len(p['records']):raise ValueError('Repeated evidence ID')
    if len({r['id'] for r in p['options']})!=len(p['options']):raise ValueError('Repeated outcome ID')
    for r in p['records']:
      if set(r)!={'id','text'} or not all(isinstance(v,str) for v in r.values()):raise ValueError('Bad record')
    for r in p['options']:
      if set(r)!={'id','description'} or not all(isinstance(v,str) for v in r.values()):raise ValueError('Bad outcome')

def serialized_evidence(p):
    validate_public(p)
    return '\n'.join(f'[{r["id"]}] {r["text"]}' for r in p['records'])

def baseline(p):
    payload={'evidence':serialized_evidence(p),'criterion':p['criterion'],
             'options':[{'letter':chr(65+i),'description':r['description']} for i,r in enumerate(p['options'])]}
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]

def generation(p,kind):
    if kind not in ('full_workspace','evidence','compact_state'):raise ValueError(kind)
    msg=baseline(p)
    return [{'role':'system','content':{'full_workspace':FULL,'evidence':EVIDENCE,'compact_state':STATE}[kind]},msg[1]]

def strict_json(text):
    def unique(ps):
      d={}
      for k,v in ps:
        if k in d:raise ValueError('Duplicate JSON key')
        d[k]=v
      return d
    return json.loads(text,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def json_complete(text,kind):
    try:obj=strict_json(text)
    except (ValueError,TypeError):return False
    return isinstance(obj,list) if kind=='evidence' else isinstance(obj,dict) if kind=='compact_state' else False

def parse_note(p,kind,text):
    """No repair of partial JSON; unknown source IDs invalidate the entire note."""
    try:
      obj=strict_json(text)
      state=''
      if kind=='evidence':ids=obj
      elif kind=='compact_state':
        if not isinstance(obj,dict) or set(obj)!={'e','s'}:raise ValueError('Wrong state schema')
        ids=obj['e'];state=obj['s']
        if type(state) is not str or len(state)>160:raise ValueError('State too long or not a string')
      else:raise ValueError('Unknown structured note kind')
      if not isinstance(ids,list) or len(ids)>6 or any(type(i) is not str for i in ids):raise ValueError('Invalid source ID array')
      if len(set(ids))!=len(ids):raise ValueError('Repeated source IDs')
      if not set(ids)<={r['id'] for r in p['records']}:raise ValueError('Unknown source IDs')
      # Preserve evidence order, not a speculative interpretation of selector order.
      ids=[r['id'] for r in p['records'] if r['id'] in ids]
      return {'valid':True,'ids':ids,'state':state,'error':None}
    except (ValueError,TypeError) as e:return {'valid':False,'ids':[],'state':'','error':str(e)}

def materialize(p,ids):
    """Copy original record text, never generated substitutes or oracle facts."""
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate ID')
    if not set(ids)<={r['id'] for r in p['records']}:raise ValueError('Unknown ID')
    return '\n'.join(f'[{r["id"]}] {r["text"]}' for r in p['records'] if r['id'] in ids)

def length_control(p,selected_ids):
    """Label-blind approximate record-length control, NOT guaranteed irrelevant.

    The selection generation is charged to this route too. Matching is by source
    character length, not exact token count; actual token costs are reported.
    """
    if not selected_ids:return []
    chosen={r['id']:r for r in p['records']}
    if not set(selected_ids)<=set(chosen):raise ValueError('Bad selector IDs')
    rng=random.Random(int(digest(['control',p])[:16],16));candidates=p['records'].copy();rng.shuffle(candidates)
    ids=[]
    for sid in selected_ids:
      target=len(chosen[sid]['text'])
      record=min(candidates,key=lambda r:abs(len(r['text'])-target));ids.append(record['id']);candidates.remove(record)
    return [r['id'] for r in p['records'] if r['id'] in ids]

def final_messages(p,mode,raw=None):
    if mode not in MODES:raise ValueError(mode)
    msgs=baseline(p)
    if mode=='baseline':return msgs,{'valid':True,'ids':[],'state':'','error':None}
    if mode=='full_workspace':
      if type(raw) is not str:raise ValueError('Missing raw workspace')
      note=raw;metadata={'valid':None,'ids':[],'state':'','error':None}
    else:
      kind='compact_state' if mode=='compact_state' else 'evidence'
      metadata=parse_note(p,kind,raw)
      if metadata['valid']:
        ids=length_control(p,metadata['ids']) if mode=='length_matched_control' else metadata['ids']
        metadata=dict(metadata,ids=ids)
        note='Source excerpts (copies of original records):\n'+materialize(p,ids)
        if kind=='compact_state':note+='\nFallible model inference: '+metadata['state']
      else:note='No valid source selection was produced. No additional evidence is supplied.'
    return msgs+[{'role':'assistant','content':'Candidate notes (verify):\n'+note},{'role':'user','content':FINAL}],metadata
