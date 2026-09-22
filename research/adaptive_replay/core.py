"""Relational replay experiment. Biology-inspired analogy, not a brain simulation.
All seed lessons are authored, not automatically learned from benchmark results.
"""
from __future__ import annotations
import hashlib,json,math,re,sqlite3,time,urllib.error,urllib.request
from pathlib import Path
REASON=('Solve the task carefully. Use concise reasoning to identify the quantities, entities, '
        'relationships, exceptions, and requested result. Check your interpretation before '
        'calculating. Finish with FINAL: <the requested answer>. Match the answer type asked '
        'for; never invent answer choices. Do not leave the final answer unfinished.')
LESSONS=[
('groups','boxes tables groups each apiece total already included inventory',
 'Bind each count to its entity. Group count times per-group amount gives a total. Different group sizes require separate products. Do not add an amount already included in a total. Contrast: boxes times items per box counts items; adding boxes to items does not. Not every number must be used.'),
('average','average mean weighted duration overall speed',
 'Identify what one observation is. Ordinary average means sum of observations divided by their count, not another available count. Weighted averages use the requested weights. Overall rate over unequal durations is total output divided by total duration. Contrast: average time for two batches divides by two; time per item divides by item count.'),
('rate','rate per hour minute speed time throughput workers simultaneously',
 'Keep rates, durations and totals distinct. Output equals rate times duration; duration equals output divided by rate. Convert time units before combining quantities. Independent simultaneous workers can add rates; sequential stages generally add times. Contrast: hours per item is the reciprocal of items per hour.'),
('fraction','half quarter third fifth fraction completed unfinished remaining portion whole',
 'A fraction refers to a particular whole and state. If fraction f of W is completed, completed amount is f*W and remaining amount is (1-f)*W. Recovering the whole from a stated part reverses this relation. Contrast: half of a 500-piece puzzle is 250 placed pieces, not 500. Do not infer completion from nominal size.'),
('percent','percent percentage discount markup margin price tax revenue cost',
 'Attach every percentage to its base. Increase multiplies the base by (1+p/100); decrease multiplies by (1-p/100). Sequential changes may compound on different bases. Reverse a percentage change by division. Contrast: markup on cost and margin on selling price use different denominators. Percentage points and relative percent changes differ.'),
('comparison','more fewer less times ratio bigger smaller twice than',
 'Write the relation direction before substituting. If A has d more than B, A=B+d; recovering B requires subtraction. If A is k times B, recovering B requires division. Keep ratios attached to ordered entities. Contrast: A has twice B does not imply B has twice A. Preserve which entity is unknown.'),
('state','remaining balance after before initial final transfers sold bought received gave',
 'Track the state after every event in order. Inflows add and outflows subtract. Distinguish intermediate, initial and final balances. Recovering the initial state undoes events in reverse order. A later fraction may apply to the remaining state. Contrast: removing half before adding ten differs from adding ten before removing half.'),
('order','swap swapped switching order alphabetical rank third first left right positions balls',
 'Represent an ordered arrangement and apply changes sequentially. Swap occupants, not the names of positions. Distinguish current rank from original rank. Contrast: successive swaps involving one position are not independent exchanges. For ordering constraints, test the proposed order against every relation; for sorting, use the requested comparison.'),
('logic','if implies true false all some none not either both boolean premise conclusion',
 'Attach predicates, negations and quantifiers to their proper scope. All A are B does not imply all B are A. Some does not imply all. P implies Q does not allow inferring P from Q alone. Contrast: not all are red permits some red; none are red permits no red. Check all constraints, not only a plausible one.'),
('space','north south east west walk turn steps navigation rotate shape orientation position',
 'Use a consistent coordinate frame. Distinguish object-relative and fixed directions. Track heading when a turn occurs and displacement when a move occurs. Returning to the same heading need not return to the same location. Separate orientation from position and preserve the relevant invariants under geometric transformations.')]

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(b):return hashlib.sha256(b).hexdigest()
def tokens(text):return re.findall(r'[a-z]+',text.lower())
def bank_records():
 return [{'id':i,'anchors':a,'lesson':l,'version':1,'provenance':'authored v0.9 acquisition library; no benchmark responses','qualification':'advisory; not a semantic proof'} for i,a,l in LESSONS]

class Memory:
 def __init__(self,path,scope):
  if not isinstance(scope,str) or not scope.strip() or len(scope)>128:raise ValueError('scope')
  self.scope=scope;Path(path).parent.mkdir(parents=True,exist_ok=True)
  self.db=sqlite3.connect(path,timeout=10)
  self.db.execute('CREATE TABLE IF NOT EXISTS relational_replay_v09(scope TEXT,id TEXT,payload TEXT,digest TEXT,PRIMARY KEY(scope,id))');self.db.commit()
 def close(self):self.db.close()
 def add(self,r):
  if set(r)!=set(bank_records()[0]) or r['qualification']!='advisory; not a semantic proof':raise ValueError('memory_schema')
  raw=canon(r)
  if len(raw)>8192:raise ValueError('memory_size')
  with self.db:self.db.execute('INSERT INTO relational_replay_v09 VALUES(?,?,?,?)',(self.scope,r['id'],raw,sha(raw.encode())))
 def read(self):
  out=[]
  for raw,h in self.db.execute('SELECT payload,digest FROM relational_replay_v09 WHERE scope=? ORDER BY id',(self.scope,)):
   if sha(raw.encode())!=h:raise ValueError('memory_integrity')
   out.append(json.loads(raw))
  return out
 def delete(self,key):
  with self.db:return self.db.execute('DELETE FROM relational_replay_v09 WHERE scope=? AND id=?',(self.scope,key)).rowcount==1

def retrieve(q,records,sham=False):
 """Transparent lexical comparator. Scores measure relevance, never correctness."""
 words=tokens(q);counts={w:words.count(w) for w in set(words)};scored=[]
 for r in records:
  anchors=set(tokens(r['anchors']));hits=sum(min(counts.get(w,0),2) for w in anchors)
  score=hits/math.sqrt(max(1,len(anchors)))
  scored.append((score,r['id'],r))
 scored.sort(key=lambda x:(-x[0],x[1]))
 picked=[x for x in scored if x[0]>0][:2]
 if sham:
  selected={x[1] for x in picked}
  picked=[x for x in sorted(scored,key=lambda x:(x[0],x[1])) if x[1] not in selected][:len(picked)]
 return [{'record':r,'retrieval_score':s} for s,i,r in picked]

def prompt(q,arm,records):
 if arm not in ('reasoning','native','replay','sham'):raise ValueError('arm')
 memories=retrieve(q,records,arm=='sham') if arm in ('replay','sham') else []
 s=REASON
 if memories:
  s+='\n\nOptional lessons from other tasks follow. They are NOT facts about this task. Use only applicable relationships, rebinding them to the actual entities and requested quantity. Ignore mismatched lessons. No special schema is required.\n'
  s+='\n'.join('- '+m['record']['lesson'] for m in memories)
 return s,memories

class Transport:
 def __init__(self,base='http://127.0.0.1:8787',timeout=240):
  from urllib.parse import urlsplit
  p=urlsplit(base)
  if p.scheme!='http' or p.hostname not in ('127.0.0.1','localhost','::1') or p.username or p.password or p.query or p.fragment or p.path not in ('','/'):
   raise ValueError('local_http_endpoint_required')
  self.base=base.rstrip('/');self.timeout=timeout
 def __call__(self,messages,native=False):
  p={'model':'local','messages':messages,'max_tokens':2048 if native else 1536,'temperature':1.0 if native else 0,'seed':1707,'stream':False,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':native}}
  if native:p.update(top_p=.95,top_k=20,min_p=0.,presence_penalty=1.5,repeat_penalty=1.)
  raw=canon(p).encode();start=time.perf_counter()
  req=urllib.request.Request(self.base+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,timeout=self.timeout) as r:
   data=r.read(2_000_001)
   if len(data)>2_000_000:raise ValueError('response_limit')
   obj=json.loads(data)
  c=obj['choices'][0]
  return {'text':c['message'].get('content') or '', 'reasoning_content':c['message'].get('reasoning_content') or '', 'finish_reason':c['finish_reason'],'usage':obj.get('usage',{}),'timings':obj.get('timings',{}),'seconds':time.perf_counter()-start,'request_sha256':sha(raw),'request_parameters':{k:v for k,v in p.items() if k!='messages'}}

def extract(text):
 matches=re.findall(r'(?:^|\n)\s*FINAL:\s*(.+)',text,re.I)
 if not matches:return None
 return matches[-1].strip()

def solve(q,arm,records,transport=None):
 if not isinstance(q,str) or not 1<=len(q)<=40000:raise ValueError('question_size')
 start=time.perf_counter();system,mem=prompt(q,arm,records);call=None;answer=None;error=None
 try:
  call=(transport or Transport())([{'role':'system','content':system},{'role':'user','content':q}],arm=='native')
  answer=extract(call['text']) if call['finish_reason']=='stop' else None
  status='model_answer_unverified' if answer is not None else 'incomplete_or_missing_answer'
 except (ValueError,KeyError,IndexError,urllib.error.URLError,TimeoutError,OSError) as e:
  status='transport_or_protocol_error';error=f'{type(e).__name__}: {e}'
 return {'arm':arm,'answer':answer,'status':status,'error':error,'calls':[] if call is None else [call],'memories':mem,'prompt_sha256':sha(system.encode()),'question_sha256':sha(q.encode()),'seconds':time.perf_counter()-start,'semantic_verification':False,'weights_updated':False}

def checks():
 import tempfile
 with tempfile.TemporaryDirectory() as d:
  m=Memory(Path(d)/'m.sqlite3','a')
  for r in bank_records():m.add(r)
  original=m.read();m.close();m=Memory(Path(d)/'m.sqlite3','a');assert m.read()==original
  q='What is the average duration for the two batches?'
  hits=retrieve(q,original);assert hits[0]['record']['id']=='average'
  sham=retrieve(q,original,True);assert len(hits)==len(sham)
  assert not {x['record']['id'] for x in hits}&{x['record']['id'] for x in sham}
  other=Memory(Path(d)/'m.sqlite3','b');assert other.read()==[];other.close()
  assert m.delete('average');assert len(m.read())==9;m.close()
 assert extract('work\nFINAL: 18')=='18' and extract('not done') is None
 assert prompt('anything','reasoning',[])[0]==REASON
 assert prompt('anything','native',[])[0]==REASON
 return {'checks':'passed','memory_records':len(LESSONS),'core_sha256':sha(Path(__file__).read_bytes()),'bank_sha256':sha(canon(bank_records()).encode())}

if __name__=='__main__':print(json.dumps(checks()))
