"""Compact evidence binding, fixed 4B inference ablation; no training/benchmark.
Public evidence IDs are chosen by the model; resolving IDs never uses the oracle.
A generated-and-discarded control reuses exactly the compact-state generation,
then matches final input length using neutral filler. Costs count that generation.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,re,statistics,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path

MODES=('direct','full_workspace','evidence_ids','compact_state','discarded_state')
P={'id':'decision0-compact-evidence-v1','seed':924731,'families':['join','judge','policy'],
 'sources_per_family':6,'rows':36,'modes':MODES,'shards':12,'max_input_tokens':4096,
 'generation_limits':{'full_workspace':160,'evidence_ids':32,'compact_state':64},
 'checkpoint':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'thinking':False,'weights_updated':False,'benchmark_calls':0,'official_score':None,
 'selection':'none; report all fixed modes and complete pairs',
 'control':'discarded_state pays for the SAME state generation; its text is removed and final token count exactly matched with neutral filler',
 'evidence_rendering':'model-generated 1-based record IDs resolved verbatim from visible evidence; invalid selection produces an explicit empty-selection note',
 'limits':'18 constructed situations, finite shared renderers; conditional compute control, not a separately generated meaningless chain of thought'}
FULL=('Build a brief, evidence-grounded decision workspace, not an answer letter. Use three named sections: Relevant evidence; Computation or governing rule; Derived result. Include only material relevant to this request. The supplied evidence is authoritative. Do not invent missing facts. Be concise; do not repeat the input or output an option letter.')
SELECT=('Select the smallest set of evidence records needed to decide the criterion. Return ONLY a JSON array of at most six distinct record numbers, such as [2,7,11]. Include necessary current rules and each link in a chain. Do not output an answer, explanation or record text. Do not assume missing facts.')
STATE=('Produce one compact evidence-grounded working state, at most 35 words. Use a single line: E: record numbers; S: linked facts or governing condition. No headings beyond E: and S:, no explanation, no final option letter, no repetition. Preserve subject identity and missingness. Finish the line within the budget.')
NOTE_LABEL='\nAuxiliary working material (unverified; check it against the evidence above):\n'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False));tmp.replace(p)
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def public(r):return {k:copy.deepcopy(r[k]) for k in ('id','state','question','options')}
def token(rng):return ''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(7))

def corpus():
 rows=[]
 for family in P['families']:
  for index in range(6):
   rng=random.Random(int(digest([P['id'],P['seed'],family,index])[:16],16));key=token(rng);source=digest([P['id'],family,index])[:24]
   items=[];pair_values=[];answers=[];supports=[]
   def add(name,text):items.append((name,text))
   noise_count=(3,8,15)[index%3]
   if family=='join':
    n=4+2*(index%2);a=[token(rng) for _ in range(n)];b=[token(rng) for _ in range(n)];dest=rng.sample(['violet','silver','ochre','jade','coral','sienna','navy','olive'],n)
    chosen=rng.sample(range(n),2);pair_values=[a[i] for i in chosen];answers=[dest[i] for i in chosen]
    add('target',f'Client {key}: active service {{EDIT}}.')
    for j in range(n):add(f'link{j}',f'Service {a[j]}: owner group {b[j]}.');add(f'route{j}',f'Group {b[j]}: delivery region {dest[j]}.')
    add('archive',f'Archived route for client {key}: {rng.choice(dest)}. Archived entries are not current.')
    question=f'Find the delivery region for client {key}. Follow the active service, its owner group, then the group region. Ignore the archived route and records for other clients.'
    options=[{'id':x,'description':f'Delivery region {x}.'} for x in dest]
    supports=[{'target',f'link{i}',f'route{i}'} for i in chosen]
   elif family=='judge':
    sub=index%3;options=[{'id':'supported','description':'The stated conclusion is supported.'},{'id':'arithmetic_error','description':'The stated numerical calculation is wrong.'},{'id':'scope_error','description':'The conclusion extends beyond the population actually checked.'},{'id':'insufficient','description':'A required observation is unrecorded; no contrary observation establishes an error.'}]
    question=f'Judge the conclusion for lot {key} against its evidence only. Select the single applicable verdict. An unrecorded fact is neither positive nor negative. Information about other lots is irrelevant.'
    if sub==0:
     n=rng.randint(38,93);bad=rng.randint(5,17);good=n-bad
     add('fact',f'Lot {key}: inspected={n}; failed={bad}; every inspected item either passed or failed.')
     add('claim',f'Conclusion for lot {key}: {{EDIT}} inspected items passed.')
     pair_values=[str(good),str(good+1)];answers=['supported','arithmetic_error'];supports=[{'fact','claim'}]*2
    elif sub==1:
     n=rng.randint(32,65);total=n+rng.randint(15,40)
     add('fact',f'Lot {key}: total={total}; inspected={n}; all inspected items passed; uninspected outcomes are unrecorded.')
     add('claim',f'Conclusion for lot {key}: {{EDIT}} passed the inspection.')
     pair_values=['Every inspected item','Every item in the entire lot'];answers=['supported','scope_error'];supports=[{'fact','claim'}]*2
    else:
     add('rule',f'Criterion for lot {key}: acceptance requires both a passed pressure test and a passed seal test. Neither test implies the other.')
     add('fact',f'Lot {key}: pressure_test=passed; seal_test={{EDIT}}.')
     add('claim',f'Conclusion for lot {key}: both required tests passed, so acceptance is established.')
     pair_values=['passed','unrecorded'];answers=['supported','insufficient'];supports=[{'rule','fact','claim'}]*2
   else:
    options=[{'id':'release','description':'Release the shipment.'},{'id':'hold','description':'Hold the active shipment.'},{'id':'reject','description':'Reject the inactive shipment.'},{'id':'unknown','description':'The policy cannot decide because required evidence is unrecorded.'}]
    question=f'Apply the current dispatch specification to shipment {key}. Evaluate priorities in order. Unrecorded facts are not false. Obsolete specifications cannot override the current specification.'
    add('version','Dispatch specification V3 is current; V1 is obsolete.')
    add('rule1','V3 priority 1: active=no means reject; active=unrecorded means unknown.')
    add('rule2','V3 priority 2: for active=yes, release if permit=yes or supervisor_waiver=yes. If both are no, hold. Otherwise return unknown.')
    add('old','V1 obsolete: any missing permit means hold even if a supervisor waiver is recorded.')
    if index%3==0:
     add('active',f'Shipment {key}: active=yes.');add('permit',f'Shipment {key}: permit=no.');add('waiver',f'Shipment {key}: supervisor_waiver={{EDIT}}.')
     pair_values=['no','yes'];answers=['hold','release'];supports=[{'version','rule1','rule2','active','permit','waiver'}]*2
    elif index%3==1:
     add('active',f'Shipment {key}: active={{EDIT}}.');add('permit',f'Shipment {key}: permit=yes.');add('waiver',f'Shipment {key}: supervisor_waiver=no.')
     pair_values=['no','yes'];answers=['reject','release'];supports=[{'version','rule1','active'},{'version','rule1','rule2','active','permit'}]
    else:
     add('active',f'Shipment {key}: active=yes.');add('permit',f'Shipment {key}: permit={{EDIT}}.');add('waiver',f'Shipment {key}: supervisor_waiver=no.')
     pair_values=['unrecorded','yes'];answers=['unknown','release'];supports=[{'version','rule1','rule2','active','permit','waiver'},{'version','rule1','rule2','active','permit'}]
   for j in range(noise_count):
    k=token(rng);add(f'noise{j}',f'Unrelated file for {k}: review count {rng.randint(30,90)}, reserve {rng.randint(50,900)} units, archive status retained. These observations concern only {k}; no other subject is identified.')
   rng.shuffle(items);rng.shuffle(options)
   # Flip endpoint order independently; both endpoints preserve the public record IDs.
   endpoint_order=[0,1];rng.shuffle(endpoint_order)
   for edit,j in enumerate(endpoint_order):
    rendered=[{'number':i+1,'text':text.replace('{EDIT}',pair_values[j])} for i,(name,text) in enumerate(items)]
    needed=sorted(i+1 for i,(name,_) in enumerate(items) if name in supports[j])
    state='Extract. Record numbers are identifiers, not priority.\n'+'\n'.join(f'[{r["number"]:02d}] {r["text"]}' for r in rendered)
    rows.append({'id':digest([source,edit])[:24],'source':source,'family':family,'edit':edit,'state':state,'question':question,'options':options,'expected':answers[j],'records':rendered,'reference_support':needed})
 return rows

def parsed_records(row):
 result={}
 for line in row['state'].splitlines():
  m=re.fullmatch(r'\[(\d+)\] (.+)',line)
  if m:
   i=int(m[1])
   if i in result:raise ValueError('Duplicate visible record number')
   result[i]=m[2]
 if not result:raise ValueError('No visible records')
 return result

def independent_reference(row):
 """Check construction from rendered evidence and criterion, no reference fields."""
 text=row['state'];q=row['question']
 if q.startswith('Find the delivery'):
  who=re.search(r'for client ([A-Z]+)',q)[1]
  a=re.search(r'Client '+who+r': active service ([A-Z]+)',text)[1]
  b=re.search(r'Service '+a+r': owner group ([A-Z]+)',text)[1]
  return re.search(r'Group '+b+r': delivery region ([a-z]+)',text)[1]
 if q.startswith('Judge'):
  who=re.search(r'for lot ([A-Z]+)',q)[1]
  if 'inspected items passed.' in text:
   n,f=map(int,re.search(r'Lot '+who+r': inspected=(\d+); failed=(\d+)',text).groups());claim=int(re.search(r'Conclusion for lot '+who+r': (\d+) inspected',text)[1]);return 'supported' if n-f==claim else 'arithmetic_error'
  if 'uninspected outcomes' in text:
   return 'scope_error' if f'Conclusion for lot {who}: Every item in the entire lot' in text else 'supported'
  s=re.search(r'Lot '+who+r': pressure_test=passed; seal_test=(\w+)',text)[1];return 'supported' if s=='passed' else 'insufficient'
 who=re.search(r'to shipment ([A-Z]+)',q)[1]
 a=re.search(r'Shipment '+who+r': active=(\w+)',text)[1]
 if a=='no':return 'reject'
 if a=='unrecorded':return 'unknown'
 p=re.search(r'Shipment '+who+r': permit=(\w+)',text)[1];w=re.search(r'Shipment '+who+r': supervisor_waiver=(\w+)',text)[1]
 return 'release' if 'yes' in (p,w) else 'hold' if p==w=='no' else 'unknown'

def select_ids(text,row):
 try:
  x=json.loads(text)
  if type(x)!=list or not 1<=len(x)<=6 or any(type(i)!=int for i in x) or len(set(x))!=len(x):raise ValueError('Expected one to six distinct integer record IDs')
  records=parsed_records(row)
  if any(i not in records for i in x):raise ValueError('ID not present in public evidence')
  return x,'\n'.join(f'[{i:02d}] {records[i]}' for i in x),None
 except (ValueError,TypeError) as e:return None,'No valid record selection was produced.',str(e)

def selftest():
 rows=corpus();assert len(rows)==36 and len({r['source'] for r in rows})==18
 seen=set()
 for r in rows:
  assert independent_reference(r)==r['expected']
  assert len(r['reference_support'])<=6
  assert parsed_records(r)=={x['number']:x['text'] for x in r['records']}
  ids,txt,err=select_ids(json.dumps(r['reference_support']),r);assert ids==r['reference_support'] and err is None
  k=digest(public(r));assert k not in seen;seen.add(k)
  a=copy.deepcopy(r);a['expected']='SECRET';a['reference_support']=[];a['records']=[];assert public(a)==public(r)
 for source in {r['source'] for r in rows}:
  a,b=[r for r in rows if r['source']==source];assert a['options']==b['options'] and a['question']==b['question'] and a['expected']!=b['expected']
  assert sum(x!=y for x,y in zip(a['state'].splitlines(),b['state'].splitlines()))==1
 assert select_ids('[true]',rows[0])[0] is None;assert select_ids('[1,1]',rows[0])[0] is None;assert select_ids('[9999]',rows[0])[0] is None
 return {'rows':36,'source_pairs':18,'corpus_sha256':digest(rows),'protocol_sha256':digest(P),'reference_checks':36}

def generation_messages(row,mode):
 system={'full_workspace':FULL,'evidence_ids':SELECT,'compact_state':STATE}[mode]
 # Outcomes carry meanings but no answer letters. References are absent.
 payload={'criterion':row['question'],'evidence':row['state'],'outcome_meanings':[o['description'] for o in row['options']]}
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]

def with_note(row,note):
 r=public(row);r['state']+=NOTE_LABEL+note;return r

def match_filler(rt,row,target_tokens,encoder):
 """Ablates only the generated note's usable text; no answer/reference access."""
 prefix=''
 seed=' pad'
 def encode(n,suffix=''):
  note=prefix+seed*n+suffix
  return note,encoder(rt.tokenizer,with_note(row,note),'baseline',P['max_input_tokens'])
 lo,hi=0,512
 while lo<hi:
  mid=(lo+hi)//2
  if encode(mid)[1]['input_tokens']<target_tokens:lo=mid+1
  else:hi=mid
 for n in range(max(0,lo-3),min(512,lo+3)+1):
  for suffix in ('',' x',' x y','.'):
   note,e=encode(n,suffix)
   if e['input_tokens']==target_tokens:return note
 raise RuntimeError('Could not construct exact-length discarded-state control')

def run(args):
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
 from flow_study import Runtime
 from prompt_variants import encode_checked
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',P);save(out/'checks.json',selftest());allrows=corpus();save(out/'cases.json',allrows)
 rt=Runtime();save(out/'runtime.json',rt.meta)
 # Warm up with a neutral one-case replay, not a target-based configuration choice.
 smoke={'id':'smoke','state':'The one parcel is turquoise.','question':'Which color is recorded?','options':[{'id':'turquoise','description':'Turquoise.'},{'id':'maroon','description':'Maroon.'}]}
 s0=rt.score(smoke,'baseline');s1=rt.score(smoke,'baseline');diff=max(abs(x-y) for x,y in zip(s0['logits'],s1['logits']));assert diff<=1e-4;save(out/'smoke.json',{'max_logit_difference':diff,'calls':2})
 # All modes of one request share a worker; balance exactly three requests per worker.
 selected=[r for i,r in enumerate(allrows) if i%P['shards']==args.shard]
 save(out/'assignment.json',{'ids':[r['id'] for r in selected],'shard':args.shard,'corpus_sha256':digest(allrows)})
 notes=[];results=[]
 with (out/'generations.jsonl').open('w') as gf,(out/'records.jsonl').open('w') as rf:
  for ix,row in enumerate(selected):
   generated={};ridx=allrows.index(row)
   gm=['full_workspace','evidence_ids','compact_state'];rot=ridx%3;gm=gm[rot:]+gm[:rot]
   for mode in gm:
    text=rt.tokenizer.apply_chat_template(generation_messages(row,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=rt.tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>P['max_input_tokens']:raise RuntimeError('Oversized generation input')
    x=rt.torch.tensor([ids]);tic=time.perf_counter()
    with rt.torch.inference_mode():
     y=rt.model.generate(input_ids=x,attention_mask=rt.torch.ones_like(x),do_sample=False,max_new_tokens=P['generation_limits'][mode],use_cache=True,pad_token_id=rt.tokenizer.eos_token_id)
    toks=y[0,len(ids):].tolist();note=rt.tokenizer.decode(toks,skip_special_tokens=True)
    gen={'id':row['id'],'source':row['source'],'family':row['family'],'mode':mode,'text':note,'token_ids':toks,'generated_tokens':len(toks),'input_tokens':len(ids),'cap':P['generation_limits'][mode],'hit_cap':len(toks)>=P['generation_limits'][mode],'seconds':time.perf_counter()-tic,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
    generated[mode]=gen;notes.append(gen);gf.write(json.dumps(gen,allow_nan=False)+'\n');gf.flush();rt.hidden=None
   selected_ids,evidence,error=select_ids(generated['evidence_ids']['text'],row)
   compact=generated['compact_state']['text'];clen=encode_checked(rt.tokenizer,with_note(row,compact),'baseline',P['max_input_tokens'])['input_tokens']
   filler=match_filler(rt,row,clen,encode_checked)
   actual_notes={'full_workspace':generated['full_workspace']['text'],'evidence_ids':evidence,'compact_state':compact,'discarded_state':filler}
   order=list(MODES);rot=ridx%len(order);order=order[rot:]+order[:rot]
   for mode in order:
    r=public(row) if mode=='direct' else with_note(row,actual_notes[mode]);rec=rt.score(r,'baseline')
    charged=None if mode=='direct' else generated['compact_state' if mode=='discarded_state' else mode]
    rec.update({'mode':mode,'source':row['source'],'family':row['family'],'edit':row['edit'],'expected':row['expected'],'correct':rec['predicted']==row['expected'],'generation_input_tokens':charged['input_tokens'] if charged else 0,'generation_output_tokens':charged['generated_tokens'] if charged else 0,'accounted_seconds':rec['request_seconds']+(charged['seconds'] if charged else 0),'generation_reused_for_control':mode=='discarded_state','selected_ids':selected_ids if mode=='evidence_ids' else None,'selection_error':error if mode=='evidence_ids' else None,'note':actual_notes.get(mode),'note_input_target':clen if mode in ('compact_state','discarded_state') else None})
    if mode in ('compact_state','discarded_state'):assert rec['input_tokens']==clen
    results.append(rec);rf.write(json.dumps(rec,allow_nan=False)+'\n');rf.flush()
    print(json.dumps({'event':'scored','shard':args.shard,'case':ix+1,'total_cases':len(selected),'mode':mode,'completed':len(results)}),flush=True)
 save(out/'receipt.json',{'records':len(results),'generations':len(notes),'records_sha256':sha(out/'records.jsonl'),'generation_sha256':sha(out/'generations.jsonl'),'protocol_sha256':digest(P),'weight_updates':0,'final_score_calls':len(results),'actual_generation_calls':len(notes),'warmup_calls':2})

def summarize(rs,gs):
 out={};base={r['id']:r for r in rs if r['mode']=='direct'}
 for mode in MODES:
  rows=[r for r in rs if r['mode']==mode];sources={r['source'] for r in rows};families={}
  for f in P['families']:
   sub=[r for r in rows if r['family']==f];ss={r['source'] for r in sub};families[f]={'n':len(sub),'correct':sum(r['correct'] for r in sub),'pairs':len(ss),'both_correct':sum(all(r['correct'] for r in sub if r['source']==s) for s in ss)}
  out[mode]={'n':len(rows),'correct':sum(r['correct'] for r in rows),'nll':statistics.mean(-math.log(max(r['probabilities'][r['labels'].index(r['expected'])],1e-15)) for r in rows),'repairs':sum(r['correct'] and not base[r['id']]['correct'] for r in rows),'regressions':sum(not r['correct'] and base[r['id']]['correct'] for r in rows),'complete_pairs':sum(all(r['correct'] for r in rows if r['source']==s) for s in sources),'families':families,'processed_input_tokens':sum(r['input_tokens']+r['generation_input_tokens'] for r in rows),'generated_tokens_charged':sum(r['generation_output_tokens'] for r in rows),'median_accounted_seconds':statistics.median(r['accounted_seconds'] for r in rows),'selection_invalid':sum(r.get('selection_error') is not None for r in rows)}
 return out

def aggregate(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rs=[];gs=[];seen=set();source_receipts=[]
 for path in sorted(Path(args.root).rglob('records.jsonl')):
  receipt=json.loads((path.parent/'receipt.json').read_text());g=path.parent/'generations.jsonl';assert sha(path)==receipt['records_sha256'] and sha(g)==receipt['generation_sha256'] and receipt['protocol_sha256']==digest(P)
  part=lines(path);gen=lines(g);assert len(part)==receipt['records'] and len(gen)==receipt['generations']
  for r in part:
   key=(r['id'],r['mode']);assert key not in seen;seen.add(key);rs.append(r)
  gs+=gen;source_receipts.append({'folder':path.parent.name,**receipt})
 assert seen=={(r['id'],m) for r in corpus() for m in MODES}
 assert len(gs)==108 and len({(g['id'],g['mode']) for g in gs})==108
 rs.sort(key=lambda r:(r['id'],r['mode']));gs.sort(key=lambda r:(r['id'],r['mode']))
 for name,items in [('records',rs),('generations',gs)]:
  (out/f'{name}.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in items))
 save(out/'results.json',{'summary':summarize(rs,gs),'protocol':P,'selection':'none; no model promotion','official_score':None});save(out/'cases.json',corpus());save(out/'source_receipts.json',source_receipts);save(out/'receipt.json',{'records_sha256':sha(out/'records.jsonl'),'generation_sha256':sha(out/'generations.jsonl'),'records':len(rs),'generations':len(gs),'protocol_sha256':digest(P)})
 print(json.dumps(summarize(rs,gs),indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('selftest')
 r=sub.add_parser('run');r.add_argument('--shard',type=int,choices=range(12),required=True);r.add_argument('--out',required=True)
 a=sub.add_parser('aggregate');a.add_argument('--root',required=True);a.add_argument('--out',required=True)
 args=parser.parse_args()
 if args.command=='selftest':print(json.dumps(selftest(),indent=2))
 else:
  try:run(args) if args.command=='run' else aggregate(args)
  except Exception as exc:
   out=Path(args.out);out.mkdir(parents=True,exist_ok=True);save(out/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc(),'complete_result':False});raise
