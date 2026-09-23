"""Frozen, off-benchmark workspace transfer test. No training or oracle at inference."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,sys,time,traceback
from datetime import datetime,timedelta,timezone
from decimal import Decimal
from pathlib import Path
PARENT=Path(__file__).resolve().parents[1]/'input_flow_v1'
if not PARENT.exists(): PARENT=Path(__file__).resolve().parent/'parent'
sys.path.insert(0,str(PARENT))
from flow_study import Runtime,save,filehash,softmax,prediction,corpus as benchmark_corpus
from prompt_variants import messages as baseline_messages
FAMILIES=('amount','elapsed','join','policy','judge','probability')
MODES=('baseline','bracket','workspace')
PROTOCOL={'id':'decision0-workspace-transfer-v1','seed':934271,'rows':72,'sources':36,
 'modes':list(MODES),'max_input_tokens':8192,'max_workspace_tokens':160,'thinking':False,
 'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'selection':'Equal-family deterministic accuracy and probability fidelity; ties fewer mean processed tokens then declared order.',
 'promotion_gate':'Workspace quality gain >=0.05 over baseline; no deterministic family loses >0.10; zero model failures.',
 'weights_updated':False,'official_score':None,'training_examples':0,'shards':8,
 'scope':'Six constructed generators, not an independent human-authored business corpus. Public evaluation is separate.'}
WORKSPACE_SYSTEM=('Build a brief, evidence-grounded decision workspace, not an answer letter. '
 'Use three named sections: Relevant evidence; Computation or governing rule; Derived result. '
 'For arithmetic show needed intermediate values with units and signs; for dates respect UTC offsets; '
 'for policies resolve versions and exceptions; for linked records identify the complete chain; '
 'for judging identify the decisive correct claim or error; for event probabilities state the relevant '
 'population and the numerical outcome distribution, not confidence in a winning label. '
 'Include only sections relevant to this request. The supplied evidence is authoritative. '
 'Do not invent missing facts. Be concise; do not repeat the input or output an option letter.')
FINAL='Check the working notes against the original evidence. Apply the original criterion. Return only the uppercase letter of one listed option.'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def visible(r):return {k:r[k] for k in ('id','state','question','options')}
def token(rng):return ''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(7))
def dollars(v):return f'{v/100:.2f}'
def make_pair(fam,index):
 rng=random.Random(int(digest([PROTOCOL['id'],fam,index])[:16],16));key=token(rng);source=digest([fam,index,'source'])[:24]
 count_noise=(2,10,26)[index%3]
 noise=[f'Record for {token(rng)} only: quantity {rng.randint(2,90)}, operating reserve {rng.randint(10,900)} units, review date 2027-02-17, team {token(rng)}. Its values do not apply to other subjects. Record retained for the next quarterly audit.' for _ in range(count_noise)]
 required=[];values=[];oracles=[];answers=[];gold=[]
 if fam=='amount':
  n=rng.randint(3,17);unit=rng.randint(131,1359);fee=rng.randint(80,230);credit=rng.randint(5,300)
  if index%2:credit=n*unit+fee+rng.randint(20,400)
  cap=n*unit+fee-credit
  required=[f'Invoice {key}: {n} units, unit charge {unit} cents.',f'Invoice {key}: handling charge {{CHANGE}} cents.',f'Invoice {key}: credit {credit} cents. Signed net totals are permitted.']
  values=[str(fee-1),str(fee+1)];answers=['below','above'];opts=[('below','Net total is strictly below the ceiling.'),('equal','Net total exactly equals the ceiling.'),('above','Net total is strictly above the ceiling.')]
  question=f'Calculate invoice {key}: units times unit charge, plus handling, minus credit. Classify its exact signed net total relative to the ceiling of {cap} cents. Use only this invoice; do not round.'
  oracles=[{'kind':fam,'key':key,'n':n,'unit':unit,'fee':int(v),'credit':credit,'boundary':cap} for v in values]
 elif fam=='elapsed':
  start=datetime(2028,1+index,3+index,21,43,tzinfo=timezone(timedelta(minutes=rng.choice([-300,120,330]))));dur=rng.choice([127,493,1467]);zone=timezone(timedelta(minutes=rng.choice([-240,0,345])))
  required=[f'Case {key}: clock starts {start.isoformat()}.',f'Case {key}: receipt {{CHANGE}}.',f'Case {key}: window lasts exactly {dur} elapsed minutes; business-day rules do not apply.']
  values=[(start+timedelta(minutes=dur+d)).astimezone(zone).isoformat() for d in (-1,1)];answers=['before','after'];opts=[('before','Receipt is before the deadline.'),('at','Receipt is exactly at the deadline.'),('after','Receipt is after the deadline.')]
  question=f'For case {key}, compare receipt time to start plus the allowed window. Respect explicit UTC offsets; all timestamps are complete. Ignore other cases.'
  oracles=[{'kind':fam,'key':key,'start':start.isoformat(),'duration':dur,'receipt':v} for v in values]
 elif fam=='join':
  contracts=[token(rng) for _ in range(6)];teams=[token(rng) for _ in range(6)];pools=rng.sample(['amber','cobalt','plum','ivory','sage','coral','teal','pearl','umber'],6)
  required=[f'Client {key}: current agreement {{CHANGE}}.']+[f'Agreement {c}: servicing team {t}.' for c,t in zip(contracts,teams)]+[f'Team {t}: current destination {p}.' for t,p in zip(teams,pools)]
  chosen=rng.sample(range(6),2);values=[contracts[j] for j in chosen];answers=[pools[j] for j in chosen];opts=[(p,f'Destination {p}.') for p in pools]
  question=f'Route client {key} by following its current agreement, that agreement\'s team, and that team\'s destination. All listed mappings are current and refer to distinct subjects.'
  oracles=[{'kind':fam,'key':key,'contract':contracts[j],'team':teams[j],'destination':pools[j]} for j in chosen]
 elif fam=='policy':
  axis=index%3;driver=('clearance','active','active')[axis]
  fixed=({'active':'yes','hold':'yes'},{'hold':'no','clearance':'no'},{'hold':'no','clearance':'no'})[axis]
  values=(['no','yes'],['no','yes'],['unrecorded','yes'])[axis];answers=(['hold','allow'],['deny','allow'],['unknown','allow'])[axis]
  required=[f'Account {key}: {k} '+('{CHANGE}' if k==driver else fixed[k])+'.' for k in ('active','hold','clearance')]
  required += ['Export policy R2 is current. R1 is obsolete. R1 rejected every held account even with clearance.',
   'R2 priority 1: if inactive is recorded (active=no), deny. Priority 2: if active is unrecorded, report insufficient evidence.',
   'R2 priority 3: for an active account, allow when hold=no OR clearance=yes. Priority 4: otherwise if hold=yes AND clearance=no, place on hold. In all remaining cases report insufficient evidence. Earlier priorities prevail.']
  question=f'Apply the current export policy to account {key}. Unrecorded is neither yes nor no. Return the unique decision under R2, not R1.'
  opts=[('allow','Allow export now.'),('deny','Deny due to recorded inactivity.'),('hold','Place on hold pending clearance.'),('unknown','Insufficient evidence to decide under the current policy.')]
  oracles=[{'kind':fam,'key':key,**fixed,driver:v} for v in values]
 elif fam=='judge':
  opts=[('supported','The proposed conclusion is supported.'),('arithmetic_error','Its numerical calculation is incorrect.'),('scope_error','It improperly expands the scope of the evidence.'),('insufficient','A necessary fact is absent, without a demonstrated contradiction.')]
  question=f'Evaluate the proposed conclusion about inspection batch {key}. Choose supported if fully warranted. Otherwise choose the one error category that applies. Do not use information about other batches.'
  if index%2==0:
   inspected=rng.randint(30,60);failed=rng.randint(3,12);good=inspected-failed
   required=[f'Batch {key}: exactly {inspected} items inspected, exactly {failed} failed. Every item either passed or failed; none pending.',f'Batch {key}: proposed conclusion: {{CHANGE}} items passed inspection.']
   values=[str(good),str(good+1)];answers=['supported','arithmetic_error'];oracles=[{'kind':fam,'key':key,'inspected':inspected,'failed':failed,'claim':int(v),'subkind':'numeric'} for v in values]
  else:
   inspected=rng.randint(30,60);total=inspected+rng.randint(10,40)
   required=[f'Batch {key}: {total} items in total. Only {inspected} were inspected; every inspected item passed. No status is known for uninspected items.',f'Batch {key}: proposed conclusion: {{CHANGE}} passed inspection.']
   values=['Every inspected item','Every item in the whole batch'];answers=['supported','scope_error'];oracles=[{'kind':fam,'key':key,'inspected':inspected,'total':total,'claim':v,'subkind':'scope'} for v in values]
 else:
  region=token(rng);other=token(rng);a=rng.randint(8,19);b=rng.randint(31,49);c=rng.randint(9,23)
  required=[f'Complete cohort {region}: pass={{CHANGE}}, rework={b}, scrap={c}. These categories are exhaustive and mutually exclusive.',f'Complete cohort {other}: pass=700, rework=90, scrap=210. This separate cohort is not part of {region}.']
  values=[str(a),str(b+c+21)];gold=[{'pass':v/(v+b+c),'rework':b/(v+b+c),'scrap':c/(v+b+c)} for v in (a,b+c+21)];answers=[max(g,key=g.get) for g in gold]
  question=f'A member is drawn uniformly from complete cohort {region} only. Return the distribution over its recorded outcome. Use the stated counts as a complete population, not a sample estimate; exclude the other cohort.'
  opts=[('pass','Member has outcome pass.'),('rework','Member has outcome rework.'),('scrap','Member has outcome scrap.')];oracles=[{'kind':fam,'key':key,'region':region,'pass':int(v),'rework':b,'scrap':c} for v in values]
 rng.shuffle(opts);options=[{'id':k,'description':d} for k,d in opts];records=required+noise;rng.shuffle(records)
 assert sum('{CHANGE}' in r for r in records)==1
 rows=[]
 for edit in range(2):
  state='Operations extract. Records apply only to their named subject.\n'+'\n'.join(f'[{i+1:03}] {r.replace("{CHANGE}",values[edit])}' for i,r in enumerate(records))
  target=gold[edit] if gold else {k:float(k==answers[edit]) for k,_ in opts}
  rows.append({'id':digest([source,edit])[:24],'source':source,'edit':edit,'family':fam,'tier':'transfer','state':state,'question':question,'options':options,'expected':answers[edit],'target_probs':target,'gold_probs':target if fam=='probability' else None,'oracle':oracles[edit],'length_stratum':index%3})
 return rows

def corpus():return [r for f in FAMILIES for i in range(6) for r in make_pair(f,i)]
def independent_reference(r):
 import re
 s=r['state'];o=r['oracle'];k=o['key'];f=r['family']
 if f=='amount':
  n,u=map(int,re.search(rf'Invoice {k}: (\d+) units, unit charge (\d+) cents',s).groups());fee=int(re.search(rf'Invoice {k}: handling charge (\d+) cents',s).group(1));credit=int(re.search(rf'Invoice {k}: credit (\d+) cents',s).group(1));boundary=int(re.search(r'ceiling of (-?\d+) cents',r['question']).group(1));v=Decimal(n)*Decimal(u)+Decimal(fee)-Decimal(credit);ans='below' if v<boundary else 'above' if v>boundary else 'equal'
 elif f=='elapsed':
  a=re.search(rf'Case {k}: clock starts (\S+)\.',s).group(1);b=re.search(rf'Case {k}: receipt (\S+)\.',s).group(1);dur=int(re.search(rf'Case {k}: window lasts exactly (\d+) elapsed minutes',s).group(1));v=(datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()-dur*60;ans='before' if v<0 else 'after' if v>0 else 'at'
 elif f=='join':
  c=re.search(rf'Client {k}: current agreement ([A-Z]+)',s).group(1);t=re.search(rf'Agreement {c}: servicing team ([A-Z]+)',s).group(1);ans=re.search(rf'Team {t}: current destination ([a-z]+)',s).group(1)
 elif f=='policy':
  vals={x:re.search(rf'Account {k}: {x} (yes|no|unrecorded)',s).group(1) for x in ('active','hold','clearance')}
  ans='deny' if vals['active']=='no' else 'unknown' if vals['active']=='unrecorded' else 'allow' if vals['hold']=='no' or vals['clearance']=='yes' else 'hold' if vals['hold']=='yes' and vals['clearance']=='no' else 'unknown'
 elif f=='judge':
  if o['subkind']=='numeric':
   n,bad=map(int,re.search(rf'Batch {k}: exactly (\d+) items inspected, exactly (\d+) failed',s).groups());v=int(re.search(rf'Batch {k}: proposed conclusion: (\d+) items',s).group(1));ans='supported' if v==n-bad else 'arithmetic_error'
  else:ans='scope_error' if 'proposed conclusion: Every item in the whole batch' in s else 'supported'
 else:
  a,b,c=map(int,re.search(rf'Complete cohort {o["region"]}: pass=(\d+), rework=(\d+), scrap=(\d+)',s).groups());return dict(zip(('pass','rework','scrap'),(x/(a+b+c) for x in (a,b,c))))
 return {x['id']:float(x['id']==ans) for x in r['options']}

def selftest():
 rows=corpus();assert len(rows)==72 and len({r['id'] for r in rows})==72
 assert len({digest({k:v for k,v in visible(r).items() if k!='id'}) for r in rows})==72
 for r in rows:
  assert independent_reference(r)==r['target_probs']
  assert r['expected']==max(sorted(r['target_probs']),key=r['target_probs'].get)
  assert set(visible(r))=={'id','state','question','options'}
 for a,b in zip(rows[::2],rows[1::2]):
  assert a['source']==b['source'] and a['options']==b['options'] and a['question']==b['question']
  assert sum(x!=y for x,y in zip(a['state'].splitlines(),b['state'].splitlines()))==1
  assert a['expected']!=b['expected']
 return {'rows':72,'sources':36,'corpus_hash':digest(rows),'families':list(FAMILIES),'max_chars':max(len(r['state'])+len(r['question']) for r in rows)}

def encode(rt,msgs,row):
 tok=rt.tokenizer;text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(text,add_special_tokens=False)
 if not 0<len(ids)<=PROTOCOL['max_input_tokens']:raise ValueError('Input exceeds limit; no truncation')
 slots=[]
 for i in range(len(row['options'])):
  letter=chr(65+i);t=tok.encode(letter,add_special_tokens=False)
  if len(t)!=1 or tok.encode(text+letter,add_special_tokens=False)!=ids+t:raise ValueError('Answer boundary changed')
  slots.append(t[0])
 return ids,slots,hashlib.sha256(text.encode()).hexdigest()
def score_messages(rt,msgs,r):
 import torch
 ids,slots,ph=encode(rt,msgs,r);x=torch.tensor([ids]);rt.hidden=None
 with torch.inference_mode():
  out=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1)
  if rt.hidden is None:raise RuntimeError('No hidden state')
  b=rt.head.bias[slots].float() if rt.head.bias is not None else None
  z=torch.nn.functional.linear(rt.hidden.float(),rt.head.weight[slots].float(),b)[0].tolist()
 del out;rt.hidden=None;p=softmax(z);labels=[o['id'] for o in r['options']]
 return {'labels':labels,'logits':z,'probabilities':p,'predicted':prediction(labels,p),'input_tokens':len(ids),'prompt_sha256':ph}
def score(rt,r,mode):
 import torch
 tick=time.perf_counter();base=baseline_messages(visible(r),'baseline')
 if mode!='workspace':
  rec=rt.score(r,'baseline' if mode=='baseline' else 'schema_bracket');rec.update(mode=mode,processed_input_tokens=rec['input_tokens'],workspace_tokens=0,workspace=None,workspace_seconds=0.)
  return rec
 wm=[{'role':'system','content':WORKSPACE_SYSTEM},base[1]]
 tok=rt.tokenizer;text=tok.apply_chat_template(wm,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(text,add_special_tokens=False)
 if not 0<len(ids)<=PROTOCOL['max_input_tokens']:raise ValueError('Workspace input limit')
 x=torch.tensor([ids]);t=time.perf_counter()
 with torch.inference_mode():
  y=rt.model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=PROTOCOL['max_workspace_tokens'],use_cache=True,pad_token_id=tok.eos_token_id)
 out=y[0,len(ids):].tolist();raw=tok.decode(out,skip_special_tokens=True);workspace_seconds=time.perf_counter()-t
 del x,y;rt.hidden=None
 final=base+[{'role':'assistant','content':'Working notes (verify against the original evidence):\n'+raw},{'role':'user','content':FINAL}]
 rec=score_messages(rt,final,r)
 rec.update(id=r['id'],mode=mode,ok=True,workspace=raw,workspace_token_ids=out,workspace_tokens=len(out),workspace_input_tokens=len(ids),processed_input_tokens=len(ids)+rec['input_tokens'],workspace_seconds=workspace_seconds,workspace_limit_hit=len(out)>=PROTOCOL['max_workspace_tokens'],workspace_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),request_seconds=time.perf_counter()-tick)
 return rec

def assignment(rows,n):
 loads=[0]*n;out=[[] for _ in range(n)]
 for i in sorted(range(len(rows)),key=lambda i:(-len(json.dumps(visible(rows[i]))),i)):
  k=min(range(n),key=lambda k:(loads[k],k));out[k].append(i);loads[k]+=len(json.dumps(visible(rows[i])))+800
 return [sorted(x) for x in out]
def run(args):
 p=Path(args.out);p.mkdir(parents=True,exist_ok=False);rows=corpus() if args.phase=='dev' else benchmark_corpus('bench');modes=MODES
 if args.phase=='bench':
  sel=json.loads(Path(args.selection).read_text());assert sel['protocol_hash']==digest(PROTOCOL) and sel['gate_passed'];modes=('baseline',sel['selected']);save(p/'selection.json',sel)
 save(p/'protocol.json',PROTOCOL);save(p/'cases.json',rows);save(p/'preflight.json',selftest());indices=assignment(rows,args.nshards)[args.shard];save(p/'assignment.json',{'ids':[rows[i]['id'] for i in indices],'corpus_hash':digest(rows),'phase':args.phase,'modes':list(modes),'nshards':args.nshards,'shard':args.shard})
 rt=Runtime();save(p/'runtime.json',rt.meta);count=0;errors=0
 with (p/'records.jsonl').open('w') as f:
  for i in indices:
   r=rows[i]
   for mode in modes[i%len(modes):]+modes[:i%len(modes)]:
    try:rec=score(rt,r,mode)
    except Exception as e:rec={'id':r['id'],'mode':mode,'ok':False,'error':repr(e),'traceback':traceback.format_exc(),'predicted':None};errors+=1
    rec.update({k:r.get(k) for k in ('family','tier','source','expected','target_probs','gold_probs','edit','length_stratum')});rec['correct']=rec['ok'] and rec['predicted']==r['expected'];rec['phase']=args.phase
    f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();count+=1
    print(json.dumps({'phase':args.phase,'shard':args.shard,'done':count,'total':len(indices)*len(modes),'mode':mode,'ok':rec['ok']}),flush=True)
 save(p/'receipt.json',{'records':count,'errors':errors,'records_sha256':filehash(p/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'weights_updated':False})
 if errors:raise RuntimeError(f'{errors} retained failures')
def tvd(r):
 q=r.get('gold_probs') or r.get('target_probs')
 if not q:return None
 if not r['ok']:return 1.
 return sum(abs(dict(zip(r['labels'],r['probabilities'])).get(k,0)-v) for k,v in q.items())/2

def metrics(rs):
 ok=[r for r in rs if r['ok']];n=len(rs);p=lambda q:sorted(r['request_seconds'] for r in ok)[round((len(ok)-1)*q)] if ok else None
 m={'n':n,'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/n,'errors':n-len(ok),'nll':sum(-math.log(max(dict(zip(r['labels'],r['probabilities'])).get(r['expected'],0),1e-15)) if r['ok'] else -math.log(1e-15) for r in rs)/n,'input_tokens':sum(r.get('processed_input_tokens',0) for r in rs),'workspace_tokens':sum(r.get('workspace_tokens',0) for r in rs),'p50_s':p(.5),'p95_s':p(.95)}
 prs=[r for r in rs if r.get('gold_probs')];m['probability_n']=len(prs);m['probability_tvd']=statistics.mean(tvd(r) for r in prs) if prs else None
 return m

def aggregate(args):
 p=Path(args.out);p.mkdir(parents=True,exist_ok=False);rows=corpus() if args.phase=='dev' else benchmark_corpus('bench');records=[];seen=set();modes=MODES
 if args.phase=='bench':
  selection=json.loads(Path(args.selection).read_text());assert selection['protocol_hash']==digest(PROTOCOL) and selection['gate_passed'];modes=('baseline',selection['selected']);save(p/'selection.json',selection)
 for f in Path(args.root).rglob('records.jsonl'):
  receipt=json.loads((f.parent/'receipt.json').read_text());a=json.loads((f.parent/'assignment.json').read_text())
  assert receipt['protocol_hash']==digest(PROTOCOL) and receipt['records_sha256']==filehash(f) and a['corpus_hash']==digest(rows)
  rs=[json.loads(x) for x in f.read_text().splitlines()];assert len(rs)==receipt['records']
  for r in rs:
   key=(r['id'],r['mode']);assert key not in seen;seen.add(key);records.append(r)
 assert seen=={(r['id'],v) for r in rows for v in modes},'Incomplete coverage'
 summary={};base={r['id']:r for r in records if r['mode']=='baseline'}
 for mode in modes:
  rs=[r for r in records if r['mode']==mode];m=metrics(rs);m['families']={f:metrics([r for r in rs if r['family']==f]) for f in sorted({r['family'] for r in rs})};m['repairs']=sum(r['correct'] and not base[r['id']]['correct'] for r in rs);m['regressions']=sum(not r['correct'] and base[r['id']]['correct'] for r in rs)
  if args.phase=='dev':
   m['quality']=sum(1-m['families'][f]['probability_tvd'] if f=='probability' else m['families'][f]['accuracy'] for f in FAMILIES)/len(FAMILIES)
   m['pairs']={f:sum(all(r['correct'] for r in rs if r['source']==s) for s in {r['source'] for r in rs if r['family']==f}) for f in FAMILIES if f!='probability'}
  else:m['tiers']={t:metrics([r for r in rs if r['tier']==t]) for t in ('easy','standard','hard')}
  summary[mode]=m
 records.sort(key=lambda r:(r['id'],r['mode']));(p/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));save(p/'cases.json',rows);save(p/'protocol.json',PROTOCOL);save(p/'summary.json',summary)
 if args.phase=='dev':
  selected=max(MODES,key=lambda m:(summary[m]['quality'],-summary[m]['input_tokens'],-MODES.index(m)))
  gain=summary['workspace']['quality']-summary['baseline']['quality'];reg=max(summary['baseline']['families'][f]['accuracy']-summary['workspace']['families'][f]['accuracy'] for f in FAMILIES if f!='probability')
  passed=selected=='workspace' and gain>=.05 and reg<=.10 and not any(m['errors'] for m in summary.values())
  save(p/'selection.json',{'selected':selected,'protocol_hash':digest(PROTOCOL),'corpus_hash':digest(rows),'gate_passed':passed,'workspace_quality_gain':gain,'worst_deterministic_family_regression':reg,'summary':summary,'no_benchmark_selection':True})
 save(p/'receipt.json',{'records':len(records),'records_sha256':filehash(p/'records.jsonl'),'protocol_hash':digest(PROTOCOL),'official_score':None})
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('selftest')
 for cmd in ('run','aggregate'):
  a=sub.add_parser(cmd);a.add_argument('--phase',choices=['dev','bench'],required=True);a.add_argument('--out',required=True);a.add_argument('--selection')
  if cmd=='run':a.add_argument('--shard',type=int,required=True);a.add_argument('--nshards',type=int,default=8)
  else:a.add_argument('--root',required=True)
 a=ap.parse_args()
 if a.cmd=='selftest':print(json.dumps(selftest(),indent=2))
 elif a.cmd=='run':run(a)
 else:aggregate(a)
