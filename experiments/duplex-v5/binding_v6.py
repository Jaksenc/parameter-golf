"""Frozen v6: larger reasoning budget and checkable operand bindings.
No model training, benchmark gold in prompts, or arbitrary generated-code execution.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, re, statistics, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numeric_boundary_fix
import study
from study import digest, save, emit, parse_number, ref_program, source_state, number_bank, program_value
MODEL=study.MODEL; REV=study.MODEL_REV
ARMS=('direct','reasoning','program','binding')
CAPS={'direct':64,'reasoning':512,'program':512,'binding':512}
EXPECTED={'model.safetensors-00001-of-00002.safetensors':'26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61','model.safetensors-00002-of-00002.safetensors':'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}

def metadata_bank(raw,state,question):
 bank=number_bank(state,question)
 for key,v in bank.items():
  text=state if v['source']=='evidence' else question
  a=text.rfind('\n',0,v['start'])+1;b=text.find('\n',v['end']);b=len(text) if b<0 else b
  line=text[a:b];row='';column='';line_id='question' if v['source']=='question' else line.split(':',1)[0]
  if line_id.startswith('table') and line_id[5:].isdigit():
   ri=int(line_id[5:]);cells=list(re.finditer(r'"(?:[^"\\]|\\.)*"',line))
   ci=next((j for j,m in enumerate(cells) if a+m.start()<=v['start'] and v['end']<=a+m.end()),None)
   row=str(raw['table'][ri][0])
   if ci is not None and ci>0 and ci<len(raw['table'][0]):column=str(raw['table'][0][ci])
  v.update(line=line_id,row=row,column=column,context=text[max(a,v['start']-60):min(b,v['end']+90)])
 return bank

def inp_only(r):return {k:r[k] for k in ('evidence','question','bank')}
def messages(r,arm):
 if arm not in ARMS:raise ValueError('unknown arm')
 base='Use only the supplied evidence. Solve the numerical question. Return ratios and percentage changes as decimal fractions (0.25 means 25%); keep other quantities in the requested units. Check the entity, exact start/end periods, units and denominator. '
 endings={
 'direct':'Return one line FINAL=<number>, then stop.',
 'reasoning':'Reason through the calculation. You have up to 512 output tokens, but prefer 3-6 concise lines. Conclude with a separate line FINAL=<number> and then stop.',
 'program':'Return one line PROGRAM=<expression>, then stop. Use only the quantity IDs, parentheses, + - * / and constants 0,1,100. Do not calculate the final number or include prose.',
 'binding':'First bind the necessary operands to their source locations. Return exactly one JSON object with keys bindings and program. bindings is a list of objects, one per distinct quantity used, each with id, line, row, column, role. Copy line, row and column EXACTLY from the bank (empty strings stay empty); role describes its intended entity/date/unit role. program is an arithmetic expression using only those IDs, parentheses, + - * / and constants 0,1,100. Do not calculate the final number. Example shape: {"bindings":[{"id":"N0","line":"table1","row":"revenue","column":"2019","role":"end revenue"}],"program":"N0/100"}. Use the actual appropriate operands, not the example.'}
 bank=[{'id':k,**{f:v[f] for f in ('value','line','row','column','context')}} for k,v in r['bank'].items()]
 user={'document':r['evidence'],'question':r['question'],'quantities':bank}
 return [{'role':'system','content':base+endings[arm]},{'role':'user','content':json.dumps(user,ensure_ascii=False)}]

def parse_response(text,arm,bank):
 s=text.strip()
 if arm in ('direct','reasoning'):
  return study.final_number(s),[],{'binding_verified':False}
 if arm=='program':
  v,used=program_value(s,bank);return v,used,{'binding_verified':False}
 if s.startswith('```'):
  lines=s.splitlines()
  if len(lines)<3 or lines[-1].strip()!='```':raise ValueError('incomplete_fence')
  s='\n'.join(lines[1:-1])
 def unique(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('duplicate_JSON_key')
   d[k]=v
  return d
 obj=json.loads(s,object_pairs_hook=unique)
 if not isinstance(obj,dict) or set(obj)!={'bindings','program'}:raise ValueError('binding_schema')
 if not isinstance(obj['program'],str) or not isinstance(obj['bindings'],list):raise ValueError('binding_types')
 v,used=program_value(obj['program'],bank);seen=[]
 for x in obj['bindings']:
  if not isinstance(x,dict) or set(x)!={'id','line','row','column','role'}:raise ValueError('binding_fields')
  key=x['id']
  if not isinstance(key,str) or key not in bank or key in seen:raise ValueError('binding_id')
  if not isinstance(x['role'],str) or not 1<=len(x['role'])<=200:raise ValueError('binding_role')
  if any(x[f]!=bank[key][f] for f in ('line','row','column')):raise ValueError('source_metadata_mismatch')
  seen.append(key)
 if sorted(seen)!=used:raise ValueError('binding_reference_coverage')
 return v,used,{'binding_verified':True,'qualification':'Source metadata and reference coverage verified; role interpretation is NOT proved','bindings':obj['bindings'],'program':obj['program']}

def prepare(root):
 import urllib.request
 root=Path(root);root.mkdir(parents=True,exist_ok=False)
 allraw=[];old=[];old_docs=set();receipts=[]
 for split,n in [('dev',8),('test',24)]:
  url=f'https://raw.githubusercontent.com/czyssrs/FinQA/{study.FINQA_REV}/dataset/{split}.json'
  with urllib.request.urlopen(url,timeout=90) as f:blob=f.read()
  sh=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
  if sh!=study.FINQA_BLOBS[split]:raise ValueError('dataset bytes changed')
  source=json.loads(blob);oldpool=[]
  for r in source:
   allraw.append((split,r))
   try:
    s=source_state(r);q=r['qa']['question'];bn=number_bank(s,q)
    if len(s)>2500 or len(q)>400 or len(bn)>64:continue
    ans,steps=ref_program(r['qa']['program']);ref=parse_number(r['qa']['exe_ans'])
    if round(float(ans),5)!=round(float(ref),5):continue
    if round(float(parse_number(str(r['qa'].get('answer','')).strip().rstrip('.'))),4)!=round(float(ref),4):continue
    if not bn or abs(ans)>10**10:continue
    doc=r['id'].rsplit('-',1)[0];oldpool.append((digest({'selection':'duplex-v5-fixed-1','id':r['id']}),r,doc))
   except (ValueError,KeyError,TypeError,ZeroDivisionError,OverflowError):continue
  chosen=0
  for _,r,doc in sorted(oldpool):
   if doc in old_docs:continue
   old_docs.add(doc);old.append(r['id']);chosen+=1
   if chosen==n:break
  if chosen!=n:raise ValueError('cannot reconstruct old exclusion population')
  receipts.append({'split':split,'rows':len(source),'git_blob':sh,'sha256':hashlib.sha256(blob).hexdigest()})
 banned={'/'.join(x.split('/')[:2]) for x in old};candidates=[];excluded=Counter()
 for split,r in allraw:
  filing='/'.join(r['id'].split('/')[:2])
  try:
   if filing in banned:raise ValueError('previously_used_filing')
   s=source_state(r);q=r['qa']['question'];bn=metadata_bank(r,s,q)
   if len(s)>3500 or len(q)>400 or len(bn)>80:raise ValueError('context_limit')
   ans,steps=ref_program(r['qa']['program']);ref=parse_number(r['qa']['exe_ans'])
   if round(float(ans),5)!=round(float(ref),5):raise ValueError('reference_program_mismatch')
   if round(float(parse_number(str(r['qa'].get('answer','')).strip().rstrip('.'))),4)!=round(float(ref),4):raise ValueError('annotation_mismatch')
   if not bn or abs(ans)>10**10:raise ValueError('range')
   row={'id':'finqa:'+r['id'],'source_split':split,'filing':filing,'evidence':s,'question':q,'bank':bn}
   gold={'value':str(ans),'reference_program':r['qa']['program'],'reference_answer':r['qa']['answer'],'steps':steps}
   candidates.append((digest(['duplex-v6-fresh-20260920',r['id']]),row,gold))
  except (ValueError,KeyError,TypeError,ZeroDivisionError,OverflowError) as e:excluded[str(e)[:70]]+=1
 selected=[];labels={};seen=set()
 for _,r,gold in sorted(candidates):
  if r['filing'] in seen:continue
  seen.add(r['filing']);r['split']='development' if len(selected)<8 else 'evaluation'
  selected.append(r);labels[r['id']]=gold
  if len(selected)==32:break
 if len(selected)!=32:raise ValueError(f'only {len(selected)} unique eligible filings')
 manifest={'version':'v6-binding-budget-1','n':32,'arms':ARMS,'caps':CAPS,'shards':16,'hash':digest(selected),'labels_hash':digest(labels),'sources':receipts,'eligible_rows':len(candidates),'eligible_filings':len({r['filing'] for _,r,g in candidates}),'excluded':dict(excluded),'v5_ids':old,'v5_filings':sorted(banned),'split_counts':dict(Counter(r['split'] for r in selected)),'source_overlap':False,'model':MODEL,'revision':REV,'training_steps':0,'gate':'Evaluation binding must beat direct, reasoning and program with paired one-sided p<=0.05/3, binding valid>=0.90, reasoning completion>=0.90; no automatic promotion','limitations':['Filtered pooled FinQA dev/test, not official FinQA evaluation','Existing labels checked numerically, not independently human-adjudicated','Whole company/year filings excluded across all prior v5 cases and new splits','Public source overlap with pretraining unknown','No JevBench fitting or evaluation','Shared enriched bank means changes versus v5 are not attributable solely to the larger token budget']}
 save(root/'requests.json',selected);save(root/'labels.json',labels);save(root/'manifest.json',manifest)
 review=[{'id':r['id'],'filing':r['filing'],'question':r['question'],'evidence':r['evidence'],**labels[r['id']]} for r in selected]
 save(root/'review-before-inference.json',review);emit('prepared',**manifest)

def run(args):
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from huggingface_hub import snapshot_download
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 out=Path('out')/('preflight' if args.preflight else f'shard-{args.shard}');out.mkdir(parents=True,exist_ok=False)
 snap=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','LICENSE*'],max_workers=4)
 hashes={}
 for f in Path(snap).glob('*.safetensors'):
  h=hashlib.sha256()
  with f.open('rb') as s:
   for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
  hashes[f.name]=h.hexdigest()
 if hashes!=EXPECTED:raise ValueError('model digest mismatch')
 tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
 t=time.perf_counter();model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(snap,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
 issues={k:v for k,v in loading.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
 if issues:raise ValueError(str(issues))
 model.eval();save(out/'runtime.json',{'model':MODEL,'revision':REV,'hashes':hashes,'load_seconds':time.perf_counter()-t,'torch':torch.__version__,'source_run':os.environ.get('GITHUB_RUN_ID'),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'caps':CAPS})
 def generate(r,arm):
  started=time.perf_counter();msg=messages(r,arm)
  prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  x=tok(prompt,return_tensors='pt',add_special_tokens=False,truncation=False);n=x['input_ids'].shape[1]
  if n>10000:raise ValueError('overlength input; no truncation')
  with torch.inference_mode():
   y=model.generate(**x,do_sample=False,max_new_tokens=CAPS[arm],use_cache=True,return_dict_in_generate=True,pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  new=y.sequences[0,n:];text=tok.decode(new,skip_special_tokens=True)
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);ended=bool(len(new) and int(new[-1]) in eos)
  rec={'id':r['id'],'arm':arm,'raw':text,'input_hash':digest(inp_only(r)),'prompt_hash':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':n,'output_tokens':len(new),'cap_hit':len(new)>=CAPS[arm] and not ended,'generation_seconds':time.perf_counter()-started}
  parse_start=time.perf_counter()
  try:
   if rec['cap_hit']:raise ValueError('generation_cap_without_EOS')
   v,used,info=parse_response(text,arm,r['bank']);rec.update(status='ok',value=str(v),source_references=used,**info)
  except (ValueError,KeyError,TypeError,SyntaxError,ZeroDivisionError,OverflowError) as e:rec.update(status='invalid',value=None,error=str(e)[:250])
  rec['verification_seconds']=time.perf_counter()-parse_start
  with (out/'records.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
  emit('prediction',id=r['id'],arm=arm,status=rec['status'],output_tokens=len(new),seconds=rec['generation_seconds'])
  return rec
 if args.preflight:
  raw={'pre_text':['A token shop reports the following.'],'table':[['Item','2020','2021'],['Green tokens','80','100'],['White tokens','5','7']],'post_text':[]}
  state=source_state(raw);question='What is the fractional increase in green tokens from 2020 to 2021?'
  r={'id':'owned-preflight','evidence':state,'question':question,'bank':metadata_bank(raw,state,question)}
  records=[generate(r,a) for a in ARMS]
  ok=all(p['status']=='ok' and Fraction(p['value'])==Fraction(1,4) for p in records)
  save(out/'complete.json',{'complete':ok,'owned_input_only':True,'records':4})
  if not ok:raise RuntimeError('Owned contract/answer failure; no evaluation fanout')
  return
 rows=json.loads(Path('data/requests.json').read_text());manifest=json.loads(Path('data/manifest.json').read_text())
 if digest(rows)!=manifest['hash'] or len(rows)!=32:raise ValueError('population mismatch')
 count=0
 for i,r in enumerate(rows):
  if i%16!=args.shard:continue
  base=number_bank(r['evidence'],r['question'])
  if {k:{f:v[f] for f in ('value','source','start','end','text')} for k,v in r['bank'].items()}!=base:raise ValueError('bank source mismatch')
  order=ARMS[i%4:]+ARMS[:i%4]
  for arm in order:generate(r,arm);count+=1
 if count!=8:raise ValueError('incomplete shard')
 save(out/'complete.json',{'complete':True,'shard':args.shard,'records':count,'hash':manifest['hash'],'model_revision':REV})

def sign_p(w,l):return sum(math.comb(w+l,k) for k in range(w,w+l+1))/2**(w+l) if w+l else 1.0
def audit(root):
 root=Path(root);rows=json.loads(Path('data/requests.json').read_text());labels=json.loads(Path('data/labels.json').read_text());manifest=json.loads(Path('data/manifest.json').read_text())
 if digest(rows)!=manifest['hash'] or digest(labels)!=manifest['labels_hash']:raise ValueError('data changed')
 records=[];done=[]
 for p in sorted(root.rglob('complete.json')):
  d=json.loads(p.read_text())
  if 'shard' in d:
   if not d['complete'] or d['hash']!=manifest['hash'] or d['records']!=8:raise ValueError('failed/incomplete shard')
   done.append(d['shard'])
 for p in sorted(root.rglob('records.jsonl')):records += [json.loads(s) for s in p.read_text().splitlines()]
 if sorted(done)!=list(range(16)) or len(records)!=128:raise ValueError('incomplete or duplicate partitions')
 byid={r['id']:r for r in rows};lookup={}
 for rec in records:
  key=rec['id'],rec['arm'];r=byid[rec['id']]
  if key in lookup or rec['input_hash']!=digest(inp_only(r)):raise ValueError('record identity mismatch')
  if rec['status']=='ok':
   v,used,_=parse_response(rec['raw'],rec['arm'],r['bank'])
   if str(v)!=rec['value']:raise ValueError('execution mismatch')
  correct=rec['status']=='ok' and round(float(Fraction(rec['value'])),5)==round(float(Fraction(labels[r['id']]['value'])),5)
  rec['correct']=correct;lookup[key]=rec
 report={}
 for split in ('development','evaluation'):
  rr=[r for r in rows if r['split']==split];report[split]={}
  for arm in ARMS:
   pp=[lookup[r['id'],arm] for r in rr]
   report[split][arm]={'n':len(rr),'correct':sum(p['correct'] for p in pp),'valid':sum(p['status']=='ok' for p in pp),'capped':sum(p['cap_hit'] for p in pp),'median_generation_seconds':statistics.median(p['generation_seconds'] for p in pp),'median_total_seconds':statistics.median(p['generation_seconds']+p['verification_seconds'] for p in pp),'median_output_tokens':statistics.median(p['output_tokens'] for p in pp)}
 comparisons={};rr=[r for r in rows if r['split']=='evaluation']
 for candidate in ('program','binding'):
  for baseline in ARMS:
   if candidate==baseline:continue
   w=sum(lookup[r['id'],candidate]['correct'] and not lookup[r['id'],baseline]['correct'] for r in rr);l=sum(not lookup[r['id'],candidate]['correct'] and lookup[r['id'],baseline]['correct'] for r in rr)
   comparisons[candidate+'-vs-'+baseline]={'repairs':w,'harms':l,'paired_one_sided_p':sign_p(w,l)}
 ev=report['evaluation'];qualified=ev['reasoning']['valid']>=22 and ev['binding']['valid']>=22 and all(comparisons['binding-vs-'+a]['paired_one_sided_p']<=.05/3 for a in ('direct','reasoning','program'))
 result={'complete':True,'n_requests':32,'generations':128,'model':MODEL,'revision':REV,'metrics':report,'comparisons':comparisons,'binding_qualified':qualified,'production_promoted':False,'training_steps':0,'manifest':manifest,'reasoning_completes':ev['reasoning']['valid']>=22,'qualified_scope':'filtered task population only; no model/leaderboard promotion','limitations':manifest['limitations']}
 save(root/'results.json',result);save(root/'records.json',records);emit('audit_complete',**result)

def test():
 raw={'pre_text':['Period ended on 31 December.'],'table':[['Metric','2019','2020'],['Sales','80','100'],['Cost','(25)','30']],'post_text':[]};s=source_state(raw);b=metadata_bank(raw,s,'Find the increase.')
 ids=[k for k,v in b.items() if v['row']=='Sales' and v['column'] in ('2019','2020')];assert len(ids)==2
 start,end=ids;expr=f'({end}-{start})/{start}';v,_,_=parse_response('PROGRAM='+expr,'program',b);assert v==Fraction(1,4)
 obj={'bindings':[{'id':k,**{f:b[k][f] for f in ('line','row','column')},'role':'sales'} for k in ids],'program':expr}
 assert parse_response(json.dumps(obj),'binding',b)[0]==v
 obj['bindings'][0]['column']='2099'
 try:parse_response(json.dumps(obj),'binding',b)
 except ValueError:pass
 else:raise AssertionError('fake source binding accepted')
 assert study.final_number('Work\nFINAL=0.25')==Fraction(1,4)
 assert any(v['value']=='-25' for v in b.values())
 inp={'evidence':s,'question':'increase?','bank':b,'answer':'LEAK','reference_program':'LEAK'}
 assert 'LEAK' not in str(messages(inp,'binding'))
 assert sign_p(10,0)==1/1024
 emit('tests',passed=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['test','prepare','run','audit']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 if a.mode=='test':test()
 elif a.mode=='prepare':prepare('data')
 elif a.mode=='run':run(a)
 else:audit('evidence')
