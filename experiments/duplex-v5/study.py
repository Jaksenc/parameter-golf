"""Duplex v5: qualify a source-bound arithmetic teacher before any training.
No Jev calls, model training, arbitrary-code execution, or benchmark-label routing.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, time
from collections import Counter
from fractions import Fraction
from pathlib import Path

FINQA_REV='0f16e2867befa6840783e58be38c9efb9229d742'
FINQA_BLOBS={'dev':'970f239f591c782b17df420c0d343b780737f3da','test':'59958c7c3bb3b21f4dff6bc912a0fe0ae710aee0'}
MODEL='Qwen/Qwen3.5-4B'; MODEL_REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
ARMS=('direct','reasoning','program')
NUM=re.compile(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])')
OPS={'add':lambda a,b:a+b,'subtract':lambda a,b:a-b,'multiply':lambda a,b:a*b,'divide':lambda a,b:a/b}
CAPS={'direct':32,'reasoning':96,'program':96}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def emit(kind,**x):print(json.dumps({'kind':kind,**x},allow_nan=False),flush=True)
def parse_number(s):
 s=str(s).strip().replace(',','')
 if s.endswith('%'):return Fraction(s[:-1])/100
 return Fraction(s)
def ref_program(s):
 parts=re.findall(r'(add|subtract|multiply|divide)\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)',s)
 residue=re.sub(r'(add|subtract|multiply|divide)\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)','',s)
 if not 1<=len(parts)<=3 or residue.replace(',','').strip():raise ValueError('unsupported_reference_program')
 values=[]
 def arg(v):
  v=v.strip()
  if re.fullmatch(r'#\d+',v):
   j=int(v[1:])
   if j>=len(values):raise ValueError('forward_reference')
   return values[j]
  if v.startswith('const_'):v=v[6:].replace('m1','-1')
  return parse_number(v)
 for op,a,b in parts:values.append(OPS[op](arg(a),arg(b)))
 return values[-1],len(parts)

def source_state(raw):
 # Explicit allowlist. NEVER use gold_inds, model_input, reference programs or answers.
 pre=raw.get('pre_text',[]);post=raw.get('post_text',[]);table=raw['table']
 lines=[f'pre{i}: {text}' for i,text in enumerate(pre)]
 lines += ['table'+str(i)+': '+json.dumps(row,ensure_ascii=False) for i,row in enumerate(table)]
 lines += [f'post{i}: {text}' for i,text in enumerate(post)]
 return '\n'.join(lines)
def number_bank(state,question):
 bank={}
 for source,text in [('evidence',state),('question',question)]:
  for match in NUM.finditer(text):
   token=match.group();v=parse_number(token)
   # Parenthesized accounting numbers denote negative values.
   if match.start()>0 and text[match.start()-1]=='(' and text[match.end():match.end()+1]==')':v=-v
   bank['N'+str(len(bank))]={'value':str(v),'source':source,'start':match.start(),'end':match.end(),'text':token}
 return bank

def program_value(output,bank):
 s=output.strip()
 if s.startswith('```'):
  lines=s.splitlines()
  if len(lines)<3 or lines[-1].strip()!='```':raise ValueError('incomplete_code_fence')
  s='\n'.join(lines[1:-1]).strip()
 if s.startswith('PROGRAM='):s=s[len('PROGRAM='):].strip()
 if len(s)>1024:raise ValueError('program_too_long')
 root=ast.parse(s,mode='eval')
 if len(list(ast.walk(root)))>96:raise ValueError('program_too_complex')
 used=[]
 def visit(n,depth=0):
  if depth>24:raise ValueError('depth_limit')
  if isinstance(n,ast.Name):
   if n.id not in bank:raise ValueError('unknown_quantity_reference')
   used.append(n.id);return Fraction(bank[n.id]['value'])
  if isinstance(n,ast.Constant) and type(n.value) is int and n.value in (0,1,100):return Fraction(n.value)
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
   v=visit(n.operand,depth+1);return -v if isinstance(n.op,ast.USub) else v
  if isinstance(n,ast.BinOp) and type(n.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div):
   a=visit(n.left,depth+1);b=visit(n.right,depth+1)
   v={ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b}[type(n.op)]()
   if v.numerator.bit_length()>4096 or v.denominator.bit_length()>4096:raise ValueError('numeric_bound')
   return v
  raise ValueError('unsupported_program_syntax')
 v=visit(root.body)
 if not used:raise ValueError('program_does_not_reference_evidence')
 return v,sorted(set(used))

def final_number(output):
 matches=re.findall(r'(?:^|\n)\s*FINAL\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?)\s*(?=$|\n)',output)
 if len(matches)!=1:raise ValueError('missing_or_duplicate_FINAL')
 return parse_number(matches[0])

def task_messages(inp,arm):
 assert set(inp)=={'id','family','split','evidence','question','bank'}
 bank='\n'.join(f'{k} = {v["value"]} ({v["source"]} characters {v["start"]}:{v["end"]}, original {v["text"]})' for k,v in inp['bank'].items())
 rule='Use only the supplied document and question. Solve the numerical question. Return ratios and percentage changes as a decimal fraction (0.25 means 25%); keep other quantities in the units requested. '
 if arm=='direct':rule+='Reply only with FINAL=<number>.'
 elif arm=='reasoning':rule+='Use at most two short calculation lines, then a separate line FINAL=<number>. Finish within 96 output tokens.'
 elif arm=='program':rule+='Do not calculate a final numeric answer. Write exactly one line PROGRAM=<arithmetic expression>. Use quantity IDs from the bank, parentheses, +, -, *, / and constants 0, 1, 100 only. No imports, function calls, assignments, comparisons, prose or numeric literals for observed quantities. Example syntax: PROGRAM=(N0-N1)/N1 . Use the IDs appropriate to THIS question. Finish within 96 output tokens.'
 else:raise ValueError('unknown_arm')
 return [{'role':'system','content':rule},{'role':'user','content':'DOCUMENT\n'+inp['evidence']+'\nQUESTION\n'+inp['question']+'\nQUANTITY BANK (automatically extracted, not selected for relevance)\n'+bank}]

def prepare(root):
 import urllib.request
 from datasets import load_dataset
 from huggingface_hub import HfApi
 root=Path(root);root.mkdir(exist_ok=True,parents=True)
 requests=[];labels={};receipts=[];exclusions={};seen_docs=set()
 for split,n in [('dev',8),('test',24)]:
  url=f'https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_REV}/dataset/{split}.json'
  with urllib.request.urlopen(url,timeout=90) as f:blob=f.read()
  sha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
  if sha!=FINQA_BLOBS[split]:raise ValueError('FinQA_asset_changed')
  raw=json.loads(blob);eligible=[];excluded=Counter()
  for r in raw:
   try:
    state=source_state(r);question=r['qa']['question'];bank=number_bank(state,question)
    if len(state)>2500 or len(question)>400 or len(bank)>64:raise ValueError('declared_context_or_bank_budget')
    answer,steps=ref_program(r['qa']['program']);ref=parse_number(r['qa']['exe_ans'])
    if round(float(answer),5)!=round(float(ref),5):raise ValueError('reference_program_answer_mismatch')
    # Exclude label-unit ambiguity before any inference: annotation must agree with executed answer.
    natural=str(r['qa'].get('answer','')).strip().rstrip('.')
    if round(float(parse_number(natural)),4)!=round(float(ref),4):raise ValueError('annotation_unit_disagreement')
    if not bank or abs(answer)>10**10:raise ValueError('invalid_numeric_range')
    doc=r['id'].rsplit('-',1)[0]
    eligible.append((digest({'selection':'duplex-v5-fixed-1','id':r['id']}),r,state,bank,answer,steps,doc))
   except (ValueError,KeyError,TypeError,ZeroDivisionError,OverflowError):excluded['ineligible_or_unchecked']+=1
  selected=[]
  for _,r,state,bank,value,steps,doc in sorted(eligible):
   if doc in seen_docs:continue
   seen_docs.add(doc);rid='finqa:'+r['id'];role='development' if split=='dev' else 'evaluation'
   inp={'id':rid,'family':'finqa','split':role,'evidence':state,'question':r['qa']['question'],'bank':bank}
   selected.append(inp);labels[rid]={'value':str(value),'exe_ans':r['qa']['exe_ans'],'program':r['qa']['program'],'reference_answer':r['qa']['answer'],'document':doc,'steps':steps,'source_split':split}
   if len(selected)==n:break
  if len(selected)!=n:raise ValueError(f'Insufficient eligible FinQA {split}: {len(selected)}')
  requests.extend(selected);receipts.append({'repo':'czyssrs/FinQA','revision':FINQA_REV,'split':split,'git_blob':sha,'total':len(raw),'eligible':len(eligible),'selected':len(selected),'source_sha256':hashlib.sha256(blob).hexdigest()});exclusions[split]=dict(excluded)
 # Initial independent retention bank. Native labels; no category routing or teacher relabeling.
 ret=[];api=HfApi()
 specs=[('stanfordnlp/snli',None,'test','nli'),('google-research-datasets/paws','labeled_final','test','paraphrase'),('google/boolq',None,'validation','qa')]
 for repo,config,heldout,family in specs:
  rev=api.dataset_info(repo).sha
  def transform(raw):
   if family=='nli':
    if raw['label'] not in (0,1,2):return None
    e=raw['premise'];q='Using only the evidence, classify this claim: '+raw['hypothesis'];opts=['Entailed: the evidence establishes the claim.','Unknown: the evidence neither establishes nor contradicts the claim.','Contradicted: the evidence establishes the claim is false.'];target=int(raw['label']);group=digest(e.strip().casefold())
   elif family=='paraphrase':
    e=raw['sentence1'];q='Does this sentence express the same meaning as the evidence? '+raw['sentence2'];opts=['Different meaning.','Same meaning.'];target=int(raw['label']);group=digest(sorted([e.strip().casefold(),raw['sentence2'].strip().casefold()]))
   else:
    e=raw['passage'];q=raw['question'];opts=['No.','Yes.'];target=int(raw['answer']);group=digest(e.strip().casefold())
   if len(e.split())>120 or len(q.split())>100:return None
   return {'id':family+':'+digest([e,q])[:24],'family':family,'state':e,'question':q,'options':opts,'target':target,'group':group}
  forbidden=set()
  for raw in load_dataset(repo,config,split=heldout,revision=rev):
   v=transform(raw)
   if v:forbidden.add(v['group'])
  candidates={}
  for raw in load_dataset(repo,config,split='train',revision=rev):
   v=transform(raw)
   if v and v['group'] not in forbidden:candidates.setdefault(v['group'],v)
  pool=sorted(candidates.values(),key=lambda x:digest(['duplex-v5-retention-1',x['id']]))
  take=pool[:48]
  if len(take)!=48:raise ValueError('Incomplete retention population')
  for i,row in enumerate(take):row['split']='replay' if i<24 else 'sentinel'
  ret.extend(take);receipts.append({'repo':repo,'revision':rev,'split':'train','eligible_groups':len(pool),'excluded_eval_groups':len(forbidden),'selected':48,'replay':24,'sentinel':24})
 save(root/'requests.json',requests);save(root/'labels.json',labels);save(root/'retention.json',ret)
 manifest={'version':'v5-teacher-qualification-1','target_n':32,'retention_n':144,'target_hash':digest(requests),'labels_hash':digest(labels),'retention_hash':digest(ret),'sources':receipts,'exclusions':exclusions,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol':{'arms':ARMS,'output_token_caps':CAPS,'weights':'unchanged','training_steps':0,'generation':'greedy nonthinking mode, one attempt per arm; reasoning explicitly requested in its arm','shards':16,'promotion':'never automatic; compare paired evaluation repairs/harms versus direct and reasoning. Do not tune after seeing scores.'},'limitations':['32 filtered numerical FinQA cases, not official full FinQA or JevBench scores','FinQA reference-program support and annotation consistency used only for eligibility and scoring','All document text/table supplied; no gold support retrieval','Data public and may overlap pretraining; no independent human adjudication performed','Source bank validates literal provenance and arithmetic, not semantic interpretation','Retention bank is an initial NLI/paraphrase/QA sample, not comprehensive capability coverage','Model-generated outputs are not calibrated probability distributions']}
 save(root/'manifest.json',manifest);emit('prepared',**manifest)
