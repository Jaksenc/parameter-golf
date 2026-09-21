"""Execution-only continuation of four timed-out v6 partitions.
Every previously saved response is preserved, including invalid and capped ones.
No source question, inference prompt, model, cap, or scoring rule changes.
"""
import ast,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
base=(root/'binding_v6.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='7c2a2d7014c9d68937dc4f6a7bb90cdf33a79da981039c861ac2837698645280'
a="  prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)\n"
b="  new=y.sequences[0,n:];text=tok.decode(new,skip_special_tokens=True)\n"
assert base.count(a)==base.count(b)==1
source=base.replace(a,a+"  if arm=='direct':prompt+='FINAL='\n").replace(b,b+"  if arm=='direct':text='FINAL='+text\n")
original=source
records=json.loads(Path('resume-existing.json').read_text());keys={(r['id'],r['arm']) for r in records}
if len(keys)!=len(records):raise RuntimeError('duplicate previous records')
rows=json.loads(Path('data/requests.json').read_text());lookup={r['id']:r for r in rows}
for record in records:
 r=lookup[record['id']];obj={k:r[k] for k in ('evidence','question','bank')}
 expected=hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 if record['input_hash']!=expected or record['arm'] not in ('direct','reasoning','program','binding'):raise RuntimeError('changed original input')
needle='  for arm in order:generate(r,arm);count+=1\n'
assert source.count(needle)==1
source=source.replace(needle,"  for arm in order:\n   if (r['id'],arm) not in EXISTING_KEYS:generate(r,arm)\n   count+=1\n")
def gen_ast(s):return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='generate']
assert gen_ast(original)==gen_ast(source)
print(json.dumps({'kind':'execution_only_resume','existing_records':len(records),'generation_function_unchanged':True,'invalid_record_retries':False,'original_run':35553557091}),flush=True)
exec(compile(source,'binding_v6.py','exec'),{'__name__':'__main__','__file__':str(root/'binding_v6.py'),'EXISTING_KEYS':keys})
