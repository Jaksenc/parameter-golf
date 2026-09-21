"""Owned-preflight repair before target inference; no output-based task selection.
The original direct arm generated explanation and hit64tokens on the owned input.
Prefill only its required FINAL= prefix. Other arms, data and scoring are unchanged.
"""
import hashlib,sys
from pathlib import Path
source=Path(__file__).with_name('binding_v6.py').read_text()
expected='7c2a2d7014c9d68937dc4f6a7bb90cdf33a79da981039c861ac2837698645280'
if hashlib.sha256(source.encode()).hexdigest()!=expected:raise RuntimeError('Unexpected frozen v6 source')
a="  prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)\n"
b="  new=y.sequences[0,n:];text=tok.decode(new,skip_special_tokens=True)\n"
if source.count(a)!=1 or source.count(b)!=1:raise RuntimeError('Unexpected generation layout')
source=source.replace(a,a+"  if arm=='direct':prompt+='FINAL='\n")
source=source.replace(b,b+"  if arm=='direct':text='FINAL='+text\n")
# Only the changed direct contract is repeated on the owned fixture. The other
# three actual preflight responses already passed under the identical model.
if '--preflight' in sys.argv:
 source=source.replace('records=[generate(r,a) for a in ARMS]',"records=[generate(r,a) for a in ('direct',)]")
 source=source.replace("'records':4}","'records':len(records)}")
print('{"kind":"owned_contract_repair","direct_prefix":"FINAL=","other_prompts_unchanged":true,"targets_unchanged":true}',flush=True)
exec(compile(source,'binding_v6.py','exec'),{'__name__':'__main__','__file__':str(Path(__file__).with_name('binding_v6.py'))})
