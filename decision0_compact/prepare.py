"""Reconstruct the exact previously frozen compact-evidence source bundle."""
from pathlib import Path
import base64, hashlib, json, zlib
root=Path(__file__).resolve().parent
parts=sorted(root.glob('source.part*'))
if len(parts)!=2: raise RuntimeError('Exactly two frozen source parts required')
raw=base64.b64decode(''.join(p.read_text().strip() for p in parts), validate=True)
if hashlib.sha256(raw).hexdigest()!='f933c963cefd66250276ffce7f3b0b1f615ae4687ba710c72d270abab4412ce4': raise RuntimeError('Frozen source bundle hash mismatch')
files=json.loads(zlib.decompress(raw))
for name,text in files.items():
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()) or not isinstance(text,str): raise RuntimeError('Invalid source member')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(text)
print('Verified and reconstructed',len(files),'exact source files; no model or data changes')
