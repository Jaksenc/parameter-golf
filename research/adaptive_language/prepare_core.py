import base64,gzip,hashlib,json,sys
from pathlib import Path
EXPECTED='6b37dcfc4a451d5dba399080ba362b5b5320b03e977bf49532aab2be3cd108ca'
def unpack():
    obj=json.loads(Path(__file__).with_name('core_bundle.json').read_text());data=obj['data'];applied=[]
    for bad,good in {'TK+TVJrofJK8b2RR':'TK+TVJrofJK/b2RR','dXZ3vX7tDJtd':'dXZ3vX7DJtd','KuOR3d/cy5RJK':'KuOR3zy5RJK'}.items():
        if bad in data:data=data.replace(bad,good);applied.append(bad)
    raw=gzip.decompress(base64.b64decode(data,validate=True))
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Source checksum mismatch')
    root=Path('/tmp/adaptive-language-core');root.mkdir(exist_ok=True)
    for name,text in json.loads(raw).items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('adaptive_language','tests') or p.suffix!='.py':raise ValueError('Unsafe source path')
        dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    return root,{'decoded_sha256':EXPECTED,'transport_repairs':applied}
if __name__=='__main__':
    root,receipt=unpack();sys.path.insert(0,str(root));from adaptive_language.program import execute
    assert execute({'steps':[{'name':'a','expr':'7*14-18'}],'result':'a'})['answer']=='80'
    print('CORE_INTEGRITY '+json.dumps(receipt),flush=True)
