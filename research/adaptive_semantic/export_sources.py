"""Download public immutable evaluation evidence only; never load a model."""
import concurrent.futures,hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path('/tmp/semantic-source-evidence')
REF='6189083af1cb093ee8029e3fd8215e85ad85fd62'
GSM='3101c7d5072418e28b9008a6636bde82a006892c'
BBH='9ee07bd481feebf959a6b59d61ea57bdcf30964d'
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'adaptive-evaluation-source-audit'})
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()
def main():
    ROOT.mkdir(exist_ok=True)
    tree=json.loads(get(f'https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/git/trees/{BBH}?recursive=1'))
    assert not tree.get('truncated')
    import re
    paths=sorted(e['path'] for e in tree['tree'] if re.fullmatch(r'bbh/[^/]+\.json',e['path']))
    assert len(paths)==27
    urls={
      'datasets/gsm8k-test.jsonl':f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/grade_school_math/data/test.jsonl',
      'sources/GSM8K-LICENSE.txt':f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/LICENSE',
      'sources/BBH-LICENSE.txt':f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH}/LICENSE'
    }
    for p in paths:urls['datasets/'+p]=f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH}/{p}'
    for p in ['research/adaptive_semantic/contracts.py','research/adaptive_semantic/engine.py','research/adaptive_semantic/run.py','.github/workflows/adaptive-semantic-v08.yml']:
        urls['frozen/'+p]=f'https://raw.githubusercontent.com/Jaksenc/parameter-golf/{REF}/{p}'
    for p in ['bootstrap.py','preflight_entry.py','prepare_core.py','core_bundle.json','benchmark.py']:
        urls['frozen/research/adaptive_language/'+p]=f'https://raw.githubusercontent.com/Jaksenc/parameter-golf/{REF}/research/adaptive_language/{p}'
    def fetch(item):
        p,url=item;raw=get(url)
        if len(raw)>8_000_000:raise ValueError('Unexpected source size')
        target=ROOT/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        return {'path':p,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,urls.items()))
    (ROOT/'MANIFEST.json').write_text(json.dumps({'source_commit':REF,'gsm_commit':GSM,'bbh_commit':BBH,'files':records,'inference':False},indent=2))
    with zipfile.ZipFile('semantic-source-evidence.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    raw=Path('semantic-source-evidence.zip').read_bytes()
    print(json.dumps({'files':len(records),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'no_inference':True}))
if __name__=='__main__':main()
