"""Public-only CPU inference bootstrap. No hosted inference, caches, or private input."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,platform,subprocess,tarfile,time,urllib.request,zipfile
from pathlib import Path
TAG='b10964'
REPO='unsloth/Qwen3.5-4B-GGUF'
BRANCH='research/adaptive-language-v07-20260921'
ROOT=Path('/tmp/adaptive-language-runtime')
def digest(b):return hashlib.sha256(b).hexdigest()
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'adaptive-language-research/0.7'})
    with urllib.request.urlopen(req,timeout=90) as r:return r.read()
def json_get(url):return json.loads(get(url))
def fetch_file(url,path,expected=None):
    h=hashlib.sha256();n=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'adaptive-language-research/0.7'}),timeout=90) as r,path.open('wb') as f:
        while True:
            b=r.read(4*1024*1024)
            if not b:break
            f.write(b);h.update(b);n+=len(b)
            if n>5_000_000_000:raise ValueError('Download exceeds public experiment size limit')
    if expected and h.hexdigest()!=expected:raise ValueError('Downloaded checksum does not match publisher')
    return {'url':url,'sha256':h.hexdigest(),'bytes':n}
def publish(name,obj):
    text=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)
    Path(name).write_text(text)
    token=os.environ.get('GITHUB_TOKEN')
    if not token:return {'local':name}
    url=f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}/contents/research/adaptive_language/results/{name}'
    payload={'message':'Record bounded public language experiment','branch':BRANCH,'content':base64.b64encode(text.encode()).decode()}
    for k in range(8):
        try:
            req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','User-Agent':'adaptive-language-research'},method='PUT')
            result=json.loads(urllib.request.urlopen(req,timeout=40).read());return {'path':name,'commit':result['commit']['sha']}
        except Exception:
            if k==7:raise
            time.sleep(k+1)
def setup(pin=None):
    ROOT.mkdir(exist_ok=True)
    started=time.perf_counter()
    release=json_get(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}')
    candidates=[a for a in release['assets'] if 'ubuntu' in a['name'].lower() and 'x64' in a['name'].lower() and not any(x in a['name'].lower() for x in ['vulkan','sycl','cuda','rocm','cann']) and a['name'].endswith(('.tar.gz','.zip','.tar.xz'))]
    if len(candidates)!=1:raise ValueError('Ambiguous CPU binary '+str([a['name'] for a in release['assets']]))
    asset=candidates[0];expected=(asset.get('digest') or '').removeprefix('sha256:') or None
    runtime=fetch_file(asset['browser_download_url'],ROOT/asset['name'],expected)
    if pin and runtime['sha256']!=pin['runtime']['sha256']:raise ValueError('Runtime changed')
    dest=ROOT/'binary';dest.mkdir(exist_ok=True)
    archive=ROOT/asset['name']
    if archive.suffix=='.zip':
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    binaries=list(dest.rglob('llama-server'))
    if len(binaries)!=1:raise ValueError('Expected one server binary')
    server=binaries[0];server.chmod(0o755)
    model_ref=pin['model_revision'] if pin else 'main'
    meta=json_get(f'https://huggingface.co/api/models/{REPO}/revision/{model_ref}?blobs=true')
    revision=meta['sha'];files=[x for x in meta['siblings'] if x['rfilename'].lower().endswith('q4_k_m.gguf') and 'mmproj' not in x['rfilename'].lower()]
    if len(files)!=1:raise ValueError('Expected one Q4_K_M model '+str([x['rfilename'] for x in meta['siblings']]))
    file=files[0];lfs=file.get('lfs',{});expected=lfs.get('sha256') or lfs.get('oid','').removeprefix('sha256:') or None
    weights=fetch_file(f'https://huggingface.co/{REPO}/resolve/{revision}/{file["rfilename"]}',ROOT/'model.gguf',expected)
    if pin and weights['sha256']!=pin['weights']['sha256']:raise ValueError('Weights changed')
    env={k:v for k,v in os.environ.items() if 'TOKEN' not in k.upper() and 'SECRET' not in k.upper() and 'KEY' not in k.upper()}
    env['LD_LIBRARY_PATH']=':'.join(sorted({str(x.parent) for x in dest.rglob('*.so*')}))
    cmd=[str(server),'-m',str(ROOT/'model.gguf'),'--host','127.0.0.1','--port','8787','-c','8192','-np','1','-t','4','-tb','4','-ngl','0','--jinja','--chat-template-kwargs','{"enable_thinking":false}','--no-webui']
    log=(ROOT/'server.log').open('w');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    for i in range(180):
        if proc.poll() is not None:raise RuntimeError('Server failed: '+(ROOT/'server.log').read_text()[-6000:])
        try:
            if json_get('http://127.0.0.1:8787/health').get('status')=='ok':break
        except Exception:pass
        time.sleep(1)
    else:proc.terminate();raise TimeoutError('Server startup')
    info={'model':REPO,'model_revision':revision,'weights':weights,'runtime':runtime,'tag':TAG,'command':cmd,'server_version':subprocess.run([str(server),'--version'],env=env,capture_output=True,text=True).stdout,'machine':platform.machine(),'platform':platform.platform(),'python':platform.python_version(),'setup_seconds':time.perf_counter()-started,'private_inputs':False,'paid_inference':False,'training':False}
    return proc,log,info

def chat(messages,tokens=768,json_mode=False):
    payload={'model':'local','messages':messages,'max_tokens':tokens,'temperature':0,'seed':1707,'stream':False,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
    if json_mode:payload['response_format']={'type':'json_object'}
    raw=json.dumps(payload).encode();t=time.perf_counter()
    req=urllib.request.Request('http://127.0.0.1:8787/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=240) as r:obj=json.loads(r.read())
    c=obj['choices'][0]
    return {'text':c['message'].get('content') or '', 'reasoning_content':c['message'].get('reasoning_content') or '', 'finish_reason':c['finish_reason'],'usage':obj.get('usage',{}),'timings':obj.get('timings',{}),'seconds':time.perf_counter()-t,'request_sha256':digest(raw)}
def main():
    proc=log=None;result={'status':'failed','run_id':os.environ.get('GITHUB_RUN_ID')}
    try:
        proc,log,info=setup();result['runtime']=info;rows=[]
        checks=[('Reply only with the integer answer.','There are 7 trays with 14 cups each. Eighteen cups are removed. How many remain?',128,False),('Reason carefully in at most 120 words, then end with FINAL: <number>.','A machine produces 18 parts each minute for 7 minutes, then loses 9 defective parts. Five identical shifts run. How many usable parts are produced?',384,False),('Return JSON only: {"expression":"an arithmetic expression"}. Do not evaluate it.','Nine bundles contain 17 items each; remove 23, then distribute the remainder equally to 5 people. How many items per person?',256,True)]
        for system,q,n,j in checks:
            row=chat([{'role':'system','content':system},{'role':'user','content':q}],n,j);rows.append(row);print('PREFLIGHT_ROW '+json.dumps(row),flush=True)
        result['rows']=rows;result['status']='completed';result['all_finished']=all(r['finish_reason']=='stop' for r in rows)
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        if proc:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if log:log.close()
        print('PREFLIGHT_RESULT '+json.dumps(result),flush=True)
        print(publish('preflight.json',result),flush=True)
    if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
