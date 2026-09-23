"""Direct full-logit implementation; all option logits are retrieved, not truncated.
The previous HTTP post-sampling backend is retained only as failed readiness evidence.
"""
from __future__ import annotations
import atexit,json,math,os,re,select,subprocess,time
from pathlib import Path
import readout
HEADER_REF='b29c606e28a01b1bc8c1351026a0fa6e616bf6c4'

def build(boot):
 root=boot.ROOT;inc=root/'native-headers';inc.mkdir(exist_ok=True);manifest=[]
 todo=[('include/llama.h','llama.h')];seen=set()
 while todo:
  path,name=todo.pop()
  if name in seen:continue
  if not re.fullmatch(r'[a-zA-Z0-9_.-]+\.h',name):raise ValueError('header_path')
  raw=boot.get(f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{HEADER_REF}/{path}');(inc/name).write_bytes(raw);seen.add(name)
  manifest.append({'path':path,'sha256':readout.digest(raw)})
  for child in re.findall(r'^\s*#\s*include\s*"([^"]+)"',raw.decode(),re.M):
   if child not in seen:todo.append(('ggml/include/'+child,child))
 libs=list((root/'binary').rglob('libllama.so'))
 if len(libs)!=1:raise ValueError('libllama_location')
 libdir=libs[0].parent;exe=root/'native-logits';cpp=Path(__file__).with_name('raw_logits.cpp')
 command=['g++','-std=c++17','-O2','-DLLAMA_SHARED','-I'+str(inc),str(cpp),'-L'+str(libdir),'-Wl,-rpath,'+str(libdir),'-lllama','-lggml','-lggml-base','-pthread','-o',str(exe)]
 # libllama is linked as -lllama: linker prefix lib + name llama.
 p=subprocess.run(command,capture_output=True,text=True,timeout=90)
 if p.returncode:raise RuntimeError('native_compile: '+p.stderr[-6000:])
 paths=sorted({str(p.parent) for p in (root/'binary').rglob('*.so*')})
 env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','KEY'))};env['LD_LIBRARY_PATH']=':'.join(paths)
 os.environ['ADAPTIVE_RAW_EXE']=str(exe);os.environ['ADAPTIVE_RAW_MODEL']=str(root/'model.gguf');os.environ['ADAPTIVE_RAW_LIBDIR']=str(libdir);os.environ['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH']
 meta={'header_commit':HEADER_REF,'headers':manifest,'cpp_sha256':readout.digest(cpp.read_bytes()),'binary_sha256':readout.digest(exe.read_bytes()),'libllama_sha256':readout.digest(libs[0].read_bytes()),'command':command,'compiler':subprocess.run(['g++','--version'],capture_output=True,text=True).stdout,'model_instances':'One formatting/generative server and one native scorer; weights share mmap pages where OS permits. Not a memory-optimal deployment.','generated_native_tokens':0,'prompt_memory_cleared_each_call':True}
 (root/'native-build.json').write_text(json.dumps(meta,indent=2));return meta

class NativeProcess:
 def __init__(self):
  exe=os.environ.get('ADAPTIVE_RAW_EXE');model=os.environ.get('ADAPTIVE_RAW_MODEL');libdir=os.environ.get('ADAPTIVE_RAW_LIBDIR')
  if not all((exe,model,libdir)):raise ValueError('explicit_native_runtime_required')
  self.log=open('/tmp/native-scorer-'+str(os.getpid())+'.log','w')
  self.p=subprocess.Popen([exe,model,libdir],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
  atexit.register(self.close)
  self.ready=self.receive(100)
  if not self.ready.get('ready'):raise ValueError('native_startup')
 def receive(self,timeout):
  ready,_,_=select.select([self.p.stdout],[],[],timeout)
  if not ready:self.close();raise TimeoutError('native_timeout')
  line=self.p.stdout.readline()
  if not line:raise RuntimeError('native_process_ended')
  return readout.loads(line)
 def evaluate(self,prompt,markers):
  if self.p.poll() is not None:raise RuntimeError('native_process_unavailable')
  if not isinstance(prompt,str) or len(prompt.encode())>readout.MAX_BYTES or not 2<=len(markers)<=26 or any(m not in readout.SYMBOLS or len(m)!=1 for m in markers) or len(set(markers))!=len(markers):raise ValueError('native_input')
  self.p.stdin.write(prompt.encode().hex()+'\t'+''.join(markers)+'\n');self.p.stdin.flush()
  out=self.receive(180)
  if not out.get('ok'):raise ValueError('native:'+str(out.get('error')))
  return out
 def close(self):
  if hasattr(self,'p') and self.p.poll() is None:
   self.p.terminate()
   try:self.p.wait(timeout=5)
   except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
  if hasattr(self,'log') and not self.log.closed:self.log.close()

class RawBackend(readout.Backend):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.native_process=None
 def close(self):
  if self.native_process:self.native_process.close();self.native_process=None
 def native(self,task,mode='input'):
  start=time.perf_counter();self.calls=[];msg,rows=readout.messages(task,mode)
  prompt=self.post('/apply-template',{'messages':msg,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt']
  if self.native_process is None:self.native_process=NativeProcess()
  request={'prompt':prompt,'markers':[r['marker'] for r in rows]};t=time.perf_counter();out=self.native_process.evaluate(prompt,request['markers'])
  self.calls.append({'path':'native-c-api','request':request,'request_sha256':readout.digest(readout.canonical(request)),'response':out,'seconds':time.perf_counter()-t})
  zs=out.get('logits')
  if not isinstance(zs,list) or len(zs)!=len(rows) or [x['marker'] for x in zs]!=request['markers']:raise ValueError('native_logit_shape')
  values=[x['logit'] for x in zs]
  if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):raise ValueError('native_logit_value')
  high=max(values);exps=[math.exp(x-high) for x in values];denom=math.fsum(exps)
  p={r['label']:v/denom for r,v in zip(rows,exps)}
  return {'ok':True,'probs':readout.distribution(p,list(p)),'source':'native_full_logit_subset_softmax','calibrated':False,'semantic_verification':False,'option_ledger':rows,'token_ids':{x['marker']:x['id'] for x in zs},'prompt_sha256':readout.digest(prompt),'calls':self.calls,'seconds':time.perf_counter()-start,'timings':{'prompt_tokens':out['prompt_tokens'],'generated_tokens':out['generated_tokens'],'native_seconds':out['seconds']},'native_ready':self.native_process.ready}
