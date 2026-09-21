"""Pinned BBEH pilot: frozen pretrained baseline, not the full Adaptive system."""
from __future__ import annotations
import argparse, base64, collections, hashlib, importlib.util, json, os
from pathlib import Path
import platform, subprocess, sys, time, urllib.request
BBEH_REF='80d12ca916b7158f22293fcf3144f4d3d854d4be'
DATA_BLOB='9a1b31285bcdd4194dd6b88cf48cdd445d3cbb59'
SCORER_BLOB='9e0550fd588fd9846265fcb0d48d89a815ff1c75'
MODEL='Qwen/Qwen3.5-4B'
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BRANCH='research/adaptive-external-benchmarks-20260920'
def sha(data): return hashlib.sha256(data).hexdigest()
def download(path,blob):
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/google-deepmind/bbeh/{BBEH_REF}/{path}',timeout=60).read()
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob: raise ValueError('Upstream integrity failure')
    return raw
def cohort(raw,n):
    obj=json.loads(raw); rows=obj if isinstance(obj,list) else obj.get('examples',obj.get('data'))
    if not isinstance(rows,list) or len(rows)!=460: raise ValueError('Expected 460 Mini examples')
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        task=r.get('task',r.get('task_name',r.get('task_type')));q=r.get('input',r.get('question'));ans=r.get('target',r.get('answer'))
        if not all(isinstance(v,str) for v in (task,q,ans)): raise ValueError(f'Unexpected fields: {list(r)}')
        groups[task].append({'id':i,'task':task,'question':q,'answer':ans})
    if len(groups)!=23: raise ValueError('Expected 23 task families')
    chosen=[]
    for task in sorted(groups):
        chosen+=sorted(groups[task],key=lambda r:sha(('adaptive-bbeh-pilot-v1\0'+r['question']).encode()))[:n]
    return sorted(chosen,key=lambda r:(r['task'],r['id']))
def worker(inp,outp):
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForImageTextToText,StoppingCriteria,StoppingCriteriaList
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(17)
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False).eval()
    manifest={'model':MODEL,'revision':REVISION,'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'machine':platform.machine(),'platform':platform.platform(),'threads':4,'dtype':'bfloat16','attention':'sdpa','parameters':sum(p.numel() for p in model.parameters()),'load_seconds':time.perf_counter()-start,'seed':17,'sampling':False,'thinking_template':False,'max_input_tokens':8192,'max_output_tokens_per_arm':512,'generation_seconds_soft_limit':90,'tools':False,'persistent_memory':False,'trained_neural_guide':False,'new_training':False,'claim':'direct vs bounded explicit reasoning baseline, not complete Adaptive architecture'}
    out=[];Path(outp).write_text(json.dumps({'manifest':manifest,'rows':out},indent=2))
    class WallStop(StoppingCriteria):
        def __init__(self): self.start=time.perf_counter()
        def __call__(self,input_ids,scores,**kwargs): return time.perf_counter()-self.start>=90
    prompts={'direct':'Solve the user task. Give only the final answer in the required format. End with: The final answer is: <answer>','reasoning':'Solve the user task carefully. Reason step by step, checking relevant constraints and exceptions. Reserve room for the answer. End with: The final answer is: <answer>'}
    for q in json.loads(Path(inp).read_text()):
        if set(q)!={'id','task','question'}:raise ValueError('Unexpected solver input; possible label leakage')
        for arm in (['direct','reasoning'] if q['id']%2==0 else ['reasoning','direct']):
            messages=[{'role':'system','content':prompts[arm]},{'role':'user','content':q['question']}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(text,return_tensors='pt',add_special_tokens=False)
            row={'id':q['id'],'task':q['task'],'arm':arm,'question_sha256':sha(q['question'].encode()),'prompt_sha256':sha(text.encode()),'input_tokens':int(enc.input_ids.shape[1])}
            start=time.perf_counter()
            if row['input_tokens']>8192:row.update(status='unsupported_context',response='',output_tokens=0)
            else:
                try:
                    wall=WallStop()
                    with torch.inference_mode():
                        ids=model.generate(**enc,max_new_tokens=512,do_sample=False,use_cache=True,stopping_criteria=StoppingCriteriaList([wall]),pad_token_id=tok.pad_token_id)
                    new=ids[0,enc.input_ids.shape[1]:];eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                    ended=len(new)>0 and int(new[-1]) in eos
                    row.update(status='ok' if ended else 'budget_limited',response=tok.decode(new,skip_special_tokens=True),output_tokens=len(new),eos_reached=ended,seconds_soft_limit_reached=time.perf_counter()-wall.start>=90)
                except Exception as exc:row.update(status='error',response='',output_tokens=None,error=f'{type(exc).__name__}: {exc}')
            row['generation_seconds']=time.perf_counter()-start;out.append(row)
            Path(outp).write_text(json.dumps({'manifest':manifest,'rows':out},indent=2))
            print('PREDICTION '+json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
    return 0
def publish(path,text):
    token=os.environ.get('GITHUB_TOKEN')
    if not token:return {'status':'not_requested'}
    url=f'https://api.github.com/repos/{os.environ["GITHUB_REPOSITORY"]}/contents/{path}'
    payload={'message':'Record bounded external benchmark receipts','content':base64.b64encode(text.encode()).decode(),'branch':BRANCH}
    headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'bounded-benchmark'}
    for attempt in range(8):
        try:
            req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='PUT')
            obj=json.loads(urllib.request.urlopen(req,timeout=45).read());return {'status':'published','commit':obj['commit']['sha'],'path':path}
        except Exception:
            if attempt==7:raise
            time.sleep(1+attempt)
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);p.add_argument('--per-task',type=int,default=2);a=p.parse_args()
    if a.worker:return worker(a.input,a.output)
    start=time.perf_counter();tmp=Path('/tmp/adaptive-bbeh');tmp.mkdir(exist_ok=True)
    raw=download('bbeh/mini/data.json',DATA_BLOB);scorer=download('bbeh/evaluate.py',SCORER_BLOB)
    selected=cohort(raw,a.per_task);rows=[r for i,r in enumerate(selected) if i%a.shards==a.shard]
    print('COHORT '+json.dumps({'total':len(selected),'shard':a.shard,'ids':[r['id'] for r in rows],'data_sha256':sha(raw),'scorer_sha256':sha(scorer)}),flush=True)
    inp=tmp/'questions.json';pred=tmp/'predictions.json';inp.write_text(json.dumps([{k:r[k] for k in ('id','task','question')} for r in rows]))
    env={k:v for k,v in os.environ.items() if k not in ('GITHUB_TOKEN','GH_TOKEN')}
    proc=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(pred)],env=env)
    result=json.loads(pred.read_text()) if pred.exists() else {'manifest':{},'rows':[]}
    sp=tmp/'official_evaluate.py';sp.write_bytes(scorer);spec=importlib.util.spec_from_file_location('official_bbeh',sp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    refs={r['id']:r['answer'] for r in rows};observed={(r['id'],r['arm']) for r in result['rows']}
    for r in rows:
        for arm in ('direct','reasoning'):
            if (r['id'],arm) not in observed:result['rows'].append({'id':r['id'],'task':r['task'],'arm':arm,'status':'not_completed','response':'','output_tokens':None,'generation_seconds':None})
    for r in result['rows']:
        r['reference']=refs[r['id']];r['correct']=bool(r['status'] in ('ok','budget_limited') and mod.evaluate_correctness(r['response'],refs[r['id']]))
    summary={}
    for arm in ('direct','reasoning'):
        rr=[r for r in result['rows'] if r['arm']==arm];summary[arm]={'correct':sum(r['correct'] for r in rr),'total':len(rr),'statuses':dict(collections.Counter(r['status'] for r in rr)),'output_tokens':sum(r.get('output_tokens') or 0 for r in rr),'generation_seconds':sum(r.get('generation_seconds') or 0 for r in rr)}
    result.update(benchmark='BBEH Mini stratified pilot',bbeh_commit=BBEH_REF,data_sha256=sha(raw),scorer_sha256=sha(scorer),shard=a.shard,shards=a.shards,per_task=a.per_task,total_selected=len(selected),selection='per-task ascending SHA256(adaptive-bbeh-pilot-v1 + NUL + original question), first n; no target use',selected_ids=[r['id'] for r in selected],summary=summary,worker_exit=proc.returncode,request_wall_seconds=time.perf_counter()-start,run_id=os.environ.get('GITHUB_RUN_ID'),source_sha256=sha(Path(__file__).read_bytes()),evaluation_isolation='worker receives only id,task,question; no answer feedback; no training or cross-case memory',limitations=['public development pilot, not full benchmark or leaderboard submission','frozen pretrained baseline only; not full Adaptive system','pretraining overlap unknown','greedy bounded decoding; budget failures retained','load/overhead reported separately; energy not measured'])
    text=json.dumps(result,indent=2);Path(f'bbeh-shard-{a.shard:02d}.json').write_text(text)
    receipt=publish(f'research/adaptive_external/results/bbeh-shard-{a.shard:02d}.json',text)
    print('BENCHMARK_SUMMARY '+json.dumps({'shard':a.shard,'summary':summary,'receipt':receipt,'worker_exit':proc.returncode}),flush=True)
    return proc.returncode
if __name__=='__main__':sys.exit(main())
