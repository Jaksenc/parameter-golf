"""Disjoint follow-up; original scored core and primary protocol stay unchanged."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import bootstrap as boot
import preflight_entry
from benchmark import PIN,GSM,SALT,score,sha
from prepare_core import unpack
PROMPT_SHA='d70e42e1309ab259d89191004e35cf10268e95de83b49559bb287eca3664f600'
def cohort():
    raw=boot.get(f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/grade_school_math/data/test.jsonl')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()];assert len(rows)==1319
    excluded={i for i,r in sorted(enumerate(rows),key=lambda p:sha((SALT+p[1]['question']).encode()))[:32]}
    remaining=[(i,r) for i,r in enumerate(rows) if i not in excluded]
    chosen=sorted(remaining,key=lambda p:sha(('adaptive-language-v07-grounding\0'+p[1]['question']).encode()))[:16]
    return [{'id':f'gsm8k:{i}','question':r['question'],'reference':r['answer'].split('####')[-1].strip()} for i,r in chosen],sorted(excluded),sha(raw)
def worker(inp,outp):
    root,_=unpack();sys.path.insert(0,str(root));from adaptive_language.engine import LocalChat,solve
    prompt=Path(__file__).with_name('grounded_prompt.txt').read_text();assert sha(prompt.encode())==PROMPT_SHA
    local=LocalChat();rows=[]
    def grounded(messages,max_tokens,json_mode=False):
        messages=[dict(m) for m in messages];messages[0]['content']=prompt
        return local(messages,max_tokens,json_mode)
    for task in json.loads(Path(inp).read_text()):
        assert set(task)=={'id','question'}
        arms=['reasoning','program','grounded_program'];k=int(sha(task['id'].encode())[:8],16)%3;arms=arms[k:]+arms[:k]
        for arm in arms:
            start=time.perf_counter()
            try:r=solve(task['question'],'program' if arm=='grounded_program' else arm,grounded if arm=='grounded_program' else local)
            except Exception as e:r={'answer':None,'status':'unexpected_error','errors':[f'{type(e).__name__}: {e}'],'calls':[],'usage':{},'seconds':time.perf_counter()-start}
            r.update(arm=arm,id=task['id']);rows.append(r);Path(outp).write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
            print('FOLLOWUP_ATTEMPT '+json.dumps({k:r.get(k) for k in ['id','arm','status','usage','seconds']}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.worker:return worker(a.input,a.output)
    proc=log=None;result={'status':'failed','shard':a.shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'prompt_sha256':PROMPT_SHA};start=time.perf_counter()
    try:
        root,core=unpack();result['core']=core
        assert sha(Path(__file__).with_name('grounded_prompt.txt').read_bytes())==PROMPT_SHA
        allrows,excluded,data_sha=cohort();tasks=[t for i,t in enumerate(allrows) if i%4==a.shard];result.update(excluded_primary_ids=excluded,selected_ids=[t['id'] for t in allrows],data_sha256=data_sha)
        temp=Path('/tmp/adaptive-grounding');temp.mkdir(exist_ok=True);inp=temp/'questions.json';out=temp/'outputs.json';inp.write_text(json.dumps([{k:r[k] for k in ['id','question']} for r in tasks]))
        proc,log,info=boot.setup(PIN);result['runtime']=info
        env={k:v for k,v in os.environ.items() if 'TOKEN' not in k.upper() and 'SECRET' not in k.upper() and 'KEY' not in k.upper()}
        try:child=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=15*60);exitcode=child.returncode
        except subprocess.TimeoutExpired:exitcode='timeout'
        rows=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in rows};lookup={r['id']:r for r in tasks}
        for t in tasks:
            for arm in ['reasoning','program','grounded_program']:
                if (t['id'],arm) not in seen:rows.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'usage':{},'seconds':None})
        for r in rows:
            t=lookup[r['id']];r.update(question=t['question'],reference=t['reference'],suite='GSM8K follow-up',correct=score(r['answer'],t['reference'],'GSM8K'))
        result.update(status='completed',worker_exit=exitcode,rows=rows)
        result['summary']={arm:{'correct':sum(r['correct'] for r in rows if r['arm']==arm),'n':sum(r['arm']==arm for r in rows),'answered':sum(r['answer'] is not None for r in rows if r['arm']==arm),'seconds':sum(r.get('seconds') or 0 for r in rows if r['arm']==arm)} for arm in ['reasoning','program','grounded_program']}
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        if proc:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if log:log.close()
        result['wall_seconds']=time.perf_counter()-start
        result['protocol']={'phase':'disjoint follow-up after one observed primary semantic error','primary_protocol_unchanged':True,'memory':False,'training':False,'native_thinking':False,'generation_budget':{'reasoning':768,'program':[512,256],'grounded_program':[512,256]},'feedback':'mechanical interpreter errors only','frontier_reference':False,'labels_solver_visible':False}
        print('FOLLOWUP_SUMMARY '+json.dumps({k:v for k,v in result.items() if k not in ['rows','excluded_primary_ids','selected_ids']}),flush=True)
        print(boot.publish(f'grounding-shard-{a.shard:02d}.json',result),flush=True)
    if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
