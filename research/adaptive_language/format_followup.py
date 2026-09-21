"""Predeclared format diagnostic, not a new model or post-hoc rescoring."""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import bootstrap as boot
import preflight_entry
from benchmark import PIN,GSM,score,sha
from grounding_followup import cohort as earlier_cohort
from prepare_core import unpack
PROMPT="Solve the task. Return only the requested answer, without explanation. End with FINAL: <answer>. Match the answer type requested by the question; do not invent answer options."
SALT='adaptive-language-v07-format-repair\0'
ARMS=('legacy_direct','format_direct','reasoning')
def cohort():
    old_follow,old_primary,data_sha=earlier_cohort()
    excluded={f'gsm8k:{i}' for i in old_primary}|{r['id'] for r in old_follow}
    raw=boot.get(f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM}/grade_school_math/data/test.jsonl')
    assert sha(raw)==data_sha
    data=[json.loads(line) for line in raw.splitlines() if line.strip()]
    available=[(i,r) for i,r in enumerate(data) if f'gsm8k:{i}' not in excluded]
    chosen=sorted(available,key=lambda p:sha((SALT+p[1]['question']).encode()))[:16]
    tasks=[{'id':f'gsm8k:{i}','question':r['question'],'reference':r['answer'].split('####')[-1].strip()} for i,r in chosen]
    assert len(excluded)==48 and len(tasks)==16 and not excluded.intersection(r['id'] for r in tasks)
    return tasks,sorted(excluded),data_sha

def worker(inp,out):
    root,_=unpack();sys.path.insert(0,str(root));from adaptive_language.engine import LocalChat,solve
    local=LocalChat();rows=[]
    def formatted(messages,max_tokens,json_mode=False):
        copied=[dict(m) for m in messages];copied[0]['content']=PROMPT
        return local(copied,max_tokens,json_mode)
    for task in json.loads(Path(inp).read_text()):
        assert set(task)=={'id','question'}
        k=int(sha(task['id'].encode())[:8],16)%3;arms=ARMS[k:]+ARMS[:k]
        for arm in arms:
            started=time.perf_counter()
            try:r=solve(task['question'],'reasoning' if arm=='reasoning' else 'direct',formatted if arm=='format_direct' else local)
            except Exception as e:r={'answer':None,'status':'unexpected_error','calls':[],'usage':{},'seconds':time.perf_counter()-started,'errors':[f'{type(e).__name__}: {e}']}
            r.update(id=task['id'],arm=arm);rows.append(r)
            Path(out).write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
            print('FORMAT_ATTEMPT '+json.dumps({k:r.get(k) for k in ('id','arm','status','seconds','usage')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--input');p.add_argument('--output');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.worker:return worker(a.input,a.output)
    assert 0<=a.shard<4
    proc=log=None;start=time.perf_counter()
    result={'status':'failed','shard':a.shard,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'prompt':PROMPT,'prompt_sha256':sha(PROMPT.encode()),'runner_sha256':sha(Path(__file__).read_bytes())}
    try:
        root,core=unpack();result['core']=core
        all_tasks,excluded,data_sha=cohort();tasks=[r for i,r in enumerate(all_tasks) if i%4==a.shard]
        result.update(excluded_ids=excluded,selected_ids=[r['id'] for r in all_tasks],data_sha256=data_sha)
        temp=Path('/tmp/adaptive-format');temp.mkdir(exist_ok=True);inp=temp/'questions.json';out=temp/'predictions.json'
        inp.write_text(json.dumps([{k:r[k] for k in ('id','question')} for r in tasks]))
        lock={'source_sha256':result['runner_sha256'],'prompt_sha256':result['prompt_sha256'],'ids':result['selected_ids'],'data_sha256':data_sha,'model_pin':PIN,'arms':ARMS,'max_tokens':768,'temperature':0,'native_thinking':False,'selection_salt':SALT,'no_memory':True}
        result['protocol_lock']=lock
        print('PROTOCOL_LOCK '+json.dumps(lock),flush=True)
        proc,log,info=boot.setup(PIN);result['runtime']=info
        env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('TOKEN','SECRET','KEY'))}
        try:exitcode=subprocess.run([sys.executable,__file__,'--worker','--input',str(inp),'--output',str(out)],env=env,timeout=10*60).returncode
        except subprocess.TimeoutExpired:exitcode='worker_timeout'
        rows=json.loads(out.read_text()) if out.exists() else [];seen={(r['id'],r['arm']) for r in rows}
        for t in tasks:
            for arm in ARMS:
                if (t['id'],arm) not in seen:rows.append({'id':t['id'],'arm':arm,'answer':None,'status':'not_completed','calls':[],'usage':{},'seconds':None})
        lookup={r['id']:r for r in tasks}
        for r in rows:
            t=lookup[r['id']];r.update(question=t['question'],reference=t['reference'],suite='GSM8K format diagnostic',correct=score(r['answer'],t['reference'],'GSM8K'))
        result.update(status='completed',worker_exit=exitcode,rows=rows)
        result['summary']={arm:{'correct':sum(r['correct'] for r in rows if r['arm']==arm),'n':sum(r['arm']==arm for r in rows),'answered':sum(r['answer'] is not None for r in rows if r['arm']==arm),'letter_answers':sum(bool(re.fullmatch(r'\(?[A-Za-z]\)?',str(r['answer']))) for r in rows if r['arm']==arm),'seconds':sum(r.get('seconds') or 0 for r in rows if r['arm']==arm)} for arm in ARMS}
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    finally:
        if proc:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if log:log.close()
        result['wall_seconds']=time.perf_counter()-start
        Path(f'format-shard-{a.shard:02d}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        print('FORMAT_SUMMARY '+json.dumps({k:v for k,v in result.items() if k not in ('rows','runtime','excluded_ids')}),flush=True)
    if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
