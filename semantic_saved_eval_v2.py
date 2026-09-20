"""Evaluation-only recovery of two already-saved pilot checkpoints.
The source jobs timed out AFTER adapter saving. No optimizer or train call here.
Preserve source failure, sampled training logs, exact weights, and every new record.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,os,shutil,sys,zipfile
from pathlib import Path

REPO='Jaksenc/parameter-golf';SOURCE_RUN=35529849663
ASSETS={
 'direct':{'artifact':10612070095,'job':106129264729,'zip_hash':'d89c92118e3b12ca9f191a8e69ef6a9836021955d9bf6e0443887bed73b7cc5a','adapter_hash':'be9919dce3953466ac155d533c67458db52eec310252824954e6a174c1f8dc4e'},
 'auxiliary':{'artifact':10611114575,'job':106129264704,'zip_hash':'7d5eecaae6f85ecf3096596100f95e4299883e6a9201dcf9e2629893da277e07','adapter_hash':'d2aca4c357b4b862602e4552c6a9cc0c78d0c7417406194247bbdbd765b7be9f'}}
BLOBS={'semantic_data_v1.py':'9a2e7846940cfc9ca405855afcf2524af82b0fe3','semantic_train_v1.py':'651f53916739407e6a5b7c81be300673f37b6562'}
DATA_HASH='2daff7f09aec40b92b5edc76241d6c3561dd0597d278e10f46bc4f9714590501'

def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False));tmp.replace(path)
def emit(kind,**kw):print(json.dumps({'kind':kind,**kw},sort_keys=True,allow_nan=False),flush=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def parse_logs(text):
    out=[]
    for line in text.splitlines():
        k=line.find('{')
        if k<0:continue
        try:r=json.loads(line[k:])
        except json.JSONDecodeError:continue
        if isinstance(r,dict) and r.get('kind') in ['protocol','loaded','training','baseline_complete','reasoning_case']:out.append(r)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ASSETS,required=True);a=ap.parse_args()
    root=Path('evaluation-'+a.arm);root.mkdir(exist_ok=False);spec=ASSETS[a.arm]
    import requests
    token=os.environ['GH_TOKEN'];headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
    def get(path,raw=False):
        r=requests.get('https://api.github.com/repos/'+REPO+path,headers=headers,timeout=90)
        r.raise_for_status();return r.content if raw else r.json()
    meta=get(f"/actions/jobs/{spec['job']}")
    if meta['status']!='completed' or meta['run_id']!=SOURCE_RUN:raise RuntimeError('Wrong/incomplete source')
    logs=get(f"/actions/jobs/{spec['job']}/logs?recovery=semantic-v2",True).decode('utf-8-sig')
    trace=parse_logs(logs);steps=[r for r in trace if r['kind']=='training']
    if [r['step'] for r in steps]!=[1,12,24,36,48,60,72,84,96,108]:raise RuntimeError('Final 108-step training record not verified')
    if any(r['arm']!=a.arm or not math.isfinite(r['loss']) or not math.isfinite(r['gradient_norm']) for r in steps):raise RuntimeError('Bad training log')
    if 'timed out after 29 minutes' not in logs:raise RuntimeError('Source failure is not the audited timeout')
    blob=get(f"/actions/artifacts/{spec['artifact']}/zip",True)
    if sha(blob)!=spec['zip_hash']:raise RuntimeError('Archive digest mismatch')
    saved=root/'source';saved.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if sum(v.file_size for v in z.infolist())>10_000_000:raise RuntimeError('Archive too large')
        for v in z.infolist():
            p=Path(v.filename)
            if p.is_absolute() or '..' in p.parts or ((v.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unsafe archive entry')
        z.extractall(saved)
    for name,digest in BLOBS.items():
        b=(saved/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=digest:raise RuntimeError('Source code changed')
    config=json.loads((saved/'adapter_config.json').read_text())
    if config['sha256']!=spec['adapter_hash'] or sha((saved/'adapter.safetensors').read_bytes())!=spec['adapter_hash']:raise RuntimeError('Wrong checkpoint')
    if config['arm']!=a.arm or config['stage']!='fact' or config['seed']!=17:raise RuntimeError('Unexpected training arm')
    source_receipt={'source_run':SOURCE_RUN,'source_job':spec['job'],'source_conclusion':meta['conclusion'],'failure':'29-minute step timeout after 108 updates and checkpoint save; original post-training predictions were not persisted','artifact':spec['artifact'],'zip_sha256':spec['zip_hash'],'adapter_sha256':spec['adapter_hash'],'sampled_training_log':trace,'training_log_sha256':sha(logs.encode()),'recorded_training_steps':108,'full_per_step_trace_available':False,'retrained':False}
    write(root/'source-receipt.json',source_receipt)
    if a.arm=='direct':
        old=get('/actions/jobs/106125396181/logs?recovery=reasoning-v2',True).decode('utf-8-sig')
        write(root/'original-reasoning.json',{'source_job':106125396181,'records':[r for r in parse_logs(old) if r['kind']=='reasoning_case'],'qualification':'Only completed capped generation records retained; not a completed six-case reasoning comparison.'})
    sys.path.insert(0,str(saved.resolve()))
    from semantic_train_v1 import Runtime
    from semantic_data_v1 import dataset,tests,digest,prompt_task,TRUTH
    if tests()['data_hash']!=DATA_HASH:raise RuntimeError('Changed population')
    allrows=dataset();rows=[r for r in allrows if r['split'] in ['wording','composition']];dev=[r for r in allrows if r['split']=='development']
    baseline=json.loads((saved/'baseline.json').read_text());refbase={r['id']:r for r in baseline}
    if [r['id'] for r in baseline]!=[r['id'] for r in rows]:raise RuntimeError('Baseline incomplete')
    rt=Runtime();basechecks=[];padding=[]
    for i in [0,35,71]:
        row=rows[i];p=rt.predict(row);ref=refbase[row['id']];err=max(abs(x-y) for x,y in zip(p['logits'],ref['logits'],strict=True))
        if err>2e-3 or p['choice']!=ref['choice'] or p['input_hash']!=ref['input_hash']:raise RuntimeError('Fresh-process baseline mismatch')
        basechecks.append({'id':row['id'],'max_logit_difference':err,'choice':p['choice']})
        with rt.torch.no_grad():z=rt.forward(rt.encode(prompt_task(row,'direct')[0],pad=True)).tolist()
        padding.append({'arm':'baseline','id':row['id'],'max_logit_difference':max(abs(x-y) for x,y in zip(z,p['logits'],strict=True)),'same_choice':max(range(6),key=z.__getitem__)==p['choice']})
    adapter=rt.attach_lora()
    from safetensors.torch import load_file
    state=load_file(str(saved/'adapter.safetensors'));expected=rt.adapter_state()
    if state.keys()!=expected.keys() or any(state[k].shape!=expected[k].shape for k in state):raise RuntimeError('Adapter layout mismatch')
    rt.load_adapter(state);predictions=[];facts=[]
    def retain(kind,record):
        with (root/'records.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,**record},sort_keys=True,allow_nan=False)+'\n')
    for i,row in enumerate(rows):
        p=rt.predict(row);predictions.append(p);retain('decision',p)
        if i%18==17:emit('evaluation_progress',arm=a.arm,completed=i+1,population=len(rows))
    write(root/'predictions.json',predictions)
    reloadchecks=[];ref={r['id']:r for r in predictions}
    rt.load_adapter(state)
    for i in [0,35,71]:
        row=rows[i];p=rt.predict(row);err=max(abs(x-y) for x,y in zip(p['logits'],ref[row['id']]['logits'],strict=True))
        if err>2e-5 or p['choice']!=ref[row['id']]['choice']:raise RuntimeError('Reload changed decisions')
        reloadchecks.append({'id':row['id'],'max_logit_difference':err})
        with rt.torch.no_grad():z=rt.forward(rt.encode(prompt_task(row,'direct')[0],pad=True)).tolist()
        padding.append({'arm':a.arm,'id':row['id'],'max_logit_difference':max(abs(x-y) for x,y in zip(z,p['logits'],strict=True)),'same_choice':max(range(6),key=z.__getitem__)==p['choice']})
    for row in dev:
        p={mode:rt.predict(row,mode) for mode in ['fact_p','fact_q']};rec={'id':row['id'],'predictions':p,'gold_facts':row['facts']};facts.append(rec);retain('fact_diagnostic',rec)
    write(root/'facts.json',facts)
    from semantic_train_v1 import stats
    metrics=stats(rows,predictions)
    report={'complete':True,'arm':a.arm,'source':source_receipt,'data_hash':DATA_HASH,'adapter':config,'runtime':rt.receipt,'baseline':baseline,'predictions':predictions,'metrics':metrics,'baseline_metrics':stats(rows,baseline),'baseline_replay':basechecks,'saved_adapter_replay':reloadchecks,'padding_diagnostics':padding,'fact_diagnostic':facts,'joint_fact_correct':sum(all(p['predictions']['fact_'+x]['choice'] is not None and TRUTH[p['predictions']['fact_'+x]['choice']]==p['gold_facts'][x] for x in ['p','q']) for p in facts),'neural_calls':120,'recovery_run':os.environ.get('GITHUB_RUN_ID'),'new_optimizer_steps':0,'limitations':['Only saved final four-FFN LoRA updates; original jobs timed out after checkpoint saving','One seed, 54 training scenarios, 108 updates; tiny synthetic discovery pilot','Closed-library rule recognition and three recurring property contexts','No completed bounded-reasoning control, real-language generalization or native Mac benchmark','Post-training facts use the original development set; no hyperparameters changed']}
    write(root/'evaluation.json',report)
    shutil.copy(__file__,root/'semantic_saved_eval_v2.py')
    emit('evaluation_complete',arm=a.arm,metrics=metrics,joint_fact_correct=report['joint_fact_correct'],fact_n=len(facts),adapter_sha256=spec['adapter_hash'],baseline_replay=basechecks,padding_diagnostics=padding)
if __name__=='__main__':main()
