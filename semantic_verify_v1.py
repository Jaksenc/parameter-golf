"""Independent process reload of completed research checkpoints, not retraining.
Source artifacts are bounded, path-checked, and complete before inference.
Also recovers original timeout records without relabeling missing generation.
Fact diagnostics are on the original development cases, not a new test set.
"""
from __future__ import annotations
import hashlib,io,json,os,zipfile
from pathlib import Path
import requests
from semantic_train_v1 import Runtime,save,filehash
from semantic_data_v1 import dataset,prompt_task,TRUTH
REPO='Jaksenc/parameter-golf';RUN=35529849663

def main():
    token=os.environ['GH_TOKEN'];root=Path('fresh-verification');root.mkdir(exist_ok=False)
    def get(path,text=False):
        r=requests.get('https://api.github.com/repos/'+REPO+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'},timeout=90)
        r.raise_for_status();return r.content if text else r.json()
    run=get(f'/actions/runs/{RUN}')
    if run['status']!='completed' or run['conclusion']!='success':raise RuntimeError('Complete successful source experiment required')
    artifacts=get(f'/actions/runs/{RUN}/artifacts')['artifacts'];receipt=[]
    for label in ['diagnostic','direct','auxiliary']:
        found=[a for a in artifacts if a['name']==f'semantic-{label}-recovery-v1' and not a['expired']]
        if len(found)!=1:raise ValueError('Missing or ambiguous artifact')
        item=found[0];blob=get(f"/actions/artifacts/{item['id']}/zip",True)
        dest=root/label;dest.mkdir()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            if sum(x.file_size for x in z.infolist())>20_000_000:raise ValueError('Oversized archive')
            for n in z.namelist():
                if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe archive path')
            z.extractall(dest)
        receipt.append({'artifact':item['id'],'label':label,'zip_sha256':hashlib.sha256(blob).hexdigest()})
    for label in ['diagnostic','direct','auxiliary']:
        for name in ['semantic_data_v1.py','semantic_train_v1.py']:
            if filehash(root/label/name)!=filehash(name):raise ValueError('Execution-source mismatch')
    original=get('/actions/jobs/106125396181/logs?archive=semantic-final',True).decode('utf-8-sig');old=[]
    for line in original.splitlines():
        start=line.find('{')
        if start<0:continue
        try:r=json.loads(line[start:])
        except json.JSONDecodeError:continue
        if isinstance(r,dict) and r.get('kind') in ['diagnostic_case','reasoning_case']:old.append(r)
    save(root/'original-timeout-records.json',{'source_job':106125396181,'log_sha256':hashlib.sha256(original.encode()).hexdigest(),'records':old,'qualification':'Four completed capped-generation calls; remaining generation incomplete. Partial non-generation records retained separately.'})
    diagnostic=json.loads((root/'diagnostic/diagnostic.json').read_text());new={r['id']:r for r in diagnostic['cases']}
    comparisons=[]
    for r in old:
        if r['kind']!='diagnostic_case':continue
        differences=[abs(x-y) for mode,p in r['predictions'].items() for x,y in zip(p['logits'],new[r['id']]['predictions'][mode]['logits'])]
        comparisons.append({'id':r['id'],'max_logit_difference':max(differences)})
    rt=Runtime();allrows=dataset();rows=[r for r in allrows if r['split'] in ['wording','composition']];selected=[rows[i] for i in [0,35,71]]
    baseline=[];padding=[]
    for row in selected:
        result=rt.predict(row);baseline.append(result)
        task,_=prompt_task(row,'direct')
        with rt.torch.no_grad():z=rt.forward(rt.encode(task,pad=True)).tolist()
        padding.append({'arm':'baseline','id':row['id'],'maximum_logit_difference':max(abs(x-y) for x,y in zip(z,result['logits'])),'same_choice':max(range(len(z)),key=z.__getitem__)==result['choice']})
    rt.attach_lora();from safetensors.torch import load_file
    checked=[];stage_records={}
    for label in ['direct','auxiliary']:
        d=json.loads((root/label/'training.json').read_text());reference={r['id']:r for r in d['predictions']};base={r['id']:r for r in d['baseline']}
        if filehash(root/label/'adapter.safetensors')!=d['adapter_sha256']:raise ValueError('Adapter bytes mismatch')
        state=load_file(str(root/label/'adapter.safetensors'));expected=rt.adapter_state()
        if state.keys()!=expected.keys() or any(state[k].shape!=expected[k].shape for k in state):raise ValueError('Adapter structure mismatch')
        rt.load_adapter(state)
        for row in selected:
            result=rt.predict(row);ref=reference[row['id']];delta=max(abs(x-y) for x,y in zip(result['logits'],ref['logits']))
            if delta>2e-3 or result['choice']!=ref['choice'] or result['input_hash']!=ref['input_hash']:raise ValueError('Fresh-process replay failed')
            checked.append({'arm':label,'id':row['id'],'maximum_logit_difference':delta,'choice':result['choice']})
            task,_=prompt_task(row,'direct')
            with rt.torch.no_grad():z=rt.forward(rt.encode(task,pad=True)).tolist()
            padding.append({'arm':label,'id':row['id'],'maximum_logit_difference':max(abs(x-y) for x,y in zip(z,result['logits'])),'same_choice':max(range(len(z)),key=z.__getitem__)==result['choice']})
        for b in baseline:
            if max(abs(x-y) for x,y in zip(b['logits'],base[b['id']]['logits']))>2e-3 or b['choice']!=base[b['id']]['choice']:raise ValueError('Fresh base differs')
        dev=[r for r in allrows if r['split']=='development'];stage_records[label]=[]
        for row in dev:
            preds={mode:rt.predict(row,mode) for mode in ['fact_p','fact_q']}
            stage_records[label].append({'id':row['id'],'predictions':preds,'gold_facts':row['facts']})
            save(root/'post-training-facts.json',stage_records)
        joint=sum(all(p['predictions']['fact_'+a]['choice'] is not None and TRUTH[p['predictions']['fact_'+a]['choice']]==p['gold_facts'][a] for a in ['p','q']) for p in stage_records[label])
        print(json.dumps({'kind':'post_training_fact_diagnostic','arm':label,'joint_correct':joint,'n':len(dev),'qualification':'Original development cases only; no tuning or model selection'}),flush=True)
    report={'passed':True,'source_run':RUN,'artifacts':receipt,'fresh_process_checkpoint_checks':checked,'baseline_replayed_cases':len(baseline),'padding_diagnostics':padding,'recovery_vs_original':comparisons,'post_training_fact_calls':72,'qualification':'New Python process and freshly loaded base; three checkpoint replay cases per trained arm, not full replication or retraining. Additional fact diagnostics reuse the 18 development cases. Padding and fact diagnostics do not alter models or prior results.'}
    save(root/'verification.json',report);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()
