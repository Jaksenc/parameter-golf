"""Resolve missing Mini task labels through exact question membership, never answers."""
import collections, concurrent.futures, hashlib, json, sys, urllib.request
from pathlib import Path
import bbeh_run as run
MEMBERSHIP=[]
def cohort(raw,n):
    obj=json.loads(raw);rows=obj if isinstance(obj,list) else obj.get('examples',obj.get('data'))
    if not isinstance(rows,list) or len(rows)!=460:raise ValueError('Expected 460 original Mini examples')
    req=urllib.request.Request(f'https://api.github.com/repos/google-deepmind/bbeh/git/trees/{run.BBEH_REF}?recursive=1',headers={'User-Agent':'adaptive-benchmark-metadata'})
    tree=json.loads(urllib.request.urlopen(req,timeout=60).read())
    if tree.get('truncated'):raise ValueError('Incomplete benchmark tree')
    entries=[e for e in tree['tree'] if e['path'].startswith('bbeh/benchmark_tasks/') and e['path'].endswith('/task.json')]
    if len(entries)!=23:raise ValueError(f'Expected 23 task files, found {len(entries)}')
    def load(e):
        rawtask=run.download(e['path'],e['sha']);data=json.loads(rawtask);examples=data if isinstance(data,list) else data['examples']
        return e,rawtask,[x['input'] for x in examples]
    membership={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for e,rawtask,questions in pool.map(load,entries):
            task=e['path'].split('/')[2]
            MEMBERSHIP.append({'path':e['path'],'blob':e['sha'],'sha256':run.sha(rawtask)})
            for question in questions:
                h=run.sha(question.encode())
                if h in membership and membership[h]!=task:raise ValueError('Ambiguous cross-task question membership')
                membership[h]=task
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        question=r['input'];answer=r['target'];task=membership.get(run.sha(question.encode()))
        if not isinstance(question,str) or not isinstance(answer,str) or task is None:raise ValueError(f'Unmatched Mini row {i}')
        groups[task].append({'id':i,'task':task,'question':question,'answer':answer})
    if len(groups)!=23 or min(map(len,groups.values()))<n:raise ValueError('Invalid stratification')
    selected=[]
    for task in sorted(groups):selected+=sorted(groups[task],key=lambda r:run.sha(('adaptive-bbeh-pilot-v1\0'+r['question']).encode()))[:n]
    print('METADATA_REPAIR '+json.dumps({'method':'exact original-question SHA256 membership in pinned full task files','group_sizes':{k:len(v) for k,v in sorted(groups.items())},'no_targets_used_for_matching_or_selection':True}),flush=True)
    return sorted(selected,key=lambda r:(r['task'],r['id']))
original_publish=run.publish
def publish(path,text):
    data=json.loads(text);data['metadata_repair']='Mini lacks task labels; restored by exact input-only matching to original task files';data['membership_manifest']=MEMBERSHIP;data['entry_source_sha256']=run.sha(Path(__file__).read_bytes());text=json.dumps(data,indent=2);Path(path.split('/')[-1]).write_text(text)
    return original_publish(path,text)
run.cohort=cohort
run.publish=publish
if __name__=='__main__':sys.exit(run.main())
