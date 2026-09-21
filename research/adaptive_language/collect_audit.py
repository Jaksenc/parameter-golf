"""Recover immutable completed receipts, verify provenance, replay programs and score."""
import collections,concurrent.futures,hashlib,json,math,statistics,sys
from pathlib import Path
import audit_results as audit
import bootstrap as boot
from benchmark import acquire,score
from grounding_followup import cohort,PROMPT_SHA
BASE='https://raw.githubusercontent.com/Jaksenc/parameter-golf/'
BRANCH='research/adaptive-language-v07-20260921'

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    ref=boot.json_get('https://api.github.com/repos/Jaksenc/parameter-golf/git/ref/heads/'+BRANCH)['object']['sha']
    root=Path('/tmp/adaptive-replay');root.mkdir(exist_ok=True)
    names=[f'language-shard-{i:02d}.json' for i in range(12)]+[f'grounding-shard-{i:02d}.json' for i in range(4)]
    def fetch(name):
        raw=boot.get(BASE+ref+'/research/adaptive_language/results/'+name);d=json.loads(raw);assert d['status']=='completed',(name,d.get('error'));(root/name).write_bytes(raw);return name,d,{'name':name,'sha256':sha(raw),'bytes':len(raw),'url':BASE+ref+'/research/adaptive_language/results/'+name}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(fetch,names))
    primary=audit.aggregate(root);primary_ids={r['id'] for r in primary['compact_rows']}
    original,_,sources=acquire(0,1);lookup={r['id']:r for r in original}
    assert set(lookup)==primary_ids and len(primary_ids)==86
    follow,excluded,ds=cohort();fl={r['id']:r for r in follow};assert len(fl)==16 and not primary_ids.intersection(fl)
    rows=[];seen=set();replayed=0;steps=0;manifest=[];setup=0;run_ids=set();primary_raw=[];follow_source_shas=set()
    for name,d,m in items:
        if name.startswith('language'):
            for r in d['rows']:
                t=lookup[r['id']];assert r['question']==t['question'] and r['reference']==t['reference'] and r['suite']==t['suite'] and r['family']==t['family'];primary_raw.append(r)
            continue
        manifest.append(m);assert d['prompt_sha256']==PROMPT_SHA and set(d['selected_ids'])==set(fl)
        assert d['runtime']['weights']['sha256']=='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
        assert d['runtime']['runtime']['sha256']=='9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'
        assert d['core']['decoded_sha256']=='6b37dcfc4a451d5dba399080ba362b5b5320b03e977bf49532aab2be3cd108ca'
        assert d['data_sha256']==ds and d['excluded_primary_ids']==excluded
        assert d['worker_exit']==0
        follow_source_shas.add(d['source_commit']);run_ids.add(d['run_id']);setup+=d['runtime']['setup_seconds']
        for r in d['rows']:
            t=fl[r['id']];assert r['question']==t['question'] and r['reference']==t['reference'];key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
            assert audit.judge(r['answer'],r['reference'],'GSM8K')==r['correct']
            assert sha(r['question'].encode())==r['question_sha256']
            cs=r.get('calls',[]);assert len(cs)<=(1 if r['arm']=='reasoning' else 2)
            assert sum(c.get('usage',{}).get('completion_tokens',0) for c in cs)<=768
            for i,c in enumerate(cs):assert c.get('usage',{}).get('completion_tokens',0)<=(768 if r['arm']=='reasoning' else 512 if i==0 else 256)
            ex=r.get('execution')
            if ex:
                assert sha(json.dumps(ex['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())==ex['program_sha256']
                value,trace=audit.replay(ex['plan'])
                if 'value' in ex:assert value==ex['value'] and trace==ex['trace'];replayed+=1;steps+=len(trace)
                assert ex['answer']==r['answer']
            rows.append(r)
    assert len(rows)==48
    summ={}
    for arm in ['reasoning','program','grounded_program']:
        rr=[r for r in rows if r['arm']==arm];times=[r.get('seconds') or 0 for r in rr];k=sum(r['correct'] for r in rr)
        summ[arm]={'correct':k,'n':len(rr),'accuracy':k/len(rr),'wilson_descriptive_95':audit.wilson(k,len(rr)),'answered':sum(r['answer'] is not None for r in rr),'statuses':dict(collections.Counter(r['status'] for r in rr)),'seconds':sum(times),'median_seconds':statistics.median(times),'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in rr),'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in rr),'calls':sum(len(r.get('calls',[])) for r in rr),'truncated_calls':sum(c.get('finish_reason')!='stop' for r in rr for c in r.get('calls',[])),'repaired_requests':sum(len(r.get('calls',[]))>1 for r in rr)}
    paired={};rl={(r['id'],r['arm']):r for r in rows}
    for a,b in [('program','grounded_program'),('reasoning','grounded_program')]:
        c=collections.Counter()
        for i in fl:
            x,y=rl[i,a]['correct'],rl[i,b]['correct'];c['repaired' if not x and y else 'broken' if x and not y else 'both_correct' if x else 'both_wrong']+=1
        n=c['repaired']+c['broken'];k=min(c['repaired'],c['broken']);c['mcnemar_exact_descriptive']=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
        paired[a+'->'+b]=dict(c)
    def outcomes(rs):
        return {arm:dict(collections.Counter(('accepted_program_correct' if r['correct'] else 'accepted_program_wrong') if r['status']=='computation_checked_semantics_unverified' else ('direct_json_correct' if r['correct'] else 'direct_json_wrong') if r.get('execution') else r['status'] for r in rs if r['arm']==arm)) for arm in sorted({r['arm'] for r in rs})}
    compact=[{k:r.get(k) for k in ['id','arm','answer','reference','correct','status','seconds','usage']} for r in rows]
    followup={'status':'audited','task_count':16,'attempts':48,'summary':summ,'paired':paired,'primary_overlap':0,'programs_independently_replayed':replayed,'steps_independently_replayed':steps,'prompt_sha256':PROMPT_SHA,'source_manifest':manifest,'run_ids':sorted(run_ids),'source_commits':sorted(follow_source_shas),'setup_seconds_sum':setup,'outcome_types':outcomes(rows),'compact_rows':compact}
    primary['outcome_types']=outcomes(primary_raw)
    for r in primary['source_manifest']:r['url']=BASE+ref+'/research/adaptive_language/results/'+r['name']
    result={'status':'audited','raw_results_commit':ref,'primary':{k:v for k,v in primary.items() if k not in ['compact_rows','by_family']},'followup':{k:v for k,v in followup.items() if k!='compact_rows'},'checks':{'upstream_question_reference_match':True,'primary_followup_disjoint':True,'model_runtime_pins_match':True,'program_trace_replay':True,'score_replay':True,'all_planned_attempts_present':True},'limitations':['No comparison with a frontier model','Small public zero-shot pilots, not official leaderboard scores','No new model weights trained','No persistent memory used in scored runs','No arbitrary Python tool; bounded expression interpreter','Same upper output-token limit is not equal total FLOPs or latency','Follow-up is a prompt revision after observing one primary semantic error; cohorts remain separate','Pretraining overlap unknown; descriptive intervals/tests not contamination or multiplicity corrected'],'audit_source_sha256':sha(Path(__file__).read_bytes()),'independent_evaluator_source_sha256':sha(Path(audit.__file__).read_bytes())}
    print('AUDIT_SUMMARY '+json.dumps(result,ensure_ascii=False),flush=True)
    print(boot.publish('audit-summary.json',result),flush=True)
    print(boot.publish('audit-compact.json',{'primary':primary['compact_rows'],'followup':compact,'by_family':primary['by_family'],'raw_results_commit':ref}),flush=True)
    # Retain changed follow-up cases as inspectable examples, not an additional score.
    ids=[i for i in fl if rl[i,'program']['correct']!=rl[i,'grounded_program']['correct']]
    examples=[{k:r.get(k) for k in ['id','question','arm','answer','reference','correct','status','execution','errors']} for r in rows if r['id'] in ids]
    print(boot.publish('audit-examples.json',{'role':'post-score diagnostic examples','rows':examples}),flush=True)
if __name__=='__main__':main()
