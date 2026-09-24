"""Complete only absent Compact Evidence records; never overwrite saved successes.

This execution helper is outside the frozen scientific module. All model settings,
source code, prompts, labels and limits remain unchanged. Existing explicit errors
are retained as measured failures, not silently retried or omitted.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sqlite3,subprocess,sys
from pathlib import Path
from compact_evidence.cases import corpus,digest
from compact_evidence.engine import assign
from compact_evidence.prompts import MODES

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def successes(directory):
    d=Path(directory);records={};calls={}
    for path in (d/'records').glob('*.json'):
        r=load(path)
        if r['status']=='ok':records[path.name]=sha(path)
    if (d/'calls.sqlite').exists():
        conn=sqlite3.connect((d/'calls.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        for key,text,h in conn.execute('SELECT key,payload,hash FROM calls'):
            obj=json.loads(text);assert digest(obj)==h;calls[key]=h
        conn.close()
    return {'successful_record_hashes':records,'successful_call_payload_hashes':calls}

def plan(args):
    root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    missing=[];records=0;bygroup=[]
    for split in ('development','replication'):
        assignments=assign(corpus(split),24)
        for i,group in enumerate(assignments):
            source=root/f'ce-{split}-{i}'
            if not source.is_dir():raise RuntimeError(f'Original source artifact unavailable: {source}')
            target=out/'runs'/split/f'shard-{i}'
            shutil.copytree(source,target,dirs_exist_ok=True)
            wanted={(r['id'],mode) for r in group for mode in MODES}
            seen=set()
            for path in (target/'records').glob('*.json'):
                row=load(path);key=(row['id'],row['mode'])
                if key in seen or key not in wanted:raise RuntimeError('Duplicate/unknown original record')
                if row['status'] not in ('ok','error'):raise RuntimeError('Unrecognized original status')
                seen.add(key)
            snap=successes(target);missing_keys=sorted(wanted-seen)
            needs_recovery=bool(missing_keys) or not (target/'receipt.json').is_file()
            if needs_recovery:missing.append({'split':split,'shard':i})
            save(target/'recovery-original.json',{'split':split,'shard':i,'original_records':len(seen),'missing_record_keys':missing_keys,**snap})
            records+=len(seen);bygroup.append({'split':split,'shard':i,'records':len(seen),'needs_recovery':needs_recovery})
    result={'original_run':36034291168,'original_records':records,'wanted_records':720,'missing_records':720-records,'matrix':{'include':missing},'groups':bygroup,'settings_unchanged':True}
    save(out/'recovery-plan.json',result)
    print(json.dumps(result,indent=2))
    if args.github_output:
        with open(args.github_output,'a') as f:
            f.write('matrix='+json.dumps(result['matrix'],separators=(',',':'))+'\n')
            f.write('needed='+('true' if missing else 'false')+'\n')

def resume(args):
    d=Path(args.out);before=load(d/'recovery-original.json')
    check=successes(d)
    if check['successful_record_hashes']!=before['successful_record_hashes'] or check['successful_call_payload_hashes']!=before['successful_call_payload_hashes']:raise RuntimeError('Recovery input differs from preserved original')
    # No --retry-errors: all existing explicit failures remain failures.
    r=subprocess.run([sys.executable,'-u','-m','compact_evidence','run','--split',args.split,'--shard',str(args.shard),'--nshards','24','--out',str(d)])
    after=successes(d)
    for name,h in before['successful_record_hashes'].items():
        if after['successful_record_hashes'].get(name)!=h:raise RuntimeError('Changed an original successful decision')
    for key,h in before['successful_call_payload_hashes'].items():
        if after['successful_call_payload_hashes'].get(key)!=h:raise RuntimeError('Changed an original successful call')
    group=assign(corpus(args.split),24)[args.shard]
    keys={(load(p)['id'],load(p)['mode']) for p in (d/'records').glob('*.json')}
    if keys!={(x['id'],m) for x in group for m in MODES}:raise RuntimeError('Recovery remains incomplete')
    if not (d/'receipt.json').exists():raise RuntimeError('Missing completed ledger receipt')
    save(d/'recovery-check.json',{'original_records_preserved':len(before['successful_record_hashes']),'original_calls_preserved':len(before['successful_call_payload_hashes']),'new_successful_calls':len(after['successful_call_payload_hashes'])-len(before['successful_call_payload_hashes']),'returncode':r.returncode,'same_scientific_code':True,'recovered_record_coverage':len(keys)})
    # Preserve explicit call errors in the report, even if a run's CLI exits 1.
    if r.returncode not in (0,1):raise RuntimeError(f'Native recovery exited {r.returncode}')

def merge(args):
    base=Path(args.base);recovered=Path(args.recovered)
    plan=load(base/'recovery-plan.json')
    for item in plan['matrix']['include']:
        split=item['split'];i=item['shard'];source=recovered/f'ce-recovered-{split}-{i}'
        if not (source/'recovery-check.json').is_file():raise RuntimeError('Missing verified recovery group')
        dest=base/'runs'/split/f'shard-{i}';before=load(dest/'recovery-original.json');after=successes(source)
        if any(after['successful_record_hashes'].get(k)!=h for k,h in before['successful_record_hashes'].items()):raise RuntimeError('Old successful row altered')
        if any(after['successful_call_payload_hashes'].get(k)!=h for k,h in before['successful_call_payload_hashes'].items()):raise RuntimeError('Old successful model call altered')
        shutil.rmtree(dest);shutil.copytree(source,dest)
    for split in ('development','replication'):
        subprocess.run([sys.executable,'-m','compact_evidence','report','--split',split,'--root',str(base/'runs'/split),'--out',str(base/f'{split}-summary.json')],check=True)
    save(base/'recovery-completed.json',{'source_run':36034291168,'original_records':plan['original_records'],'new_mode_records':plan['missing_records'],'complete_cases_and_modes':720,'recovery_groups':plan['matrix']['include'],'no_retraining':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('plan');a.add_argument('--root',required=True);a.add_argument('--out',required=True);a.add_argument('--github-output')
    a=subs.add_parser('resume');a.add_argument('--split',required=True,choices=('development','replication'));a.add_argument('--shard',required=True,type=int);a.add_argument('--out',required=True)
    a=subs.add_parser('merge');a.add_argument('--base',required=True);a.add_argument('--recovered',required=True)
    a=p.parse_args();{'plan':plan,'resume':resume,'merge':merge}[a.command](a)
