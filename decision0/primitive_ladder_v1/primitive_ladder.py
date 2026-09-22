"""Locate numerical decision failures without changing weights or consulting JevBench.

Five distinct diagnostic views; only original_end_to_end has no privileged
component inputs. Two outcome orderings are fixed and both are reported.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, re, sys, traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

STAGES=('original_end_to_end','bind_operands','execute_given_operands',
        'compare_given_result','original_plus_result')
PROTOCOL={'id':'decision0-primitive-ladder-v1','families':['amount','elapsed'],
 'sources_per_family':8,'endpoints_per_source':3,'stages':list(STAGES),
 'outcome_orders':['forward','reverse'],'scored_calls':480,'shards':8,
 'weights':'unchanged Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'updates':0,'benchmark_calls':0,'selection':'none; all predefined views reported',
 'diagnostic_privilege':'Every view except original_end_to_end changes task information or supplies correct intermediate values.',
 'data_origin':'New constructed situations; not an independently reviewed language corpus.'}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def money(c):return f'${c//100}.{c%100:02d}'
def tag(rng):return ''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(8))
def visible(r):return {k:r[k] for k in ('id','state','question','options')}
def reference(r):
    o=r['oracle']
    if r['family']=='amount':
        total=int(o['quantity'])*Decimal(o['unit'][1:])+Decimal(o['fee'][1:])-Decimal(o['credit'][1:]);cap=Decimal(o['cap'][1:])
        assert int(total*100)==o['result_cents']
        return 'below' if total<cap else 'above' if total>cap else 'equal'
    dt=datetime.fromisoformat(o['event'])-datetime.fromisoformat(o['start'])
    m=dt.total_seconds()/60
    assert m==o['elapsed_minutes']
    return 'before' if m<o['allowed_minutes'] else 'after' if m>o['allowed_minutes'] else 'at'

def corpus():
    rows=[]
    for family in PROTOCOL['families']:
      for source_index in range(8):
        rng=random.Random(int(digest(['primitive-ladder-sep22',family,source_index])[:16],16))
        key=tag(rng);source=digest([family,key])[:24];gap=rng.choice([1,7,31]);offsets=[-gap,0,gap]
        order=[0,1,2];rng.shuffle(order)
        if family=='amount':
          n=rng.randint(3,11);unit=rng.randint(121,489);fee=rng.randint(101,219);credit=rng.randint(11,89);cap=n*unit+fee-credit
          results=[cap+d for d in offsets];values=[money(fee+d) for d in offsets]
          records=[f'Order {key}: quantity {n}; unit price {money(unit)}.',f'Order {key}: handling fee {{VALUE}}.',f'Order {key}: credit {money(credit)}; cap {money(cap)}.']
          for _ in range(4):records.append(f'Order {tag(rng)}: quantity {rng.randint(3,11)}; unit price {money(rng.randint(121,489))}; handling fee {money(rng.randint(101,219))}; credit $0.50; cap $28.31.')
          endlabels=['below','equal','above']
          enddesc=['The order total is strictly below its cap.','The order total equals its cap.','The order total is strictly above its cap.']
          q=f'For order {key}, compare its exact total with its cap. Total = quantity times unit price, plus handling fee, minus credit. Use only this order. Do not round intermediate amounts.'
          binddesc=[f'Quantity {n}; unit price {money(unit)}; handling fee {v}; credit {money(credit)}; cap {money(cap)}.' for v in values]
          execdesc=[f'The exact total is {money(v)}.' for v in results]
        else:
          allowed=rng.randint(120,1920)
          start=datetime(2028,rng.randint(1,11),rng.randint(1,25),rng.randint(0,23),rng.randint(0,59),tzinfo=timezone(timedelta(minutes=rng.choice([-300,0,120]))))
          zone=timezone(timedelta(minutes=rng.choice([-210,60,345])))
          values=[(start+timedelta(minutes=allowed+d)).astimezone(zone).isoformat() for d in offsets];results=[allowed+d for d in offsets]
          records=[f'Case {key}: start {start.isoformat()}; allowed interval {allowed} minutes.',f'Case {key}: submission {{VALUE}}.','Use real elapsed time, not business time. Explicit UTC offsets control.']
          for _ in range(4):records.append(f'Case {tag(rng)}: start {(start+timedelta(minutes=rng.randint(-1000,1000))).isoformat()}; allowed interval {rng.randint(120,1920)} minutes; status pending.')
          endlabels=['before','at','after'];enddesc=['The submission is strictly before its deadline.','The submission is exactly at its deadline.','The submission is strictly after its deadline.']
          q=f'Compare submission {key} with its deadline: its start plus its allowed interval. Respect explicit UTC offsets; other cases are irrelevant.'
          binddesc=[f'Start {start.isoformat()}; submission {v}; allowed interval {allowed} minutes.' for v in values]
          execdesc=[f'The exact elapsed time is {v} minutes.' for v in results]
        rng.shuffle(records)
        for endpoint in range(3):
          state='\n'.join(f'Record {i+1}. '+s.replace('{VALUE}',values[endpoint]) for i,s in enumerate(records))
          if family=='amount':
            o={'key':key,'quantity':n,'unit':money(unit),'fee':values[endpoint],'credit':money(credit),'cap':money(cap),'result_cents':results[endpoint]}
            execution_state=f'Quantity {n}; unit price {money(unit)}; handling fee {values[endpoint]}; credit {money(credit)}.'
            execution_q='Compute quantity times unit price, plus handling fee, minus credit. Which listed exact total is correct?'
            comparison_state=f'Computed order total: {money(results[endpoint])}. Spending cap: {money(cap)}.'
            comparison_q='Compare the supplied order total with the supplied spending cap. No additional calculation or records are required.'
            certificate=f'Verified intermediate calculation for order {key}: its exact total is {money(results[endpoint])}. This note does not classify it relative to the cap.'
          else:
            o={'key':key,'start':start.isoformat(),'event':values[endpoint],'allowed_minutes':allowed,'elapsed_minutes':results[endpoint]}
            execution_state=f'Start timestamp: {start.isoformat()}. Submission timestamp: {values[endpoint]}.'
            execution_q='Compute exact elapsed minutes from start to submission, respecting the explicit UTC offsets. Which listed value is correct?'
            comparison_state=f'Computed elapsed time: {results[endpoint]} minutes. Allowed interval: {allowed} minutes.'
            comparison_q='Compare computed elapsed time with the allowed interval: strictly less means before, equal means at, and strictly greater means after the deadline.'
            certificate=f'Verified intermediate calculation for case {key}: elapsed time from start to submission is exactly {results[endpoint]} minutes. This note does not classify it relative to the allowed interval.'
          for stage in STAGES:
            ss,qq=state,q
            labels=endlabels;desc=enddesc;answer=endlabels[endpoint]
            if stage=='bind_operands':qq=f'Copy the recorded inputs belonging to {key}, without performing arithmetic. Which complete tuple matches its records?';labels=['tuple0','tuple1','tuple2'];desc=binddesc;answer=labels[endpoint]
            elif stage=='execute_given_operands':ss,qq=execution_state,execution_q;labels=['value0','value1','value2'];desc=execdesc;answer=labels[endpoint]
            elif stage=='compare_given_result':ss,qq=comparison_state,comparison_q
            elif stage=='original_plus_result':ss=state+'\n'+certificate
            opts=[{'id':labels[j],'description':desc[j]} for j in order]
            row={'id':digest([source,endpoint,stage])[:24],'source':source,'family':family,'stage':stage,'endpoint':endpoint,'state':ss,'question':qq,'options':opts,'expected':answer,'original_expected':endlabels[endpoint],'oracle':o,'privileged':stage!='original_end_to_end'}
            assert reference(row)==row['original_expected']
            rows.append(row)
    return rows

def checks():
    rows=corpus();assert len(rows)==240 and len({r['id'] for r in rows})==240
    assert len({r['source'] for r in rows})==16
    for r in rows:
      assert set(visible(r))=={'id','state','question','options'}
      assert r['expected'] in [o['id'] for o in r['options']]
      assert reference(r)==r['original_expected']
      if r['stage']=='original_end_to_end':
        key=r['oracle']['key'];s=r['state']
        if r['family']=='amount':
          n,p=re.search(rf'Order {key}: quantity (\d+); unit price (\$\d+\.\d{{2}})',s).groups()
          fee=re.search(rf'Order {key}: handling fee (\$\d+\.\d{{2}})',s).group(1)
          credit,cap=re.search(rf'Order {key}: credit (\$\d+\.\d{{2}}); cap (\$\d+\.\d{{2}})',s).groups()
          t=int(n)*Decimal(p[1:])+Decimal(fee[1:])-Decimal(credit[1:]);b=Decimal(cap[1:])
          assert ('below' if t<b else 'above' if t>b else 'equal')==r['expected']
        else:
          start,window=re.search(rf'Case {key}: start ([^;\s]+); allowed interval (\d+) minutes',s).groups()
          event=re.search(rf'Case {key}: submission ([^\s]+)\.',s).group(1)
          diff=(datetime.fromisoformat(event)-datetime.fromisoformat(start)).total_seconds()-int(window)*60
          assert ('before' if diff<0 else 'after' if diff>0 else 'at')==r['expected']
    for family in PROTOCOL['families']:
      for stage in STAGES:
        rs=[r for r in rows if r['family']==family and r['stage']==stage]
        assert len(rs)==24 and set(Counter(r['expected'] for r in rs).values())=={8}
        for source in {r['source'] for r in rs}:
          g=sorted([r for r in rs if r['source']==source],key=lambda r:r['endpoint'])
          assert len(g)==3 and g[0]['options']==g[1]['options']==g[2]['options']
          if stage=='original_end_to_end':assert sum(a!=b for a,b in zip(g[0]['state'].splitlines(),g[1]['state'].splitlines()))==1
    return {'rows':len(rows),'sources':16,'calls':2*len(rows),'sha256':digest(rows),'reference_checks':len(rows),'passed':True}

def run(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
    from flow_study import Runtime,save,filehash
    from prompt_variants import encode_checked
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=checks();save(out/'protocol.json',PROTOCOL);save(out/'data_checks.json',report)
    rows=corpus();sources=sorted({r['source'] for r in rows});assignment={s:i%8 for i,s in enumerate(sources)};assigned=[r for r in rows if assignment[r['source']]==args.shard]
    rt=Runtime();save(out/'runtime.json',rt.meta)
    for r in assigned:
      encode_checked(rt.tokenizer,visible(r),'baseline',8192)
      rev=copy.deepcopy(r);rev['options'].reverse();encode_checked(rt.tokenizer,visible(rev),'baseline',8192)
    records=[]
    with (out/'records.jsonl').open('w') as f:
      for r in assigned:
        for order in PROTOCOL['outcome_orders']:
          v=copy.deepcopy(r)
          if order=='reverse':v['options'].reverse()
          rec=rt.score(v,'baseline')
          rec.update({k:r[k] for k in ('source','family','stage','endpoint','expected','privileged')});rec['outcome_order']=order;rec['correct']=rec['predicted']==r['expected']
          f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();records.append(rec)
          print(json.dumps({'phase':'primitive','shard':args.shard,'done':len(records),'total':len(assigned)*2}),flush=True)
    save(out/'receipt.json',{'rows':len(assigned),'records':len(records),'sha256':filehash(out/'records.jsonl'),'protocol_sha256':digest(PROTOCOL),'corpus_sha256':report['sha256'],'weights_changed':False})

def aggregate(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'input_flow_v1'))
    from flow_study import save,filehash
    rows=corpus();reference_by_id={r['id']:r for r in rows};records=[];seen=set()
    for f in Path(args.root).rglob('records.jsonl'):
      rec=json.loads((f.parent/'receipt.json').read_text())
      assert rec['sha256']==filehash(f) and rec['protocol_sha256']==digest(PROTOCOL) and rec['corpus_sha256']==digest(rows)
      for line in f.read_text().splitlines():
        r=json.loads(line);key=(r['id'],r['outcome_order']);assert key not in seen;seen.add(key)
        assert r['expected']==reference_by_id[r['id']]['expected'];records.append(r)
    assert seen=={(r['id'],o) for r in rows for o in PROTOCOL['outcome_orders']}
    summary={}
    for family in PROTOCOL['families']:
      summary[family]={}
      for stage in STAGES:
        rs=[r for r in records if r['family']==family and r['stage']==stage];m={}
        for order in PROTOCOL['outcome_orders']:
          one=[r for r in rs if r['outcome_order']==order]
          m[order]={'n':len(one),'correct':sum(r['correct'] for r in one),'complete_triplets':sum(all(r['correct'] for r in one if r['source']==s) for s in {r['source'] for r in one})}
        lookup={(r['id'],r['outcome_order']):r for r in rs}
        m['order_stable']=sum(lookup[(i,'forward')]['predicted']==lookup[(i,'reverse')]['predicted'] for i in {r['id'] for r in rs})
        m['both_orders_correct']=sum(lookup[(i,'forward')]['correct'] and lookup[(i,'reverse')]['correct'] for i in {r['id'] for r in rs})
        summary[family][stage]=m
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save(out/'results.json',{'summary':summary,'protocol':PROTOCOL,'records':len(records),'weights_changed':False,'official_score':None})
    save(out/'data.json',rows)
    (out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in sorted(records,key=lambda r:(r['id'],r['outcome_order']))))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('check')
    a=sub.add_parser('run');a.add_argument('--shard',type=int,choices=range(8),required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',required=True);a.add_argument('--out',required=True)
    args=p.parse_args()
    if args.mode=='check':print(json.dumps(checks(),indent=2))
    else:
      try:{'run':run,'aggregate':aggregate}[args.mode](args)
      except Exception as e:
        path=Path(args.out);path.mkdir(parents=True,exist_ok=True);(path/'FAILED.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()}));raise
