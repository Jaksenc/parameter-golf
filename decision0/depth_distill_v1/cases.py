"""Independent, paired training/holdout scenarios. References never enter prompts."""
from __future__ import annotations
import hashlib, json, random
from datetime import datetime, timedelta, timezone

FAMILIES = ('amount','elapsed','policy','join','probability','intent','entailment','judge')

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def visible(row):
    return {k:row[k] for k in ('id','state','question','options')}

def corpus(split):
    if split not in ('train','held'): raise ValueError(split)
    rows=[]
    for fi,family in enumerate(FAMILIES):
      for source in range(6):
        rng=random.Random(int(digest(['depth-distill-v1',split,family,source])[:16],16))
        tag=''.join(rng.choice('BCDFGHJKLMNPQRSTVWXYZ') for _ in range(5))
        scenarios=[]
        if family=='amount':
          n=rng.randint(3,17); price=rng.randint(12,87); fee=rng.randint(1,30); credit=rng.randint(1,30)
          total=n*price+fee-credit
          ds=rng.sample([-1,0,1],2)
          opts=[('below','The final total is strictly below the cap.'),('equal','The final total equals the cap.'),('above','The final total is strictly above the cap.')]
          order=list(range(4));rng.shuffle(order)
          for edit,d in enumerate(ds):
            cap=total+d
            records=[f'{tag}: {n} units; unit price {price} cents.',f'{tag}: fee {fee} cents; credit {credit} cents.',f'{tag}: cap {cap} cents.',f'Other order QZ: 77 units, cap 120 cents.']
            state='\n'.join(f'Entry {j+1}: {records[i]}' for j,i in enumerate(order))
            q=f'For order {tag}, multiply units by unit price, add the fee, subtract the credit. Compare that total with its cap. All amounts are exact cents.'
            y='below' if total<cap else 'above' if total>cap else 'equal'
            scenarios.append((state,q,y,{'n':n,'price':price,'fee':fee,'credit':credit,'cap':cap,'total':total,'support':[order.index(i) for i in (0,1,2)]}))
        elif family=='elapsed':
          base=datetime(2026,rng.randint(1,10),rng.randint(1,20),rng.randint(0,20),rng.choice([0,15,30,45]),tzinfo=timezone.utc)
          window=rng.randint(45,1800);deadline=base+timedelta(minutes=window)
          ds=rng.sample([-1,0,1],2);offset=timezone(timedelta(hours=rng.choice([-5,-3,0,2,5])))
          opts=[('before','Submission is before the exact deadline.'),('at','Submission is exactly at the deadline.'),('after','Submission is after the deadline.')]
          for edit,d in enumerate(ds):
            event=(deadline+timedelta(minutes=d)).astimezone(offset)
            state=f'Case {tag}: start={base.isoformat()}, allowed interval={window} minutes, submitted={event.isoformat()}. The interval is elapsed time, not business time.'
            q=f'Classify the submission for {tag} relative to start plus the allowed interval. Respect the explicit UTC offsets.'
            scenarios.append((state,q,'before' if d<0 else 'after' if d>0 else 'at',{'start':base.isoformat(),'minutes':window,'submission':event.isoformat()}))
        elif family=='policy':
          # One counterfactual makes the final decision change; the task's three-valued rule is explicit.
          opts=[('allow','All permission conditions are established.'),('deny','A required permission condition is explicitly violated.'),('unknown','The necessary evidence is incomplete and there is no explicit disqualifier.')]
          def oracle(a,h,c):
            if a=='no':return 'deny'
            if h=='yes' and c=='no':return 'deny'
            if a=='yes' and (h=='no' or c=='yes'):return 'allow'
            return 'unknown'
          states=[(a,h,c) for a in ('yes','no','unrecorded') for h in ('yes','no','unrecorded') for c in ('yes','no','unrecorded')]
          s0=rng.choice(states)
          # Change exactly one field, not the rule or distractors.
          candidates=[s for s in states if sum(a!=b for a,b in zip(s0,s))==1 and oracle(*s)!=oracle(*s0)]
          s1=rng.choice(candidates)
          order=list(range(5));rng.shuffle(order)
          for s in (s0,s1):
            rec=[f'{tag}: active={s[0]}.',f'{tag}: hold={s[1]}.',f'{tag}: clearance={s[2]}.','Unrelated record MK: active=no, hold=yes, clearance=no.','Obsolete rule: a hold always blocks. This rule has been replaced.']
            state='\n'.join(f'Record {j+1}: {rec[i]}' for j,i in enumerate(order))
            q=f'Current rule for {tag}: allow only if active and either not on hold or cleared. Deny for known inactive status, or a known hold with known lack of clearance. Otherwise return unknown if required facts are unrecorded. Ignore obsolete rules.'
            scenarios.append((state,q,oracle(*s),{'active':s[0],'hold':s[1],'clearance':s[2],'support':[order.index(i) for i in range(3)]}))
        elif family=='join':
          k=rng.choice([4,5,6]);names=rng.sample(['Iris','Milo','Vera','Leon','Alma','Remy','Juno','Ezra','Nora','Otis','Tess','Hugo'],k)
          teams=rng.sample(['Tern','Reed','Moss','Dune','Cove','Pine','Fern','Ash'],k)
          pools=rng.sample(['amber','cobalt','sage','coral','plum','silver','ivory','ochre'],k)
          target=rng.randrange(k);old=rng.randrange(k);new=rng.choice([j for j in range(k) if j!=old])
          opts=[(p,f'Assign to destination {p}.') for p in pools]
          order=list(range(k*2));rng.shuffle(order)
          for idx in (old,new):
            member=[f'{name} belongs to {teams[idx if j==target else j]}.' for j,name in enumerate(names)]
            mapping=[f'Team {t} uses destination {p}.' for t,p in zip(teams,pools)]
            rec=member+mapping;state=' '.join(rec[i] for i in order)
            q=f'Follow the current person-to-team and team-to-destination mappings. Which destination is assigned to {names[target]}?'
            scenarios.append((state,q,pools[idx],{'member':names[target],'team':teams[idx],'destination':pools[idx],'support':[order.index(target),order.index(k+idx)]}))
        elif family=='probability':
          k=rng.choice([3,4,5,6]);labels=rng.sample(['blue','green','red','gold','black','white','orange','purple'],k)
          counts=[rng.randint(4,40) for _ in labels];edit_i=rng.randrange(k);more=rng.randint(10,45)
          opts=[(v,f'The selected item has color {v}.') for v in labels]
          for edit in (0,1):
            cc=counts.copy();cc[edit_i]+=edit*more;total=sum(cc);target={v:n/total for v,n in zip(labels,cc)}
            state=f'Warehouse {tag}, complete ready-to-ship inventory: '+', '.join(f'{v}={n}' for v,n in zip(labels,cc))+'. Quarantined stock is excluded: blue=99, green=37.'
            q=f'Select uniformly from the ready-to-ship inventory in warehouse {tag}. Return the probabilities of the listed colors, using the complete counts and excluding quarantined stock.'
            scenarios.append((state,q,target,{'counts':dict(zip(labels,cc)),'sum':total}))
        elif family=='intent':
          topics=[('cancel','cancellation','cancel my membership'),('refund','refunds','refund my latest purchase'),('address','address changes','change my shipping address'),('extend','extensions','extend my reservation'),('track','package tracking','track my current parcel')]
          chosen=rng.choice(topics);key,noun,request=chosen
          opts=[('none','No present request for any listed action; discussion, history, quotation, and hypothetical interest alone do not count.')]+[(k,f'The speaker currently requests action: {r}.') for k,n,r in topics]
          positives=[f'Please {request}. I am asking you to do this now.',f'After considering it, I have decided: {request}, please.',f'I previously declined. My instruction now is to {request}.']
          negatives=[f'Can you explain your policy on {noun}? Do not make any changes or take action.',f'My colleague said "{request}" yesterday. I am only quoting them; I am not making that request.',f'I had considered asking you to {request}, but decided not to. Please leave everything as it is.']
          scenarios=[(rng.choice(negatives),'Classify the present speaker\'s actionable request using the outcome definitions.', 'none',{'present_action':'none'}),(rng.choice(positives),'Classify the present speaker\'s actionable request using the outcome definitions.',key,{'present_action':key})]
        elif family=='entailment':
          person=rng.choice(['Ari','Bo','Cleo','Dara','Emil','Faye','Gus','Hana']);obj=rng.choice(['parcel','folder','ticket','crate']);color=rng.choice(['teal','violet','orange','white']);other=rng.choice([c for c in ['teal','violet','orange','white'] if c!=color])
          opts=[('supported','The claim follows from the passage.'),('contradicted','The passage establishes that the claim is false.'),('unknown','The passage establishes neither the claim nor its negation.')]
          claim=f'{person} has the {color} {obj}.'
          modes=rng.sample(['supported','contradicted','unknown'],2)
          for mode in modes:
            passage={'supported':f'{person} has exactly one {obj}. It is {color}.','contradicted':f'{person} has no {color} {obj}; their only {obj} is {other}.','unknown':f'{person} is waiting near a {color} {obj}. The owner is not identified.'}[mode]
            scenarios.append((passage,f'Assess only this claim against the supplied passage: {claim}',mode,{'relation':mode,'claim':claim}))
        elif family=='judge':
          n=rng.randint(3,14);price=rng.randint(12,60);fee=rng.randint(1,20);total=n*price+fee
          opts=[('correct','The candidate is correct in both method and result.'),('arithmetic','The method is correct but the arithmetic result is wrong.'),('method','The candidate applies the wrong operation or omits a required term.'),('unsupported','The passage does not supply enough information to determine the answer.')]
          variants=[(f'{n}*{price}+{fee}={total} cents.','correct'),(f'{n}*{price}+{fee}={total+rng.choice([-2,-1,1,2])} cents.','arithmetic'),(f'{n}*{price}={n*price} cents; the fee is ignored.','method')]
          picks=rng.sample(variants,2)
          for answer,label in picks:
            state=f'Order: {n} units at {price} cents each, plus a mandatory {fee}-cent fee. Candidate answer: {answer}'
            q='Judge this answer to the total-cost question. A correct answer must include the mandatory fee and use exact arithmetic.'
            scenarios.append((state,q,label,{'n':n,'price':price,'fee':fee,'candidate':answer,'error_type':label,'total':total}))
        order=list(range(len(opts)));rng.shuffle(order)
        ordered=[{'id':opts[j][0],'description':opts[j][1]} for j in order]
        for edit,(state,question,answer,oracle) in enumerate(scenarios):
            if split=='held':
                # Formatting shift only; reference semantics unchanged.
                state='Evidence record:\n'+state
                question='Use no unstated assumptions. '+question
            target=answer if isinstance(answer,dict) else {k:float(k==answer) for k,d in opts}
            expected=max(sorted(target),key=lambda k:target[k])
            rows.append({'id':digest([split,family,source,edit])[:20],'source':digest([split,family,source])[:20],'split':split,'family':family,'edit':edit,'state':state,'question':question,'options':ordered,'expected':expected,'target_probs':target,'oracle':oracle,'teacher_eligible':family in ('policy','join','intent','entailment')})
    return rows

def checks():
    sets={s:corpus(s) for s in ('train','held')};seen=set()
    for split,rows in sets.items():
      assert len(rows)==96 and len({r['id'] for r in rows})==96
      for r in rows:
        assert len(r['options']) in range(2,7)
        assert abs(sum(r['target_probs'].values())-1)<1e-12
        key=digest({k:r[k] for k in ('state','question','options')})
        assert key not in seen,('visible duplicate',r['id']);seen.add(key)
        assert not (set(visible(r)) & {'expected','oracle','target_probs','source','family','edit'})
      for source in {r['source'] for r in rows}:
        pair=[r for r in rows if r['source']==source];assert len(pair)==2
        assert pair[0]['options']==pair[1]['options']
        assert pair[0]['target_probs']!=pair[1]['target_probs']
        if pair[0]['family']!='probability':assert pair[0]['expected']!=pair[1]['expected']
    return {'counts':{k:len(v) for k,v in sets.items()},'source_counts':{k:len({r['source'] for r in v}) for k,v in sets.items()},'hashes':{k:digest(v) for k,v in sets.items()},'visible_overlap':0,'families':FAMILIES}

if __name__=='__main__':print(json.dumps(checks(),indent=2))
