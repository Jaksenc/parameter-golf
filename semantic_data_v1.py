"""Controlled semantic-localization pilot. Synthetic, NOT independent human data.
Model inputs contain only rendered evidence, rules, and candidate definitions.
T/F/U use strong-Kleene truth; unknown also represents unresolved equal-source conflict.
"""
from __future__ import annotations
import hashlib, itertools, json, random
from collections import Counter

OPS=['and','or','xor','same','except','neither','implies','not_both']
ASTS=[['and','p','q'],['or','p','q'],['xor','p','q'],['same','p','q'],['and','p',['not','q']],['not',['or','p','q']],['or',['not','p'],'q'],['not',['and','p','q']]]
FORMULAS=['p AND q','p OR q','p XOR q','p EQUIVALENT q','p AND (NOT q)','NOT (p OR q)','(NOT p) OR q','NOT (p AND q)']
OUTCOMES=['Apply the adjustment to the target only.','Leave the target unchanged.','Request clarification before changing anything.','Apply the adjustment to the other object only.','Apply the adjustment to every object.','Delete the target object.']
TRUTH=[1,0,-1]
TRUTH_WORDS={1:'true',0:'false',-1:'unknown'}
CONTEXTS=[('selected','locked'),('tagged','pinned'),('featured','hidden')]

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
def negate(x):return -1 if x==-1 else 1-x

def evaluate(tree, facts):
    if isinstance(tree,str):return facts[tree]
    op=tree[0];a=evaluate(tree[1],facts)
    if op=='not':return negate(a)
    b=evaluate(tree[2],facts)
    if op=='and':return 0 if 0 in (a,b) else (1 if a==b==1 else -1)
    if op=='or':return 1 if 1 in (a,b) else (0 if a==b==0 else -1)
    if op in ('xor','same'):
        return -1 if -1 in (a,b) else int((a!=b) if op=='xor' else (a==b))
    raise ValueError('Unsupported operator')

def independent_reference(op,p,q):
    functions={'and':lambda p,q:p and q,'or':lambda p,q:p or q,'xor':lambda p,q:p!=q,'same':lambda p,q:p==q,'except':lambda p,q:p and not q,'neither':lambda p,q:not(p or q),'implies':lambda p,q:(not p) or q,'not_both':lambda p,q:not(p and q)}
    possible={int(functions[op](a,b)) for a in ([False,True] if p==-1 else [bool(p)]) for b in ([False,True] if q==-1 else [bool(q)])}
    return possible.pop() if len(possible)==1 else -1

def wording(op,p,q,version):
    if version==0:
        forms=[f'the target is both {p} and {q}',f'the target is {p} or {q} or both',f'exactly one of {p} and {q} holds for the target',f'the target is both {p} and {q}, or neither',f'the target is {p} but not {q}',f'the target is neither {p} nor {q}',f'the target is not {p}, or is {q}, or both',f'the target is not both {p} and {q}']
    elif version==1:
        forms=[f'the target is {p}, and also {q}',f'at least one of {p} and {q} holds for the target',f'the target is either {p} or {q}, but not both',f"the target's {p} and {q} statuses have the same truth value",f'the target is {p}, with {q} as an exclusion',f'the target is not {p} and is not {q}',f'the target is either not {p} or {q}',f'at least one of not {p} and not {q} holds for the target']
    else:
        forms=[f'the target is {p} as well as {q}',f'one or both of {p} and {q} holds for the target',f'one, but only one, of {p} and {q} holds for the target',f"the target's {p} status agrees in truth value with its {q} status",f'the target meets the {p} criterion and does not meet the {q} criterion',f'none of these holds for the target: {p}; {q}',f'the target satisfies the disjunction of not {p} and {q}',f'it is false that the target is both {p} and {q}']
    return 'Adjust the target if and only if '+forms[OPS.index(op)]+'.'

def make_case(split,op,p,q,context,version,serial):
    a,b=CONTEXTS[context]; target='Item-'+digest({'namespace':split,'serial':serial})[:6];other='Other-'+digest({'other':split,'serial':serial})[:6]
    def statement(prop,v):
        if v==-1:
            if serial%2:
                return f'Equally current records disagree: {target} is {prop}; {target} is not {prop}.'
            return f'The {prop} status of {target} is not reported.'
        if version==0:return f'{target} is '+('' if v else 'not ')+prop+'.'
        if version==1:return f'Record for {target}: {prop} is '+('confirmed' if v else 'explicitly ruled out')+'.'
        return f'The review '+('confirms' if v else 'denies')+f' that {target} is {prop}.'
    statements=[statement(a,p),statement(b,q),f'{other} is {a} and {b}. Do not edit {other}.']
    rng=random.Random(int(digest({'split':split,'serial':serial,'order':17})[:12],16));rng.shuffle(statements)
    evidence=f'Target: {target}. '+ ' '.join(statements)
    rule=wording(op,a,b,version)
    facts={'p':p,'q':q};val=evaluate(ASTS[OPS.index(op)],facts);meaning={1:0,0:1,-1:2}[val]
    order=list(range(6));rng.shuffle(order)
    return {'id':f'{split}-{serial:03d}','split':split,'group':f'{split}-{context}','operator':op,'facts':facts,'properties':[a,b],'target_name':target,'evidence':evidence,'rule':rule,'ast':ASTS[OPS.index(op)],'truth':val,'action':meaning,'order':order,'answer':order.index(meaning),'context':context}

def dataset():
    rows=[];states=list(itertools.product(TRUTH,repeat=2))
    for split,version,ops in [('train',0,OPS[:6]),('development',1,OPS[:6]),('wording',2,OPS[:6]),('composition',2,OPS[6:])]:
        serial=0
        for j,op in enumerate(ops):
            chosen=range(9) if split!='development' else [(j+k*3)%9 for k in range(3)]
            for si in chosen:
                p,q=states[si];context=(j+si)%3
                rows.append(make_case(split,op,p,q,context,version,serial));serial+=1
    return rows

POLICY='Use only the target evidence. Unreported or conflicting facts are unknown, not false. AND is false if any operand is false; OR is true if any operand is true. Otherwise unknown propagates. If the condition is true adjust the target only; if false leave it unchanged; if unknown request clarification.'

def prompt_task(row,mode):
    a,b=row['properties'];common=f"Evidence:\n{row['evidence']}\nInstruction:\n{row['rule']}\nPolicy:\n{POLICY}"
    if mode=='direct':
        return {'content':common,'options':[OUTCOMES[j] for j in row['order']]},row['answer']
    if mode.startswith('fact_'):
        atom=mode[-1];prop=a if atom=='p' else b
        return {'content':f"Evidence:\n{row['evidence']}\nFor target {row['target_name']} only, is {prop} true, explicitly false, or unknown? Equal-source conflicts count as unknown.",'options':['True.','False.','Unknown.']},TRUTH.index(row['facts'][atom])
    if mode=='rule':
        return {'content':f"Instruction:\n{row['rule']}\np means the target is {a}. q means the target is {b}. Which expression matches the instruction's condition? XOR means exactly one; EQUIVALENT means equal truth values.",'options':FORMULAS},OPS.index(row['operator'])
    if mode=='execute':
        return {'content':f"Given p={TRUTH_WORDS[row['facts']['p']]}, q={TRUTH_WORDS[row['facts']['q']]}, evaluate {FORMULAS[OPS.index(row['operator'])]}. {POLICY}",'options':['True.','False.','Unknown.']},TRUTH.index(row['truth'])
    raise ValueError(mode)

def tests():
    for op,ast in zip(OPS,ASTS):
        for p,q in itertools.product(TRUTH,repeat=2):assert evaluate(ast,dict(p=p,q=q))==independent_reference(op,p,q)
    rows=dataset();assert Counter(r['split'] for r in rows)=={'train':54,'development':18,'wording':54,'composition':18}
    assert len({r['id'] for r in rows})==144
    assert len({r['evidence'] for r in rows})==144
    for row in rows:
        assert row['order'][row['answer']]=={1:0,0:1,-1:2}[row['truth']]
        p,t=prompt_task(row,'direct');assert 0<=t<len(p['options'])
        assert row['operator'] not in ('implies','not_both') or row['split']=='composition'
        for mode in ['fact_p','fact_q','rule','execute']:
            p,t=prompt_task(row,mode);assert 0<=t<len(p['options'])
    return {'truth_tables':72,'cases':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),'gold_actions':{s:dict(Counter(r['action'] for r in rows if r['split']==s)) for s in ['train','development','wording','composition']},'data_hash':digest(rows),'limitations':['synthetic controlled wording, 3 correlated property contexts per split','No independent human adjudication or real customer inputs','No free-form rule parser; rule-stage candidates are a closed library','Composition holds out formula combinations, not primitive NOT/AND/OR operations']}
if __name__=='__main__':print(json.dumps(tests(),sort_keys=True))
