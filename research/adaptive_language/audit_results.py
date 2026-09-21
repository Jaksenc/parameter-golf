"""Independent scoring, bounded scalar replay, and failure-inclusive aggregation."""
import ast,collections,hashlib,json,math,operator,re,statistics,sys
from datetime import datetime,timedelta
from fractions import Fraction
from pathlib import Path
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Mod:operator.mod}
CMPS={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
def scalar(expr,env):
    assert isinstance(expr,str) and len(expr)<=2048
    tree=ast.parse(expr,mode='eval');assert sum(1 for _ in ast.walk(tree))<=256
    def walk(n,depth=0):
        assert depth<=24
        w=lambda x:walk(x,depth+1)
        if isinstance(n,ast.Constant):
            if type(n.value) in (bool,str):return n.value
            assert type(n.value) in (int,float)
            literal=ast.get_source_segment(expr,n);assert len(literal)<=100 and math.isfinite(float(n.value));return Fraction(literal)
        if isinstance(n,ast.Name):return {'true':True,'false':False,**env}[n.id]
        if isinstance(n,(ast.List,ast.Tuple)):return [w(x) for x in n.elts]
        if isinstance(n,ast.UnaryOp):
            v=w(n.operand)
            if isinstance(n.op,ast.Not):return not v
            return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp):
            a,b=w(n.left),w(n.right)
            if isinstance(n.op,ast.BitXor):assert type(a) is bool and type(b) is bool;return a!=b
            if isinstance(n.op,ast.Pow):assert b.denominator==1 and abs(b)<=12;return a**int(b)
            if isinstance(n.op,ast.FloorDiv):return Fraction(a//b)
            return OPS[type(n.op)](a,b)
        if isinstance(n,ast.Compare):
            vals=[w(n.left)]+[w(x) for x in n.comparators]
            return all(CMPS[type(op)](a,b) for op,a,b in zip(n.ops,vals,vals[1:]))
        if isinstance(n,ast.BoolOp):
            vals=[w(x) for x in n.values];return all(vals) if isinstance(n.op,ast.And) else any(vals)
        if isinstance(n,ast.Subscript):return w(n.value)[int(w(n.slice))]
        if isinstance(n,ast.Call):
            assert isinstance(n.func,ast.Name) and not n.keywords
            args=[w(a) for a in n.args];name=n.func.id
            if name=='date_add':return (datetime.strptime(args[0],'%Y-%m-%d')+timedelta(days=int(args[1]))).strftime('%m/%d/%Y')
            if name=='count':return Fraction(args[0].count(args[1]))
            if name=='len':return Fraction(len(args[0]))
            if name=='abs':return abs(args[0])
            if name=='ceil':return Fraction(math.ceil(args[0]))
            if name=='floor':return Fraction(math.floor(args[0]))
            if name=='round':return Fraction(round(args[0]))
            vals=args[0] if len(args)==1 and isinstance(args[0],list) else args
            if name=='sum':return sum(vals,Fraction())
            if name=='min':return min(vals)
            if name=='max':return max(vals)
            if name=='sorted':return sorted(vals)
        raise AssertionError('Unrecognized accepted program')
    return walk(tree.body)
def serial(v):
    if isinstance(v,Fraction):return {'numerator':v.numerator,'denominator':v.denominator}
    if isinstance(v,list):return [serial(x) for x in v]
    return v
def replay(plan):
    assert isinstance(plan,dict)
    if set(plan)=={'answer'}:return None,[]
    assert set(plan)<={'result','steps'} and len(plan.get('steps',[]))<=32
    env={};trace=[]
    for step in plan.get('steps',[]):
        v=scalar(step['expr'],env);env[step['name']]=v;trace.append({'name':step['name'],'expr':step['expr'],'value':serial(v)})
    return serial(scalar(plan['result'],env)),trace
def judge(answer,reference,suite):
    if answer is None:return False
    def norm(s):return ' '.join(str(s).strip().lower().replace('**','').removesuffix('.').split())
    a,b=norm(answer),norm(reference)
    if suite=='GSM8K':
        try:return Fraction(a.replace(',','').removeprefix('$'))==Fraction(b.replace(',',''))
        except (ValueError,ZeroDivisionError):return False
    return a==b or (re.fullmatch(r'\(?[a-z]\)?',a) is not None and re.fullmatch(r'\(?[a-z]\)?',b) is not None and a.strip('()')==b.strip('()'))
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [c-r,c+r]
def aggregate(directory):
    files=sorted(Path(directory).glob('language-shard-*.json'));rows=[];manifest=[];cohort=set();seen=set();replays=0;steps=0;runids=set();setup=0
    for file in files:
        raw=file.read_bytes();d=json.loads(raw);assert d.get('status')=='completed',(file,d.get('error'))
        assert d['core']['decoded_sha256']=='6b37dcfc4a451d5dba399080ba362b5b5320b03e977bf49532aab2be3cd108ca'
        assert d['runtime']['weights']['sha256']=='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
        assert d['runtime']['runtime']['sha256']=='9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'
        cohort.add(d['cohort_sha256']);runids.add(d['run_id']);setup+=d['runtime']['setup_seconds']
        manifest.append({'name':file.name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        for r in d['rows']:
            key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
            assert judge(r['answer'],r['reference'],r['suite'])==r['correct']
            if r.get('question_sha256'):assert hashlib.sha256(r['question'].encode()).hexdigest()==r['question_sha256']
            calls=r.get('calls',[]);assert len(calls)<=(2 if r['arm']=='program' else 1)
            assert sum(c.get('usage',{}).get('completion_tokens',0) for c in calls)<=768
            for i,c in enumerate(calls):assert c.get('usage',{}).get('completion_tokens',0)<=(512 if i==0 and r['arm']=='program' else 256 if i else 768)
            ex=r.get('execution')
            if ex:
                plan=ex['plan'];h=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert h==ex['program_sha256']
                value,trace=replay(plan)
                if 'value' in ex:assert value==ex['value'] and trace==ex['trace'];replays+=1;steps+=len(trace)
                assert ex['answer']==r['answer']
            rows.append(r)
    assert len(files)==12 and len(rows)==258 and len(cohort)==1
    summary={}
    for suite in ['ALL','GSM8K','BBH']:
        sub=[r for r in rows if suite=='ALL' or r['suite']==suite];summary[suite]={}
        for arm in ['direct','reasoning','program']:
            rr=[r for r in sub if r['arm']==arm];n=len(rr);good=sum(r['correct'] for r in rr);times=[r['seconds'] for r in rr if r.get('seconds') is not None]
            summary[suite][arm]={'correct':good,'n':n,'accuracy':good/n,'wilson_descriptive_95':wilson(good,n),'answered':sum(r['answer'] is not None for r in rr),'statuses':dict(collections.Counter(r['status'] for r in rr)),'seconds':sum(times),'median_seconds':statistics.median(times),'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in rr),'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in rr),'calls':sum(len(r.get('calls',[])) for r in rr),'truncated_calls':sum(c.get('finish_reason')!='stop' for r in rr for c in r.get('calls',[])),'repaired_requests':sum(len(r.get('calls',[]))>1 for r in rr)}
    paired={}
    for suite in ['ALL','GSM8K','BBH']:
        sub=[r for r in rows if suite=='ALL' or r['suite']==suite];lookup={(r['id'],r['arm']):r for r in sub};ids=sorted({r['id'] for r in sub})
        for a,b in [('direct','reasoning'),('direct','program'),('reasoning','program')]:
            counts=collections.Counter()
            for i in ids:
                va,vb=lookup[i,a]['correct'],lookup[i,b]['correct'];counts['repaired' if not va and vb else 'broken' if va and not vb else 'both_correct' if va else 'both_wrong']+=1
            discord=counts['repaired']+counts['broken'];k=min(counts['repaired'],counts['broken']);p=min(1.,2*sum(math.comb(discord,j) for j in range(k+1))/2**discord) if discord else 1.
            paired[suite+':'+a+'->'+b]={**dict(counts),'mcnemar_exact_descriptive':p}
    family={}
    for r in rows:
        a=family.setdefault(r['family'],{}).setdefault(r['arm'],{'correct':0,'n':0});a['correct']+=r['correct'];a['n']+=1
    compact=[{k:r.get(k) for k in ['id','arm','suite','family','answer','reference','correct','status','seconds','usage']} for r in rows]
    return {'status':'audited','task_count':len({r['id'] for r in rows}),'attempts':len(rows),'summary':summary,'paired':paired,'by_family':family,'source_manifest':manifest,'run_ids':sorted(runids),'programs_independently_replayed':replays,'steps_independently_replayed':steps,'setup_seconds_sum':setup,'source_weight_consistency':True,'no_frontier_comparison':True,'scope':'Public zero-shot pilot; correlated BBH families; descriptive intervals/tests, no multiplicity or contamination correction; system rather than weight improvement.','compact_rows':compact}
if __name__=='__main__':
    result=aggregate(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('compact_rows','by_family','source_manifest')},indent=2))
