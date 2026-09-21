"""A bounded expression interpreter. Never uses eval/exec or opens resources.
Execution validity and source-constant checks DO NOT prove semantic correctness.
"""
from __future__ import annotations
import ast, datetime as dt, fractions, json, math, operator, re
from collections import deque
F=fractions.Fraction
STRUCTURAL={F(x) for x in (-1,0,1,2,3,4,5,7,10,12,24,60,100,360,365,1000)}

class Rejected(ValueError): pass

def clean(code):
    code=code.strip()
    if code.startswith('```'):
        code=re.sub(r'^```(?:python|py)?\s*','',code)
        code=code.split('```',1)[0].strip()
    if len(code)>6000: raise Rejected('program too long')
    return code

def serial(x):
    if isinstance(x,F): return x.numerator if x.denominator==1 else f'{x.numerator}/{x.denominator}'
    if isinstance(x,(int,bool,str)) or x is None:return x
    if isinstance(x,float):
        if not math.isfinite(x):raise Rejected('nonfinite value')
        return str(x)
    if isinstance(x,(tuple,list,range)):return [serial(y) for y in x]
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    raise Rejected('unsupported value type')

def integer(x):
    n=int(x)
    if n!=x:raise Rejected('integer required')
    return n

def bounded_range(*args):
    r=range(*(integer(x) for x in args))
    if len(r)>256:raise Rejected('range too large')
    return list(r)

def clock(s):
    p=str(s).split(':');h,m=map(int,p)
    if not 0<=h<24 or not 0<=m<60:raise Rejected('invalid clock')
    return h*60+m

def hhmm(n):
    n=integer(n)
    if n<0 or n>=1440*32:raise Rejected('invalid clock result')
    return f'{n//60%24:02d}:{n%60:02d}'+(f' (+{n//1440}d)' if n>=1440 else '')

def days(a,b):return (dt.date.fromisoformat(str(b))-dt.date.fromisoformat(str(a))).days

def follow(mapping,start,n):
    n=integer(n)
    if not isinstance(mapping,dict) or not 0<=n<=256:raise Rejected('invalid traversal')
    cur=start
    for _ in range(n):cur=mapping[cur]
    return cur

def affine(x,a,b,m,n):
    x,a,b,m,n=map(integer,(x,a,b,m,n))
    if not 1<=m<=1000000 or not 0<=n<=256:raise Rejected('invalid recurrence')
    for _ in range(n):x=(a*x+b)%m
    return x

def reachable(edges,start,end):
    if isinstance(edges,dict):edges=[(a,b) for a,bs in edges.items() for b in (bs if isinstance(bs,list) else [bs])]
    if not isinstance(edges,list) or len(edges)>256:raise Rejected('invalid graph')
    q=deque([start]);seen={start}
    while q:
        x=q.popleft()
        if x==end:return True
        for a,b in edges:
            if a==x and b not in seen:seen.add(b);q.append(b)
        if len(seen)>256:raise Rejected('graph too large')
    return False

def choose(n,k):
    n,k=integer(n),integer(k)
    if not 0<=k<=n<=256:raise Rejected('invalid combination')
    return math.comb(n,k)

FUNCS={'sum':sum,'len':len,'min':min,'max':max,'abs':abs,'all':all,'any':any,'sorted':sorted,
       'range':bounded_range,'enumerate':lambda x:list(enumerate(x)), 'zip':lambda *x:list(zip(*x)),
       'int':integer,'str':str,'round':lambda x,n=0:round(x,integer(n)),'days':days,'clock':clock,'hhmm':hhmm,'follow':follow,
       'affine':affine,'reachable':reachable,'comb':choose}
BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
     ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
CMP={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,
     ast.GtE:operator.ge,ast.In:lambda a,b:a in b,ast.NotIn:lambda a,b:a not in b}

class VM:
    def __init__(self,code,state,question):
        self.code=clean(code);self.trace=[];self.reads=[];self.steps=0;self.ops=0
        text=json.dumps({'S':state,'Q':question},ensure_ascii=False)
        def exact(x):
            if isinstance(x,float):return F(str(x))
            if isinstance(x,list):return [exact(y) for y in x]
            if isinstance(x,dict):return {k:exact(v) for k,v in x.items()}
            return x
        self.env={'S':exact(state),'Q':exact(question)}
        self.allowed=STRUCTURAL|{F(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',text)}
        self.allowed|={abs(x) for x in self.allowed}
        self.literal_checks=[]
    def check(self,v):
        if isinstance(v,int) and v.bit_length()>4096:raise Rejected('integer overflow budget')
        if isinstance(v,F) and max(v.numerator.bit_length(),v.denominator.bit_length())>4096:raise Rejected('fraction overflow budget')
        if isinstance(v,(str,list,tuple,dict,range)) and len(v)>4096:raise Rejected('container budget')
        if isinstance(v,float) and not math.isfinite(v):raise Rejected('nonfinite')
        return v
    def record(self,node,v,read=False):
        self.ops+=1
        if len(self.trace)<40:self.trace.append({'expression':ast.get_source_segment(self.code,node),'value':serial(v)})
        if read:self.reads.append(ast.get_source_segment(self.code,node))
        return v
    def bind(self,target,value,env):
        if isinstance(target,ast.Name) and target.id not in ('S','Q') and not target.id.startswith('_'):env[target.id]=value
        elif isinstance(target,(ast.Tuple,ast.List)) and len(target.elts)==len(value):
            for a,b in zip(target.elts,value):self.bind(a,b,env)
        else:raise Rejected('invalid comprehension binding')
    def walk(self,n,env=None,depth=0):
        env=self.env if env is None else env;self.steps+=1
        if self.steps>10000 or depth>45:raise Rejected('execution budget')
        w=lambda x,e=env:self.walk(x,e,depth+1)
        if isinstance(n,ast.Constant):
            v=n.value
            if type(v) in (int,float):
                v=F(str(v));ok=v in self.allowed;self.literal_checks.append({'literal':str(v),'allowed':ok})
                if not ok:raise Rejected('numeric literal absent from input and structural whitelist')
            elif not isinstance(v,(str,bool)) and v is not None:raise Rejected('invalid literal')
        elif isinstance(n,ast.Name):
            if n.id not in env:raise Rejected('unknown variable '+n.id)
            v=env[n.id]
        elif isinstance(n,(ast.List,ast.Tuple)):v=[w(x) for x in n.elts]
        elif isinstance(n,ast.Dict):
            if any(x is None for x in n.keys):raise Rejected('dictionary expansion not allowed')
            v={w(k):w(x) for k,x in zip(n.keys,n.values)}
        elif isinstance(n,ast.Subscript):
            a,b=w(n.value),w(n.slice)
            if isinstance(b,F):b=integer(b)
            v=self.record(n,a[b],True)
        elif isinstance(n,ast.Slice):v=slice(*(integer(w(x)) if x is not None else None for x in (n.lower,n.upper,n.step)))
        elif isinstance(n,ast.BinOp):
            if type(n.op) not in BIN:raise Rejected('operator not allowed')
            a,b=w(n.left),w(n.right)
            if isinstance(n.op,ast.Pow) and (abs(b)>32 or integer(b)!=b):raise Rejected('power budget')
            if isinstance(n.op,ast.Mult) and (isinstance(a,(list,str)) or isinstance(b,(list,str))):raise Rejected('sequence multiplication prohibited')
            if type(a) in (int,float):a=F(str(a))
            if type(b) in (int,float):b=F(str(b))
            v=self.record(n,BIN[type(n.op)](a,b))
        elif isinstance(n,ast.UnaryOp):
            a=w(n.operand)
            if isinstance(n.op,ast.USub):v=-a
            elif isinstance(n.op,ast.UAdd):v=+a
            elif isinstance(n.op,ast.Not):v=not a
            else:raise Rejected('unary operator prohibited')
        elif isinstance(n,ast.Compare):
            a=w(n.left);v=True
            for op,bn in zip(n.ops,n.comparators):
                if type(op) not in CMP:raise Rejected('comparison prohibited')
                b=w(bn)
                if not CMP[type(op)](a,b):v=False;break
                a=b
            self.record(n,v)
        elif isinstance(n,ast.BoolOp):
            v=w(n.values[0])
            for x in n.values[1:]:
                if isinstance(n.op,ast.And) and not v:break
                if isinstance(n.op,ast.Or) and v:break
                v=w(x)
        elif isinstance(n,ast.IfExp):v=w(n.body) if w(n.test) else w(n.orelse)
        elif isinstance(n,(ast.ListComp,ast.GeneratorExp,ast.DictComp)):
            rows=[]
            def rec(i,e):
                if i==len(n.generators):
                    rows.append((w(n.key,e),w(n.value,e)) if isinstance(n,ast.DictComp) else w(n.elt,e));return
                g=n.generators[i]
                if g.is_async:raise Rejected('async prohibited')
                iterable=w(g.iter,e)
                if len(iterable)>256:raise Rejected('comprehension budget')
                for x in iterable:
                    ee=e.copy();self.bind(g.target,x,ee)
                    if all(w(c,ee) for c in g.ifs):rec(i+1,ee)
                    if len(rows)>256:raise Rejected('comprehension output budget')
            rec(0,env.copy());v=dict(rows) if isinstance(n,ast.DictComp) else rows
        elif isinstance(n,ast.Call):
            if not isinstance(n.func,ast.Name) or n.func.id not in FUNCS or n.keywords:raise Rejected('call prohibited')
            args=[w(x) for x in n.args]
            v=self.record(n,FUNCS[n.func.id](*args))
        else:raise Rejected('syntax prohibited: '+type(n).__name__)
        return self.check(v)
    def run(self):
        tree=ast.parse(self.code,mode='eval')
        if not isinstance(tree.body,ast.Dict):raise Rejected('return a dictionary expression')
        if len(list(ast.walk(tree)))>1000:raise Rejected('AST budget')
        value=self.walk(tree.body)
        if not value:return {'accepted':False,'reason':'empty_program','code':self.code}
        if not self.ops:raise Rejected('no computation or source access')
        output=serial(value)
        if len(json.dumps(output))>6000:raise Rejected('result too large')
        return {'accepted':True,'code':self.code,'result':output,'trace':self.trace,'source_accesses':sorted(set(self.reads)),
                'numeric_literal_checks':self.literal_checks,'steps':self.steps,
                'scope':'Exact execution of proposed expression; not verification of interpretation or final answer.'}

def execute(code,state,question):
    try:return VM(code,state,question).run()
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ZeroDivisionError,SyntaxError,RecursionError) as e:
        return {'accepted':False,'reason':type(e).__name__+': '+str(e)[:200],'code':str(code)[:6000]}
