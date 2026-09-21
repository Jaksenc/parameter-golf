"""Partial semantic contracts: quoted quantities, dimensional algebra, exact execution.

Checks literal support and declared unit consistency, NOT natural-language truth.
The interpreter never executes Python code. Previously evaluated core is unchanged.
"""
from __future__ import annotations
import ast
from fractions import Fraction
import re
from typing import Any
from adaptive_language.program import Interpreter, ProgramError, bounded, render, serial, sha, parse_plan

VERSION = 'quantity-contracts-0.8.0'
NAME = re.compile(r'[a-z][a-z0-9_]{0,31}')
SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(21)))
SMALL.update({'half': Fraction(1,2), 'quarter': Fraction(1,4), 'third': Fraction(1,3), 'twice': 2, 'double': 2, 'dozen': 12, 'couple': 2})
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20,100,10)))
ALIASES = {'dollar':'usd','dollars':'usd','cent':'cent','cents':'cent','hours':'hour','minutes':'minute','seconds':'second','days':'day','weeks':'week','years':'year','items':'item','pieces':'piece','boxes':'box','people':'person','persons':'person','miles':'mile','feet':'foot','pounds':'pound','liters':'liter'}
CONSTANTS = {'constant:one': ('1','1'), 'constant:two':('2','1'), 'constant:percent':('100','1'), 'conversion:minutes_per_hour':('60','minute/hour'), 'conversion:hours_per_day':('24','hour/day'), 'conversion:days_per_week':('7','day/week'), 'conversion:cents_per_dollar':('100','cent/usd'), 'conversion:dozen':('12','1')}


def normalized(text: str) -> str:
    return ' '.join(text.casefold().split())


def literal_values(text: str) -> set[Fraction]:
    """Conservative recognition of numeric tokens and small English quantities.

    Does not infer which entity a literal describes. A longer quote may contain
    several numbers. Unrecognized word expressions fail closed rather than guess.
    """
    out: set[Fraction] = set()
    for m in re.finditer(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?!\w|\.\d)', text):
        s=m.group().replace(',','')
        if len(s)>100: continue
        try: out.add(Fraction(s))
        except (ValueError,ZeroDivisionError): pass
    words=re.findall('[a-z]+',text.lower())
    for i,w in enumerate(words):
        if w in SMALL: out.add(Fraction(SMALL[w]))
        if w in TENS:
            n=TENS[w]
            if i+1<len(words) and words[i+1] in SMALL and isinstance(SMALL[words[i+1]],int) and SMALL[words[i+1]]<10:
                out.add(Fraction(n+SMALL[words[i+1]]))
            out.add(Fraction(n))
    return out


def units(source: str) -> dict[str,int]:
    if not isinstance(source,str) or not 1<=len(source)<=96: raise ProgramError('unit_size')
    if source=='1': return {}
    try: tree=ast.parse(source,mode='eval')
    except SyntaxError as e: raise ProgramError('invalid_unit') from e
    if sum(1 for _ in ast.walk(tree))>32: raise ProgramError('unit_limit')
    def walk(n,depth=0):
        if depth>10: raise ProgramError('unit_depth')
        if isinstance(n,ast.Name) and NAME.fullmatch(n.id): return {ALIASES.get(n.id,n.id):1}
        if isinstance(n,ast.Constant) and type(n.value) is int and n.value==1: return {}
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.Div)):
            return combine(walk(n.left,depth+1),walk(n.right,depth+1),-1 if isinstance(n.op,ast.Div) else 1)
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow):
            k=None
            if isinstance(n.right,ast.Constant) and type(n.right.value) is int: k=n.right.value
            elif isinstance(n.right,ast.UnaryOp) and isinstance(n.right.op,ast.USub) and isinstance(n.right.operand,ast.Constant) and type(n.right.operand.value) is int: k=-n.right.operand.value
            if k is not None and abs(k)<=6: return {u:p*k for u,p in walk(n.left,depth+1).items() if p*k}
        raise ProgramError('invalid_unit_syntax')
    return walk(tree.body)


def combine(a:dict[str,int],b:dict[str,int],sign:int=1)->dict[str,int]:
    out=dict(a)
    for k,v in b.items(): out[k]=out.get(k,0)+sign*v
    out={k:v for k,v in out.items() if v}
    if len(out)>8 or any(abs(v)>24 for v in out.values()): raise ProgramError('unit_complexity_limit')
    return out


def infer(source:str,env:dict[str,dict[str,int]],values:Interpreter)->dict[str,int]:
    try: tree=ast.parse(source,mode='eval')
    except (SyntaxError,RecursionError) as e: raise ProgramError('invalid_expression') from e
    if sum(1 for _ in ast.walk(tree))>256: raise ProgramError('ast_limit')
    def walk(n,depth=0):
        if depth>24: raise ProgramError('computation_limit')
        if isinstance(n,ast.Name):
            if n.id not in env: raise ProgramError('unknown_variable:'+n.id)
            return env[n.id]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            if n.value not in (0,1): raise ProgramError('declare_quantity_before_use:'+str(n.value))
            return {}
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return walk(n.operand,depth+1)
        if isinstance(n,ast.BinOp):
            a,b=walk(n.left,depth+1),walk(n.right,depth+1)
            if isinstance(n.op,(ast.Add,ast.Sub,ast.Mod)):
                if a!=b: raise ProgramError('incompatible_units:'+str(a)+' vs '+str(b))
                return a
            if isinstance(n.op,(ast.Mult,ast.Div,ast.FloorDiv)): return combine(a,b,1 if isinstance(n.op,ast.Mult) else -1)
            if isinstance(n.op,ast.Pow):
                if b: raise ProgramError('dimensioned_exponent')
                k=values.expression(ast.get_source_segment(source,n.right))
                if not isinstance(k,Fraction) or k.denominator!=1 or abs(k)>12: raise ProgramError('integer_exponent_required')
                return {u:p*int(k) for u,p in a.items() if p*k}
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:
            name=n.func.id
            if name in ('ceil','floor','abs','round') and len(n.args)==1: return walk(n.args[0],depth+1)
            if name in ('min','max') and n.args:
                us=[walk(x,depth+1) for x in n.args]
                if any(u!=us[0] for u in us): raise ProgramError('incompatible_function_units')
                return us[0]
        raise ProgramError('quantity_syntax_not_supported:'+type(n).__name__)
    return walk(tree.body)


def execute_contract(question:str,plan:dict[str,Any])->dict[str,Any]:
    if not isinstance(question,str) or not 1<=len(question)<=40000: raise ProgramError('question_size')
    if not isinstance(plan,dict): raise ProgramError('object_required')
    if set(plan)=={'answer'}:
        from adaptive_language.program import execute
        r=execute(plan);r['contract_checked']=False;r['semantic_verification']=False;return r
    expected={'target','quantities','steps','result','answer_unit'}
    if set(plan)!=expected: raise ProgramError('contract_fields:'+','.join(sorted(expected)))
    target=plan['target']
    if not isinstance(target,str) or not 4<=len(target)<=300 or normalized(target) not in normalized(question): raise ProgramError('target_not_quoted_from_question')
    qs=plan['quantities'];ss=plan['steps']
    if not isinstance(qs,list) or not 1<=len(qs)<=24 or not isinstance(ss,list) or len(ss)>24: raise ProgramError('quantity_or_step_limit')
    interpreter=Interpreter();env={};bindings=[];trace=[];sources=[]
    def check_name(name):
        if not isinstance(name,str) or not NAME.fullmatch(name) or name in env or name in ('true','false'): raise ProgramError('variable_name')
    for q in qs:
        if not isinstance(q,dict) or set(q)!={'name','value','unit','source'}: raise ProgramError('quantity_schema')
        name=q['name'];check_name(name)
        text=q['value'];src=q['source']
        if not isinstance(text,str) or len(text)>100 or not re.fullmatch(r'[-+]?(?:[0-9,]+(?:\.[0-9]+)?|\.[0-9]+)(?:/[0-9]+)?',text): raise ProgramError('quantity_value_string_required')
        try: value=bounded(Fraction(text.replace(',','')))
        except (ValueError,ZeroDivisionError) as e: raise ProgramError('invalid_quantity_value:'+name) from e
        unit=units(q['unit'])
        if not isinstance(src,str) or not 1<=len(src)<=180: raise ProgramError('source_size:'+name)
        origin='quoted_literal'
        if src in CONSTANTS:
            literal,u=CONSTANTS[src]
            if value!=Fraction(literal) or unit!=units(u): raise ProgramError('invalid_known_constant:'+name)
            origin='declared_constant'
        else:
            if normalized(src) not in normalized(question): raise ProgramError('source_not_in_question:'+name)
            if value not in literal_values(src): raise ProgramError('literal_not_supported_by_quote:'+name)
            sources.append(src)
        interpreter.env[name]=value;env[name]=unit
        bindings.append({'name':name,'value':serial(value),'unit':unit,'source':src,'origin':origin})
    for step in ss:
        if not isinstance(step,dict) or set(step)!={'name','expr'}: raise ProgramError('step_schema')
        name=step['name'];check_name(name);expr=step['expr']
        if not isinstance(expr,str) or not 1<=len(expr)<=2048: raise ProgramError('expression_size')
        try:
            unit=infer(expr,env,interpreter);value=interpreter.expression(expr)
        except ProgramError as e: raise ProgramError('step:'+name+':'+str(e)) from e
        if not isinstance(value,Fraction): raise ProgramError('numeric_contract_required')
        interpreter.env[name]=value;env[name]=unit
        trace.append({'name':name,'expr':expr,'value':serial(value),'unit':unit})
    expr=plan['result']
    if not isinstance(expr,str) or not 1<=len(expr)<=2048: raise ProgramError('expression_size')
    unit=infer(expr,env,interpreter)
    if unit!=units(plan['answer_unit']): raise ProgramError('result_unit_mismatch:'+str(unit))
    value=interpreter.expression(expr)
    if not isinstance(value,Fraction): raise ProgramError('numeric_contract_required')
    all_uses=set()
    for expression in [s['expr'] for s in ss]+[expr]:
        all_uses.update(n.id for n in ast.walk(ast.parse(expression,mode='eval')) if isinstance(n,ast.Name))
    warnings=['unused_quantity:'+q['name'] for q in qs if q['name'] not in all_uses]
    return {'answer':render(value),'value':serial(value),'status':'literal_and_unit_checked_semantics_unverified','contract_checked':True,'semantic_verification':False,'target':target,'bindings':bindings,'trace':trace,'unit':unit,'warnings':warnings,'program_sha256':sha(plan),'question_sha256':sha(question),'contract_version':VERSION}
