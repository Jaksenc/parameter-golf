"""Frozen, comparable prompts. No reference answers enter the solver."""
from __future__ import annotations
import hashlib
import time
from adaptive_language.engine import extract_final, TransportError
from adaptive_language.program import ProgramError, parse_plan, execute
from .contracts import execute_contract

REASON = ('Solve the task carefully. Use concise reasoning to identify the quantities, entities, '
          'relationships, exceptions, and requested result. Check your interpretation before '
          'calculating. Finish with FINAL: <the requested answer>. Match the answer type asked '
          'for; never invent answer choices. Do not leave the final answer unfinished.')
PROGRAM = '''Translate the task into a small exact executable plan when useful. Return ONE JSON object, no prose.
Distinguish entities, what each quantity measures, units/time basis, and the requested relationship. Use descriptive lowercase variable names, not single-letter aliases. Keep groups, per-unit and total quantities distinct. Preserve qualifiers such as half-completed, remaining, already included, and changes over time.
Format: {"steps":[{"name":"boxes","expr":"6"},{"name":"items_per_box","expr":"11"},{"name":"returned","expr":"5"}],"result":"boxes*items_per_box-returned"}.
Every expr and result is an expression STRING. Previous named steps are available. Compute derived quantities rather than guess their values. Prefer at most 12 concise steps.
Supported: + - * / // % **, parentheses, comparisons, True False, and or not, boolean ^, flat lists, list indexing, min,max,sum,len,sorted,abs,ceil,floor,round (nearest-even integer), count(list,value), date_add("YYYY-MM-DD",integer_days) returning MM/DD/YYYY. Arithmetic is exact rational. Integer exponents must have absolute value <=12. No Python code, imports, loops, attributes, dictionaries or comprehensions inside expressions.
If these operations do not help, reason internally and return {"answer":"the requested answer"}. Match the answer type requested by the question; never invent answer choices. Execution checks computation, not interpretation.'''
CONTRACT = '''Return ONE JSON object describing a source-linked quantity calculation. No prose.
First identify what the question asks and preserve every relevant qualifier: completed fraction, remaining amount, per-unit versus total, changed price, or different entity. Use this schema:
{"target":"exact quote of the requested quantity from the question","quantities":[{"name":"descriptive_name","value":"numeric literal or fraction","unit":"item","source":"short exact quote containing that quantity"}],"steps":[{"name":"derived_name","expr":"expression using previously declared names"}],"result":"expression","answer_unit":"item"}
Source quotes must appear in the question and contain the declared number (small English numbers, half, quarter, twice and dozen are recognized). Use singular consistent units such as usd, hour, item, piece, item/box or usd/hour. Dimensionless counts/fractions use unit "1". Multiplication/division combines/cancels units; addition requires equal units. The result must have answer_unit. Do not confuse mathematical unit consistency with correct entity relationships.
Declare raw numbers in quantities. Use named quantities for derived expressions; only literals 0 and 1 can appear directly in steps. Supported expression operators: + - * / // % **, parentheses, min,max,abs,ceil,floor,round. No loops, imports or arbitrary Python. Names: lowercase identifiers <=32 characters; at most 24 quantities/steps. Prefer a short plan.
A needed mathematical 2 or 100 can use source "constant:two" or "constant:percent", unit "1". Known conversion sources: "conversion:minutes_per_hour" value "60" unit "minute/hour"; "conversion:hours_per_day" value "24" unit "hour/day"; "conversion:days_per_week" value "7" unit "day/week"; "conversion:cents_per_dollar" value "100" unit "cent/usd". Never label an invented task fact as a constant.
For nonnumerical tasks or when these operations are unsuitable, reason internally and return {"answer":"the requested answer"}. Do not invent answer choices. Checks verify quoted literals and declared unit algebra, not the correctness of your interpretation.'''
ARMS=('reasoning','program','contract')


def solve(question,arm,transport):
    if not isinstance(question,str) or not 1<=len(question)<=40000: raise ValueError('question_size')
    if arm not in ARMS: raise ValueError('unknown_arm')
    system={'reasoning':REASON,'program':PROGRAM,'contract':CONTRACT}[arm]
    messages=[{'role':'system','content':system},{'role':'user','content':question}]
    calls=[];errors=[];receipts=[];answer=None;execution=None;status='no_answer';t=time.perf_counter()
    try:
        response=transport(messages,1536 if arm=='reasoning' else 1024,arm!='reasoning');calls.append(response)
        if arm=='reasoning':
            answer=extract_final(response['text']) if response['finish_reason']=='stop' else None
            status='model_answer_unverified' if answer else 'incomplete_or_missing_answer'
        else:
            for i in range(2):
                try:
                    if response['finish_reason']!='stop': raise ProgramError('incomplete_generation')
                    plan=parse_plan(response['text'])
                    execution=execute_contract(question,plan) if arm=='contract' else execute(plan)
                    execution['plan']=plan
                    answer=execution['answer'];status=execution['status'];receipts.append({'attempt':i,'accepted':True});break
                except ProgramError as e:
                    errors.append(str(e));receipts.append({'attempt':i,'accepted':False,'error':str(e)})
                    if i==1: status='program_failed';break
                    repair=messages+[{'role':'assistant','content':response['text']}, {'role':'user','content':
                        'The plan failed a mechanical check: '+str(e)+'. Correct the complete JSON using the original question. '
                        'Recheck entities and qualifiers rather than merely changing labels to pass. Maximum 512 output tokens. '
                        'A direct {"answer":"..."} is allowed for an unsuitable tool; it remains unverified.'}]
                    response=transport(repair,512,True);calls.append(response)
    except TransportError as e: errors.append(str(e));status='transport_error'
    usage={k:sum(int(c.get('usage',{}).get(k) or 0) for c in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    return {'arm':arm,'answer':answer,'status':status,'calls':calls,'execution':execution,'errors':errors,'receipts':receipts,'usage':usage,'seconds':time.perf_counter()-t,'question_sha256':hashlib.sha256(question.encode()).hexdigest(),'prompt_sha256':hashlib.sha256(system.encode()).hexdigest(),'semantic_verification':False,'weights_updated':False}
