"""Input-dependent constrained decoding, not semantic truth verification.

A grammar can require fields and restrict raw quantity literals to observed
mentions or named constants. It cannot determine the intended relationships.
"""
from __future__ import annotations
import hashlib,json,re,time,urllib.request
from fractions import Fraction
from adaptive_language.engine import LocalChat,TransportError
from adaptive_language.program import ProgramError,parse_plan
from .contracts import CONSTANTS,literal_values,execute_contract
from .engine import CONTRACT,REASON


def schema_for(question):
    if not isinstance(question,str) or not 4<=len(question)<=6000:raise ValueError('schema_question_size')
    identifier={'type':'string','pattern':'^[a-z][a-z0-9_]{0,31}$'}
    unit={'type':'string','minLength':1,'maxLength':96}
    choices=[];seen=set()
    # Original-character positions are used; Unicode case-folding cannot shift spans.
    pattern=r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?(?!\w|\.\d)|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|half|quarter|third|twice|double|dozen|couple)\b'
    for m in re.finditer(pattern,question,re.I):
        quote=m.group()
        if len(quote)>100:continue
        for value in literal_values(quote):
            key=str(value),quote
            if key in seen:continue
            seen.add(key);choices.append((str(value),quote,None))
        if len(choices)>64:raise ValueError('too_many_source_literals')
    for source,(value,u) in CONSTANTS.items():choices.append((value,source,u))
    quantity=[]
    for value,source,u in choices:
        quantity.append({'type':'object','properties':{'name':identifier,'value':{'const':value},'unit':{'const':u} if u else unit,'source':{'const':source}},'required':['name','value','unit','source'],'additionalProperties':False})
    # A source-faithful task excerpt is supplied mechanically, not evaluated as an entailment.
    trimmed=question.rstrip();end=len(trimmed)
    delimiters=[m.end() for m in re.finditer(r'[.!?]\s+|\n',trimmed)]
    start=delimiters[-1] if delimiters else 0
    target=trimmed[start:]
    if not 4<=len(target)<=300:target=trimmed[-250:]
    plan={'type':'object','properties':{'target':{'const':target},'quantities':{'type':'array','items':{'anyOf':quantity},'minItems':1,'maxItems':24},'steps':{'type':'array','items':{'type':'object','properties':{'name':identifier,'expr':{'type':'string','minLength':1,'maxLength':2048}},'required':['name','expr'],'additionalProperties':False},'maxItems':24},'result':{'type':'string','minLength':1,'maxLength':2048},'answer_unit':unit},'required':['target','quantities','steps','result','answer_unit'],'additionalProperties':False}
    fallback={'type':'object','properties':{'answer':{'type':'string','minLength':1,'maxLength':4096}},'required':['answer'],'additionalProperties':False}
    return {'anyOf':[plan,fallback]}


class ConstrainedChat(LocalChat):
    def __init__(self,question,*args,**kwargs):
        super().__init__(*args,**kwargs);self.schema=schema_for(question)
    def __call__(self,messages,max_tokens,json_mode=False):
        if not json_mode:return super().__call__(messages,max_tokens,False)
        payload={'model':self.model,'messages':messages,'max_tokens':max_tokens,'temperature':0,'seed':1707,'stream':False,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_object','schema':self.schema}}
        raw=json.dumps(payload,ensure_ascii=False).encode();started=time.perf_counter()
        req=urllib.request.Request(self.url,data=raw,headers={'Content-Type':'application/json'},method='POST')
        try:
            with self.opener.open(req,timeout=self.timeout) as response:
                data=response.read(1_000_001)
                if len(data)>1_000_000:raise TransportError('Response size limit')
                obj=json.loads(data)
            c=obj['choices'][0];text=c['message'].get('content') or ''
            if not isinstance(text,str):raise TransportError('Invalid content')
            return {'text':text,'reasoning_content':c['message'].get('reasoning_content') or '', 'finish_reason':c['finish_reason'],'usage':obj.get('usage',{}),'timings':obj.get('timings',{}),'seconds':time.perf_counter()-started,'request_sha256':hashlib.sha256(raw).hexdigest(),'schema_sha256':hashlib.sha256(json.dumps(self.schema,sort_keys=True).encode()).hexdigest()}
        except (OSError,ValueError,KeyError,IndexError) as e:raise TransportError(type(e).__name__+': '+str(e)[:300]) from e
