"""Narrow record-ID syntax repair, built after the first completed shard.

Not used by the frozen primary experiment. Accepts only a complete bracketed list
of ASCII decimal IDs. Leading zeros are permitted because public records display
zero-padded IDs. No free-text scraping, correct-answer lookup, fuzzy matching,
missing-record recovery, or permission to resolve nonexistent IDs.
"""
from __future__ import annotations
import json,re
from typing import Mapping

FORM=re.compile(r'\[\s*([0-9]{1,6}(?:\s*,\s*[0-9]{1,6})*)\s*\]',re.ASCII)

def resolve(text:str, records:Mapping[int,str], max_records:int=6)->dict:
    if not isinstance(text,str) or len(text)>512:
        raise ValueError('Selection must be a short string')
    if type(max_records) is not int or not 1<=max_records<=64:
        raise ValueError('Invalid selection limit')
    match=FORM.fullmatch(text.strip())
    if match is None:raise ValueError('Expected one complete bracketed decimal-ID list')
    raw=re.split(r'\s*,\s*',match.group(1));ids=[int(t,10) for t in raw]
    if not 1<=len(ids)<=max_records or len(set(ids))!=len(ids):
        raise ValueError('Selection size or duplicate-ID violation')
    if any(i not in records for i in ids):raise ValueError('Selected ID absent from public evidence')
    if any(type(i) is not int or not isinstance(t,str) for i,t in records.items()):raise ValueError('Malformed public record mapping')
    return {'ids':ids,'normalized':json.dumps(ids,separators=(',',':')),
            'leading_zero_normalized':any(len(t)>1 and t.startswith('0') for t in raw),
            'evidence':'\n'.join(f'[{i:02d}] {records[i]}' for i in ids)}
