"""Frozen-input ablations. Reused readouts are explicit, never new inference.

This research helper accepts no reference labels and changes neither runtime policy
nor probabilities. Each logical arm still includes the full readout cost.
"""
from __future__ import annotations
import copy,hashlib
from .contracts import Task
from .core import from_first,finish_pass,evidence_payload


def compare(task: Task, first, core, observations, *, case_id):
    workspace=from_first(task,first)
    rows={};actual_calls=[]
    if workspace.candidate is None:
        for mode in ('review','rules','shared'):
            rows[mode]={'result':finish_pass(task,first,core),
                        'new_readout_calls':0,'reused_readout_from':None}
        return {'arms':rows,'actual_calls':actual_calls}
    order=['review','rules','shared']
    offset=int(hashlib.sha256(case_id.encode()).hexdigest()[:8],16)%3
    order=order[offset:]+order[:offset]
    existing={}
    class Record:
        def call(self,messages,cap):
            call=core.call(messages,cap);actual_calls.append(copy.deepcopy(call));return call
    class Replay:
        def __init__(self,call):self.saved=call
        def call(self,messages,cap):
            if self.saved['request']['messages']!=messages or self.saved['request']['max_tokens']!=cap:
                raise ValueError('reused_readout_prompt_mismatch')
            return copy.deepcopy(self.saved)
    for mode in order:
        obs=observations.get(mode) if mode!='review' else None
        if mode!='review' and obs is None:
            rows[mode]={'result':finish_pass(task,first,core),'new_readout_calls':0,'reused_readout_from':None}
            continue
        other='shared' if mode=='rules' else 'rules'
        reuse=(mode!='review' and other in existing and observations.get(other) is not None
               and evidence_payload(obs)==evidence_payload(observations[other]))
        used=Replay(existing[other]['calls'][-1]) if reuse else Record()
        result=finish_pass(task,first,used,observation=obs,review=mode=='review')
        existing[mode]=result
        rows[mode]={'result':result,'new_readout_calls':0 if reuse else 1,
                    'reused_readout_from':other if reuse else None}
    return {'arms':rows,'actual_calls':actual_calls}
