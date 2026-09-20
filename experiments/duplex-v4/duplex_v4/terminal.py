"""The v3 terminal FFN factorization, with explicit numeric rounding boundaries.
Only the terminal down projection may change; earlier updates invalidate caches.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

class TerminalAdapter(torch.nn.Module):
    def __init__(self, input_size, hidden_size, rank=8, scale=2.0):
        super().__init__();self.scale=float(scale)
        self.a=torch.nn.Parameter(torch.empty(rank,input_size,dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros(hidden_size,rank,dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def delta(self,x):
        return (F.linear(F.linear(x.float(),self.a),self.b)*self.scale).to(x.dtype)
    def forward(self,cache,indices=None):
        def get(k):return cache[k] if indices is None else cache[k][indices]
        x,r,y=get('x'),get('r'),get('y')
        h=r+(y+self.delta(x))
        normalized=qwen_rms(h,cache['norm_weight'],cache['norm_epsilon'])
        return F.linear(normalized.float(),cache['answer_weight'],cache.get('answer_bias'))

def qwen_rms(h,weight,epsilon):
    """Zero-centred RMSNorm; every real cache must verify this against native norm."""
    v=h.float()
    return (v*torch.rsqrt(v.square().mean(-1,keepdim=True)+float(epsilon))*(1+weight.float())).to(h.dtype)

def baseline_logits(cache):
    h=cache['r']+cache['y']
    return F.linear(qwen_rms(h,cache['norm_weight'],cache['norm_epsilon']).float(),
                    cache['answer_weight'],cache.get('answer_bias'))
