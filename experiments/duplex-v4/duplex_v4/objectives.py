"""Paired semantic supervision and baseline-centred controls.

This combination is an experimental method, not a claim of an original theorem
or a guarantee of transfer. All arms see exactly the same answer examples.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class LossSpec:
    anchor: float = 0.0
    invariant: float = 0.0
    change: float = 0.0
    contrast_margin: float = 2.0

ARMS = {
    'direct': LossSpec(),
    'anchored': LossSpec(anchor=0.05),
    'structured': LossSpec(anchor=0.05, invariant=0.1, change=0.02),
}

def centered(z): return z - z.mean(dim=-1, keepdim=True)

def meaning_logits(z, order):
    """order[i,j] is the semantic identity in input slot j; no gold is needed."""
    if z.ndim != 2 or order.shape != z.shape: raise ValueError('Alignment shape mismatch')
    expected=torch.arange(z.shape[-1],device=order.device).expand_as(order)
    if not torch.equal(order.sort(dim=-1).values,expected):raise ValueError('Not a candidate permutation')
    return z.gather(-1,order.argsort(dim=-1))

def semantic_losses(z, z0, order, targets, families, spec: LossSpec):
    """families is [G,8], with 6 invariant views, a fact edit, a rule edit.
    Model output z has 6 slots per request. Targets are input-slot targets.
    Test labels and test families must never be supplied to this function.
    """
    if z.shape!=z0.shape or z.shape[-1]!=6: raise ValueError('Six-option shapes required')
    if families.ndim!=2 or families.shape[1]!=8:raise ValueError('Eight views per source required')
    if families.numel()!=len(z) or sorted(families.flatten().tolist())!=list(range(len(z))):
        raise ValueError('Families must partition the population')
    if not torch.isfinite(z).all() or not torch.isfinite(z0).all():raise ValueError('Nonfinite logits')
    s=meaning_logits(z,order)
    truth=order.gather(1,targets[:,None]).squeeze(1)
    base_indices=families[:,0]
    if not torch.equal(truth[families[:,:6]],truth[base_indices,None].expand(-1,6)):
        raise ValueError('Invalid same-meaning relation')
    if torch.any(truth[families[:,6:]]==truth[base_indices,None]):
        raise ValueError('Change edge has unchanged gold')
    ce=F.cross_entropy(z,targets)
    anchor=(centered(z)-centered(z0)).square().mean()
    logp=F.log_softmax(s,dim=-1)
    p=logp.exp()
    group_p=p[families[:,:6]]
    avg=group_p.mean(dim=1,keepdim=True)
    invariant=(group_p*(logp[families[:,:6]]-avg.clamp_min(1e-30).log())).sum(-1).mean()
    i=base_indices[:,None].expand(-1,2).flatten()
    j=families[:,6:].flatten()
    a,b=truth[i],truth[j]
    # A difference of semantic log odds cancels any shared semantic bias.
    contrast=(s[i,a]-s[i,b])-(s[j,a]-s[j,b])
    change=F.relu(spec.contrast_margin-contrast).square().mean()
    total=ce+spec.anchor*anchor+spec.invariant*invariant+spec.change*change
    return total,{'ce':ce,'anchor':anchor,'invariant':invariant,'change':change}
