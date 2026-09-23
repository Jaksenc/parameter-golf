"""Loss implementation for the NEXT matched training experiment.

No training is launched by this module. Correct source-defined targets anchor
relationships; agreement alone is never treated as ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor

@dataclass(frozen=True)
class Relation:
    left: int
    right: int
    # Column-stochastic fine-to-coarse or permutation map (right K, left K).
    mapping: Tensor
    # Zero for invariance/renaming/partitioning; exact target change for evidence edits.
    delta: Tensor
    kind: str

@dataclass(frozen=True)
class Losses:
    total: Tensor
    supervised: Tensor
    relational: Tensor


def compute_loss(logits: Sequence[Tensor], targets: Sequence[Tensor],
                 relations: Sequence[Relation], weight: float = 0.25) -> Losses:
    if not logits or len(logits) != len(targets) or not 0 <= weight <= 10:
        raise ValueError('Invalid batch or relation coefficient')
    probabilities, terms = [], []
    for z, q in zip(logits, targets):
        if z.ndim != 1 or z.shape != q.shape or z.numel() < 2:
            raise ValueError('Each input requires a vector with at least two choices')
        if not torch.isfinite(z).all() or not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError('Nonfinite logits or invalid target')
        if q.requires_grad or not torch.isclose(q.sum(), q.new_tensor(1.0), atol=1e-7, rtol=0):
            raise ValueError('Targets must be fixed, normalized distributions')
        logp = torch.log_softmax(z, dim=-1)
        terms.append(-(q * logp).sum())
        probabilities.append(logp.exp())
    supervised = torch.stack(terms).mean()
    relation_terms = []
    for edge in relations:
        if edge.kind not in {'irrelevant', 'permutation', 'partition', 'evidence_delta'}:
            raise ValueError('Unspecified semantic relation')
        if not (0 <= edge.left < len(logits) and 0 <= edge.right < len(logits)) or edge.left == edge.right:
            raise ValueError('Invalid relation endpoints')
        left, right = edge.left, edge.right
        A, d = edge.mapping, edge.delta
        if A.shape != (logits[right].numel(), logits[left].numel()) or d.shape != logits[right].shape:
            raise ValueError('Map shape mismatch')
        if A.requires_grad or d.requires_grad or not torch.isfinite(A).all() or not torch.isfinite(d).all() or (A < 0).any():
            raise ValueError('Maps must be fixed finite nonnegative data')
        if not torch.allclose(A.sum(0), A.new_ones(A.shape[1]), atol=1e-7, rtol=0):
            raise ValueError('Probability map must preserve total mass')
        if edge.kind != 'evidence_delta' and not torch.equal(d, torch.zeros_like(d)):
            raise ValueError('Only a verified evidence edit may have a nonzero target change')
        if not torch.allclose(targets[right], A @ targets[left] + d, atol=1e-7, rtol=0):
            raise ValueError('Claimed relationship contradicts targets')
        residual = probabilities[right] - A @ probabilities[left] - d
        relation_terms.append(residual.square().sum())
    relational = torch.stack(relation_terms).mean() if relation_terms else supervised * 0
    # Both arms can compute this same graph. A zero coefficient is the matched SFT control.
    return Losses(supervised + weight * relational, supervised, relational)
