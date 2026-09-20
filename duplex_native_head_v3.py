"""Restricted FP32 output projection for dedicated decision-only model instances."""
from contextlib import contextmanager


def make_allowed_head(base,codes):
    """Restricted projection, not post-hoc full-vocabulary slicing."""
    import torch
    if not isinstance(base,torch.nn.Linear):raise TypeError('This path requires an ordinary Linear output head')
    if not codes or len(set(codes))!=len(codes) or min(codes)<0 or max(codes)>=base.out_features:raise ValueError('Invalid answer-token IDs')
    class AllowedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weight',base.weight[codes].detach().float().clone())
            self.register_buffer('bias',None if base.bias is None else base.bias[codes].detach().float().clone())
        def forward(self,hidden):
            return torch.nn.functional.linear(hidden.float(),self.weight,self.bias)
    return AllowedHead()

@contextmanager
def scoped_head(model,codes):
    base=model.get_output_embeddings();replacement=make_allowed_head(base,codes)
    try:
        model.set_output_embeddings(replacement)
        yield replacement
    finally:model.set_output_embeddings(base)
