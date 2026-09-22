"""One-update process-supervision integration test on actual Qwen3.5-4B.

This is not a model-quality experiment. No public benchmark or hidden answers.
A capture-only output head avoids constructing unused prompt x vocabulary
logits; target positions are projected in chunks with the original native head.
The ordinary head is restored and its last-position output is checked.
"""
from __future__ import annotations
import argparse,json,sys,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'depth_distill_v1'))
import study as parent

def run(out):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    p=Path(out);p.mkdir(parents=True,exist_ok=False)
    protocol={'id':'decision0-process-supervision-preflight-v1','model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','updates':1,'seed':95423,'lr':3e-5,'kind':'integration preflight, not capability measurement','benchmark_calls':0}
    parent.save(p/'protocol.json',protocol)
    quantity,price,fee,credit=4,121,150,27
    expected={'product':quantity*price,'subtotal':quantity*price+fee,'total':quantity*price+fee-credit}
    assert expected=={'product':484,'subtotal':634,'total':607}
    conversation=[{'role':'system','content':'Return exactly one JSON object with integer-cent values for product, subtotal, and total. Product is quantity times unit price. Subtotal adds the handling fee. Total subtracts the credit.'},
      {'role':'user','content':'The order has 4 units at 121 cents each, a 150-cent handling fee, and a 27-cent credit. Compute the three requested intermediate values using exact cents.'}]
    answer=json.dumps(expected,separators=(',',':'))
    rt=parent.Runtime();parent.save(p/'runtime.json',rt.meta);torch.manual_seed(protocol['seed'])
    prompt=rt.tokenizer.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=rt.tokenizer.encode(prompt,add_special_tokens=False);suffix=rt.tokenizer.encode(answer,add_special_tokens=False)
    full=rt.tokenizer.encode(prompt+answer,add_special_tokens=False)
    if not suffix or full!=prefix+suffix or len(full)>512:raise RuntimeError('Token-boundary or length mismatch')
    x=torch.tensor([full]);labels=torch.tensor([[-100]*len(prefix)+suffix]);mask=labels[:,1:]!=-100
    if int(mask.sum())!=len(suffix):raise RuntimeError('Incorrect assistant-only loss mask')
    rt.hook.remove();original=rt.model.get_output_embeddings()
    with torch.no_grad(): baseline_logits=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1).logits[0,-1].float().clone()
    class Capture(torch.nn.Module):
        def __init__(self):super().__init__();self.hidden=None
        def forward(self,h):self.hidden=h;return h[...,:1]*0
    capture=Capture();rt.model.set_output_embeddings(capture)
    def loss_and_hidden():
        capture.hidden=None
        rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=0)
        h=capture.hidden
        if h is None or h.shape[:2]!=x.shape:raise RuntimeError('Missing full normalized hidden sequence')
        hs=h[:,:-1][mask];ys=labels[:,1:][mask];loss=h.sum()*0
        for k in range(0,len(ys),4):
            z=F.linear(hs[k:k+4].to(original.weight.dtype),original.weight,original.bias).float()
            loss=loss+F.cross_entropy(z,ys[k:k+4],reduction='sum')
        return loss/len(ys),h
    with torch.no_grad():
        before,h=loss_and_hidden();reconstructed=F.linear(h[:,-1:].to(original.weight.dtype),original.weight,original.bias)[0,-1].float()
        equivalence=float((reconstructed-baseline_logits).abs().max())
    if equivalence>1e-4:raise RuntimeError(f'Native head reconstruction mismatch {equivalence}')
    ad=parent.Adapter(rt,protocol['seed']);initial=parent.tensor_hash(ad.state())
    with torch.no_grad():zero,_=loss_and_hidden()
    if abs(float(zero)-float(before))>1e-5:raise RuntimeError('Zero adapter changes sequence loss')
    parent.save(p/'prepared.json',{'prompt_tokens':len(prefix),'target_tokens':len(suffix),'sequence_tokens':len(full),'expected':expected,'initial_loss':float(before),'zero_update_loss':float(zero),'native_head_equivalence_max_error':equivalence,'initial_sha256':initial})
    rt.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});rt.model.enable_input_require_grads();rt.model.train()
    opt=torch.optim.AdamW(ad.factors.parameters(),lr=protocol['lr'],weight_decay=0);opt.zero_grad(set_to_none=True);tick=time.perf_counter()
    loss,h=loss_and_hidden()
    if not loss.requires_grad or not torch.isfinite(loss):raise RuntimeError('Invalid supervised gradient')
    loss.backward();grads=[float(f.b.grad.float().norm()) if f.b.grad is not None else None for f in ad.factors]
    if any(g is None or not math.isfinite(g) or g<=0 for g in grads):raise RuntimeError('Missing intermediate-block gradients')
    if any(v.grad is not None for v in rt.model.parameters()):raise RuntimeError('Frozen backbone received gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(ad.factors.parameters(),1.0));opt.step();elapsed=time.perf_counter()-tick
    cp=p/'one_update_diagnostic.safetensors';save_file(ad.state(),str(cp))
    rt.model.eval();rt.model.gradient_checkpointing_disable();rt.model.disable_input_require_grads()
    with torch.no_grad():after,_=loss_and_hidden()
    ad.enabled=False
    with torch.no_grad():restored,_=loss_and_hidden()
    with torch.no_grad():
        for f in ad.factors:f.b.zero_()
    ad.load(cp);ad.enabled=True
    with torch.no_grad():reloaded,_=loss_and_hidden()
    restore_error=abs(float(before)-float(restored));reload_error=abs(float(after)-float(reloaded))
    if max(restore_error,reload_error)>1e-5:raise RuntimeError('Checkpoint loss round-trip failed')
    rt.model.set_output_embeddings(original);ad.enabled=False
    with torch.no_grad(): final=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True,logits_to_keep=1).logits[0,-1].float()
    head_restore_error=float((final-baseline_logits).abs().max())
    if head_restore_error>1e-4:raise RuntimeError('Original output head was not restored')
    receipt={'scope':protocol['kind'],'optimizer_updates':1,'prompt_tokens':len(prefix),'assistant_target_tokens':len(suffix),'target':expected,'B_gradient_norms':grads,'gradient_norm':norm,'updated_blocks':sum(bool(f.b.abs().sum()>0) for f in ad.factors),'before_teacher_forced_loss':float(before),'after_teacher_forced_loss':float(after),'native_projection_max_error':equivalence,'base_restoration_loss_error':restore_error,'same_process_reload_loss_error':reload_error,'head_restoration_logit_error':head_restore_error,'update_seconds':elapsed,'peak_rss_mib':parent.memory(),'checkpoint_sha256':parent.filehash(cp),'free_running_generations':0,'capability_improvement_established':False,'official_score':None}
    parent.save(p/'receipt.json',receipt);parent.emit(**receipt)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    try:run(args.out)
    except Exception as exc:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True);parent.save(p/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc(),'success':False});raise
