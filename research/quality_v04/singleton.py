"""Exact, pre-score numerical-execution correction after padded-batch preflight failed.
The optimizer still receives four examples per update. Neither data nor labels,
learning rate, adapter size, sample order, loss, or evaluation splits are changed.
"""
from pathlib import Path
import hashlib
OLD='d49d5681cfaf4ddc2c9cb03448735b0a725a847eb1028907a2d73e43695e8ea9'
NEW='58f5c86df2942f607a66b1fc1f0a90b4b08883a3067f80fe1e1d255617740c27'
def repair():
    path=Path(__file__).with_name('train.py');s=path.read_text();h=hashlib.sha256(path.read_bytes()).hexdigest()
    if h==NEW:return
    if h!=OLD:raise ValueError('Unexpected training-source identity')
    s=s.replace("'batch_size':4,'weight_updates'", "'effective_batch_size':4,'microbatch_size':1,'weight_updates'")
    s=s.replace("'max_input_tokens':1024,'training_budget_seconds':5400", "'max_input_tokens':1024,'microbatch_size':1,'gradient_accumulation':4,'training_budget_seconds':5400")
    start=s.index('    # Readiness is numerical/protocol-only;');end=s.index('    class Factors',start)
    s=s[:start]+'''    # Singleton native execution avoids the failed padded-batch numerical path.
    # This preserves complete four-example optimizer groups through accumulation.
    first=batches[0]
    with torch.inference_mode():
        singleton=[score([k])[0] for k in first]
    save(out/'singleton_probe.json',{'ids':first,'logits':[x.tolist() for x in singleton],
      'mode':'one unpadded forward per example; no approximate batch-equivalence claim'})
'''+s[end:]
    s=s.replace('with torch.inference_mode():zero=score(first)','with torch.inference_mode():zero=[score([k])[0] for k in first]')
    s=s.replace('zip(zero,batched)','zip(zero,singleton)')
    old="model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();keys=batches[i];zs=score(keys)\n            loss=sum(torch.nn.functional.cross_entropy(z[None],torch.tensor([truth[k]])) for k,z in zip(keys,zs))/len(keys)\n            if not torch.isfinite(loss):raise ValueError('nonfinite loss')\n            loss.backward()"
    new="""model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();keys=batches[i];loss_total=0.
            for key in keys:
                z=score([key])[0]
                loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([truth[key]]))/len(keys)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();loss_total+=float(loss.detach())"""
    s=s.replace(old,new).replace("'loss':float(loss.detach())","'loss':loss_total")
    s=s.replace('with torch.inference_mode():restored=score(first)','with torch.inference_mode():restored=[score([k])[0] for k in first]')
    s=s.replace('zip(restored,batched)','zip(restored,singleton)')
    if hashlib.sha256(s.encode()).hexdigest()!=NEW:raise ValueError('Repair differs from tested singleton source')
    path.write_text(s)
if __name__=='__main__':
    repair()
    from .acquire import main
    main()
