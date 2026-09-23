"""Research CLI for recorded replay or a trained Learn v16 adapter.

Live mode loads the pinned Qwen3.5-4B model and needs substantially more memory
than this small adapter archive. It is not an authenticated or hardened server.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, sys, time
from pathlib import Path


def validate(payload):
    if not isinstance(payload,dict) or set(payload)-{'id','state','question','labels'}:
        raise ValueError('Only id, state, question and labels are accepted; targets are forbidden')
    if not {'state','question','labels'}<=set(payload):raise ValueError('Missing required fields')
    labels=payload['labels'];q=payload['question']
    if not isinstance(labels,list) or not 2<=len(labels)<=5 or any(not isinstance(x,str) or not x or len(x)>200 or '\n' in x or '\r' in x for x in labels):raise ValueError('Research domain requires2–5 single-line string labels')
    if len(set(labels))!=len(labels):raise ValueError('Duplicate label')
    if not isinstance(q,dict) or set(q)-{'type','instructions','criteria'} or q.get('type')!='choice':raise ValueError('This pilot was trained on Choice only')
    if not isinstance(q.get('instructions'),str) or not q['instructions']:raise ValueError('Instructions required')
    if not isinstance(q.get('criteria'),dict) or set(q['criteria'])!=set(labels) or any(not isinstance(x,str) for x in q['criteria'].values()):raise ValueError('Criteria/label mismatch')
    result={'id':str(payload.get('id','request')),'state':payload['state'],'question':q,'labels':labels}
    encoded=json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False)
    if len(encoded)>60000:raise ValueError('Research request bound')
    return json.loads(encoded)


def record_reply(payload,observation,method,execution):
    import math
    row=validate(payload);p=observation['probabilities'];z=observation['logits']
    if len(p)!=len(row['labels']) or len(z)!=len(p) or any(not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>3e-7:
        raise ValueError('Invalid measured vector')
    index=max(range(len(p)),key=p.__getitem__)
    if index!=max(range(len(z)),key=z.__getitem__):raise ValueError('Answer/logit inconsistency')
    return {'id':row['id'],'answer':row['labels'][index],'probabilities':dict(zip(row['labels'],p)),
            'probability_origin':'measured allowed-code softmax, temperature1, float32',
            'calibration_fitted':False,'method':method,'execution':execution,
            'generated_reasoning_tokens':0,'semantic_correctness_verified':False,
            'scope':'Small synthetic learning-capability pilot; no demonstrated general-domain or JevBench improvement.'}


class AdapterRuntime:
    def __init__(self,data_root,model_dir):
        data_root=Path(data_root).resolve();model_dir=Path(model_dir).resolve()
        sys.path.insert(0,str(data_root))
        import torch
        from safetensors.torch import load_file
        import learn_v16 as train
        from reconstruct_v1 import Runtime
        self.train=train;self.model_dir=model_dir;self.torch=torch
        freeze=json.loads((data_root/'learn-prepared/freeze.json').read_text())
        for filename,expected in freeze['sources'].items():
            if train.filehash(data_root/filename)!=expected:raise ValueError('Frozen runtime file changed: '+filename)
        self.receipt=json.loads((model_dir/'complete.json').read_text())
        if self.receipt['status']!='complete' or self.receipt['steps']!=64 or self.receipt['updated_layers']!=32:raise ValueError('Incomplete checkpoint')
        if train.filehash(model_dir/'adapter.safetensors')!=self.receipt['checkpoint_sha256']:raise ValueError('Checkpoint checksum')
        self.rt=Runtime(data_root/'reconstruction-inputs')
        self.targets,self.factors,self.hooks=train.make_factors(torch,self.rt.model,self.receipt['seed'])
        state=load_file(str(model_dir/'adapter.safetensors'))
        if train.tensors_digest(state)!=self.receipt['final_tensor_sha256']:raise ValueError('Tensor checksum')
        self.factors.load_state_dict(state,strict=True)
        self.rt.model.eval()
        self.method=f"{self.receipt['arm']}-{self.receipt['seed']}-step64"
    def decide(self,payload):
        import measure_v14
        row=validate(payload);t=self.torch;rt=self.rt
        prompt=rt.tokenizer.apply_chat_template(measure_v14.arm_messages(row,'','semantic_codes'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=rt.tokenizer.encode(prompt,add_special_tokens=False)
        codes=[rt.tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(len(row['labels']))]
        if len(ids)>self.train.MAX_TOKENS or any(len(c)!=1 for c in codes):raise ValueError('Token bound or unsupported codes')
        for i,c in enumerate(codes):
            if rt.tokenizer.encode(prompt+chr(65+i),add_special_tokens=False)!=ids+c:raise ValueError('Code continuation boundary')
        rt.head.codes=[c[0] for c in codes];start=time.perf_counter()
        with t.inference_mode():
            z=rt.model(input_ids=t.tensor([ids]),attention_mask=t.ones((1,len(ids)),dtype=t.long),logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float()
            if not t.isfinite(z).all():raise ValueError('Nonfinite output')
            p=t.softmax(z,-1)
        obs={'logits':z.tolist(),'probabilities':p.tolist()};answer=record_reply(row,obs,self.method,'live one-forward neural inference')
        answer.update({'tokens':len(ids),'seconds':time.perf_counter()-start,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'checkpoint_sha256':self.receipt['checkpoint_sha256']})
        return answer


def replay(data_root,seed,arm,item_id,phase='after'):
    data_root=Path(data_root)
    cases=json.loads((data_root/'learn-prepared/records.json').read_text())
    row=next((r['input'] for r in cases if r['input']['id']==item_id),None)
    if row is None:raise ValueError('Unknown recorded request')
    obs=json.loads((data_root/'trained'/f'learn-v16-model-{seed}-{arm}'/(phase+'.json')).read_text())
    value=next((o for o in obs if o['id']==item_id),None)
    if value is None:raise ValueError('Request was not evaluated in that phase')
    return record_reply(row,value,f'{arm}-{seed}-{phase}','saved-record replay; no new neural computation')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');commands=parser.add_subparsers(dest='mode',required=True)
    r=commands.add_parser('replay');r.add_argument('--seed',type=int,choices=(16101,16102,16103),required=True);r.add_argument('--arm',choices=('supervised','relational'),required=True);r.add_argument('--id',required=True);r.add_argument('--phase',choices=('baseline','after'),default='after')
    l=commands.add_parser('live');l.add_argument('input');l.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    if args.mode=='replay':out=replay(args.data,args.seed,args.arm,args.id,args.phase)
    else:
        payload=json.loads(Path(args.input).read_text());validate(payload)
        with contextlib.redirect_stdout(sys.stderr):rt=AdapterRuntime(args.data,args.checkpoint);out=rt.decide(payload)
    print(json.dumps(out,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
