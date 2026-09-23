from pathlib import Path
import json,time,hashlib
import causal_v17_data as d

def main():
 import torch,measure_v14
 from reconstruct_v1 import Runtime
 r=json.load(open('causal-prepared/records.json'));m=json.load(open('causal-prepared/manifest.json'))
 assert d.digest(r)==m['record_hash'];rt=Runtime(Path('reconstruction-inputs'));check=rt.check();rows=[]
 out=Path('baseline');out.mkdir(exist_ok=False)
 for i in m['evaluation_indices']:
  row=r[i]['input'];prompt=rt.tokenizer.apply_chat_template(measure_v14.arm_messages(row,'','semantic_codes'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=rt.tokenizer.encode(prompt,add_special_tokens=False);codes=[rt.tokenizer.encode(chr(65+j),add_special_tokens=False) for j in range(len(row['labels']))]
  assert len(ids)<=768 and all(len(c)==1 for c in codes)
  for j,c in enumerate(codes):assert rt.tokenizer.encode(prompt+chr(65+j),add_special_tokens=False)==ids+c
  rt.head.codes=[c[0] for c in codes];t=time.perf_counter()
  with torch.inference_mode():z=rt.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),logits_to_keep=1,use_cache=False,return_dict=True).logits[0,-1].float();p=torch.softmax(z,-1)
  x={'id':row['id'],'record_index':i,'logits':z.tolist(),'probabilities':p.tolist(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_sha256':d.digest(row),'tokens':len(ids),'seconds':time.perf_counter()-t}
  rows.append(x)
  with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
  if len(rows)%32==0:print(json.dumps({'baseline_completed':len(rows),'total':len(m['evaluation_indices'])}),flush=True)
 d.write(out/'complete.json',{'complete':True,'records':len(rows),'runtime':rt.receipt,'fixture':check,'record_hash':m['record_hash'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
if __name__=='__main__':main()
