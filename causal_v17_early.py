"""Fixed early-trajectory evaluation; uses the unchanged v16 evaluator.
Only checkpoint step/hash arguments change. No training or checkpoint selection.
"""
from pathlib import Path
import argparse,json,hashlib
import evaluate_learn_v16_saved as old
H={
'16101-relational':{1:'1f237ed752781d22320d72952e6872a9da86d453acde5f2d737f040e280c3cfb',16:'13801ae49e1ddc85d23fca3182c641e6e6785dfe2a3f3d7a4d76936c3494d133'},
'16101-supervised':{1:'7a3dd19c8696e66ad75eb68ce28d830b8cf1739df48c234ecb390efd4bf932b7',16:'060286a5f38ed5f3931dfd69fbdf3940bea6ccac318aa469371465807dd1bf18'},
'16102-relational':{1:'e1853427dadaad336e8fe6badbdf025c9b29d8695cfb4d9451c7fddab1b211aa',16:'5f295a58b60286e73c25b234ccd58cb30ec0dcac0abb9c259665c8218725035d'},
'16102-supervised':{1:'a762009108a247ef0a72fc8cb373b73949e99bb0b8392e5631d65e5ee6e941e3',16:'b69297b4e13cd58ac91a8eac0ac1c6b2419ea12ccc4a9e7f642a4cc8aa847b85'},
'16103-relational':{1:'7b8a38cb0b018d4cfbd0c4d4cb674057ff1ed99386be9aaa8b517458b8a346d4',16:'2d79c2a138da123d4993bf21e90842daa15395a91605641647f29ecd99b0751b'},
'16103-supervised':{1:'e8c4c17ccadd72880366d060ea42d6d454a2c7db726dc64bea3db74e36783cb6',16:'e9566829e43ca10496ef69833cdba297cf20bed316a85a6fc6210513f8cf827b'}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['supervised','relational'],required=True);p.add_argument('--step',type=int,choices=[1,16],required=True);p.add_argument('--out',required=True);a=p.parse_args()
 k=f'{a.seed}-{a.arm}';old.STEP=a.step;old.EXPECTED={k:H[k][a.step]}
 old.run(Path('.'),Path(a.out),a.seed,a.arm)
 f=Path(a.out)/'complete.json';r=json.loads(f.read_text())
 r['unique_fit_inputs_by_evaluated_step']=r.pop('unique_fit_inputs_by_step32')
 r['evaluation_wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 r['scope']='All saved step1/16 checkpoints evaluated; availability-independent fixed audit, not an early-stopping model selection.'
 f.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False))
if __name__=='__main__':main()
