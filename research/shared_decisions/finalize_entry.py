"""Transport-only recovery. Fixed decoded digest must match before any inference."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path

def verified_manifest():
 p=Path(__file__).with_name('final_observations.json');before=p.read_bytes();obj=json.loads(before);data=obj['data'];repairs=[]
 for bad,good in [('DPK3/+P/A','DPK3+P/A'),('AvZXCd5qad5qADhc','AvZXCd5qADhc'),('SSlXAuQUWZjYgQGn','SSlXAuQUZgQGn')]:
  if bad in data:data=data.replace(bad,good);repairs.append(bad)
 raw=gzip.decompress(base64.b64decode(data,validate=True))
 expected='1cc65bb7be2b604a02c7aa81daff2dd19594be1aef664fdd0eddb76846659a5f'
 if hashlib.sha256(raw).hexdigest()!=expected or obj['decoded_sha256']!=expected:raise ValueError('frozen_observation_bytes_not_recovered')
 entries=json.loads(raw)
 if len(entries)!=15:raise ValueError('observation_count')
 obj['data']=data;p.write_text(json.dumps(obj,separators=(',',':')))
 receipt={'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'decoded_sha256':expected,'transport_repairs':repairs,'observation_entries':len(entries),'semantic_content_changed':False}
 Path('manifest-transport-receipt.json').write_text(json.dumps(receipt,indent=2));print('TRANSPORT_VERIFIED '+json.dumps(receipt),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--verify-only',action='store_true');a=p.parse_args();verified_manifest()
 if not a.verify_only:
  import finalize_comparison
  if a.shard is None or not 0<=a.shard<8:raise ValueError('shard')
  finalize_comparison.run(a.shard,Path('readouts'))
