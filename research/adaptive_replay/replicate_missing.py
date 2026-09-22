"""Declared replication of one upload-lost shard, never a replacement primary score."""
from pathlib import Path
import base64,gzip,hashlib,json,zipfile
import run as primary
EXPECTED={'core.py':'ccc7ee849074094456d1479f77c09802b9ac2b409253028b21d4813313c52e0e','run.py':'d4ed2c982ea65e994d6c999feb1bf4a789a29d4fe16660fc7b55eb9f3ea01e66'}

def main():
 for name,h in EXPECTED.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
 print('REPLICATION_DECLARATION '+json.dumps({'original_run':'35679439488','original_job':'106593890419','shard':3,'reason':'Artifact bytes uploaded but finalization failed with 403. Original scorer log says 4/4 for each arm. This is a separately reported replication, not original raw output recovery.','source_locks':EXPECTED}),flush=True)
 try:primary.run(3,False)
 finally:
  p=Path('replay-shard-03.json')
  if p.exists():
   d=json.loads(p.read_text());d['replication']={'original_run':'35679439488','original_job':'106593890419','original_summary':{a:{'correct':4,'n':4} for a in ('reasoning','replay','sham')},'not_original_bytes':True}
   raw=json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False).encode();Path('replay-shard-03-replication.json').write_bytes(raw)
   # Public benchmark-only emergency receipt; no credentials or user-private data.
   encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
   print('BACKUP_SHA256 '+hashlib.sha256(raw).hexdigest(),flush=True)
   for i in range(0,len(encoded),3000):print(f'BACKUP_PART_{i//3000:03d} '+encoded[i:i+3000],flush=True)
  with zipfile.ZipFile('replication-source.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(__file__,'replicate_missing.py')

if __name__=='__main__':main()
