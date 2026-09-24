"""One acquisition-only recovery. No scored answer or failed inference is retried.
Original continuation shard 23 failed before loading a model: release metadata API
returned a rate-limit response. Reuse the already acquired immutable release/model
metadata locally; all public asset downloads still pass original SHA256 checks.
"""
from __future__ import annotations
import json,os,zipfile
from pathlib import Path
import resume_entry
r=resume_entry.r
boot=r.exp.boot
old_get=boot.json_get
RUNTIME='9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443'
WEIGHTS='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
REV='e87f176479d0855a907a41277aca2f8ee7a09523'
def cached_metadata(url):
 if url=='https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10964':
  return {'assets':[{'name':'llama-b10964-bin-ubuntu-x64.tar.gz','digest':'sha256:'+RUNTIME,'browser_download_url':'https://github.com/ggml-org/llama.cpp/releases/download/b10964/llama-b10964-bin-ubuntu-x64.tar.gz'}]}
 if url==f'https://huggingface.co/api/models/unsloth/Qwen3.5-4B-GGUF/revision/{REV}?blobs=true':
  return {'sha':REV,'siblings':[{'rfilename':'Qwen3.5-4B-Q4_K_M.gguf','lfs':{'sha256':WEIGHTS}}]}
 return old_get(url)
boot.json_get=cached_metadata
if __name__=='__main__':
 print('ACQUISITION_RECOVERY '+json.dumps({'original_run':'36013756279','original_job':'107680850874','shard':23,'reason':'metadata HTTP 403 rate limit before model load; zero completed inference','metadata_source':'existing native preflight verified acquisition receipt, run 35913116027','model_bytes_unchanged':True,'no_new_metadata_api_request':True}),flush=True)
 try:r.run(23)
 finally:
  p=Path('resumed-native-23.json')
  if p.exists():
   d=json.loads(p.read_text());d['acquisition_recovery']={'original_run':'36013756279','original_job':'107680850874','metadata_reused':True,'original_inference_attempts':0,'wrapper_sha256':r.readout.digest(Path(__file__).read_bytes())};p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))
  with zipfile.ZipFile('cached-metadata-source.zip','w',zipfile.ZIP_DEFLATED) as z:
   for name in ('resume_cached_metadata.py','resume_entry.py','resume_frozen.py'):z.write(Path(__file__).with_name(name),name)
