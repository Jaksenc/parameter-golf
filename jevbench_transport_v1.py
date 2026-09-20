"""Transport-only recovery of frozen public JevBench comparison.
No task, model, scoring, partition or selection changes. GitHub API credentials
must not be forwarded to the separately signed artifact-storage origin.
"""
from __future__ import annotations
import argparse,hashlib,os,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import jevbench_public_v1 as study
ORIGINAL_SHA256='14cd69ec52ee1d75bebea28de354101bfe61faed5c9b4543f8203d60a5d17242'
class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urlsplit(newurl).scheme!='https':raise ValueError('HTTPS redirects only')
        new=super().redirect_request(req,fp,code,msg,headers,newurl)
        if new is not None and urlsplit(req.full_url).netloc!=urlsplit(newurl).netloc:
            new.remove_header('Authorization')
            new.remove_header('Cookie')
        return new

def safe_get(url,auth=False):
    parsed=urlsplit(url)
    if parsed.scheme!='https':raise ValueError('HTTPS source required')
    if auth and parsed.hostname!='api.github.com':raise ValueError('Credentials scoped to GitHub API')
    headers={'User-Agent':'Duplex-public-benchmark/1.0'}
    if auth:headers.update(Authorization='Bearer '+os.environ['GH_TOKEN'],Accept='application/vnd.github+json')
    opener=urllib.request.build_opener(ScopedRedirect())
    with opener.open(urllib.request.Request(url,headers=headers),timeout=120) as r:data=r.read(25_000_001)
    if len(data)>25_000_000:raise ValueError('Download exceeds bounded resource size')
    return data

def test_redirect():
    handler=ScopedRedirect();r=urllib.request.Request('https://api.github.com/repos/o/r/actions/artifacts/1/zip',headers={'Authorization':'Bearer fixture','Cookie':'fixture'})
    other=handler.redirect_request(r,None,302,'Found',{},'https://storage.example.test/artifact?signature=fixture')
    assert other.get_header('Authorization') is None and other.get_header('Cookie') is None
    same=handler.redirect_request(r,None,302,'Found',{},'https://api.github.com/another')
    assert same.get_header('Authorization')=='Bearer fixture'
    try:handler.redirect_request(r,None,302,'Found',{},'http://storage.example.test/artifact')
    except ValueError:pass
    else:raise AssertionError('Downgrade accepted')

def main():
    if hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest()!=ORIGINAL_SHA256:raise RuntimeError('Frozen benchmark source changed')
    test_redirect();study.self_test();study.get=safe_get
    study.emit('transport_recovery',source_sha256=ORIGINAL_SHA256,previous_failed_run=35540061862,changes='Strip API credentials on cross-origin HTTPS redirects only; no model/task/scoring changes')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--shard',type=int);ap.add_argument('--audit',type=int);a=ap.parse_args()
    if a.prepare:
        rows,manifest=study.prepare('jevbench-prepared',adapters=True)
        study.emit('prepared',cases=len(rows),data_hash=manifest['data_hash'],verified_adapter_downloads=2,partition_sizes=[len(x) for x in manifest['partitions']])
    elif a.shard is not None:study.run(a.shard)
    elif a.audit is not None:
        study.audit(a.audit)
        (Path('jevbench-audit')/'jevbench_transport_v1.py').write_bytes(Path(__file__).read_bytes())
    else:ap.error('Choose an execution mode')
if __name__=='__main__':main()
