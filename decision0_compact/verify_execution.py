"""Execution preflight only. No changes to model, cases, prompts or outcomes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
EXPECTED={
'__init__.py':'95a7b94ffb6579c8492e549706ce28f8fc4d7a905116f13e99eb7e58ff9681fd',
'__main__.py':'44fa51b13910376b60be5464183a3d4ed899d02f3e42671b504501bc05677d90',
'cases.py':'1b9769f316634c8ec539d9d44bb0865b8dd4393bf8cef3efb5c26b72179e2110',
'engine.py':'c9c8d4bda4e648fc29688ba7808a16b9c2efc3ca8a6073eca92898fa2862cbbb',
'ledger.py':'2c33bc171a9c8b43a6eed4cd036cab50e1df4e1a05082548a5d07feb40770928',
'oracles.py':'4d6911e34113abb1588a4b71e0ed029692b756a12efd1f1795f9094d096c07b1',
'prompts.py':'f99e20c455a5eac8e54573bb4f62c377d24577cab92fc28e35d2b64c8fdb56dc',
'protocol.py':'19fb807a3876b53cee8837d1871d3840ca697015f21c6afc70c23cdbddb0bb28',
'report.py':'7582756a89af21ef92e15a5ea645414ca839d4f4e67f2a7399c6daf50cac54e3',
'runtime.py':'fdfb27468c9513fd5f8a0521f475273b91216ba64cd38721d9525a0e8af50e00'}
for name,want in EXPECTED.items():
    got=hashlib.sha256((ROOT/'compact_evidence'/name).read_bytes()).hexdigest()
    if got!=want:raise RuntimeError(f'Frozen source changed: {name}: {got} != {want}')
from compact_evidence.__main__ import prepare,validate
from compact_evidence.cases import digest
from compact_evidence.protocol import PROTOCOL
if digest(PROTOCOL)!='3c1e3f63d900ab3d77362a46aa7c98964a52175c71425ecff962142d13cad24a':raise RuntimeError('Protocol changed')
if not (ROOT/'data').exists():prepare(ROOT/'data')
v=validate()
assert v['splits']['development']['public_corpus_sha256']=='2150ec4888a36137fd6f82d4c8c605b44afbc60351d0c3b92f12c38401f7d019'
assert v['splits']['replication']['public_corpus_sha256']=='90c5daf83a908066aa546190c9d71e35af9024cb60a6389fcced17bbab9af348'
print(json.dumps({'frozen_source_files':len(EXPECTED),'original_archive_sha256':'ab042c1339a61fcbdd5a6d333b122798c9a82302ac1db198aa10a4e0f95ed39d','validation':v},indent=2))
