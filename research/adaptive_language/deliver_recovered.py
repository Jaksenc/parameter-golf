"""Recover a completed experiment. No inference, training, or score-driven changes."""
from __future__ import annotations
import hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import bootstrap as boot
import collect_audit
from prepare_core import unpack
RAW_REF = 'ca58761d9360a4068149e18453050669af43c3ef'
SNAPSHOT_REF = '61ffbf1ba7c3cbc4072d52718e5b2dca95d00375'
BRANCH = 'research/adaptive-language-v07-20260921'
BASE = f'https://raw.githubusercontent.com/Jaksenc/parameter-golf/{SNAPSHOT_REF}/research/adaptive_language/'
def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
def main() -> None:
    original = json.loads(boot.get(BASE + 'results/audit-summary.json'))
    saved_get = boot.json_get
    def pinned_get(url: str):
        if url == 'https://api.github.com/repos/Jaksenc/parameter-golf/git/ref/heads/' + BRANCH:
            return {'object': {'sha': RAW_REF}}
        return saved_get(url)
    capture = {}
    def capture_publish(name: str, obj):
        capture[name] = obj
        return {'captured_locally': name}
    boot.json_get = pinned_get
    boot.publish = capture_publish
    collect_audit.main()
    rerun = capture['audit-summary.json']
    for section in ('primary', 'followup'):
        assert rerun[section]['summary'] == original[section]['summary'], section
        assert rerun[section]['paired'] == original[section]['paired'], section
        assert rerun[section]['source_manifest'] == original[section]['source_manifest'], section
    assert rerun['checks'] == original['checks']
    assert all(rerun['checks'].values())
    root, transport = unpack()
    env = {k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('TOKEN','SECRET','KEY'))}
    env['PYTHONPATH'] = str(root)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(root/'tests')], cwd=root,
                           env=env, capture_output=True, text=True, timeout=120)
    print('CORE_TESTS\n' + tests.stdout + tests.stderr, flush=True)
    assert tests.returncode == 0, 'Original language-core tests failed'
    delivery = Path('/tmp/adaptive-language-delivery/Adaptive-Language-v0.7')
    if delivery.exists():
        raise FileExistsError('Refuse to overwrite a previous delivery')
    delivery.mkdir(parents=True)
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        dest=delivery/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    exp=delivery/'experiment';exp.mkdir()
    here=Path(__file__).parent
    names=['bootstrap.py','preflight_entry.py','prepare_core.py','core_bundle.json','benchmark.py',
           'audit_results.py','collect_audit.py','grounding_followup.py','grounded_prompt.txt']
    source_manifest=[]
    for name in names:
        original_bytes=boot.get(BASE+name)
        assert here.joinpath(name).read_bytes()==original_bytes, 'Changed evaluated source: '+name
        exp.joinpath(name).write_bytes(original_bytes)
        source_manifest.append({'path':name,'sha256':digest(original_bytes)})
    results=exp/'results';results.mkdir()
    for path in sorted(Path('/tmp/adaptive-replay').glob('*.json')):
        shutil.copy2(path,results/path.name)
    for name,obj in capture.items():
        results.joinpath(name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    for name in ('preflight.json','preflight-r2.json'):
        results.joinpath(name).write_bytes(boot.get(BASE+'results/'+name))
    (delivery/'TEST_OUTPUT.txt').write_text(tests.stdout+tests.stderr)
    receipt={'status':'passed','source_snapshot':SNAPSHOT_REF,'raw_results_commit':RAW_REF,
             'recovery_run_id':os.environ.get('GITHUB_RUN_ID'), 'transport':transport,
             'primary_attempts':rerun['primary']['attempts'],'followup_attempts':rerun['followup']['attempts'],
             'core_tests_exit_code':tests.returncode,'original_summary_exactly_reproduced':True,
             'checks':rerun['checks'],'source_manifest':source_manifest,
             'new_model_inference':False,'new_training':False,'private_inputs':False,
             'only_recovered_results':True,'raw_results_included':True,'model_weights_included':False}
    (delivery/'REPLAY_VERIFICATION.json').write_text(json.dumps(receipt,indent=2))
    notices=delivery/'sources';notices.mkdir()
    for name,url in {
        'GSM8K-LICENSE.txt':'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE',
        'BBH-LICENSE.txt':'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/LICENSE'
    }.items():
        notices.joinpath(name).write_bytes(boot.get(url))
    (delivery/'NOTICE.md').write_text('# Scope and attribution\n\nRecovered user research code and recorded public benchmark outputs. No new license is assigned to the user project. Benchmark excerpts retain their upstream notices in sources/. No model weights or font files are bundled. Execution checks do not certify semantic interpretation.\n')
    (delivery/'README.md').write_text('# Adaptive Language v0.7\n\nRecovered, tested language-core source and complete recorded benchmark outputs. This is a small-model language integration, not frontier-level intelligence.\n\nOriginal model: unsloth/Qwen3.5-4B-GGUF Q4_K_M; pinned revision and hashes are in experiment/results/. Primary pilot: 86 questions, 258 attempts. Separate grounding follow-up: 16 questions, 48 attempts. Persistent memory was disabled in scored runs. No new weights were trained.\n\nRun original core tests with `python -m pytest -q tests`. The experiment scripts document full reproduction but need external downloads and a compatible local runtime. They do not start automatically. Raw questions, reference answers, generations, execution traces, and audit results are included under experiment/results/.\n')
    manifest={str(p.relative_to(delivery)):digest(p.read_bytes()) for p in sorted(delivery.rglob('*')) if p.is_file()}
    (delivery/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    target=Path('Adaptive-Language-v0.7-recovered.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(delivery.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(delivery.parent)))
    assert target.stat().st_size < 8_000_000
    print('DELIVERY_RECEIPT '+json.dumps({'archive':str(target),'bytes':target.stat().st_size,
          'sha256':digest(target.read_bytes()),'files':len(manifest)+1,'audit':receipt}),flush=True)
if __name__=='__main__':
    main()
