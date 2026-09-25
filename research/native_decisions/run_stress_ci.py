"""Bounded paired native vs schema-only GGUF probe on fixed authored stress tasks.

For GitHub Actions: put alongside independent_stress.py in research/native_decisions.
Never send oracle targets to the worker process. Journals are append-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import independent_stress
import readout
import schema_control

SHARDS = 4
GROUPS_PER_FAMILY = 2
EXPECTED_DATA_SHA = 'e9cce349225dc2c60f2a7fa9cb20d25e2f2c2ede26ffdf5e1a4c6a480c4fc9df'
EXPECTED_GENERATOR_SHA = 'e54206564c6e392d683f6968e66650246105985318a7384466da8852ba4e5b65'


def roster():
    if readout.digest(Path(independent_stress.__file__).read_bytes()) != EXPECTED_GENERATOR_SHA:
        raise ValueError('generator source changed')
    rows = independent_stress.build()
    manifest = independent_stress.audit(rows)
    if manifest['data_sha256'] != EXPECTED_DATA_SHA:
        raise ValueError('dataset identity changed')
    families = defaultdict(set)
    for r in rows:
        families[r['family']].add(r['group'])
    selected = {g for f, group_set in families.items() for g in sorted(
        group_set, key=lambda x: hashlib.sha256(('stress-probe-v1:' + x).encode()).hexdigest())[:GROUPS_PER_FAMILY]}
    result = [r for r in rows if r['group'] in selected]
    assert len(result) == 96 and len(selected) == 8
    return result, manifest, sorted(selected)


def worker(input_path, journal_path):
    b = schema_control.exact_backend.RawBackend()
    try:
        with journal_path.open('x') as stream:
            for packet in json.loads(input_path.read_text()):
                if set(packet) != {'id', 'task'} or set(packet['task']) != {'state', 'question', 'labels'}:
                    raise ValueError('unexpected worker fields')
                arms = schema_control.ARMS if int(readout.digest(packet['id'])[:8], 16) % 2 == 0 else schema_control.ARMS[::-1]
                for arm in arms:
                    attempt = b.solve(packet['task'], arm) if arm == 'native_canonical' else schema_control.schema_solve(packet['task'], b)
                    result = {'id': packet['id'], 'arm': arm, 'ok': attempt['ok'],
                              'probs': attempt.get('probs'), 'error': attempt.get('error'),
                              'seconds': attempt.get('seconds'), 'calls': attempt.get('calls'),
                              'task_sha256': readout.digest(readout.canonical(packet['task']))}
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    print('ATTEMPT ' + json.dumps({k: result.get(k) for k in ('id', 'arm', 'ok', 'error', 'seconds')}), flush=True)
    finally:
        b.close()


def run(shard):
    if not 0 <= shard < SHARDS:
        raise ValueError('invalid shard')
    schema_control.check_source()
    rows, manifest, selected = roster()
    subset = [r for i, r in enumerate(rows) if i % SHARDS == shard]
    packet_path = Path('/tmp/independent-stress-packets.json')
    packet_path.write_text(json.dumps([{'id': r['id'], 'task': r['task']} for r in subset]))
    journal = Path(f'independent-stress-journal-{shard:02d}.jsonl')
    receipt = Path(f'independent-stress-shard-{shard:02d}.json')
    output = {'status': 'failed', 'shard': shard, 'run_id': os.environ.get('GITHUB_RUN_ID'),
              'source_commit': os.environ.get('GITHUB_SHA'), 'generator': manifest,
              'selected_groups': selected, 'cases': subset, 'planned_attempts': len(subset) * 2,
              'protocol': {'arms': list(schema_control.ARMS), 'checkpoint': schema_control.exp.PIN,
                           'raw_logits': 'native_canonical', 'generation': 'schema_keyed',
                           'training': False, 'quantization': 'Q4_K_M', 'calibration': False,
                           'model_weight_identity_required': True, 'retry': False,
                           'pilot_only': True, 'whole_stress_set': False}}
    proc = log = None
    start = time.perf_counter()
    try:
        proc, log, info = schema_control.setup()
        output['runtime'] = info
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN', 'SECRET', 'KEY'))}
        try:
            code = subprocess.run([sys.executable, __file__, '--worker', '--input', str(packet_path),
                                   '--journal', str(journal.resolve())], env=env, timeout=3300).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
        attempts = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
        output.update(status='completed' if code == 0 and len(attempts) == len(subset) * 2 else 'incomplete',
                      worker_exit=code, attempts=attempts, received_attempts=len(attempts))
    except Exception as exc:
        output['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        schema_control.exp.stop(proc, log)
        output['wall_seconds'] = time.perf_counter() - start
        receipt.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        print('RECEIPT ' + json.dumps({k: output.get(k) for k in ('status', 'error', 'planned_attempts', 'received_attempts', 'worker_exit')}), flush=True)
    if output['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--input', type=Path)
    p.add_argument('--journal', type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        rows, manifest, groups = roster()
        print(json.dumps({'n': len(rows), 'groups': groups, 'data_sha256': manifest['data_sha256']}))
    elif args.worker:
        worker(args.input, args.journal)
    else:
        run(args.shard)
