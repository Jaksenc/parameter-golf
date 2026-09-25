"""Invoke the official JevBench Runner with the frozen local candidate.
No evaluator scoring, tasks, or sealed content is copied or modified here.
"""
import argparse
import json
import time
from pathlib import Path
from decision0_probability_submission import Decision0Adapter, MODEL_NAME, REVISION, TEMPERATURE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', required=True, help='Comma-separated evaluator-owned JSONL paths')
    parser.add_argument('--out', required=True, help='New private directory; never overwritten')
    args = parser.parse_args()
    from jevbench.tasks import load_jsonl, dataset_hash
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    tasks = []
    for path in args.tasks.split(','):
        tasks.extend(load_jsonl(path.strip()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    adapter = Decision0Adapter()
    tick = time.perf_counter()
    adapter.load()
    load_seconds = time.perf_counter() - tick
    # Public service price remains None. Budget reservation is not a tariff.
    ledger = Ledger(str(out / 'ledger.jsonl'), cap_usd=0.0)
    runner = Runner(adapter, ledger, raw_dir=str(out / 'raw'), default_reserve_usd=0.0)
    records = runner.run_all(tasks, results_path=str(out / 'results.jsonl'), delay_s=0.0)
    manifest = {'model': MODEL_NAME, 'model_revision': REVISION, 'temperature': TEMPERATURE,
        'adapter': adapter.name, 'dataset_hash': dataset_hash(tasks), 'planned': len(tasks),
        'attempted': len(records), 'completed_full_suite': False,
        'scope': 'Only the task files supplied by the evaluator; no full-score claim inferred from count.',
        'price_input_per_m': None, 'price_output_per_m': None,
        'cost_basis': adapter.cost_basis, 'loading_seconds_excluded_from_request_times': load_seconds,
        'latency_basis': 'local CPU, sequential official Runner; no network hop',
        'sealed_data_policy': 'Evaluator-owned inputs/results stay in the supplied private output directory.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    failures = sum(r.get('status') == 'failed' for r in records)
    print(json.dumps({'attempted': len(records), 'planned': len(tasks), 'failures': failures,
                      'output_directory': str(out)}))
    return 0 if len(records) == len(tasks) and not failures else 3


if __name__ == '__main__':
    raise SystemExit(main())
