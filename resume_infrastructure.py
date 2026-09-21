"""Infrastructure-only continuation of a partially completed frozen shard.

No model/prompt/interpreter/routing change. Only inputs without a complete prior
record are retried. The merged receipt retains every completed prior record and
names the continuation source and provenance. Do not use to repair model outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import run_evidence_v4
E = run_evidence_v4.experiment


def load_prefix(path: Path, planned: list[dict]) -> tuple[list[dict], str | None]:
    records = []
    truncated = None
    if not path.exists():
        return records, truncated
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Only an interrupted final line can be retried; never skip a middle row.
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise
            truncated = hashlib.sha256(line).hexdigest()
            break
        if len(records) >= len(planned):
            raise ValueError('Extra prior record')
        task = planned[len(records)]
        if row['id'] != task['id'] or row['input_sha256'] != E.digest(E.inp(task)):
            raise ValueError('Prior records are not an exact valid plan prefix')
        # Complete row contains every previously specified measurement.
        for key in ('native', 'compile', 'reason', 'execution', 'predictions'):
            if key not in row:
                raise ValueError('Incomplete JSON record, not an interrupted line')
        records.append(row)
    return records, truncated


def main(root: Path, prior: Path, out: Path, shard: int) -> None:
    if not 0 <= shard < E.SHARDS:
        raise ValueError('Invalid shard')
    if out.exists():
        raise ValueError('Output already exists')
    out.mkdir(parents=True)
    original_partition = E.partition
    plan = original_partition(root)
    planned = plan[shard]
    rows, truncated = load_prefix(prior / 'records.jsonl', planned)
    prior_preflight = json.loads((prior / 'preflight.json').read_text())
    if prior_preflight['source'] != E.sha(E.__file__):
        raise ValueError('Parent source identity mismatch')
    remaining = planned[len(rows):]
    new_dir = out / 'continuation'
    if remaining:
        updated = [list(x) for x in plan]
        updated[shard] = remaining
        E.partition = lambda _root: updated
        try:
            E.run(root, new_dir, shard, False)
        finally:
            E.partition = original_partition
        new = [json.loads(line) for line in (new_dir / 'records.jsonl').read_text().splitlines()]
        if len(new) != len(remaining):
            raise ValueError('Continuation incomplete')
        for record, task in zip(new, remaining):
            if record['id'] != task['id'] or record['input_sha256'] != E.digest(E.inp(task)):
                raise ValueError('Continuation assignment mismatch')
        final = rows + new
        shutil.copy2(new_dir / 'preflight.json', out / 'preflight.json')
    else:
        final = rows
        shutil.copy2(prior / 'preflight.json', out / 'preflight.json')
    if len(final) != len(planned):
        raise ValueError('Merged shard remains incomplete')
    with (out / 'records.jsonl').open('w') as f:
        for r in final:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    receipt = {
        'count': len(final), 'shard': shard, 'source_sha256': E.sha(E.__file__),
        'population_sha256': E.digest(planned),
        'records_sha256': E.sha(out / 'records.jsonl'), 'pilot': False,
        'infrastructure_recovery': {
            'retained_prior_rows': len(rows), 'continued_rows': len(remaining),
            'prior_records_sha256': E.sha(prior / 'records.jsonl') if (prior / 'records.jsonl').exists() else None,
            'interrupted_last_line_sha256': truncated,
            'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'method_change': False,
        },
    }
    E.write(out / 'complete.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('reconstruction-inputs'))
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    args = p.parse_args()
    main(args.root, args.prior, args.out, args.shard)
