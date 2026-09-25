"""Frozen Probability v13 serving boundary for maintainer-operated JevBench.

Only the complete-probability primary is exposed. This is a CPU research
reference, not a claim of fast Jev-class serving. No gold, corpus, or cache is
consulted at inference. Public and sealed prompts need not be logged.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'Decision-0 Probability v13 (full-calibrated)'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TEMPERATURE = 1.0218971486541166
FIT_HASH = 'd23db64c1bcd5bc5b153014261dd9d06043051da9c811ff41be54d34af04de27'
CAP = 480
CORE_SHA256 = {
    'contrast_v7.py': '52771fdb5d8c7e5315c6f49ac87b725ba1244c0fe5ba4f5a0674ec4de3778d0b',
    'handoff_v5.py': '2fe06521028bed0ecae5fe1584aba3eae0f6852d5c2699d51d6703c3ade8dfb4',
    'evidence_v4.py': 'a0184e91a1aec54ec1ca200313e1c9ed51c6e456056f3b9ab0441a370077ae31',
    'reconstruct_v1.py': '62a30f102e16067632303065cd997fd0691fe0f5630f82c68333916a9d9ad0ec',
}
REQUEST_FIELDS = ('id', 'state', 'question', 'labels')


def verify_core(root: Path = ROOT) -> None:
    for name, expected in CORE_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source mismatch: {name}')


def canonical_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve explicit labels order; sort object keys as in the v13 experiment."""
    if not isinstance(payload, dict) or set(payload) != set(REQUEST_FIELDS):
        raise ValueError('Expected exactly id, state, question, labels')
    q, labels = payload['question'], payload['labels']
    if not isinstance(labels, list) or not 2 <= len(labels) <= 16:
        raise ValueError('Exactly 2 to 16 labels supported')
    if any(not isinstance(x, str) or not x or '\n' in x or '\r' in x for x in labels):
        raise ValueError('Labels must be nonempty single-line strings')
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate labels')
    if not isinstance(q, dict) or q.get('type') not in ('choice', 'noul', 'score'):
        raise ValueError('Unsupported question type')
    if not isinstance(q.get('instructions'), str) or not isinstance(q.get('criteria'), (dict, list)):
        raise ValueError('Instructions and criteria are required')
    if q['type'] == 'score' and (len(labels) > 10 or labels != [str(i) for i in range(len(labels))]):
        raise ValueError('Score domain must be 0..K-1, K <= 10')
    if q['type'] == 'noul' and {s.lower() for s in labels} not in ({'no', 'yes'}, {'false', 'true'}):
        raise ValueError('Noul domain requires no/yes or false/true')
    return json.loads(json.dumps({**payload, 'id': str(payload['id'])},
                                sort_keys=True, ensure_ascii=False, allow_nan=False))


def task_payload(task: Any) -> dict[str, Any]:
    """Explicit allowlist: evaluation-only attributes never enter model messages."""
    return canonical_request({k: task[k] if isinstance(task, dict) else getattr(task, k)
                              for k in REQUEST_FIELDS})


def calibrated_vector(logits: list[float]) -> list[float]:
    if not 2 <= len(logits) <= 16 or any(not math.isfinite(float(z)) for z in logits):
        raise ValueError('Incomplete or nonfinite probability readout')
    top = max(logits)
    weights = [math.exp((float(z) - top) / TEMPERATURE) for z in logits]
    total = sum(weights)
    return [w / total for w in weights]


def infer(rt: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Same frozen generation and readout functions as the measured v13 primary."""
    import contrast_v7 as old
    row = canonical_request(payload)
    started = time.perf_counter()
    trace = old.generate(rt, old.e4.messages(row, 'reason'), CAP)
    # A valid textual FINAL never bypasses the unconditional all-option readout.
    observation = old.h5.readout(rt, row, trace['text'])
    logits = observation['logits']
    if len(logits) != len(row['labels']):
        raise ValueError('Missing label probability')
    values = calibrated_vector(logits)
    winner = min(range(len(values)), key=lambda i: (-values[i], row['labels'][i]))
    input_tokens = trace['input_tokens'] + observation['input_tokens']
    output_tokens = trace['output_tokens']
    return {
        'ok': True, 'model': MODEL_NAME,
        'probs': dict(zip(row['labels'], values)), 'label': row['labels'][winner],
        'latency_s': time.perf_counter() - started,
        'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                  'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
                  'total_tokens': input_tokens + output_tokens},
        'raw': {'runtime': {'model_revision': REVISION, 'generated_tokens': output_tokens,
                           'draft_limit': CAP, 'temperature': TEMPERATURE,
                           'calibration_fit_record_hash': FIT_HASH,
                           'probability_origin': 'FP32 projection on allowed decision-code rows; conditional softmax',
                           'autoregressive_draft': True, 'trained_adapter': False},
                'draft_sha256': hashlib.sha256(trace['text'].encode()).hexdigest(),
                'readout_prompt_sha256': observation['prompt_sha256']},
    }


def load_runtime() -> Any:
    verify_core()
    from reconstruct_v1 import Runtime
    # CPU BF16/FP32 reference. Runtime itself pins and checks both weight files.
    rt = Runtime(ROOT / 'reconstruction-inputs')
    rt.receipt['no_generation'] = False  # This submitted path explicitly generates.
    rt.receipt['submission_path'] = 'full_calibrated'
    return rt


class Decision0Adapter:
    """Uses the official Runner/DecisionResult without changing official scoring."""
    name = 'decision0_probability_v13'
    model = MODEL_NAME
    endpoint = 'local-cpu'
    cost_basis = 'self_hosted_cpu_unmetered'
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, runtime: Any = None):
        self.runtime = runtime
        self.lock = threading.Lock()

    def load(self):
        if self.runtime is None:
            self.runtime = load_runtime()

    def run(self, task):
        from jevbench.adapters.base import DecisionResult
        start = time.perf_counter()
        try:
            payload = task_payload(task)
            with self.lock:
                self.load()
                result = infer(self.runtime, payload)
            return DecisionResult(adapter=self.name, ok=True, probs=result['probs'],
                probs_source='native', model=self.model, status=200,
                latency_s=time.perf_counter() - start, usage=result['usage'], raw=result['raw'])
        except Exception as exc:
            # No exception text: some underlying libraries include prompt snippets.
            return DecisionResult(adapter=self.name, ok=False, probs_source='native',
                model=self.model, status=422 if isinstance(exc, ValueError) else 500,
                error=f'Inference failed ({type(exc).__name__})',
                latency_s=time.perf_counter() - start)

    def reserve_estimate(self, task):
        # Reservation zero is NOT a price: there is no metered provider account.
        return 0.0


def serve(host: str, port: int) -> None:
    rt = load_runtime()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not write sealed prompt bodies or routine request logs.

        def send_json(self, status, value):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path != '/health':
                self.send_json(404, {'ok': False})
                return
            self.send_json(200, {'ready': True, 'model': MODEL_NAME, 'revision': REVISION})

        def do_POST(self):
            if self.path != '/run':
                self.send_json(404, {'ok': False})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Body limit')
                body = json.loads(self.rfile.read(length))
                result = infer(rt, task_payload(body['task']))
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError):
                self.send_json(400, {'ok': False, 'error': 'Invalid or unsupported request'})
            except Exception as exc:
                self.send_json(500, {'ok': False, 'error': f'Inference failed ({type(exc).__name__})'})

    # Intentionally sequential: the historical runtime mutates its selected head.
    HTTPServer((host, port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('predict')
    one.add_argument('input', help='Canonical request JSON, without answers or metadata')
    server = sub.add_parser('serve')
    server.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1', 'localhost'])
    server.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args.host, args.port)
    else:
        payload = canonical_request(json.loads(Path(args.input).read_text()))
        with contextlib.redirect_stdout(sys.stderr):
            result = infer(load_runtime(), payload)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
