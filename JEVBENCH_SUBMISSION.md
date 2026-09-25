# Decision-0 Probability v13 — full JevBench evaluation request

**Candidate:** `Decision-0 Probability v13 (full-calibrated)`.
**Status:** runnable research reference submitted for maintainer-operated full evaluation; no full official score or sealed result is claimed.

## Why this candidate

This is the strongest broadly evaluated **complete-probability quality candidate** in this project with a frozen, reproducible public-task run. The older Contrast label-selection pipeline scored 203/231 but did not provide a matching complete calibrated distribution on every path. More recent adapters improved narrow synthetic probability tasks but did not establish robust transfer or retention. Neither is substituted here. We have not established which project variant maximizes the current four-axis composite; this choice prioritizes verified general-task quality with complete probabilities, not an unmeasured claim of best speed or cost.

## Exact inference

- Base: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; original weights unchanged, no trained adapter.
- A single ordinary reasoning draft: greedy, thinking disabled, at most **480 new tokens**. Uses the existing `contrast_v7.generate` and `evidence_v4.messages(..., 'reason')` without modifications.
- An **unconditional** second forward readout receives the original evidence plus full draft. It uses `handoff_v5.readout`, projecting BF16-backbone hidden states onto allowed answer-code output rows in FP32.
- The emitted distribution is conditional softmax over those code logits with fixed temperature **1.0218971486541166**. It is not a one-hot encoding of the generated final answer, and not the probabilities of verbalized numerical strings.
- The selected answer is derived from that same distribution; exact ties use lexicographic label order.
- No numeric solver, task-ID lookup, cached benchmark answer, route selection, online learning, new temperature fitting, or benchmark-key access occurs at inference.
- Up to 16 labels; Score supports levels 0..K-1 for K<=10; Noul uses no/yes or false/true. Generation input plus 480 and the final readout must each fit the existing 16,000-token bounds. Unsupported requests fail explicitly, never silently truncate.

The new boundary verifies SHA256 for all four original inference-source files at startup. The original runtime pins and verifies both model-weight shard hashes. Its historical module header mentions a no-generation experiment; **this submitted path does generate** as described above. Only its weight loader and selected projection are reused.

## Historical evidence, not a new v1.4 result

From the preserved Probability v13 experiment on benchmark commit `7128f5cf445ca41e4da1bdc7c84f97926250724d`:

| Population | Correct | Calibrated target log loss | Brier loss |
|---|---:|---:|---:|
| 231 public JevBench items | 201/231 (87.01%) | 0.453916 | 0.210474 |
| Earlier 64-item BBH sample | 46/64 | 0.830447 | 0.432903 |
| Separate 73-item source check | 67/73 | 0.375531 | 0.171039 |

These data have been repeatedly inspected during project development. They are not fresh confirmation or a complete JevBench score. The archived public Jev result was 200/231; that one-answer difference is not a matched live deployment comparison or established superiority.

The historical measured complete CPU phase sum averaged **117.96 seconds** publicly (generation and readout partly cross-run). This implementation is not optimized GPU serving, and we do not claim Jev-class latency or competitive current cost. The current harmonic-mean score and speed/cost penalties may strongly penalize it. Self-hosted price is **unknown**, not zero. Both price fields remain null; a zero budget reservation merely means there is no metered provider account.

The original report is `PROBABILITY_V13_COMPLETED_RESULTS.md` on this branch. Original inference run: https://github.com/Jaksenc/parameter-golf/actions/runs/35781209019 . The September 25 submission audit replayed all **441 recorded full-path distributions** with zero vector or answer changes, including 201/231 public correctness. This is offline equivalence, not 441 new neural calls. The boundary has **24 unit tests**. Any live smoke is identified separately by its CI receipt.

## Calibration and exposure disclosure

One scalar temperature was selected from the previously fixed 161-point grid on **73 source-development examples only**: 32 SciQ, 32 BoolQ, and 9 authored probability tasks. The 73 source-check examples and all 295 public/BBH examples were excluded from fitting. Fit-record hash: `d23db64c1bcd5bc5b153014261dd9d06043051da9c811ff41be54d34af04de27`.

No pretrained weights are trained by this candidate. Nevertheless, public JevBench inputs, labels, and error analyses were used repeatedly to develop and choose inference methods. We explicitly disclose **public-benchmark development exposure**; this is not a benchmark-blind method. Unknown pretraining exposure is not ruled out. We have not accessed or evaluated the maintainer's 308 sealed questions/keys. Do not give us those private files or use them to tune this entry.

SciQ source terms are CC-BY-NC-3.0; BoolQ source terms are CC-BY-SA-3.0. The scalar and this experimental package are supplied for research evaluation, not commercial-use clearance. Qwen weights remain subject to their upstream terms. No base weights, source-dataset text, private task text, credentials, or model API keys are redistributed in the submission additions. Existing repository and third-party licenses continue to apply.

## Run on evaluator-controlled hardware

Use a Linux CPU with BF16 support and sufficient memory for the unquantized 4B model. The reference uses four CPU threads. A GPU port has not been validated for this submission; hardware-dependent results must be identified separately.

```sh
# Check out the immutable submission commit linked in the benchmark request.
git clone https://github.com/Jaksenc/parameter-golf.git decision0
cd decision0
git checkout <SUBMISSION_COMMIT_FROM_ISSUE>
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-submission.txt
python -m unittest -v test_decision0_submission

# Keep the official harness in a separate directory. This snapshot was inspected
# for the integration; maintainers may use a newer compatible official snapshot.
git clone https://github.com/fstandhartinger/jevbench.git ../official-jevbench
git -C ../official-jevbench checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a
export PYTHONPATH="$PWD/../official-jevbench${PYTHONPATH:+:$PYTHONPATH}"

# These task files belong to the evaluator. Outputs must stay private and outside
# either repository. Pass the complete current suite for an official full run.
python run_decision0_jevbench.py \
  --tasks /evaluator/private/legacy.jsonl,/evaluator/private/sealed.jsonl \
  --out /evaluator/private/results/decision0-probability-v13
```

This uses the official `Runner`, scoring and ledger unchanged, with a small in-process `Decision0Adapter`. The adapter copies only `id/state/question/labels` from the task. The id is not included in model prompts. Explicit `task.labels` order is preserved: reconstructing labels from sorted Choice criteria would change 119 historical public inputs' code maps. Do not replace this transport by a TypeSafe wrapper that discards explicit label order.

For direct integration, import `Decision0Adapter` from `decision0_probability_submission`, call `load()` once before timing, then `run(task)`. The evaluator controls the tasks, raw result directory, scoring, cost model and publication. A run manifest never infers that a full suite was measured merely because some number of items completed.

Optional loopback service:

```sh
python decision0_probability_submission.py serve --port 8765
```

It exposes `GET /health` and `POST /run` with `{"task": {"id": ..., "state": ..., "question": ..., "labels": [...]}}`, compatible with the official `RemoteInprocAdapter`. Metadata is stripped before inference. It binds only to loopback, has no prompt log or cross-request answer cache, and executes sequentially because the historical projection head is mutable. The remote adapter's 120-second default can be too short for this CPU reference; the in-process driver has no such HTTP timeout. It is not a public hosted endpoint.

## Requested official action

Please evaluate the pinned candidate on the complete current JevBench protocol, including all evaluator-held legacy and sealed decisions, and measure Intelligence, Calibration, Speed and Cost under the same published rules as other entries. Private question text, keys and per-item sealed outputs should remain on evaluator-controlled hardware. Only a completed maintainer measurement can determine official eligibility, score and rank. This package does not claim any of those in advance.
