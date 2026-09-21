# Jev-like reconstruction v1 — completed results

Completed September 21, 2026 UTC (September 20 New York). Main branch unchanged.

## Result

Primary independently trained reconstruction: **183/231 (79.22%)** on all public JevBench items. Fresh native Qwen3.5-4B restricted readout control: **182/231 (78.79%)**. Pinned published Jev 1.13.0 outcomes on identical public IDs: **200/231 (86.58%)**. This is not a full official leaderboard score or submission, and it does not match Jev.

| Configuration | Correct | NLL | Multiclass Brier | Ten-bin ECE |
|---|---:|---:|---:|---:|
| Native readout | 182/231 | 0.5122 | 0.2806 | 0.0536 |
| Native plus independently fitted temperature | 182/231 | 0.4950 | 0.2744 | 0.0252 |
| Trained head plus temperature, predeclared primary | 183/231 | 0.5340 | 0.2871 | 0.0498 |
| Trained head plus temperature and two-order averaging | 182/231 | 0.4933 | 0.2734 | 0.0438 |

The head fixed four baseline errors and introduced three. The accuracy change is +0.43 percentage points, with a diagnostic 95% group-bootstrap interval of -1.76 to +2.64 percentage points (10,000 draws, 195 groups). Its proper probability losses worsened. There is no established meaningful quality improvement from head training.

| System | Easy | Standard | Hard | Total |
|---|---:|---:|---:|---:|
| Reconstruction primary | 48/48 | 67/72 | 68/111 | 183/231 |
| Published Jev 1.13.0 | 48/48 | 71/72 | 81/111 | 200/231 |
| Published SemIf | 48/48 | 71/72 | 68/111 | 187/231 |

External rows are archived published results, not new Jev API calls. The reference is results/v1.2/jevbench-v1.2-per-task.json at benchmark commit 7128f5cf445ca41e4da1bdc7c84f97926250724d; its Git blob is f9993fc449faf54ad61729f692504faf33d0e7ba. The entire 231-ID population is matched. Full JevBench contains 534 decisions; 303 nonpublic decisions, production performance and pricing were not evaluated.

## Reconstruction boundary

A frozen Qwen/Qwen3.5-4B backbone (851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), FP32 permitted-answer projection, learned bounded residual probability head, independent temperature fitting, and deterministic typed Noul/Choice/Score outputs. No autoregressive answer generation, Jev API calls or Jev-output distillation. This is not recovered Jev weights, architecture or RLCD.

The residual matrix has 40,960 slots, 10,240 nonzero trained values. The selected checkpoint is step 192 of 256. Temperature: 1.122462; native calibration control: 1.289370. Checkpoint SHA256: a4802566ef90a1d6a654d059f94202c92ed381e04083272f120f041d5bd0659b.

Corrected data: 301 unique training, 73 development, 73 calibration inputs; two candidate-order augmentations per input. A pre-outcome audit removed 33 repeated synthetic inputs. SciQ/BoolQ and independent executable tasks supplied supervision. This is research-only because SciQ supervision is noncommercial.

## Diagnostics and verification

The hardest primary families were temporal/numeric (4/15), multi-hop (9/18), and long-policy (10/19), accounting for 29 of 48 errors. A bounded plus/minus-one residual provably cannot correct 14 native gold-logit gaps greater than two. Reversing candidate order changed 27 native decisions; averaging did not improve accuracy.

All 1,422 native hidden/logit records completed across 16 ARM CPU workers. Every worker checked both model-weight SHA256 hashes and a restricted-readout versus full-vocabulary FP32 fixture. All 16 fixtures matched exactly. The 36 local tests passed. Independent FP64 replay checked 924 main-arm distributions with maximum absolute probability error 4.30795e-7.

The initial corrected fitting job failed before fitting because packaging was missing. Complete features were recovered; fitting completed locally. After dependency repair, a separate remote refit reproduced the trained checkpoint byte-for-byte. Metric reductions agree within 3.89e-16.

Extraction run: https://github.com/Jaksenc/parameter-golf/actions/runs/35554698513

Successful unchanged fit replay: https://github.com/Jaksenc/parameter-golf/actions/runs/35555975138

Artifact: jev-reconstruction-verified-results (10620755716). The conversation release additionally contains all features, source files, typed facade, tests, controls, predictions and reproducibility documentation.

Native CPU forward latency: median 5.950 seconds, p95 48.654 seconds; excludes learned-head overhead, model loading and production network. No GPU or Jev-production speed equivalence is claimed.

Limitations: public benchmark reused from earlier experiments, not a fresh sealed test; single seed; exact deduplication is not semantic/pretraining contamination exclusion. The real backbone uses serial isolated questions, no shared-prefix optimization; Score levels are jointly encoded; Choice cap is 16, not 255. The typed facade is separately unit-tested, not claimed as a live production endpoint.
