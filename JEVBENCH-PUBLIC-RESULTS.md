# Duplex — completed JevBench public comparison

20 September 2026. All 231 public items and all four prespecified configurations completed: 924 primary predictions, no model/schema errors, no truncation. Maximum observed serialized input: 4,235 tokens. The 231 FP32 SemIf shadow distributions reuse the same forward passes and are not a fifth inference run.

Main inference run: https://github.com/Jaksenc/parameter-golf/actions/runs/35540256799 . Source commit `cffb438318eff0aa1e477c2fd77b18f1db841660`. Benchmark commit `7128f5cf445ca41e4da1bdc7c84f97926250724d`; SemIf commit `ca3ba65f142967030ecb453346e94d6f476a69df`; Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## Measured results

| Configuration | Easy /48 | Standard /72 | Hard /111 | All /231 | Mean log loss |
|---|---:|---:|---:|---:|---:|
| Unchanged 4B, Duplex scorer | 48 | 63 | 63 | 174 (75.3%) | 0.547 |
| Direct-answer adapter | 48 | 62 | 63 | 173 (74.9%) | 0.650 |
| Answer-plus-fact adapter | 48 | 65 | 60 | 173 (74.9%) | 1.493 |
| Unchanged 4B, SemIf native scorer | 48 | 66 | 70 | 184 (79.7%) | 0.508 |
| SemIf FP32 same-forward control | 48 | 66 | 68 | 182 (78.8%) | 0.512 |

The native SemIf interface repairs 16 baseline errors and breaks 6 correct answers, net +10. Its FP32 control repairs 15 and breaks 7, net +8. The direct adapter repairs 12 and breaks 13; the fact adapter repairs 3 and breaks 4. Neither trained checkpoint earns promotion.

The fact adapter's hard-tier log loss is 2.631 versus base 0.931. Native SemIf improves hard accuracy while hard log loss is slightly worse at 0.947; the result is not uniform superiority in every metric. Native SemIf has four exact top ties; official lexical tie-breaking was retained. Four choices differ between its native and FP32 output projections, with a net two-correct native advantage on these cases. This is not evidence that lower precision generally improves reasoning.

## Scope and uncertainty

This is the public subset, not a rank or score against the complete 534-decision board. All four configurations share the same CPU-loaded base weights and complete inputs; SemIf uses unmodified upstream prompt and direct scoring functions, not its published CUDA/MLX deployment. The two learned adapters use our base scorer's identical prompt and numerical path. No model training, calibration fitting, prompt search or test-based checkpoint selection occurred in this benchmark.

An independent standard-library replay verifies all 924 native distributions, input identities, scores and repair/harm counts and agrees with the pinned official scorer. The 231 tasks form 195 provided source groups. A paired tier-stratified 5,000-resample group bootstrap gives an approximate conditional 95% interval of +0.43 to +8.23 percentage points for native SemIf minus the base scorer, and -0.43 to +7.36 for its FP32 control. These are not simultaneous testing guarantees, seed uncertainty or proof of general-domain superiority.

Observed median per-call times: base 4.58s, direct adapter 4.63s, fact adapter 4.62s, native SemIf 5.95s. P95 is roughly 48–50s. These are serial calls within sixteen ARM CPU VMs, excluding model loading, HTTP/network transit and service overhead. They are not Mac or production GPU benchmarks. No monetary cost or leaderboard composite is invented.

## Published reference alignment — not rerun

The pinned maintainer artifact `results/v1.2/jevbench-v1.2-per-task.json` has reference outcomes for the exact same 231 public IDs. Jev 1.13.0: 200/231 (86.6%), including 81/111 hard. Maintainer's SemIf: 187/231 (81.0%), including 68/111 hard. Our measured SemIf is 184/231, not the published 187. Different loaders, serialization/runtime and serving deployments preclude claiming an exact reproduction. No live Jev API call was made.

Reference source: https://github.com/fstandhartinger/jevbench/blob/7128f5cf445ca41e4da1bdc7c84f97926250724d/results/v1.2/jevbench-v1.2-per-task.json . Receipt run `35541136320`; git blob `f9993fc449faf54ad61729f692504faf33d0e7ba`.

## Interpretation

The strongest observed gain comes from the inference interface, not the two trained adapters. Make the SemIf-style interface the leading candidate for a fresh baseline comparison, with numerical precision explicitly fixed. The public tasks are now regression evidence, not a training or calibration corpus. No model or production policy is promoted by this report.

Hard-family signals to investigate on separate development data: temporal/numeric base 4/15 and SemIf 4/15; multi-hop 8/18 versus 10/18; long policy 9/19 versus 12/19; hard judging 10/17 versus 13/17. Small slices and post-hoc inspection limit generality.

Completed-audit ZIP SHA256: `26a4fa285e09d8e8c7eff7540969ca713e6f71ab5a24b165c207329f918a4dfa`. Input population SHA256: `3801797ad34852a40869c051c5b7727c852b436f9ad5504ce82d48fbcfb84edf`. The earlier transport run `35540061862` remains failed; recovery stripped GitHub API credentials from cross-origin artifact redirects without changing the frozen inference source or task/scoring protocol.
