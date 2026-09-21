# Orbit v2 — completed results, September 21, 2026

## Result: not promoted

The validation-selected method, synchronize_0.1, scored **184/231 (79.65%)** on reused public JevBench, versus **182/231 (78.79%)** for the calibrated native readout. On the new executable stress test it regressed from **40/128 (31.25%)** to **31/128 (24.22%)**. This is not a general upgrade.

| Method | JevBench correct | JevBench NLL | JevBench Brier | Fresh correct | Fresh NLL | Fresh Brier |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated native | 182/231 | 0.498294 | 0.275973 | 40/128 | 1.246270 | 0.713267 |
| Two-order arithmetic mean | 180/231 | 0.501575 | 0.276255 | 38/128 | 1.211481 | 0.696614 |
| Three-order arithmetic mean | 184/231 | 0.501500 | 0.279168 | 38/128 | 1.225683 | 0.705431 |
| Three-order geometric mean | 183/231 | 0.500353 | 0.277757 | 39/128 | 1.223426 | 0.702302 |
| Orbit primary | 184/231 | 0.491134 | 0.274848 | 31/128 | 1.236728 | 0.709193 |

Primary JevBench tier scores: 48/48 easy, 67/72 standard, 69/111 hard. Published Jev 1.13.0 scored 200/231 on identical public IDs; these are archived benchmark-author outcomes, not new API calls. No Jev reference exists for the new generated tasks. No full 534-decision leaderboard composite, production speed/cost or submission is claimed.

The primary corrected 6 native errors and introduced 4 on JevBench. Its +0.866 percentage-point change has a diagnostic 95% group-bootstrap interval [-1.732,+3.525] with 10,000 resamples of 195 source groups. Against matched-compute three-order averaging it had the same accuracy and a small NLL difference whose interval includes zero.

On the fresh set it corrected 5 and broke 14, a -7.031-point change; item-bootstrap interval [-14.063,-0.781]. These intervals are conditional on the reused public set or four generated task templates, not general natural-language guarantees. Exploratory intervals are not multiplicity-adjusted.

## New mechanism and frozen selection

Orbit tests y_v = s + P_v b + e_v, separating centered candidate-aligned evidence, presentation-slot nuisance and unexplained interaction. It solves a regularized constrained least-squares problem per input, with a training-only bias prior.

Identity plus reversal identifies floor(K/2) bias directions; identity plus a one-step K-cycle identifies K-1 centered directions under the additive hypothesis. A third distinct view supplies a lack-of-fit diagnostic. Binary reversal equals the cycle and is not counted as an independent third observation. This is a mathematical property of the proposed factorization, not a recovered Jev architecture or a guarantee that the invariant component is correct.

Eleven candidates were preregistered: native, two/three-order arithmetic averaging, geometric averaging, two global-bias strengths, and five ridge settings. Global priors used 301 training inputs without answer labels. Selection used 73 development inputs; temperatures used a separate 73 calibration inputs. The primary was selected by development cross-entropy, then publicly frozen before evaluation. All eleven arms remain in the release; no alternative is retrospectively promoted.

Selected ridge: 0.1. Temperature: 1.0717734625362931. Native comparison temperature: 1.1755479062836087. Its probability metrics differ slightly from v1 because this experiment recalibrates on 73 single-order inputs, whereas v1's native temperature fit used both order augmentations. Native logits and decisions are unchanged.

## Fresh stress test and diagnosis

128 independently generated, reference-checked tasks: 32 each of ledger arithmetic, sequential scheduling, pointer traversal and repeated affine modular updates. Seed 902731; 2–5 candidates. Neither their answer labels nor outcomes were used for selection or calibration.

Native vs primary by family: ledger 7/32 vs 6/32; schedule 9/32 vs 7/32; pointer 9/32 vs 7/32; affine 15/32 vs 11/32. These concern the tested non-generative direct-readout configuration, not the backbone's maximum capabilities with explicit reasoning or tools.

For K>2, median unexplained order-variation fraction was 24.56% on 157 public items and 34.94% on 96 fresh items. Binary cases have no residual degrees of freedom and are excluded. These are lack-of-fit measures, not correctness probabilities.

39 fresh tasks were unanimously wrong across every tested native ordering. At least one order was correct on 73/128, but that union is not an upper bound on other estimators. Both missing computation and unreliable selection across views remain plausible failure mechanisms; the data do not uniquely attribute causation.

## Execution and verification

760 new neural records completed across 16 ARM CPU workers, plus 16 readout fixtures and 16 native anchors. All anchor logits exactly matched the prior run. Each worker verified both pretrained weight hashes. Main was unchanged.

36 mathematical, generator and facade unit tests passed. The facade tests use an injected runtime; the real neural extractor separately ran live. An independent full-design-matrix solver and independent probability arithmetic checked all 3,949 evaluation distributions, max error 2.9421e-15. Aggregate metric error was at most 2.2204e-16. Offline refitting reproduced the selected arm, numerical priors, temperatures and all predictions exactly.

The primary requires two forwards for binary decisions and three otherwise: mean 2.680 on JevBench, 2.750 on fresh tasks. There is no shared-prefix optimization or new backbone training. More passes did not establish a useful upgrade.

## Provenance

Protocol commit: 539ab40a6a03d2ec5dd778ca805a4d440d3b8d38.
Implementation lock: 704be06486031887fbc74cd2f74f0f66fe063978.
Selection freeze: 7eab3588f5b8ddc4ed06a5789f0421de3c5788f7.
Frozen model SHA256: a5c9e9fd57f8531d4df936bc682db7eb064d35e8595ff9be47a8d7d7e0c60757.
Execution: https://github.com/Jaksenc/parameter-golf/actions/runs/35617181813 .
Completed probe artifact: 10647746294, SHA256 1dbce74898ece615fa5c291b708bc60d80e95cf1fba935ed27577311af32b196.

## Novelty and limits

PriDe (arXiv:2309.03882) already studies option-ID priors and cyclic permutations; Set-LLM (arXiv:2505.15433) studies architectural invariance; graph-potential decomposition has established mathematical prior art (arXiv:0811.1067). Orbit's newly engineered contribution within this project is a rank-aware sparse intervention, per-input regularized joint estimator and residual diagnostic, not a literature-wide priority claim.

The complete source, fitted parameters, all eleven arms, predictions, original and new records, tests, mathematical derivation and replay instructions are in the conversation release. No recovered Jev weights, RLCD training, full hidden benchmark, commercial-use clearance or hosted deployment is claimed. The old public benchmark and small development/calibration sets were previously used. The new four-template stress test is not a comprehensive generalization claim. Original rubric/semantic IDs remain in prompts. The scientific next hypothesis is intermediate-state supervision or independently verifiable computation, not another claim that reshuffling answers creates reasoning.
