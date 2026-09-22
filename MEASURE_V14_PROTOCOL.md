# Measure v14 — frozen mechanism protocol, September 22, 2026

Goal: distinguish incorrect distribution calculation from a readout that fails to represent the requested random variable. This follows Probability v13's correct-draft/wrong-vector failure. No distillation or model-weight update is performed here.

## Fixed primary and controls

Primary: numeric_mass. The same frozen Qwen3.5-4B checkpoint returns a short JSON array of nonnegative numerical weights in the declared label order. Deterministic code normalizes the weights by their sum. These are model-expressed probability estimates, not the softmax probabilities of the tokens used to write them. A malformed, nonnumeric, negative, all-zero or wrong-length output falls back to the legacy code readout on exactly the same evidence/note; invalid generations stay in every denominator. Report parse validity and fallback-inclusive outcomes separately. No answer-key repair, threshold fitting, selection of an alternative primary, or fictitious confidence vector.

Control 1: unchanged Handoff single-best-answer code readout used by Probability v13. Control 2: the same code map and user bytes, but a fixed system instruction to represent a draw from the requested categorical law rather than always its mode. Code distributions are softmax at T=1 conditional on the allowed single-token code set; this experiment does not physically sample those codes or claim that all-vocabulary generation has the same distribution.

All three channels see the same original evidence and available note. Numerical output changes the system instruction and required output syntax, so it is not a clean one-word prompt intervention or equal-compute comparison. No temperature is fitted or selected.

## Factorial task design

32 newly generated source scenarios, eight each from four authored families: explicit probabilities; finite active-bag counts with a distractor bag; counts conditioned on a selected stratum; two-stage bag mixtures. Seed 141260922. Three exhaustive mutually exclusive labels; exact rational target distributions and unique modal labels. Current realized event targets are all strictly positive. Distribution mode positions are 10/13/9, and the smallest leading-probability gap is 0.01. The sample does not test every possible support/boundary case.

Each scenario has two questions with identical state/labels: the law of the unobserved next event, and the determinate identity of the unique most likely category under exactly specified parameters. The event target is q; the latter target is a point mass on argmax(q). A point-mass task target is not a claim that the model knows its answer with certainty or that its empirical self-confidence is calibrated.

Each question has source_only and reference_note conditions. The reference note intentionally supplies the correct event distribution, exact fractions, decimal approximations and modal category. This is an oracle-information/readout-fidelity diagnostic, never an end-to-end reasoning or autonomous probability-estimation result. The note is generated before neural evaluation and identical across the two semantic questions. The source_only primary contains no supplied solution.

Total: 64 questions, 128 input/note views, 256 new code readouts and 128 numerical generations with a fixed 96-token cap. Sixteen workers, eight views each. Parent native anchors/fixtures are extra verification calls, not benchmark samples. There are only 32 distinct underlying worlds and four authored families, not 128 independent tasks or a broad natural-language benchmark. No new JevBench score is computed.

## Analysis fixed before outcomes

Primary endpoint: source_only event-distribution TVD, numerical-mass versus unchanged legacy code readout. Also report vector squared error sum((p-q)^2), equivalent to excess expected multiclass Brier loss; mode correctness is a diagnostic, not the event-probability endpoint. Determinate-question outcomes must be reported to test semantic collapse. Report every arm/condition, exact-support violations, strict infinite log-loss counts, and a separately labeled clipped log loss with epsilon=1e-12. Do not hide zero probability on a possible event by silently clipping it.

Paired uncertainty uses 10,000 resamples of worlds within the four source-family strata. Semantic-pair and reference-note observations remain grouped. Intervals are descriptive, unadjusted for multiple questions/analyses, and conditional on this single model and authored generator. No external/pretraining-contamination claim.

The primary may fail. Diagnostics do not replace it. Reference-note improvement is not evidence of unaided computation. If a known distribution is supplied and preserved, count it as faithful extraction/transport, not newly discovered reasoning.

## Execution, boundaries and provenance

Use only standard public-repository CPU runners, pinned existing dependencies, no paid provider or Jev API. Model Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Parent artifact: probability-v13-recovered-complete, run 35781209019. All parent inference source hashes, both weight shard hashes, a native anchor, and the full/restricted projection fixture are checked per worker. Failed work is retained explicitly; recovery, if needed, may select only missing complete views. Flush and fsync each matched view.

Frozen scientific source SHA256: 1e74fb444e03b0e783b760d014cf27ef4db739e70769a0cc80e7f2271691280c.
Jobs SHA256: 232d5768433c644aa999ffe552299e903f7356fa748dcb5932f89b3b049e1a1a.
Evaluation SHA256: 379a31bb3f3a84129372f1b916fd358c23ffe4b779c01f86edb482428c579c0a.
Worlds SHA256: b9a65f850f05c78ec804c893ccc9580ff8e9975403e4da119430bd88a269e73a.
Pre-inference local tests: 38 passed. Independent ticket enumeration checks all 32 target laws. No new model output inspected.

Relevant primary research: Pournemat et al., Reasoning Under Uncertainty (arXiv:2509.10739); Baldelli et al., Probabilistic Calibration Is a Trainable Capability in Language Models (arXiv:2605.11845). Those distinguish estimation/sampling and distributional calibration; this is not their replication or a claim to invent numerical probability elicitation. The project contribution is the paired-estimand, shared-evidence and supplied-reference test before teaching current readout errors to a student.

No general model superiority, production cost/latency, authentic semantic proof, calibrated self-knowledge, exact Jev internals, or full benchmark composite is claimed. Main is unchanged.
