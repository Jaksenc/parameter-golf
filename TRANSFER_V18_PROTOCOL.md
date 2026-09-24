# Transfer v18 — pre-outcome protocol, September 24, 2026

Goal: test whether v17.1's direct-learning gain transfers without retraining or recalibration and whether unrelated decision capability is retained. No new optimizer updates, drafts, judges, Jev API calls or answer-conditioned routes. All six completed step32 high-rate adapters are included: event_high and mixed_high, seeds17101/17102/17103. Unchanged frozen Qwen3.5-4B is measured on every input. No best seed selection or ensemble averaging of outputs. Main is not modified.

## Frozen population and contrasts

24 new generated probability worlds: four existing mechanism families (counts, conditioning, mixture, Bayes), with outcome counts2,3,4,5,6,8. Counts6/8, vocabulary and the row-oriented grammar were absent from v17 training. Each world supplies five requests: familiar source rendering; new row-oriented source; uniform replication of sample-I counts; a relevant addition of19 sample-I items in its first category; and the modal-answer question. The first four ask for event distributions, the last for the uniquely most likely category. Primary base worlds have unique modes; optional edit ranking uses first-maximum if a tie occurs. Two count strata additionally have a zero sample-I count; exact zero event targets occur in counts/conditioning/Bayes but not necessarily mixtures.

The row-oriented rendering introduces more irrelevant fields as well as wording/organization changes. Its contrast with familiar rendering is therefore a compound representation/distractor intervention, NOT a clean paraphrase-only causal effect. New vocabulary and category counts are shared within each pair. These remain project-authored variants of known mathematical mechanisms, not independently authored probability language or an independent-task-family benchmark.

56 separately authored deterministic retention items:8 per seven BBH families. Formal fallacies are excluded because that source was exhaustively inspected in prior specialist work. Retained source commit9ee07bd481feebf959a6b59d61ea57bdcf30964d, validated git blob hashes. Selection seed180924271; fixed exclusions were reconstructed from21 input-bearing members in five prior conversation releases,1524 distinct source strings. Exclusion uses normalized source-stem matching without answer outcomes. The exclusions cover those recovered project evaluations, not unknown other conversations or pretraining. Public BBH training contamination is unknown. All source answer keys remain unchanged.

Total176 requests,1232 direct distributions (seven model states). There are24 independent generated worlds and56 external items, not176 independent mechanisms or1232 independent test examples. Generated groups stay together in resampling. No new public JevBench score is claimed.

## Probability calibration and endpoints

Temperatures are copied exactly from v17.1's separately calibrated32-world split and frozen in source before this experiment. No transfer input, answer or outcome selects a temperature or checkpoint. Every output is scored raw and with its pre-existing requested-quantity temperature. Applying the modal-question temperature to external BBH is explicitly a secondary transport diagnostic, not an independently calibrated external classifier.

Primary: frozen-calibrated event_high minus unchanged mean target CE and squared-vector error on the24 new row-oriented base event requests. Mixed_high is a fully reported comparator, not a post-outcome replacement. Report raw losses/TVD and per-seed results, plus outcome-count/family strata with their small denominators. Report uniform reference. No literal-zero smoothing; if positive target mass receives zero report strict infinite loss explicitly.

Retention guardrail: raw allowed-label accuracy on56 BBH items, repairs/regressions and per-seed results, proper losses separately. A nonsignificant loss or no observed accuracy change is not a noninferiority certificate. No tolerance or production-approval claim is retrofitted from results.

Replication consistency and response-to-relevant-edit residuals are diagnostics, anchored to independently calculated target changes. Absolute accuracy/proper losses remain primary; constant wrong distributions can satisfy invariance. Familiar-versus-row-oriented probability differences are also diagnostic.

Uncertainty:10000 crossed paired-seed and world resamples within the four generated family strata; external resample paired seeds and items within seven source-family strata. Three seeds and small strata limit population interpretations. No adjustment for multiple endpoints, repeated project research, or unknown source dependence. Calibrated intervals are conditional on prior temperature estimates.

## Execution integrity

Same pinned pretrained revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and v17 semantic_codes readout. Same BF16 backbone/FP32 head and low-rank factor implementation, no generation.16 standard public-repository ARM CPU workers,11 matched tasks each. Each verifies two base weight shards, full/restricted head fixture, a saved baseline anchor, all six saved adapted-anchor outputs, and baseline restoration. Checkpoint files are hash-bound. Input cap1600 tokens, no silent truncation; errors stop visibly, no output omission or baseline substitution.

Only validated id/state/question/labels are inference inputs; targets and mechanism metadata are separate. Model call order varies deterministically per input to reduce ordering-time confounding. Every complete seven-state record is fsynced. Timing is per forward, excluding model load, adapter switching, tokenization, unused controls and serving overhead; no production latency or cost comparison.

Preflight:18 local tests passed, including source-text-independent finite-ticket target checking on generated scenarios and additional generator seeds. Frozen source SHA256d265a8761e7670fcfa721a1f24dd94a2b01197b9fb3d797a4eaf3f88c8af4608; label-free jobs hasha70f10784224deb6da8c991979c3d0d8b2b6aee787566c6fcd6ad03ee1d995c9; evaluation hash5d5ff0cf6434de8d945a2eebc42b1cebffa2a1555b2cfebb48949514f13bdf90.
Parent clean artifact10777138863/run35912706972 SHA25698c6a1a3f0e4f39429e8605cc7944513b2cbf0f905390e65e7338a08a4a3962d. Six final factor hashes and seven old temperatures are encoded in scientific source. All test outcomes, including regressions, will be retained. No literature-wide novelty or recovered-Jev equivalence claim.
