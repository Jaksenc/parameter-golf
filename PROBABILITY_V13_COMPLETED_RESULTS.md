# Probability v13 — complete measured distributions, not solved calibration

Completed September 22, 2026. This continuation recovered the already completed neural experiment, fitted the three predeclared scalar temperatures, audited all outcomes, and packaged the release. It launched no duplicate neural inference and updated no pretrained weights.

## Main result

The primary is ordinary480 reasoning followed unconditionally by the existing Handoff source-plus-draft all-option readout. Its reported answer is the argmax of the actual reported probability distribution, even when that differs from a completed textual FINAL answer. All 441 inputs have measured native, full-draft, and FINAL-line-masked logits. The masked path is a diagnostic, not a promoted alternative.

| Matched population | Native probabilities | Ordinary480 text control | Full probability readout, primary | FINAL-line masking, diagnostic |
|---|---:|---:|---:|---:|
| Public JevBench | 182/231 | 202/231 | 201/231 | 201/231 |
| Earlier v7 external BBH sample | 36/64 | 46/64 | 46/64 | 47/64 |
| Separate source-check inputs | 68/73 | 67/73 | 67/73 | 63/73 |

All datasets are historical. The 295 public/BBH drafts are reused unchanged; their all-option observations are new. The 146 source drafts were generated in this neural experiment. Source textual controls use the full readout if FINAL is missing, so incomplete cases are not an independent native-fallback arm.

The primary repairs one public textual-answer error and introduces two: net -1/231. Its descriptive 95% paired source-group interval is [-2.09,+0.88] percentage points. Externally it repairs one and introduces one. Primary public tiers: 48/48 easy, 70/72 standard, 83/111 hard.

Archived Jev 1.13.0 remains 200/231 on identical public IDs. Full readout is correct on 15 Jev misses and misses 14 Jev answers. A one-answer lead is not a live comparison, full 534-decision score, statistical superiority or model equivalence. The 303 nonpublic decisions, production price, matched serving performance and full composite remain unavailable. Historical Contrast203/231 cannot be attached to this different probability-output interface.

## Calibration and probability quality

Each path fits its own positive temperature only on the same 73 source-development inputs, preserving soft targets. The other 73 source-check and 295 public/BBH records do not fit parameters. Exact canonical input overlap is zero, not a guarantee of task-family or pretraining disjointness. The 301 older source-training examples are unused here. Fixed grid: T=2^(i/32), i=-64 through96, including identity and spanning0.25 to8. Parameters were persisted before test reduction.

Selected T: native1.4142135623730951; full1.0218971486541166; masked1.241857812073484. No grid boundaries were selected. A post-outcome source-only bootstrap shows broad full-path temperature sensitivity, approximately0.738 to1.354 at its2.5/97.5 percentiles. It does not replace the fit or establish target-domain calibration.

| Population | Native calibrated target log loss | Full calibrated target log loss | Native Brier | Full Brier |
|---|---:|---:|---:|---:|
| Public231 | 0.496864 | 0.453916 | 0.255484 | 0.210474 |
| External64 | 0.963687 | 0.830447 | 0.503388 | 0.432903 |
| Source check73 | 0.256920 | 0.375531 | 0.099929 | 0.171039 |

Losses use supplied gold distributions for probability targets, not one-hot modal labels. Public full-minus-native target log-loss difference -0.04295 has a descriptive interval [-0.13984,+0.05889]; Brier difference -0.04501 has interval [-0.10380,+0.01500]. External intervals also include zero. Source-check losses worsen. These historical, unadjusted finite-sample means do not establish uniform or general probabilistic improvement.

The full-path temperature is nearly identity: public raw/calibrated loss0.455591/0.453916, with a tiny difference whose interval includes zero. Source-check raw/calibrated loss0.374652/0.375531 slightly worsens. A single scalar did not solve probability estimation.

## A correct distribution in the draft is not preserved automatically

Public item hard-opus-b-probability-02 has relevant counts4,13,3 out of20 for bad push, upstream-provider slowness and hardware fault. The draft computes20%,65%,15% correctly and selects upstream_provider. The calibrated full readout emits4.02%,46.91%,49.07%, switching to hardware. The masked readout emits6.09%,57.23%,36.69%. Independent source-count and code-label mapping checks rule out a simple reporting-order error; no oracle was used to patch predictions.

Across the ten exact-distribution public items, mean TVD is0.251326 for calibrated native,0.281317 for full,0.217897 for masked. The primary worsens that small distribution-estimation subset despite better aggregate proper-loss means publicly.

The readout prompt asks for the single best answer and scores A/B/C decision tokens. Those are genuine neural code preferences; they are not automatically calibrated frequencies of mutually exclusive source events. Being confident that a cause is the most likely does not mean that cause occurs with near-certainty. This identifies a probability-semantics/training-objective concern, not a proven sole cause of every error. Positive temperature scaling cannot fix a reversed ranking.

Another public regression raises a cosmetic-only incident from ordinal level0 to1 despite a correct draft. Conversely, the readout repairs a wrong draft on newsletter consent. It is not simply copying all textual answers.

## Masking remains diagnostic

Removing only explicit FINAL lines changes431/441 drafts; the other10 produce identical readout logits. Earlier answer cues remain. Calibrated public log loss falls from0.453916 to0.399981 and Brier from0.210474 to0.186890, with unchanged201-correct total. There are two repairs, two regressions and one wrong-to-wrong answer change. Public paired masked-minus-full log-loss interval[-0.09086,-0.01901]. Raw comparisons at common T=1 also improve, from0.455591 to0.413561, but calibrated comparisons additionally change temperature.

Externally masking changes46/64 to47/64. Source-check accuracy falls67/73 to63/73: four regressions, zero repairs. No masked-path gate or primary replacement was selected using these outcomes. This is not a verified de-anchoring or generally harmless postprocessor.

## Cost

Public mean phase accounting rises103.348s ordinary480 to117.957s full probabilities (+14.14%); external101.233s to110.160s (+8.82%); source-check62.539s to69.157s (+10.58%). Public/BBH combine archived generation and new readouts, not a new integrated serving benchmark. Source phases run within this experiment. Native screening is not charged to the primary. These four-thread ARM CPU values exclude loading and do not claim GPU, production-price or Jev-endpoint equivalence. No speedup is established.

## Execution and failure recovery

Retained new neural work:441 native passes,882 all-option readouts,146 source generations and12,387 generated tokens.295 old evaluation drafts remain unchanged. Original worker19 failed its native anchor before saving any task. Its failing values were not logged, so cause and magnitude remain unknown. Three diagnostic repeats matched the archived anchor exactly. Six recovery workers completed only18 missing inputs with unchanged tolerance; all423 successful original rows are preserved.29 successful worker receipts agree on model hashes and exact native/projection fixtures. A static base-class no_generation=true field is stale for this wrapper; actually_generative flags and recorded traces establish the real calls.

This continuation did source-only fitting, scoring, audits and saved-record CLI execution, not another neural benchmark. The live wrapper is injected-runtime tested; the underlying generation/readout functions ran in the completed experiment. No separate live standalone neural replay is claimed here.

The original locked scorer encountered a NumPy Boolean JSON-serialization error locally. A separate replay_v13.py converts only NumPy scalar types at export. All four pre-outcome locked source files, numerical methods, fit and predictions remain unchanged. The failed export is not counted as successful validation.

## Verification and release

52 tests pass:37 original plus15 replay/interface tests. Independent50-digit Decimal arithmetic checks2,646 raw/calibrated distributions, reconstructs441 targets and repeats all three temperature choices. Maximum probability difference7.78e-16, aggregate metric difference1.12e-16. The official scorer validates1,386 public distribution outcomes.441 native logits/prompts and12 prior fallback vectors/prompts match parent observations exactly. Independent provenance replay covers423 originals,18 recovered,146 parent-source and295 parent-evaluation inputs, plus882 code-position maps.

Three actual saved-record CLI cases exercise Choice, Score and Noul. Expected ordinal levels and binary probabilities derive from measured vectors. No text answer is assigned an invented one-hot distribution or concentration mislabeled as correctness certainty.

A fresh extraction verified194 file hashes, passed52 tests, refitted identical calibration parameters and reproduced all12 result JSON files byte-for-byte under Python3.13.5/NumPy2.3.5. This is offline recorded-data replay, not full neural re-execution or semantic proof.

Release Probability-v13-Completed.zip SHA256:e60e2efd14f37df54dd592ffeea99162760da14fdb65664c531911dd19feae07.
Recovery artifact10718082590 SHA256:a72a69078e8aa894be273ed5e6e04e147c55f7e2ac9c0eeb952ad9feaad037a6.
Main run35776811846, inference commit55e1c317077e80ac9a365281b1512217e0e2e980.
Recovery35781209019, commit647b56295dd2b5f3525d3b02e3c5645f454cf5bc.
Diagnostic35780530571. Model Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Main unchanged; no official submission or hosted deployment. Derived calibration data retain underlying source terms, including SciQ noncommercial supervision; no pretrained weights are distributed.

## Decision

Retain the measured complete probability baseline, not a claim of calibrated-model superiority. Before distillation, distinguish best-answer uncertainty from event distributions, test whether known stochastic targets survive readout, and compare changes on new data with proper losses and actual computation. Blindly distilling current vectors could teach readout errors as well as useful reasoning. Then compare ordinary direct supervision against rationale-assisted adaptation with matched training budgets. No new training result is claimed.

Primary references: Guo et al., On Calibration of Modern Neural Networks (arXiv1706.04599); Yoon et al., Reasoning Models Better Express Their Confidence (arXiv2505.14489). Their confidence-elicitation setting differs from this restricted-code readout; this is not a replication or literature-wide novelty claim.
