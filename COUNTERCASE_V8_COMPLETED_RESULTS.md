# Countercase v8 — completed and audited

## Decision: do not promote

The predeclared primary scored **197/231 (85.28%)** on reused public JevBench versus **202/231 (87.45%)** for ordinary 480-token reasoning. On 64 different, new-to-project BIG-Bench Hard items it scored **49/64 (76.56%)** versus the same-run ordinary control's **50/64 (78.13%)**. Countercase used about3.2 times the ordinary control's measured serial CPU phase time. No new weights were trained; this is a fixed system-prompt/inference experiment on Qwen3.5-4B, not reconstructed Jev/RLCD.

| Frozen method | Public same231 IDs | External same64 IDs |
|---|---:|---:|
| Ordinary480 | 202/231 | 50/64 |
| Countercase480, primary | 197/231 | 49/64 |
| Archived Jev1.13.0 reference | 200/231 | Not evaluated |

The previous Contrast primary's203/231 remains historical; this round does not improve it. Its old external48/64 is a different sample and cannot be directly compared with the new49/64. No current live Jev ranking, full534-decision composite, hidden-set outcome, production speed/cost/calibration win, API extraction, or official submission is claimed.

Public repairs/regressions:11/16; delta−2.16 percentage points, descriptive95% paired group-bootstrap interval[−6.61,+2.13], discordant-pair p=0.4421. External repairs/regressions:5/6; delta−1.56 points, interval[−10.94,+7.81], p=1.0. These do not establish a population-level decline or improvement. There is no basis to promote a much more expensive method. Intervals use10,000 draws,195 public source groups, and within-family paired item resampling externally; no adjustment for repeated benchmark use or multiple comparisons.

Public ordinary/Countercase tier counts:47/48 vs48/48 easy;71/72 vs72/72 standard;84/111 vs77/111 hard.

## What was tested

One fixed prompt asks the model to identify entity/time/quantifier/polarity, construct constraints or calculations, and try to falsify the leading answer and strongest rival without changing given evidence. It distinguishes contradiction from insufficient support and countermodels from actual events. No second generator, judge, voting, task-family routing, confidence threshold, training, or answer-dependent method selection.

Both arms share canonical user evidence, greedy decoding, a480-token output ceiling, strict completed-FINAL parsing, and source-plus-draft categorical fallback. System prompt and actual generated length differ. Public standard traces are reused intact fromv7; external both arms are live with fixed randomized execution order. Hybrid generated labels have no invented probabilities; fallback distributions are uncalibrated.

## Completion bottleneck, with causal limits

Public mean generated tokens:ordinary126.52 versusCountercase407.90. Public cap/fallback counts:ordinary9/231 versusCountercase121/231. Public mean CPU phase sums:103.35s versus323.78s, ratio3.13.

External mean tokens:ordinary142.84 versusCountercase458.64. Cap counts5/64 versus49/64; fallback counts6/64 versus49/64. External phase means106.59s versus341.44s, ratio3.20. Equal maximum ceilings are not equal actual computation. These four-thread ARM measurements include conditional readout, exclude loading/serving/lost work, and are not optimized endpoint timings. Public is cross-run descriptive, external within-run paired.

All16 public regressions occurred on capped Countercase drafts. Among110 finished Countercase public drafts,Countercase110 correct versusordinary108; among121 capped drafts,87 versus94. This is post-outcome, method-dependent stratification, not causal proof that truncation explains the regression or that more tokens fix it. External finished drafts are only9/15 correct, so completion is not a correctness certificate. No completion-based selector was fitted or promoted.

One public invoice case requires final half-up rounding after an unrounded discount/tax calculation. Exact arithmetic:19.95*0.85=16.9575;16.9575*1.0825=18.35649375;rounded=18.36. The supplied response said18.35. Ordinary reasoning correctly rejected it. Countercase restated constraints, checked intermediates and reached the rounding section, but was cut off before evaluating it; its fallback incorrectly accepted the response. The included Decimal audit independently confirms the arithmetic, not a general semantic diagnosis.

## Useful diversity, not a replacement

The fixed diagnostic found six newly correct answers among22 cases missed by BOTH v7 long320 and blind160. IDs:hard-opus-a-long_policy-04;hard-opus-a-long_policy-13;hard-opus-b-tradeoff-07;hard-sol-a-multi_hop-09;hard-sol-b-judge_hard-18;original-intent-02-0. The oracle union of those exact recorded pools rises209/231 to215/231. This is not an achieved ensemble, trained selector, forecast, or test-selected improvement. Sixteen cases remain without a correct candidate.

External family counts,8 each,ordinary/Countercase:causal7/7;dates8/8;pronoun3/3;formal fallacies6/8;seven-object deduction7/6;sports5/3;temporal7/8;shuffled tracking7/6. Tiny task-family cells do not validate a router.

Formal-fallacy example054:premises notF and F=>E do not entail notE. AssignmentF=false,E=true satisfies premises and falsifies the conclusion. Ordinary incorrectly says valid;Countercase's readout correctly says invalid. An audit-only truth-assignment check verifies this illustration; the Countercase draft itself was capped. This is not a general English-to-proof checker.

The motivating Failing to Falsify paper(arXiv2604.02485) studies interactive rule discovery with feedback. Countercase provides no external feedback on imagined counterexamples. Its negative result concerns this broad fixed prompt, not falsification as an epistemology. A valid countermodel is different from a plausible-sounding model-generated story. BBH reference:arXiv2210.09261. No literature-wide novelty claim.

## Recovery and audits

Original run35665066216 retained140 complete matched input records. Existing recovery run35668827839 completed155 missing records, preserving all140 originals. The original per-worker limit was35 minutes. The availability guard was widened after the unexpectedly large missing population; scientific source, prompts, cap, arms, and selection did not change. Recovery completed September21 New York time(September22,01:40 UTC).

This continuation located the already completed recovery artifact and performed no additional neural calls. An unnecessary duplicate recovery helper created during rediscovery was removed unexecuted; no duplicate recovery workflow ran. Main is unchanged.

Retained new inference:359 generated outputs/132,721 generated tokens;176 new categorical readouts;295 unique inputs;231 archived standard public traces. Counts exclude unfinished/lost work. All32 original shards lacked completion receipts;all32 recovery shards completed.

Current verification:53 passing unit tests(the45 locked tests plus8 recovery tests), with10,000 deterministic parser fixtures inside one test. All5 locked source hashes match. Independent parser/control replay reconstructs590 labels;official scorer agrees on462 public-arm outcomes. Separate probability arithmetic checks185 distributions including9 archived, max error2.92e-16. All64 original/recovery worker receipts agree on pretrained weight hashes and exactly matching native anchors/full-restricted fixtures. Every original/recovered row is mapped back to its planned input;all231 archived controls match thev7 release exactly.

All64 external source texts, keys and selected indices were independently replayed. The exclusion reference contains421 v6 inputs plus295 v7 inputs. These new-to-project public examples are not a private or pretraining-contamination-free holdout. Source fidelity is not independent proof of all original answer keys.

The standalone wrapper is unit-tested;the actual scientific solve function ran in the recorded experiment. No separate live standalone neural replay occurred in this continuation. The released archive's fresh extraction verified287 file hashes, passed53 tests, and reproduced all9 result JSON files byte-for-byte in Python3.13.5/NumPy2.3.5. This is recorded-data replay, not a second full neural benchmark.

## Release and provenance

Release archive SHA256:b0dabe794664de7f90292364966c47946ef02fe9b06d7192699cecebbaf1032a.
Recovery artifact10673231963 SHA256:51698d3179737cfd9966a427688e8ca76b9e0030ddf6bb2d23d45f4655099fd8.
Scientific commit03ae69ac70aa04ab599cd3af3926dbd72a4e0ce9.
Protocolc8c1812a6a8d871dfdf79874abe29c3f6cfef612.
Scientific sourcee50085c81381f226a632e06efbb7d740f276acd013a3931e0c18343d15a55ffd.
ModelQwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
JevBench7128f5cf445ca41e4da1bdc7c84f97926250724d;BBH9ee07bd481feebf959a6b59d61ea57bdcf30964d.

The next justified hypothesis is a compact, independently checkable countermodel for actual entailment tasks, not a universal skeptical-prose prompt. It needs source-premise coverage checks, an exact witness verifier, fresh evaluation and a concise ordinary-reasoning control at measured compute parity. No such new system or improvement is claimed in this result.
