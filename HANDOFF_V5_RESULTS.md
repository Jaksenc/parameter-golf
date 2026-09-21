# Handoff v5 — completed September 21, 2026

## Result: the 64-token primary is not promoted

All 24 shards completed. The predeclared primary, handoff64, scored **186/231 (80.52%)** on reused public JevBench, below the stronger v4 reasoning control at **194/231 (83.98%)**. On the 64 reused English computation cases it dropped from **63/64 to 38/64**, with 25 regressions and no repairs.

The predeclared longer secondary, handoff160, scored **196/231 (84.85%)**, retained 63/64 on the old English cases, and reached 62/62 unique new policy cases. It fixed seven public errors and introduced five. Its +0.866-point change versus the stronger control has a descriptive 95% paired group-bootstrap interval [-2.11,+3.88] points. This is not an established general improvement or a retroactively selected primary.

Published Jev 1.13.0 scored **200/231 (86.58%)** on exactly the same public IDs. These are archived benchmark-author outcomes, not new Jev API calls. The longer variant is still four answers behind. No full 534-decision composite, hidden-set score, production speed/cost comparison, or official submission is claimed.

| Configuration | Public JevBench | Reused English | New policy, unique cases |
|---|---:|---:|---:|
| Native readout | 182/231 | 31/64 | 61/62 |
| Prior 160-token reasoning, native fallback | 194/231 | 63/64 | 62/62 |
| Typed readout, no draft | 176/231 | 32/64 | 60/62 |
| Typed readout, neutral placeholder | 178/231 | 33/64 | 56/62 |
| Forced readout, 64-token prefix | 186/231 | 37/64 | 62/62 |
| Forced readout, full draft | 195/231 | 62/64 | 62/62 |
| Handoff64 — primary | 186/231 | 38/64 | 62/62 |
| Handoff160 — secondary | 196/231 | 63/64 | 62/62 |

The new generator produced 64 scheduled policy slots, but an input-only audit found one duplicated pair: 62 unique canonical cases, 31 distinct pairs. The original 64 slots and all predictions are retained; a first-occurrence deduplicated sensitivity and dependence-aware bootstrap are reported. The duplicate pair was identified and committed before full outcome reduction, without selecting on correctness. Both handoff arms scored 64/64 nominally and 62/62 uniquely, with both members correct in all 31 distinct pairs. The native baseline already scored 61/62, so this narrow four-template probe is weak evidence for superiority.

## Mechanism and controls

The frozen Qwen3.5-4B supplies a reasoning draft. Handoff preserves a complete exact FINAL answer by the boundary; otherwise a separate forward on original evidence plus draft scores unique single-token uppercase decision codes and takes their argmax. Allowed-label membership is guaranteed conditional on successful input and numeric validation, not correctness or calibration. The old baseline already returned a valid native fallback label, so this is not new output coverage.

The old 295 reasoning/native records are replayed unchanged from v4; original generation token IDs were not saved, so old prefixes are retokenized text, not exact original-KV interruption replays. The 64 new policy slots generated live token IDs and timing taps. There is no KV reuse optimization, new model training, calibration fitting, Jev model extraction or reconstructed RLCD. Hybrid label-only paths have no fabricated probability vector; four actual typed branches retain raw uncalibrated distributions.

## What failed

Public tiers for handoff64: 48/48 easy, 71/72 standard, 67/111 hard. Prior reasoning: 47/48, 71/72, 76/111. Handoff160: 47/48, 71/72, 78/111. Archived Jev: 48/48, 71/72, 81/111.

On reused English, early handoff reduced ledger 16/16 to 4/16, schedule 16/16 to 6/16 and pointer traversal 15/16 to 12/16. Threshold cases stayed 16/16. The 64-token primary corrected ten and broke eighteen public cases versus the stronger control: -3.463 points, 95% descriptive interval [-7.93,+0.88]. Its reused-English change was -39.063 points, interval [-51.56,-28.13].

Example: a job starts 09:15, runs 13 minutes, pauses five, then runs nine. Correct answer: 555+13+5+9=582. The draft at token 64 is cut inside the numeral 573, leaving 57. The forced readout chooses 577; the full draft and the no-draft control choose 582. A post-outcome diagnostic detects split numerals in 11/64 reused-English cases, including nine of the 25 regressions. In public JevBench it finds twelve such cases, including four of eighteen regressions. It uses the later suffix to detect splits and is not an online stopping algorithm or causal proof. No boundary-aware repair was retroactively applied.

At the long boundary, the strict interrupted-label parser marks forty public outputs unfinished; native fallback is correct on nineteen, full-draft readout on twenty-one. The historical parser marked thirty-nine unfinished; stricter parsing adds one case but changes no baseline correctness count. The long recovery improvement is small and includes five regressions.

The exact descriptive decomposition: 64-token reasoning with native fallback would score 185 public answers; typed recovery adds one to reach 186. Stopping earlier loses nine relative to the 194-answer longer control. For reused English, stopping loses 27 and typed recovery restores two. These are algebraic counterfactual reductions of existing observations, not extra neural runs or selected alternative models.

## Compute and verification

Fresh policy generation averages 75.70 tokens, not 160; the 64-prefix averages 63.70. Same-run CPU phase sums average **50.89 seconds** for handoff64 versus **53.95 seconds** for the longer path: approximately **5.68%** less, not the nominal 60% cap reduction. Medians are 52.93 and 54.02 seconds. They include an extra full-source readout when needed, but are phase sums rather than a deployed end-to-end benchmark. No public-case speedup is claimed because old 64-token boundary timings were not saved.

Completed 1,436 typed forwards, 64 new drafts / 4,845 generated tokens, 64 fresh native decisions, 24 native anchors and 24 full/restricted fixtures. Anchors and fixtures agree exactly. All 2,872 arm labels replay; official scoring agrees on all 1,848 public-arm outcomes. Separate probability arithmetic matches all 1,436 vectors within 4.44e-16, and an independent parser/control implementation reconstructs 2,513 labels across seven policies. The historical comparator is retained, not falsely claimed as independently reimplemented.

A separately executed actual-64-token entry point matches all three fixed main-run prefixes, answers and readout probabilities exactly. It is a runtime check, not a three-case performance benchmark. The suite passes 55 unit tests plus 20,000 deterministic parser-fuzz fixtures. The independent audit initially omitted string normalization for eighteen integer Score targets; it stopped on the mismatch, was repaired and rerun. Main analysis, official scores and inference were unchanged. Both this and the duplicate discovery are documented.

The full source, records, code maps, prefixes, timings, tests, numerical audits and reproduction instructions are in the conversation release. Prior art includes s1 (arXiv:2501.19393) and Entropy After </Think> (arXiv:2509.26522). This is project-specific engineering and evaluation, not a literature-wide novelty claim. The result motivates testing completion-aware boundaries, but no such improvement is claimed here.

## Provenance

Main run: https://github.com/Jaksenc/parameter-golf/actions/runs/35640245638
Main artifact: 10658744882; SHA256 09f70d66330681e70fbf3504c69459b8abd6928fef6b9d7f1f2e28e7a74881c1.
Bounded runtime: https://github.com/Jaksenc/parameter-golf/actions/runs/35640801516
Inference commit: 726aef026dd7fd30d7a472e47c69cc6156dac143.
Protocol: 6d3aebb235787111a54470fc23802683bab2aa42.
Analysis lock: ea5c1905265111312d8137023cb7b71423073eff.
Duplicate audit: 396246772e6e6f022cdbbf06ec8ba471c8c28afb.
Model: Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Benchmark: 7128f5cf445ca41e4da1bdc7c84f97926250724d; Jev comparison Git blob f9993fc449faf54ad61729f692504faf33d0e7ba.
Main branch unchanged. No Jev API calls or new weights. Reused public benchmark and authored policy templates do not establish general superiority.
