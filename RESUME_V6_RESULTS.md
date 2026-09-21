# Resume v6 — completed and audited, September 21, 2026

## Result

The predeclared primary, resume_complete, scored **200/231 (86.58%)** on reused public JevBench, matching the benchmark author's archived Jev 1.13.0 count. Prior Handoff160 was **196/231 (84.85%)**. This is a public accuracy tie, not recovered Jev weights, model equivalence, superior calibration, or a full leaderboard win. Resume is correct on 14 cases Jev missed, and Jev is correct on 14 Resume misses.

On 64 new harder generated English cases, the primary improved **44/64 (68.75%) to 51/64 (79.69%)**, with seven repairs and no regressions versus Handoff160. Six repairs were scheduling and one was ledger arithmetic. The fresh set has four authored task families, not independent external authorship or broad natural-language coverage.

| Fixed arm | Public JevBench | New harder English | Reused English | Old policies, unique |
|---|---:|---:|---:|---:|
| Native direct readout | 182/231 | 21/64 | 31/64 | 61/62 |
| Prior reasoning with native fallback | 194/231 | 41/64 | 63/64 | 62/62 |
| Handoff160 | 196/231 | 44/64 | 63/64 | 62/62 |
| Resume complete, primary | 200/231 | 51/64 | 63/64 | 62/62 |
| Same continuation, native fallback | 200/231 | 52/64 | 63/64 | 62/62 |
| Rollback then continuation | 200/231 | 51/64 | 63/64 | 62/62 |
| Rollback then immediate readout | 199/231 | 43/64 | 63/64 | 62/62 |
| Published Jev 1.13.0 | 200/231 | Not evaluated | Not evaluated | Not evaluated |

The 52/64 secondary is not substituted as the primary after seeing outcomes. All seven arms remain in the release.

Public tiers for primary: **47/48 easy, 71/72 standard, 82/111 hard**. Archived Jev: 48/48, 71/72, 81/111. A one-answer hard-tier lead offset by an easy miss is not demonstrated hard-reasoning superiority.

Fresh family counts, 16 each (native / Handoff / primary): ledger 3/11/12; schedule 6/8/14; alternating two-map pointer traversal 2/13/13; interacting policy 10/12/12. An independent parser of the authored English grammars verified every fresh answer key. That parser is audit-only and never enters inference or routing.

## Uncertainty

Primary versus Handoff160 public: six repairs, two regressions, **+1.73 percentage points**; descriptive 95% paired source-group bootstrap interval **[-0.45,+4.27]** points, 10,000 draws of 195 groups. Exact discordant-pair p=0.2891. Public improvement is not statistically established.

Primary versus prior reasoning public: eight repairs, two regressions, +2.60 points, interval [0.00,+5.38], p=0.1094.

Fresh primary versus Handoff: seven repairs, zero regressions, **+10.94 points**, item-bootstrap interval **[+4.69,+18.75]**, exact discordant-pair p=0.015625. These are exploratory, not multiplicity-adjusted, conditional on four generated families and one frozen model. Reused public benchmark results are not a fresh sealed generalization test.

## What was implemented

Use the existing 160-token reasoning draft. Preserve a strictly complete allowed FINAL answer. Otherwise continue the same assistant draft for up to 160 additional greedy tokens, stopping at a newline-terminated valid FINAL or EOS. If still unfinished, use the unchanged original-evidence-plus-draft categorical readout. No confidence, correctness, or task-family routing; no training or test-selected parameters.

Forty public drafts lacked completed answers. The other 191 were preserved, not regenerated. Across old cohorts, 41 incomplete drafts plus 64 new inputs required new inference; 316 old completed cases were preserved. Total unique reduced population: 421. Fresh cases generated their base drafts live. Old drafts are text-reprefill interventions, not original token-ID/KV-cache resumption. Fresh continuation also re-prefills. Same pinned Qwen3.5-4B backbone; no new weights, Jev API calls, or Anchor neural executor.

Public routed subset: 25/40 became terminally complete, 19 correct; remaining 15 used typed fallback, six correct. Total routed correctness rose 21 to25. The preserved subset was175/191, including16 already-completed errors this gate cannot repair. Fresh43 completed bases were preserved,36 correct;21 extended,16 completed/15 correct, five unfinished/all wrong under primary. Completion is a routing signal, not a semantic certificate.

A new scheduling case (resume-6230927-schedule-09) stopped at `Stage 5: 785 +`. Immediate typed readout and immediate rollback/readout chose option d incorrectly. Continuation completed **785 + 34 = 819**, emitted option b, and matched the independently checked answer. Rollback then continuation also succeeded. This illustrates one repair without hiding the two public regressions.

## Rollback controls and remaining counterargument

Rollback drops at most32 retokenized tokens to the last visible punctuation/newline, using no future suffix. It is lexical, not a semantic completion detector. Of34 changed public prefixes,32 regenerated the exact discarded visible text. Fresh:15 of16. Rollback-then-continuation tied primary counts but traded one repair for one regression publicly and used4497 versus4158 additional public tokens; fresh1730 versus1598. Much rollback compute replayed the same work rather than exploring a new branch.

Rollback without continuation reached199/231 but43/64 fresh, below the44/64 baseline. Three of the primary's six public repairs were also repaired by this cheaper control; public gains cannot all be uniquely attributed to generation.

An uninterrupted320-token maximum could produce much of the same benefit and naturally stop early atEOS. This full-population control was not run. The result supports giving unfinished drafts more computation, not claiming that this gate beats a simple longer decode, cached continuation, or a learned controller on efficiency.

## Compute and verification

Primary added4158 public tokens: exactly18.0 per public input on average (103.95 per routed case). Fresh added1598 tokens,24.97 per input. Fresh mean total generated tokens rose131.58 to156.55 (+18.98%). Measured same-run four-thread ARM CPU policy phase sums rose **94.57 to111.21 seconds (+17.60%)**. Median94.86 seconds unchanged because most cases did not extend; p95135.58 to213.86 seconds. These exclude research-only controls, model loading and production serving overhead; they include re-prefill. No speedup or optimized GPU equivalence is claimed.

Main execution completed all16 inference shards and aggregation:64 new base generations,113 unique continuation generations,19352 generated tokens,64 fresh native baseline calls,114 new categorical readouts,16 native anchors and16 full/restricted-readout fixtures. Memoized duplicate prefixes count once. Interface verification runs are separate.

The independent control/parser reducer reconstructs **2947 policy labels**. Official scoring agrees on **1617 public-arm outcomes**. Separate probability arithmetic checks172 stored vector records including aliases, maximum error1.55e-16. All16 anchors and readout fixtures match exactly. These are audits of recorded data, not a full neural rerun or calibrated-probability proof. Forty pre-outcome unit tests plus six interface serialization tests passed:46 total.

Standalone in-memory requests initially serialized nested JSON keys differently from the frozen saved inputs. The release's decide_canonical.py canonicalizes validated requests to the tested format. Main inference, policy and scores are unchanged. Three separate canonical live fixtures reproduce the main answer, token counts, branch and full trace hash exactly (two preserved, one continued). Their original noncanonical receipts are retained with different traces, not mislabeled as equivalent. The extended fixture was selected for runtime branch coverage, not an accuracy estimate.

One older auxiliary English case also had different JSON key order between its v4 base prompt and v6 continuation. Its unchanged correct output is not presented as a clean continuation effect. All40 public and all21 continued new-English base/continuation prompt hashes match, so headline comparisons are unaffected.

A fresh release extraction verified151 file hashes, passed46 tests, and reproduced all predictions, summary, probability audit, diagnostics, fresh-key audit and interface audit byte-for-byte. This is offline replay of recorded evidence, not repeating the entire4B neural experiment. Release ZIP SHA256:76c376fdd2c0a846132edde7e448c2b2a0a6df9f6b74eb629b178697ed0903f3.

## Provenance and scope

Main run: https://github.com/Jaksenc/parameter-golf/actions/runs/35645701540
Main artifact10661727832, SHA256 f810f8c687e00504c8dc8723aaf1528a44ce6845a23c18898ea36cdfb09758c7.
Canonical live replay: https://github.com/Jaksenc/parameter-golf/actions/runs/35649171138
Protocol commit3c32b0d98210b2477a9b4243700357ebe741d5c4; inference commitfdde8c7d37bf490a892e3bc01cca2e1875f34cf7; analysis-lock commitbbbd0deb3cc22b39b849944bb8352ddc5485af96.
Model Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Benchmark revision7128f5cf445ca41e4da1bdc7c84f97926250724d; archived Jev comparison blobf9993fc449faf54ad61729f692504faf33d0e7ba. Matched231 public IDs only; remaining303 of534 decisions, full composite, calibration, comparable production speed/cost untested.

Budget extension/forcing has prior art (s1,2501.19393); PUMA(2605.17672) distinguishes answer readiness and reasoning convergence. These were researched, not replicated or surpassed. Project contribution is a tested preserve/continue policy, rollback controls, and transfer to harder authored English cases. No literature-wide priority claim, main-branch change, official submission, or hosted deployment.
