# Contrast v7 — completed and independently audited, September 21, 2026

## Result and claim boundary

The frozen primary, contrast_consensus, scored **203/231 (87.88%)** on reused public JevBench, versus the benchmark author's archived Jev 1.13.0 **200/231 (86.58%)**. This is a three-item public-count lead, not a full 534-decision benchmark win, statistical establishment of superiority, model equivalence, or a production speed/cost/calibration comparison. No new weights were trained and no Jev API calls were made.

A simpler uninterrupted 480-token control scored **202/231**. Contrast scored **48/64** on a new-to-project external BIG-Bench Hard sample, compared with **46/64** for the longer-decode control. Contrast consumed approximately1.88 times its mean measured serial CPU phase time publicly and1.81 times externally. The added two-order gate is not promoted as an established improvement.

| Frozen arm | Public JevBench | External BBH sample |
|---|---:|---:|
| Native direct readout | 182/231 | 36/64 |
| Uninterrupted320-token reasoning | 200/231 | 45/64 |
| Uninterrupted480-token reasoning | 202/231 | 46/64 |
| Blind constraint-first challenger160 | 197/231 | 42/64 |
| Single-order judge | 204/231 | 48/64 |
| Contrast two-order consensus, primary | 203/231 | 48/64 |
| Archived Jev1.13.0 on identical public IDs | 200/231 | Not evaluated |

The204-answer ablation is retained, not substituted as the primary after seeing outcomes. Primary tier counts:48/48 easy,71/72 standard,84/111 hard. Archived Jev:48/48,71/72,81/111.

## Paired uncertainty

Primary versus320-token: five repairs/two regressions, +1.30 percentage points, descriptive95% source-group bootstrap interval[-0.87,+3.56]. Primary versus480-token: five repairs/four regressions, +0.43 points, interval[-2.16,+3.03]. Public gains are not statistically established.

Primary versus archived Jev:14 Contrast-only correct and11 Jev-only correct. A post-outcome descriptive interval for the+1.30-point difference is[-2.97,+5.56]. This is not a contemporaneous matched-serving comparison.

External primary versus320-token: three repairs/zero regressions, +4.69 points, family-stratified item-bootstrap interval[0.00,+9.38], exact discordant-pair p=0.25. Versus480-token: three repairs/one regression, +3.13 points, interval[-1.56,+9.38]. Each source family contributes eight items. These exploratory intervals use10,000 resamples,195 public source groups, or within-family external resampling; no multiplicity/repeated-use adjustment or generalization guarantee.

## What was implemented

A320-token base draft and a blind160-token constraint-first challenger are generated from original evidence. The challenger does not see the base proposal. If their allowed answer labels differ, the same frozen model judges the two answer/draft proposals twice with their presentation order reversed. Switch only if both select the challenger; otherwise preserve the base. Both candidates and judges share weights, so they are not statistically independent or semantic verification.

A live480-token decode supplies actual first320 token IDs and timing taps for the shorter control. No old archived prefix was substituted for the new base. Categorical fallbacks use the unchanged Handoff readout when a completed exact FINAL label is unavailable. The actual standalone primary entry point runs320+160, without unused longer-control/native-baseline work.

## Lessons from stronger controls

The uninterrupted320-token control reproduces **all231 Resume v6 final answers exactly**, not only the same200-correct total. On this recorded public set, the earlier continuation controller did not add an accuracy advantage over a simpler longer decode.

The candidates disagree on26 public and17 external inputs. Judge order changes the selected proposal on six and five respectively. Relative to the single-order ablation, requiring order consensus blocks only one additional public switch, and it was a correct repair. It blocks no additional external switch. Thus the second judge adds cost without observed benefit here; this does not prove that one judge is generally reliable.

Of28 public primary errors,22 have neither candidate correct and six have a correct candidate that selection misses. Of16 external errors,13 lack a correct candidate and three are selection errors. The oracle ceilings for choosing between these fixed recorded proposals are209/231 and51/64, not achieved model scores. More voting among the same wrong proposals cannot repair those22 and13 cases.

On a pronoun-ambiguity example, the base chooses the employee's car while the challenger and judges recover the ambiguous antecedent. On a fulfillment-policy regression, authorization was withdrawn and copied notes explicitly could not restore it. The base correctly chose level0; the challenger and both judges wrongly selected level2, authorized but unused. Agreement across orderings did not verify the policy interpretation. A warranty example additionally shows that a correct answer label can accompany an impossible date in its explanation.

## External sample and cost

BBH sources are independently authored, public since2022, and selected before model outcomes using pinned bytes/seed and normalized input-stem exclusion against421 prior inputs. The64 cases cover eight families, not the full BBH benchmark; pretraining contamination is unknown. No Jev result exists for this sample. All64 source texts/answer keys and selected indices were independently replayed.

External correct counts per eight cases,320/480/primary: causal6/6/6; dates5/5/5; disambiguation3/3/4; formal fallacies8/8/8; seven-object deduction4/5/4; sports4/4/6; temporal8/8/8; shuffled tracking7/7/7. The gain over320 tokens is one pronoun and two sports cases, not a broad across-family improvement.

Measured public mean CPU phase sums:320-token99.35s;480-token103.35s;primary194.17s. External means96.62s,101.23s,183.61s. Public mean generated tokens:126.52 for480-token versus214.02 forprimary. Equal maximum token ceilings are not equal actual tokens or FLOPs. These are serial phase sums on four-thread ARM CPUs, excluding loading/serving/unused controls, not an optimized deployment or comparison to Jev endpoint latency.

## Recovery and verification

Original run35651891496 retained285 complete rows; shard5 lost nine in an upload failure, shard25 retained nine of ten before cancellation. Recovery35660386499 re-executed only the ten missing inputs. All285 originals are unchanged. All295 inputs have all six outputs.

Retained inference:295 native forwards,590 generations/66,593 generated tokens,81 unique fallback forwards and86 judge forwards. Counts exclude unavailable work and separate runtime fixtures. Forty-one surviving nonempty-worker receipts verify consistent model hashes, native anchors and full/restricted-readout fixtures. Every one of231 native prompt hashes and logit vectors matches the previous baseline exactly.

Current independent replay reconstructs1,770 policy labels. Official scoring agrees on1,386 public outcomes. Independent probability arithmetic checks169 stored vectors including aliases, max error1.64e-16. Four separately executed actual-policy fixtures match answers, counts, full trace hashes, readouts and judge observations except timings. Three were preselected; one was post-outcome disagreement-coverage selection, not an accuracy estimate.

The original local analysis code did not survive in the available artifact; its historical hashes/test receipts remain explicitly historical. This recovered release uses a new independent reducer and **41 passing unit tests**, including10,000 parser-fuzz fixtures. The frozen scientific source and primary never changed. A fresh extraction verified254 file hashes and replayed tests and all eight result JSON files byte-for-byte. This is offline recorded-data replay, not another full neural run.

Release SHA256:95fac11739c747f98d079e736769f45ce95fc934d00435dd64cbafe92d004ccd.

## Provenance

Scientific commit:b49f89b8f2f82ec3da53bd0d4b72f86c5cbf0e85.
Scientific source SHA256:52771fdb5d8c7e5315c6f49ac87b725ba1244c0fe5ba4f5a0674ec4de3778d0b.
Protocol:821d0b465870500513ee6bd793f7739dfebd78b2.
Recovery workflow:ace85749b032553cb832cc076429843bfd8f1696.
Model:Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Benchmark:7128f5cf445ca41e4da1bdc7c84f97926250724d; published Jev comparison blobf9993fc449faf54ad61729f692504faf33d0e7ba.
BBH:9ee07bd481feebf959a6b59d61ea57bdcf30964d.
Completed recovery artifact10667542726, SHA256 ac1f4016ca4c34c1996e24b59c98d991eafa2ddf2d8022d3d62bf58a5f48e5d8.
Main branch unchanged. No official submission or hosted deployment.

Related primary research includes Chain-of-Verification(arXiv2309.11495), intrinsic self-correction limitations(arXiv2310.01798), and BBH(arXiv2210.09261). Contrast is project-specific engineering and evaluation, not a literature-wide novelty priority claim. The next test should target missing correct candidates or genuinely informative task-disjoint verification, with new evaluation examples and actual compute controls, rather than adding same-model votes or selecting the best-looking current ablation.
