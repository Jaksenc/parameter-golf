# Bridge v15 — completed and independently replayed

Completed September 22, 2026 New York time (September 23 UTC). Main branch unchanged. Both main and secondary-control inference completed without recovery. No pretrained weight updates, Jev API calls, paid inference, or new JevBench evaluation occurred.

## Main result

Using the EXACT SAME ordinary model-generated draft on 24 new event-law questions, numerical-mass output reduced mean TVD from 0.2463456605 to0.0244197559, a90.09% reduction versus the legacy code head. Squared-vector error fell0.1113207141 to0.0096785135. The stronger distribution-aware code control yielded0.2070600131 TVD and0.0888137449 squared error. Numerical mass is88.21% lower in TVD than that control.

| Same draft,24 event questions | Legacy codes | Semantic codes | Numerical mass, primary |
|---|---:|---:|---:|
| Mean TVD | 0.246346 | 0.207060 | 0.024420 |
| Squared-vector loss | 0.111321 | 0.088814 | 0.009679 |
| Correct modal category | 16/24 | 19/24 | 24/24 |
| False-zero positive-target cases | 0 | 0 | 0 |
| Mean complete phase sum | 223.16s | 224.06s | 268.79s |

Numerical improves23/ties1 against legacy; the tie is a predeclared fallback. Against semantic it improves23/worsens1. Primary paired TVD difference−0.221926, descriptive95% interval[−0.267147,−0.178955]; squared-loss difference−0.101642, interval[−0.141812,−0.068414]. Against semantic TVD difference−0.182640, interval[−0.236039,−0.126251]. These use10000 paired world resamples within four authored family strata, six worlds each. No multiple-comparison, independent-grammar or model-variation guarantee. Squared error is excess expected multiclass Brier, not total Brier. No calibration was fitted.

This closes the missing autonomous same-draft comparison rather than using oracle calculations as a performance result. It remains a narrow event-estimation diagnostic. Mode correctness24/24 does not mean perfect distributions:14 numerical vectors are within1e-10 maximum-coordinate error.

## Supports and semantic limits

24 worlds,48 event/modal questions,192 support views. Two-to-five exhaustive labels, zero-event cases, unique modes. Families: working/reserve counts; accepted/rejected conditional counts; urn mixtures; Bayes conditioning on a passed assay. The source and labels are shared within each event/modal pair, but questions and ordinary drafts differ.

Numerical event TVD by support: source only0.202503; privileged selected raw inputs plus procedural hint0.131205; ordinary model draft0.024420; privileged exact calculation1.86e-11. Oracle24/24 within1e-10 is readout fidelity, not autonomous calculation. Selected-evidence improvement mixes correct selection, simpler representation and procedure hints, not a pure causal estimate of retrieval. Model-draft readouts also retain original source, so copying versus recomputation is not isolated.

On the separate modal-answer questions, same-draft numerical output is worse: legacy22/24, semantic21/24, numerical20/24. Numerical mode squared loss0.357023 versus0.146693 legacy and0.123486 semantic. Three numerical mode outputs assign zero to the correct answer, yielding infinite strict log loss. The deterministic target is not a claim of justified model certainty. No semantic router is fitted from these outcomes and numerical output is not promoted as a universal replacement.

## Concrete learning target

A mixture draft explicitly computes normalize(0.4*n_L+0.6*n_R) instead of0.4*n_L/sum(n_L)+0.6*n_R/sum(n_R). For L=[4,34,28,25],R=[7,3,5,23], true law is[0.12810873,0.19681897,0.20202429,0.47304800]; numerical output is[0.09797297,0.26013514,0.23986486,0.40202703], matching the wrong rule to2.78e-17.

Replicating every L ticket four times while retaining its0.4 selection weight leaves the true distribution unchanged. The wrong rule changes its modal label from silver to ochre. This is an independently checked mathematical intervention, not another neural measurement or a patched answer. It motivates within-component replication examples for training.

A post-outcome diagnostic finds three mixture cases account for98.50% of numerical model-draft event TVD. They include an invalid truncated array/fallback, the wrong mixture normalization order, and inaccurate arithmetic despite correct main equations. The sample does not establish one universal failure cause. A separate Bayesian draft uses an incorrect common prior denominator that cancels in the posterior; a correct final answer does not certify all teacher steps.

## Cost and format failures

Numerical event readout alone averages57.20s versus11.57s legacy and12.46s semantic. Including the same ordinary draft, numerical adds20.45% to legacy complete phase time. These are four-thread ARM CPU phase sums; semantic is a separate control run. No GPU, production-price or Jev-endpoint comparison.

188/192 numerical generations parse. Two are all-zero arrays and two are incomplete at the96-token limit. All four use the predeclared same-view legacy fallback and remain scored. Four of48 ordinary drafts hit their480-token cap. Strict infinite losses remain explicit; clipped loss is a separate diagnostic, not altered predictions.

## What is implemented for the next training stage

A differentiable soft-target CE plus verified relation loss, ||p_right−A*p_left−delta||², is implemented and gradient-tested. The ordinary-supervision control receives exactly the same examples and may compute the same graph with relation weight zero. A and delta must agree with independently verified target laws. The relation loss adds a constraint, not new ground truth; consistently wrong invariant predictions can satisfy it.

Initial preparation:960 examples/720 relations over120 new worlds,768 fit/192 check. Edits cover distractors, real evidence changes and label-order permutations. Post-diagnostic extension:120 mixture replication examples/60 relations over30 additional worlds,96 fit/24 check. All targets are independently reconstructed from actual text. These are same-grammar capability preflights, not independent-family transfer sets.

Tests reject a false mode-partition relation: event mass[.30,.29,.41] merges to[.59,.41], but mapping the fine modal label does not produce the correct coarse modal label. Real evidence changes may also leave the target unchanged. No optimizer update, pretrained adapter fit, full trainer execution or matched trained-checkpoint result exists in this release. These are prepared data and objective components, not a trained student.

## Execution and validation

Retained neural work:48 ordinary generations/12536 tokens;192 numerical generations/7153 tokens;384 code readouts;24 native anchors and24 projection fixtures. Total240 generations/19689 generated tokens. Main run35802731023; semantic run35803306938. All24 inference workers and both aggregates succeeded, with no recovery or duplicates.

Independent reducers reconstruct48 source targets by separate ticket enumeration, reparse192 arrays, recompute384 softmax vectors, and rebuild576 probability/answer outputs. All notes, control code maps and24 weight receipts agree. Main maximum probability differences6.66e-16 softmax and3.89e-16 numerical normalization. Frozen main and semantic analysis sources match pre-outcome locks. The stronger semantic-code control was committed before any new main outputs were inspected.

47 current tests pass:20 scientific,11 independent analysis,4 wrapper and12 objective/gradient/semantic-boundary tests. A later integrity-helper key-name typo was repaired before validation; frozen scientific code and metrics never changed. Three actual saved-record CLI cases include a correct event answer, wrong modal answer and fallback. The live wrapper has injected-runtime tests; its underlying inference functions ran, but no separate live wrapper replay is claimed.

Fresh release extraction verifies203 file hashes, passes47 tests, and reproduces15 result/preparation JSONs byte-for-byte, including seven main result files, plus three CLI outputs. This is recorded-data replay and deterministic data regeneration, not a second neural run or weight training.

Release: Bridge-v15-Completed.zip; SHA256 d3517eff22ae2cdb15d3a3972cbaf22284410fe56ab7c49108cd8207d78eed21.
Main artifact10727691526 SHA256 a246ce2889ca98629949171fb489cdd5cf2221fe112d56f1c7f9226c441779be.
Semantic artifact10727197839 SHA256810599a9fe3d0b6ba05741544d7bc146fb0de2d44167dca4449f7f03fb2563ea.
Scientific source a58cb1240ee9d662f91bcd24cc5a2e7f9a44ae45b61a4e182e5d53f0fb844b98.
Model Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
Protocol1405ba0e867a249c007a73f5a28f514b860453af; inferencef53eb711db875d1599244284a51a33b46bb5610b.
Semantic declaration4baacf7c6579500ebd6c069da142466043411d43; workflow7e0efc2b2232725cd110fc002b0a198c43ba8069.

Decision: retain this measured event-law reference and test direct learning of correct probabilistic composition using independent targets. Do not distill all draft statements or numerical estimates unquestioned. No general Jev superiority or literature-wide novelty claim.
