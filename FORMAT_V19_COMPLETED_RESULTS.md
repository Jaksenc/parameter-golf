# Format v19 — completed and audited September 24, 2026

## Decision and scope

No established advantage from varied-format training over the matched prose-only control. Both improve probability quality versus the unchanged model on this new generated cohort, but the varied-versus-single primary differences are small and their intervals include zero. One of three paired seeds regresses. No checkpoint or seed is promoted. No new JevBench score, full composite, Jev API use, or production-superiority claim.

Run36033383857 had already completed when this continuation retrieved its artifact. Current work recovered records, independently verified arithmetic and checkpoints, and packaged the experiment. It did not launch duplicate pretrained training or inference. Main remains unchanged.

## Matched design

Same frozen Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; six rank-four/scale-two adapters with1,507,328 parameters on all32 MLP down projections. Seeds19101/19102/19103, single and varied training. AdamW1e-4, weight decay0.01, clip1, event cross entropy times0.5,32 real updates each, no relational objective, modal supervision, generated rationale or privileged calculation.

Single uses field-grouped prose. Varied assigns each scheduled world to prose, category-grouped records or category-keyed JSON, approximately10–11 examples per format. Corresponding updates share the same world, complete facts including excluded inventory, target, label order, initialization and optimizer. Each world is presented once, not explicitly paired with multiple renderings. Formatting also changes fact order and token count, so this is not isolated typography or equal-FLOP training. Table representation is absent from training and calibration.

192 worlds:128 fit-pool,32 calibration,32 test. Each seed uses32 fit worlds. Four authored mathematical mechanisms,2–5 categories, sharp/diffuse target strata. Test worlds are deliberately shortcut-discriminating, not representative arbitrary English. Evaluation contains252 requests, including128 format/noise cells,64 replication/relevant-edit variants,32 calibration and28 historical retention cases. Repeated views and seeds are not independent worlds.

## Primary raw event estimates on32 table-plus-noise test worlds

| Method | Target CE | Squared-vector error | Leading-category accuracy |
|---|---:|---:|---:|
| Unchanged |1.199191|0.154508|34.38%|
| Uniform |1.196873|0.141733|Tied ranking, not informative|
| Prose-only, seed mean |1.136377|0.108555|50.00%|
| Varied-format, seed mean |1.135332|0.106206|48.96%|

Squared-vector error is excess expected multiclass Brier. Varied-minus-single CE=-0.001045, descriptive95% interval[-0.019996,+0.025275]; squared=-0.002349, interval[-0.013661,+0.012510]. Seeds19101/19102 improve,19103 worsens on both. No claim of equivalence follows from the inconclusive comparison.

Against unchanged, squared error improves29.74% for single and31.26% for varied. All six adapters improve raw primary proper losses over unchanged. Single CE difference=-0.062814, interval[-0.109649,-0.019823]; varied=-0.063858, interval[-0.114224,-0.019663]. Ten thousand crossed paired-seed/within-family-world resamples; conditional on three seeds and32 authored worlds, no multiplicity or prior-exploration adjustment. These are individual-model means, not deployed ensemble predictions.

## Factorial information: table versus excluded inventory

| Raw mean squared error | Prose/no noise | Prose/noise | Table/no noise | Table/noise |
|---|---:|---:|---:|---:|
| Unchanged |0.126395|0.133500|0.130843|0.154508|
| Single |0.085857|0.093482|0.088921|0.108555|
| Varied |0.080230|0.099684|0.082206|0.106206|

Table-only mean CE effects: +0.009582 unchanged,-0.000072 single,-0.003173 varied; all intervals include zero. Adding noise to prose gives+0.008543,+0.012832,+0.028873. Adding noise to tables gives+0.033996,+0.026955,+0.035531. Most effect intervals and all interaction intervals include zero; varied prose-noise interval is[+0.006121,+0.056592]. These are controlled total input interventions, not proof of an internal attention mechanism. Noise adds a column/field and length, not only a semantic distractor.

A post-outcome maximum varied-arm table-noise CE example has ACTIVE counts[8,300,4,2,27] for ivory,ochre,silver,jade,amber. Ochre's true probability300/341 is87.98%. Adding excluded SPARE counts[14,5,39,14,43] must preserve that law. Varied seeds' ochre probabilities fall40.41→24.22%,39.22→18.69%,37.08→20.21%; all three switch from ochre to amber. Unchanged also switches. This illustrates a defect the treatment failed to repair, not a defect uniquely caused by training or fresh confirmation.

## Calibration, consistency and retention

Secondary one-temperature calibration uses only32 prose/noise calibration worlds and the fixed121-point grid[0.25,16]. Table test rendering never enters fitting. Calibrated table/noise CE is1.182586 unchanged,1.136789 single,1.135251 varied; squared error0.139356,0.108103,0.106206. Varied-single CE=-0.001538, interval[-0.018712,+0.017897]. Raw primary is not replaced by this secondary result. Intervals condition on fitted temperatures.

Raw replication inconsistency is0.003022 unchanged,0.012783 single,0.005944 varied; calibrated0.000888,0.005435,0.003561. Varied is less unstable than single on this measurement but still more unstable than unchanged. Correctness and invariance are distinct; no generally learned probability algorithm is established.

Retention is a historical subset from v18: first four sorted IDs from each of seven BBH families, selected without using correctness. Unchanged17/28; single seeds17,17,15; varied18,17,17. Varied repairs/regressions are4/3,2/2,4/4. No general retention or noninferiority guarantee, and no transported event temperature is represented as an externally calibrated confidence model.

## Compute

Each arm has32 presentations per model. Mean input tokens:12128 single,12349 varied (+1.82%). Mean logged update time1004.44s versus1021.27s (+1.68%), excluding some checkpoint/probe/load costs. Forward-only means7.50–7.53s for all states, fixed call ordering, four-thread ARM CPUs. No optimized serving advantage, billing or Jev endpoint comparison.

## Validation and provenance boundary

Complete neural execution:192 actual optimizer updates,1764 final probability vectors,192 probe vectors, six training trajectories and16 evaluation workers. Independent source parser checks608 generated targets and exact same-fact matched schedules; full generator replay reproduces636 records. All28 historical retention selections/targets match parent records.

Separate Torch-float64 arithmetic reproduces3528 raw/secondary metric vectors and all seven calibration fits. Maximum metric error6.66e-16; calibration-grid error4.44e-16; stored FP32 vector discrepancy1.28e-7. Six full-state files equal final adapter tensors; every AdamW counter is32; all18 factor snapshots retained; A and B change in all32 layers between8 and32. All22 runtime/weight/anchor receipts agree, restoration errors zero. Current42 tests pass:22 scientific,11 repository analysis,9 added arithmetic/interface tests. Synthetic fixtures do not enter neural outcomes.

All15 scientific source-lock hashes match. The recovered repository scorer matches Git blob e2b58c7d2c536d250884b1ed89eb36d391a3e029, committed17:26:11UTC, but its SHA256f357045b4c19cc11e2728cf61cd5d63d15f9f339fd94adb0930f2f873f1a7b9c differs from the earlier analysis-lock declaration19c7bfa22aedd90712a55d6c4e04dbdc26d5b7679b686adadef4027974b5435c. Cause not established; original lock preserved. Inference/data/test hashes match, separate arithmetic verifies outputs, no scientific treatment changed. Exact original scorer-lock reproduction is not claimed.

Standalone saved-record interface examples are replay-only and include the distractor mistake. No live deployment or new pretrained calls were made by the local audit. The full conversation release retains source, all raw records, all adapter snapshots and optimizer states, tests, independent numerical checks and a fresh-extraction receipt; full pretrained backbone weights are not redistributed.

Next: retain ordinary supervision as the simpler experimental control and unchanged inference as deployment reference. A focused follow-up can vary eligibility while holding representation/numeric ranges fixed: excluded-count changes should preserve the distribution; changing the eligible group should change it. This is proposed, not implemented or demonstrated here; no new optimizer or universal relation loss is justified by this experiment alone.

Run36033383857; inference414ddb341e40714572213a11badfdc60c70ca318; scorer92886430d8f3955f278adf564a91aa20e908bcbf. Raw artifact10826083731,201528767bytes,SHA256195756b73b1ec5b23909d8e11c3cd88bdbbfe82387d033002083e2a75f8d0702. Trainerd951d0df4f5915c9dc07d026fb0c255a9d48c3e201a83c0abfe66664db835d2a; dataset18039ff25498f57ff9ff2f3ac8082e35aa33dc03ef40f46bcce45adef1a4fb99. No new JevBench score or claimed recovered Jev internals.
