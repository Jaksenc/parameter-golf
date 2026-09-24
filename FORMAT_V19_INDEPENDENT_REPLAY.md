# Format v19 — independent replay, September 24, 2026

The completed run 36033383857 was discovered and recovered rather than duplicated. This response performed source/target, full-state, arithmetic and experimental-design audits; it made no new pretrained forward pass or optimizer update. Main and the frozen scientific source are unchanged.

## Decision

Varied-format training has not established an advantage over single-format training in the matched experiment. Both beat the unchanged model on the raw primary table-plus-distractor probability losses. Do not credit ordinary supervised learning gains to the augmentation treatment or promote a general model from this narrow test.

Six models: seeds19101/19102/19103, single/varied,32 updates each. Paired updates share the same mathematical world, complete facts including excluded inventory, exact target law, label order, initialization, optimizer and loss weight. Single always sees field-grouped prose; varied sees a balanced assignment of prose, category records or JSON. Each world is presented once per arm, not three times to the varied arm. The Markdown table is absent from training and calibration. The pinned Qwen3.5-4B base remains frozen; each rank4/scale2 adapter has1,507,328 trainable parameters on32 MLP down projections. AdamW lr1e-4, weight decay0.01, clip1, loss0.5*soft-target CE. No relationship loss, modal supervision, generated reasoning or oracle note.

192 worlds:128 fit pool,32 calibration,32 test. Each seed sees32 fit worlds. Four authored mathematical mechanisms,2–5 categories, deliberately shortcut-discriminating test worlds. Evaluation252 requests:32 calibration,128 factorial views,32 replication variants,32 relevant edits,28 historical BBH retention questions. Related views and seeds are not independent worlds.

## Raw primary, table+noise,32 worlds, means over paired seeds

| Model | Target cross entropy | Squared-vector error | Correct leading category |
|---|---:|---:|---:|
| Unchanged |1.199191|0.154508|34.375%|
| Single format |1.136377|0.108555|50.000%|
| Varied format |1.135332|0.106206|48.958%|
| Uniform |1.196873|0.141732|Uninformative tied ranking|

Squared-vector error is excess expected multiclass Brier. Varied-minus-single CE=-0.001045, descriptive95% interval[-0.019941,+0.024935]; squared difference=-0.002349, interval[-0.013571,+0.012602]. Two seeds favor varied, one favors single. No equivalence or superior augmentation is established.

Single-minus-unchanged CE=-0.062814, interval[-0.109955,-0.020636]; squared error=-0.045952, interval[-0.081387,-0.014258]. All three seeds improve both raw proper losses. Mean squared-error reductions relative to unchanged are29.74% single and31.26% varied. The31% is not attributable specifically to format diversity.

Intervals:10000 crossed paired-seed/within-family whole-world resamples, variants linked, three seeds and32 worlds, no multiplicity correction, no independent-grammar/general-language guarantee.

## Factor separation: raw cross entropy

| Source | Unchanged | Single | Varied |
|---|---:|---:|---:|
| Prose without irrelevant inventory |1.155613|1.109494|1.102974|
| Prose with irrelevant inventory |1.164156|1.122326|1.131847|
| Table without irrelevant inventory |1.165195|1.109422|1.099801|
| Table with irrelevant inventory |1.199191|1.136377|1.135332|

The clean layout change has little mean effect in this sample. Adding excluded inventory causes a larger point-estimate penalty, not uniformly statistically established across cells. Varied does not reliably reduce that penalty. This is not proof that distractors alone explain v18, which used different source organization and samples.

Illustration selected after outcomes: ACTIVE counts silver30,jade11,amber2 imply69.77%,25.58%,4.65%. Seed19101 varied returns57.44%,19.38%,23.18% from the clean table. Add an explicitly excluded SPARE column4,12,25: prediction becomes40.30%,19.19%,40.51%, changing the leading category to amber. The source law is unchanged. This demonstrates sensitivity, not the internal implementation of a particular wrong pooling formula. No target-based repair is applied.

## Other controls

Each model receives its own temperature from the fixed121-point[0.25,16] grid on only32 prose/noise calibration worlds. Calibrated primary CE:1.182586 unchanged,1.136789 single,1.135250 varied; squared error0.139356,0.108103,0.106210. The small varied-single gap remains. Raw primary is not replaced.

Raw replication inconsistency:0.003022 unchanged,0.012783 single,0.005944 varied. Varied is less inconsistent than single but remains worse than unchanged; prediction consistency is not correctness. Calibration changes absolute scales but not the conclusion that invariance is incomplete.

Historical retention28 items from seven BBH families, not a new external holdout: unchanged17/28; single seeds17,17,15; varied18,17,17. Every trained model introduces some regressions. No retention or noninferiority guarantee.

Training tokens total36385 single versus37046 varied(+1.82%); recorded training phases3013.32s versus3063.82s(+1.68%) across three workers each. Equal updates are not equal tokens/FLOPs. These sums exclude loading, checkpoint/evaluation overhead, are not parallel wall-clock or production serving results.

## Verification

192 real updates,1764 recorded evaluation vectors. All six full-state checkpoints have optimizer counters32 for all64 parameter states; saved factors, histories and source/data bindings agree. All factors changed between8 and32. All evaluation restoration/anchor errors are zero. Six trajectories resumed across separate16-step processes. No diagnostic optimizer trial steps exist in this trainer.

Independent audit decodes608 controlled-source target distributions, checks28 historical source rows/keys,96 matched training-world pairs and all16 inference shards. It recomputes3528 raw/calibrated metric vectors. Separate60-digit Decimal CE/squared differences are at most4.0e-15/1.7e-15. StoredFP32 probability discrepancy is1.28e-7. Torch-float64 independently reproduces all seven temperature fits and121-point grids. Current44 tests pass:22 frozen experiment tests and22 new independent tests. These local checks are not another neural run.

Keep single-format as the simpler research control and unchanged as deployment reference. The next possible test is matched counterfactual changes to irrelevant versus relevant evidence, not an assumption that more formats or another loss fixes the problem. No new JevBench score, complete composite, recovered Jev internals or model-superiority claim.

Raw artifact10826083731,201528767 bytes,SHA256195756b73b1ec5b23909d8e11c3cd88bdbbfe82387d033002083e2a75f8d0702. Source commit414ddb341e40714572213a11badfdc60c70ca318; trainer SHA256d951d0df4f5915c9dc07d026fb0c255a9d48c3e201a83c0abfe66664db835d2a; record hash18039ff25498f57ff9ff2f3ac8082e35aa33dc03ef40f46bcce45adef1a4fb99. Model revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Independent bootstrap seed190924514; not claimed byte-identical to any other unrecovered reducer.

Prior art: Sclar et al., arXiv2310.11324 documents prompt-format sensitivity, not that this particular augmentation succeeds. The contribution is the matched treatment and separately controlled evaluation, not invention of augmentation.
