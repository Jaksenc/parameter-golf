# Learn v16 — saved step32 audit, September 23, 2026

## Result and scope

Six actually trained low-rank Qwen3.5-4B adapters were evaluated at the latest checkpoint saved by every run: step32. Both ordinary soft-target supervision and verified-relation supervision worsen held-out probability quality relative to the unchanged model. No checkpoint is promoted. The original predeclared64-update primary remains INCOMPLETE; this is an availability-defined intermediate diagnostic selected before post-training predictions, not a retrospectively chosen best checkpoint or a recovered final model.

Original jobs logged366 updates total (60,61,60,61,62,62), but saved only steps1,16,32. No final64 weights, optimizer moments or RNG state survived. This continuation performed zero new optimizer updates and completed six inference-only evaluations. Resetting AdamW around stored weights would not be exact resumption.

## Matched held-out results

There are28 unique event questions and28 unique modal-answer questions from10 worlds, each evaluated for three paired seeds. Repeated seeds and transformations are not independent source examples. The16 fit-pool probes are reported separately. Seven of the ten held-out worlds are mixtures, with one each from counts, conditioning and Bayes; this is a small same-grammar pilot, not external-language generalization.

| Held-out event law, mean across seeds | Unchanged | Supervised | Relational |
|---|---:|---:|---:|
| Target log loss |1.254269|1.447954|1.411101|
| Squared-vector error |0.075998|0.158784|0.150276|
| TVD |0.200165|0.281321|0.275665|
| Correct modal category |32.14%|38.10%|35.71%|

Squared-vector error is excess expected multiclass Brier. Selecting the modal category does not substitute for estimating the event law. All six models worsen held-out event log loss AND squared error over the same baseline. Relational is less bad than supervised for two seeds, worse for the third.

| Held-out determinate modal answer | Unchanged | Supervised | Relational |
|---|---:|---:|---:|
| Target log loss |1.286573|2.227208|2.100793|
| Squared-vector error |0.750113|0.904776|0.892999|
| Correct answer, mean |50.00%|44.05%|44.05%|

Relational-minus-supervised event log-loss difference is-0.036854; descriptive95% crossed paired-seed/world bootstrap interval[-0.143043,+0.016399]. Squared-error difference-0.008508, interval[-0.046190,+0.008795]. Ten thousand resamples, three seeds and ten same-grammar worlds, no multiplicity correction. No statistically established or general benefit follows. A post-outcome equal-world weighting sensitivity also leaves both arms worse than baseline on event proper losses.

## Fit behavior and failed relation transfer

On eight fit-pool modal probes, mean correct rises62.50% to95.83% for both trained arms; log loss falls0.666279 to0.153710/0.150043. Fit-pool event squared error falls0.136400 to0.076285/0.073274. However, only13-15 of all16 fit-pool probes appeared during the first32 steps for a seed; this is not a fully seen-set metric, nor held-out evidence.

Held-out predictions become more concentrated: event entropy1.186111 to0.957623/0.981478; modal entropy1.006667 to0.481389/0.512283 nats. Proper test losses worsen. This pattern raises a narrow-fitting/overconfidence concern without uniquely identifying memorization, objective interference, optimization instability or insufficient steps.

For held-out event pairs replicating every ticket in one urn, the true mixture law is unchanged. Mean squared prediction inconsistency rises from0.003061 baseline to0.013656 supervised/0.013479 relational; every seed worsens. Label-permutation residuals worsen too. Some other relations improve in some seeds; all are retained.

A shared modal regression: L chosen with probability0.1 has counts[29,1,33,28]; R with0.9 has[12,20,1,3], labels[cobalt,ochre,ivory,jade]. Ochre is uniquely most probable with event probability0.5010989011. Baseline picks ochre with36.63% answer-code probability. All six trained models instead pick ivory. Seed16102 relational assigns ochre0.85% and ivory97.49%. The case is selected post-outcome for illustration, not new confirmation; all observations remain available. Event probability and answer-code probability are different quantities.

## What actually learned

Pinned base: Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Original weights frozen. Rank4 FP32 factors with scale2 on all32 language MLP down projections:1,507,328 trainable parameters per model. Both A and B changed in every layer in all six step32 snapshots. AdamW lr1e-4, weight decay0.01, clip1.0. No generated draft, oracle note, benchmark target, teacher probabilities or calibration enters direct inference.

Seeds16101/16102/16103 pair identical initial factors, pair order, data, optimizer and computation. Both compute mean two-input soft-target CE and the same relation graph; its coefficient is0 or0.25. Independently known q_right=A q_left+delta holds for all edges. Pair-gradient replay matches observed logits exactly in every retained training log and original parameters receive no gradients.

Fit pool96 inputs/16 worlds, check56 inputs/10 worlds,100 relations total. At step32:32 pair presentations/64 input presentations, covering53,53,52 unique fit records for the three seeds. This is half the planned64-edge schedule, not a full epoch or a general-training conclusion. The first32 training steps consumed2451.98-2511.85 seconds per model on four-thread ARM CPUs. Hook adapters are not merged; no inference-overhead or serving-price advantage is claimed.

The relation residual can be written||e_right-A e_left||² for prediction errors e=p-q. It constrains error relationships, not independently their correctness. Equal wrong predictions can satisfy invariance. The supervised loss is necessary, but these updates did not establish transfer of the intended probabilistic computation.

## Recovery, infrastructure and verification

Original archive10730888996, run35808253851, SHA2566f271f3caafc78b35505a6b58abaccbee43fed07f9cfed418ad224696fd8a359. Six checkpoints are evaluated unchanged by run35867355646; artifact10754525171 SHA256f29ca03acd9abeffb4b1dcf7cd5223677cb35dd642782918bed5b2d6c0c8b4e5. All workers and aggregation succeeded. Original64 remains incomplete.

Recovery protocolbfda51778e7f92c26861b016c2e33d3bed0f6323; evaluation commit4ab901f7fe942086d630cffada1f90287db1f914; analysis lock3659287631d6bbccf62f55c30a0fe1493d77937c. Parent scientific source SHA25688d3ce9ebae554adbb1728b45254bfe7a87043a26d1264fe0fad55077054b587.

New evaluation:432 trained probability outputs,462 counted model forwards including probes, plus projection fixtures. Independent60-digit arithmetic checks504 distinct vectors including72 identical archived base observations. Maximum float64 difference5.55e-16; storedFP32 difference1.13e-7. All152 actual-source targets,100 exact relations, paired schedules, tensor changes, zero adapters, reloads and baseline restoration verify. No canonical or selected normalized-semantic split overlap; unknown semantic or pretraining exposure is not ruled out.

44 current tests pass. The original missing parent exclusion-test data file was restored unchanged; the initial failed local test is not claimed as success. The old failed-turn structured-model scores are not recovered and are not presented as evidence.

A separate atomic CPU checkpoint helper saves parameters, optimizer, all three random streams, schedule position and source/data bindings. On a dropout-using20-step toy run interrupted at7, full-state resume exactly matches losses/weights; resetting only AdamW causes parameter difference0.06679. The helper is not yet integrated into a new full4B training run and cannot recreate the lost original state. Checksums detect corruption, not malicious authorship.

Three saved-record CLI examples run, including a wrong answer. The live wrapper is injected-runtime tested; its underlying factor/readout path ran in the checkpoint evaluation, but no separate live-wrapper benchmark is claimed. Full pretrained base weights are not distributed; all eighteen saved low-rank snapshots are retained.

A fresh extraction verified139 file hashes, passed44 tests and reproduced13 result JSON files byte-for-byte, including all three CLI outputs. Replay re-reads recorded predictions and saved weights; it is not a second full neural run. Release Learn-v16-Checkpoint-Audit.zip,98,119,568 bytes, SHA25639aa0bd3e9ca89c6f24e134dd7cf480ba33810bb0b0b0ba684729ba7874e7047.

Decision: retain the unchanged direct model. Complete a properly resumable, newly identified matched trajectory before scaling or claiming learning gains. Tested worlds are now diagnostics, not fresh confirmation. These results do not tell us that more steps, different rank, a new learning rate or separate event/mode training will work. No new JevBench score, teacher distillation, calibrated-model claim, paid provider, main modification, official submission, or Jev-superiority result.

Prior art: LoRA(arXiv2106.09685); PyTorch saving/loading documentation. Actual training and reliable recovery are engineering progress, not a new architecture or literature-wide invention.