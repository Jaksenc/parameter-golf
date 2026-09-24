# Transfer v18 — completed September 24, 2026

## Decision

The fixed transfer gate is not passed. Event-only adapters retain a modest gain on new probability worlds in familiar formatting. On reorganized, distractor-heavy sources, frozen-calibrated log loss improves slightly but squared error worsens slightly; neither paired interval excludes zero. Uniform probabilities outperform all three method means on both primary proper losses. External retention counts improve slightly in all six adapters, but individual regressions and small samples prevent a general preservation claim. No adapter is promoted.

This is a newly executed test, not replay of the old evaluation. Run36012803018 completed all16 workers and aggregate. It produced1232 direct probability vectors on176 requests, comparing the unchanged model with all six saved32-update adapters (event_high/mixed_high, seeds17101/17102/17103). No training, generation, recalibration, Jev API call, new JevBench score, or main-branch modification.

## Primary:24 new row-oriented event worlds, temperatures frozen from v17.1

| Model | Target CE | Squared-vector error | Correct leading category |
|---|---:|---:|---:|
| Unchanged |1.475913|0.135344|37.50%|
| Event-only, mean over seeds |1.464582|0.136731|47.22%|
| Mixed, mean over seeds |1.472229|0.137865|44.44%|
| Uniform reference |1.443115|0.129981|Arbitrary tied ranking|

Squared error is excess expected multiclass Brier. Event-only minus unchanged CE=-0.011331, descriptive95% interval[-0.042458,+0.020101]; squared error=+0.001387, interval[-0.013383,+0.019406]. Raw event-only CE improves1.577077 to1.510608 and squared error0.167964 to0.153262, but raw results do not replace the predeclared frozen-calibration endpoint. All seeds/arms remain reported.

## Representation and cardinality

The24 worlds cover four mathematical mechanisms (counts, conditioning, mixtures, Bayes) and2,3,4,5,6,8 labels, with new category names and six true-zero event worlds. Five requests per world supply familiar rendering, row-oriented rendering, count replication, relevant count change and the modal-answer question. Whole worlds and seeds remain linked in resampling. These remain authored versions of known mathematical mechanisms, not independently authored probability language.

On familiar renderings, calibrated squared loss is0.101184 unchanged,0.091523 event-only,0.093334 mixed. Event-only CE is1.386307 versus1.405179 unchanged; paired CE interval[-0.040581,-0.002305], squared interval[-0.022787,+0.000177]. Row-oriented event-only squared loss rises to0.136731 and leading-category correctness falls75% to47.22%. The new source also adds irrelevant fields, so this contrast does not isolate grammar from distractor effects.

On the eight6/8-label worlds, calibrated CE is2.054131 unchanged,2.004352 event-only and2.027905 mixed; uniform is1.935601. Relative model improvement is not mastery of the unseen cardinalities. No per-family or code-position correction is fitted.

## Retention

56 separately authored BBH questions, eight from seven families. Fixed source/index exclusions were reconstructed from recovered prior project inputs; formal fallacies omitted because its corpus had been extensively inspected. BBH pretraining exposure and unknown other project inputs remain unruled-out.

Unchanged33/56; event-only seeds34/56,34/56,36/56; mixed seeds34/56,35/56,34/56. Event-only mean accuracy change+2.98 points, interval[-5.95,+12.50]; mixed+2.38, interval[-4.76,+10.12]. Event-only repair/regression counts by seed are2/1,7/6,7/4; mixed1/0,5/3,6/5. These counts do not certify noninferiority or general improvement. Old modal temperatures transported to BBH worsen each arm's mean CE versus raw output; this is a secondary calibration-transport diagnostic, not a fitted external confidence model.

## Consistency and example

Calibrated replication squared inconsistency:0.002848 unchanged,0.009774 event-only,0.007290 mixed. Raw replication also worsens0.009145 to0.020334/0.023672. Calibrated relevant-edit delta error improves slightly0.081544 to0.078269/0.078642. Absolute quality and consistency remain separate.

A post-outcome example: eligible basalt45/quartz17/garnet43 implies[42.86%,16.19%,40.95%]. Scaling all eligible counts by five must preserve the law. Event-only seed17102 instead changes[37.45%,46.25%,16.31%] to[63.84%,14.76%,21.40%]. The leading answer happens to become correct, but the probability change is unjustified. This is an illustration, not a new model-selection criterion.

## Verification and limits

Independent reduction checks176 actual-source targets/keys,1232 raw vectors,2464 raw/frozen-calibrated metric rows, all16 complete shards, six checkpoint hashes,96 adapted-anchor checks,16 baseline anchors and restoration. All anchor/restoration errors are zero. StoredFP32 discrepancy max1.18e-7; independent60-digit softmax difference6.66e-16. SeparateTorch-float64 arithmetic reproduces all2464 metric vectors within2.22e-15. Thirty-six tests pass; three saved-record CLI examples are explicitly replay, not additional neural inference.

All seven old temperature pairs remain unchanged. No target-domain fit, best seed or alternative-primary selection. Intervals use10000 crossed paired-seed/within-family-world resamples, conditional on three seeds,24 generated worlds or56 external items, with no multiplicity adjustment or propagation of old temperature uncertainty.

Mean forward-only CPU timings approximately9.215s unchanged,9.254s event-only,9.250s mixed. Excludes loading, adapter switching, tokenization, serving and research controls; unmerged hooks, no optimized production or Jev-endpoint comparison.

The conversation release contains executable source, six unchanged final adapters, all records, protocols, tests, nine numerical result files, and a separate fresh-extraction validation receipt. Full pretrained weights are not distributed. Offline result replay is not a second neural benchmark.

Next justified test: same-format versus multiple-format training with the same mathematical worlds, target laws and total presentations, withholding some renderers and distractor arrangements. Evaluate format-only, distractor-only and combined interventions separately. This is a proposed causal test, not a demonstrated fix. Keep the unchanged deployment reference.

Raw artifact10813762938,34379830 bytes,SHA2565615f165cec06946b8c0876c73a77b0fa7e26d750de4dd9e530918dc202b5f3f.
Scientific source d265a8761e7670fcfa721a1f24dd94a2b01197b9fb3d797a4eaf3f88c8af4608.
Inference commit90e64a18bd78283a98ed9193ecf1db17a5f2929a;protocol d5e23db7a3da83f5dd9a115d33e93ffbe4d891a3;analysis lock3c5fcccc61700c08ffc9efb4ba9634018cd07cf7.
ModelQwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a;BBH commit9ee07bd481feebf959a6b59d61ea57bdcf30964d.
