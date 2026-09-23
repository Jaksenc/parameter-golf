# Causal v17.1 — completed clean factorial, audited September 23, 2026

## Completion and scope

Clean run35912706972 (commit c23e27357dbdf3521c3f4c034f0679b4c6a334d2) was already complete when this continuation retrieved it. No duplicate pretrained inference or training was launched in this continuation. All twelve trajectories restarted from paired initializations, completed32 real updates, transferred full state between distinct16-update processes, and retained final predictions. The contaminated run35900447533 is excluded.

The deployed repair deep-copies optimizer state on every restoration, checks moment fingerprints across discarded diagnostic trials, verifies real optimizer counters after each actual update, and verifies optimizer state in checkpoint roundtrips. All12 final AdamW states have counter32. Original parameter/projection/restoration records agree. This is not a reset of AdamW around contaminated checkpoints.

Four treatments: event-only or event-plus-modal supervision at learning rates1e-4 and1e-5, three paired seeds17101/17102/17103. Event loss weight is0.5 in all arms; modal loss weight is0 or0.5. Both gradients/computational branches are computed in all arms. Same examples, initialization, schedule and32-step budget within each seed. Clipping and AdamW dynamics are part of the treatment.

Frozen Qwen3.5-4B base, rank-four adapters on32 language MLP down projections,1,507,328 trainable parameters. No relation loss, generated rationale, teacher probability, Jev API, or oracle note. Each seed sees32 distinct fit worlds selected from128. There are32 independent calibration worlds and32 test worlds from four authored mechanism grammars. Evaluation contains256 requests including related variants, not256 independent worlds. No new JevBench score or general-model superiority claim.

## Results: base event distributions on32 test worlds, means over three seeds

| Treatment | Raw target CE | Raw squared-vector error | TVD | Leading category correct |
|---|---:|---:|---:|---:|
| Unchanged |1.203942|0.144456|0.266355|46.875%|
| Uniform reference |1.196873|0.132419|0.272305|Arbitrary tied ranking|
| Event-only,1e-4 |1.143255|0.106022|0.234341|55.208%|
| Mixed,1e-4 |1.167010|0.121515|0.249253|56.250%|
| Event-only,1e-5 |1.201847|0.143441|0.265216|46.875%|
| Mixed,1e-5 |1.205966|0.145956|0.267304|45.833%|

Squared-vector error is excess expected multiclass Brier. Event-only/high improves both raw proper losses in all three seeds; means improve CE5.04% and squared error26.61%. CE difference versus unchanged:-0.060686, descriptive95% interval[-0.107477,-0.019230]. Squared difference:-0.038434, interval[-0.074213,-0.011377]. Intervals use10000 crossed paired-seed/whole-world resamples, conditional on three seeds and32 authored worlds, with no multiplicity correction. No best-seed or best-checkpoint promotion.

Mixed-minus-event at1e-4 raises raw event CE by0.023754 (interval[-0.001966,+0.048283]) and squared error0.015493 (interval[+0.000747,+0.030333]). Lowering the rate tenfold nearly eliminates learning at this32-step budget; it is not established as a remedy. This says nothing about eventual convergence at a smaller rate.

## Calibration qualification

A temperature is fitted independently for each model/quantity using only32 calibration questions of that quantity, on the predeclared121-point grid0.25–16. Unchanged receives the same treatment. No test-selected temperature.

| Treatment | Calibrated event CE | Calibrated squared error |
|---|---:|---:|
| Unchanged |1.174612|0.125200|
| Event-only,1e-4 |1.138126|0.101414|
| Mixed,1e-4 |1.138598|0.102657|
| Event-only,1e-5 |1.174012|0.125035|
| Mixed,1e-5 |1.175670|0.125945|

The mean event-only improvement survives independent calibration. But the mixed-minus-event gap becomes0.000472 CE (interval[-0.016403,+0.017970]). Mixed uses softer event temperatures1.625–1.932 versus1.366–1.516. This supports a confidence-scale contribution; it does not establish that modal supervision substantially destroys event reasoning information after calibration. Calibrated intervals are conditional on the fitted temperatures.

## Limits exposed by controls

Raw modal accuracy is50.00% unchanged,52.08% event-only/high,48.96% mixed/high,and50.00% for both low-rate arms. Raw modal CE worsens slightly under event-only/high and more under mixed/high. After separate modal calibration, their mean CE is1.0541 unchanged,1.0364 event-only/high,and1.0020 mixed/high. Ranking and distribution quality must remain distinct.

Event-only/high raises replication inconsistency0.004229 to0.005800 and paraphrase inconsistency0.011400 to0.013928; distractor inconsistency also worsens. Its relevant-edit delta error improves slightly. Better base estimates are not proof that the desired invariance/general algorithm transferred.

On eight fixed two-category probe worlds,event-only/high CE is0.7921 before training,0.7970 at1,0.8166 at8,0.7437 at16,and0.6924 at32. The earlier eight-step view would have suggested the wrong endpoint conclusion. These probes neither select a checkpoint nor represent all category counts.

A post-outcome illustration: class counts jade5/silver8 and pass likelihoods9/10,1/10 give P(jade|pass)=45/53=84.906%. Unchanged reports32.740%; event-only/high seeds report37.153%,51.060%,47.925%. Estimates improve but remain far from exact; two still rank the wrong outcome. All alternatives are preserved.

## Verification and remaining diagnostic caveat

The independent audit checks512 exact source targets and4288 stored vectors including probes, max storedFP32 probability discrepancy1.40e-7. It verifies384 actual update ledgers,72 discarded trials,3072 final trained outputs,960 probes,and256 reused baseline outputs. Full-state/factor equality, paired initializations/schedules, all12 final optimizer counters,and recorded zero restoration errors pass.

A separate Torch-float64 implementation reproduces26 calibration fits and every121-point grid(max loss difference2.22e-15). Decimal60 arithmetic reproduces52 base metric summaries(max difference6.66e-16). All384 combined gradient norms reconstruct within4.11e-6. Forty-five current tests pass:25 original/deployed-repair tests and20 independent arithmetic/source/reducer tests. No model is run by these offline checks.

The deployed repaired trainer is not identical to the later stronger safe_diagnostics helper in the previous local package. Its trial before-loss uses grad-enabled execution and after-loss uses no_grad. Local trial loss differences are therefore not promoted as execution-mode-controlled causal proof. The core optimizer-aliasing defect is fixed; the full factorial conclusions do not rely on those local diagnostic deltas.

One relevant-edit event variant has a tied mode; its probability target is valid and its optional ranking statistic uses first-maximum convention. All primary base worlds have unique modes. Current examples remain same-grammar and deliberately shortcut-discriminating. Original two-category probes were previously inspected during the invalid run; no scientific treatment was chosen from them.

Decision: retain the unchanged deployment reference. Carry event-only/high and calibrated mixed/high into an untouched transfer/retention comparison, retaining all paired seeds and proper losses. Do not add a relationship objective merely because one current arm looks best. This is a clean bounded learning gain, not a JevBench or production win.

Raw archive causal-v17-clean-complete.zip:463450491 bytes,SHA25698c6a1a3f0e4f39429e8605cc7944513b2cbf0f905390e65e7338a08a4a3962d,artifact10777138863. Trainer SHA256c46f27f29b44f83426830d2c4a3ff265f6c6211952c98167d24d3248f44c7cad. Model revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Main branch unchanged.
