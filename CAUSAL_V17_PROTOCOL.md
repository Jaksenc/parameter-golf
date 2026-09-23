# Causal v17 — fixed before new training outcomes, September 23, 2026

Question: does adding modal-answer supervision worsen event-probability learning, and is the existing step size responsible? This is a bounded causal pilot, not a scale-up, convergence study, recovered Jev or performance promise.

First evaluate EVERY saved v16 step1 and step16 checkpoint (three seeds, two old objectives, 72 old evaluation inputs). Step32 observations already exist. Those old worlds are diagnostic, not untouched confirmation. No checkpoint is selected or promoted from this trajectory audit. Old64 endpoint remains incomplete.

## New factorial training

Same pinned Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, rank4 FP32 factors on all32 language MLP down projections, scale2, original weights frozen, existing semantic_codes readout with T=1. No generated reasoning, output-format change, relational loss or teacher estimates. AdamW weight_decay0.01, gradient clip1.0; learning rates1e-4 and1e-5. Three paired seeds17101,17102,17103. Four arms event_high/mixed_high/event_low/mixed_low.

Every step uses one matched event/modal world pair. Compute and retain BOTH full parameter gradients in every arm. Apply 0.5*g_event + 0.5*w*g_modal, where w is0 or1. Thus event examples, event loss weight, both computational branches, selected32 worlds, and initialization are identical across the four arms within a seed. Different clipping/optimizer directions are part of the treatment; gradients are not artificially rescaled to equalize them.

Each trajectory has32 fixed optimizer updates. It is deliberately divided into16+16 updates in distinct processes with full-state transfer. Save model factors, optimizer, Python/NumPy/PyTorch CPU RNG, source/data/schedule bindings and progress atomically every step. Save factor-only checkpoints at1,8,16,32 and fixed probe predictions at0,1,8,16,32. No best-checkpoint selection or optional early stopping. Failing jobs retain committed state; no silent reset of AdamW. Resume checks fixed-runtime environment and probe logits. Full-state helper passed a dropout-using toy exact-resume test; actual Qwen roundtrips and cross-process checks are required before claiming live resumption.

At steps1,16,32 perform diagnostic event-only and modal-only tentative updates from the SAME factors/optimizer/RNG state, evaluate both losses on the current training pair, then restore the state before the actual assigned update. These discarded counterfactual steps are counted separately. They establish only local effects on those inputs; no general interference conclusion from gradient cosine alone. Actual event/modal gradients, cosine, clipping norm and losses are logged every step.

## Data and evaluation

Generator seed170923261. 192 distinct numerical worlds:128 fit-pool,32 calibration,32 final-test. Four authored mechanisms(counts, conditioning, mixture, Bayes), two target-entropy strata, two-to-five categories. All event targets here are positive, modes unique; zeros and ties remain untested. For each seed select exactly one fit world from every family/category/entropy stratum:32 unique training worlds,64 input presentations. This is NOT training on all128 fit worlds. The pool has half its worlds selected to disagree in modal label with a specified wrong computation. Every calibration/test world must discriminate its specified shortcut; this is an intentional challenge distribution, not population-representative sampling. The generator rejects shortcut ties before rotation, a pre-inference test fix.

Final evaluation:32 test base event questions and32 test modal questions, plus128 test event variants(replication, irrelevant edit, relevant edit, paraphrase). A separate64-question calibration split gets no optimizer updates. All related variants stay within one world/split. The new evaluation contains256 requests, not256 independent worlds. Fixed learning-curve probes are16 questions from eight TWO-CATEGORY test worlds; their curves must not be generalized to all category counts. No probe outcome changes optimizer choices. No independent-authorship benchmark or general capability-retention claim.

Record data hash26ce5f59a61db89f66582935ffd3461c0e386a87487682819a5c23a7a9d4228e; world hash6a51e65fc9da603db67992d7c4062c3935fad6a60560ebdec7606eb37e30effb. Targets independently enumerated; current19 preflight tests pass. Source hash manifest is retained before new model outputs. Label-free model requests contain id,state,question,labels only.

## Frozen reduction

Primary endpoints: raw cross-entropy and squared-vector error on32 base test event questions. Publish every arm/seed versus unmodified and uniform baselines; factorial contrasts mixed-minus-event at each rate and low-minus-high within each objective. Publish modal accuracy/proper loss, entropy, all transformation outcomes, paired delta error, and the two-category probe curves. Treat seed and world as paired sampling units;10000 crossed seed/world resamples give descriptive intervals conditional on three seeds and32 authored worlds, not broad guarantees. No outcome-dependent arm promotion.

Fit one scalar temperature per completed model and requested quantity using only the32 calibration questions of that quantity: fixed121-point log2 grid[-2,4]. Keep raw outcomes primary; report independent-test calibrated scores separately. Baseline receives its own calibration under the same rule. Temperature cannot change rankings. Never tune on test or old outcomes. No invented one-hot uncertainty vectors; all outputs are measured code logits/softmax. No full JevBench score is generated.

Costs: count all forward/backward work and discarded diagnostic updates. Separate loading, active training, evaluation and checkpoint transfer. Standard public-repository CPU workers only; no paid provider calls, Jev API use or main-branch changes. Equal graph does not assert identical wall-clock timing. Numerical source parsing, gradient addition and checkpoint recovery are tested components, not novel research claims. Prior art: LoRA(arXiv2106.09685), task-gradient interference(arXiv2001.06782), PyTorch general checkpoint guidance.
