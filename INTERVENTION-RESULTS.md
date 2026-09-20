# Verified intervention experiment — completed 20 September 2026

## Decision

Retain the unchanged Qwen3.5-4B baseline. All 768 planned benchmark calls completed, but no correction policy earned admission. The prespecified selector chose baseline on development, before certification/evaluation labels could influence selection. No pretrained weights were changed or student trained.

Model/source execution commit: `f3bdeb83e03efb11633fd13228b4a22e58d129ab`.
Run: https://github.com/Jaksenc/parameter-golf/actions/runs/35525372043 .
Successful audit job: `106118068775`.
Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Raw numeric scores and complete results: `evidence/intervention-v1/scores.json` and `results.json` at immutable commit `463f417c22a361b0a4076bce8a0a4627897721fc`.
Float32 array SHA256: `550557a8be9d8e97364b95cbb180044749d16e0dea36bbf8cf84e05bb41fc5e7`.

## Design

128 previously unselected exact-source groups: 64 BoolQ passages, 64 PAWS sentence pairs. Splits: 24 development, 64 certification, 40 evaluation. Six views per case: original, reversed option order, logical complement NOT(P), complement with reversed order, and two option rotations. Candidate meanings are transported back to the original True/False/Unknown space. NOT fixes Unknown. Output transport is exhaustively verified, but the model's understanding of the complemented prompt is not a mathematical guarantee.

Previously used exact-source groups were excluded. All views of a source stay together. This is fresh to the project, not guaranteed absent from pretraining; no new independent human adjudication was done. Both source datasets have binary labels, so Unknown is available but never gold. Exact deduplication does not establish general independence. These are derived diagnostic subsets, not full benchmark or production UI results.

The 768 benchmark calls exclude six warmup integration calls. Execution used standard public Linux ARM CPUs, original BF16 model weights and FP32 answer-row projection, no truncation, and exact model/input checks. No Jev, paid model APIs, rented GPU, native Mac performance test or production action was used.

## Prespecified results

| Method | Calls/case | Development /24 | Certification /64 | Evaluation /40 | Evaluation NLL |
|---|---:|---:|---:|---:|---:|
| Unmodified 4B | 1 | 23 | 51 | 33 | 0.516485 |
| Original + complement | 2 | 21 | 47 | 35 | 0.553539 |
| Original + reversed order | 2 | 22 | 50 | 34 | 0.496698 |
| Four-view complement/order projection | 4 | 22 | 52 | 33 | 0.543269 |
| Four-order projection | 4 | 22 | 50 | 33 | 0.499360 |
| Prespecified selected policy: baseline | 1 at deployment | 23 | 51 | 33 | 0.516485 |

NLL is mean negative log probability of the correct outcome, lower better. These are restricted candidate probabilities, not calibrated real-world success probabilities. The selector included nine KL-limited/consensus guard candidates and baseline. No guarded candidate repaired a baseline mistake in this sample. Some shadow candidates altered confidence without changing actions; those are distinct from the stricter standalone overlay, which preserves the full baseline payload when no accepted outcome correction is made.

The two-call complement repairs two and harms zero on final evaluation, a one-sided paired sign-test p-value of 0.25. It repairs zero and harms six across the earlier splits. Across all128, descriptively INCLUDING development/certification, it scores103 versus baseline107. Four-view projection repairs3 and harms3, also107/128. There is no justified production promotion. The displayed baseline certification zero-repair/zero-harm result compares baseline to itself; it does not certify that an active correction is risk-free.

Observed median summed forward/tokenization call times: baseline4.067s, complement2 8.147s, order2 8.133s, orbit4 16.349s, order4 16.226s. These are same-shard sequential sums per case, not end-to-end service/Mac latency; loading, host/proof checks are excluded. Call counts match, exact token counts need not. No bounded reasoning-generation control was run.

## Post-hoc diagnosis, not policy selection

`evidence/intervention-v1/view-diagnostic.json` independently describes the saved scores. Across all128, individual-view accuracy is107 original,103 reversed order,68 complement,56 complement+reverse,100 first rotation,107 second rotation. The model is substantially less competent on the complemented representation despite the formally correct output transport. Negation/symbolic framing/prompt-distribution effects are plausible, not causally separated explanations.

For cross-entropy, convexity gives `loss(mean aligned_logits,y) <= mean loss(aligned_logits,y)`. The reference is average-view loss, NOT original-view loss. Projection can satisfy this inequality while degrading a stronger original view.

A gold-aware selector across all six views would reach127/128; that is an oracle diagnostic, not an achievable policy or evidence that the correct view can be recognized without labels. Twenty of21 baseline errors have a correct alternative view, but the tested guards did not reliably identify one. Do not tune new correction rules on this now-exposed evaluation set.

## Implemented correction layer and next step

The accompanying local research package implements proof-checked three-valued expressions, meaning-aligned reverse-KL symmetry projection, exponential-geodesic KL trust regions, default-deny baseline preservation, and a prospective optional-stopping-safe paired-improvement e-process.37 local tests passed, including a fresh-extraction check and numerical agreement with the executed study primitives. It is a separate score-bundle library, not a completed integration into the original app's production adapters. No production e-process certificate was issued.

The advanced mathematical components use established methods; no original theorem, guaranteed model improvement or individual no-harm guarantee is asserted. Logical proof, bounded probability movement, statistical evidence and action authorization are separate obligations.

Next, qualify each intervention/verifier's standalone capability on new development sources before letting it modify4B. Compare symbolic and natural contradiction/compliance formulations, include genuine unknown evidence, and learn residual correction reliability from separate source groups and baseline successes/failures. Keep unmodified4B and matched-cost ordering/reasoning controls. Only a successful fresh correction result justifies later intervention-informed student training.
