# Finite-state intermediate-supervision study — separate from Witness

A learned bank of32 stochastic17x17 transition matrices has9,248 trainable logits. Sixteen actions add residues modulo17 and sixteen multiply nonzero residues. Each step propagates the model's own probability distribution; it never receives the gold intermediate state as its next input.

Primary comparison: same near-uniform random initialization (logit SD0.01), same programs, architecture, Adam optimizer,600 updates, batch64 and forward computations. Only the loss differs: terminal-only cross-entropy versus mean cross-entropy on every intermediate state. Intermediate supervision provides more labels; this is not an equal-information-budget experiment.

Training lengths2–6. Evaluation covers all544 primitive input/action transitions plus2,048 programs at each length2,6,12,24. Training seeds0,1,2 share a fixed test population. With terminal-only supervision length24 accuracy was5.957%,5.127%,5.957%. Intermediate supervision reached100% on every tested length and every primitive transition in all three seeds; its length24 NLL remained0.328–0.340, not zero. Chance is1/17, approximately5.88%.

An exact-uniform control showed the same direction. An exploratory identity-initialization control, added after inspecting primary results and documented before its own run, used4*identity logits with no operation targets. Terminal-only length24 accuracy was5.811%,5.908%,6.348%; intermediate supervision again reached100%. These are robustness checks, not retroactively preregistered primary evidence.

Mathematical mechanism: for uniform row-stochastic U and any zero-row-sum tangent dT, dT U=0. A uniformly mixing suffix erases the terminal effect of changes in an earlier transition. Intermediate losses provide local gradients before suffix mixing. This is a property of the chosen architecture, not a proved explanation of transformer or Jev errors.

All18 checkpoints were saved. Independent NumPy probability propagation reproduced all measured accuracies and NLL within3.754e-7. A zero-row-sum test gave residual7.38e-17; the isolated nonterminal gradient through an exact uniform suffix was zero.

These are finite-state composition results, not language understanding, unseen operators, arbitrary arithmetic, JevBench results or new4B backbone weights. Different curriculum, architecture, optimization or training budget could improve terminal-only learning. The full source, checkpoints, protocols, per-seed metrics and independent audit are in the conversation release's learning-study folder. This study ran locally on CPU and did not use Jev API calls or paid compute.
