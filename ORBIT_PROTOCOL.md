# Orbit v2 — pre-outcome research protocol

Goal: test whether answer-order sensitivity is an identifiable nuisance, not assume that invariance produces truth. This is an independently engineered method, not recovered Jev or a claim of first-ever novelty. Relevant prior art: PriDe (arXiv:2309.03882), Set-LLM (arXiv:2505.15433).

## Mathematical hypothesis

For K choices and order v, centered native logits aligned to semantic candidates satisfy y_v = s + P_v b + interaction_v. Both s and b are centered. Identity plus reversal cannot identify reversal-even components of b; rank(I-R)=floor(K/2). Identity plus a one-step cyclic permutation identifies K-1 centered bias directions under this additive model. A third, reversed view makes the model overdetermined for K>2 and exposes interaction residuals. K=2 reversal equals the cyclic shift and is not treated as an independent observation.

We will implement graph-synchronized, regularized nuisance separation: minimize sum_v ||y_v-s-P_v b||^2 + lambda ||b-b_prior||^2, subject to sum(s)=sum(b)=0. Priors use only independent training inputs. Hyperparameters and choice of method use only development labels; scalar temperature uses only calibration labels. The native unmodified control must remain a selectable outcome.

Candidate family is fixed: native, reversal arithmetic averaging, three-view arithmetic averaging, three-view centered-logit averaging, globally learned slot-bias correction on native logits, and per-input joint nuisance separation with ridge lambda in {0,0.1,1,10,100}. Global bias is fitted on training input order differences without using correctness labels, separately for observed K, and falls back to centered zero for unobserved K. For global correction, strengths are {0.5,1}; no benchmark-dependent correction strength. Per-input methods use all unique observed orders. Candidate selection minimizes development mean target cross-entropy, ties resolved by fewer views then lexicographic arm name. Report every arm, without retroactively selecting a benchmark winner. Freeze selected arm before opening new benchmark/fresh-test outcomes. Calibrate each arm separately on the same independent calibration inputs, using the existing fixed 121-point log-temperature grid [0.25,4]. Temperatures are not included in development selection.

Primary comparison: selected method versus native with separately fitted temperature. Matched-compute controls: simple three-view averaging and centered-logit averaging. Report accuracy, NLL, Brier, ECE, paired cluster bootstrap on reused JevBench, order-fit residuals, required views, and measured new forward cost. No full leaderboard composite, speed equivalence or production-price claims.

## Data and execution

Reuse the pinned Qwen3.5-4B native/reversed logits and 301/73/73 deduplicated train/development/calibration inputs from v1. Add exactly one cyclic-order forward for eligible K>2 original items. For K=2, reuse the existing reversed output because it is the identical permutation. Original public JevBench 231 items remain evaluation-only. This is a reused public diagnostic, not a fresh sealed test.

Add 128 new independently generated executable tasks, never used for training, method selection or calibration: ledger arithmetic, sequential scheduling, pointer traversal and affine modular-state updates, 32 each, with 2-5 choices. The generator uses seed 902731, independent reference calculations and no benchmark text. Their inputs and labels are stored separately. New tasks test generalization only; no modifications may be chosen based on their model outcomes.

All orders preserve semantic candidate labels, changing only presentation slots. The architecture's behavior under semantic cross-references is a limitation, and future arbitrary production inputs must reject or separately handle positional references. Published Jev outcomes are for the old matched-ID public set only; no Jev comparison is available for the new executable set.

Every extraction shard verifies pretrained weight hashes, full/readout fixture, exact prior native prompt hash and numerical native-anchor agreement. No generated answer text or Jev API queries. No public-main modification or leaderboard submission. Scientific findings may be negative; no forced promotion.

## Boundaries

This experiment does not update the backbone, reconstruct RLCD, establish universal calibration, or show that counterfactual order observations measure correctness. A cyclic intervention can identify additive position bias mathematically while neural content/order interactions may violate the hypothesis. That distinction is itself an intended test outcome.
