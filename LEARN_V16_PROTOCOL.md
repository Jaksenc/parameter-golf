# Learn v16 — pretrained probability learning-capability pilot

Primary comparison: relational supervision versus ordinary soft-target cross-entropy, paired by initialization seed and training order. Both use the exact same inputs, targets, 64 relation-pair presentations, graph, optimizer, and fixed final checkpoint. Seeds16101/16102/16103. Report every seed and unmodified-base result; no best-seed, best-step or best-metric promotion.

## Model and objective

Pinned Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Freeze the original pretrained parameters. Add rank4 FP32 factors with scale2 to the down projection of all32 language MLP blocks. A is Kaiming-initialized; B starts at zero. Train both factors with AdamW, learning rate1e-4, weight decay0.01 and global gradient clipping1.0. These are actual learned inference-changing weights, not post-hoc temperature fitting. They do not constitute full backbone finetuning or recovered Jev/RLCD.

All requests use the unchanged v14 semantic_codes system/user prompt, with no note, draft, oracle, teacher prediction or answer field. The model directly predicts allowed-code logits in one forward pass. Code-softmax at temperature1 gives reported estimates and the matching argmax answer. No autoregressive answer generation or calibration fitting occurs.

Pair loss: mean soft-target CE over the two inputs, plus lambda times ||p_right - A*p_left - delta||². Lambda=0 for supervised control and0.25 for relational primary. The verified Bridge loss is reused. Both arms compute the relation even when its coefficient is zero. Targets are reconstructed from actual text independently of the generator. Mode questions retain deterministic point targets; event questions retain complete rational distributions. Consistency by itself does not establish truth.

Memory-constrained gradients: compute both observed logits without their model graphs; differentiate the exact small paired loss with respect to those logits; recompute each input under autodiff and propagate its already-computed output gradient. Both objectives follow the same sequence. There is no stale model update between the two passes. Every recomputed logit must agree with its observation within2e-4 or the job fails. Non-reentrant activation checkpointing, no cache, and no nonzero dropout. This is extra training computation, not an inference acceleration. Gradients on base weights are prohibited; all32 initial B gradients and all32 final factor changes must be nonzero.

## Data and evaluation

Select, by a fixed input/group hash, subsets of Bridge's previously prepared but untrained worlds.96 unique fit examples from16 worlds;56 check examples from10 completely disjoint worlds. Training is exactly one shuffled pass over64 fit edges,128 example presentations with repeated endpoints. Transformations include irrelevant-count edits, genuine-count edits, reversed label order, and replication of every ticket in one mixture component. Same generated grammar and2–5 labels as parent; this is a small learning-capability pilot, NOT independently authored/general-domain transfer.

Each seed evaluates the identical72 inputs before and after training:56 held-out checks plus16 explicitly labeled training probes. The training probes cannot be reported as held-out results. Intermediate checkpoints at1/16/32/64 are archival only; final64 is fixed. No early stopping or hyperparameter selection on check outcomes. Canonical inputs and whole worlds do not cross splits. The original Bridge diagnostic inputs are excluded. No public JevBench input is trained or newly evaluated.

Frozen records hash069c87039a67b933922ef9dafcd2fc8fd47b2fe675114b55d6d9d7eb171ecf14;relations hash9d29f2a0a47d27ee4c291a7a1e35cb6475532e816992774cf62a51520c43a639. Complete examples, group assignments and edge schedules must be included in the release. Source targets are synthetic exact laws, not Jev or Qwen supervision.

Primary reported endpoints: held-out target log loss and squared-vector loss, per seed and separately for event versus modal queries. Also report TVD, modal-label correctness, false zeros, impossible-event mass, relation residual and mixture replication behavior. No metric may substitute for a failed primary after outcomes. More consistency with worse target losses is not improvement. Use the same72 baseline observations to verify zero-adapter/reload/restoration; compare all six baseline records for consistency. Any uncertainty intervals are descriptive, group-clustered, and cannot turn10 same-grammar held-out worlds into a broad confidence guarantee.

## Execution and checks

Standard public GitHub ARM CPU runners only; six paired model jobs, bounded wall-clock timeout; no paid GPU/API fallback. Assert public repository, hash model shards, validate source fixture, token/label boundaries and unchanged training data. Persist every step and checkpoint; resource failure remains explicit rather than represented as completed training. No original base parameter may enter the optimizer. Verify checkpoint reload and disabling adapters reproduces the original probe. All25 local tests passed before launching, including independent parameter-gradient replay and source-level targets. CI repeats them before all6 model jobs.

Report actual optimizer updates, trainable parameter count, every checkpoint and training loss record, before/after distributions, peak memory and all forward/backward phase counts. Hook-based adapters are not merged; do not claim zero inference overhead. No comparison to production Jev latency/cost or full534-decision composite is available. Main branch unchanged; no official submission, hosted deployment or literature-wide novelty claim.

The first question is whether this limited direct adaptation learns usefully at all and whether the relational term earns its cost beyond matched ordinary supervision. Negative or unstable results are retained. Scaling to more data/steps or different layers requires a new experiment, not an outcome-selected rewrite of this pilot.

References: Hu et al., LoRA, arXiv2106.09685; official Transformers gradient-checkpointing documentation. These support the implementation pattern, not an expected benchmark gain.
