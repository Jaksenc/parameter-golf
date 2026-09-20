# Duplex v3 research results — 20 September 2026

## Decision

The learning sanity check now passes, but no new trained model earns production or leaderboard promotion. Keep the unadapted Qwen3.5-4B reference with explicit FP32 scoring. The new reusable local core distinguishes conditional answer scores from event forecasts and exact execution. Do not enable the old fact-supervision adapter.

## Actual new learning result

Run `35541733965`, job `106160512702`, source commit `676d56ae9da46c81e2984a6c3759a3337293d34a` completed successfully. Model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The new method caches the frozen inputs and residuals to the terminal language FFN, then optimizes a rank8/scale2 adapter on its down projection. It has 94,208 trainable parameters. The cached computation includes the actual final RMS normalization and FP32 allowed output rows. This is an exact factorization for this placement, not a novel theorem or a general cache for earlier-layer updates. Inputs are unpadded and identical between training and inference. SemIf-style framing is used.

| Training set | Baseline correct | Trained correct | Baseline NLL | Trained NLL | Updates |
|---|---:|---:|---:|---:|---:|
| Micro12 | 8/12 | 12/12 | 0.66444 | 0.01110 | 50 |
| Full54 | 30/54 | 54/54 | 1.09576 | 0.03481 | 110 |

Independent starts, seed41, full-batch AdamW lr0.0005, no weight decay, gradient clip1. Stopping uses only training fit: three consecutive 10-step checks with all examples correct and NLL below0.1. No JevBench inputs or labels were used.

This is NOT generalization success. On already exposed controlled new wording, full54 training changes30/54 to26/54 and NLL0.933 to2.349. On exposed formula combinations,10/18 becomes11/18 and NLL0.899 becomes1.152. Across those72 cases the adapter loses three correct decisions. Small recurring synthetic contexts, one seed and multiple simultaneous recipe changes prevent a broad or single-cause claim.

The backbone cache takes785.70 seconds. Optimizer-only loops take0.107 and0.273 seconds respectively; those figures exclude loading, cache building and replay, and must not be called end-to-end training times. Eight full-network post-training replay cases agree within9.54e-6 maximum logit error, with identical choices. Baseline cache checks are within1.15e-5; baseline restoration is exact on the checked input. Saved checkpoint hashes and reported accuracy/NLL were independently verified after download.

Learning artifact SHA256: `e017bb08d0f523ddbd9ca93fd87b6df287faa6a4cfade6f01582e8feb8da62d7`.
Micro adapter: `a27494fcec4e50b0d57754110f43b1cc67a19f92783e49e8583e785c08d18804`.
Full adapter: `87ebcbfe7f8d2a21adce4d230b4d3f29343f887f9498f2b5b1976ff04cbc8efb`.

## Actual new inference optimization

Run `35542033181`, job `106161305622`, source commit `8c32a796009f4276fd9f3d2f1dba277239295493` completed successfully. Restricted output projection avoids materializing all vocabulary logits before recomputing the allowed rows in FP32.

Six development inputs, three repeats, alternating order, one shared CPU-loaded model:18 paired comparisons,36 measured calls plus2 warmups. Maximum paired logit discrepancy0.0. Median reference duration6.17543s versus5.91084s for the restricted head:4.2846% lower median latency. All18 observed pairs are faster, but these are correlated small CPU measurements, not full JevBench, native Mac or production-throughput claims.

Head artifact SHA256: `6c8df5224371c096cbbed3328a87d597a06afd480b38e6ace180e7a64e4261c6`.

## Further diagnosis of the earlier benchmark

Independent replay of the924 previous predictions shows1 wrong answer above95% confidence for the original baseline,7 for the direct adapter,34 for the fact adapter and3 for native SemIf. The fact adapter's median centered-logit scaling versus base is3.679 with median cosine0.973: predominantly stronger versions of similar preferences, not reliable new corrections. This is descriptive geometry, not proof of a unique causal mechanism.

## Implemented local core and boundaries

The delivered core uses fixed SemIf-compatible framing, explicit probability meaning, a scoped FP32 allowed-row head, request hashes and no automatic experimental adapter. A separate bounded executor implements rational arithmetic, exact thresholds, count-based event probabilities and offset-aware time differences over source-bound programs. It does not interpret arbitrary language or certify semantic translation. The event soft-target objective is separately tested, not trained on the benchmark.36 local tests pass, separate from the real-model jobs.

Experiment source is in this branch. A connector safety-status block prevented posting the combined core file; it is delivered in the downloadable research bundle rather than claimed as a repository update.

## Durable research rules

- Working fit is necessary diagnostic evidence, not transfer or product quality.
- Preserve a raw strong same-backbone control; do not conflate framing with learning.
- An event probability is not the model's confidence that an event is the modal answer.
- Mathematical correctness of execution does not prove the natural-language interpretation.
- More correlated calls are not free intelligence; include speed and cost effects.
- Do not train or calibrate on exposed JevBench gold; the231 public items are regression only.
- No official full534-case score/rank exists for this configuration.
- Keep Jev development calls blocked under the existing permission restriction.

Research consulted: Calibrate Before Use (2102.09690); Program of Thoughts (2211.12588); Distilling Step-by-Step (2305.02301); LoRA (2106.09685); LoRA Learns Less and Forgets Less (2405.09673); LoRA+ (2402.12354); Bayesian Low-rank Adaptation (2308.13111); calibration (1706.04599), uncertainty under shift (1906.02530), Lost in the Middle (2307.03172), SGLang (2312.07104), official Qwen source and pinned JevBench composite code. No cited paper's gains are treated as predicted Duplex gains.

Next: varied, checked semantic training sources with controlled direct-answer exposure, followed by fresh transfer evaluation. Independently validate language-to-program faithfulness and representative probability calibration. Only then compress a demonstrated capability. No further model jobs are launched by this report.
