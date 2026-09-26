# Adaptive Quality: completed v0.4 experiment

Audited September 26, 2026. Model training and evaluation executed September 25, 2026 in Actions run **36149615951**, frozen source **3f143520e2fd946f99de619522fbe0a3c7a9d8ab**. The earlier blocked-status report missed this later successful run. No new pretrained inference or optimizer updates were launched in this recovery.

## Complete, matched training

Both grouped and shuffled arms completed **64 optimizer updates on the same 256 unique training decisions**: 192 authored decisions and 64 selected human-labeled BoolQ training examples. They use the same Qwen3.5-4B BF16 backbone, seed 7041, and byte-identical initial rank-4 adapters (1,507,328 trainable parameters across 32 MLP output projections). Actual execution uses singleton gradient accumulation, four examples per optimizer update. The original padded-batch precursor failed its numerical-equivalence check and was stopped.

Each arm has 392 saved base/adapted evaluations: **784 records total**. There are 180 ordinary evaluation tasks plus 16 privileged normalized representations of transfer tasks, not 784 independent questions. Base logits match exactly across the two runs. No missing evaluation records remain.

| Evaluation | Unadapted base | Grouped | Shuffled |
|---|---:|---:|---:|
| Authored transfer | 34/64 | 45/64 | 41/64 |
| Composition | 26/32 | 30/32 | 28/32 |
| BoolQ validation/retention subset | 16/32 | 23/32 | 24/32 |
| Exposed JevBench easy/original slice | 16/20 | 18/20 | 18/20 |
| Development | 19/32 | 22/32 | 23/32 |
| Privileged normalized-input diagnostic | 5/16 | 11/16 | 10/16 |

Grouped transfer repairs 15 base errors and breaks four correct answers. Grouped versus shuffled repairs five and breaks one. This is one seed and changes both minibatch composition and update order; it does not isolate a universal benefit from grouping.

## Important limits

- Composition is imbalanced: the base rejects every case. Grouped correctly permits 4/6 eligible cases and rejects 26/26 ineligible ones; balanced accuracy rises from 50% to 83.3%. **Neither adapter solves an entire 16-case truth table (0/2).**
- Entire four-case transfer groups correct: base 2/16, grouped 5/16, shuffled 3/16.
- BoolQ subset is balanced and length-limited, not a full benchmark score. Base says yes to all 32. Grouped repairs 16 false cases but loses nine previously correct true cases. Shuffled repairs 14 and loses six. Neither is a no-regression retention result.
- Both JevBench repairs are the same already-exposed scenario, original-policy-05-0/1. They were also repaired in the earlier v0.3 pilot. **No new independent JevBench gain, hard-tier result, sealed evaluation or leaderboard rank is established.**
- Primary authored scores retain recorded canonical-ledger tie behavior. A separate lexical-tie sensitivity analysis does not replace predictions; public-slice scoring agrees with the preserved upstream scorer.
- The actual completed run predates the separate local v0.4.1 transactional-checkpoint repair. The two implementations are not conflated.

## Verification

Recovered original artifact SHA-256 values:

- Grouped, artifact 10875938235: `e2ddce0dfed63ce5c92c2bceabc0d562b67f4a9c5cd5ed45f0bc8f3ccc18733f`
- Shuffled, artifact 10876337171: `2a26f0506cf8f0874f39c8bb9d6598d18687c60d34f38404a87f7439de40962b`

Final adapter SHA-256:

- Grouped: `294cccbf73fcec5d3a268961aa4ddf0f63ecd8bc4acaf1fb5d4c43bf7b812bac`
- Shuffled: `4fbb9e06e6a918b73cd48484d866fb23c7050b9939fc688f21001644aba72cca`

Independent audit reconstructs both training schedules, 320 authored labels, BoolQ subset selection from 12,697 archived JSONL rows, native distributions from all 784 recorded logit vectors, and public-slice scores. Checkpoint tensors match saved step-64 optimizer state. **38 new audit/runtime tests and 53 preserved repair tests pass.** These are software tests, not additional model-quality evidence. The audit does not rerun the 4B forward pass; Parquet bytes are hash-verified, but local independent Parquet decoding was unavailable.

Both adapters remain explicit research options; no default model is replaced. Original source, failed records and trained checkpoints are preserved in the companion evidence package. This documentation-only commit does not start a training workflow.
