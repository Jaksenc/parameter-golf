# Duplex v4 — completed benchmark results

Completed 20 September 2026 New York / 21 September UTC. **Every trained variant underperforms the unchanged Qwen3.5-4B reference on the JevBench public subset. No trained checkpoint earns promotion.**

Actual training/evaluation run: https://github.com/Jaksenc/parameter-golf/actions/runs/35545727146 . Execution commit `fa66e9505b53da42e232f7670bd53771e42dee10`. Frozen-prefix features came from run `35545335396`.

## Prespecified comparison

Direct, anchored, and structured objectives each ran seeds41/73/101 for400 updates, using the same384 training requests and96 development requests. Rank8, scale2,94,208 trainable terminal FFN parameters on pinned Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; base weights remained frozen. Checkpoints, including zero adaptation, were selected by development NLL. Method selection averaged the same three seeds. No benchmark input, gold, prompt search, or calibration was used for fitting or selection.

Development selected **anchored**: mean NLL1.314503 versus direct1.333262, structured1.318214, and unchanged1.940253. This was locked before transfer and JevBench. Selection SHA256: `c9359d60863005866d91d26c14112add5fbc3e3b3b7c33a965283c00c39361de`.

## JevBench public results

231 public tasks; mean values are across training replicas, not an ensemble or 693 independent tasks. Seed order is41,73,101. Lower NLL is better.

| Configuration | Correct per seed /231 | Mean accuracy | Hard correct per seed /111 | Mean NLL |
|---|---:|---:|---:|---:|
| Unchanged 4B, SemIf framing + FP32 |182|78.79%|68|0.5122|
| Direct training |164,163,167|71.28%|58,57,58|0.6558|
| Anchored, development-selected |160,173,172|72.87%|59,61,61|0.6284|
| Structured semantic-change objective |150,170,171|70.85%|55,62,61|0.6717|

The anchored replicas repair17,15,17 baseline errors while breaking39,24,27 previously correct decisions. Even the best anchored replica scores nine fewer correct answers than unchanged4B. The structured objective offers no external gain in any replica.

The unchanged reference reproduces the prior FP32 SemIf-control logits exactly on all231 public inputs (maximum difference0.0). The older BF16-native SemIf184/231 row is not the numeric comparator here. This experiment fixes FP32 scoring for all methods.

All231 public items and all10 configurations completed, producing2,310 score vectors. They use231 shared frozen-prefix forwards plus72 full-network adapter replay checks and8 baseline restorations:311 physical full-model forwards. This is not nine independent complete-model inference runs. Maximum external replay discrepancy0.0001678466796875; checked choices matched. No model/schema failure or truncation was accepted as a valid result.

This is not the complete534-decision board, official ranking, or a serving-cost benchmark. The public data have been exposed repeatedly and are regression evidence, not a new private holdout. No Jev endpoint was called or hosting price/composite invented.

## Controlled transfer results

Each split contains96 requests from12 generated source families with eight correlated views each. This is generated language, not independent human-authored production data. Means are across the three matched seeds.

| Split | Unchanged correct /96 | Direct mean | Anchored mean | Structured mean |
|---|---:|---:|---:|---:|
| New wording |43|42.00|49.00|47.00|
| New domains |40|41.67|41.67|42.67|
| Unseen truth functions |40|38.67|39.67|41.67|

Anchored wording improves by six answers on average, but formula-composition accuracy is flat. Structured has only a small composition increase on this narrow population. All methods reduce transfer NLL, without a correspondingly strong accuracy increase. Neither result warrants a broad capability claim.

The288 distinct transfer requests yield2,592 trained score vectors plus their shared unchanged reference. Eighteen additional full-network checkpoint replays passed, maximum difference1.1444091796875e-5, with exact baseline restoration. Bootstrap intervals resample source families and do not prove general-domain transfer.

## Independent verification

After downloading the actual artifacts, a separate standard-library implementation verified input identities, every saved probability distribution, labels, counts, NLL/Brier, repair/harm counts, all400 update records per candidate, and selected-checkpoint hashes. It imports neither the training metric helper nor the official benchmark scorer. External maximum probability discrepancy4.440892098500626e-16; maximum NLL/Brier discrepancy1.1102230246251565e-16. Internal float32-versus-independent loss difference1.0904408487277806e-7. All90 full-network spot checks across transfer/external evaluation passed; they are not full second runs or retraining replications.

Training artifact SHA256: `ba4db02aabbfdbedcde34d6ead84f360af81e82712893d3fcabef0b44c1f12aa`.
Transfer artifact SHA256: `b02cb9623175404f72b8f3a1d192c3c9249f80ae437f79233ed30f5aa18394c6`.
External artifact SHA256: `6359c00223f846f8f54e51e2a7f35e7614044c766a31f2e4a73005a88f71498f`.

## Execution repairs

Run35545041177 exposed a genuine compatibility bug: pinned Qwen3_5RMSNorm exposes `eps`, not `variance_epsilon`. Two attribute accesses were corrected without changing the epsilon value, inputs, prompts, objectives, seeds or update schedule. Its failed status is preserved.

A32-row ARM-cache replay on x86 showed a0.000347 maximum logit difference and about1.68e-6 probability difference, with no changed sampled argmaxes. The original x86 fit failed its strict cache check. Training was moved to ARM instead of loosening the check or regenerating data. All16 completed cache partitions were reused. The corrected same-ISA run completed all3,600 optimizer steps.

## Interpretation and next decision

This v4 recipe produces modest in-distribution wording improvements but negative transfer on the external benchmark. Formal correctness of source labels and a semantic consistency objective do not establish reusable semantic learning. Do not select the best public-test seed, train on JevBench answers, or construct an after-the-fact benchmark-family router.

A small descriptive temporal/numeric slice improves from3/15 to5,5,7/15 for anchored training, while long-policy performance falls from11/19 to8/19 for every anchored replica. That does not establish an identifiable useful specialist or justify overriding the overall result.

Keep unchanged4B as the reference. Further training should use separate diverse source tasks and explicit general-capability retention controls, with fresh validation rather than more tuning against this exposed benchmark. Both original development-only selection and all losing candidates remain visible.
