# Duplex v6 — completed teacher qualification

Completed20September2026 New York /21September UTC. All128 planned numerical outputs are preserved and independently checked. No training, distillation, deployment or JevBench submission occurred. Neither program teacher qualifies.

## Actual results

32 previously unselected company/year filings:8development,24evaluation. All v5 filings excluded. Same unchanged Qwen3.5-4B checkpoint, common enriched quantity bank and evidence, greedy decoding, native thinking mode disabled. Direct cap64 with prefilled FINAL=; reasoning/program/binding cap512. This is not the prior v5 sample, so differences cannot be attributed to budget alone.

| Method | Correct /24 | Valid /24 | Capped /24 | Median completed-response time |
|---|---:|---:|---:|---:|
| Direct number |5|24|0|98.1s|
| Reasoning then number |11|24|0|201.9s|
| Compact program + exact executor |4|15|0|98.8s|
| Program + explicit operand bindings |7|17|0|194.7s|

Primary accuracy uses the frozen five-decimal reference check. Development correct counts are1,5,3,1 respectively; one development reasoning response was capped and retained as failure. All evaluation reasoning responses completed.

Binding versus direct:3repairs/1harm,p=.3125; versus plain programs:5/2,p=.2265625; versus reasoning:2/6,p=.96484375 in the improvement direction. Binding validity17/24 fails90%. The predeclared qualification requires it to beat all three alternatives with paired one-sided p<=.05/3 and both binding/reasoning validity>=90%; it remains false. No threshold was lowered.

Time includes prompt/tokenization/generation and subsequent parsing/execution, excludes loading, queues, artifact transport and aborted attempts. Six recovered completions use different same-class ARM workers. These are not GPU/Mac/campaign/production-serving cost measurements. Generation outputs are not physical forward-pass counts.

## Source quality and sensitivities

Filters left105questions from72filings out of2030 pooled native FinQA dev/test rows. This is a heavily filtered derived task, not official or representative FinQA. The24evaluation filings span21companies. Source independence and absence from pretraining are not established.

Seven evaluation references were flagged BEFORE target inference for wording, date, unit, count, or sign ambiguity; one development reference was also flagged. All original labels remain primary. Excluding the seven preflagged evaluation records gives direct5/17,reasoning10/17,program4/17,binding7/17. This is assistant-reviewed sensitivity, not independent human adjudication. A separately labeled post-hoc half-percent numeric-rounding sensitivity gives direct5/24,reasoning13/24,program4/24,binding7/24; it does not replace the primary gate.

## Interpretation

This completed reasoning control is stronger than both tested program interfaces on this sample. It does not prove programs are intrinsically worse: nine compact-program and seven binding evaluation responses were rejected by the output contract. The restricted grammar permits literal constants only0,1,100. A legitimate mean `(N1+N2)/2` is rejected. Invented table indexing, trailing prose, inline code fences and surplus cited quantities cause other failures. These are partly our interface design limitations and must not be reported as pure reasoning failures.

Ignoring only source-copy metadata requirements in a post-hoc diagnostic leaves19binding expressions executable and9correct; two correct calculations were rejected solely by metadata validation. This is not an authorized rescored policy. Incorrect units, periods, operands and operations remain among executable results. Valid citations and arithmetic do not prove faithful interpretation.

Example: MO2014's exact `(5070-4535)/4535` is correct in compact/binding programs, while reasoning's final0.12 fails the fixed precision check. PNC2015's genuine two-value mean is rejected because2 is forbidden. AES2010's authentic binding expression evaluates to zero despite the semantic error. No outputs were repaired or relabeled after observation.

## Actual system improvement

The downloadable package adds automatic source-provenance attachment to compact expressions. It authenticates request hashes, source offsets and numeric literals, then attaches the source spans without model-generated citation JSON or another neural call. Replayed on all20valid compact-program outputs across both splits, it preserved every value. It explicitly reports semantic_interpretation_verified=false and authorization_to_act=false. This is postprocessing verification, not a new accuracy arm; it does not repair invalid expressions or choose better operands.

## Execution provenance

Main run35553557091, commitc5ab503a032a055d1d6f905b351a0e9ddaee40dd, saved122responses; four workers exceeded24-minute job limits. Recovery35555099244, commit8fa8494e6ac1ff46ef5169cc367b804514099873, completed exactly six absent pairs. No recorded invalid/capped/wrong response was retried or overwritten. All original fields are unchanged; generation-function AST is unchanged. Four interrupted decoder attempts and five owned preflight generations are separate. Recovery audit job106198802088 succeeded.

The original owned preflight caught direct generation explaining rather than returning only a number. Only its FINAL= prefix was repaired and revalidated before target inference. Earlier failures remain recorded.

All20worker receipts match pinned model weights and source. Independent Pratt-parser replay checks all128input identities, status decisions, values, reference labels, aggregate counts, paired repairs/harms, and1598source quantity spans. It imports neither the study's scorer nor its AST executor. Maximum input8286tokens, no silent truncation.46local tests pass, including500random arithmetic comparisons and provenance integrity controls. These are software checks, separate from neural accuracy.

Model revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
FinQA revision0f16e2867befa6840783e58be38c9efb9229d742.
Input hash485956e198fcaf2a01c021c00ccde5f629ac53316323797b9504420e2441bbd8.
Completed evidence ZIP SHA256204970cbacc3da0ade5b091ecc0d905d1217dc0c886f9f44d00dd020c0ec90f9.

## Next decision

Keep unchanged4B and do not distill these unqualified teachers. Before another held-out study, validate a compact, valid-by-construction program grammar with explicit mean/count/unit-conversion operators, automatically attached provenance, and a declared numerical precision contract. Do not tune against these exposed outputs or treat syntax validity as semantic truth. No new model or leaderboard performance is claimed.
