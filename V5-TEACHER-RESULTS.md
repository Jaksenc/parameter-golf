# Duplex v5 — completed teacher qualification

Completed20September2026 New York /21September UTC. Actual run: https://github.com/Jaksenc/parameter-golf/actions/runs/35551040554 . Execution commit `212cb31b60396cc74a2efba4add58215aa6395e7`.

## Result and decision

The source-bound program path substantially improves over direct numeric generation on this filtered FinQA diagnostic, but does not pass the prespecified requirement to beat both direct generation and capped reasoning. No training, distillation, deployment or leaderboard promotion occurred. The unchanged Qwen3.5-4B weights remain the reference.

All16 inference shards and the audit completed. There are96 numerical generations (32cases x3arms) and144 native retention scores. Eight owned preflight calls are separate. These are240 study outputs, not240 physical model forwards: generation performs multiple decoder steps.

| Arm | Development correct /8 | Evaluation correct /24 | Evaluation valid /24 | Median evaluation generation time |
|---|---:|---:|---:|---:|
| Direct number |1|5 (20.8%)|24|36.67s|
| Concise reasoning then number |3|11 (45.8%)|13|93.87s|
| Source-bound program + exact arithmetic |6|15 (62.5%)|24|40.24s|

Scoring matches values rounded to five decimals. The predeclared half-percent-tolerance sensitivity gives6/24 direct,11/24 reasoning,15/24 program. The direct/program gap is not solely precision or formatting.

Programs repair10 direct errors and harm0 of the five previously correct cases (paired one-sided sign p=.0009765625). They repair7 reasoning errors and harm3 (p=.171875). The gate requires both comparisons p<=.025 and>=90% valid programs; it remains FALSE. It is legitimate to investigate the direct-generation improvement, not to mark the full gate passed.

## Crucial reasoning-budget qualification

Direct generation is capped at32 output tokens; program and concise-reasoning arms at96. All use greedy decoding with native thinking mode disabled, and one attempt each. On evaluation,11 reasoning outputs hit the96-token cap and remain failures. Among the13 completed reasoning outputs, reasoning gets11 correct versus8 for programs. All seven program repairs versus reasoning occur on capped responses; all three harms occur on completed responses. This is a post-hoc diagnostic, not a replacement denominator.

Thus the program route is promising under the tested completion budget, not proved semantically superior to adequately budgeted reasoning. Timings are CPU prompt/tokenization/generation measurements within each source's shared ARM VM, excluding loading and the subsequent deterministic parser/executor. No GPU, Mac, serving-price, JevBench composite or rank is claimed.

## Meaningful implemented mechanism

All three numerical arms receive the same entire supplied FinQA excerpt/table and mechanically extracted quantity bank. Quantities carry exact rational values plus original source offsets/text. Only the program arm emits a restricted expression over these quantity IDs. The executor permits bounded +,-,*,/ with constants0,1,100, rejects calls/imports/attributes/invented quantity literals and verifies every referenced literal. Numerical reference programs, final answers, gold_inds and retrieved model_input are not forwarded to any model.

Example: AWK2018 reports0.6million repurchased shares costing45million dollars. Direct generation returns0.75; the model-written expression N28/N26 executes to75. Concise reasoning also succeeds. Conversely, DISCA2011 selects December2008 and2010 values instead of the requested September2008 andDecember2011 interval. Its arithmetic is valid while the semantic binding is wrong.

All32 programs execute,21 match the source references; nine evaluation programs disagree. Executability is not a semantic certificate.

## Scope and reference quality

Selection was fixed before outputs. Filters leave44/883 FinQA dev and48/1147 test cases eligible before selecting8 and24. This is a heavily filtered1–3operation numerical subproblem, not a representative/full FinQA or JevBench score. Exact report-page groups are separate across splits;24evaluation cases include22filings (threeAON2007pages). Supplementary filing-level sign checks preserve the direct-comparison result but do not establish broad independence. No absence-from-pretraining claim.

FinQA revision `0f16e2867befa6840783e58be38c9efb9229d742`. Its native program/answer consistency was checked for eligibility, but that does not prove faithful language interpretation. Post-hoc inspection flags BLL2012page31question4: question names DJ US Containers & Packaging (107.76), while reference uses97.13 from S&P500. Two decline questions also require sign/denominator adjudication. All published counts retain original labels and every case; nothing was removed or relabeled after results.

## Initial retention population established

144 native training-split examples:48SNLI,48PAWS,48BoolQ. Each source has24replay candidates and24sentinels, source-group disjoint. Eligible official test/validation source groups used by earlier probes were excluded before selection; pretraining and all semantic near-duplicates cannot be ruled out.

| Source | Replay correct /24 | Sentinel correct /24 |
|---|---:|---:|
| SNLI |19|18|
| PAWS |20|19|
| BoolQ |20|18|

Total114/144. Baseline logits, probabilities, original native labels, input hashes and source groups are saved. This is an initial reference bank, not a measured capability-preservation result: no weights were updated. It is not comprehensive coverage of long policies, arbitrary labels, numerical tasks or production behavior. Sentinel cases are not training data; known baseline mistakes are not automatically gold for distillation.

## Verification and implementation recovery

Original run35550920040 failed unit tests before target data/inference. An explicit wrapper repaired the number lexer so sentence-final numbers and percentages are not dropped/backtracked. Original failure/source remain preserved. Model, prompts, token caps, selection policy and scoring were not tuned against target results.

Completed ZIP SHA256 `17d2ec4deaa09f79972e3feb0acf2d019911c2708e5dc80eb2c4e6155cb2a089`. Separate standard-library replay checks all96 numeric outputs and144 probability vectors. Successful program outputs are recalculated with an independent Pratt parser rather than the executed AST interpreter. Maximum probability discrepancy2.220446049250313e-16. Repaired21unit tests and250separate random-expression comparisons plus6rejection controls pass. These tests are distinct from neural accuracy.

Model `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, unchanged BF16 weights and FP32 retention output rows. No Jev calls, paid inference API, student training or new benchmark submission.

## Next step

Keep the program route an unpromoted teacher candidate. On new independently checked source groups, compare it against a properly completing reasoning control and improve evidence/date/unit binding rather than arithmetic. Only after a qualified teacher advantage should we train with verified supervision plus independent retention controls. Do not train on these now-exposed qualification evaluation cases or lower the gate after seeing results.
