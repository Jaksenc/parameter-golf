# Witness v9 — completed September 22, 2026

## Result and decision

A proof-producing logic specialist improved a fixed, previously unused 64-item BBH formal-fallacy sample from **47/64 (73.44%) to 56/64 (87.50%)**, nine repairs and zero regressions. The original frozen primary accepted 30 cases, all30 agreeing with source labels; ordinary480 got21 of those30 correct. Unsupported cases preserved the measured ordinary480 output. This is a controlled-grammar algorithm plus a frozen model fallback, not new model weights or recovered Jev/RLCD.

**Public JevBench did not improve.** All231 public cases fell outside the specialist's exact entailment interface, so every answer remains the archived ordinary480 result: **202/231**. Historical Contrast203/231 and archived Jev200/231 are not new specialist gains. No full534-decision composite, hidden-set evaluation, calibrated probability, comparable serving performance or official submission is claimed.

The English compiler is **not promoted as a trusted semantic verifier**. Broader audits found wrong translations even when the formal certificates checked correctly. The release defaults to explicitly formal inputs; English interpretation is an experimental opt-in with known remaining defects.

| Method | Matched64 unused logic cases | Symbolic replacements |
|---|---:|---:|
| Ordinary480 | 47/64 | 0 |
| Frozen Witness v9 hybrid, primary | 56/64 | 30 |
| v9.2 guarded hybrid, post-audit sensitivity | 55/64 | 28 |

The primary gain is14.06 percentage points; descriptive paired item-bootstrap95% interval[6.25,23.44], exact discordant-pair p=0.00390625. Resampling uses10000 draws within one shared synthetic template generator. This does not establish independent-template/general-English performance or remove repeated-research/multiplicity concerns. The guarded sensitivity is not a retroactively selected primary or a new confirmation result.

## Mechanism

The entire recognized argument is compiled to equality-free, nonnested unary-predicate classical logic. The kernel tests premises AND NOT conclusion. It returns a finite countermodel when satisfiable, or an exhaustive unary-type contradiction certificate when unsatisfiable. A separately written checker uses dictionaries/Boolean evaluation rather than the solver's integer-type/set implementation. There are at most12 predicates and4096 types; nonempty domains, no existential import from universals, no equality or two-variable relations. This is exact only for the stated formal fragment.

Source hashing, complete span accounting and reparsing bind the proof to the compiler output. The source grammar is shared by that binding check; it is **not independently semantically verified**. Ambiguous and/or parenthesizations retained by the grammar must agree before a label is accepted, but other lexical readings are not exhaustively represented.

Example repair: E implies NOT B, F implies NOT E, therefore F implies NOT B. Assignment E=false,B=true,F=true satisfies both premises and falsifies the conclusion. The ordinary model said valid on the source item; the checked countermodel yields invalid. No model-generated prose or answer-key lookup is used in that proof.

## Staged error discovery — original results retained

| Fixed version / new source population | Total | Accepted | Source-label matches among accepted |
|---|---:|---:|---:|
| Original v9, all234 unused BBH formal rows including64 sample | 234 | 103 | 102 |
| v9.1 rejection guard, first nonoverlapping original-source sample | 1000 | 390 | 386 |
| v9.2 rejection guard, second disjoint original-source sample | 1000 | 370 | 369 |

These different populations are not a version-comparison learning curve. Neither1000-case audit has a complete neural fallback evaluation, so conditional agreement must not be called full accuracy. All samples come from the same formal-fallacy generator family; public pretraining contamination is unknown.

The initial error treated **Nothing** as a named individual. v9.1 adds abstention for quantifier/pronoun entity words and was frozen before the first1000 audit. That audit exposed three real errors that absorbed **however** into a distinct predicate. v9.2 adds rejection for discourse/modal words inside predicates and was frozen before the second1000 sample. A residual error splits the brand **Bumble and bumble shampoo** at its lexical and, producing the wrong negation scope. No further parser repair was selected from that second confirmation set.

One first1000 source disagreement appears to be a key error: original index3231 labels an argument valid although T=false,M=true,S=false,P=false satisfies its premises NOT T=>M and (NOT S AND NOT M)=>NOT M while falsifying (NOT P AND NOT S)=>T. A separate Boolean audit verifies this countermodel. All official counts retain the original valid label; no source keys were silently corrected.

Both guards only add abstention. Known remaining lexical counterexamples preclude treating accepted English as a universal correctness certificate. Explicit formal inputs avoid this translation step, but the implementation is research-grade, not a security-hardened service.

## Cost and controls

The30 accepted cases could skip generation. Observed ordinary480 CPU phase mean126.47s. Recorded-policy replay, skipping those30 model calls and adding local symbolic time, estimates68.55s mean, a45.79% reduction. Symbolic processing averages0.33ms across64 checks. Neural and symbolic timing environments differ; this is accounting over recorded phases, **not an integrated production benchmark**. All64 neural baselines actually ran for the matched comparison. They generated10506 tokens; counterfactual fallback-only generation would total5680. No global Jev speed claim.

The compiler was manually developed on16 previously used formal-fallacy examples plus the source task README. Frozen implementation hashes precede new evaluations. The234 pool excludes those16; the64 subset is seeded before baseline inference. First1000 excludes all250 BBH formal rows; second1000 also excludes first1000. Exact indices and source bytes are retained. These are specialized parser evaluations, not arbitrary-language reasoning or trained-model generalization.

## Execution and verification

All16 baseline workers completed. Retained neural work:64 ordinary480 generations,10506 tokens, one categorical fallback, plus16 native anchors and16 projection fixtures. All weight hashes agree; anchors and full/restricted fixture logits match exactly. Independent reduction reproduces64 neural labels and192 labels across three recorded policies. The fallback probability arithmetic agrees within8.46e-17. All32 accepted interpretations in the64 sample pass separate proof arithmetic.

Current55 unit tests include300 random theories independently compared against exhaustive finite-domain models. A mutation audit checks1000 valid proof fixtures and rejects4000 mutated source/formula/certificate fixtures. These are implementation tests, not a machine-checked proof of the whole program or of unrestricted semantic fidelity.

Release: Witness-v9-Completed.zip, SHA256 **075480dadfb593e80e9d8d0a74418671c3e3cffa9db5d6af2a47dfd26ab8c464**. It contains full implementation, formal/experimental-English CLI, all wrong answers and abstentions, raw baseline traces, original-source bytes, locks and reproduction code. Replay regenerates13 numerical result files with identical predictions/proofs/selection/counts after excluding explicitly listed wall-clock fields. Offline neural-record replay is not a second neural benchmark.161 file hashes are included in the manifest.

## Provenance

Branch research/witness-v9-20260922; main unchanged.
Protocol commit8ded213b013522420e65919ab84eeedb7db9ca64.
Baseline run https://github.com/Jaksenc/parameter-golf/actions/runs/35736267292 ; artifact10698253726 SHA256 a56fd2117496e654e8c734ca9001c92f2fc6b672afb1d88dff4b9d0dda0acafe.
Source-only audit run35736714324, artifact10697182930.
Guard locks f1b8efabb38cc8f29bbc59aff59d0b91bfe38506 and9e5d9fb66df13d8ef928117a0afc7599985652ef.
Model Qwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
JevBench7128f5cf445ca41e4da1bdc7c84f97926250724d; BBH9ee07bd481feebf959a6b59d61ea57bdcf30964d.
Original BIG-bench task blobfca6d6d6a232084cc31ebd95bda332571cbf0bf5. Source authors Gregor Betz, Christian Voigt, Kyle Richardson; canaries and licences retained. No benchmark weight training, Jev API calls or official submission.

Prior art: Logic-LM(arXiv2305.12295), Faithful Chain-of-Thought(arXiv2301.13379), and classical monadic logic. The contribution within this project is executable selective solving with independently checked formal witnesses and aggressive translation auditing, not a literature-wide invention. The next technical target is semantic coverage without meaning changes; another same-model endorsement would not resolve the shared-parser trust boundary.
