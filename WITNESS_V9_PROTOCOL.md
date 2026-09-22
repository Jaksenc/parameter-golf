# Witness v9 — frozen protocol, September 22, 2026

Primary: witness_checked_else_standard480. A closed-grammar recognizer compiles an entire explicit entailment argument to equality-free monadic classical first-order logic. It enumerates unary types to find a countermodel to premises AND NOT conclusion, or an exhaustive contradiction certificate. A separately written checker verifies every certificate using a different arithmetic implementation. Unsupported syntax or conflicting alternative and/or scopes causes abstention, preserving the exact ordinary480 baseline answer. No benchmark-ID lookup, target-dependent routing, confidence threshold, learned verifier, model training, Jev query or majority voting.

The compiler is deliberately not a general natural-language semantic verifier. It recognizes only fixed argument wrappers and an explicitly documented controlled grammar. Predicate phrases are uninterpreted unary properties; there is no implicit world knowledge, existential import, identity or binary variable relation. Complete source coverage is conditional on this grammar. The checker independently validates logic arithmetic but trusts/reuses the grammar for source binding. It rejects altered premises or source hashes. Ambiguous unmarked conjunction/disjunction scopes are enumerated; a verdict is emitted only if every supported interpretation agrees. Interpretations outside the grammar are not certified.

Development: 16 already evaluated formal-fallacy questions from v7/v8, plus generated unit fixtures. All 16 development cases are supported and agree with source targets; this is not a held-out result. The source task README, not unseen evaluation examples, was consulted for semantics. Predicate normalization, wrappers and constructions were developed on those 16 items. No claim of universal language coverage or literature-wide invention.

Evaluation, fixed before opening new outcomes: all 231 reused public JevBench cases against archived ordinary480 (202 correct), all 234 unused formal-fallacy BBH source rows for supported-coverage/conditional-accuracy diagnostics, and a seeded random 64-row subset for a matched ordinary480 baseline and hybrid comparison. Seed 920260922. Exclusion uses normalized input stems against all v6/v7/v8 previously used inputs. Source is the pinned formal_fallacies.json from BBH commit9ee07bd481feebf959a6b59d61ea57bdcf30964d. These are new-to-project public template-generated examples, not independent grammar designs or pretraining-contamination-free holdouts. The eight other-domain v8 families remain untouched; no claim that a specialized formal-logic result implies broad Jev superiority.

Counterfactual invalidity witnesses and valid exhaustive certificates must be independently checked on every accepted evaluation. Report every unsupported input and every accepted wrong verdict, not only proof successes. Compare the hybrid versus ordinary480 on all64 sampled items; unsupported items retain the measured baseline. Report simple item-bootstrap intervals as conditional and warn of shared generator/template dependence. Report logical engine latency separately from GPU/CPU neural latency. The policy would skip model calls on accepted items; an offline paired study computes the baseline for all64 to support attribution. No fabricated probability=1 for certified labels.

The ordinary baseline uses the same pinned Qwen3.5-4B, 480-token greedy generation, strict FINAL parser and unchanged categorical fallback as v8. It receives only id/state/question/labels, not certificates or targets. Public baseline is archived and not regenerated. Only64 new neural baseline outputs are required. All worker model hashes, full/restricted output fixture and native anchors are verified. Main unchanged; public free CPU workflow only; no official submission/full534 composite/calibration claim.

Frozen local implementation hashes, before new evaluation outcomes:
logic.py 08adba184151463c527fb70eda94095ae32c2d6a199b9dd05968f961d4582bfd
compiler.py c60cf8de28682ff03b52c4a0a6e336128cf8d26a8793c8c52bd9e3369ea3d03e
checker.py f9242d41ff074201776e05fab237e7930391d5cac3187150f94d594b7660ce1e
adapter.py 8e86336cb65c1589389e470a25fb2d9d1f17958d049157fda4fb544e7e1a587e
select_v9.py dff76d2b432e8ce4c6602851a5e112274312a1a4f5dda0abdceda71b4562fd9d
tests/test_witness.py 09f59d10a8032505a19ae7fff1d3fb45daf9180770df13f2afa7c642218988da
41 tests passed, including150 random theories independently compared with exhaustive finite-domain models. All source canaries/licences are retained. Model- and dataset-independent certificates are project-specific engineering; prior art includes Logic-LM(arXiv2305.12295), Faithful CoT(arXiv2301.13379), and BIG-bench Formal Fallacies/Syllogisms with Negation.

64-sample canonical SHA256 d76b02feb42cb8df551ef119c9f724836118978d35a769861893cb02836b8afa
234-pool canonical SHA256 720c5f967eb85c29cd2bb6ba4b7d9561fb82d34f90ba7014004ded16fa0b515c
No new evaluation outcome inspected as of this lock. No post-outcome parser repair may replace the frozen primary. Any such repair is explicitly separate and requires fresh evaluation.
