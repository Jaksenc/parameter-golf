# Pre-outcome static audit — Evidence v4

Before inspecting any pilot or new evaluation model output, static code review found that the original numeric regex omitted values followed by a sentence-ending period. The generated access-rule family also used necessary-condition-only wording ('only when') for a necessary-and-sufficient oracle. These are corrected by run_evidence_v4.py: numbers may end before punctuation, and the generated policy now says 'if and only if'. No new outcome was used to choose either repair.

The original run 35631052486 is superseded and cancelled through the replacement workflow. Its artifacts are not included in primary results. Repaired execution is run 35631351186. Original source is retained unchanged for audit. All four arms, token cap 160, seeds, oracle labels, greedy decoding, fallback policy and item counts remain fixed. Fresh text and its input hashes change only for the iff repair.

Base source SHA256: a0184e91a1aec54ec1ca200313e1c9ed51c6e456056f3b9ab0441a370077ae31
Repair SHA256: cf62a0790fe714a3317ae8a8e4710866eed87a4172fb27e0908a880f2f026778
Effective combined source hash: db7e73bcea3b497213bfe252ad4f154d8f110f321bee23f9fc17f82518b89851
Repaired fresh set hash: 2b22556cfa24de4ebc78f08f8ed1c71bb5461460b29830e22b0b580f5c912f3c
Repaired label-free full input hash: 56991da6d70f3a78c5aa9165ec2b15133c05bffde69c78dd6842c1e2bde837f3

As of 2026-09-21T17:21:26.799344+00:00, all 34 local interpreter/generator tests passed and new outputs remained uninspected. Reduction script analyze.py SHA256: 7d865e2796337474ce3a32b4df176d6c6a65dc6ec8f67440f6ba894d3b465bbc. It checks every input hash, complete shards, frozen source, raw-expression replay, and official label-only scoring. It does not invent distributions for label-only arms. Same-program replay is not represented as an independent interpreter implementation.
