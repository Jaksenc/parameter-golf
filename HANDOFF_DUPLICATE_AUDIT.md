# Handoff v5 canonical-input audit

A static audit during inference, before any complete benchmark outcome reduction, found that the fresh generator's uniqueness assertion included each input ID. That assertion therefore did not establish unique textual cases.

Exactly two canonical duplicate groups occur across the entire 359-slot evaluation, when equality is defined on {state, question, labels} without ID:

- handoff-new-exception-02-0 and handoff-new-exception-03-0
- handoff-new-exception-02-1 and handoff-new-exception-03-1

Thus the new set has 64 scheduled case slots but **62 unique canonical inputs**, organized as 32 nominal policy pairs but **31 distinct pairs**. The full experiment has 359 scheduled slots and 357 unique canonical inputs. There are no other exact duplicates across the combined population.

The frozen inference, primary handoff64, and all eight arms are unchanged. Do not drop or replace the duplicate records, alter prompts or regenerate a more favorable test. Preserve the nominal full-population metrics for the exact predeclared run, report a canonical-deduplicated sensitivity with the first occurrence retained, and jointly cluster the duplicate exception pair for uncertainty. Show nominal and unique-pair correctness separately. The original fixed analysis is preserved alongside the corrected dependence-aware reduction.

The correction was motivated solely by canonical input equality, not model success or failure. The separate three-case bounded-runtime smoke outputs had already been inspected, and three completed shards were used for structural/numerical/prefix checks; no full benchmark scores were available when this issue was identified. Neither the nominal nor deduplicated authored set is a broad external language benchmark. No inference policy or model weights change as a result of this audit.
