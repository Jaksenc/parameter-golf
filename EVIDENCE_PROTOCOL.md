# Evidence v4 — protocol fixed before new model outcomes

Primary: evidence_exact, a source-linked expression compiler over frozen Qwen3.5-4B followed by a bounded exact interpreter, falling back to that input's newly measured native direct answer when compilation is rejected. This is an end-to-end natural-language binding experiment, not a trained backbone or recovered Jev/RLCD. Anchor v3's supplied-table executor is not substituted for a language binder.

Population: every one of the 231 public JevBench tasks at the already-pinned v1 snapshot, plus 64 new generated English problems (16 ledger, schedule, threshold, pointer each; seed 927431). All public benchmark labels are withheld from inference. Fresh templates were authored in this experiment; they are not external natural-language benchmarks. Benchmark is reused diagnostic data. Full 534-decision composite remains unavailable.

Arms, all reported: native direct categorical readout; short explicit reasoning with FINAL label; compiler CLAIM (the model's predicted result of its own emitted program); exact execution of that identical program. Every malformed/missing label, unsupported expression or failed source contract returns the native answer, never omission or oracle repair. Compiler and reasoning each get max 160 greedy output tokens; actual token counts and latency reported. Compiler prompts are longer, so equal cap is not equal FLOPs. No inference confidence threshold is tuned and no benchmark-based arm promotion.

Source contract: numeric operands addressed through a deterministic source-number registry; copied strings must occur in evidence or labels; expression must execute a state reference and an allowed operation. AST is evaluated by a custom whitelist interpreter, not Python eval/exec. Source presence is not semantic entailment: wrong relation, negation, actor or unit bindings can pass. It is not a truth or security guarantee against arbitrary adversarial resource exhaustion.

Numeric roots map only to unique purely numeric option descriptions; Boolean roots only to explicit yes/no or true/false pairs; string roots to exact labels or uniquely matching descriptions. No task-ID lookup, benchmark-specific solver, answer access or forced repair. Untested or uncalibrated probabilities are not invented for generated/executed labels.

A four-case disjoint pilot (seed 928430) checks runtime and generation compatibility. It is not used for prompt or parameter selection. The full inference jobs may start after pilot infrastructure succeeds regardless of pilot quality. Any infrastructure fixes are separately recorded; no deletion of failed attempts. Every new native result is retained. No main-branch modification, paid API calls, Jev API queries, or official submission.

Comparison: same-program CLAIM versus exact result isolates execution changes conditional on the emitted program, with a separately reported intersection of parse-valid/claim-valid cases. Full hybrid arms measure the practical fallback behavior. Primary paired uncertainty uses source-group bootstrap for JevBench and item bootstrap within the finite generated set; no claim of general natural-language confidence intervals.

Frozen source SHA256: a0184e91a1aec54ec1ca200313e1c9ed51c6e456056f3b9ab0441a370077ae31.
Fresh inputs/labels SHA256: 6484c6587d5a95331d8fdc60af1939ffeef3cdef19c5ab8cb67aebf4a8680adc.
Label-free full input population hash: be13a37b76c9eb545685f16943fe45ac32ce63fb196314d9769e1ae4fbf30ffb.

Thirty interpreter/generator tests passed locally before inference. References: PAL arXiv:2211.10435; execution-guided semantic parsing arXiv:1807.03100; ambiguous semantic parsing arXiv:2306.00824. This project's source-number operands and same-compiled-program control are engineering extensions, not a literature-wide novelty claim.
