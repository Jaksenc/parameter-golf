# Evidence v4 — completed and audited, September 21, 2026

The interrupted experiment is now complete: 254 original complete matched records were preserved and only 41 missing inputs were rerun. All 295 inputs (231 public JevBench plus 64 new generated English problems) have all four outputs. Model, source text, prompts, output cap 160, interpreter and fallback policy were unchanged.

## Main result

| Configuration | Public JevBench | Fresh English tasks |
|---|---:|---:|
| Native restricted readout | 182/231 (78.79%) | 31/64 (48.44%) |
| Compiler CLAIM, native fallback on missing label | 147/231 (63.64%) | 36/64 (56.25%) |
| Short explicit reasoning, native fallback on missing FINAL | 194/231 (83.98%) | 63/64 (98.44%) |
| Evidence exact execution, predeclared primary | 180/231 (77.92%) | 40/64 (62.50%) |
| Published Jev 1.13.0, identical public IDs | 200/231 (86.58%) | Not measured |

The new compiler/interpreter mechanism is not promoted. The stronger control is short reasoning: it fixed 20 native errors and introduced 8, +5.19 percentage points on the reused public benchmark. Descriptive 95% paired group-bootstrap interval: [0.45, 9.87] points, 10,000 draws of 195 source groups, no multiplicity correction. The primary fixed 1 and introduced 3: -0.87 points, interval [-2.62, 0.86].

Tier counts (easy/standard/hard): native 48/48, 66/72, 68/111; reasoning 47/48, 71/72, 76/111; primary 48/48, 65/72, 67/111; published Jev 48/48, 71/72, 81/111. Reasoning remains six correct answers behind published Jev. This is no full 534-decision leaderboard score, production speed/cost comparison, or official submission.

## Attribution and failure mechanism

Only 12/231 public programs were accepted; 7 were correct, versus 9 correct native answers on that same accepted subset. On 64 fresh problems, 16 programs were accepted, all ledger tasks, all correct; the compiler's own CLAIM was correct on only 5 of those 16. Thus exact evaluation corrected 11 claims on that narrow family. The other fresh families—schedule, threshold policy and pointer traversal—had zero accepted programs.

The broad public CLAIM-versus-execution score difference is not a pure execution gain because the fallback sets differ. On the same accepted public programs, CLAIM was 9/12 and exact execution 7/12. That intersection prevents crediting a fallback advantage to the interpreter.

A recorded policy failure requires both a receipt and purchase within 30 days. The customer bought 12 days ago without a receipt. The accepted program is n(0) >= n(1), checking 30 >= 12 and allowing the refund. Arithmetic and source references are valid; the receipt obligation is omitted. Other accepted programs ignore JSON shape or trailing whitespace requirements. Source membership does not prove semantic entailment.

Seventeen compiler prompts exceeded the frozen 10,000-input-token cap after source-registry expansion and skipped generation. Public rejection counts include 77 source-index errors, 32 invalid expressions, 27 unsourced numeric literals and 23 expressions without an executed operation. These are limitations of this frozen prompt/DSL, not proof of the model's maximum capability.

Reasoning hit its 160-token cap on 41 public cases; 39 had no valid final label and fell back. A guaranteed typed final readout or changed budget is an untested next hypothesis, not a retroactive score adjustment.

## Compute, verification and scope

Frozen model: Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. No v4 training and no Jev API calls. Anchor v3 neural execution is not used; exact interpretation is the stronger supplied-program control.

Retained experiment: 295 native decisions, 573 actual generated responses and 37,506 generated tokens. The original completion field generations=590 counts slots, including 17 compiler preflight skips; it is not the number of neural generations. The inherited native Runtime no_generation=true field is likewise not descriptive of this explicitly generative experiment.

All 231 native prompt hashes and logits exactly match the earlier baseline. Both pretrained weight shards were checked on 16 original and 16 recovery workers. All 32 full/restricted-readout fixtures and 16 recovery native anchors matched exactly.

44 local tests passed. Reconstructed all 1,180 output labels; pinned official label scoring agreed on all 924 public-arm outcomes. A separate AST value interpreter with separate source-number extraction and label mapping reproduced all 28 accepted-program values and labels. This is not semantic verification or a hardened security proof.

Measured four-thread ARM CPU public medians: native 6.01 seconds, native plus compiler 29.94 seconds, native plus reasoning 69.25 seconds. Excludes model loading, serving/network and interpreter time. Equal 160-output-token caps are not equal compute because input lengths and actual output lengths differ. No probabilities, calibration metrics or composite scores are invented for label-only outputs.

Original run: https://github.com/Jaksenc/parameter-golf/actions/runs/35631351186
Recovery run: https://github.com/Jaksenc/parameter-golf/actions/runs/35637799747
Recovery artifact: 10657251769, SHA256 43f2b0f5c0bac60b8d208b0a965815e1d9f50f8a0a1c0d84f1af5ba069c611e3.
Published Jev comparison: benchmark commit 7128f5cf445ca41e4da1bdc7c84f97926250724d, per-task Git blob f9993fc449faf54ad61729f692504faf33d0e7ba; all 231 IDs matched. Archived outcomes, not a fresh Jev run.

PAL (arXiv:2211.10435) and execution-guided semantic parsing (arXiv:1807.03100) are prior art. Source-addressed operands and same-program attribution are project-specific engineering, not a claim of literature-wide novelty. The public benchmark is reused, the fresh set has four authored templates, and no deployment or main-branch change is claimed. Full source, raw records, program audit, tests and reproduction instructions are in the conversation release.
