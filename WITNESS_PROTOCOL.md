# Witness v3 — frozen execution-feedback protocol

Question: does verified execution of a model-proposed intermediate program improve decisions beyond generating the same program without its results?

## Architecture and attribution

Keep the exact Qwen3.5-4B backbone and native FP32 permitted-answer readout from v1. Add a bounded 160-token, non-thinking, greedy program-generation pass. Interpret the proposed dictionary expression using an explicitly whitelisted AST interpreter, never eval/exec or unrestricted Python. Source data are S, rubric Q. Arithmetic uses exact rational values; helpers cover calendar differences, clock arithmetic, successor traversal, bounded affine updates, combinations and reachability. Source numeric literals are checked against supplied inputs or a declared structural/unit whitelist. This checks numeric provenance, not semantic relevance or interpretation correctness.

Three arms, fixed in advance: native; program-only (same generated code, no execution result); Witness (same code plus its executed result). The primary is Witness with separately fitted scalar temperature. If the program is empty, rejected or cannot fit the readout context, fall back to the unmodified native result; preserve all failures and their costs. No output-text answer parsing: final probabilities remain the model's native logits restricted to allowed slots. Execution success never becomes probability 1 or a claim of semantic correctness. Compare primary against both native and program-only; all arms reported.

This is new inference computation, NOT trained new backbone weights, recovered Jev, or reconstructed RLCD. Generation and runtime expenses are included. PAL, Program of Thoughts, and Faithful CoT are prior art. Project novelty is the source-checked, fail-closed execution-feedback experiment with controlled program-only attribution and typed probabilistic readout, not a first-ever claim.

## Population

503 inputs: all 231 public JevBench decisions, all 128 previous Orbit stress tasks (now explicitly reused), 48 new calibration examples, and 96 new held-out examples spanning cashflow, calendar, route traversal, recurrence, sampling and threshold logic. Seeds: 930571 calibration; 2819043 fresh. New cases use independently computed reference answers; calibration and fresh inputs are distinct, with longer traversal/update chains in fresh. Some surfaces are prose and some structured data. New fresh examples are never used for method selection or calibration.

Canonical input hash: e1ba7ebeb5f1ddeb9b1c578c9b3834ed55fd114e602a5fb6ff09b4fbf2101e2b.
Calibration labeled-data hash: 9856b3f2adfdff8e2480c8d1a19dad0d29b8412771d221f81d9ac85a83ede378.
Fresh labeled-data hash: b38f13c264f0a8f90a6d6630f2e4c6052efb186f459be379e3c64ef82d82303a.

Each arm's temperature is fitted on the new 48 calibration inputs using the fixed 121-point log grid [0.25,4]. Freeze fitted state before inspecting evaluation outcomes. No task-dependent mixing weights or retrospective primary substitution. Use original official JevBench scoring; report accuracy, NLL, Brier, ECE, paired source-group bootstrap and execution acceptance/failure rates. Fresh intervals are conditional on generated templates. No full hidden-set composite or production cost claim.

## Execution

Free standard public GitHub ARM CPU runners only, separate research branch; main unchanged. No paid API, GPU rental, Jev extraction, private-data upload, deployment or leaderboard submission. The separate Jaksenc/decision-0 repository still returned 404; this branch is the existing compute host, not a project merger into main. Publish a self-contained source/result archive independently.

All 16 workers verify model hashes, restricted/full-readout fixture and an archived native anchor. 503 fresh native readouts provide within-run controls. Preserve raw programs, execution certificates, prompts hashes, per-arm logits, failures and timing. Never omit a failed task. Generation-boundary or dependency repairs may be made only as documented execution fixes; a scientific change after outcomes requires a separate experiment.

Hypothesis may fail: a correct interpreter can execute a wrong formalization, and post-execution neural scoring can ignore correct evidence. These are distinct error sources to diagnose, not assumptions of success.
