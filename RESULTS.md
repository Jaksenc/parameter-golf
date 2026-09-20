# Completed public decision baseline — September 20, 2026

Two unchanged pretrained checkpoints were evaluated on the same 128 requests using standard public GitHub-hosted CPU runners. No paid API, GPU, Actions artifact/cache upload, private project data, or model training was used. This is a small fixed-prompt diagnostic, not a model release, full benchmark, or a Parameter Golf competition submission.

| Task | n | GLiClass-large-v3.0 | Qwen3.5-4B |
|---|---:|---:|---:|
| Synthetic rule/evidence choices | 64 | 32 | 56 |
| SNLI inference | 16 | 7 | 14 |
| PAWS paraphrase | 16 | 6 | 15 |
| BoolQ yes/no | 16 | 12 | 13 |
| CLINC-derived domain routing | 16 | 10 | 9 |
| All cases, descriptive only | 128 | 67 | 107 |
| Both rule-change endpoints correct | 32 pairs | 0 | 24 |
| Both evidence-change endpoints correct | 32 pairs | 0 | 24 |

Counts are correct decisions, not percentages. Qwen's one tie counts as unsuccessful. Pairs reuse the synthetic cases; they are not additional independent observations. The 64 synthetic cases derive from only four noun/source blocks, and candidate order varies between pair endpoints. Public-source pretraining overlap is unknown. Task wrappers and descriptions were not independently optimized for each model. No independently adjudicated real-world performance, calibration, general superiority, Jev comparison, or model-level speedup is claimed.

## Actual execution records

- Qwen successful job: https://github.com/Jaksenc/parameter-golf/actions/runs/35516884391/job/106094227284
- GLiClass repaired successful job: https://github.com/Jaksenc/parameter-golf/actions/runs/35517134423/job/106094856168
- Independent arithmetic audit: https://github.com/Jaksenc/parameter-golf/actions/runs/35517378478/job/106095489708

The initial matrix's GLiClass job failed because the wrapper does not expose `prepare_input`. The repair workflow changes only `pipeline.prepare_input` to `pipeline.pipe.prepare_input`. No inputs or model weights changed. The enclosing first workflow remains marked failed although its Qwen job succeeded.

The separate audit recomputed all 256 probability vectors and selected answers, all task metrics, and both paired metrics from actual completed job logs. Maximum probability discrepancy: 2.220446049250313e-16. Input hashes and target vectors matched across models. This checks arithmetic and population agreement, not independent human correctness labels.

Input SHA-256: `a433530b7db1bf5fb43e279ea8a0c5c8ace96d2759a260e25e8d601e10e96de2`.

Qwen revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; 4,539,265,536 total loaded parameters including multimodal components; BF16; text-only, thinking disabled, direct answer-letter scoring, no cache. Standard BF16 output-head rounding remains; the tie was not broken in favor of any answer.

GLiClass revision: `e065d1844f913a9aa611cf33623a9538b8aa8841`; 438,668,802 parameters; FP32; official pipeline with complete outcome descriptions.

Both checkpoints loaded with no missing, unexpected, or mismatched keys. Actual dependencies: torch 2.10.0+cpu, transformers 5.17.0, datasets 4.8.5, huggingface_hub 1.32.0; GLiClass source at `40baa67cdea577449bc3f6a251646377b2cb0a6c`.

The inference loops took 183.35 seconds (Qwen) and 138.01 seconds (GLiClass), excluding downloads/loading. These are not a fair speed ratio: separate VMs, potentially different CPUs, different precision and pipelines. No GPU, multi-question batching, or concurrent-serving performance was measured.

## Scope

Public-only helper code and public dataset references were used. The main branch was rechecked unchanged at `0f5145101cc5639ea9e93c39635251ce5922190a`. No private project checkpoint/code/corpus was published. No new weights were produced. All jobs are completed.

GitHub's standard public runners are free under https://docs.github.com/en/billing/concepts/product-billing/github-actions . Logs do not consume artifact storage allowance. No account invoice audit was performed; cost scope is established by the selected free runner class and absence of paid storage/services. This is not a claim of free large-scale/private GPU training.
