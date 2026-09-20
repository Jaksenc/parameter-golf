# Public decision-probe fusion audit — 2026-09-20

## Evidence and scope

Completed arithmetic audit: https://github.com/Jaksenc/parameter-golf/actions/runs/35522867676/job/106109871021 . Audit code and workflow commit: `8002535aa497f703d2b75d588cf61b743fa1236c`.

The audit verified 1,200 genuine pretrained prediction records (400 inputs × 3 models), including input identity, checkpoint revisions, completed jobs, partition completeness, and recomputed softmax/argmax. Maximum probability discrepancy was 0.0. It loaded no pretrained weights and performed no new model inference. Temperatures, static pooling weights, and a tiny confidence-feature gate were fitted only on the declared calibration/fitting cases; pretrained model parameters were not trained.

The original 4B x86 job stopped after 27 calibration cases and is excluded. Four successful ARM shards from run `35520447897` supplied all 400 Qwen4B predictions with the same native scoring functions. GLiClass and Qwen0.8B came from successful jobs in run `35518735165`.

Data hash: `4309b5304f50e5f740a334f3bd8d7f8ac6a4cd9761a8822caddb843575131f0d`.
Inference-source SHA256: `b7267efe6fb8431b303490bafba0986bf899bb21c3b4677f90d36ff1de4695ef`.

## Results

Primary test: 144 controlled cases in six correlated noun/source blocks. Public regression: 64 previously inspected examples, 16 each from SNLI, PAWS, BoolQ, and derived CLINC domain routing; these are not full benchmark scores. NLL means mean negative log probability of the correct candidate; lower is better.

| System | Primary correct / 144 | Primary NLL | Public regression correct / 64 |
|---|---:|---:|---:|
| Qwen3.5-4B, raw | 122 | 0.4198543713 | 51 |
| GLiClass Instruct Large v1.0, raw | 74 | 1.6048441248 | 31 |
| Qwen3.5-0.8B, raw | 64 | 1.3785359384 | 27 |
| Uniform raw probability pool | 76 | 0.7817199014 | 49 |
| Uniform temperature-fitted pool | 104 | 0.7220480289 | 51 |
| Fit-selected 10%-increment weight grid | 122 | 0.7257247671 | 51 |
| Continuous fitted static pool | 122 | 0.5795283333 | 51 |
| Confidence-feature gate | 122 | 0.5667334532 | 51 |
| Fit-selected small-to-large cascade | 122 | 0.6475764186 | 48 |

No tested pool or gate improved overall primary accuracy or overall regression accuracy over Qwen4B. The confidence-feature gate made exactly the same choices as Qwen4B on all 144 primary cases. Its small fit loss improvement did not become an accuracy gain on the test.

The grid selected 100% Qwen4B. The continuous pool's test weights were approximately 98.91% Qwen4B, 0.62% GLiClass, and 0.47% small Qwen. The gate's mean test weights were approximately 91.63%, 5.78%, and 2.58%, respectively. Calibration temperatures were Qwen4B 0.25, GLiClass 2.3784142300054425, and small Qwen 1.189207115002721. Qwen4B's temperature fit worsened primary NLL from 0.420 to 0.726 and regression NLL from 0.379 to 0.676. Temperature fitting is not a guarantee of calibration under a changed task distribution.

## Diagnostic failures

Qwen4B: atomic rules 46/48, AND/NAND 48/48, XOR/XNOR 28/48. Complementary-rule pairs both correct: 50/72; evidence-flip pairs both correct: 56/96. Reversing candidate order changed the semantic answer in 6/48 atomic-case pairs. Adding the irrelevant note changed one answer in 48. The gate reproduced these same primary failures and option-order changes.

GLiClass: rule pairs 2/72 and evidence pairs 4/96. Small Qwen: 0/72 and 0/96. Stability of a largely constant answer is not semantic correctness.

A gold-aware selector among the three existing top choices could obtain 127/144 primary and 58/64 regression. This is an oracle diagnostic, NOT a deployed selector or a general upper bound on other fusion methods. GLiClass recovered 5 of Qwen4B's 22 primary errors; small Qwen recovered 2, without adding further primary coverage beyond GLiClass.

The cascade threshold was 0.8, chosen on fitting data. It still called Qwen4B on 133/144 primary cases, saving 11 large-model calls while adding a small-model call to every case. It fell to 48/64 regression and 43/48 irrelevant-note cases, versus Qwen4B's 51/64 and 47/48. No end-to-end cascade latency claim was measured.

## Limitations and decision

Calibration and fitting used 48 cases each and only atomic rules. The primary test introduced AND and XOR families. Candidate rotation is confounded with noun within a source block, and only two of six candidate action meanings can be correct. The primary cases are a narrow literal grammar, not production UI requests. Public examples may have pretraining overlap and were already inspected. This is a text-only restricted-candidate forward-pass probe, not free-form generation, native macOS inference, or an end-to-end application benchmark. Separate x86/ARM VMs cannot establish a resource-matched speed advantage.

Keep raw Qwen4B as the current reference, not as a certified deployment model. Do not promote this three-model blend. Retire these exposed cases to regression tests. The next useful comparison is fresh real-language source groups, a matched-compute single-model control, and semantic extraction followed by deterministic rule execution. No Jev, paid model API, GPU rental, private source/corpus publication, artifact upload, cache upload, or default-branch merge was used in this audit.
