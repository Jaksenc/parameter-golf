# Pre-outcome split correction — reconstruction v1

This correction was fixed while run 35554698513 was extracting features. No reconstruction predictions or benchmark outcomes had been inspected.

The synthetic generator uses four Boolean states and repeats some threshold values. Source IDs were distinct but some canonical inputs overlapped between the training/development/calibration partitions. The original automatic fit is therefore diagnostic, not the final primary result.

## Corrected primary protocol

Deduplicate canonical JSON `{state, question, labels}`, excluding the record ID. Iterate partitions in order training, development, calibration, and retain only the first input occurrence. Require identical targets for duplicates; abort on conflicts. Retain 301 unique training inputs, 73 development inputs, and 73 calibration inputs (447 total). Two candidate orderings remain augmentations, not independent observations.

The removed IDs are:

synthetic-0005, synthetic-0011, synthetic-0014, synthetic-0023, synthetic-0026, synthetic-0027, synthetic-0029, synthetic-0032, synthetic-0035, synthetic-0038, synthetic-0041, synthetic-0044, synthetic-0047, synthetic-0050, synthetic-0053, synthetic-0056, synthetic-0059, synthetic-0060, synthetic-0062, synthetic-0065, synthetic-0068, synthetic-0069, synthetic-0071, synthetic-0074, synthetic-0075, synthetic-0077, synthetic-0080, synthetic-0083, synthetic-0086, synthetic-0089, synthetic-0092, synthetic-0093, synthetic-0095.

All model weights, extraction prompts, benchmark inputs, native logits, hidden states, training loss, learning rate, regularization, calibration grid, checkpoint selection and official scoring remain unchanged. Refit only the tiny decision head and temperature from retained training/development/calibration features. Persist its checkpoint before evaluating the same 231 public JevBench inputs. The corrected calibrated single-order head remains the primary configuration; normal/reversed averaging remains secondary.

No further modifications are to be chosen based on benchmark outcomes. The existing public benchmark has been used in prior experiments; this is not a fresh sealed test.
