# Causal v17: early trajectory audit completed September 23, 2026

All 12 predeclared v16 step1/16 checkpoint evaluations completed: three seeds, two prior objectives, two saved steps, 72 records each (864 new predictions). The unchanged and step32 vectors are retained from the old matched audit. No optimizer updates were made by this trajectory audit and no old checkpoint is promoted. The old64-update primary remains incomplete.

Held-out event outcomes: 28 questions drawn from ten old worlds, transformed variants and three seeds dependent. Metrics are averaged across seeds, not an ensemble. These are diagnostic data already used in the project, not fresh confirmation.

| Step | Supervised CE | Supervised squared error | Supervised modal accuracy | Relational CE | Relational squared error | Relational modal accuracy |
|---|---:|---:|---:|---:|---:|---:|
|0|1.254269|0.075998|32.14%|1.254269|0.075998|32.14%|
|1|1.255329|0.076418|32.14%|1.253714|0.075569|33.33%|
|16|1.387100|0.154451|50.00%|1.383126|0.150654|48.81%|
|32|1.447954|0.158784|38.10%|1.411101|0.150276|35.71%|

There is little event-loss movement after the first saved update. By step16 squared error is about twice baseline while leading-category accuracy has increased. Worsening is not confined to the final32 checkpoint. Only these saved timepoints exist, so the audit cannot locate the exact intervening update or establish a cause. The new objective-by-step-size experiment and analysis were fixed before inspecting these outputs.

Independent reduction replays all input and prompt mappings, recomputes stored probabilities, verifies all checkpoint receipts and restoration/reload errors, and preserves all seeds and both objectives. Timing fields differ between old baseline workers and are not mistaken for changed predictions. Full results include fit-pool and modal-question curves separately.

Run35898948971; artifact10769456895, SHA256db226c8f2f577ccf32c36f6bd46f52f3f0eed89af8f50dcd9e9b7cfc06466326. Scientific wrapper commitcfcaa3f50ca3405714b7945fcf9b74f27ccf28b3; workflow7d9bbf3ffa77dea841d23fcf7456f971ae07669d. New factorial run35900447533 is separate. No new JevBench result, calibration fitting or general-model superiority is claimed.