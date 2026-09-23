# Learn v16 recovery: matched checkpoint audit, September 23, 2026

The original 64-update primary experiment did not finish. All six runs were cancelled before final evaluation. Their logs retain 60,60,61,61,62,62 completed updates, but each archive contains weights only at steps 1,16,32. There is no optimizer state, RNG state, or final64 checkpoint. Loading step32 weights into a new AdamW optimizer is NOT exact resumption; do not claim it is. The later unsaved updates cannot be recovered from their scalar loss logs.

This continuation performs an explicitly intermediate, availability-defined diagnostic. Evaluate the largest persisted checkpoint shared by every seed and arm: step32. The choice is made from saved-file availability before any post-training predictions are observed, not by held-out score. Original primary64 remains incomplete. No optimizer updates, new training policy, or best-step/seed selection in this audit.

Use the exact frozen source inputs, prompts, output head, code labels, low-rank factor implementation, and the 72 original evaluation indices: 56 held-out variants of10 worlds, plus16 labeled training probes. Six models: seeds16101,16102,16103, each ordinary soft-target supervision and lambda0.25 verified relation supervision. At step32 both arms of each seed have the same32 edge presentations,64 input presentations, initial parameters, optimizer settings, and per-step computation. All other recorded checkpoints/logs remain in the release. Report the number of unique fit examples actually seen at this endpoint.

Unmodified comparison uses the preserved72 native baseline observations, checked for equality across all six archives. Each new worker verifies fixed pretrained weight hashes, the existing readout fixture, actual zero-adapter equivalence, expected checkpoint SHA256, input/prompt/code-map consistency, and base restoration. Save every new output, including wrong predictions, with input hash and FP32 logits/probabilities. No oracle labels appear in inference prompts. No generation, calibration fitting, public JevBench scoring, or paid resources.

Endpoints are held-out target cross entropy, squared-vector error, TVD and modal-category agreement, separately for event and mode questions; report both supervised and relational arms against baseline and each other for EVERY seed. Source targets are exact rational laws. Aggregate paired comparisons are descriptive over three seeds and ten generated source worlds; resample whole worlds and paired seeds, never count repeated seeds/variants as independent questions. Report fit probes separately, plus relation residuals and data/representation limitations. Do not label improvement as established on new grammars or general English.

Parent artifact: learn-v16-evidence, run35808253851, artifact10730888996, archive SHA2566f271f3caafc78b35505a6b58abaccbee43fed07f9cfed418ad224696fd8a359. Original source learn_v16.py SHA25688d3ce9ebae554adbb1728b45254bfe7a87043a26d1264fe0fad55077054b587. Full results absent from original artifact; its completion.json explicitly says partial.

Step32 checkpoint hashes:
16101/supervised b7b2a1da8015abb2d205bed06878d335de7111e55ec212b6e00899a9742e6b4f
16101/relational 4178b503143f4c0e8c64e30717021d2733a9b1f98071998205d45c48d71a452c
16102/supervised 2fb43e01029bc540e72ffc6ec7896b10e515c2448177b3db656f3d0f90ea39ab
16102/relational 766a05cd00e2a34f33a46d72a941f309535730a87a2c88adb39bb6b3facd228c
16103/supervised 4f2f731ab7a715f5a70b4b1602cb93b3c675ccdefbfac73859ca767516e37368
16103/relational a7491cd19c1818a6fccfd7aa74a61105c35b1496f872d46d92d4e5d412c7574e

The interrupted local structured-representation experiment is mentioned in earlier progress but its source/results are not present in this artifact. Do not present those unrecovered numbers as reproduced evidence or merge them with this pretrained-language comparison. The scope of this recovery is actual saved Qwen adapters.

Correct resumable training should later save model factors, optimizer state, RNG state, schedule position, source/data hashes, and progress atomically. That infrastructure repair does not retroactively reconstruct the missing state. Main stays unchanged, no official submission, no Jev endpoint calls or claimed new composite.
