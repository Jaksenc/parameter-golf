# Contrast v7 recovery audit — September 21, 2026

The original inference run 35651891496 saved 285 complete matched records. Shard 5 finished inference but lost its artifact upload, leaving nine unavailable records. Shard 25 saved nine of ten before cancellation. Recovery run 35660386499 reruns only the ten inputs lacking complete saved records. The original 285 rows are preserved exactly, including all incorrect outputs. Recovery selection uses availability and input hashes, not expected answers, confidence, or correctness.

The scientific source is unchanged: contrast_v7.py SHA256 52771fdb5d8c7e5315c6f49ac87b725ba1244c0fe5ba4f5a0674ec4de3778d0b. Same model revision, prompts, 320/480/160 ceilings, six arms, categorical fallbacks and order-consensus switching policy. Recovery source SHA256 3ab640ed13bae34df92e9fdd35035855441c1b6a8b445b12eb54248ad43e164f. The recovered-data aggregate must contain all 295 unique planned IDs before any full-population score is reported.

The earlier local analysis implementation was not included in the surviving source/artifact bundle. Its historical hash and test receipts remain in the archive, but are not represented as a currently reproduced implementation. A new independent offline reducer implements the frozen policy and scoring, plus independent source-selection, native-anchor, probability and interface checks. No model policy or primary selection was modified by this replacement. Current release preflight: 41 unit tests passed, including 10,000 parser-fuzz fixtures inside one test. These are distinct from the earlier claimed 58-test suite, which has not been recovered/re-run.

Current offline source hashes, fixed before reducing full-population outcomes:

- audit_and_score.py: a9d36f844234ac1c665ff65885ce467379b27242e21ff696a2e7823173170140
- audit_provenance.py: 61b5c3dc72e593979881994cca9c9478cc6c018a6b94d19a3af37804656ee77a
- test_contrast.py: 43bcdfd31f761831b76bcb1b9ab6effaa4a6ba699111c681b4181f71495f832b
- decide_contrast.py: d234f421f9477bc3f9579659443c924fe21cf1190ac9c5189d88e9a66b82caf8

The standalone runtime entry point is the previously tested source, not a rewritten policy. The three predeclared live fixtures and the one post-outcome disagreement-coverage fixture are retained separately. None is an independent performance estimate. Source-data matching verifies provenance and sample selection, not the truth of all original answer keys or absence of pretrained-model contamination.
