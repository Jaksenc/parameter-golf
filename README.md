# Public pretrained decision baseline

Isolated research branch for bounded, zero-paid-service inference tests. This is not a Parameter Golf competition submission or a Decision-0 model release. The main branch is not changed.

Uses only public pretrained checkpoints, public datasets, and new synthetic examples authored here. No private project code, data, credentials, or trained weights are uploaded.

Runs standard public GitHub-hosted CPU runners only. No paid GPUs, API services, Actions artifact uploads, cache uploads, remote storage, or scheduled jobs. Results are printed to ordinary workflow logs. One run is bounded by its timeout. No GPU performance or model-training claims.

The small fixed-prompt evaluation is diagnostic, not a full public benchmark, independently reviewed dataset, or proof of general superiority. Public evaluation examples may overlap upstream pretraining. Rules-only and evidence-only controls on the synthetic task each have a 50% ceiling; correlated pairs are not independent observations.

To reproduce: install CPU PyTorch, Transformers, datasets, SentencePiece and the pinned GLiClass source in the workflow; run `python benchmark.py --model gliclass` or `python benchmark.py --model qwen`. Model and dataset revisions and dependency versions are recorded at runtime.
