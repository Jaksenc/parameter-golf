# Duplex v4 controlled transfer execution

This imports the previously delivered v4 experiment. Data, objectives, 400-update schedule, seeds 41/73/101, and development-only selection are unchanged. All 63 original tests passed before execution. Data is original generated English, not independently human-reviewed examples.

The workflow first extracts model features from the fixed 384 training and 96 development inputs, then fits all nine candidates. All transfer inputs are withheld from the trainer and checkpoint selection. A separate model-feature stage and full-network replay evaluate 288 transfer inputs. JevBench public evaluation is a further, separate stage; no JevBench input or label is used in fitting, calibration, checkpoint selection or prompt tuning.

The model is pinned Qwen/Qwen3.5-4B, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Experimental rank8 terminal adapters are not production promoted automatically. This branch does not modify main.

Prompt framing is derived from SemIf, MIT; include its license before redistributing source. Base model weights retain their own license and are not stored here.
