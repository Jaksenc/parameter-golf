# Attribution controls, fixed before reconstruction outcomes

The calibrated single-order residual head remains the primary configuration. Add the following diagnostic controls using already planned hidden states/logits and the same independent corrected calibration split:

1. Native readout plus scalar temperature only. The original fitting routine already fits `native_comparison_temperature` independently on calibration inputs. This distinguishes any gain from post-hoc calibration from a gain due to newly learned decision weights.
2. Native readout averaged across normal/reversed candidate orders, both at temperature 1 and at that same native calibration temperature. These distinguish order-averaging gains from residual-head gains.

No model parameters or temperatures are selected against benchmark outputs. No extra backbone inference is required. Evaluate all controls on exactly the same 231 public items using the pinned official scorer. These are secondary comparisons, not candidates for retroactive primary selection.

Report paired changes and group-resampling uncertainty where metadata supports grouping; treat this public repeatedly used corpus as diagnostic rather than a fresh sealed generalization test.
