# Measure v14 — completed September 22, 2026

## Result and decision

The fixed numerical-mass primary reduces source-only event mean TVD from **0.240996 to0.185367** (23.08% relative), but raises squared-vector error from0.128531 to0.134931, assigns zero probability to a possible event in five of32 worlds, and takes4.08x the legacy readout's measured phase time. It is **not promoted as a general replacement**.

The narrower positive result is **32/32 supplied event distributions preserved exactly** by numerical output, versus mean TVD0.258179 for legacy codes and0.237753 for semantic codes given the identical correct reference notes. Those notes intentionally provide the solution; this is readout-fidelity evidence, not autonomous calculation or new general-model intelligence.

No weights trained, no temperatures fitted, no Jev API use, no new JevBench score, no full534-decision composite or production-performance claim. All16 inference workers and aggregation completed, with no recovery or duplicate inference.

## Frozen comparison

| Source-only event distribution,32 worlds | Legacy best-answer codes | Distribution-aware codes | Numerical weights, primary |
|---|---:|---:|---:|
| Mean TVD | 0.240996 | 0.214243 | 0.185367 |
| Mean squared-vector error | 0.128531 | 0.100018 | 0.134931 |
| Correct modal category, diagnostic | 20/32 | 20/32 | 18/32 |
| False-zero / infinite strict log-loss cases | 0 | 0 | 5 |
| Clipped cross-entropy, epsilon1e-12 | 1.073880 | 1.031267 | 1.861811 |
| Mean readout seconds | 5.955 | 6.906 | 24.267 |

The primary improves22 worlds and worsens10; paired TVD difference-0.055628 has descriptive95% interval[-0.110875,-0.002446]. Semantic-code difference-0.026753 has interval[-0.054113,-0.000574]. Intervals use10000 paired world resamples within four authored families and are not multiple-comparison-adjusted, independent-grammar or broad-language guarantees. Squared-vector loss equals excess expected multiclass Brier; strict log loss is infinite for positive target mass assigned zero. Clipped metrics do not modify or conceal those output zeros.

## Mechanism and causal limits

Legacy is the unchanged Handoff single-best-answer code readout. Semantic codes use the same code map/user text with a fixed system instruction to represent the requested law, not always its mode. Neither code channel physically samples; reported softmax is conditional on the allowed tokens.

Numerical output uses one greedy generation of at most96 tokens, requesting a JSON array of nonnegative weights in labels order. Software normalizes their positive sum. These are model-expressed estimates, not token-softmax likelihoods of their spelling. Invalid output falls back to the same-input legacy vector, but all128 numerical outputs parsed successfully, none hit the cap, and no fallback ran. Model errors here cannot be blamed on malformed output or the numerical token ceiling.

This corrects the earlier overly restrictive premise that legitimate probability estimates must come from raw logits. A normalized numerical array is a valid estimate, not a calibration or correctness guarantee. Changing the channel also changes instructions and adds generation work; it is not a clean single-word intervention or equal-FLOP comparison.

## Task semantics

Seed141260922 produces32 worlds,8 each: explicit probabilities, active/spare bag counts, flagged/clear conditional counts, and bag mixtures. Three exhaustive colors, positive exact rational event probabilities and unique modes. Each world has event-law and modal-answer questions with identical evidence/labels, under source-only and correct-reference-note conditions:64 questions,128 views,not128 independent worlds.

The event target is q. The question asking the determinate identity of its unique mode has target e_argmax(q). That target does not imply the imperfect model has calibrated certainty about its own answer. The supplied reference gives exact fractions, decimals and modal identity. No model-generated ordinary reasoning draft is used in this round, so it does not measure an end-to-end ordinary480-plus-numerical replacement.

| Mean TVD,32 worlds each | Legacy | Semantic codes | Numerical |
|---|---:|---:|---:|
| Event, source only | 0.240996 | 0.214243 | 0.185367 |
| Event, correct reference | 0.258179 | 0.237753 | 0.000000 |
| Modal answer, source only | 0.398500 | 0.340319 | 0.436979 |
| Modal answer, correct reference | 0.062572 | 0.089806 | 0.153906 |

Source-only modal accuracy is21/32,21/32,17/32 respectively; numerical output has12 infinite correct-answer log-loss cases. Reference-note modal accuracy is32/32,31/32,29/32; numerical output has two false-zero correct-answer cases. Preserving known event mass does not establish semantic switching or epistemic confidence calibration.

## Concrete failure separation

FLAGGED counts7 amber,8 violet,10 cobalt (CLEAR counts3,34,3 are ineligible) imply[28%,32%,40%]. Numerical output from source alone gives[70%,30%,0%]. Given the correct calculation, legacy codes give[3.96%,3.05%,92.99%], while numerical output preserves[28%,32%,40%]. On the paired modal-answer question with the same correct note, numerical output incorrectly gives[0,1,0]; legacy gives97.10% to the correct cobalt answer. All wrong outputs remain in the release and saved-record CLI examples.

Numerical source-only event vectors are exact in9/32 cases: all8 explicit-percentage cases and one count case. Family mean TVD legacy/numerical:explicit0.1910/0;counts0.2929/0.3661;conditional0.2954/0.2198;mixture0.1846/0.1556. A post-outcome non-explicit-only cut gives0.257656/0.247156 TVD but0.139434/0.179908 squared error. No family selector is fitted or promoted. A conventional exact interpreter can solve the known finite mechanisms perfectly; the experiment concerns neural estimation/readout behavior, not superiority to exact arithmetic.

## Compute and audit

New neural work:128 numerical generations totaling2509 tokens;256 code readouts;16 native anchors and16 full/restricted-projection fixtures. Mean source-only event numerical time24.267s versus5.955s legacy;reference24.116s versus7.970s. Four-thread ARM CPU phases exclude model loading, unused experimental controls and serving overhead. They are not optimized GPU measurements or Jev endpoint comparisons. Correct reference construction is oracle assistance, not measured autonomous reasoning work.

All four pre-outcome source/analysis/test hashes remain unchanged. The independent reducer reconstructs32 event laws from actual English source text by enumerating equiprobable tickets, checks64 target pairs, recomputes256 softmax vectors with60-digit arithmetic, independently parses128 arrays with rational arithmetic, and reconstructs384 probability/answer outputs. Max softmax difference5.55e-16;numeric normalization difference0. All16 workers match both model hashes and exact native/projection fixtures. All128 raw views match planned input/note/channel records.

Current54 tests pass:46 pre-outcome plus8 later interface tests. Three actual saved-record CLI paths cover source-event, reference-event and reference-mode, including wrong answers. They explicitly report recorded replay, not live generation. The standalone live wrapper is injected-runtime tested; underlying channel functions ran in the main neural experiment. No separate live standalone replay is claimed.

The matched dataset has only three outcomes,four authored mechanisms,positive event targets and unique modes. Larger label sets,zero-event targets,ties,independent grammar shift,unknown mechanisms and generic answer correctness calibration are untested.

## Next decision and provenance

Keep numerical probability mass as an explicitly identified output channel, not a universally superior default or unquestioned distillation teacher. The next end-to-end control should give the SAME model-generated ordinary draft to numerical versus code readouts on unused deterministic and stochastic tasks, without oracle notes. Direct training should use independently known targets for the requested quantity, not blindly copy either defective channel.

Run https://github.com/Jaksenc/parameter-golf/actions/runs/35796368589 .
Complete artifact10724905763 SHA256 a136c54afba35f9ba7f291f9ec53dc4f48c961ff41278f66df9c2d7e5ebf2687.
Scientific commit74129edd0d06b56a2cd22b45339b762e4734cded;inference3eb76e01c58ad4696521ad6a176b9f121a1ce736;protocol320e0cb89e53ef76691bae887edf2a1b611dda57;analysis locke637b67d3550a72c9fffa31d4e3d32d23d7022ea.
ModelQwen/Qwen3.5-4B revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Main unchanged.
Primary prior art: Reasoning Under Uncertainty(arXiv2509.10739), Probabilistic Calibration Is a Trainable Capability in Language Models(arXiv2605.11845). These motivate distinct mode,estimation,sampling targets; no literature-wide invention or replication claim.
