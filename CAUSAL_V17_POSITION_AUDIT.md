# Static source audit: position balance, September 23, 2026

This check used only frozen inputs, exact targets and the selected schedules. No clean-run trained outcomes were inspected to choose the diagnostic. The running factorial, seeds, targets, objective weights, step sizes and data remain unchanged.

The full 128-world fit pool is approximately balanced in modal code positions. That does NOT imply the actually selected 32-world trajectory is balanced. For four-option problems, all three seeds have zero selected training worlds whose modal category occupies slot0. The test contains eight four-option worlds: four have slot0 as the mode and four slot1; slots2 and3 are absent. The calibration four-option subset has the same slot0/slot1 distribution.

Actual training four-option modal position counts:
- seed17101: slot1=2,slot2=3,slot3=3.
- seed17102: slot1=3,slot2=4,slot3=1.
- seed17103: slot1=3,slot2=4,slot3=1.

This is a real assignment distribution shift, not a miscomputed target. The same matched schedule still applies to every arm of a seed, so the factorial comparison estimates treatment effects under this data design. It does not alone establish that event and modal supervision are intrinsically incompatible, or that one optimizer setting is generally preferable. A model learning a true permutation-independent operation should tolerate code-position changes, but a small adaptation can instead reinforce positional shortcuts.

Do not silently repair the data, remove four-option cases, or promote a best subgroup after outcomes. Report all prespecified primary results and separately inspect cardinality-specific behavior. Any subsequent confirmation should balance actual exposure and test-code positions, not merely the fit pool. This note supplements rather than changes the original protocol.

The earlier target-sharpness gap is substantially reduced: mean normalized event-target entropy in the selected training worlds is0.7735/0.7723/0.7702 for the three seeds, versus0.7990 in the test. That controls one earlier concern but does not remove the position imbalance or the intentionally shortcut-discriminating test selection.
