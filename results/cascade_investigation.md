# Why the cascade "replicated" in the original but not seeds 1-2 — resolved

Question: original cascade suppressed the wood code 1.00->0.03 under the SAE wood
feature; seeds 1-2 showed 1.00->1.00 / 0.97. Why?

## E1/E2 (cascade_diagnose.py): teacher-forced vs closed-loop
Teacher-forcing the baseline action sequence reproduces the split (orig wood->0.03,
seeds ->1.00/0.97), and random/CAV are inert with comparable KL/MAE. => NOT policy
steering and NOT a generic perturbation; it is a world-model, feature-specific effect.

## E3 (cascade_reselect.py): causal re-selection
The cascade picks the wood feature by DETECTOR LIFT (decodability). Sweeping the wood
features by teacher-forced generative gating shows every agent HAS a gate:
  original: f187   (lift rank 1/21)  wood within -> 0.00   [gate == best decoder, lucky]
  seed1:    f255   (lift rank 25/29, lift 4x)  -> 0.10      [gate is a low-lift feature]
  seed2:    f996   (lift rank 9/19,  lift 22x) -> 0.13      [gate is a low-lift feature]
=> The discrepancy was a FEATURE-SELECTION artifact. Generative control dissociates
from decodability even within the SAE dictionary.

## Gate-feature cascade (imagine_cascade --feat-override): does write drive?
With each agent's GATE feature (downstream tools, specific: off-tree unchanged, random inert):
  original f187: wood 1.00->0.03 | wood tools 0.80->0.78 (FLAT)    | return 2.56->2.54
  seed1    f255: wood 1.00->0.23 | wood tools 0.78->0.10 (CASCADE) | return 2.58->3.56
  seed2    f996: wood 1.00->0.12 | wood tools 0.72->0.03 (CASCADE) | return 2.22->3.01
=> In the SEEDS write DOES drive; the original's non-cascade is the exception.
(return rise = imagined-return artifact off-distribution; achievement cascade is the reliable signal.)

## Path-1 mechanism (hud_gate_check.py): the HUD shortcut governs it
Teacher-forced HUD-strip vs world-region MAE under each gate (world MAE comparable across all):
  original f187: HUD MAE 0.76  world 9.69  ratio 12.7  -> HUD wood PERSISTS  -> no cascade
  seed1    f255: HUD MAE 4.02  world 11.96 ratio 3.0   -> HUD wood REMOVED   -> cascade
  seed2    f996: HUD MAE 3.03  world 8.45  ratio 2.8   -> HUD wood REMOVED   -> cascade

## Conclusion
Write drives behaviour through the tech tree IFF the lesion also removes the concept's
frame-carried (HUD) copy. f187 is a code-only feature: it deletes the wood code while the
HUD still renders wood, so downstream tools re-derive it (no cascade) -- the smoking gun.
The seed gates remove the HUD copy too, so behaviour cascades. The HUD-region MAE under the
lesion predicts the cascade. The 2->3 (write->drive) seam is thus CONDITIONAL on frame
re-derivation -- one mechanism (the HUD shortcut / quasi-Markovian dynamics), not "codes
don't gate behaviour".
