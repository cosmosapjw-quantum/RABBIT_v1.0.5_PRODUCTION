# PR-D Spec — Noisy synthetic-null false-positive-rate test

Audit ref: BD598 D-R3 (P2).

## Problem (verified)

All existing synthetic-null tests (`test_inference_synthetic_null_full_forward.py`,
`test_inference_null_recovery.py`) feed the model's own **noiseless** prediction
back as data — recovery-by-construction. The spurious Σ_H>0 detection rate under
realistic observational noise is therefore untested. Without it, a future Σ_H
"detection/exclusion" statement has no calibrated false-positive baseline.

## Design (reuse, fast)

Model predictions do not depend on the noise, so the (Σ_H, η) grid is solved
ONCE via `canonical_forward_solver` (the only solver cost); the N noise
realizations are cheap likelihood evaluations on the precomputed grid.

- Truth: FLRW (Σ_H=0) at fiducial η.
- N=300 draws of (Y_p, D/H) ~ Normal(truth, σ_obs), σ_obs = (4.0e-3, 2.9e-7).
- Per draw: profile loglike over η, find best Σ_H; count a "detection" iff
  best Σ_H>0 AND ΔlnL(best vs Σ_H=0) > 1.92 (≈95%, one-sided boundary mixture).
- Gate: FPR < 0.10 (nominal ~2.5–5% expected for a Σ_H≥0 boundary parameter).

Seeded RNG (`default_rng(0)`) → deterministic. Marked `production`+`slow`.

## Interpretation

- PASS → the null is calibrated; spurious-detection rate is controlled, a
  prerequisite for any future detection claim. Combined with PR-C (Σ_H is not
  independently identified at the null), this bounds over-claim risk.
- FAIL → real miscalibration finding (the model manufactures spurious shear
  preference under noise); would be filed as a new P-level risk, not forced green.

## Acceptance

- New test green (or a recorded miscalibration finding). Reuses the existing
  forward model + likelihood; no new production surface. `/review` clean.
