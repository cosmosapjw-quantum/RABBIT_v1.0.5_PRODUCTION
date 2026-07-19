"""tests/test_inference_noisy_null_fpr.py — BD598 PR-D (D-R3).

Calibrated false-positive-rate test for a spurious Σ_H "detection". Every prior
synthetic-null test fed the model's own NOISELESS prediction back as data
(recovery-by-construction), so the spurious-detection rate was untested. Here we
draw many NOISY (Y_p, D/H) realizations at the FLRW truth and measure how often a
profile-likelihood scan spuriously prefers Σ_H > 0 at ~95% significance.

This is the calibrated prerequisite for any future Σ_H detection/exclusion claim:
if the FPR were far above nominal, a "detection" could not be trusted.

Efficiency: model predictions are noise-independent, so the (Σ_H, η) grid is
solved ONCE; the N noise realizations are cheap likelihood evaluations.

Scope / limitations (this is a GATE-CHECK, not a formal calibration study):
  * single seeded RNG, N=300 → a point estimate, no confidence interval;
  * coarse η grid (5 pts) and Σ_H grid (6 pts) approximate the profile;
  * N_q=8 solver discretization (~1e-4 in Y_p) is ~40× below the YP_ERR=4e-3
    noise, so it does not materially move the FPR.
The bound is therefore conservative (0.05); a true nominal rate for a one-sided
boundary parameter is ~2.5%. A formal calibration (fine grid, many seeds,
bootstrap CI) is deferred — this gate only guards against a grossly miscalibrated
null that would invalidate a future detection claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver

# Profile-likelihood significance for 1 extra parameter at a one-sided boundary.
# ΔlnL > 1.92 ≈ 95% (χ²_1 / 2 for the Σ_H ≥ 0 boundary mixture).
DELTA_LNL_95 = 1.92
N_REALIZATIONS = 300
# Observational σ used both to add noise and in the likelihood (calibrated case).
YP_ERR = 4.0e-3
DH_ERR = 2.9e-7


@pytest.fixture(scope="module")
def grid_predictions():
    """Precompute [Y_p, D/H] on a (Σ_H, η) grid once (the only solver cost)."""
    sigma_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    eta_grid = np.array([5.90e-10, 6.00e-10, 6.104e-10, 6.20e-10, 6.30e-10])
    yp = np.empty((sigma_grid.size, eta_grid.size))
    dh = np.empty((sigma_grid.size, eta_grid.size))
    for i, s in enumerate(sigma_grid):
        for j, e in enumerate(eta_grid):
            r = canonical_forward_solver(Sigma_H=float(s), eta=float(e), N_q=8, backend="scipy")
            assert r.success, r.metadata
            yp[i, j] = r.Yp
            dh[i, j] = r.DH
    return sigma_grid, eta_grid, yp, dh


@pytest.mark.production
@pytest.mark.slow
def test_noisy_flrw_null_shear_signal_below_noise_floor(grid_predictions):
    """No spurious-detection regime: the Σ_H shear signal sits below the noise.

    BD599 W2-R2 (self-correction of PR-D): this is NOT a calibration guarantee.
    Because the shear→Y_p response is far below σ_obs across the prior range, a
    significant spurious Σ_H detection would need a ~5.9σ aligned (Y_p,D/H) draw,
    unreachable in N=300 — so the empirical rate is pinned at 0 by geometry, not
    by demonstrated calibration. We record the rate as an honest observation that
    the null does not manufacture spurious shear preference at this noise level;
    a reachable-regime false-positive-rate calibration is deferred.
    """
    sigma_grid, eta_grid, yp, dh = grid_predictions
    i0 = int(np.argmin(np.abs(sigma_grid)))          # Σ_H = 0 row
    j0 = int(np.argmin(np.abs(eta_grid - 6.104e-10)))  # fiducial η
    yp_truth, dh_truth = yp[i0, j0], dh[i0, j0]

    rng = np.random.default_rng(0)
    detections = 0
    for _ in range(N_REALIZATIONS):
        yp_obs = yp_truth + rng.normal(0.0, YP_ERR)
        dh_obs = dh_truth + rng.normal(0.0, DH_ERR)
        loglike = -0.5 * (((yp_obs - yp) / YP_ERR) ** 2 + ((dh_obs - dh) / DH_ERR) ** 2)
        profile = loglike.max(axis=1)                # profile over η
        best_i = int(np.argmax(profile))
        if sigma_grid[best_i] > 0.0 and (profile[best_i] - profile[i0]) > DELTA_LNL_95:
            detections += 1

    fpr = detections / N_REALIZATIONS
    print(f"\nnoisy-null spurious Σ_H rate = {fpr:.4f} "
          f"(N={N_REALIZATIONS}, ΔlnL>{DELTA_LNL_95}, σ_obs=({YP_ERR},{DH_ERR}); "
          f"signal below noise floor — geometry-pinned, not a calibration)")
    # Honest observation, not a calibration claim: no spurious-detection regime here.
    assert fpr < 0.05, f"unexpected spurious Σ_H rate {fpr:.3f} (signal-below-noise regime)"
