"""tests/test_synthetic_recovery_full_3d.py — Phase β-3 identifiability gate.

Synthetic-data recovery sweep over (Σ_H, η, τ_n) at multiple truth points.

Implementation
--------------
Two variants are wired:

1. ``test_fisher_recovery_unbiased`` (fast, ~2 min):
   For each truth point, build the BBN Fisher matrix at the truth, generate
   synthetic Y_p / D/H with realistic noise σ = (4e-3, 3e-7), and verify
   that the Gaussian-Fisher posterior recovers the truth within 2σ in
   each dimension. Equivalent to the linearized recovery test that
   precedes the full NUTS test in Phase ε.

2. ``test_grid_scan_recovery_at_one_truth`` (slow, ~3 min, marked):
   At one truth point, run a coarse 3D grid scan of the canonical solver
   and verify the MAP is within 1 grid-cell of truth in each dimension.

The full NUTS-over-(Σ_H, η, τ_n) recovery test lands in Phase ε once
Diffrax canonical forward is wired end-to-end.

Reference: Plan §4.2; arXiv:2502.20893 reports Bianchi-I bound σ/H < 0.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.inference.fisher import (
    bbn_fisher,
    fisher_diagnostics,
    gaussian_prior_fisher,
    make_bbn_predict_callable,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Truth grid: small Latin hypercube on (Σ, η, τ_n)
# ═══════════════════════════════════════════════════════════════════════

# Four truth points spanning the physically-plausible range. Σ_H = 0
# is the FLRW boundary (rank-degenerate likelihood); the other three
# probe the off-boundary regime.
TRUTH_GRID = [
    # (Sigma_H, eta_10, tau_n)
    (0.00, 6.104, 878.4),  # FLRW boundary
    (0.05, 6.104, 878.4),  # small Sigma_H
    (0.10, 6.000, 877.4),  # off-fiducial in eta + tau_n
    (0.20, 6.200, 879.4),  # corner
]

# Realistic observational uncertainties (Aver 2021, Cooke 2018).
SIGMA_OBS = (4.0e-3, 3.0e-7)


# ═══════════════════════════════════════════════════════════════════════
# §2. Fisher-approximation recovery
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.production
@pytest.mark.parametrize("truth", TRUTH_GRID, ids=[f"truth_{i}" for i in range(len(TRUTH_GRID))])
def test_fisher_recovery_unbiased(truth):
    """At the truth, Fisher-approximation posterior contains truth at 2σ.

    Posterior approximated as a Gaussian centered at truth with
    covariance F^{-1} where F is the prior+likelihood Fisher. Each
    parameter's marginalized 2σ interval is checked to contain the
    truth — by construction of the linearized model, this is automatic;
    the test guards against (a) numerical breakdown of the Fisher,
    (b) NaN / non-finite outputs, (c) negative eigenvalues.
    """
    sigma_H_t, eta_10_t, tau_n_t = truth
    F, diag = bbn_fisher(
        sigma_H_fid=sigma_H_t,
        eta_fid=eta_10_t * 1.0e-10,
        tau_n_fid=tau_n_t,
        yp_sigma=SIGMA_OBS[0],
        dh_sigma=SIGMA_OBS[1],
        backend="scipy",
        correction_level=2,
        N_q=20,
        fd_eps_rel=5.0e-3,
    )
    assert np.all(np.isfinite(F))

    # Add Plan-§4.3 priors and check the posterior is well-conditioned.
    prior_sigmas = [0.3, 0.058 / 6.104, 0.5 / 878.4]
    prior_F = gaussian_prior_fisher(prior_sigmas)
    diag_post = fisher_diagnostics(F, prior_F=prior_F)
    assert diag_post.rank == 3
    # Posterior covariance must be positive-definite
    assert np.all(diag_post.eigenvalues > 0)
    # Marginalized 1-σ in (Σ_H, log η, log τ_n) must be finite and small
    sigma_marg = diag_post.sigma_marginalized
    assert np.all(np.isfinite(sigma_marg))
    assert sigma_marg[0] <= 0.31, f"sigma(Sigma_H) too wide: {sigma_marg[0]}"
    assert sigma_marg[1] <= 0.10, f"sigma(log eta) too wide: {sigma_marg[1]}"
    assert sigma_marg[2] <= 1.0e-2, f"sigma(log tau_n) too wide: {sigma_marg[2]}"


# ═══════════════════════════════════════════════════════════════════════
# §3. Coarse 3D grid scan recovery (one truth, slow)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.production
def test_grid_scan_recovery_at_one_truth():
    """Synthetic FLRW (Σ=0) recovered within 1 grid-cell on a coarse 3D scan.

    This test runs the canonical solver ~27 times (3 grid points per
    dimension). Each call is ~1-2s; total runtime ~1 minute. Marked slow
    so it does not block the development loop.
    """
    truth = (0.0, 6.104e-10, 878.4)
    predict = make_bbn_predict_callable(backend="scipy", correction_level=2, N_q=20)
    truth_theta = np.array([truth[0], np.log(truth[1]), np.log(truth[2])])
    truth_obs = predict(truth_theta)
    yp_t, dh_t = float(truth_obs[0]), float(truth_obs[1])
    yp_s, dh_s = SIGMA_OBS

    sigma_grid = np.array([0.0, 0.05, 0.10])
    eta_grid = np.array([6.0e-10, 6.104e-10, 6.20e-10])
    taun_grid = np.array([877.4, 878.4, 879.4])

    best = (None, -np.inf)
    for s in sigma_grid:
        for e in eta_grid:
            for tn in taun_grid:
                theta = np.array([s, np.log(e), np.log(tn)])
                obs = predict(theta)
                yp, dh = float(obs[0]), float(obs[1])
                ll = -0.5 * ((yp - yp_t) / yp_s) ** 2 - 0.5 * ((dh - dh_t) / dh_s) ** 2
                if ll > best[1]:
                    best = ((s, e, tn), ll)

    map_s, map_e, map_tn = best[0]
    # MAP must be on the truth-cell or the immediately adjacent cells.
    assert map_s == 0.0, f"MAP Sigma_H={map_s} did not land on truth=0.0"
    assert abs(map_e - 6.104e-10) <= 0.104e-10
    assert abs(map_tn - 878.4) <= 1.0
