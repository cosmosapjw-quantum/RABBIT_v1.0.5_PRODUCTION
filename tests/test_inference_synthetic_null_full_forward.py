from __future__ import annotations

import numpy as np
import pytest

from rabbit.inference.forward_likelihood import (
    BBNLikelihood,
    Observation,
    canonical_forward_solver,
    make_canonical_forward_model,
)


@pytest.mark.production
@pytest.mark.slow
def test_full_forward_null_scan_does_not_prefer_spurious_typeI_shear():
    """Synthetic FLRW data must peak at Sigma_H=0 after marginalizing eta."""
    truth = canonical_forward_solver(Sigma_H=0.0, eta=6.104e-10, N_q=8, backend="scipy")
    assert truth.success, truth.metadata

    observations = [
        Observation("Y_p", truth.Yp, 5.0e-4),
        Observation("D/H", truth.DH, 3.0e-7),
    ]
    model = make_canonical_forward_model(N_q=8, backend="scipy")
    likelihood = BBNLikelihood(model, observations=observations)

    sigma_grid = np.array([0.0, 0.05, 0.10, 0.20])
    eta_grid = np.array([6.00e-10, 6.104e-10, 6.20e-10])
    loglike = np.empty((len(sigma_grid), len(eta_grid)))

    for i, sigma in enumerate(sigma_grid):
        for j, eta in enumerate(eta_grid):
            loglike[i, j] = likelihood.log_likelihood(Sigma_H=float(sigma), eta=float(eta))

    best = np.unravel_index(np.argmax(loglike), loglike.shape)
    best_sigma = sigma_grid[best[0]]
    profile = np.max(loglike, axis=1)

    assert best_sigma == 0.0
    assert profile[0] > profile[-1] + 1.0
    assert profile[0] >= np.max(profile)
    assert np.all(np.isfinite(loglike))


@pytest.mark.production
@pytest.mark.slow
def test_full_forward_null_profile_eta_tau_no_spurious_typeI_shear():
    """FLRW synthetic data must keep Sigma_H=0 preferred after eta/tau profiling."""
    truth = canonical_forward_solver(
        Sigma_H=0.0,
        eta=6.104e-10,
        tau_n=878.4,
        N_q=6,
        backend="scipy",
    )
    assert truth.success, truth.metadata

    observations = [
        Observation("Yp", truth.Yp, 5.0e-4),
        Observation("DH", truth.DH, 3.0e-7),
    ]
    model = make_canonical_forward_model(N_q=6, backend="scipy")
    likelihood = BBNLikelihood(model, observations=observations)

    sigma_grid = np.array([0.0, 0.10, 0.20])
    eta_grid = np.array([6.04e-10, 6.104e-10, 6.18e-10])
    tau_grid = np.array([877.9, 878.4, 878.9])
    loglike = np.empty((len(sigma_grid), len(eta_grid), len(tau_grid)))

    for i, sigma in enumerate(sigma_grid):
        for j, eta in enumerate(eta_grid):
            for k, tau_n in enumerate(tau_grid):
                loglike[i, j, k] = likelihood.log_likelihood(
                    Sigma_H=float(sigma),
                    eta=float(eta),
                    tau_n=float(tau_n),
                )

    profile = np.max(loglike, axis=(1, 2))
    best_sigma = sigma_grid[int(np.argmax(profile))]

    assert np.all(np.isfinite(loglike))
    assert best_sigma == 0.0
    assert profile[0] >= np.max(profile)
    assert profile[0] > profile[-1] + 0.5
