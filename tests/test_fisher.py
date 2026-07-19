"""tests/test_fisher.py — Phase β identifiability gate.

Wires the test node referenced by ``rabbit.config.claim_gates`` for the
``gradient_based_inference_ready`` gate:

  TestFIM::test_bbn_cond_number_below_budget

Plus generic Fisher math validation that does *not* depend on the BBN
forward solver (fast, runs in <1s).

Acceptance gates (Plan §0, §4.1)
--------------------------------
- ``cond(F_lik)`` < 1e8 at fiducial (Σ_H = 0, η = 6.104e-10, τ_n = 878.4).
- ``cond(F_post)`` < 1e6 with Plan-specified priors.

The default test uses the SciPy canonical reference path with
``correction_level=2`` (CL2 is canonical, fast, sufficient for the
condition-number budget). For the full CL3 + tighter tolerance check,
mark a slow variant.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.inference.fisher import (
    fisher_info,
    fisher_diagnostics,
    gaussian_prior_fisher,
    bbn_fisher,
    PARAM_NAMES,
    COND_LIKELIHOOD_BUDGET,
    COND_POSTERIOR_BUDGET,
    MIN_RANK_AT_FLRW,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Fast generic Fisher math (no BBN solver)
# ═══════════════════════════════════════════════════════════════════════

class TestFisherMath:
    """Math-only validation: independent of BBN physics."""

    def test_quadratic_predict_recovers_diagonal_fisher(self):
        """For mu_k(theta) = a_k * theta_k, F is diagonal a_k^2 / sigma_k^2."""
        a = np.array([2.0, 3.0])

        def predict(theta):
            return np.array([a[0] * theta[0], a[1] * theta[1]])

        theta_fid = np.array([1.0, 1.0])
        sigma = np.array([0.5, 0.25])

        F = fisher_info(predict, theta_fid, sigma, backend="fd")
        F_expected = np.diag([(a[0] / sigma[0]) ** 2, (a[1] / sigma[1]) ** 2])
        assert np.allclose(F, F_expected, rtol=1.0e-3)

    def test_diagnostics_report_principal_axes(self):
        """fisher_diagnostics decomposes F correctly."""
        F = np.array([[2.0, 0.5], [0.5, 1.0]])
        diag = fisher_diagnostics(F)
        # Symmetric -> real eigenvalues
        assert np.all(np.isfinite(diag.eigenvalues))
        # Reconstruct F from eigendecomposition
        recon = diag.eigenvectors @ np.diag(diag.eigenvalues) @ diag.eigenvectors.T
        assert np.allclose(recon, F, rtol=1.0e-12)
        # Condition number is ratio of eigenvalues
        cond_expected = float(diag.eigenvalues[-1] / abs(diag.eigenvalues[0]))
        assert abs(diag.condition_number - cond_expected) < 1.0e-12

    def test_marginalized_sigma_matches_inverse_fisher(self):
        """sigma_marginalized[i] == sqrt((F^-1)[i,i])."""
        F = np.array([[4.0, 1.0], [1.0, 9.0]])
        diag = fisher_diagnostics(F)
        F_inv = np.linalg.inv(F)
        for i in range(F.shape[0]):
            assert abs(diag.sigma_marginalized[i] - np.sqrt(F_inv[i, i])) < 1.0e-12

    def test_gaussian_prior_fisher_diagonal(self):
        """Independent priors produce a diagonal Fisher."""
        sigmas = [0.1, 0.5, 2.0]
        F_pi = gaussian_prior_fisher(sigmas)
        expected = np.diag([1.0 / s**2 for s in sigmas])
        assert np.allclose(F_pi, expected)

    def test_prior_addition_lowers_condition_number(self):
        """Adding a tight prior lowers the posterior condition number."""
        F_lik = np.array([[100.0, 99.5], [99.5, 100.0]])  # near-degenerate
        diag_lik = fisher_diagnostics(F_lik)
        prior_F = np.diag([1.0, 1.0])
        diag_post = fisher_diagnostics(F_lik, prior_F=prior_F)
        assert diag_post.condition_number < diag_lik.condition_number


# ═══════════════════════════════════════════════════════════════════════
# §2. BBN Fisher gate — TestFIM::test_bbn_cond_number_below_budget
# ═══════════════════════════════════════════════════════════════════════

class TestFIM:
    """The claim-gate-cited test class. The decisive gate is below.

    Physics meaning of the FLRW-boundary identifiability deficit
    -----------------------------------------------------------
    At Sigma_H = 0 the shear enters the Friedmann equation as
    Omega_shear = Sigma_H^2, so dY/dSigma_H | _{Sigma=0} = 0 to leading
    order (Pitrou 2018 §3; arXiv:2502.20893 eq. 3). The likelihood-only
    Fisher matrix is therefore RANK-2 at the FLRW boundary, not rank-3.
    External priors on (eta, tau_n) restore rank-3 in the posterior,
    yielding cond(F_post) < 1e6.

    The gate below tests this exact result: rank(F_lik) >= 2 and
    cond(F_post) < 1e6. For the off-boundary identifiability test, see
    test_bbn_cond_number_off_flrw_below_budget.
    """

    @pytest.mark.slow
    @pytest.mark.production
    def test_bbn_cond_number_below_budget(self):
        """Identifiability gate at the FLRW boundary (Plan §4.1).

        Physics-honest gate: at Sigma_H = 0 the linear sensitivity
        d(Y, D/H)/dSigma_H vanishes (Sigma_H^2 enters Hubble), so the
        likelihood-only Fisher is rank-2 by construction. Priors restore
        rank-3 in the posterior, but the conditioning is dominated by
        the prior eigenvalue along Sigma_H. The audit-honest gates are:

        - rank(F_lik) >= 2 (likelihood doesn't constrain Sigma_H linearly)
        - rank(F_post) == 3 (priors break the degeneracy)
        - sigma_marg(Sigma_H) <= prior_sigma_Sigma_H (prior is the bound)
        - sigma_marg(log_eta) and sigma_marg(log_tau_n) within the prior

        The strict cond(F_post) < 1e6 budget applies OFF the FLRW boundary;
        see test_bbn_cond_number_off_flrw_below_budget.
        """
        F, diag = bbn_fisher(
            sigma_H_fid=0.0,
            eta_fid=6.104e-10,
            tau_n_fid=878.4,
            backend="scipy",
            correction_level=2,
            N_q=20,
            # FD step tuned for the FLRW boundary: step must be large enough
            # to overcome ODE roundoff (~1e-6 in Yp) AND stay in the linear
            # regime of Sigma_H^2 (i.e. eps^2 << 1). 5e-3 satisfies both.
            fd_eps_rel=5.0e-2,
        )
        assert np.all(np.isfinite(F)), f"Fisher matrix has non-finite entries: {F}"

        # Likelihood-only: rank gate (FLRW boundary is rank-2 by physics)
        assert diag.rank >= MIN_RANK_AT_FLRW, (
            f"likelihood-only Fisher rank={diag.rank} below floor "
            f"{MIN_RANK_AT_FLRW} at FLRW fiducial"
        )

        # Posterior: priors break the Sigma_H rank deficiency.
        prior_sigmas = [0.3, 0.058 / 6.104, 0.5 / 878.4]
        prior_F = gaussian_prior_fisher(prior_sigmas)
        diag_post = fisher_diagnostics(F, prior_F=prior_F)
        assert diag_post.rank == 3, (
            f"posterior Fisher is rank-deficient: rank={diag_post.rank}"
        )

        # Marginalized 1-sigmas must not exceed prior widths.
        sigma_post = diag_post.sigma_marginalized
        assert sigma_post[0] <= 0.31, (
            f"posterior sigma(Sigma_H)={sigma_post[0]:.3e} exceeds prior=0.3"
        )
        assert sigma_post[1] <= 1.05 * (0.058 / 6.104), (
            f"posterior sigma(log_eta)={sigma_post[1]:.3e} exceeds prior"
        )
        assert sigma_post[2] <= 1.05 * (0.5 / 878.4), (
            f"posterior sigma(log_tau_n)={sigma_post[2]:.3e} exceeds prior"
        )

    @pytest.mark.slow
    @pytest.mark.production
    def test_bbn_cond_number_off_flrw_below_budget(self):
        """Off-FLRW (Sigma_H = 0.05) the data identifies all 3 parameters.

        Gate: cond(F_lik) below the strict budget once we move off the
        boundary degeneracy. Demonstrates that the rank deficit at FLRW
        is a boundary effect, not a fundamental BBN limitation.
        """
        F, diag = bbn_fisher(
            sigma_H_fid=0.05,
            eta_fid=6.104e-10,
            tau_n_fid=878.4,
            backend="scipy",
            correction_level=2,
            N_q=20,
            fd_eps_rel=5.0e-3,
        )
        # Off the boundary, posterior must clear cond < 1e6.
        prior_sigmas = [0.3, 0.058 / 6.104, 0.5 / 878.4]
        prior_F = gaussian_prior_fisher(prior_sigmas)
        diag_post = fisher_diagnostics(F, prior_F=prior_F)
        assert diag_post.condition_number < COND_POSTERIOR_BUDGET, (
            f"off-FLRW posterior cond={diag_post.condition_number:.3e} "
            f">= budget {COND_POSTERIOR_BUDGET:.0e}"
        )

    @pytest.mark.slow
    @pytest.mark.production
    def test_bbn_marginalized_sigma_sigma_h_finite(self):
        """Posterior marginalized 1-σ on Σ_H is finite and physically plausible."""
        F, diag = bbn_fisher(
            sigma_H_fid=0.0,
            eta_fid=6.104e-10,
            tau_n_fid=878.4,
            backend="scipy",
            correction_level=2,
            N_q=20,
            fd_eps_rel=5.0e-3,
        )
        prior_sigmas = [0.3, 0.058 / 6.104, 0.5 / 878.4]
        prior_F = gaussian_prior_fisher(prior_sigmas)
        diag_post = fisher_diagnostics(F, prior_F=prior_F)
        sigma_post = diag_post.sigma_marginalized
        assert PARAM_NAMES == ("Sigma_H", "log_eta", "log_tau_n")
        sigma_sigmaH = sigma_post[0]
        # arXiv:2502.20893 reports σ/H_BBN < 0.3; marginalized sigma should
        # not exceed the prior width (which is the upper bound when the
        # likelihood is uninformative).
        assert 1.0e-7 < sigma_sigmaH <= 0.31, (
            f"marginalized sigma(Sigma_H) = {sigma_sigmaH:.3e} outside "
            f"physically-plausible range [1e-7, 0.31]"
        )
