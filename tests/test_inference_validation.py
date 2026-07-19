"""T5: Inference synthetic null and recovery tests.

PURPOSE:
  Validate inference pipeline with real forward solver on synthetic data.
  Addresses audit finding: "no full-forward synthetic null/recovery."

TESTS:
  T5-2: Null test — FLRW synthetic data → posterior includes Σ_H=0
  T5-3: Recovery test — Σ_H=0.3 synthetic data → posterior recovers Σ_H≈0.3
  T5-4: Canonical forward model wrapper runs and returns finite values

DESIGN:
  Uses grid_scan (not MCMC) for deterministic, fast validation.
  Uses canonical_forward_solver (real ODE, not surrogate).
  Smoke tier: N_q=6 for speed (~15s per evaluation).
"""
from __future__ import annotations

import numpy as np
import pytest

from rabbit.inference.forward_likelihood import (
    BBNLikelihood, ForwardModel, BBNPrediction, Observation,
    canonical_forward_solver, make_canonical_forward_model,
    grid_scan, confidence_interval_1d,
)


# ═══════════════════════════════════════════════════════════════
# T5-4: Canonical forward model basic functionality
# ═══════════════════════════════════════════════════════════════

class TestCanonicalForwardModel:
    """Verify the canonical forward solver wrapper."""

    def test_canonical_solver_runs(self):
        """canonical_forward_solver returns finite values at FLRW."""
        pred = canonical_forward_solver(Sigma_H=0.0, N_q=6)
        assert pred.success
        assert 0.2 < pred.Yp < 0.3, f"Y_p = {pred.Yp}"
        assert 1e-6 < pred.DH < 1e-4, f"D/H = {pred.DH}"

    def test_canonical_solver_anisotropic(self):
        """canonical_forward_solver at Σ=0.1 returns Y_p > FLRW."""
        pred_0 = canonical_forward_solver(Sigma_H=0.0, N_q=6)
        pred_1 = canonical_forward_solver(Sigma_H=0.1, N_q=6)
        assert pred_1.Yp > pred_0.Yp

    def test_make_canonical_forward_model(self):
        """make_canonical_forward_model creates working ForwardModel."""
        fm = make_canonical_forward_model(N_q=6)
        pred = fm.predict(Sigma_H=0.0)
        assert pred.success
        assert pred.Yp > 0.2

    def test_likelihood_finite(self):
        """BBNLikelihood returns finite log-likelihood."""
        fm = make_canonical_forward_model(N_q=6)
        like = BBNLikelihood(fm)
        ll = like.log_likelihood(Sigma_H=0.0)
        assert np.isfinite(ll), f"log L = {ll}"


# ═══════════════════════════════════════════════════════════════
# T5-2: Synthetic null test
# ═══════════════════════════════════════════════════════════════

class TestSyntheticNull:
    """Null test: FLRW data → posterior consistent with Σ_H=0."""

    @pytest.mark.slow
    def test_null_sigma_grid_scan(self):
        """Inject FLRW data, scan Σ_H, verify peak near 0."""
        # Generate "observed" data from FLRW
        truth = canonical_forward_solver(Sigma_H=0.0, N_q=6)
        assert truth.success

        # Observations = truth ± realistic errors
        obs_yp = Observation("Y_p", truth.Yp, 0.004)
        obs_dh = Observation("D/H", truth.DH, 0.03e-5)

        fm = make_canonical_forward_model(N_q=6)
        like = BBNLikelihood(fm, observations=[obs_yp, obs_dh])

        # Grid scan over Σ_H
        sigma_grid = np.linspace(0.0, 0.5, 11)
        log_likes = []
        for s in sigma_grid:
            ll = like.log_likelihood(Sigma_H=s)
            log_likes.append(ll)
            print(f"  Σ_H={s:.2f}: log L = {ll:.4f}")

        log_likes = np.array(log_likes)

        # Peak should be at or near Σ_H=0
        peak_idx = np.argmax(log_likes)
        peak_sigma = sigma_grid[peak_idx]
        print(f"  Peak at Σ_H = {peak_sigma:.2f}")

        # The peak should be in the first 3 bins (Σ ≤ 0.1)
        # because FLRW data shouldn't prefer anisotropy
        assert peak_sigma <= 0.15, \
            f"Null test failed: peak at Σ_H={peak_sigma:.2f}, expected near 0"

        # Log-likelihood should decrease monotonically (or nearly so) from peak
        # Allow small non-monotonicity from numerical noise
        assert log_likes[0] >= log_likes[-1] - 1.0, \
            "Null test: Σ_H=0.5 should not be preferred over Σ_H=0"


# ═══════════════════════════════════════════════════════════════
# T5-3: Synthetic recovery test
# ═══════════════════════════════════════════════════════════════

class TestSyntheticRecovery:
    """Recovery test: Σ_H=0.3 data → posterior recovers Σ_H≈0.3."""

    @pytest.mark.slow
    def test_recovery_sigma_grid_scan(self):
        """Inject Σ=0.3 data, scan Σ_H, verify peak near 0.3."""
        # Generate "observed" data from Σ_H=0.3
        truth = canonical_forward_solver(Sigma_H=0.3, N_q=6)
        assert truth.success

        # Observations = truth ± realistic errors
        obs_yp = Observation("Y_p", truth.Yp, 0.004)
        obs_dh = Observation("D/H", truth.DH, 0.03e-5)

        fm = make_canonical_forward_model(N_q=6)
        like = BBNLikelihood(fm, observations=[obs_yp, obs_dh])

        # Grid scan
        sigma_grid = np.linspace(0.0, 0.6, 13)
        log_likes = []
        for s in sigma_grid:
            ll = like.log_likelihood(Sigma_H=s)
            log_likes.append(ll)
            print(f"  Σ_H={s:.2f}: log L = {ll:.4f}")

        log_likes = np.array(log_likes)

        peak_idx = np.argmax(log_likes)
        peak_sigma = sigma_grid[peak_idx]
        print(f"  Peak at Σ_H = {peak_sigma:.2f}")

        # Peak should be within ±0.15 of true value 0.3
        assert abs(peak_sigma - 0.3) < 0.15, \
            f"Recovery failed: peak at Σ_H={peak_sigma:.2f}, expected ~0.3"
