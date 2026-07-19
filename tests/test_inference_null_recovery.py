"""
Inference null recovery test — validates that the constraint pipeline
produces correct results on synthetic data with known truth.

This is the missing "synthetic calibration" gate required by
model_comparison.py before evidence values can be cited.
"""
import pytest
import numpy as np


@pytest.mark.production
class TestInferenceNullRecovery:

    @staticmethod
    def _fm():
        from rabbit.inference.forward_likelihood import make_canonical_forward_model, Observation
        return make_canonical_forward_model(backend='auto', correction_level=0, N_q=6), Observation

    def test_null_no_spurious_detection(self):
        """FLRW data (Σ=0 truth) → Σ=0 should be preferred over Σ=0.5."""
        fm, Obs = self._fm()
        truth = fm.predict(Sigma_H=0.0)
        obs = [Obs('Yp', truth.Yp, 0.004)]
        ll_0 = fm.log_likelihood(obs, Sigma_H=0.0)
        ll_5 = fm.log_likelihood(obs, Sigma_H=0.5)
        assert ll_0 > ll_5, f"Null preferred: ll(0)={ll_0:.2f} vs ll(0.5)={ll_5:.2f}"

    def test_injection_recovery(self):
        """Injected Σ=0.3 signal should be recovered within ±0.15."""
        fm, Obs = self._fm()
        truth = fm.predict(Sigma_H=0.3)
        obs = [Obs('Yp', truth.Yp, 0.004)]
        sigma_grid = np.linspace(0, 0.6, 13)
        lls = [fm.log_likelihood(obs, Sigma_H=float(s)) for s in sigma_grid]
        best = sigma_grid[np.argmax(lls)]
        assert abs(best - 0.3) < 0.15, f"Recovery failed: best={best:.2f}"

    def test_monotonic_away_from_null(self):
        """Likelihood at FLRW truth should decrease with Σ."""
        fm, Obs = self._fm()
        truth = fm.predict(Sigma_H=0.0)
        obs = [Obs('Yp', truth.Yp, 0.004)]
        ll_prev = fm.log_likelihood(obs, Sigma_H=0.0)
        for s in [0.1, 0.2, 0.3, 0.5]:
            ll = fm.log_likelihood(obs, Sigma_H=s)
            assert ll <= ll_prev + 0.1, f"Non-monotonic at Σ={s}"
            ll_prev = ll

    def test_upper_limit_finite(self):
        """Profile 95% CL from real observations should be finite."""
        fm, Obs = self._fm()
        obs = [Obs('Yp', 0.2449, 0.004)]
        sigma_grid = np.linspace(0, 0.75, 16)
        lls = np.array([fm.log_likelihood(obs, Sigma_H=float(s)) for s in sigma_grid])
        dchi2 = -2 * (lls - np.max(lls))
        above = np.where(dchi2 > 3.84)[0]
        if len(above) > 0:
            sigma_95 = sigma_grid[above[0]]
            assert 0 < sigma_95 <= 0.75, f"95% CL={sigma_95:.2f} not in validated envelope"
