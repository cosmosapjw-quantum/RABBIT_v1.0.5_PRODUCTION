"""T4: Backend parity harness.

PURPOSE:
  Quantify solver backend agreement on identical physics.
  Addresses audit finding: "component parity ≠ full physics parity."

DESIGN:
  Run the same FLRW BBN problem through two SciPy methods (Radau, BDF)
  at identical config. Compare observables.

  JAX full-driver parity is NOT tested here because the JAX path is
  explicitly not the reference pipeline (solver_config.py lines 8-10).
  Component-level JAX parity is covered by test_j01/j02/j04/j06.

ACCEPTANCE CRITERIA:
  AC1: Radau vs BDF |ΔY_p| < 1e-5
  AC2: Radau vs BDF |Δ(D/H)/D/H| < 0.1%
  AC3: Both succeed without solver failure
"""
from __future__ import annotations

import pytest
from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.solver_config import SolverConfig, SolverMethod


def _run_with_method(method: SolverMethod, rtol=1e-8):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0, N_q=6, n_reactions=12,
        correction_level=0, enable_teff=False,
        solver=SolverConfig(method=method, rtol=rtol, atol=1e-10, max_step=0.1),
    )
    return run_full_coupled_typeI(cfg)


class TestBackendParity:
    """Radau vs BDF on identical FLRW physics."""

    def test_radau_bdf_yp_parity(self):
        """AC1: |ΔY_p| < 1e-5 between Radau and BDF."""
        r_radau = _run_with_method(SolverMethod.RADAU)
        r_bdf = _run_with_method(SolverMethod.BDF)
        delta = abs(r_radau.observables.Yp - r_bdf.observables.Yp)
        print(f"Radau Y_p = {r_radau.observables.Yp:.8f}")
        print(f"BDF   Y_p = {r_bdf.observables.Yp:.8f}")
        print(f"|ΔY_p|    = {delta:.2e}")
        assert delta < 1e-5, f"|ΔY_p| = {delta:.2e} exceeds 1e-5"

    def test_radau_bdf_dh_parity(self):
        """AC2: |Δ(D/H)/D/H| < 0.1% between Radau and BDF."""
        r_radau = _run_with_method(SolverMethod.RADAU)
        r_bdf = _run_with_method(SolverMethod.BDF)
        rel = abs(r_radau.observables.DH - r_bdf.observables.DH) / r_radau.observables.DH
        print(f"Radau D/H = {r_radau.observables.DH:.6e}")
        print(f"BDF   D/H = {r_bdf.observables.DH:.6e}")
        print(f"|Δ(D/H)|/DH = {rel:.2e}")
        assert rel < 1e-3, f"D/H relative diff = {rel:.2e} exceeds 0.1%"


class TestBackendParityAnisotropic:
    """Radau vs BDF on identical anisotropic physics."""

    @pytest.mark.slow
    def test_radau_bdf_anisotropic(self):
        """Same test at Σ=0.1 for the matched reduced anisotropic path."""
        cfg_base = dict(
            Sigma_H_plus=0.1, N_q=6, n_reactions=12,
            correction_level=0, enable_teff=False,
        )
        r_radau = run_full_coupled_typeI(FullCoupledConfig(
            solver=SolverConfig(method=SolverMethod.RADAU, rtol=1e-8, atol=1e-10), **cfg_base))
        r_bdf = run_full_coupled_typeI(FullCoupledConfig(
            solver=SolverConfig(method=SolverMethod.BDF, rtol=1e-8, atol=1e-10), **cfg_base))
        delta = abs(r_radau.observables.Yp - r_bdf.observables.Yp)
        print(f"Anisotropic Radau Y_p = {r_radau.observables.Yp:.8f}")
        print(f"Anisotropic BDF   Y_p = {r_bdf.observables.Yp:.8f}")
        print(f"|ΔY_p| = {delta:.2e}")
        assert delta < 1e-4, f"Anisotropic |ΔY_p| = {delta:.2e} exceeds 1e-4"
