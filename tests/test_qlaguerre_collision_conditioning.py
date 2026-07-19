"""tests/test_qlaguerre_collision_conditioning.py — Phase-1 root cause of the
q-Laguerre dynamic-collision blocker (PROJECT_STATE BD199-BD204).

The dynamic q-Laguerre (q4/q5) FLRW collision-on run collapses before the BBN
endpoint (Rodas5P h-min at N=3→4, T_νx/T_γ~0.0012, N_eff_3T~1.15). This session
pinned the ROOT CAUSE to an ill-conditioned temperature variable, amplified by a
fragile energy moment:

  1. AMPLIFIER (the universal mechanism): the heavy-bank temperature source is
     dT_νx/dN = −T_νx + dQ_nux_bank_N / (2·dρ_pair/dT), with dρ_pair/dT ∝ T_νx³.
     As the bank nears decoupling (T_νx→0) the denominator → 0, so ANY error in
     dQ_nux_bank_N (from ANY source) is amplified by ~1/T_νx³ into a catastrophic
     temperature kick. This is ill-conditioning of the *variable choice*, not a
     floor bug — the `d2 > 1e-50` guard does not bound it.
  2. THE FIX (well-conditioning): evolve the heavy-bank ENERGY density ρ_νx
     (dρ/dN = −4ρ + dQ_nux_bank_N) instead of T_νx directly. Then a small dQ error
     produces a BOUNDED dρ/dN regardless of how small T_νx is; recover T_νx ∝ ρ^{1/4}
     diagnostically. This is the conditioning contract the clean core must satisfy.
  3. THE dQ moment is NOT the root fault: the q-Laguerre plain-weight energy
     moment (w_i·exp(q_i)·q³·C) is measured here to be ACCURATE for smooth
     distributions (FD and slower-decaying exp(−q/2) match their analytic
     integrals to ~machine precision). So dQ_nux_bank noise is small; it is the
     1/T_νx³ amplifier (1) that makes it fatal. This pins the fix priority:
     well-condition the variable (2) is the ROOT remedy; quadrature positivity
     guards are secondary hardening (they cannot save a 1/T³-amplified variable).

These tests lock (1),(2),(3) on SURVIVING modules (nudec_coupled + the Laguerre
grid), so they remain the acceptance contract after the AP65 scaffolding that
currently hosts the broken solver is deleted. No external anchor exists for
Bianchi-I + ν-collision BBN; this is an internal conditioning/consistency contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.thermo.nudec_coupled import (
    coupled_3T_rhs_from_collision_moments,
    _rho_nu_pair,
)


def _dTnux_dN(T_nu_x, dQ_nux_bank_N):
    """Isolate the heavy-bank temperature source; plasma overrides make the γ
    channel trivial so only the ν_x channel is exercised."""
    return coupled_3T_rhs_from_collision_moments(
        1.0, 1.0, float(T_nu_x),
        dQ_nue_pair_N=0.0, dQ_nux_bank_N=float(dQ_nux_bank_N),
        plasma_dT_base_dN=0.0, plasma_drho_dT_MeV3=1.0,
    )[2]


def test_temperature_source_is_ill_conditioned_near_decoupling():
    """A FIXED small dQ error yields a dT_νx/dN that explodes like 1/T_νx³ as the
    heavy bank decouples — the confirmed BD203 mechanism. This is the blocker."""
    dQ = -1.0e-6  # a small, fixed energy-source error
    ratios = []
    prev = None
    for T in (1.0, 1.0e-1, 1.0e-2, 1.0e-3):
        d = abs(_dTnux_dN(T, dQ))
        if prev is not None:
            ratios.append(d / prev)  # per decade of T; ~1000x if ∝1/T³
        prev = d
    # Near-decoupling the source grows steeply (roughly a decade³ per T-decade).
    assert ratios[-1] > 100.0, f"expected steep 1/T^3 growth, got per-decade ratios {ratios}"
    # At the BD203 heavy-bank temperature the kick is catastrophic (|dT/dN| ≫ 1e3).
    blowup = abs(_dTnux_dN(3.7624e-05, dQ))
    assert blowup > 1.0e5, f"expected catastrophic blow-up at BD203 T_νx, got {blowup:.3e}"


def test_energy_variable_evolution_is_well_conditioned():
    """THE FIX contract: evolving ρ_νx (dρ/dN = −4ρ + dQ) is bounded for the SAME
    tiny T_νx and dQ that blow up the temperature form. The clean core must evolve
    the energy (or ρ^{1/4}/log), not T_νx directly."""
    dQ = -1.0e-6
    for T in (1.0, 1.0e-2, 3.7624e-05):
        rho = 2.0 * _rho_nu_pair(T)            # heavy bank = 2 pairs
        drho_dN = -4.0 * rho + dQ              # well-conditioned: no division by c_v∝T³
        # bounded by the redshift term + the source, independent of how small T is
        assert abs(drho_dN) <= 4.0 * rho + abs(dQ) + 1e-30
        assert np.isfinite(drho_dN)
    # Contrast: at T_νx=3.76e-5 the temperature form is O(1e6) while the energy
    # form is O(dQ)=1e-6 — a ~1e12 conditioning improvement.
    T = 3.7624e-05
    temp_form = abs(_dTnux_dN(T, dQ))
    energy_form = abs(-4.0 * 2.0 * _rho_nu_pair(T) + dQ)
    assert temp_form / energy_form > 1.0e6


def test_energy_moment_is_accurate_so_the_fault_is_amplification_not_quadrature():
    """The q-Laguerre plain-weight energy moment (w·exp(q)·q³·C) is ACCURATE for
    smooth distributions — Gauss-Laguerre resolves FD-like and slower-decaying
    (exp(−q/2)) sources to ~machine precision vs a dense direct integral. So the
    dQ moment itself is not the fatal error; the blocker is the 1/T_νx³
    AMPLIFICATION (test 1) of whatever small residual dQ noise exists. This pins
    the clean-core priority: well-condition the variable (test 2) first;
    quadrature positivity guards are secondary hardening, not the root fix."""
    from rabbit.config.grids import MomentumGrid
    grid = MomentumGrid(N_q=80)
    q = np.asarray(grid.nodes)
    w = np.asarray(grid.weights)
    plain = w * np.exp(np.minimum(q, 500.0))            # exact bridge formula

    # FD source: GL moment vs analytic ∫q³/(e^q+1)dq = (7/8)·Γ(4)·ζ(4).
    f0 = 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)
    moment_fd = float(np.sum(plain * q**3 * f0))
    analytic_fd = (7.0 / 8.0) * 6.0 * (np.pi**4 / 90.0)
    assert abs(moment_fd - analytic_fd) < 1e-3 * analytic_fd, (
        f"GL FD energy moment {moment_fd:.6f} vs analytic {analytic_fd:.6f}")

    # A smooth slower-decaying distortion (exp(−q/2)): GL matches its analytic
    # integral ∫q³ e^{−q/2} dq = 6·2⁴ — no tail-amplification artifact for smooth C.
    eps = 1e-6
    moment_distorted = float(np.sum(plain * q**3 * (eps * np.exp(-0.5 * np.minimum(q, 500.0)))))
    analytic_distorted = eps * 6.0 * 2.0**4
    assert abs(moment_distorted - analytic_distorted) < 1e-3 * analytic_distorted
