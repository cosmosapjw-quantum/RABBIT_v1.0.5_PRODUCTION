"""tests/test_dynamic_collision_core.py — PR-C2: well-conditioned energy-variable 3T thermo.

Locks the clean-core fix for the q-Laguerre collision collapse (BD205): evolve the
neutrino bank ENERGY (dρ/dN = −4ρ + dQ) instead of its temperature (which divides by
c_v ∝ T³ → 0 at decoupling). Proves the energy form is the SAME physics as the
temperature form away from decoupling, but bounded where the temperature form diverges.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.thermo.nudec_coupled import (
    _drho_nu_pair_dT,
    coupled_3T_rhs_from_collision_moments,
)
from rabbit.collisions.dynamic_collision_core import (
    N_PAIRS_NUE,
    N_PAIRS_NUX,
    coupled_3T_energy_rhs,
    dnu_bank_energy_dN,
    neutrino_distribution_from_modes,
    nu_bank_T_from_energy,
    nu_bank_energy_from_T,
    spectral_energy_moment,
    spectral_number_moment,
)

_PLASMA = dict(plasma_dT_base_dN=0.0, plasma_drho_dT_MeV3=1.0)  # deterministic γ channel


@pytest.mark.parametrize("T", [10.0, 1.0, 0.1, 1e-3, 3.7624e-05])
@pytest.mark.parametrize("n_pairs", [N_PAIRS_NUE, N_PAIRS_NUX])
def test_energy_temperature_roundtrip(T, n_pairs):
    rho = nu_bank_energy_from_T(T, n_pairs)
    assert nu_bank_T_from_energy(rho, n_pairs) == pytest.approx(T, rel=1e-12)
    assert nu_bank_T_from_energy(0.0, n_pairs) == 0.0  # decoupling limit guarded


@pytest.mark.parametrize("T_nux", [1.0, 0.5, 0.1, 1e-2])
def test_energy_rhs_equals_temperature_form_away_from_decoupling(T_nux):
    """The ν energy channels are the exact chain-rule image of the temperature-form
    dT/dN (both finite away from decoupling), and the γ channel is untouched."""
    T_gamma, T_nue = 2.0, 1.5
    dQ_nue, dQ_nux = 3.0e-4, -5.0e-4
    rho_nue = nu_bank_energy_from_T(T_nue, N_PAIRS_NUE)
    rho_nux = nu_bank_energy_from_T(T_nux, N_PAIRS_NUX)

    dTg_t, dTnue_t, dTnux_t = coupled_3T_rhs_from_collision_moments(
        T_gamma, T_nue, T_nux, dQ_nue_pair_N=dQ_nue, dQ_nux_bank_N=dQ_nux, **_PLASMA)
    dTg_e, drho_nue_e, drho_nux_e = coupled_3T_energy_rhs(
        T_gamma, rho_nue, rho_nux, dQ_nue_pair_N=dQ_nue, dQ_nux_bank_N=dQ_nux, **_PLASMA)

    # γ channel identical
    assert dTg_e == pytest.approx(dTg_t, rel=1e-12, abs=1e-30)
    # ν channels: dρ/dN == (dρ_bank/dT)·(dT/dN) from the temperature form
    drho_nue_expect = N_PAIRS_NUE * _drho_nu_pair_dT(T_nue) * dTnue_t
    drho_nux_expect = N_PAIRS_NUX * _drho_nu_pair_dT(T_nux) * dTnux_t
    assert drho_nue_e == pytest.approx(drho_nue_expect, rel=1e-9, abs=1e-30)
    assert drho_nux_e == pytest.approx(drho_nux_expect, rel=1e-9, abs=1e-30)


def test_energy_rhs_is_bounded_where_temperature_form_diverges():
    """At the BD203 near-decoupled T_νx the temperature form blows up (O(1e6)); the
    energy form stays O(dQ) — the conditioning fix, on the production function."""
    T_gamma, T_nue, T_nux = 1.0, 1.0, 3.7624e-05
    dQ = -1.0e-6
    rho_nux = nu_bank_energy_from_T(T_nux, N_PAIRS_NUX)

    _, _, dTnux_t = coupled_3T_rhs_from_collision_moments(
        T_gamma, T_nue, T_nux, dQ_nue_pair_N=0.0, dQ_nux_bank_N=dQ, **_PLASMA)
    _, _, drho_nux_e = coupled_3T_energy_rhs(
        T_gamma, nu_bank_energy_from_T(T_nue, N_PAIRS_NUE), rho_nux,
        dQ_nue_pair_N=0.0, dQ_nux_bank_N=dQ, **_PLASMA)

    assert abs(dTnux_t) > 1.0e5                    # temperature form: catastrophic
    assert abs(drho_nux_e) < 1.0e-5                # energy form: bounded ~|dQ|
    assert np.isfinite(drho_nux_e)


def test_bank_energy_rhs_is_pure_linear_relaxation():
    """dρ/dN = −4ρ + dQ, independent of any vanishing heat capacity."""
    assert dnu_bank_energy_dN(2.0, 0.5) == pytest.approx(-4.0 * 2.0 + 0.5)
    # a tiny bank energy with a tiny source stays tiny (no amplification)
    assert abs(dnu_bank_energy_dN(1e-20, -1e-6)) < 1e-5


# ── PR-C3: A-mode logit distribution + spectral moments ──

def _grid(n_q=80):
    from rabbit.config.grids import MomentumGrid
    g = MomentumGrid(N_q=n_q)
    return np.asarray(g.nodes), np.asarray(g.weights)


def test_equilibrium_spectral_moments_match_analytic():
    """f₀ = 1/(e^q+1): ∫q³f₀dq = 7π⁴/120, ∫q²f₀dq = 3ζ(3)/2 (convention-free check
    of the q-Laguerre moment integrator)."""
    from scipy.special import zeta
    q, w = _grid(80)
    f0 = 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)
    assert spectral_energy_moment(f0, q, w) == pytest.approx(7.0 * np.pi**4 / 120.0, rel=1e-6)
    assert spectral_number_moment(f0, q, w) == pytest.approx(1.5 * float(zeta(3)), rel=1e-6)


def test_logit_distribution_is_positive_and_bounded_for_arbitrary_modes():
    """The logit form guarantees f∈(0,1) for ANY finite A-modes — the realizability
    the clean core relies on (no separate positivity clamp needed)."""
    q, _ = _grid(20)
    n_modes, n_angles = 3, 4
    # deterministic but nontrivial A-modes (no RNG in this env); large amplitudes too
    A = np.outer(np.array([2.0, -3.0, 1.5]), np.cos(q)) * 5.0    # (n_modes, n_q)
    basis = np.array([[1.0, 0.5, -0.5, -1.0],
                      [0.0, 1.0, -1.0, 0.0],
                      [-1.0, 0.3, 0.3, -1.0]])                   # (n_modes, n_angles)
    f = neutrino_distribution_from_modes(A, q, basis)
    assert f.shape == (q.size, n_angles)
    assert np.all(f > 0.0) and np.all(f < 1.0)
    assert np.all(np.isfinite(f))


def test_zero_modes_recover_equilibrium_fermi_dirac():
    """A-modes = 0 ⟹ logit = q ⟹ f = 1/(e^q+1) on every angle."""
    q, _ = _grid(20)
    n_modes, n_angles = 3, 4
    A = np.zeros((n_modes, q.size))
    basis = np.ones((n_modes, n_angles))
    f = neutrino_distribution_from_modes(A, q, basis)
    f0 = 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)
    for a in range(n_angles):
        np.testing.assert_allclose(f[:, a], f0, rtol=1e-12, atol=1e-15)


# ── PR-C4: first-principles collision energy source ──

def _collision_quadrature(n_q=24):
    from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature
    # evaluators require q_nodes == nue_y3_nodes == pair_y2_nodes (same laggauss order).
    return build_fixed_collision_quadrature(
        N_q=n_q, N_nue_y2=n_q, N_nue_y3=n_q, N_pair_y2=n_q, N_pair_leg=16)


def _fd(q, x=1.0):
    """Fermi-Dirac at temperature x·T_gamma, sampled in units of T_gamma (q = p/T_gamma)."""
    return 1.0 / (np.exp(np.minimum(q / x, 500.0)) + 1.0)


def test_collision_energy_transfer_vanishes_at_equilibrium():
    """Detailed balance: neutrinos and e± both Fermi-Dirac at T_gamma ⟹ the first-principles
    energy transfer and the statistical residual both vanish."""
    from rabbit.collisions.dynamic_collision_core import neutrino_collision_energy_transfer
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    f_eq = _fd(q, 1.0)
    dQ_eq, db_eq = neutrino_collision_energy_transfer(
        f_eq, f_eq, quadrature=quad, T_gamma_MeV=2.0, species="nue")

    # colder neutrinos (T_ν = 0.8 T_γ) → a nonzero heated reference to compare against
    f_cold = _fd(q, 0.8)
    dQ_hot, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, species="nue")

    assert db_eq < 1e-12, f"detailed-balance residual should vanish at equilibrium, got {db_eq:.2e}"
    assert abs(dQ_eq) < 1e-8 * abs(dQ_hot), (
        f"equilibrium energy transfer {dQ_eq:.3e} must be ≪ heated {dQ_hot:.3e}")


def test_collision_energy_transfer_heats_cold_neutrinos():
    """T_γ > T_ν ⟹ net energy flows plasma→ν (dQ > 0); a colder ν bank is heated harder."""
    from rabbit.collisions.dynamic_collision_core import neutrino_collision_energy_transfer
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    dQ_08, _ = neutrino_collision_energy_transfer(
        _fd(q, 0.8), _fd(q, 0.8), quadrature=quad, T_gamma_MeV=2.0, species="nue")
    dQ_05, _ = neutrino_collision_energy_transfer(
        _fd(q, 0.5), _fd(q, 0.5), quadrature=quad, T_gamma_MeV=2.0, species="nue")
    assert dQ_08 > 0.0 and dQ_05 > 0.0, "colder-than-plasma neutrinos must be heated (dQ>0)"
    assert dQ_05 > dQ_08, "a colder bank (larger T_γ−T_ν) is heated harder"


def test_nux_neutral_current_transfer_below_nue():
    """ν_x (neutral current only) couples more weakly than ν_e (charged+neutral), so at the
    same thermal offset its energy transfer is smaller."""
    from rabbit.collisions.dynamic_collision_core import neutrino_collision_energy_transfer
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    f_cold = _fd(q, 0.7)
    dQ_nue, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, species="nue")
    dQ_nux, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, species="nux")
    assert 0.0 < dQ_nux < dQ_nue


# ── PR-C5: multi-species bank assembly (dQ_nue_pair_N, dQ_nux_bank_N) ──

def test_bank_sources_vanish_at_equilibrium():
    from rabbit.collisions.dynamic_collision_core import collision_bank_energy_sources_per_efold
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    f_eq = _fd(q, 1.0)
    dQ_pair, dQ_bank = collision_bank_energy_sources_per_efold(
        f_eq, f_eq, quadrature=quad, T_gamma_MeV=2.0, H_MeV=1.0)
    # compare against a heated reference for a convention-free "≈0" check
    f_cold = _fd(q, 0.7)
    ref_pair, ref_bank = collision_bank_energy_sources_per_efold(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, H_MeV=1.0)
    assert abs(dQ_pair) < 1e-8 * abs(ref_pair)
    assert abs(dQ_bank) < 1e-8 * abs(ref_bank)


def test_bank_sources_heat_cold_neutrinos_and_scale_with_hubble():
    from rabbit.collisions.dynamic_collision_core import collision_bank_energy_sources_per_efold
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    f_cold = _fd(q, 0.7)
    dQ_pair, dQ_bank = collision_bank_energy_sources_per_efold(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, H_MeV=1.0)
    assert dQ_pair > 0.0 and dQ_bank > 0.0
    # per-e-fold source scales as 1/H
    dQ_pair_h2, _ = collision_bank_energy_sources_per_efold(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, H_MeV=2.0)
    assert dQ_pair_h2 == pytest.approx(0.5 * dQ_pair, rel=1e-12)


def test_nux_bank_applies_degeneracy_factor():
    """dQ_nux_bank_N == BANK_DEGENERACY[NUX] × (single ν_x transfer) / H (= 4× — 2 flavors × ν+ν̄)."""
    from rabbit.collisions.dynamic_collision_core import (
        collision_bank_energy_sources_per_efold, neutrino_collision_energy_transfer)
    from rabbit.collisions.species import BANK_DEGENERACY, Species
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    f_cold = _fd(q, 0.7)
    single_nux, _ = neutrino_collision_energy_transfer(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, species="nux")
    _, dQ_bank = collision_bank_energy_sources_per_efold(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=2.0, H_MeV=1.0)
    assert dQ_bank == pytest.approx(float(BANK_DEGENERACY[Species.NUX]) * single_nux, rel=1e-12)


def test_first_principles_bank_source_agrees_in_sign_with_calibrated_table():
    """Cross-model sanity: the first-principles ν_e-pair source heats ν when T_γ>T_ν, the same
    sign as the trusted calibrated table (nudec_coupled.total_energy_transfer). (Magnitudes are
    NOT compared — different models/normalizations.)"""
    from rabbit.collisions.dynamic_collision_core import collision_bank_energy_sources_per_efold
    from rabbit.thermo.nudec_coupled import total_energy_transfer
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    T_gamma, x = 2.0, 0.85
    f_cold = _fd(q, x)  # ν at T_ν = 0.85 T_γ
    dQ_pair, _ = collision_bank_energy_sources_per_efold(
        f_cold, f_cold, quadrature=quad, T_gamma_MeV=T_gamma, H_MeV=1.0)
    dQ_nue_table, _ = total_energy_transfer(T_gamma, x * T_gamma, x * T_gamma)
    assert np.sign(dQ_pair) == np.sign(dQ_nue_table) and dQ_pair != 0.0


# ── PR-C6: coupled FLRW dynamic-collision RHS (the driver kernel) ──

def _cold_modes(q, x):
    """Logit distortion A giving f = FD(q/x) (ν at T = x·T_γ). A>0 for x<1 (colder)."""
    return q * (1.0 / x - 1.0)


def test_flrw_rhs_equilibrium_is_steady_and_finite():
    """A = 0 (f = FD): the distribution RHS ≈ 0 (detailed balance) and dT_γ is finite —
    equilibrium is a (near) fixed point of the collision, plasma cools normally."""
    from rabbit.collisions.dynamic_collision_core import flrw_dynamic_collision_rhs
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    dA_nue, dA_nux, dT_gamma = flrw_dynamic_collision_rhs(
        np.zeros_like(q), np.zeros_like(q), 2.0, quadrature=quad)
    assert np.max(np.abs(dA_nue)) < 1e-6 and np.max(np.abs(dA_nux)) < 1e-6
    assert np.isfinite(dT_gamma)


def test_flrw_rhs_stays_finite_for_arbitrarily_cold_neutrinos():
    """The blocker DIVERGED as T_νx→0 (dT_νx/dN ∝ 1/T_νx³ → −4e6 and growing). The clean
    driver kernel stays FINITE for arbitrarily cold ν, because the distribution logit rate is
    bounded by the physical collision-to-Hubble ratio Γ/H (set by the plasma T_γ), NOT by how
    cold the neutrinos are. Large-but-finite values are genuine Γ/H stiffness (ν still coupled
    at 2 MeV) that a stiff solver handles — not a blow-up."""
    from rabbit.collisions.dynamic_collision_core import flrw_dynamic_collision_rhs
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    peaks = []
    for x in (0.5, 0.1, 0.01):                     # progressively colder ν
        A_cold = _cold_modes(q, x)
        dA_nue, dA_nux, dT_gamma = flrw_dynamic_collision_rhs(A_cold, A_cold, 2.0, quadrature=quad)
        assert np.all(np.isfinite(dA_nue)) and np.all(np.isfinite(dA_nux)) and np.isfinite(dT_gamma)
        peaks.append(max(np.max(np.abs(dA_nue)), np.max(np.abs(dA_nux))))
    # Does NOT diverge like 1/T³: a 50× colder ν (0.5→0.01) would blow the OLD form up by
    # ~1.25e5×; here the logit rate is bounded by Γ/H and grows far more mildly.
    assert peaks[-1] < 100.0 * peaks[0], f"logit rate must not diverge as ν cools; peaks={peaks}"


def test_flrw_rhs_heats_cold_neutrinos_toward_equilibrium():
    """A cold ν bank is driven back UP toward equilibrium: the induced dρ_ν/dN = ∫q³(df/dN) > 0
    (df/dN = −f(1−f)·dA)."""
    from rabbit.collisions.dynamic_collision_core import (
        flrw_dynamic_collision_rhs, spectral_energy_moment)
    from rabbit.transport.augmented_pstf_distribution import fermi_dirac_from_logit
    quad = _collision_quadrature()
    q = np.asarray(quad.q_nodes)
    qw = np.asarray(quad.q_weights)
    A_cold = _cold_modes(q, 0.7)
    dA_nue, _, _ = flrw_dynamic_collision_rhs(A_cold, A_cold, 2.0, quadrature=quad)
    f = fermi_dirac_from_logit(q + A_cold)
    df_dN = -f * (1.0 - f) * dA_nue           # d/dN of the occupancy
    drho_nue_dN = spectral_energy_moment(df_dN, q, qw)
    assert drho_nue_dN > 0.0, "cold neutrinos must gain energy (heated toward equilibrium)"
