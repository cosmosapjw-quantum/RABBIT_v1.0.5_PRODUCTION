"""tests/test_physical_collision_operator_reference.py — B4-PR1: numpy reference / parity oracle.

Pins the EXACT contract of the production numpy collision operator
(``rabbit.collisions.projected_operator.PhysicalCollisionOperator``), which the SciPy tier-2 driver
(``full_coupled_typeI.py``) uses as its incomplete-decoupling source. This is:

  (a) the first DEDICATED unit test of the operator's contract -- detailed balance, heating/cooling
      sign, the exact source/damping decomposition, the new-equilibrium null, the species coupling
      ratio, and the closed-form C_ell0 / C_ell2; and
  (b) the PARITY ORACLE for the forthcoming JAX twin (B4): the twin must reproduce ``evaluate`` to
      machine precision against the independent closed-form spec encoded here BEFORE collisions can
      be wired into the JAX characteristic transport path (removing the ``forward_likelihood.py``
      ValueError that currently blocks ``enable_collisions`` on the JAX characteristic backend).

The model (projected_operator.py docstring):
    C_0(q) = kappa0 * (Gamma_s/H) * [ Psi0_target(q) - Psi0(q) ],  kappa0 = 1
    C_2(q) = - kappa2 * (Gamma_s/H) * Psi2(q),                     kappa2 = 5/3
    Psi0_target(q) = (T_g - T_nu)/T_g * q * (1 - f0(q))            (0 when T_g == T_nu)
    Gamma_s/H per species scales linearly in the coupling a_s.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MomentumGrid
from rabbit.collisions.kernels import (
    A_TOTAL_NUE,
    A_TOTAL_NUX,
    COUPLING_RATIO_NUE_NUX,
)
from rabbit.collisions.projected_operator import (
    KAPPA_ELL0,
    KAPPA_ELL2,
    PhysicalCollisionOperator,
    collision_rate_per_efold,
)

_N_Q = 20
_H = 1.0e0  # H_inv_sec [s^-1]; fixed so the Gamma/H ratios are exercised deterministically


def _op(n_species: int = 3) -> PhysicalCollisionOperator:
    return PhysicalCollisionOperator(MomentumGrid(N_q=_N_Q), n_species=n_species)


def _f0(q: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


def _expected_target(op: PhysicalCollisionOperator, T_g: float, T_nu: float) -> np.ndarray:
    """Independent closed-form Psi0_target (the spec the JAX twin must match)."""
    q = op.grid.nodes
    if abs(T_g - T_nu) < 1e-15 * max(abs(T_g), abs(T_nu), 1e-30):
        return np.zeros_like(q)
    return (T_g - T_nu) / max(T_g, 1e-30) * q * (1.0 - _f0(q))


def _expected_C(op, Psi0, Psi2, T_g, T_nu, H):
    """Independent closed-form (C0, C2) from the documented formula -- NOT a call to evaluate()."""
    target = _expected_target(op, T_g, T_nu)
    ns = op.n_species
    C0 = np.zeros((ns, op.grid.N_q))
    C2 = np.zeros((ns, op.grid.N_q))
    for s in range(ns):
        gh = collision_rate_per_efold(T_g, T_nu, H, op.a_coupling[s])
        C0[s] = KAPPA_ELL0 * gh * target - KAPPA_ELL0 * gh * Psi0[s]
        C2[s] = -KAPPA_ELL2 * gh * Psi2[s]
    return C0, C2


# ── 1. The exact closed form (the strongest oracle pin) ──────────────────────────────

def test_evaluate_matches_closed_form_spec():
    """evaluate() reproduces the documented closed form for arbitrary Psi0/Psi2 to machine precision.
    This is THE spec the JAX twin must match -- it locks the kappa factors, the source/damping
    structure, and the per-species Gamma/H composition, not just smoke properties."""
    op = _op()
    rng = np.random.default_rng(0)
    Psi0 = rng.standard_normal((3, _N_Q)) * 0.01
    Psi2 = rng.standard_normal((3, _N_Q)) * 0.01
    T_g, T_nu = 2.0, 1.6
    r = op.evaluate(Psi0, Psi2, T_g, T_nu, _H)
    C0_exp, C2_exp = _expected_C(op, Psi0, Psi2, T_g, T_nu, _H)
    np.testing.assert_allclose(r.C_ell0, C0_exp, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.C_ell2, C2_exp, rtol=0, atol=1e-15)


@pytest.mark.parametrize("ns", [1, 2, 3])
def test_closed_form_holds_for_each_species_count(ns):
    """The closed form holds for n_species in {1,2,3} -- pins the per-species composition/indexing
    (the twin must honor whatever species count the driver passes, not just 3)."""
    op = _op(n_species=ns)
    rng = np.random.default_rng(10 + ns)
    Psi0 = rng.standard_normal((ns, _N_Q)) * 0.01
    Psi2 = rng.standard_normal((ns, _N_Q)) * 0.01
    r = op.evaluate(Psi0, Psi2, 2.0, 1.5, _H)
    C0_exp, C2_exp = _expected_C(op, Psi0, Psi2, 2.0, 1.5, _H)
    np.testing.assert_allclose(r.C_ell0, C0_exp, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.C_ell2, C2_exp, rtol=0, atol=1e-15)
    assert r.Gamma_over_H.shape == (ns,)


def test_source_damping_decomposition_is_exact():
    """source_ell0 + damping_ell0 == C_ell0 exactly; damping is the only Psi0-dependent part."""
    op = _op()
    rng = np.random.default_rng(1)
    Psi0 = rng.standard_normal((3, _N_Q)) * 0.02
    r = op.evaluate(Psi0, np.zeros((3, _N_Q)), 2.0, 1.5, _H)
    np.testing.assert_array_equal(r.source_ell0 + r.damping_ell0, r.C_ell0)
    # damping is exactly -kappa0 * Gamma/H * Psi0 per species
    for s in range(3):
        np.testing.assert_allclose(
            r.damping_ell0[s], -KAPPA_ELL0 * r.Gamma_over_H[s] * Psi0[s], rtol=0, atol=1e-15)


# ── 2. Detailed balance, heating/cooling sign ────────────────────────────────────────

def test_detailed_balance_zero_at_equal_temperature():
    """T_g == T_nu and Psi == 0 -> C0 == C2 == 0 exactly (thermal equilibrium)."""
    op = _op()
    z = np.zeros((3, _N_Q))
    r = op.evaluate(z, z, 1.0, 1.0, _H)
    assert np.max(np.abs(r.C_ell0)) == 0.0
    assert np.max(np.abs(r.C_ell2)) == 0.0
    assert r.total_energy_transfer == 0.0


def test_heating_when_photon_hotter():
    """T_g > T_nu with Psi == 0 GENERATES a positive source (incomplete-decoupling heating)."""
    op = _op()
    z = np.zeros((3, _N_Q))
    r = op.evaluate(z, z, 2.0, 1.5, _H)
    assert np.all(r.C_ell0 >= 0.0)        # source = kappa0 * Gamma/H * target, target >= 0
    assert np.max(r.C_ell0) > 0.0
    assert r.total_energy_transfer > 0.0  # net neutrino heating


def test_cooling_when_photon_colder():
    """T_g < T_nu reverses the sign: net neutrino cooling."""
    op = _op()
    z = np.zeros((3, _N_Q))
    r = op.evaluate(z, z, 1.5, 2.0, _H)
    assert np.all(r.C_ell0 <= 0.0)
    assert r.total_energy_transfer < 0.0


def test_new_equilibrium_null_at_target():
    """Psi0 == Psi0_target (the bath-temperature distribution) -> C0 == 0: source and damping cancel.
    A strong non-trivial null (docstring property 3) the JAX twin must also satisfy."""
    op = _op()
    T_g, T_nu = 2.0, 1.4
    target = _expected_target(op, T_g, T_nu)
    Psi0 = np.tile(target, (3, 1))
    r = op.evaluate(Psi0, np.zeros((3, _N_Q)), T_g, T_nu, _H)
    # The null is source - damping = kappaGamma*(target - target); residual is float roundoff at the
    # SCALE of the source it cancels (Gamma/H is large at H=1 s^-1), so test it RELATIVE to that scale.
    scale = max(float(np.max(np.abs(r.source_ell0))), 1e-300)
    assert np.max(np.abs(r.C_ell0)) < 1e-13 * scale


# ── 3. kappa2 / kappa0 ratio + species coupling ratio ────────────────────────────────

def test_quadrupole_damps_faster_by_kappa_ratio():
    """At equal temperature (no source) the quadrupole damps faster than the monopole by exactly
    kappa2/kappa0 = 5/3."""
    op = _op()
    rng = np.random.default_rng(2)
    Psi = rng.standard_normal((3, _N_Q)) * 0.01
    r = op.evaluate(Psi, Psi, 1.0, 1.0, _H)  # equal T -> C0 is pure damping
    for s in range(3):
        # C2 == (kappa2/kappa0) * C0 elementwise (multiplicative form -- no division, so no
        # divide-by-zero hazard if a random Psi entry lands near 0).
        np.testing.assert_allclose(
            r.C_ell2[s], (KAPPA_ELL2 / KAPPA_ELL0) * r.C_ell0[s], rtol=1e-12, atol=0)


def test_species_coupling_ratio():
    """Gamma/H scales linearly in the species coupling: nu_e / nu_x == COUPLING_RATIO_NUE_NUX, and
    the two nu_e rows are identical."""
    op = _op()
    z = np.zeros((3, _N_Q))
    r = op.evaluate(z, z, 2.0, 1.5, _H)
    assert r.Gamma_over_H[0] == r.Gamma_over_H[1]  # nu_e, nu_e-bar share A_TOTAL_NUE
    np.testing.assert_allclose(
        r.Gamma_over_H[0] / r.Gamma_over_H[2], A_TOTAL_NUE / A_TOTAL_NUX, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        A_TOTAL_NUE / A_TOTAL_NUX, COUPLING_RATIO_NUE_NUX, rtol=1e-9, atol=0)


# ── 4. Input-shape broadcast equivalence (the twin must honor both) ──────────────────

def test_1d_input_broadcasts_like_explicit_tile():
    """A 1-D (N_q,) Psi input is tiled across species -- identical to passing the explicit 2-D form."""
    op = _op()
    rng = np.random.default_rng(3)
    p0 = rng.standard_normal(_N_Q) * 0.01
    p2 = rng.standard_normal(_N_Q) * 0.01
    r1 = op.evaluate(p0, p2, 2.0, 1.5, _H)
    r2 = op.evaluate(np.tile(p0, (3, 1)), np.tile(p2, (3, 1)), 2.0, 1.5, _H)
    np.testing.assert_array_equal(r1.C_ell0, r2.C_ell0)
    np.testing.assert_array_equal(r1.C_ell2, r2.C_ell2)


def test_energy_transfer_is_quadrature_of_C0():
    """energy_transfer[s] == sum( w * e^q * q^3 * f0 * C0[s] ) -- pins the energy-weight quadrature
    the twin must reproduce for the plasma back-reaction coupling."""
    op = _op()
    z = np.zeros((3, _N_Q))
    r = op.evaluate(z, z, 2.0, 1.5, _H)
    q = op.grid.nodes
    w_energy = op.grid.weights * np.exp(q) * q**3 * _f0(q)
    for s in range(3):
        np.testing.assert_allclose(
            r.energy_transfer[s], np.sum(w_energy * r.C_ell0[s]), rtol=0, atol=1e-15)
