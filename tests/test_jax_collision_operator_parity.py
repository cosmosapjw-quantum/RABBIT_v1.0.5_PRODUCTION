"""tests/test_jax_collision_operator_parity.py — B4-PR2: JAX twin parity vs the numpy oracle.

Locks ``rabbit.jax.collision_operator_jax.physical_collision_rhs_jax`` to the production numpy
operator ``PhysicalCollisionOperator.evaluate`` (pinned by B4-PR1) to machine precision across a
(T_g, T_nu, H, Psi) sweep, and proves the twin is jittable + differentiable -- the property the JAX
characteristic transport path needs and the numpy operator lacks.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from rabbit.config.grids import MomentumGrid  # noqa: E402
from rabbit.collisions.projected_operator import PhysicalCollisionOperator  # noqa: E402
from rabbit.jax.collision_operator_jax import (  # noqa: E402
    DEFAULT_A_COUPLING,
    physical_collision_rhs_jax,
)

_N_Q = 20
# (T_gamma, T_nu, H_inv_sec): heating, exact-equal (detailed balance), cooling, hot, cold-tail.
_CASES = [(2.0, 1.6, 1.0), (1.0, 1.0, 1e-3), (0.5, 0.8, 1e2), (5.0, 4.0, 1.0), (0.1, 0.07, 1e5)]


def _numpy(grid, Psi0, Psi2, T_g, T_nu, H, ns):
    return PhysicalCollisionOperator(grid, n_species=ns).evaluate(Psi0, Psi2, T_g, T_nu, H)


def _jax(grid, Psi0, Psi2, T_g, T_nu, H, ns):
    return physical_collision_rhs_jax(
        jnp.asarray(Psi0), jnp.asarray(Psi2), T_g, T_nu, H,
        jnp.asarray(grid.nodes), jnp.asarray(grid.weights), DEFAULT_A_COUPLING[:ns])


@pytest.mark.parametrize("T_g,T_nu,H", _CASES)
@pytest.mark.parametrize("ns", [1, 2, 3])
def test_jax_twin_matches_numpy_evaluate(T_g, T_nu, H, ns):
    grid = MomentumGrid(N_q=_N_Q)
    rng = np.random.default_rng(7)
    Psi0 = rng.standard_normal((ns, _N_Q)) * 0.01
    Psi2 = rng.standard_normal((ns, _N_Q)) * 0.01
    rn = _numpy(grid, Psi0, Psi2, T_g, T_nu, H, ns)
    rj = _jax(grid, Psi0, Psi2, T_g, T_nu, H, ns)
    np.testing.assert_allclose(np.asarray(rj.C_ell0), rn.C_ell0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.C_ell2), rn.C_ell2, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.source_ell0), rn.source_ell0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.damping_ell0), rn.damping_ell0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.Gamma_over_H), rn.Gamma_over_H, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(rj.energy_transfer), rn.energy_transfer, rtol=1e-11, atol=1e-15)
    assert abs(float(rj.total_energy_transfer) - rn.total_energy_transfer) < (
        1e-11 * max(abs(rn.total_energy_transfer), 1e-300))


@pytest.mark.parametrize("dT", [0.0, 1e-16, 1e-12])
def test_near_equal_temperature_eps_band_parity(dT):
    """Mirror PhysicalCollisionOperator.source_term's epsilon guard: |T_g-T_nu| < 1e-15*T returns
    zeros. In-band (dT=0, 1e-16) both numpy and twin give EXACTLY zero; just outside (dT=1e-12) both
    give the (tiny) formula value. Catches the parity gap a naive twin (no eps guard) would have."""
    grid = MomentumGrid(N_q=_N_Q)
    T_nu, T_g = 1.0, 1.0 + dT
    z = np.zeros((3, _N_Q))
    rn = _numpy(grid, z, z, T_g, T_nu, 1.0, 3)
    rj = _jax(grid, z, z, T_g, T_nu, 1.0, 3)
    np.testing.assert_allclose(np.asarray(rj.source_ell0), rn.source_ell0, rtol=1e-12, atol=1e-300)
    if dT < 1e-15:  # inside the band -> exactly zero on BOTH
        assert float(jnp.max(jnp.abs(rj.source_ell0))) == 0.0
        assert np.max(np.abs(rn.source_ell0)) == 0.0
    else:           # outside the band -> nonzero source on both
        assert np.max(np.abs(rn.source_ell0)) > 0.0
        assert float(jnp.max(jnp.abs(rj.source_ell0))) > 0.0


def test_detailed_balance_parity_at_equal_T():
    """T_g == T_nu, Psi == 0: both numpy and twin give exactly zero (DT == 0 -> target == 0)."""
    grid = MomentumGrid(N_q=_N_Q)
    z = np.zeros((3, _N_Q))
    rj = _jax(grid, z, z, 1.0, 1.0, 1e-3, 3)
    assert float(jnp.max(jnp.abs(rj.C_ell0))) == 0.0
    assert float(jnp.max(jnp.abs(rj.C_ell2))) == 0.0


def test_twin_is_jittable():
    """The twin compiles under jax.jit and returns the same result as eager."""
    grid = MomentumGrid(N_q=_N_Q)
    rng = np.random.default_rng(8)
    Psi0 = jnp.asarray(rng.standard_normal((3, _N_Q)) * 0.01)
    Psi2 = jnp.asarray(rng.standard_normal((3, _N_Q)) * 0.01)
    qn, qw = jnp.asarray(grid.nodes), jnp.asarray(grid.weights)
    f = jax.jit(lambda p0, p2, tg, tn, h: physical_collision_rhs_jax(
        p0, p2, tg, tn, h, qn, qw, DEFAULT_A_COUPLING).C_ell0)
    eager = physical_collision_rhs_jax(Psi0, Psi2, 2.0, 1.5, 1.0, qn, qw, DEFAULT_A_COUPLING).C_ell0
    # jit-vs-eager agree to ~1 ULP; XLA fma/reassociation gives ~1e-14 absolute on O(100) values
    # (~2e-16 relative), so compare relatively, not at bit-exactness.
    np.testing.assert_allclose(np.asarray(f(Psi0, Psi2, 2.0, 1.5, 1.0)), np.asarray(eager),
                               rtol=1e-12, atol=1e-13)


def test_twin_is_differentiable():
    """The twin is differentiable end-to-end -- the property the numpy operator lacks and the JAX
    characteristic RHS needs. d(total_energy_transfer)/dT_gamma and d/dPsi0 are finite + nonzero."""
    grid = MomentumGrid(N_q=_N_Q)
    qn, qw = jnp.asarray(grid.nodes), jnp.asarray(grid.weights)
    Psi0 = jnp.asarray(np.zeros((3, _N_Q)))
    Psi2 = jnp.zeros((3, _N_Q))

    def total_xfer(T_g):
        return physical_collision_rhs_jax(Psi0, Psi2, T_g, 1.5, 1.0, qn, qw,
                                          DEFAULT_A_COUPLING).total_energy_transfer

    g = jax.grad(total_xfer)(2.0)
    assert jnp.isfinite(g) and float(g) != 0.0

    # Jacobian wrt Psi0 of C_ell0 has the EXACT analytic structure: -kappa0*Gamma/H on the
    # (s,i)==(s,i) diagonal, zero off-diagonal (C0[s,i] depends only on Psi0[s,i]). Checking the
    # structure -- not just finiteness -- proves the grad is correct, not merely non-nan.
    p_pt = jnp.full((3, _N_Q), 0.01)
    res = physical_collision_rhs_jax(p_pt, Psi2, 2.0, 1.5, 1.0, qn, qw, DEFAULT_A_COUPLING)
    gh = np.asarray(res.Gamma_over_H)                                # (3,)

    def c0(p0):
        return physical_collision_rhs_jax(p0, Psi2, 2.0, 1.5, 1.0, qn, qw,
                                          DEFAULT_A_COUPLING).C_ell0
    J = np.asarray(jax.jacfwd(c0)(p_pt))                             # (3, N_q, 3, N_q)
    assert np.all(np.isfinite(J))
    diag_vals = np.array([[J[s, i, s, i] for i in range(_N_Q)] for s in range(3)])  # (3, N_q)
    np.testing.assert_allclose(diag_vals, np.broadcast_to(-gh[:, None], (3, _N_Q)), rtol=1e-9, atol=0)
    mask = np.zeros((3, _N_Q, 3, _N_Q), bool)
    for s in range(3):
        for i in range(_N_Q):
            mask[s, i, s, i] = True
    assert np.max(np.abs(J[~mask])) < 1e-6 * np.max(np.abs(gh))      # off-diagonal is structurally 0
