"""tests/test_jax_teff_collision_bridge_parity.py — B4-PR3: JAX gather-scatter bridge parity.

Locks ``rabbit.jax.teff_collision_bridge_jax.apply_gather_scatter_collision_jax`` to the numpy
bridge ``transport.teff_collision_bridge.apply_gather_scatter_collision`` to machine precision, and
proves the bridge (gather -> collide -> scatter) is jittable + differentiable end-to-end -- the
property the JAX characteristic RHS needs before collisions can be wired in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from rabbit.config.grids import MomentumGrid  # noqa: E402
from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision  # noqa: E402
from rabbit.jax.teff_collision_bridge_jax import (  # noqa: E402
    apply_gather_scatter_collision_jax,
)

_N_Q, _N_MU = 20, 8
_CASES = [(2.0, 1.6, 1.0), (1.0, 1.0, 1e-3), (0.5, 0.8, 1e2), (5.0, 4.0, 1.0)]


@pytest.fixture
def rays():
    """A physically-plausible ray state: small shear-like energy shift + near-unit Jacobian."""
    mu, w0 = np.polynomial.legendre.leggauss(_N_MU)
    I = 0.05 * mu
    J = 1.0 + 0.1 * mu
    grid = MomentumGrid(N_q=_N_Q)
    return I, J, w0, grid.nodes, grid.weights


@pytest.fixture(autouse=True)
def _default_bridge_env(monkeypatch):
    """The numpy bridge reads RABBIT_COLLISION_QMAX/RELAX from the env; force the defaults (80, 1.0)
    so it matches the twin's default q_cap/relax for an apples-to-apples parity comparison."""
    monkeypatch.delenv("RABBIT_COLLISION_QMAX", raising=False)
    monkeypatch.delenv("RABBIT_COLLISION_BRIDGE_RELAX", raising=False)


@pytest.mark.parametrize("T_g,T_nu,H", _CASES)
def test_bridge_parity_at_production_Nq80_exercises_q_cap_path(T_g, T_nu, H):
    """BD601 P3: the standalone parity test runs only at N_q=20 (max Gauss-Laguerre
    node ~66 < q_cap=80), so numpy↔jax parity at the PRODUCTION grid N_q=80 (the
    resolution the live driver uses) was uncovered, and the ``exp(min(q, q_cap))``
    clamp arithmetic never ran. This locks parity at N_q=80, where the nodes reach
    ~297 so the clamp branch is exercised for 31 nodes.

    HONESTY NOTE (measured): at N_q=80 the clamp is numerically a NO-OP — the
    ``f0 ~ exp(-q)`` factor in ρ_ref suppresses the high-q tail, so ρ_ref is
    bit-identical clamped vs un-clamped (ratio 1.0). The clamp is a finite-ness
    guard that would only bite on grids reaching ``q ≳ 709`` (and even there
    ``nan_to_num`` + the f0 suppression make it largely redundant). So this test's
    value is PRODUCTION-RESOLUTION parity coverage + clamp-path coverage, NOT a
    demonstration that the clamp is load-bearing."""
    n_q, n_mu, q_cap = 80, 8, 80.0
    mu, w0 = np.polynomial.legendre.leggauss(n_mu)
    I = 0.05 * mu
    J = 1.0 + 0.1 * mu
    grid = MomentumGrid(N_q=n_q)
    qn, qw = np.asarray(grid.nodes), np.asarray(grid.weights)

    # The grid must reach past q_cap so the clamp arithmetic actually runs.
    assert (qn > q_cap).sum() > 0, "N_q=80 grid must extend past q_cap to exercise the clamp path"

    rn = apply_gather_scatter_collision(I, J, w0, qn, qw, T_g, T_nu, H, compute_tangency=True)
    rj = apply_gather_scatter_collision_jax(
        jnp.asarray(I), jnp.asarray(J), jnp.asarray(w0), jnp.asarray(qn), jnp.asarray(qw),
        T_g, T_nu, H)
    np.testing.assert_allclose(np.asarray(rj.f_monopole), rn.f_monopole, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.C_monopole), rn.C_monopole, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.delta_I), rn.delta_I, rtol=1e-10, atol=1e-18)
    assert abs(float(rj.delta_rho_nu) - rn.delta_rho_nu) < 1e-10 * max(abs(rn.delta_rho_nu), 1e-300)
    assert np.all(np.isfinite(np.asarray(rj.delta_I)))


@pytest.mark.parametrize("T_g,T_nu,H", _CASES)
def test_bridge_twin_matches_numpy(rays, T_g, T_nu, H):
    I, J, w0, qn, qw = rays
    rn = apply_gather_scatter_collision(I, J, w0, qn, qw, T_g, T_nu, H, compute_tangency=True)
    rj = apply_gather_scatter_collision_jax(
        jnp.asarray(I), jnp.asarray(J), jnp.asarray(w0), jnp.asarray(qn), jnp.asarray(qw),
        T_g, T_nu, H)
    np.testing.assert_allclose(np.asarray(rj.f_monopole), rn.f_monopole, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.theta_per_ray), rn.theta_per_ray, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(rj.C_monopole), rn.C_monopole, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rj.delta_I), rn.delta_I, rtol=1e-10, atol=1e-18)
    assert abs(float(rj.delta_rho_nu) - rn.delta_rho_nu) < 1e-10 * max(abs(rn.delta_rho_nu), 1e-300)
    np.testing.assert_allclose(np.asarray(rj.tangency_D2), rn.tangency_D2, rtol=1e-9, atol=1e-12)
    assert bool(jnp.all(jnp.isfinite(rj.delta_I)))  # the scatter guards keep delta_I finite


@pytest.mark.parametrize("r", [0.5, 2.0])
def test_relax_scales_delta_I_linearly(rays, r):
    """relax (RABBIT_COLLISION_BRIDGE_RELAX) is applied LAST as a linear scale on delta_I -- so
    twin(relax=r) == r * twin(relax=1). Locks the relax-last ordering that mirrors the numpy bridge."""
    I, J, w0, qn, qw = (jnp.asarray(a) for a in rays)
    base = apply_gather_scatter_collision_jax(I, J, w0, qn, qw, 2.0, 1.5, 1.0, relax=1.0).delta_I
    scaled = apply_gather_scatter_collision_jax(I, J, w0, qn, qw, 2.0, 1.5, 1.0, relax=r).delta_I
    np.testing.assert_allclose(np.asarray(scaled), r * np.asarray(base), rtol=1e-12, atol=1e-18)


def test_bridge_detailed_balance(rays):
    """Isotropic rays (I==0 -> f_mono==f_eq) AND T_g==T_nu -> no source, no damping -> delta_I==0."""
    _, _, w0, qn, qw = rays
    I0 = jnp.zeros(_N_MU)
    J1 = jnp.ones(_N_MU)
    rj = apply_gather_scatter_collision_jax(I0, J1, jnp.asarray(w0), jnp.asarray(qn),
                                            jnp.asarray(qw), 1.0, 1.0, 1e-3)
    assert float(jnp.max(jnp.abs(rj.C_monopole))) < 1e-12
    assert float(jnp.max(jnp.abs(rj.delta_I))) < 1e-14


def test_bridge_heating_sign(rays):
    """T_g > T_nu drives net neutrino heating: delta_rho_nu > 0 (regardless of ray state)."""
    I, J, w0, qn, qw = rays
    rj = apply_gather_scatter_collision_jax(
        jnp.asarray(I), jnp.asarray(J), jnp.asarray(w0), jnp.asarray(qn), jnp.asarray(qw),
        2.0, 1.5, 1.0)
    assert float(rj.delta_rho_nu) > 0.0


def test_bridge_is_jittable(rays):
    I, J, w0, qn, qw = rays
    Ij, Jj, w0j, qnj, qwj = map(jnp.asarray, (I, J, w0, qn, qw))
    f = jax.jit(lambda i, jj, tg, tn, h: apply_gather_scatter_collision_jax(
        i, jj, w0j, qnj, qwj, tg, tn, h).delta_I)
    eager = apply_gather_scatter_collision_jax(Ij, Jj, w0j, qnj, qwj, 2.0, 1.5, 1.0).delta_I
    np.testing.assert_allclose(np.asarray(f(Ij, Jj, 2.0, 1.5, 1.0)), np.asarray(eager),
                               rtol=1e-12, atol=1e-15)


def test_bridge_is_differentiable(rays):
    """End-to-end differentiable through gather->collide->scatter: d(delta_rho_nu)/dT_gamma and the
    Jacobian of delta_I wrt the ray energy shifts I are finite (and the T-grad is nonzero)."""
    I, J, w0, qn, qw = rays
    Jj, w0j, qnj, qwj = map(jnp.asarray, (J, w0, qn, qw))

    def drho(T_g):
        return apply_gather_scatter_collision_jax(
            jnp.asarray(I), Jj, w0j, qnj, qwj, T_g, 1.5, 1.0).delta_rho_nu
    g = jax.grad(drho)(2.0)
    assert jnp.isfinite(g) and float(g) != 0.0

    def dI(Iv):
        return apply_gather_scatter_collision_jax(
            Iv, Jj, w0j, qnj, qwj, 2.0, 1.5, 1.0).delta_I
    Jac = jax.jacfwd(dI)(jnp.asarray(I))
    assert bool(jnp.all(jnp.isfinite(Jac)))
