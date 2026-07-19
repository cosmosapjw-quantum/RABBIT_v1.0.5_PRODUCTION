"""rabbit.jax.teff_collision_bridge_jax — JAX twin of the gather-scatter collision bridge (B4).

The JAX characteristic transport state is RAYS-ONLY (I_j, J_j per ray), with no moment hierarchy
slice, so collisions cannot be applied as a Psi0/Psi2 RHS term. The SciPy characteristic path
(full_coupled_typeI.py) instead routes collisions through a GATHER-SCATTER bridge
(transport/teff_collision_bridge.apply_gather_scatter_collision):

    1. GATHER : rays -> per-ray Theta_j = exp(-2 I_j) -> angle-averaged monopole f_mono(q)
    2. COLLIDE: Psi0 = (f_mono - f_eq)/f_eq -> PhysicalCollisionOperator.evaluate(Psi0, 0, ...)
    3. SCATTER: isotropic energy conservation -> delta_I_j = -(delta_rho_nu/(8 rho_ref)) per ray

This module is the differentiable, jittable JAX twin of that bridge, parity-locked to the numpy
version (tests/test_jax_teff_collision_bridge_parity.py). The COLLIDE step reuses the parity-locked
operator twin ``physical_collision_rhs_jax`` (B4-PR2). It is the prerequisite for wiring collisions
into ``_rhs_core`` (and removing the ``forward_likelihood.py`` ValueError) -- that live wiring is a
separate PR; this module only builds + locks the bridge, it does not touch the driver.

Constants/grid weights are reproduced verbatim from the numpy bridge so parity is exact. The numpy
bridge reads ``RABBIT_COLLISION_QMAX`` (default 80) and ``RABBIT_COLLISION_BRIDGE_RELAX`` (default 1)
from the environment per call; here they are explicit static args (read once by the caller).

PARITY IS NOT PHYSICAL VALIDATION (external audit BD601, 2026-06-30). The collide step routes through
the CALIBRATED RTA model (``physical_collision_rhs_jax`` / ``PhysicalCollisionOperator``; ``_C_RATE``
calibrated to ``N_eff ~ 3.044``, linearized source, momentum-independent ``Gamma``, no nu-nu), and the
scatter is an ISOTROPIC Born-level energy map (``delta_I = -delta_rho/(8 rho_ref)``). Machine-precision
agreement with the numpy bridge is ENGINEERING fidelity, **NOT a physical validation of the collision
physics**.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rabbit.jax.collision_operator_jax import DEFAULT_A_COUPLING, physical_collision_rhs_jax

#: Defaults matching transport.teff_collision_bridge env vars.
DEFAULT_Q_CAP = 80.0
DEFAULT_RELAX = 1.0


class JAXGatherScatterResult(NamedTuple):
    """Mirror of transport.teff_collision_bridge.GatherScatterResult (all arrays jnp)."""
    delta_I: jnp.ndarray         # (N_mu,) correction to ray energy-shift integrals
    delta_rho_nu: jnp.ndarray    # scalar net energy transfer to the nu sector
    C_monopole: jnp.ndarray      # (N_q,) summed monopole collision integral
    theta_per_ray: jnp.ndarray   # (N_mu,) per-ray effective temperatures
    f_monopole: jnp.ndarray      # (N_q,) angle-averaged monopole
    tangency_D2: jnp.ndarray     # scalar diagnostic (degenerate; ~0)


def _f0_jax(q: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (jnp.exp(jnp.minimum(q, 500.0)) + 1.0)


def gather_monopole_jax(I, J, w0, q_nodes, theta=None):
    """Angle-average the boosted-FD rays to the monopole: f_mono(q) = 1/2 * sum_j w_j J_j f(q/Theta_j).
    Mirrors transport.teff_collision_bridge.gather_monopole."""
    if theta is None:
        theta = jnp.exp(-2.0 * I)
    qa = q_nodes[:, None] / theta[None, :]                 # (N_q, N_mu)
    f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500.0)) + 1.0)
    return 0.5 * f_vals @ (w0 * J)


def _tangency_D2_jax(I, J, w0, q_nodes):
    """Reproduce tangency_diagnostic (degenerate: its P2 term is the constant -0.75, so D2 ~ 0).
    Diagnostic only -- not used in the scatter RHS; reproduced for faithful parity."""
    theta = jnp.exp(-2.0 * I)
    theta_mean = 0.5 * jnp.sum(w0 * J * theta)
    theta_var = 0.5 * jnp.sum(w0 * J * (theta - theta_mean) ** 2)
    T2 = theta_var / jnp.maximum(theta_mean ** 2, 1e-30)
    # m=0 Legendre term: 0.5*(3*0-1) = -0.5; the numpy expression collapses to the constant -0.75.
    P2 = 0.5 * (3.0 * (0.5 * jnp.sum(w0 * J * (theta / theta_mean) * (-0.5))))
    D2 = jnp.maximum(T2 - P2 ** 2, 0.0)
    return jnp.sqrt(jnp.maximum(D2, 0.0))


def apply_gather_scatter_collision_jax(
    I, J, w0, q_nodes, q_weights, T_gamma, T_nu, H_inv_sec,
    a_coupling=DEFAULT_A_COUPLING, q_cap: float = DEFAULT_Q_CAP, relax: float = DEFAULT_RELAX,
    compute_tangency: bool = True,
) -> JAXGatherScatterResult:
    """JAX twin of ``apply_gather_scatter_collision``.

    Parameters mirror the numpy version. ``q_cap`` / ``relax`` are the
    ``RABBIT_COLLISION_QMAX`` / ``RABBIT_COLLISION_BRIDGE_RELAX`` knobs (static).
    """
    n_q = q_nodes.shape[0]
    f0 = _f0_jax(q_nodes)

    # ── 1. Gather ──
    theta = jnp.exp(-2.0 * I)
    f_mono = gather_monopole_jax(I, J, w0, q_nodes, theta=theta)

    # ── 2. Tangency (diagnostic only) ──
    D2 = _tangency_D2_jax(I, J, w0, q_nodes) if compute_tangency else jnp.asarray(0.0)

    # ── 3. Collide (3-species isotropic; reuse the parity-locked operator twin) ──
    Psi0 = (f_mono - f0) / jnp.maximum(f0, 1e-30)
    n_sp = a_coupling.shape[0]
    Psi0_sp = jnp.broadcast_to(Psi0, (n_sp, n_q))
    Psi2_sp = jnp.zeros((n_sp, n_q))
    coll = physical_collision_rhs_jax(
        Psi0_sp, Psi2_sp, T_gamma, T_nu, H_inv_sec, q_nodes, q_weights, a_coupling)
    C_mono = jnp.sum(coll.C_ell0, axis=0)                  # (N_q,)

    # ── 4. Scatter (isotropic energy conservation; mirror scatter_collision_to_rays) ──
    # delta_rho_weight uses the UNcapped exp; bridge_weight caps the exp at q_cap (and drops f0),
    # exactly as transport.teff_collision_bridge does.
    delta_rho_weight = q_weights * jnp.exp(q_nodes) * q_nodes ** 3 * f0
    bridge_weight = jnp.nan_to_num(
        q_weights * jnp.exp(jnp.minimum(q_nodes, q_cap)) * q_nodes ** 3,
        nan=0.0, posinf=0.0, neginf=0.0)
    rho_ref = jnp.sum(bridge_weight * f0)
    # BD612-F6: propagate a non-finite runtime collision field (parity with the numpy
    # bridge, which no longer nan_to_num's C_monopole/delta_I) so a bad state fails loud
    # instead of being silently zeroed.
    delta_rho = jnp.sum(delta_rho_weight * C_mono)
    frac = delta_rho / jnp.maximum(jnp.abs(rho_ref), 1e-30)
    delta_I = jnp.full_like(I, -frac / 8.0)
    # Mirror scatter_collision_to_rays' guard ORDER exactly: (a) numpy early-returns zeros when
    # rho_ref (a deterministic quadrature CONSTANT) is non-finite or <= 1e-300 — a construction
    # guard, kept; (b) it scales by relax LAST. A non-finite delta_I from a bad C now PROPAGATES
    # in both twins (no delta_I nan_to_num), so the two stay parity-identical on all inputs.
    bad_ref = jnp.logical_or(jnp.logical_not(jnp.isfinite(rho_ref)), rho_ref <= 1.0e-300)
    delta_I = jnp.where(bad_ref, jnp.zeros_like(I), delta_I)
    delta_I = relax * delta_I

    return JAXGatherScatterResult(
        delta_I=delta_I,
        delta_rho_nu=delta_rho,
        C_monopole=C_mono,
        theta_per_ray=theta,
        f_monopole=f_mono,
        tangency_D2=D2,
    )
