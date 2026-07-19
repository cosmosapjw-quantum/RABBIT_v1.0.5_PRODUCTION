"""Continuous q-advection helpers for the full phase-space ray driver.

The first landed tier-3 path keeps q-advection inside the continuous RHS so it
fits the existing Rodas5P event/error-control contract. Exact PCHIP remap is
retained as an oracle in tests and audits, not as the in-stage update.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _build_forward_diff_matrix_np(q_nodes: np.ndarray) -> np.ndarray:
    n_q = q_nodes.shape[0]
    diff = np.zeros((n_q, n_q), dtype=np.float64)
    for i in range(n_q - 1):
        h = q_nodes[i + 1] - q_nodes[i]
        diff[i, i] = -1.0 / h
        diff[i, i + 1] = 1.0 / h
    h_last = q_nodes[-1] - q_nodes[-2]
    # Negative q-velocity uses forward differencing; the high-q inflow state
    # is clamped to zero, so the ghost value vanishes.
    diff[-1, -1] = -1.0 / h_last
    return diff


def _build_backward_diff_matrix_np(q_nodes: np.ndarray) -> np.ndarray:
    n_q = q_nodes.shape[0]
    diff = np.zeros((n_q, n_q), dtype=np.float64)
    # Positive q-velocity uses backward differencing; the low-q inflow state
    # is clamped to f(q_min), so the one-sided derivative is zero.
    for i in range(1, n_q):
        h = q_nodes[i] - q_nodes[i - 1]
        diff[i, i - 1] = -1.0 / h
        diff[i, i] = 1.0 / h
    return diff


def _build_centered_diff_matrix_np(q_nodes: np.ndarray) -> np.ndarray:
    n_q = q_nodes.shape[0]
    diff = np.zeros((n_q, n_q), dtype=np.float64)

    for i in range(1, n_q - 1):
        h_m = q_nodes[i] - q_nodes[i - 1]
        h_p = q_nodes[i + 1] - q_nodes[i]
        diff[i, i - 1] = -h_p / (h_m * (h_m + h_p))
        diff[i, i] = (h_p - h_m) / (h_m * h_p)
        diff[i, i + 1] = h_m / (h_p * (h_m + h_p))

    h0 = q_nodes[1] - q_nodes[0]
    h1 = q_nodes[2] - q_nodes[1]
    diff[0, 0] = -(2.0 * h0 + h1) / (h0 * (h0 + h1))
    diff[0, 1] = (h0 + h1) / (h0 * h1)
    diff[0, 2] = -h0 / (h1 * (h0 + h1))

    hm = q_nodes[-2] - q_nodes[-3]
    hp = q_nodes[-1] - q_nodes[-2]
    diff[-1, -3] = hp / (hm * (hm + hp))
    diff[-1, -2] = -(hm + hp) / (hm * hp)
    diff[-1, -1] = (2.0 * hp + hm) / (hp * (hm + hp))
    return diff


@lru_cache(maxsize=16)
def cached_q_advection_operators(N_q: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return `(forward, backward, centered)` q-advection operators on Laguerre nodes.

    Each operator already includes the multiplicative `q` factor in `q df/dq`.
    """
    q_nodes_np, _ = np.polynomial.laguerre.laggauss(int(N_q))
    q_diag = np.diag(q_nodes_np.astype(np.float64))
    forward = q_diag @ _build_forward_diff_matrix_np(q_nodes_np)
    backward = q_diag @ _build_backward_diff_matrix_np(q_nodes_np)
    centered = q_diag @ _build_centered_diff_matrix_np(q_nodes_np)
    return (
        jnp.asarray(forward),
        jnp.asarray(backward),
        jnp.asarray(centered),
    )


@jax.jit
def _pchip_slopes(q_nodes: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Return monotone cubic slopes for the exact-remap oracle."""
    h = q_nodes[1:] - q_nodes[:-1]
    d = (f[1:] - f[:-1]) / jnp.maximum(h, 1e-300)
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    interior = jnp.where(
        (d[:-1] * d[1:]) > 0.0,
        (w1 + w2) / (w1 / d[:-1] + w2 / d[1:] + 1e-300),
        0.0,
    )
    return jnp.concatenate([d[:1], interior, d[-1:]])


@jax.jit
def _hermite_interp(
    q_nodes: jnp.ndarray,
    f: jnp.ndarray,
    slopes: jnp.ndarray,
    q_query: jnp.ndarray,
) -> jnp.ndarray:
    idx = jnp.clip(jnp.searchsorted(q_nodes, q_query, side="right") - 1, 0, q_nodes.shape[0] - 2)
    q0 = q_nodes[idx]
    q1 = q_nodes[idx + 1]
    f0 = f[idx]
    f1 = f[idx + 1]
    s0 = slopes[idx]
    s1 = slopes[idx + 1]
    h = q1 - q0
    t = (q_query - q0) / jnp.maximum(h, 1e-300)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * f0 + h10 * h * s0 + h01 * f1 + h11 * h * s1


@jax.jit
def semi_lagrangian_q_advect(
    f: jnp.ndarray,
    q_nodes: jnp.ndarray,
    dI: jnp.ndarray,
) -> jnp.ndarray:
    """Oracle exact-remap transport `f(q) -> f(q * exp(-2 dI))` via PCHIP."""
    q_query = q_nodes * jnp.exp(-2.0 * dI)
    slopes = _pchip_slopes(q_nodes, f)
    advected = _hermite_interp(q_nodes, f, slopes, q_query)
    advected = jnp.where(q_query > q_nodes[-1], 0.0, advected)
    advected = jnp.where(q_query < q_nodes[0], f[0], advected)
    return jnp.clip(advected, 0.0, 1.0)


@jax.jit
def apply_continuous_q_advection_jax(
    f_species_rays: jnp.ndarray,
    coeff_per_ray: jnp.ndarray,
    q_forward_op: jnp.ndarray,
    q_backward_op: jnp.ndarray,
) -> jnp.ndarray:
    """Return `df/dN = a_j q df/dq` for each `(species, ray, q)` block.

    `coeff_per_ray[j] = Sigma_plus * P2(mu_j)` in the LRS collisionless
    transport equation. Upwinding follows the sign of the characteristic
    velocity `v = -a_j q`.
    """

    def _apply_one_ray(coeff: jnp.ndarray, f_ray: jnp.ndarray) -> jnp.ndarray:
        # v = -coeff * q. Negative velocity -> forward stencil, positive -> backward.
        op = jnp.where(coeff >= 0.0, q_forward_op, q_backward_op)
        return coeff * (op @ f_ray)

    def _apply_one_species(f_rays: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_apply_one_ray)(coeff_per_ray, f_rays)

    return jax.vmap(_apply_one_species)(f_species_rays)


@jax.jit
def apply_centered_q_advection_jax(
    f_species_rays: jnp.ndarray,
    coeff_per_ray: jnp.ndarray,
    q_centered_op: jnp.ndarray,
) -> jnp.ndarray:
    """Diagnostic centered q-advection operator used in audits/tests."""

    def _apply_one_ray(coeff: jnp.ndarray, f_ray: jnp.ndarray) -> jnp.ndarray:
        return coeff * (q_centered_op @ f_ray)

    def _apply_one_species(f_rays: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(_apply_one_ray)(coeff_per_ray, f_rays)

    return jax.vmap(_apply_one_species)(f_species_rays)
