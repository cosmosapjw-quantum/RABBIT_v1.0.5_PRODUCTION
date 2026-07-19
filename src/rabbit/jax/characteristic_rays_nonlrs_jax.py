"""JAX-compatible non-LRS characteristic-ray primitives on S^2.

This module provides the angular quadrature and direction-map building blocks
used by the opt-in generic-Type-I characteristic driver
``transport_mode="characteristic_nonlrs"``.  The surface is candidate scope:
the primitives are exact for orthogonal diagonal Type I, while driver-level
promotion is controlled by the backend capability registry.

The conventions match the existing LRS characteristic helpers:

- the polar quadrature is Gauss-Legendre in ``mu = cos(theta)``;
- the azimuthal average is a uniform midpoint rule in ``phi``;
- the returned ``w_s2`` weights sum to ``2`` so the existing
  ``0.5 * (...)`` monopole convention remains valid in the LRS reduction.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)

_SQRT3 = float(np.sqrt(3.0))
_TWOPI = float(2.0 * np.pi)


@lru_cache(maxsize=8)
def setup_ray_grid_S2(N_theta: int, N_phi: int):
    """Build a tensor-product S^2 quadrature grid.

    Parameters
    ----------
    N_theta, N_phi
        Number of polar and azimuthal nodes. ``N_theta`` uses a
        Gauss-Legendre rule in ``mu = cos(theta)``; ``N_phi`` uses the
        midpoint rule on ``[0, 2pi)``.

    Returns
    -------
    tuple of JAX arrays
        ``(mu0, phi0, w_s2, X0, signs)`` flattened over the full tensor grid.
        ``w_s2`` follows the LRS average convention and therefore sums to ``2``.
    """
    n_theta = int(N_theta)
    n_phi = int(N_phi)
    if n_theta < 2:
        raise ValueError(f"N_theta must be >= 2, got {N_theta!r}.")
    if n_phi < 1:
        raise ValueError(f"N_phi must be >= 1, got {N_phi!r}.")

    mu0_1d, w_mu_1d = leggauss(n_theta)
    phi0_1d = (np.arange(n_phi, dtype=np.float64) + 0.5) * (_TWOPI / float(n_phi))

    mu0 = np.broadcast_to(mu0_1d[:, None], (n_theta, n_phi)).reshape(-1)
    phi0 = np.broadcast_to(phi0_1d[None, :], (n_theta, n_phi)).reshape(-1)
    w_s2 = np.broadcast_to(
        w_mu_1d[:, None] * (1.0 / float(n_phi)),
        (n_theta, n_phi),
    ).reshape(-1)
    X0 = mu0**2 / np.maximum(1.0 - mu0**2, 1e-30)
    signs = np.where(mu0 >= 0.0, 1.0, -1.0)
    return (
        jnp.asarray(mu0, dtype=jnp.float64),
        jnp.asarray(phi0, dtype=jnp.float64),
        jnp.asarray(w_s2, dtype=jnp.float64),
        jnp.asarray(X0, dtype=jnp.float64),
        jnp.asarray(signs, dtype=jnp.float64),
    )


@jax.jit
def _direction_components_and_log_norm_jax(
    mu0: jnp.ndarray,
    phi0: jnp.ndarray,
    S_plus: jnp.ndarray,
    S_minus: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the current direction vector under diagonal Bianchi-I shear.

    In the orthogonal Type-I Wainwright-Hsu basis the integrated shear
    generator is diagonal, so the path-ordered exponential collapses to
    axis-wise stretching:

    ``diag(exp(-(S_+ + sqrt(3) S_-)),
           exp(-(S_+ - sqrt(3) S_-)),
           exp( 2 S_+))``.

    Renormalising the stretched vector gives the exact null-direction map.
    """
    sin_theta0 = jnp.sqrt(jnp.maximum(1.0 - mu0 * mu0, 0.0))
    ex0 = sin_theta0 * jnp.cos(phi0)
    ey0 = sin_theta0 * jnp.sin(phi0)
    ez0 = mu0

    log_abs_ex0 = jnp.where(jnp.abs(ex0) > 0.0, jnp.log(jnp.abs(ex0)), -jnp.inf)
    log_abs_ey0 = jnp.where(jnp.abs(ey0) > 0.0, jnp.log(jnp.abs(ey0)), -jnp.inf)
    log_abs_ez0 = jnp.where(jnp.abs(ez0) > 0.0, jnp.log(jnp.abs(ez0)), -jnp.inf)

    log_scale_x = -(S_plus + _SQRT3 * S_minus)
    log_scale_y = -(S_plus - _SQRT3 * S_minus)
    log_scale_z = 2.0 * S_plus

    log_norm_sq = jnp.logaddexp(
        2.0 * log_abs_ex0 + 2.0 * log_scale_x,
        jnp.logaddexp(
            2.0 * log_abs_ey0 + 2.0 * log_scale_y,
            2.0 * log_abs_ez0 + 2.0 * log_scale_z,
        ),
    )
    log_norm = 0.5 * log_norm_sq

    ex = jnp.sign(ex0) * jnp.exp(log_abs_ex0 + log_scale_x - log_norm)
    ey = jnp.sign(ey0) * jnp.exp(log_abs_ey0 + log_scale_y - log_norm)
    ez = jnp.sign(ez0) * jnp.exp(log_abs_ez0 + log_scale_z - log_norm)
    return ex, ey, ez, log_norm


@jax.jit
def _direction_components_current_jax(
    mu0: jnp.ndarray,
    phi0: jnp.ndarray,
    S_plus: jnp.ndarray,
    S_minus: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ex, ey, ez, _log_norm = _direction_components_and_log_norm_jax(
        mu0, phi0, S_plus, S_minus
    )
    return ex, ey, ez


@jax.jit
def mu_phi_current_jax(
    mu0: jnp.ndarray,
    phi0: jnp.ndarray,
    S_plus: jnp.ndarray,
    S_minus: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward map ``(mu0, phi0) -> (mu, phi)`` for generic Type I.

    The map is exact for orthogonal Type I because the shear generator stays
    diagonal in the fixed Wainwright-Hsu frame, so the integrated stretch
    operators commute.
    """
    ex, ey, ez = _direction_components_current_jax(mu0, phi0, S_plus, S_minus)
    return ez, jnp.arctan2(ey, ex)


@jax.jit
def intensity_shift_nonlrs_jax(
    mu0: jnp.ndarray,
    phi0: jnp.ndarray,
    S_plus: jnp.ndarray,
    S_minus: jnp.ndarray,
) -> jnp.ndarray:
    """Closed-form characteristic energy shift for generic orthogonal Type I.

    If ``A`` is the diagonal shear-stretch map acting on the initial direction
    ``e0`` and ``norm = ||A e0||``, then the transported FD argument rescales as
    ``q -> q * norm`` in the project convention. The characteristic intensity
    shift therefore satisfies ``exp(2 I) = norm``.
    """
    _ex, _ey, _ez, log_norm = _direction_components_and_log_norm_jax(
        mu0, phi0, S_plus, S_minus
    )
    return 0.5 * log_norm


@jax.jit
def jacobian_nonlrs_jax(
    mu0: jnp.ndarray,
    phi0: jnp.ndarray,
    S_plus: jnp.ndarray,
    S_minus: jnp.ndarray,
) -> jnp.ndarray:
    """Exact S² area Jacobian for the non-LRS characteristic forward map.

    For the unit-sphere map ``e = A e0 / ||A e0||`` with positive diagonal
    stretch matrix ``A``, the push-forward of the solid-angle measure is

    ``J = det(A) / ||A e0||^3``.

    The orthogonal Type-I shear stretches are trace-free, so
    ``det(A) = exp(-(Sx + Sy + Sz)) = 1`` and the Jacobian collapses to
    ``J = ||A e0||^{-3}``.
    """
    _ex, _ey, _ez, log_norm = _direction_components_and_log_norm_jax(
        mu0, phi0, S_plus, S_minus
    )
    return jnp.exp(-3.0 * log_norm)


@jax.jit
def stress_plus_kernel_S2_jax(mu: jnp.ndarray) -> jnp.ndarray:
    """Angular kernel for ``Pi_+`` on the unit sphere."""
    return 0.5 * (3.0 * mu * mu - 1.0)


@jax.jit
def stress_minus_kernel_S2_jax(mu: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Real ``|m| = 2`` quadrupole kernel in the project convention."""
    return -(0.5 * _SQRT3) * (1.0 - mu * mu) * jnp.cos(2.0 * phi)


@jax.jit
def extract_stress_plus_S2(
    I: jnp.ndarray,
    J: jnp.ndarray,
    mu: jnp.ndarray,
    w_s2: jnp.ndarray,
    f_nu: float,
) -> jnp.ndarray:
    """S^2-weighted ``Pi_+`` extractor in the LRS-compatible convention."""
    kernel = stress_plus_kernel_S2_jax(mu)
    return f_nu * jnp.sum(w_s2 * J * kernel * jnp.exp(-8.0 * I))


@jax.jit
def extract_stress_minus_S2(
    I: jnp.ndarray,
    J: jnp.ndarray,
    mu: jnp.ndarray,
    phi: jnp.ndarray,
    w_s2: jnp.ndarray,
    f_nu: float,
) -> jnp.ndarray:
    """S^2-weighted ``Pi_-`` extractor in the real ``Y_2^{±2}`` convention."""
    kernel = stress_minus_kernel_S2_jax(mu, phi)
    return f_nu * jnp.sum(w_s2 * J * kernel * jnp.exp(-8.0 * I))


@jax.jit
def extract_monopole_S2(
    I: jnp.ndarray,
    J: jnp.ndarray,
    w_s2: jnp.ndarray,
    q_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """Angle-averaged transported monopole on the S^2 quadrature."""
    alpha = jnp.exp(2.0 * I)
    qa = q_nodes[:, None] * alpha[None, :]
    f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500.0)) + 1.0)
    return 0.5 * f_vals @ (w_s2 * J)
