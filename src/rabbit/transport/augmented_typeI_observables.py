"""Observable projections for augmented Type I neutrino distributions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentedStressMoments:
    """Energy-weighted angular moments from an augmented distribution."""

    rho_weighted: np.ndarray
    monopole_q: np.ndarray
    pi_plus_tilde: np.ndarray
    pi_minus_tilde: np.ndarray | None = None


def lrs_quadrupole_kernel(mu_nodes: np.ndarray) -> np.ndarray:
    """Return the LRS `Sigma_+` quadrupole kernel `(3 mu^2 - 1) / 2`."""

    mu = np.asarray(mu_nodes, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu_nodes must contain only finite values.")
    return 0.5 * (3.0 * mu**2 - 1.0)


def non_lrs_quadrupole_kernels(
    mu_nodes: np.ndarray,
    phi_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(W_plus, W_minus)` kernels on an S2 angular grid."""

    mu = np.asarray(mu_nodes, dtype=float)
    phi = np.asarray(phi_nodes, dtype=float)
    if mu.shape != phi.shape:
        raise ValueError("mu_nodes and phi_nodes must have the same shape.")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(phi)):
        raise ValueError("mu_nodes and phi_nodes must contain only finite values.")
    W_plus = lrs_quadrupole_kernel(mu)
    W_minus = 0.5 * np.sqrt(3.0) * (1.0 - mu**2) * np.cos(2.0 * phi)
    return W_plus, W_minus


def angular_monopole(
    f: np.ndarray,
    angular_weights: np.ndarray,
) -> np.ndarray:
    """Return normalized angular monopole with shape `(..., n_q)`."""

    f_arr = np.asarray(f, dtype=float)
    weights = np.asarray(angular_weights, dtype=float)
    if f_arr.ndim < 2:
        raise ValueError("f must have shape (..., n_q, n_angles).")
    if weights.shape != (f_arr.shape[-1],):
        raise ValueError("angular_weights must match the final axis of f.")
    if not np.all(np.isfinite(f_arr)) or not np.all(np.isfinite(weights)):
        raise ValueError("f and angular_weights must contain only finite values.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("angular_weights must have positive total weight.")
    return np.einsum("...qa,a->...q", f_arr, weights) / weight_sum


def _energy_integral(
    f: np.ndarray,
    q_energy_weights: np.ndarray,
    angular_weights: np.ndarray,
    angular_kernel: np.ndarray | None = None,
) -> np.ndarray:
    f_arr = np.asarray(f, dtype=float)
    q_weights = np.asarray(q_energy_weights, dtype=float)
    a_weights = np.asarray(angular_weights, dtype=float)
    if f_arr.ndim < 2:
        raise ValueError("f must have shape (..., n_q, n_angles).")
    if q_weights.shape != (f_arr.shape[-2],):
        raise ValueError("q_energy_weights must match the q axis of f.")
    if a_weights.shape != (f_arr.shape[-1],):
        raise ValueError("angular_weights must match the angular axis of f.")
    if not np.all(np.isfinite(f_arr)):
        raise ValueError("f must contain only finite values.")
    if np.any((f_arr < -1.0e-15) | (f_arr > 1.0 + 1.0e-15)):
        raise ValueError("f must lie within [0, 1] up to numerical tolerance.")
    if not np.all(np.isfinite(q_weights)) or not np.all(np.isfinite(a_weights)):
        raise ValueError("quadrature weights must contain only finite values.")

    if angular_kernel is None:
        kernel = np.ones_like(a_weights)
    else:
        kernel = np.asarray(angular_kernel, dtype=float)
        if kernel.shape != a_weights.shape:
            raise ValueError("angular_kernel must match angular_weights.")
        if not np.all(np.isfinite(kernel)):
            raise ValueError("angular_kernel must contain only finite values.")

    return np.einsum("...qa,q,a,a->...", f_arr, q_weights, a_weights, kernel)


def stress_moments_from_distribution(
    f: np.ndarray,
    q_energy_weights: np.ndarray,
    angular_weights: np.ndarray,
    W_plus: np.ndarray,
    *,
    W_minus: np.ndarray | None = None,
    rho_floor: float = 1.0e-300,
) -> AugmentedStressMoments:
    """Compute energy-weighted Type I stress moments from `f(q, angle)`.

    `q_energy_weights` must already include the desired momentum measure
    for `integral dq q^3 (...)`.  For Gauss-Laguerre nodes this usually
    means passing the combined `w_i exp(q_i) q_i^3` weights.
    """

    if rho_floor <= 0.0:
        raise ValueError("rho_floor must be positive.")

    f_arr = np.asarray(f, dtype=float)
    q_weights = np.asarray(q_energy_weights, dtype=float)
    a_weights = np.asarray(angular_weights, dtype=float)
    Wp = np.asarray(W_plus, dtype=float)
    if f_arr.ndim < 2:
        raise ValueError("f must have shape (..., n_q, n_angles).")
    if q_weights.shape != (f_arr.shape[-2],):
        raise ValueError("q_energy_weights must match the q axis of f.")
    if a_weights.shape != (f_arr.shape[-1],):
        raise ValueError("angular_weights must match the angular axis of f.")
    if Wp.shape != a_weights.shape:
        raise ValueError("angular_kernel must match angular_weights.")
    if not np.all(np.isfinite(f_arr)):
        raise ValueError("f must contain only finite values.")
    if np.any((f_arr < -1.0e-15) | (f_arr > 1.0 + 1.0e-15)):
        raise ValueError("f must lie within [0, 1] up to numerical tolerance.")
    if not np.all(np.isfinite(q_weights)) or not np.all(np.isfinite(a_weights)):
        raise ValueError("quadrature weights must contain only finite values.")
    if not np.all(np.isfinite(Wp)):
        raise ValueError("angular_kernel must contain only finite values.")
    weight_sum = float(np.sum(a_weights))
    if weight_sum <= 0.0:
        raise ValueError("angular_weights must have positive total weight.")

    rho = np.einsum("...qa,q,a->...", f_arr, q_weights, a_weights)
    plus = np.einsum("...qa,q,a,a->...", f_arr, q_weights, a_weights, Wp)
    denom = np.maximum(rho, float(rho_floor))
    pi_plus = plus / denom

    pi_minus = None
    if W_minus is not None:
        Wm = np.asarray(W_minus, dtype=float)
        if Wm.shape != a_weights.shape:
            raise ValueError("angular_kernel must match angular_weights.")
        if not np.all(np.isfinite(Wm)):
            raise ValueError("angular_kernel must contain only finite values.")
        minus = np.einsum("...qa,q,a,a->...", f_arr, q_weights, a_weights, Wm)
        pi_minus = minus / denom

    return AugmentedStressMoments(
        rho_weighted=rho,
        monopole_q=np.einsum("...qa,a->...q", f_arr, a_weights) / weight_sum,
        pi_plus_tilde=pi_plus,
        pi_minus_tilde=pi_minus,
    )


__all__ = [
    "AugmentedStressMoments",
    "angular_monopole",
    "lrs_quadrupole_kernel",
    "non_lrs_quadrupole_kernels",
    "stress_moments_from_distribution",
]
