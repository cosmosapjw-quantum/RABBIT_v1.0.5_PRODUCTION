"""Reusable JAX transport operators for flat and curved Bianchi kernels.

This module separates three reusable layers that were previously embedded in
``rhs_typeI.py``:

1. momentum-space projector construction,
2. hierarchy -> physical observable projection (monopoles, anisotropic stress),
3. streaming RHS assembly.

The immediate goal is not a full curved production hierarchy.  The goal is to
factor the Type-I exact pieces into operators that can be reused by future
Class-A curved transport drivers with the same quadrature and projection
conventions.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

A_FEEDBACK = 6.0
B_QUADRUPOLE = 8.0 / 15.0
D_DAMPING = 4.0


@jax.jit
def fd_equilibrium(q_nodes: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (jnp.exp(q_nodes) + 1.0)


@jax.jit
def build_energy_weight(q_nodes: jnp.ndarray, q_weights: jnp.ndarray) -> jnp.ndarray:
    """Physical q^3 f0(q) measure on a standard Laguerre grid.

    Standard Laguerre weights integrate ``∫ e^{-q} g(q) dq``.  Therefore the
    physical energy weight for reduced quadrupoles is

        W_i = w_i e^{q_i} q_i^3 f0(q_i).
    """
    f0_eq = fd_equilibrium(q_nodes)
    return q_weights * jnp.exp(q_nodes) * q_nodes ** 3 * f0_eq


@jax.jit
def project_species_quadrupoles(psi_species_quad: jnp.ndarray, energy_weight: jnp.ndarray) -> jnp.ndarray:
    """Project per-species quadrupoles to reduced π̃ values.

    Parameters
    ----------
    psi_species_quad : array, shape (n_species, N_q)
    energy_weight : array, shape (N_q,)
    """
    norm = jnp.maximum(jnp.sum(energy_weight), 1e-100)
    return jnp.sum(psi_species_quad * energy_weight[None, :], axis=1) / norm


@jax.jit
def species_mean_stress(pi_tilde_per_species: jnp.ndarray) -> jnp.ndarray:
    """Identical-species reduced-model average.

    This preserves the current Type-I reduced-model convention; future curved
    drivers can replace this with an energy-weighted species sum without
    changing the projector layer.
    """
    return jnp.mean(pi_tilde_per_species)


@partial(jax.jit, static_argnums=(1, 2))
def extract_aniso_stress_operator(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    f_nu: jnp.ndarray,
) -> tuple:
    """Extract Π₊, Π₋ with reusable quadrature/projector operators."""
    n_species = psi_flat.shape[0] // (n_ell * N_q)
    psi = psi_flat.reshape(n_species, n_ell, N_q)
    weight = build_energy_weight(q_nodes, q_weights)

    pi_tilde_plus_per_species = project_species_quadrupoles(psi[:, 1, :], weight)
    pi_tilde_plus = species_mean_stress(pi_tilde_plus_per_species)
    pi_plus = A_FEEDBACK * f_nu * pi_tilde_plus

    pi_minus = 0.0
    if n_ell >= 3:
        pi_tilde_minus_per_species = project_species_quadrupoles(psi[:, 2, :], weight)
        pi_tilde_minus = species_mean_stress(pi_tilde_minus_per_species)
        pi_minus = A_FEEDBACK * f_nu * pi_tilde_minus
    return pi_plus, pi_minus


@partial(jax.jit, static_argnums=(1, 2))
def extract_pi_tilde_per_species(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Extract per-species reduced anisotropic stress π̃_s for Teff correction.

    Returns π̃ for each species, shape (n_species,).  For the standard
    6-species layout: indices 0=ν_e, 1=ν̄_e, 2=ν_μ, 3=ν̄_μ, 4=ν_τ, 5=ν̄_τ.
    """
    n_species = psi_flat.shape[0] // (n_ell * N_q)
    psi = psi_flat.reshape(n_species, n_ell, N_q)
    weight = build_energy_weight(q_nodes, q_weights)
    return project_species_quadrupoles(psi[:, 1, :], weight)


@partial(jax.jit, static_argnums=(1, 2, 4))
def extract_monopole_distributions_operator(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    q_nodes: jnp.ndarray,
    species_pair: tuple[int, int] = (0, 1),
) -> tuple:
    """Extract physical f(q)=f_eq(1+Ψ0) for a species pair."""
    n_species = psi_flat.shape[0] // (n_ell * N_q)
    psi = psi_flat.reshape(n_species, n_ell, N_q)
    f0_eq = fd_equilibrium(q_nodes)
    s0, s1 = species_pair
    f_a = f0_eq * (1.0 + psi[s0, 0, :])
    f_b = f0_eq * (1.0 + psi[s1, 0, :])
    return f_a, f_b


@partial(jax.jit, static_argnums=(1, 2, 3))
def extract_monopole_perturbation_operator(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    species_idx: int = 0,
) -> jnp.ndarray:
    n_species = psi_flat.shape[0] // (n_ell * N_q)
    psi = psi_flat.reshape(n_species, n_ell, N_q)
    return psi[species_idx, 0, :]


@partial(jax.jit, static_argnums=(3, 4))
def apply_flat_streaming_rhs(
    Sigma_plus: jnp.ndarray,
    Sigma_minus: jnp.ndarray,
    psi_flat: jnp.ndarray,
    n_ell: int = 2,
    n_species: int = 6,
) -> jnp.ndarray:
    """Type-I collisionless streaming RHS, ℓ=2 PSTF closure (vectorized).

    Exact to LINEAR order in accumulated shear only (ℓ>=4 truncated); use the
    method-of-characteristics path for the ℓ-untruncated production transport
    (BD599 BIA-1).
    """
    total = psi_flat.shape[0]
    N_q = total // (n_species * n_ell)
    dpsi = jnp.zeros((n_species, n_ell, N_q), dtype=psi_flat.dtype)
    source_plus = -B_QUADRUPOLE * Sigma_plus
    dpsi = dpsi.at[:, 1, :].set(source_plus)
    if n_ell >= 3:
        source_minus = -B_QUADRUPOLE * Sigma_minus
        dpsi = dpsi.at[:, 2, :].set(source_minus)
    return dpsi.reshape(-1)
