"""Class A transport — reduced κ-cascade branch.

Provides curved streaming with a scalar curvature-cascade parameter κ
and exact Type I reduction at κ=0. This is NOT the exact curved PSTF
hierarchy (which requires m-decomposition and ℓ→ℓ±1 coupling).

Validation:
  - Type I (κ=0): exact reduction to flat operator (1e-15)
  - Type II (κ>0): monopole-quadrupole coupling activated
  - Floor damping D=4 matches linearized PSTF convention
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.config.conventions import BianchiType
from rabbit.jax.geometry_classA_jax import build_type_mask
from rabbit.jax.transport_ops_jax import B_QUADRUPOLE, D_DAMPING, apply_flat_streaming_rhs


@jax.jit
def effective_kappa_from_curvature(N1: float, N2: float, N3: float, type_mask: jnp.ndarray) -> jnp.ndarray:
    """Minimal scalar curvature-cascade proxy.

    The goal is only a reusable substrate for future curved transport work.
    Type I gives exactly zero; representative curved types give a finite
    positive κ that activates monopole-quadrupole mixing.
    """
    N = type_mask * jnp.asarray([N1, N2, N3], dtype=jnp.float64)
    return jnp.sqrt(jnp.sum(N ** 2))


@partial(jax.jit, static_argnums=(3, 4))
def classA_transport_rhs(
    Sigma_plus: jnp.ndarray,
    Sigma_minus: jnp.ndarray,
    psi_flat: jnp.ndarray,
    n_ell: int = 2,
    n_species: int = 6,
    *,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Curved streaming substrate with exact Type-I reduction.

    For κ = 0 this returns the exact flat Type-I operator.
    For κ > 0 and n_ell >= 2 the leading curved couplings are

        dΨ0/dN = -(2/3) κ Ψ2
        dΨ2/dN = -(8/15) Σ - D Ψ2 + (2/5) κ Ψ0

    with the same floor damping D = 4 used by the curved reference template.
    """
    total = psi_flat.shape[0]
    N_q = total // (n_species * n_ell)
    psi = psi_flat.reshape(n_species, n_ell, N_q)
    dpsi = jnp.zeros_like(psi)

    # Monopole <-> quadrupole coupling for curved transport scaffold.
    dpsi = dpsi.at[:, 0, :].set(-(2.0 / 3.0) * kappa * psi[:, 1, :])
    dpsi = dpsi.at[:, 1, :].set(
        -B_QUADRUPOLE * Sigma_plus - D_DAMPING * psi[:, 1, :] + (2.0 / 5.0) * kappa * psi[:, 0, :]
    )

    if n_ell >= 3:
        dpsi = dpsi.at[:, 2, :].set(-B_QUADRUPOLE * Sigma_minus - D_DAMPING * psi[:, 2, :])

    dpsi_curved = dpsi.reshape(total)
    dpsi_flat = apply_flat_streaming_rhs(Sigma_plus, Sigma_minus, psi_flat, n_ell=n_ell, n_species=n_species)
    return jnp.where(kappa == 0.0, dpsi_flat, dpsi_curved)


def classA_transport_rhs_for_type(
    Sigma_plus: float,
    Sigma_minus: float,
    psi_flat: jnp.ndarray,
    bianchi_type: BianchiType | str,
    N1: float,
    N2: float,
    N3: float,
    *,
    n_ell: int = 2,
    n_species: int = 6,
) -> jnp.ndarray:
    mask = build_type_mask(bianchi_type)
    kappa = effective_kappa_from_curvature(N1, N2, N3, mask)
    return classA_transport_rhs(
        Sigma_plus,
        Sigma_minus,
        psi_flat,
        n_ell=n_ell,
        n_species=n_species,
        kappa=kappa,
    )
