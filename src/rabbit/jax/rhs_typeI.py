"""
rabbit.jax.rhs_typeI — JAX-native Type I geometry + transport RHS.

Pure algebra — trivial port from numpy. No interpolation, no quadrature.

Geometry:
    dΣ₊/dN = −(2−q)Σ₊ + Π₊     q = 1 + Σ²
    dΣ₋/dN = −(2−q)Σ₋ + Π₋

Transport (collisionless; ℓ=2 PSTF closure, exact to LINEAR order in accumulated
shear S=∫Σ dN only — NOT unqualified exact; ℓ>=4 + angular-drift content is
truncated and grows with S. Use transport_mode=CHARACTERISTIC for the
ℓ-untruncated production path. BD599 BIA-1):
    dΨ₀/dN = 0
    dΨ₂⁺/dN = −(8/15)Σ₊
    dΨ₂⁻/dN = −(8/15)Σ₋    (generic, n_ell=3)
"""
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

from rabbit.jax.transport_ops_jax import (
    A_FEEDBACK,
    B_QUADRUPOLE,
    D_DAMPING,
    apply_flat_streaming_rhs,
    extract_aniso_stress_operator,
    extract_monopole_distributions_operator,
    extract_monopole_perturbation_operator,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Geometry RHS
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def compute_q_radiation(Sigma_sq: jnp.ndarray) -> jnp.ndarray:
    """Deceleration parameter q = 1 + Σ² (radiation domination)."""
    return 1.0 + Sigma_sq


@jax.jit
def typeI_geometry_rhs(
    Sigma_plus: jnp.ndarray,
    Sigma_minus: jnp.ndarray,
    pi_plus: jnp.ndarray,
    pi_minus: jnp.ndarray,
) -> tuple:
    """dΣ₊/dN, dΣ₋/dN for Type I orthogonal Bianchi.

    Parameters
    ----------
    Sigma_plus, Sigma_minus : float
        Hubble-normalized shear.
    pi_plus, pi_minus : float
        Anisotropic stress from transport sector.

    Returns
    -------
    (dSigma_plus_dN, dSigma_minus_dN)
    """
    Sigma_sq = Sigma_plus ** 2 + Sigma_minus ** 2
    q = compute_q_radiation(Sigma_sq)
    damping = -(2.0 - q)  # = -(1 - Σ²)

    dSp = damping * Sigma_plus + pi_plus
    dSm = damping * Sigma_minus + pi_minus
    return dSp, dSm


# ═══════════════════════════════════════════════════════════════════════
# §2. Transport hierarchy RHS (collisionless ℓ=2 closure; exact to linear order
#     in accumulated shear only — see module docstring / BD599 BIA-1)
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnums=(3,))
def typeI_transport_rhs(
    Sigma_plus: jnp.ndarray,
    Sigma_minus: jnp.ndarray,
    psi_flat: jnp.ndarray,
    n_ell: int = 2,
) -> jnp.ndarray:
    """dΨ/dN for the collisionless Type I PSTF hierarchy.

    Thin wrapper around the reusable flat streaming operator.
    """
    return apply_flat_streaming_rhs(Sigma_plus, Sigma_minus, psi_flat, n_ell=n_ell, n_species=6)


# ═══════════════════════════════════════════════════════════════════════
# §3. Projectors: hierarchy → physics
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnums=(1, 2))
def extract_aniso_stress(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    f_nu: jnp.ndarray,
) -> tuple:
    """Backward-compatible wrapper around the reusable projector operator."""
    return extract_aniso_stress_operator(psi_flat, N_q, n_ell, q_nodes, q_weights, f_nu)




@partial(jax.jit, static_argnums=(1, 2))
def extract_monopole_distributions(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
    q_nodes: jnp.ndarray,
) -> tuple:
    """Backward-compatible wrapper around the reusable monopole extractor."""
    return extract_monopole_distributions_operator(psi_flat, N_q, n_ell, q_nodes, species_pair=(0, 1))

@partial(jax.jit, static_argnums=(1, 2))
def extract_monopole_f0(
    psi_flat: jnp.ndarray,
    N_q: int,
    n_ell: int,
) -> jnp.ndarray:
    """Extract the ν_e monopole perturbation Ψ₀(q) for species 0."""
    return extract_monopole_perturbation_operator(psi_flat, N_q, n_ell, species_idx=0)
