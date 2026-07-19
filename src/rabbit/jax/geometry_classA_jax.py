"""JAX-native Class A Bianchi geometry — reduced branch.

Validated geometry kernel for all 6 Class A types (I, II, VI₀, VII₀, VIII, IX)
in the Wainwright–Hsu formalism. Type-mask dispatch removes inactive curvature
coordinates before dynamics are assembled.

This module imports common algebra from geometry_bianchi_base and adds
Class A–specific dispatch (type masks, full RHS assembly).

Validation status:
  - Type I reduction: exact to machine precision (1e-12)
  - Type II Collins-Stewart attractor: equilibrium verified (dΣ/dN < 1e-12)
  - Friedmann constraint residual: < 1e-12 for all types

State: y_geo = (Σ₊, Σ₋, N₁, N₂, N₃)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from jax import config as jax_config
import jax.numpy as jnp

jax_config.update("jax_enable_x64", True)

from rabbit.config.conventions import BianchiType
from rabbit.jax.geometry_bianchi_base import (
    SQRT3,
    gauss_curvature_K,
    curvature_sources,
    curvature_invariants,
    compute_q,
    compute_Omega,
    friedmann_residual,
    shear_rhs,
    structure_rhs,
    get_type_mask_jax,
    CLASS_A_TYPES,
)


# Backward-compatible type mask dict (Class A only)
TYPE_MASKS = {
    BianchiType.TYPE_I: jnp.array([0.0, 0.0, 0.0]),
    BianchiType.TYPE_II: jnp.array([1.0, 0.0, 0.0]),
    BianchiType.TYPE_VI0: jnp.array([0.0, 1.0, 1.0]),
    BianchiType.TYPE_VII0: jnp.array([0.0, 1.0, 1.0]),
    BianchiType.TYPE_VIII: jnp.array([1.0, 1.0, 1.0]),
    BianchiType.TYPE_IX: jnp.array([1.0, 1.0, 1.0]),
}


@dataclass(frozen=True)
class ClassAGeometryResult:
    dSigma_plus_dN: float
    dSigma_minus_dN: float
    dN1_dN: float
    dN2_dN: float
    dN3_dN: float
    q: float
    Omega: float
    K: float
    S_plus: float
    S_minus: float
    constraint_residual: float
    type_mask: tuple[float, float, float]


def build_type_mask(bianchi_type: BianchiType | str) -> jnp.ndarray:
    if not isinstance(bianchi_type, BianchiType):
        bianchi_type = BianchiType(bianchi_type)
    return TYPE_MASKS[bianchi_type]


def apply_type_mask_to_curvature(N1: float, N2: float, N3: float, type_mask: Iterable[float]) -> jnp.ndarray:
    mask = jnp.asarray(type_mask, dtype=jnp.float64)
    N = jnp.asarray([N1, N2, N3], dtype=jnp.float64)
    return mask * N


# Backward-compatible wrappers delegating to base module
def compute_curvature_invariants_jax(N1: float, N2: float, N3: float):
    N1 = jnp.asarray(N1, dtype=jnp.float64)
    N2 = jnp.asarray(N2, dtype=jnp.float64)
    N3 = jnp.asarray(N3, dtype=jnp.float64)
    return curvature_invariants(N1, N2, N3)


def compute_Omega_classA_jax(Sigma_plus: float, Sigma_minus: float, N1: float, N2: float, N3: float) -> jnp.ndarray:
    Sigma_sq = jnp.asarray(Sigma_plus, dtype=jnp.float64) ** 2 + jnp.asarray(Sigma_minus, dtype=jnp.float64) ** 2
    K = gauss_curvature_K(jnp.asarray(N1, dtype=jnp.float64),
                          jnp.asarray(N2, dtype=jnp.float64),
                          jnp.asarray(N3, dtype=jnp.float64))
    return compute_Omega(Sigma_sq, K)


def compute_q_classA_jax(Sigma_plus: float, Sigma_minus: float, N1: float, N2: float, N3: float) -> jnp.ndarray:
    """Deceleration parameter for radiation: q = 1 + Σ² - K.

    Using q = 2Σ² + Ω and Ω = 1 - Σ² - K for Class A gives
        q = 1 + Σ² - K.
    This recovers the Collins–Stewart equilibrium q = 1 at
    Σ₊ = 1/4 and K = 1/16.
    """
    Sigma_sq = jnp.asarray(Sigma_plus, dtype=jnp.float64) ** 2 + jnp.asarray(Sigma_minus, dtype=jnp.float64) ** 2
    K = gauss_curvature_K(jnp.asarray(N1, dtype=jnp.float64),
                          jnp.asarray(N2, dtype=jnp.float64),
                          jnp.asarray(N3, dtype=jnp.float64))
    return compute_q(Sigma_sq, K)


def compute_classA_constraint_residual_jax(Sigma_plus: float, Sigma_minus: float, N1: float, N2: float, N3: float, Omega: float | None = None) -> jnp.ndarray:
    Sigma_sq = jnp.asarray(Sigma_plus, dtype=jnp.float64) ** 2 + jnp.asarray(Sigma_minus, dtype=jnp.float64) ** 2
    K, _, _ = curvature_invariants(jnp.asarray(N1, dtype=jnp.float64),
                                    jnp.asarray(N2, dtype=jnp.float64),
                                    jnp.asarray(N3, dtype=jnp.float64))
    if Omega is None:
        Omega = compute_Omega(Sigma_sq, K)
    return friedmann_residual(Sigma_sq, K, jnp.asarray(Omega, dtype=jnp.float64))


def compute_classA_geometry_rhs_jax(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    *,
    pi_shear_plus: float = 0.0,
    pi_shear_minus: float = 0.0,
    type_mask: Iterable[float] | None = None,
) -> ClassAGeometryResult:
    if type_mask is None:
        type_mask = jnp.array([1.0, 1.0, 1.0])
    masked_N = apply_type_mask_to_curvature(N1, N2, N3, type_mask)
    N1m, N2m, N3m = masked_N

    K, S_plus, S_minus = curvature_invariants(N1m, N2m, N3m)
    Sigma_sq = jnp.asarray(Sigma_plus, dtype=jnp.float64)**2 + jnp.asarray(Sigma_minus, dtype=jnp.float64)**2
    q = compute_q(Sigma_sq, K)
    Omega = compute_Omega(Sigma_sq, K)

    dSp, dSm = shear_rhs(
        jnp.asarray(Sigma_plus, dtype=jnp.float64),
        jnp.asarray(Sigma_minus, dtype=jnp.float64),
        q, S_plus, S_minus, pi_shear_plus, pi_shear_minus)

    dN1, dN2, dN3 = structure_rhs(N1m, N2m, N3m, q,
                                   jnp.asarray(Sigma_plus, dtype=jnp.float64),
                                   jnp.asarray(Sigma_minus, dtype=jnp.float64))

    residual = friedmann_residual(Sigma_sq, K, Omega)

    return ClassAGeometryResult(
        dSigma_plus_dN=float(dSp),
        dSigma_minus_dN=float(dSm),
        dN1_dN=float(dN1),
        dN2_dN=float(dN2),
        dN3_dN=float(dN3),
        q=float(q),
        Omega=float(Omega),
        K=float(K),
        S_plus=float(S_plus),
        S_minus=float(S_minus),
        constraint_residual=float(residual),
        type_mask=tuple(float(x) for x in jnp.asarray(type_mask)),
    )
