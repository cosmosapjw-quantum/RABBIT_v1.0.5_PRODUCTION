"""
rabbit.jax.geometry_classB_jax — Class B Bianchi geometry kernel.

6 Class B types (V, IV, III, VI_h, VII_h, VI₋₁/₉) in Wainwright–Hsu formalism.
Imports common algebra from geometry_bianchi_base; adds the frame variable A.

State: y_geo = (Σ₊, Σ₋, N₁, N₂, N₃, A)  — 6 DOF

Key differences from Class A:
  1. Frame variable A with evolution dA/dN = (q + 2Σ₊)A
  2. Friedmann constraint: Ω = 1 − Σ² − K − cA²
  3. Deceleration: q = 1 + Σ² − K − cA²
  4. c = 3 generic, c = 15/4 for exceptional VI₋₁/₉

Physics (from RABBIT_report.tex §Eigenvalue Analysis):
  Class B modifies BBN through expansion ONLY (via cA² in Friedmann),
  NOT through altered shear dynamics. The σ–π eigenvalues are nearly
  unchanged: Δα < 0.02 even at 3A² = 0.25.

Type recovery:
  V:      A ≠ 0, N_i = 0 → K = S = 0, pure frame effect on Hubble
  IV:     A ≠ 0, N₁ ≠ 0 → frame + single curvature
  III:    A ≠ 0, N₂, N₃ ≠ 0 (h = −1)
  VI_h:   A ≠ 0, N₂, N₃ ≠ 0 (h < 0, h ≠ −1)
  VII_h:  A ≠ 0, N₂, N₃ ≠ 0 (h > 0)
  VI₋₁/₉: A ≠ 0, N₂, N₃ ≠ 0 (exceptional, c = 15/4)

References:
  Wainwright & Ellis 1997, §6.2
  Ellis & MacCallum 1969
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
    frame_variable_rhs,
    classB_c_factor,
    get_type_mask_jax,
    CLASS_B_TYPES,
)


# ═══════════════════════════════════════════════════════════════
# §1. Type masks (N_i active flags — A handled separately)
# ═══════════════════════════════════════════════════════════════

TYPE_MASKS_B = {
    BianchiType.TYPE_V:      jnp.array([0.0, 0.0, 0.0]),  # A only
    BianchiType.TYPE_IV:     jnp.array([1.0, 0.0, 0.0]),  # A + N₁
    BianchiType.TYPE_III:    jnp.array([0.0, 1.0, 1.0]),  # A + N₂, N₃
    BianchiType.TYPE_VIH:    jnp.array([0.0, 1.0, 1.0]),  # A + N₂, N₃
    BianchiType.TYPE_VIIH:   jnp.array([0.0, 1.0, 1.0]),  # A + N₂, N₃
    BianchiType.TYPE_VI_M19: jnp.array([0.0, 1.0, 1.0]),  # A + N₂, N₃ (exceptional)
}


# ═══════════════════════════════════════════════════════════════
# §2. Result dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClassBGeometryResult:
    dSigma_plus_dN: float
    dSigma_minus_dN: float
    dN1_dN: float
    dN2_dN: float
    dN3_dN: float
    dA_dN: float
    q: float
    Omega: float
    K: float
    cA_sq: float
    S_plus: float
    S_minus: float
    constraint_residual: float
    c_factor: float
    type_mask: tuple


# ═══════════════════════════════════════════════════════════════
# §3. Dispatch helpers
# ═══════════════════════════════════════════════════════════════

def build_classB_type_mask(bianchi_type: BianchiType | str) -> jnp.ndarray:
    """Return N_i mask for a Class B type."""
    if not isinstance(bianchi_type, BianchiType):
        bianchi_type = BianchiType(bianchi_type)
    if bianchi_type not in TYPE_MASKS_B:
        raise ValueError(f"{bianchi_type} is not a Class B type. "
                         f"Class B types: {list(TYPE_MASKS_B.keys())}")
    return TYPE_MASKS_B[bianchi_type]


def get_c_factor(bianchi_type: BianchiType | str) -> float:
    """Return Friedmann c-factor for a Class B type."""
    name = bianchi_type.value if isinstance(bianchi_type, BianchiType) else str(bianchi_type)
    return classB_c_factor(name)


# ═══════════════════════════════════════════════════════════════
# §4. Full geometry RHS
# ═══════════════════════════════════════════════════════════════

def compute_classB_geometry_rhs_jax(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    *,
    pi_shear_plus: float = 0.0,
    pi_shear_minus: float = 0.0,
    type_mask: Iterable[float] | None = None,
    c_factor: float = 3.0,
) -> ClassBGeometryResult:
    """Full Class B geometry RHS.

    State: (Σ₊, Σ₋, N₁, N₂, N₃, A)
    Friedmann: Ω = 1 − Σ² − K − cA²
    Deceleration: q = 1 + Σ² − K − cA²

    Parameters
    ----------
    A : float
        WH frame variable (expansion-normalized).
    c_factor : float
        Friedmann coefficient for A² (3.0 generic, 3.75 for VI₋₁/₉).
    """
    if type_mask is None:
        type_mask = jnp.array([1.0, 1.0, 1.0])
    mask = jnp.asarray(type_mask, dtype=jnp.float64)
    N_arr = mask * jnp.asarray([N1, N2, N3], dtype=jnp.float64)
    N1m, N2m, N3m = N_arr[0], N_arr[1], N_arr[2]

    Sp = jnp.asarray(Sigma_plus, dtype=jnp.float64)
    Sm = jnp.asarray(Sigma_minus, dtype=jnp.float64)
    A_val = jnp.asarray(A, dtype=jnp.float64)

    # Curvature from N_i (same as Class A)
    K, S_plus, S_minus = curvature_invariants(N1m, N2m, N3m)

    # Class B extension: cA² enters Friedmann and q
    Sigma_sq = Sp**2 + Sm**2
    cA_sq = c_factor * A_val**2

    q = compute_q(Sigma_sq, K, cA_sq)
    Omega = compute_Omega(Sigma_sq, K, cA_sq)

    # Shear (same form; curvature sources from N_i only)
    dSp, dSm = shear_rhs(Sp, Sm, q, S_plus, S_minus, pi_shear_plus, pi_shear_minus)

    # Structure constants (same eigenvalues as Class A)
    dN1, dN2, dN3 = structure_rhs(N1m, N2m, N3m, q, Sp, Sm)

    # Frame variable (Class B specific)
    dA = frame_variable_rhs(A_val, q, Sp)

    # Constraint residual
    residual = friedmann_residual(Sigma_sq, K, Omega, cA_sq)

    return ClassBGeometryResult(
        dSigma_plus_dN=float(dSp),
        dSigma_minus_dN=float(dSm),
        dN1_dN=float(dN1),
        dN2_dN=float(dN2),
        dN3_dN=float(dN3),
        dA_dN=float(dA),
        q=float(q),
        Omega=float(Omega),
        K=float(K),
        cA_sq=float(cA_sq),
        S_plus=float(S_plus),
        S_minus=float(S_minus),
        constraint_residual=float(residual),
        c_factor=float(c_factor),
        type_mask=tuple(float(x) for x in mask),
    )


# ═══════════════════════════════════════════════════════════════
# §5. Type-dispatched convenience
# ═══════════════════════════════════════════════════════════════

def compute_classB_rhs_for_type(
    Sigma_plus: float, Sigma_minus: float,
    N1: float, N2: float, N3: float, A: float,
    bianchi_type: BianchiType | str,
    *,
    pi_shear_plus: float = 0.0,
    pi_shear_minus: float = 0.0,
) -> ClassBGeometryResult:
    """Geometry RHS with automatic type mask and c-factor dispatch."""
    mask = build_classB_type_mask(bianchi_type)
    c = get_c_factor(bianchi_type)
    return compute_classB_geometry_rhs_jax(
        Sigma_plus, Sigma_minus, N1, N2, N3, A,
        pi_shear_plus=pi_shear_plus,
        pi_shear_minus=pi_shear_minus,
        type_mask=mask,
        c_factor=c,
    )


# ═══════════════════════════════════════════════════════════════
# §6. Reduction checks
# ═══════════════════════════════════════════════════════════════

def classB_reduces_to_classA_when_A_zero(
    Sigma_plus: float, Sigma_minus: float,
    N1: float, N2: float, N3: float,
) -> bool:
    """Verify that Class B with A=0 reproduces Class A geometry."""
    from rabbit.jax.geometry_classA_jax import compute_classA_geometry_rhs_jax
    rhs_B = compute_classB_geometry_rhs_jax(
        Sigma_plus, Sigma_minus, N1, N2, N3, A=0.0, c_factor=3.0)
    rhs_A = compute_classA_geometry_rhs_jax(
        Sigma_plus, Sigma_minus, N1, N2, N3)
    return (abs(rhs_B.dSigma_plus_dN - rhs_A.dSigma_plus_dN) < 1e-14
            and abs(rhs_B.K - rhs_A.K) < 1e-14
            and abs(rhs_B.dA_dN) < 1e-14)
