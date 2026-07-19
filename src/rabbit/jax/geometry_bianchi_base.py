"""
rabbit.jax.geometry_bianchi_base — Common Bianchi geometry interface.

Shared algebra for all Bianchi types (Class A and Class B).
Class-specific modules (geometry_classA_jax, future geometry_classB_jax)
import from here instead of duplicating.

Hierarchy:
    geometry_bianchi_base.py   ← this file (common algebra)
    ├── geometry_classA_jax.py  (type masks, full RHS for 6 types)
    └── geometry_classB_jax.py  (future: frame variable A, c-factor)

Common functions (shared by all 12 Bianchi types):
    §1. Gauss curvature K and curvature sources S₊, S₋
    §2. Deceleration parameter q and Friedmann constraint Ω
    §3. Shear evolution RHS (same form for A and B)
    §4. Structure constant evolution eigenvalues
    §5. Bianchi class identification

Class B extension points (not implemented here):
    - Frame variable: dA/dN = (q + 2Σ₊)A
    - Friedmann: Ω = 1 - Σ² - K - cA²  (c=3 generic, c=15/4 for VI₋₁/₉)
    - q = 1 + Σ² − K − cA²

References:
    Wainwright & Ellis 1997, §6.2
    Ellis & MacCallum 1969
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SQRT3 = jnp.sqrt(3.0)


# ═══════════════════════════════════════════════════════════════
# §1. Curvature invariants (common to all types)
# ═══════════════════════════════════════════════════════════════

@jax.jit
def gauss_curvature_K(N1: jnp.ndarray, N2: jnp.ndarray, N3: jnp.ndarray) -> jnp.ndarray:
    """Gauss curvature invariant K from WH structure-constant variables.

    K = (1/12)(N₁² + N₂² + N₃² − 2N₁N₂ − 2N₂N₃ − 2N₃N₁)

    Positive K: open types (II, VI₀, VIII with mixed signs)
    Negative K: closed type (IX)
    Zero K: flat (I) and flat-curvature (VII₀)
    """
    return (1.0 / 12.0) * (
        N1**2 + N2**2 + N3**2
        - 2.0 * N1 * N2 - 2.0 * N2 * N3 - 2.0 * N3 * N1
    )


@jax.jit
def curvature_sources(N1: jnp.ndarray, N2: jnp.ndarray, N3: jnp.ndarray):
    """Wainwright–Hsu curvature source terms S₊, S₋.

    S₊ = (1/6)[(N₂−N₃)² − N₁(2N₁−N₂−N₃)]
    S₋ = (√3/6)(N₃−N₂)(N₁−N₂−N₃)

    These enter the shear evolution as dΣ/dN = ... − S.
    For Class A: this is exact.
    For Class B: same formula (A does not contribute to curvature sources).
    """
    S_plus = (1.0 / 6.0) * ((N2 - N3)**2 - N1 * (2.0 * N1 - N2 - N3))
    S_minus = (SQRT3 / 6.0) * (N3 - N2) * (N1 - N2 - N3)
    return S_plus, S_minus


@jax.jit
def curvature_invariants(N1: jnp.ndarray, N2: jnp.ndarray, N3: jnp.ndarray):
    """Compute K, S₊, S₋ together."""
    K = gauss_curvature_K(N1, N2, N3)
    S_plus, S_minus = curvature_sources(N1, N2, N3)
    return K, S_plus, S_minus


# ═══════════════════════════════════════════════════════════════
# §2. Deceleration parameter and Friedmann constraint
# ═══════════════════════════════════════════════════════════════

@jax.jit
def compute_q(Sigma_sq: jnp.ndarray, K: jnp.ndarray, cA_sq: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Deceleration parameter for radiation domination.

    For radiation, q = 2Σ² + Ω. With Friedmann
        Ω = 1 − Σ² − K − cA²,
    this gives
        q = 1 + Σ² − K − cA².

    Class A: cA² = 0.
    Class B: cA² = c × A² where c = 3 (generic) or 15/4 (VI₋₁/₉).

    Verified at Collins–Stewart: Σ² = 1/16, K = 1/16, q = 1.
    """
    return 1.0 + Sigma_sq - K - cA_sq


@jax.jit
def compute_Omega(Sigma_sq: jnp.ndarray, K: jnp.ndarray, cA_sq: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Friedmann constraint: Ω = 1 − Σ² − K − cA².

    Must be > 0 for physical solutions.
    """
    return 1.0 - Sigma_sq - K - cA_sq


@jax.jit
def friedmann_residual(Sigma_sq: jnp.ndarray, K: jnp.ndarray, Omega: jnp.ndarray,
                       cA_sq: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Friedmann constraint residual: |1 − Σ² − K − cA² − Ω|."""
    return jnp.abs(1.0 - Sigma_sq - K - cA_sq - Omega)


# ═══════════════════════════════════════════════════════════════
# §3. Shear evolution (same form for all types)
# ═══════════════════════════════════════════════════════════════

@jax.jit
def shear_rhs(Sigma_plus: jnp.ndarray, Sigma_minus: jnp.ndarray,
              q: jnp.ndarray,
              S_plus: jnp.ndarray, S_minus: jnp.ndarray,
              Pi_plus: jnp.ndarray = 0.0, Pi_minus: jnp.ndarray = 0.0):
    """Shear evolution equations (universal for all Bianchi types).

    dΣ₊/dN = −(2−q)Σ₊ − S₊ + Π₊
    dΣ₋/dN = −(2−q)Σ₋ − S₋ + Π₋

    −(2−q)Σ: cosmological friction (Hubble damping)
    −S: curvature source (zero for Type I)
    +Π: neutrino anisotropic stress backreaction
    """
    damping = -(2.0 - q)
    dSp = damping * Sigma_plus - S_plus + Pi_plus
    dSm = damping * Sigma_minus - S_minus + Pi_minus
    return dSp, dSm


# ═══════════════════════════════════════════════════════════════
# §4. Structure constant evolution eigenvalues
# ═══════════════════════════════════════════════════════════════

@jax.jit
def structure_eigenvalues(q: jnp.ndarray, Sigma_plus: jnp.ndarray,
                          Sigma_minus: jnp.ndarray):
    """Eigenvalues for N_i evolution: dN_i/dN = λ_i × N_i.

    λ₁ = q − 4Σ₊
    λ₂ = q + 2Σ₊ + 2√3 Σ₋
    λ₃ = q + 2Σ₊ − 2√3 Σ₋

    These are the diagonal entries of q·I + 2Σ in the WH basis.
    Same for Class A and Class B N_i variables.
    """
    lam1 = q - 4.0 * Sigma_plus
    lam2 = q + 2.0 * Sigma_plus + 2.0 * SQRT3 * Sigma_minus
    lam3 = q + 2.0 * Sigma_plus - 2.0 * SQRT3 * Sigma_minus
    return lam1, lam2, lam3


@jax.jit
def structure_rhs(N1: jnp.ndarray, N2: jnp.ndarray, N3: jnp.ndarray,
                  q: jnp.ndarray, Sigma_plus: jnp.ndarray,
                  Sigma_minus: jnp.ndarray):
    """Structure constant evolution: dN_i/dN = λ_i × N_i.

    Common to Class A and Class B.
    Type mask should be applied BEFORE calling this function.
    """
    lam1, lam2, lam3 = structure_eigenvalues(q, Sigma_plus, Sigma_minus)
    return lam1 * N1, lam2 * N2, lam3 * N3


# ═══════════════════════════════════════════════════════════════
# §5. Class B extension: frame variable
# ═══════════════════════════════════════════════════════════════

@jax.jit
def frame_variable_rhs(A: jnp.ndarray, q: jnp.ndarray,
                       Sigma_plus: jnp.ndarray) -> jnp.ndarray:
    """Frame variable evolution (Class B only).

    dA/dN = (q + 2Σ₊) × A

    Active for types V, IV, III, VI_h, VII_h, VI₋₁/₉.
    For Class A: A ≡ 0, this is never called.
    """
    return (q + 2.0 * Sigma_plus) * A


def classB_c_factor(bianchi_type_str: str) -> float:
    """Friedmann constraint coefficient for the frame variable.

    Ω = 1 − Σ² − K − c × A²

    c = 3    for generic Class B (V, IV, III, VI_h, VII_h)
    c = 15/4 for the exceptional type VI₋₁/₉
    """
    key = bianchi_type_str.upper()
    if "VI_M19" in key or "-1/9" in key:
        return 15.0 / 4.0
    return 3.0


# ═══════════════════════════════════════════════════════════════
# §6. Bianchi class identification
# ═══════════════════════════════════════════════════════════════

CLASS_A_TYPES = frozenset({"TYPE_I", "TYPE_II", "TYPE_VI0", "TYPE_VII0", "TYPE_VIII", "TYPE_IX"})
CLASS_B_TYPES = frozenset({"TYPE_V", "TYPE_IV", "TYPE_III", "TYPE_VIH", "TYPE_VIIH", "TYPE_VI_M19"})

def is_class_A(bianchi_type_str: str) -> bool:
    key = bianchi_type_str.upper()
    return key in CLASS_A_TYPES or f"TYPE_{key}" in CLASS_A_TYPES

def is_class_B(bianchi_type_str: str) -> bool:
    key = bianchi_type_str.upper()
    return key in CLASS_B_TYPES or f"TYPE_{key}" in CLASS_B_TYPES

def has_frame_variable(bianchi_type_str: str) -> bool:
    """Class B types have the frame variable A ≠ 0."""
    return is_class_B(bianchi_type_str)

def has_curvature(bianchi_type_str: str) -> bool:
    """Types with nonzero spatial curvature (all except I and V)."""
    key = bianchi_type_str.upper()
    return key not in {"TYPE_I", "TYPE_V", "I", "V"}


# ═══════════════════════════════════════════════════════════════
# §7. Type mask registry (Class A + Class B)
# ═══════════════════════════════════════════════════════════════

# N_i active masks: which structure constant variables are nonzero.
# Class A: (N₁, N₂, N₃) only
# Class B: (N₁, N₂, N₃) + separate A variable
TYPE_MASK_REGISTRY = {
    # Class A
    "TYPE_I":    (0.0, 0.0, 0.0),
    "TYPE_II":   (1.0, 0.0, 0.0),
    "TYPE_VI0":  (0.0, 1.0, 1.0),
    "TYPE_VII0": (0.0, 1.0, 1.0),
    "TYPE_VIII": (1.0, 1.0, 1.0),
    "TYPE_IX":   (1.0, 1.0, 1.0),
    # Class B (N_i masks; A is handled separately)
    "TYPE_V":     (0.0, 0.0, 0.0),   # A only, no N_i
    "TYPE_IV":    (1.0, 0.0, 0.0),   # A + N₁
    "TYPE_III":   (0.0, 1.0, 1.0),   # A + N₂, N₃
    "TYPE_VIH":   (0.0, 1.0, 1.0),   # A + N₂, N₃
    "TYPE_VIIH":  (0.0, 1.0, 1.0),   # A + N₂, N₃
    "TYPE_VI_M19": (0.0, 1.0, 1.0),  # A + N₂, N₃ (exceptional)
}


def get_type_mask(bianchi_type_str: str) -> tuple:
    """Return (N₁_active, N₂_active, N₃_active) mask for any Bianchi type."""
    key = bianchi_type_str.upper()
    if key in TYPE_MASK_REGISTRY:
        return TYPE_MASK_REGISTRY[key]
    # Try with TYPE_ prefix
    prefixed = f"TYPE_{key}"
    if prefixed in TYPE_MASK_REGISTRY:
        return TYPE_MASK_REGISTRY[prefixed]
    raise ValueError(f"Unknown Bianchi type: {bianchi_type_str}")


def get_type_mask_jax(bianchi_type_str: str) -> jnp.ndarray:
    """Return mask as JAX array."""
    return jnp.array(get_type_mask(bianchi_type_str), dtype=jnp.float64)
