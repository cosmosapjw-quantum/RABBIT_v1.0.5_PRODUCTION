"""
rabbit.jax.tilt_jax — JAX tilt evolution for all Bianchi types.

Tilt = bulk velocity v of matter relative to symmetry-surface normal.
During radiation (γ=4/3), tilt GROWS (FLRW eigenvalue +1).
BBN requires v₀ ≲ 10⁻⁷ for physical results.

Key physics:
  dv_α/dN = v_α(1−v²)/G × [(2−q)(1−(γ−1)v²)/G + λ_α − (3γ−4)]

where λ_α are shear eigenvalues (trace-free):
  λ₁ = −Σ₊ − √3 Σ₋
  λ₂ = −Σ₊ + √3 Σ₋
  λ₃ = 2Σ₊

For scalar tilt (LRS): v = v₃ only, coupling through λ₃ = 2Σ₊.

Hubble correction: H_tilted = H_orth / √(1 − v²)

The tilt RHS is IDENTICAL for all Bianchi types (A and B) because
the shear eigenvalues depend only on (Σ₊, Σ₋), not on curvature
or the frame variable. The curvature enters only through q.

References:
  Hewitt, Bridson, Wainwright (2001) GRG 33, 65
  Wainwright & Ellis (1997) Ch. 7
"""
from __future__ import annotations

from functools import partial
import numpy as _np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SQRT3 = jnp.sqrt(3.0)
GAMMA_RAD = 4.0 / 3.0
GAMMA_CRIT = 14.0 / 9.0  # ≈ 1.5556
_BOOST_MU16_NP, _BOOST_W16_NP = _np.polynomial.legendre.leggauss(16)
BOOST_MU16 = jnp.asarray(_BOOST_MU16_NP, dtype=jnp.float64)
BOOST_W16 = jnp.asarray(_BOOST_W16_NP, dtype=jnp.float64)


# ═══════════════════════════════════════════════════════════════
# §1. Shear eigenvalues for tilt coupling
# ═══════════════════════════════════════════════════════════════

@jax.jit
def shear_eigenvalues_tilt(Sigma_plus: jnp.ndarray,
                           Sigma_minus: jnp.ndarray):
    """Trace-free shear eigenvalues that enter the tilt equation.

    λ₁ = −Σ₊ − √3 Σ₋
    λ₂ = −Σ₊ + √3 Σ₋
    λ₃ = 2Σ₊

    Sum = 0 (trace-free). Same for all Bianchi types.
    """
    return (
        -Sigma_plus - SQRT3 * Sigma_minus,
        -Sigma_plus + SQRT3 * Sigma_minus,
        2.0 * Sigma_plus,
    )


# ═══════════════════════════════════════════════════════════════
# §2. Scalar tilt RHS (LRS — single component v₃)
# ═══════════════════════════════════════════════════════════════

@jax.jit
def tilt_rhs_scalar(v: jnp.ndarray, q: jnp.ndarray,
                    Sigma_plus: jnp.ndarray,
                    gamma: float = GAMMA_RAD) -> jnp.ndarray:
    """Scalar (LRS) tilt evolution: dv/dN.

    Uses λ₃ = 2Σ₊ coupling. Full nonlinear.

    For radiation (γ=4/3): (3γ−4) = 0.
    At FLRW (Σ=0, q=1): dv/dN ≈ v (exponential growth).
    """
    v_sq = v**2
    G = 1.0 + (gamma - 1.0) * v_sq
    eos_term = 3.0 * gamma - 4.0  # = 0 for radiation
    friction = (2.0 - q) * (1.0 - (gamma - 1.0) * v_sq) / G
    lam3 = 2.0 * Sigma_plus
    bracket = friction + lam3 - eos_term
    # Suppress at v → 1
    return jnp.where(v_sq < 0.99, v * (1.0 - v_sq) / G * bracket, 0.0)


@partial(jax.jit, static_argnames=("axis",))
def tilt_rhs_principal_axis(
    v: jnp.ndarray,
    q: jnp.ndarray,
    Sigma_plus: jnp.ndarray,
    Sigma_minus: jnp.ndarray,
    *,
    axis: int = 3,
    gamma: float = GAMMA_RAD,
) -> jnp.ndarray:
    """Scalar tilt evolution along one diagonal principal axis.

    The diagonal WH runner evolves only diagonal shear variables.  A tilted
    perfect fluid is therefore self-contained in that reduced state only when
    its velocity is aligned with one principal axis; mixed-axis vector tilt
    would source off-diagonal anisotropic stress and needs a larger frame
    state.  ``axis=3`` reproduces the legacy scalar-v3 equation exactly.
    """
    if axis not in (1, 2, 3):
        raise ValueError(f"tilt principal axis must be 1, 2, or 3; got {axis!r}.")
    v_sq = v**2
    gamma_j = jnp.asarray(gamma, dtype=jnp.float64)
    G = 1.0 + (gamma_j - 1.0) * v_sq
    eos_term = 3.0 * gamma_j - 4.0
    friction = (2.0 - q) * (1.0 - (gamma_j - 1.0) * v_sq) / G
    lam1, lam2, lam3 = shear_eigenvalues_tilt(Sigma_plus, Sigma_minus)
    if axis == 1:
        lam = lam1
    elif axis == 2:
        lam = lam2
    else:
        lam = lam3
    bracket = friction + lam - eos_term
    return jnp.where(v_sq < 0.99, v * (1.0 - v_sq) / G * bracket, 0.0)


# ═══════════════════════════════════════════════════════════════
# §3. Vector tilt RHS (3-component, general non-LRS)
# ═══════════════════════════════════════════════════════════════

@jax.jit
def tilt_rhs_vector(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray,
                    q: jnp.ndarray,
                    Sigma_plus: jnp.ndarray,
                    Sigma_minus: jnp.ndarray,
                    gamma: float = GAMMA_RAD):
    """3-component tilt evolution: dv_α/dN.

    Each component couples to its own shear eigenvalue λ_α.
    """
    v_sq = v1**2 + v2**2 + v3**2
    G = 1.0 + (gamma - 1.0) * v_sq
    eos_term = 3.0 * gamma - 4.0
    friction = (2.0 - q) * (1.0 - (gamma - 1.0) * v_sq) / G
    prefactor = (1.0 - v_sq) / G

    lam1, lam2, lam3 = shear_eigenvalues_tilt(Sigma_plus, Sigma_minus)

    suppress = v_sq < 0.99
    dv1 = jnp.where(suppress, v1 * prefactor * (friction + lam1 - eos_term), 0.0)
    dv2 = jnp.where(suppress, v2 * prefactor * (friction + lam2 - eos_term), 0.0)
    dv3 = jnp.where(suppress, v3 * prefactor * (friction + lam3 - eos_term), 0.0)
    return dv1, dv2, dv3


# ═══════════════════════════════════════════════════════════════
# §4. Frame corrections for BBN
# ═══════════════════════════════════════════════════════════════

@jax.jit
def tilt_hubble_factor(v_sq: jnp.ndarray) -> jnp.ndarray:
    """H_tilted / H_orthogonal = 1 / √(1 − v²).

    This is the dominant BBN effect: tilt increases effective
    expansion rate, freezing out more neutrons → higher Y_p.
    """
    return 1.0 / jnp.sqrt(jnp.maximum(1.0 - v_sq, 1e-20))


@jax.jit
def tilted_normal_energy_density_factor(
    v_sq: jnp.ndarray,
    gamma: float = GAMMA_RAD,
) -> jnp.ndarray:
    """Normal-frame T00 factor for a tilted perfect fluid.

    For ``p=(gamma-1) rho`` and fluid 4-velocity
    ``u^a = Gamma(n^a + v^a)``,

        T_ab n^a n^b / rho = gamma Gamma^2 - (gamma - 1)
                            = 1 + gamma Gamma^2 v^2.

    This is distinct from the legacy pure ``Gamma`` Hubble factor.
    """
    gamma_lorentz_sq = 1.0 / jnp.maximum(1.0 - v_sq, 1e-20)
    return 1.0 + jnp.asarray(gamma, dtype=jnp.float64) * gamma_lorentz_sq * v_sq


@jax.jit
def tilt_hubble_stress_energy_factor(
    v_sq: jnp.ndarray,
    gamma: float = GAMMA_RAD,
) -> jnp.ndarray:
    """H/H_orthogonal from the tilted-fluid normal-frame energy density."""
    return jnp.sqrt(tilted_normal_energy_density_factor(v_sq, gamma=gamma))


@jax.jit
def tilt_omega_correction(Omega_orth: jnp.ndarray, v_sq: jnp.ndarray,
                          gamma: float = GAMMA_RAD) -> jnp.ndarray:
    """Tilted Friedmann: Ω_tilt = Ω_orth × G / (G + (2−γ)v²)."""
    G = 1.0 + (gamma - 1.0) * v_sq
    denom = G + (2.0 - gamma) * v_sq
    return Omega_orth * G / jnp.maximum(denom, 1e-20)


@jax.jit
def boosted_fd_legendre_moments(
    q_nodes: jnp.ndarray,
    v: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plasma-frame ``l=0,1,2`` moments of a boosted FD background.

    For a plasma moving with principal-axis speed ``v`` relative to the
    geometry-normal frame, a neutrino with plasma-frame energy ``q*T_nu`` has

        E_geo / T_nu = Gamma * q * (1 + v mu_plasma).

    The returned coefficients follow

        f(mu) = sum_l f_l P_l(mu),  f_l = (2l+1)/2 int dmu f(mu) P_l(mu).

    The monopole reduces exactly to the equilibrium FD spectrum at ``v=0``,
    while the dipole and quadrupole vanish to quadrature precision.
    """
    v_sq = v**2
    gamma_lorentz = 1.0 / jnp.sqrt(jnp.maximum(1.0 - v_sq, 1e-20))
    q_geo = gamma_lorentz * q_nodes[:, None] * (1.0 + v * BOOST_MU16[None, :])
    f_mu = 1.0 / (jnp.exp(jnp.clip(q_geo, 0.0, 500.0)) + 1.0)
    p1 = BOOST_MU16[None, :]
    p2 = 0.5 * (3.0 * BOOST_MU16[None, :] ** 2 - 1.0)
    f0 = 0.5 * jnp.sum(BOOST_W16[None, :] * f_mu, axis=1)
    f1 = 1.5 * jnp.sum(BOOST_W16[None, :] * f_mu * p1, axis=1)
    f2 = 2.5 * jnp.sum(BOOST_W16[None, :] * f_mu * p2, axis=1)
    return jnp.clip(f0, 0.0, 1.0), f1, f2


@jax.jit
def boosted_fd_monopole(q_nodes: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Plasma-frame neutrino monopole of a geometry-frame FD background."""
    f0, _, _ = boosted_fd_legendre_moments(q_nodes, v)
    return f0


@jax.jit
def tilt_anisotropic_stress_scalar(
    v_sq: jnp.ndarray,
    Omega_tilted: jnp.ndarray,
    gamma: float = GAMMA_RAD,
):
    """Perfect-fluid anisotropic stress from scalar tilt along the 3-axis.

    For a tilted perfect fluid,

        π_ab / (3H²) = γ Ω Γ² v_<a v_b>

    with Γ² = 1/(1-v²).  A scalar v₃ tilt has trace-free diagonal
    ``(-1/3, -1/3, +2/3)`` and therefore maps to
    ``Π_+ = γ Ω Γ² v² / 3`` and ``Π_- = 0`` in the diagonal WH basis.
    """
    gamma_lorentz_sq = 1.0 / jnp.maximum(1.0 - v_sq, 1e-20)
    pi_plus = (jnp.asarray(gamma, dtype=jnp.float64) * Omega_tilted * gamma_lorentz_sq * v_sq) / 3.0
    return pi_plus, jnp.array(0.0, dtype=jnp.float64)


@partial(jax.jit, static_argnames=("axis",))
def tilt_anisotropic_stress_principal_axis(
    v_sq: jnp.ndarray,
    Omega_tilted: jnp.ndarray,
    *,
    axis: int = 3,
    gamma: float = GAMMA_RAD,
):
    """Perfect-fluid ``(Π_+, Π_-)`` for principal-axis tilt.

    ``π_ab/(3H²) = γ Ω Γ² (v_a v_b - v² δ_ab/3)``.  Projecting the
    diagonal trace-free tensor onto the WH basis gives:

    axis 1: Π_+ = -C/6, Π_- = -C/(2√3)
    axis 2: Π_+ = -C/6, Π_- = +C/(2√3)
    axis 3: Π_+ = +C/3, Π_- = 0

    where ``C = γ Ω Γ² v²``.
    """
    if axis not in (1, 2, 3):
        raise ValueError(f"tilt principal axis must be 1, 2, or 3; got {axis!r}.")
    gamma_lorentz_sq = 1.0 / jnp.maximum(1.0 - v_sq, 1e-20)
    c = jnp.asarray(gamma, dtype=jnp.float64) * Omega_tilted * gamma_lorentz_sq * v_sq
    if axis == 1:
        return -c / 6.0, -c / (2.0 * SQRT3)
    if axis == 2:
        return -c / 6.0, c / (2.0 * SQRT3)
    return c / 3.0, jnp.array(0.0, dtype=jnp.float64)


# ═══════════════════════════════════════════════════════════════
# §5. FLRW eigenvalue (diagnostic)
# ═══════════════════════════════════════════════════════════════

def tilt_eigenvalue_flrw(gamma: float = GAMMA_RAD) -> float:
    """Linearized eigenvalue at FLRW: λ = 7 − 9γ/2.

    γ = 4/3 (radiation): λ = +1 → GROWS
    γ = 14/9 (critical):  λ = 0 → marginal
    γ = 2 (stiff):        λ = −2 → decays
    """
    return 7.0 - 4.5 * gamma
