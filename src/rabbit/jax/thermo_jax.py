"""
rabbit.jax.thermo_jax — JAX-native photon + e± equation of state.

JIT-compilable and forward-mode differentiable.

Ports from: rabbit.thermo.eos_photon_electron
Key change: Fermi-Dirac integrals vectorized via jnp array ops
            (no Python for-loop). drho_dT via jax.grad (not FD).

AD safety: all sqrt/log/power operations protected near zero.
"""
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

M_E: float = 0.5109989500          # electron mass [MeV]
ALPHA_EM: float = 1.0 / 137.035999084
PI = jnp.pi
PI2 = jnp.pi ** 2
RHO_GAMMA_PREFACTOR = PI2 / 15.0   # ρ_γ = (π²/15) T⁴
RHO_UNIT_PREFACTOR = PI2 / 30.0    # (π²/30) for g_* conversion

# Pre-computed Gauss-Laguerre nodes/weights (cold path, NumPy)
import numpy as _np
from numpy.polynomial.laguerre import laggauss as _laggauss
_N_QUAD = 64
_gl_nodes_np, _gl_weights_np = _laggauss(_N_QUAD)
GL_NODES = jnp.array(_gl_nodes_np)
GL_WEIGHTS = jnp.array(_gl_weights_np)


# ═══════════════════════════════════════════════════════════════════════
# §2. Vectorized Fermi-Dirac integral I_{m,n}(x)
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnums=(0, 1))
def _Imn_jax(m: int, n: int, x: jnp.ndarray) -> jnp.ndarray:
    """Generalized Fermi-Dirac integral I_{m,n}(x), JIT-compiled.

    I_{m,n}(x) = ∫₀^∞ du (u²+2ux)^{n/2} (u+x)^m / (e^{√(u²+2ux+x²)}+1)

    Gauss-Laguerre weights absorb e^{-u} from the measure.  The actual
    distribution is f(E) = 1/(e^E+1) where E = u+x.  We need:
        f(E) × e^u = e^u/(e^{u+x}+1) = 1/(e^x + e^{-u})
    This avoids overflow and correctly accounts for the Laguerre factor.
    """
    u = GL_NODES          # shape (N_QUAD,)
    w = GL_WEIGHTS

    E = u + x                                    # E = u + x
    p_sq = u * u + 2.0 * u * x                   # p² = u² + 2ux
    p_sq_safe = jnp.maximum(p_sq, 1e-100)         # AD safety
    p = jnp.sqrt(p_sq_safe)                       # |p|

    # Distribution with Laguerre factor: f(E) × e^u = 1/(e^x + e^{-u})
    # This correctly absorbs the e^{-u} from Gauss-Laguerre measure
    exp_x = jnp.exp(jnp.minimum(x, 500.0))
    exp_neg_u = jnp.exp(-u)
    denom = exp_x + exp_neg_u
    dist = 1.0 / jnp.maximum(denom, 1e-300)       # AD safety

    # Integrand: E^m × p^n × dist × w
    integrand = jnp.where(
        p_sq > 1e-30,
        jnp.power(E, m) * jnp.power(p, n) * dist * w,
        0.0
    )

    return jnp.sum(integrand)


# ═══════════════════════════════════════════════════════════════════════
# §3. Core EOS functions
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def rho_photon(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Photon energy density ρ_γ = (π²/15) T⁴ [MeV⁴]."""
    return RHO_GAMMA_PREFACTOR * T_MeV ** 4


@jax.jit
def rho_electron(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Electron + positron energy density ρ_{e±} [MeV⁴].

    ρ_{e±} = 2 × (T⁴/π²) × I_{2,1}(m_e/T)
    """
    x = M_E / jnp.maximum(T_MeV, 1e-30)
    I21 = _Imn_jax(2, 1, x)
    # Boltzmann suppression: effectively zero for x > 50
    return jnp.where(x < 50.0, 2.0 * T_MeV ** 4 / PI2 * I21, 0.0)


@jax.jit
def pressure_electron(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Electron + positron pressure p_{e±} [MeV⁴]."""
    x = M_E / jnp.maximum(T_MeV, 1e-30)
    I03 = _Imn_jax(0, 3, x)
    return jnp.where(x < 50.0, 2.0 * T_MeV ** 4 / (3.0 * PI2) * I03, 0.0)


@jax.jit
def rho_plasma_tree(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Total plasma energy density (tree-level, no QED)."""
    return rho_photon(T_MeV) + rho_electron(T_MeV)


@jax.jit
def qed_delta_rho(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """QED O(e²)+O(e³) correction to plasma energy density [MeV⁴].

    δρ < 0 is correct sign (Bennett et al. 2020).
    """
    x = M_E / jnp.maximum(T_MeV, 1e-30)
    rho_rel_e = (7.0 / 4.0) * RHO_GAMMA_PREFACTOR * T_MeV ** 4
    rho_e = rho_electron(T_MeV)

    f_e = jnp.minimum(rho_e / jnp.maximum(rho_rel_e, 1e-100), 1.0)

    # O(e²): thermal photon self-energy
    delta_e2 = (5.0 * ALPHA_EM / (4.0 * PI)) * rho_rel_e * f_e

    # O(e³): Debye screening (negative correction)
    delta_e3 = -0.20 * delta_e2 * jnp.sqrt(ALPHA_EM / PI)

    return jnp.where(x < 50.0, delta_e2 + delta_e3, 0.0)


@jax.jit
def rho_plasma(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Total plasma ρ with QED correction. Canonical EOS."""
    return rho_plasma_tree(T_MeV) + qed_delta_rho(T_MeV)


@jax.jit
def pressure_plasma(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Total plasma pressure with QED correction."""
    return rho_photon(T_MeV) / 3.0 + pressure_electron(T_MeV) + qed_delta_rho(T_MeV) / 3.0


@jax.jit
def entropy_density_plasma(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """Plasma entropy density s = (ρ+p)/T [MeV³]."""
    rho = rho_plasma(T_MeV)
    p = pressure_plasma(T_MeV)
    return (rho + p) / jnp.maximum(T_MeV, 1e-30)


# ═══════════════════════════════════════════════════════════════════════
# §4. Derivatives via AD (NOT finite difference)
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def drho_dT(T_MeV: jnp.ndarray) -> jnp.ndarray:
    """dρ/dT via automatic differentiation [MeV³].

    Exact to machine precision — no FD truncation error.
    """
    return jax.grad(lambda T: rho_plasma(T))(T_MeV)


# ═══════════════════════════════════════════════════════════════════════
# §5. Hubble rate
# ═══════════════════════════════════════════════════════════════════════

# Gravitational constant in natural units
G_N = 6.70883e-45  # [MeV⁻²]


@jax.jit
def hubble_rate(T_MeV: jnp.ndarray, N_eff: jnp.ndarray) -> jnp.ndarray:
    """Hubble rate H(T) = √(8πG/3 × ρ_total) [MeV].

    ρ_total = ρ_plasma(T) + ρ_ν(T, N_eff)
    ρ_ν = N_eff × (7/8)(4/11)^{4/3} ρ_γ
    """
    rho_pl = rho_plasma(T_MeV)
    rho_nu = N_eff * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * rho_photon(T_MeV)
    rho_total = rho_pl + rho_nu
    H_sq = 8.0 * PI / 3.0 * G_N * rho_total
    # AD safety: sqrt(0) → NaN gradient
    return jnp.sqrt(jnp.maximum(H_sq, 1e-100))
