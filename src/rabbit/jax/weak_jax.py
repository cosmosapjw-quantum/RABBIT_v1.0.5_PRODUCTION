"""
rabbit.jax.weak_jax — JAX-native Born-level weak n↔p rates.

Ports from: rabbit.weak.live_rates + rabbit.weak.channels
Scope: Born level (CL0) only. CL1–CL3 corrections deferred to J-EXT.

The full weak rates are functionals of the neutrino monopole distribution
f₀(q), but this JAX module currently implements only the Born-level
equilibrium-FD approximation parameterized by T_ν. It does NOT yet accept
an arbitrary distorted monopole f₀(q).

Six channels:
  n→p: (a) ν_e+n→p+e⁻  (b) e⁺+n→p+ν̄_e  (c) n→p+e⁻+ν̄_e
  p→n: (d) p+e⁻→n+ν_e  (e) p+ν̄_e→n+e⁺  (f) p+e⁻+ν̄_e→n

Normalization: K = 1/(τ_n × I₀), I₀ ≈ 1.63609 (neutron decay integral).
"""
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants (m_e units)
# ═══════════════════════════════════════════════════════════════════════

M_E_MEV = 0.5109989500
Q_NP_MEV = 1.29333236
Q_DIM = Q_NP_MEV / M_E_MEV    # ≈ 2.5310 (dimensionless mass splitting)
TAU_N = 878.4                   # neutron lifetime [s]

# Gauss-Legendre quadrature for channel integrals (cold path)
import numpy as _np
_N_LEG = 64
_leg_x, _leg_w = _np.polynomial.legendre.leggauss(_N_LEG)
LEG_X = jnp.array(_leg_x)
LEG_W = jnp.array(_leg_w)


# ═══════════════════════════════════════════════════════════════════════
# §2. I₀ normalization integral
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def compute_I0_born(q: float = Q_DIM) -> jnp.ndarray:
    """Neutron decay phase-space integral I₀.

    I₀ = ∫₁^q dε_e ε_e (q−ε_e)² √(ε_e²−1)
    """
    # Map [−1,1] → [1, q]
    eps_e = 0.5 * (q - 1.0) * (LEG_X + 1.0) + 1.0
    jac = 0.5 * (q - 1.0)

    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    integrand = eps_e * (q - eps_e) ** 2 * p_e
    return jnp.sum(LEG_W * integrand) * jac


# ═══════════════════════════════════════════════════════════════════════
# §3. Fermi-Dirac distributions
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def fermi_dirac_dimless(eps: jnp.ndarray, T_dimless: jnp.ndarray) -> jnp.ndarray:
    """Standard FD: f(ε) = 1/(exp(ε/T)+1), T in m_e units."""
    arg = jnp.minimum(eps / jnp.maximum(T_dimless, 1e-30), 500.0)
    return 1.0 / (jnp.exp(arg) + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §4. Channel integrals (Born level, equilibrium FD distributions)
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def _channel_a_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (a): ν_e + n → p + e⁻.  ε_ν semi-infinite.

    I_a = ∫₀^∞ dε_ν  ε_ν² ε_e p_e f_ν(ε_ν)(1−f_e(ε_e))
    where ε_e = ε_ν + q.
    """
    # Map [−1,1] → [0, E_max] with E_max ~ 20T_ν
    E_max = jnp.maximum(20.0 * T_nu, 30.0)
    eps_nu = 0.5 * E_max * (LEG_X + 1.0)
    jac = 0.5 * E_max

    eps_e = eps_nu + Q_DIM
    p_e_sq = eps_e ** 2 - 1.0
    p_e = jnp.sqrt(jnp.maximum(p_e_sq, 1e-100))

    f_nu = fermi_dirac_dimless(eps_nu, T_nu)
    f_e = fermi_dirac_dimless(eps_e, T_e)

    integrand = jnp.where(
        p_e_sq > 0,
        eps_nu ** 2 * eps_e * p_e * f_nu * (1.0 - f_e),
        0.0,
    )
    return jnp.sum(LEG_W * integrand) * jac


@jax.jit
def _channel_b_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (b): e⁺ + n → p + ν̄_e.  ε_e semi-infinite, ε_e > 1.

    I_b = ∫₁^∞ dε_e  ε_e p_e ε_ν² f_e(ε_e)(1−f_ν(ε_ν))
    where ε_ν = ε_e + q.
    """
    E_max = jnp.maximum(20.0 * T_e, 30.0)
    eps_e = 0.5 * E_max * (LEG_X + 1.0) + 1.0
    jac = 0.5 * E_max

    eps_nu = eps_e + Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))

    f_e = fermi_dirac_dimless(eps_e, T_e)     # positron distribution
    f_nu_bar = fermi_dirac_dimless(eps_nu, T_nu)

    integrand = eps_e * p_e * eps_nu ** 2 * f_e * (1.0 - f_nu_bar)
    return jnp.sum(LEG_W * integrand) * jac


@jax.jit
def _channel_c_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (c): n → p + e⁻ + ν̄_e.  Bounded: ε_e ∈ [1, q].

    I_c = ∫₁^q dε_e  ε_e p_e ε_ν² (1−f_e)(1−f_ν)
    where ε_ν = q − ε_e.
    """
    eps_e = 0.5 * (Q_DIM - 1.0) * (LEG_X + 1.0) + 1.0
    jac = 0.5 * (Q_DIM - 1.0)

    eps_nu = Q_DIM - eps_e
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))

    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = fermi_dirac_dimless(eps_nu, T_nu)

    integrand = jnp.where(
        eps_nu > 0,
        eps_e * p_e * eps_nu ** 2 * (1.0 - f_e) * (1.0 - f_nu),
        0.0,
    )
    return jnp.sum(LEG_W * integrand) * jac


@jax.jit
def _channel_d_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (d): p + e⁻ → n + ν_e.  ε_e > q (threshold)."""
    E_max = jnp.maximum(20.0 * T_e, 30.0)
    eps_e = 0.5 * E_max * (LEG_X + 1.0) + Q_DIM
    jac = 0.5 * E_max

    eps_nu = eps_e - Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))

    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = fermi_dirac_dimless(eps_nu, T_nu)

    integrand = jnp.where(
        eps_nu > 0,
        eps_e * p_e * eps_nu ** 2 * f_e * (1.0 - f_nu),
        0.0,
    )
    return jnp.sum(LEG_W * integrand) * jac


@jax.jit
def _channel_e_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (e): p + ν̄_e → n + e⁺.  ε_ν > q+1 (threshold for e⁺ creation)."""
    E_max = jnp.maximum(20.0 * T_nu, 30.0)
    eps_nu = 0.5 * E_max * (LEG_X + 1.0) + Q_DIM + 1.0
    jac = 0.5 * E_max

    eps_e = eps_nu - Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))

    f_nu = fermi_dirac_dimless(eps_nu, T_nu)
    f_e = fermi_dirac_dimless(eps_e, T_e)

    integrand = jnp.where(
        eps_e > 1.0,
        eps_nu ** 2 * eps_e * p_e * f_nu * (1.0 - f_e),
        0.0,
    )
    return jnp.sum(LEG_W * integrand) * jac


@jax.jit
def _channel_f_integral(T_e: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    """Channel (f): p + e⁻ + ν̄_e → n.  Same kinematics as (c), reverse."""
    eps_e = 0.5 * (Q_DIM - 1.0) * (LEG_X + 1.0) + 1.0
    jac = 0.5 * (Q_DIM - 1.0)

    eps_nu = Q_DIM - eps_e
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))

    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = fermi_dirac_dimless(eps_nu, T_nu)

    integrand = jnp.where(
        eps_nu > 0,
        eps_e * p_e * eps_nu ** 2 * f_e * f_nu,
        0.0,
    )
    return jnp.sum(LEG_W * integrand) * jac


# ═══════════════════════════════════════════════════════════════════════
# §5. Total rates (Born level)
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def compute_born_rates(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
) -> tuple:
    """Compute Born-level n↔p rates [s⁻¹].

    Parameters
    ----------
    T_gamma_MeV : float
        Photon temperature [MeV].
    T_nu_MeV : float
        Neutrino temperature [MeV].
    tau_n : float
        Neutron lifetime [s].

    Returns
    -------
    (lambda_np, lambda_pn) in s⁻¹.
    """
    # Convert to m_e units
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    # Normalization: K = 1/(τ_n × I₀)
    I0 = compute_I0_born()
    K = 1.0 / (tau_n * I0)

    # n→p channels
    I_a = _channel_a_integral(T_e, T_nu)
    I_b = _channel_b_integral(T_e, T_nu)
    I_c = _channel_c_integral(T_e, T_nu)

    # p→n channels
    I_d = _channel_d_integral(T_e, T_nu)
    I_e = _channel_e_integral(T_e, T_nu)
    I_f = _channel_f_integral(T_e, T_nu)

    lambda_np = K * (I_a + I_b + I_c)
    lambda_pn = K * (I_d + I_e + I_f)

    return lambda_np, lambda_pn


@jax.jit
def equilibrium_Xn(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
) -> jnp.ndarray:
    """Equilibrium neutron fraction X_n = λ_pn/(λ_np + λ_pn)."""
    lnp, lpn = compute_born_rates(T_gamma_MeV, T_nu_MeV, tau_n)
    return lpn / jnp.maximum(lnp + lpn, 1e-100)
