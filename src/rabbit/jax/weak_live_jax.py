"""
rabbit.jax.weak_live_jax — JAX-native live weak rates from transport monopoles.

Born-level and CL1/CL2-corrected functionals of transported ν_e and ν̄_e
monopole distributions. Uses the same channel structure and quadrature
conventions as the SciPy live rate path, with a JAX-native not-a-knot cubic
spline on the transport monopole grid. The current end-to-end driver wiring
uses this only in explicitly marked experimental branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from rabbit.jax.weak_jax import (
    M_E_MEV, Q_DIM, fermi_dirac_dimless,
)
from rabbit.weak.quadrature import (
    DEFAULT_WEAK_QUADRATURE,
    FAST_INFERENCE_WEAK_QUADRATURE,
    HIGHRES_WEAK_QUADRATURE,
)
from rabbit.weak.corrections import weak_correction_factor as _weak_correction_factor_np
from rabbit.weak.finite_mass import finite_mass_scalar_correction as _finite_mass_scalar_correction_np

jax.config.update("jax_enable_x64", True)

from rabbit.jax.weak_corrections_jax import weak_correction_factor_jax
from rabbit.jax.weak_finite_mass_jax import (
    finite_mass_angular_coefficients_jax,
    finite_mass_scalar_correction_jax,
    get_I0_cl3_jax,
)


@dataclass(frozen=True)
class LiveWeakBudgetJAX:
    lambda_np: float
    lambda_pn: float
    I0: float
    correction_level: int
    weak_mode: str


@dataclass(frozen=True)
class LiveWeakQuadratureSpecJAX:
    """Metadata for the JAX live-weak quadrature locked at import time."""

    mode: str
    N_laguerre: int
    N_legendre: int


LIVE_WEAK_QUADRATURE_SPEC = LiveWeakQuadratureSpecJAX(
    mode="production",
    N_laguerre=DEFAULT_WEAK_QUADRATURE.N_laguerre,
    N_legendre=DEFAULT_WEAK_QUADRATURE.N_legendre,
)
LIVE_WEAK_VALIDATION_QUADRATURE_SPEC = LiveWeakQuadratureSpecJAX(
    mode="validation",
    N_laguerre=HIGHRES_WEAK_QUADRATURE.N_laguerre,
    N_legendre=HIGHRES_WEAK_QUADRATURE.N_legendre,
)
LIVE_WEAK_FAST_CANDIDATE_QUADRATURE_SPEC = LiveWeakQuadratureSpecJAX(
    mode="inference_fast_candidate",
    N_laguerre=FAST_INFERENCE_WEAK_QUADRATURE.N_laguerre,
    N_legendre=FAST_INFERENCE_WEAK_QUADRATURE.N_legendre,
)

_N_LAG = LIVE_WEAK_QUADRATURE_SPEC.N_laguerre
_N_LEG = LIVE_WEAK_QUADRATURE_SPEC.N_legendre
_lag_x = DEFAULT_WEAK_QUADRATURE.lag_nodes
_lag_w = DEFAULT_WEAK_QUADRATURE.lag_weights
_leg_x = DEFAULT_WEAK_QUADRATURE.leg_nodes
_leg_w = DEFAULT_WEAK_QUADRATURE.leg_weights
LAG_X = np.asarray(_lag_x, dtype=float)
LAG_W = np.asarray(_lag_w, dtype=float)
LEG_X = np.asarray(_leg_x, dtype=float)
LEG_W = np.asarray(_leg_w, dtype=float)
LAG_EXP = np.exp(LAG_X)

# Fixed bounded-channel (c/f) kinematics on the Legendre grid. These are
# temperature-independent, so we precompute them once and reuse them across
# every exact weak-rate RHS evaluation.
CF_JAC = 0.5 * (Q_DIM - 1.0)
CF_EPS_E = 0.5 * (Q_DIM - 1.0) * (LEG_X + 1.0) + 1.0
CF_EPS_NU = Q_DIM - CF_EPS_E
CF_MASK = (CF_EPS_NU > 0.0) & (CF_EPS_E > 1.0)
CF_P_E = np.sqrt(np.maximum(CF_EPS_E ** 2 - 1.0, 1e-100))
CF_PS = CF_EPS_NU**2 * CF_EPS_E * CF_P_E
CF_BASE_WEIGHT = CF_JAC * np.where(CF_MASK, LEG_W * CF_PS, 0.0)


# Locked normalisation integrals for the live-weak kernels.
# These are numerically frozen against the pre-PRR7 reference evaluation so
# the module no longer pays a heavy import-time penalty recomputing them.
_I0_CL0 = 1.6361003656429858
_I0_CL1 = 1.6916641768135807
_I0_CL2 = 1.7171274576843225
_I0_CL3 = get_I0_cl3_jax()  # CL2 (Coulomb+Sirlin) × FM (recoil+WM)

# Fixed correction factors for the bounded c/f channels. For CL1 and CL2 the
# c and f channels share the same Coulomb/Sirlin factors; for CL3 the finite
# mass correction distinguishes them, so keep both arrays.
CF_CORR_CL1 = _weak_correction_factor_np(CF_EPS_E, 'c', True, False)
CF_CL2_C = _weak_correction_factor_np(CF_EPS_E, 'c', True, True)
CF_CL2_F = _weak_correction_factor_np(CF_EPS_E, 'f', True, True)
CF_CORR_CL2 = CF_CL2_C
CF_CORR_CL3_C = CF_CL2_C * _finite_mass_scalar_correction_np(CF_EPS_E, CF_EPS_NU, 'c')
CF_CORR_CL3_F = CF_CL2_F * _finite_mass_scalar_correction_np(CF_EPS_E, CF_EPS_NU, 'f')


@jax.jit
def build_not_a_knot_matrix(q_nodes: jnp.ndarray) -> jnp.ndarray:
    """Linear operator mapping node values y_i to spline second derivatives M_i.

    The cubic interpolator matches SciPy/FITPACK's default not-a-knot choice
    used in the reference live weak path.  For fixed q_nodes this returns the
    matrix P such that M = P @ y.
    """
    n = q_nodes.shape[0]
    h = q_nodes[1:] - q_nodes[:-1]

    A = jnp.zeros((n, n), dtype=q_nodes.dtype)
    bmat = jnp.zeros((n, n), dtype=q_nodes.dtype)

    # Not-a-knot boundaries
    A = A.at[0, 0].set(h[1])
    A = A.at[0, 1].set(-(h[0] + h[1]))
    A = A.at[0, 2].set(h[0])

    A = A.at[n - 1, n - 3].set(-h[-1])
    A = A.at[n - 1, n - 2].set(h[-2] + h[-1])
    A = A.at[n - 1, n - 1].set(-h[-2])

    # Interior spline equations
    idx = jnp.arange(1, n - 1)
    A = A.at[idx, idx - 1].set(h[:-1])
    A = A.at[idx, idx].set(2.0 * (h[:-1] + h[1:]))
    A = A.at[idx, idx + 1].set(h[1:])

    # RHS matrix: b = B @ y
    coeff_im1 = 6.0 / h[:-1]
    coeff_i = -6.0 * (1.0 / h[:-1] + 1.0 / h[1:])
    coeff_ip1 = 6.0 / h[1:]
    bmat = bmat.at[idx, idx - 1].set(coeff_im1)
    bmat = bmat.at[idx, idx].set(coeff_i)
    bmat = bmat.at[idx, idx + 1].set(coeff_ip1)

    return jnp.linalg.solve(A, bmat)


@jax.jit
def _prepare_logit_residual(q_nodes: jnp.ndarray, f_q: jnp.ndarray) -> jnp.ndarray:
    r"""Return δ(q) defined by logit(f(q)) = -q + δ(q).

    Interpolating the logit residual, rather than f(q) directly, makes the
    equilibrium Fermi-Dirac spectrum an exact fixed point of the reconstruction
    for any N_q. This removes the severe low-N_q oscillations seen with the
    previous not-a-knot cubic interpolation on sparse Laguerre nodes, while
    also eliminating the dense O(N_q^2) spline-matrix multiply from the hot
    live-weak path.
    """
    eps = jnp.asarray(1e-14, dtype=f_q.dtype)
    f_safe = jnp.clip(f_q, eps, 1.0 - eps)
    return jnp.log(f_safe) - jnp.log1p(-f_safe) + q_nodes


@jax.jit
def _interp_monopole_epsilon_M(eps_nu: jnp.ndarray, T_nu: jnp.ndarray,
                               q_nodes: jnp.ndarray, f_q: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """Evaluate transported monopoles via linear interpolation of δ(q).

    For backward compatibility with internal call sites the helper name and
    signature are preserved, but ``M`` now stores the logit residual δ(q)
    rather than cubic-spline second derivatives.
    """
    del f_q
    q_eval = eps_nu / jnp.maximum(T_nu, 1e-30)
    idx = jnp.searchsorted(q_nodes, q_eval, side='right') - 1
    idx = jnp.clip(idx, 0, q_nodes.shape[0] - 2)

    x_i = q_nodes[idx]
    x_ip1 = q_nodes[idx + 1]
    h_i = jnp.maximum(x_ip1 - x_i, 1e-30)
    t = (q_eval - x_i) / h_i
    delta_eval = (1.0 - t) * M[idx] + t * M[idx + 1]
    delta_eval = jnp.where(q_eval <= q_nodes[0], M[0], delta_eval)
    delta_eval = jnp.where(q_eval >= q_nodes[-1], 0.0, delta_eval)

    vals = 1.0 / (1.0 + jnp.exp(q_eval - delta_eval))
    return vals


@jax.jit
def _interp_moment_epsilon_linear(
    eps_nu: jnp.ndarray,
    T_nu: jnp.ndarray,
    q_nodes: jnp.ndarray,
    moment_q: jnp.ndarray,
) -> jnp.ndarray:
    """Linearly interpolate a signed neutrino angular moment in q=E_nu/T_nu."""
    q_eval = eps_nu / jnp.maximum(T_nu, 1e-30)
    idx = jnp.searchsorted(q_nodes, q_eval, side='right') - 1
    idx = jnp.clip(idx, 0, q_nodes.shape[0] - 2)

    x_i = q_nodes[idx]
    x_ip1 = q_nodes[idx + 1]
    h_i = jnp.maximum(x_ip1 - x_i, 1e-30)
    t = (q_eval - x_i) / h_i
    vals = (1.0 - t) * moment_q[idx] + t * moment_q[idx + 1]
    vals = jnp.where(q_eval <= q_nodes[0], moment_q[0], vals)
    vals = jnp.where(q_eval >= q_nodes[-1], 0.0, vals)
    return vals


def _finite_mass_moment_occupation(eps_e, eps_nu, channel, f0, f1, f2):
    """Return angular average of K(mu) f(mu) through l=2."""
    K0, K1, K2 = finite_mass_angular_coefficients_jax(eps_e, eps_nu, channel)
    return K0 * f0 + (K1 * f1) / 3.0 + (K2 * f2) / 5.0


def _finite_mass_moment_blocking(eps_e, eps_nu, channel, f0, f1, f2):
    """Return angular average of K(mu) [1-f(mu)] through l=2."""
    K0, K1, K2 = finite_mass_angular_coefficients_jax(eps_e, eps_nu, channel)
    return K0 * (1.0 - f0) - (K1 * f1) / 3.0 - (K2 * f2) / 5.0


@jax.jit
def _channel_a_live(T_e, T_nu, q_nodes, f_nue_q, M_nue):
    eps_nu = LAG_X * T_nu
    eps_e = eps_nu + Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    integrand = ps * f_nu * (1.0 - f_e) * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


@jax.jit
def _channel_b_live(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    eps_e = 1.0 + LAG_X * T_e
    eps_nu = eps_e + Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu_bar = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    integrand = ps * f_e * (1.0 - f_nu_bar) * LAG_EXP
    return T_e * jnp.sum(LAG_W * integrand)


@jax.jit
def _channel_c_live(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    integrand = CF_BASE_WEIGHT * (1.0 - f_e) * (1.0 - f_nu)
    return jnp.sum(integrand)


@jax.jit
def _channel_d_live(T_e, T_nu, q_nodes, f_nue_q, M_nue):
    eps_e = Q_DIM + LAG_X * T_e
    eps_nu = eps_e - Q_DIM
    mask = (eps_nu > 0) & (eps_e > 1.0)
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    integrand = ps * f_e * (1.0 - f_nu) * LAG_EXP
    return T_e * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


@jax.jit
def _channel_e_live(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    eps_nu = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e = eps_nu - Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    integrand = ps * f_nu * (1.0 - f_e) * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


@jax.jit
def _channel_f_live(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    integrand = CF_BASE_WEIGHT * f_e * f_nu
    return jnp.sum(integrand)


@jax.jit
def compute_live_born_rates_from_monopoles(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple:
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    I0 = _I0_CL0
    K = 1.0 / (tau_n * I0)

    del spline_matrix

    M_nue = _prepare_logit_residual(q_nodes, f_nue_monopole)
    M_nuebar = _prepare_logit_residual(q_nodes, f_nuebar_monopole)

    I_a = _channel_a_live(T_e, T_nu, q_nodes, f_nue_monopole, M_nue)
    I_b = _channel_b_live(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_c = _channel_c_live(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_d = _channel_d_live(T_e, T_nu, q_nodes, f_nue_monopole, M_nue)
    I_e = _channel_e_live(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_f = _channel_f_live(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)

    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn


@jax.jit
def compute_live_born_rates_from_shared_monopole(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple:
    """CL0 live weak kernel for the LRS shared-monopole characteristic path."""
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    I0 = _I0_CL0
    K = 1.0 / (tau_n * I0)

    del spline_matrix

    M_shared = _prepare_logit_residual(q_nodes, f_monopole)

    I_a = _channel_a_live(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_b = _channel_b_live(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_c = _channel_c_live(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_d = _channel_d_live(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_e = _channel_e_live(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_f = _channel_f_live(T_e, T_nu, q_nodes, f_monopole, M_shared)

    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, I0


def _channel_a_live_corrected(T_e, T_nu, q_nodes, f_nue_q, M_nue, correction_level):
    eps_nu = LAG_X * T_nu
    eps_e = eps_nu + Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    corr = weak_correction_factor_jax(eps_e, 'a', correction_level >= 1, correction_level >= 2)
    integrand = ps * f_nu * (1.0 - f_e) * corr * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_b_live_corrected(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar, correction_level):
    eps_e = 1.0 + LAG_X * T_e
    eps_nu = eps_e + Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu_bar = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    corr = weak_correction_factor_jax(eps_e, 'b', correction_level >= 1, correction_level >= 2)
    integrand = ps * f_e * (1.0 - f_nu_bar) * corr * LAG_EXP
    return T_e * jnp.sum(LAG_W * integrand)


def _channel_c_live_corrected(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar, correction_level):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    corr = CF_CORR_CL1 if correction_level == 1 else CF_CORR_CL2
    integrand = CF_BASE_WEIGHT * (1.0 - f_e) * (1.0 - f_nu) * corr
    return jnp.sum(integrand)


def _channel_d_live_corrected(T_e, T_nu, q_nodes, f_nue_q, M_nue, correction_level):
    eps_e = Q_DIM + LAG_X * T_e
    eps_nu = eps_e - Q_DIM
    mask = (eps_nu > 0) & (eps_e > 1.0)
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    corr = weak_correction_factor_jax(eps_e, 'd', correction_level >= 1, correction_level >= 2)
    integrand = ps * f_e * (1.0 - f_nu) * corr * LAG_EXP
    return T_e * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_e_live_corrected(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar, correction_level):
    eps_nu = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e = eps_nu - Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    corr = weak_correction_factor_jax(eps_e, 'e', correction_level >= 1, correction_level >= 2)
    integrand = ps * f_nu * (1.0 - f_e) * corr * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_f_live_corrected(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar, correction_level):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    corr = CF_CORR_CL1 if correction_level == 1 else CF_CORR_CL2
    integrand = CF_BASE_WEIGHT * f_e * f_nu * corr
    return jnp.sum(integrand)


# ═══════════════════════════════════════════════════════════════
# CL3 corrected channel functions: CL2 (Coulomb+Sirlin) × FM (recoil+WM)
# ═══════════════════════════════════════════════════════════════

def _cl3_correction(eps_e, eps_nu, channel):
    """Combined CL3 multiplicative correction = CL2 × FM."""
    cl2 = weak_correction_factor_jax(eps_e, channel, True, True)
    fm = finite_mass_scalar_correction_jax(eps_e, eps_nu, channel)
    return cl2 * fm


def _channel_a_live_cl3(T_e, T_nu, q_nodes, f_nue_q, M_nue):
    eps_nu = LAG_X * T_nu
    eps_e = eps_nu + Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    corr = _cl3_correction(eps_e, eps_nu, 'a')
    integrand = ps * f_nu * (1.0 - f_e) * corr * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_b_live_cl3(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    eps_e = 1.0 + LAG_X * T_e
    eps_nu = eps_e + Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu_bar = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    corr = _cl3_correction(eps_e, eps_nu, 'b')
    integrand = ps * f_e * (1.0 - f_nu_bar) * corr * LAG_EXP
    return T_e * jnp.sum(LAG_W * integrand)


def _channel_c_live_cl3(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    integrand = CF_BASE_WEIGHT * (1.0 - f_e) * (1.0 - f_nu) * CF_CORR_CL3_C
    return jnp.sum(integrand)


def _channel_d_live_cl3(T_e, T_nu, q_nodes, f_nue_q, M_nue):
    eps_e = Q_DIM + LAG_X * T_e
    eps_nu = eps_e - Q_DIM
    mask = (eps_nu > 0) & (eps_e > 1.0)
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nue_q, M_nue)
    corr = _cl3_correction(eps_e, eps_nu, 'd')
    integrand = ps * f_e * (1.0 - f_nu) * corr * LAG_EXP
    return T_e * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_e_live_cl3(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    eps_nu = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e = eps_nu - Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_nu = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    corr = _cl3_correction(eps_e, eps_nu, 'e')
    integrand = ps * f_nu * (1.0 - f_e) * corr * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_f_live_cl3(T_e, T_nu, q_nodes, f_nuebar_q, M_nuebar):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nu = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f_nuebar_q, M_nuebar)
    integrand = CF_BASE_WEIGHT * f_e * f_nu * CF_CORR_CL3_F
    return jnp.sum(integrand)


def _channel_a_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    eps_nu = LAG_X * T_nu
    eps_e = eps_nu + Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f0 = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f2_q)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    cl2 = weak_correction_factor_jax(eps_e, 'a', True, True)
    fm_occ = _finite_mass_moment_occupation(eps_e, eps_nu, 'a', f0, f1, f2)
    integrand = ps * fm_occ * (1.0 - f_e) * cl2 * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_b_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    eps_e = 1.0 + LAG_X * T_e
    eps_nu = eps_e + Q_DIM
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f0 = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f2_q)
    cl2 = weak_correction_factor_jax(eps_e, 'b', True, True)
    fm_block = _finite_mass_moment_blocking(eps_e, eps_nu, 'b', f0, f1, f2)
    integrand = ps * f_e * fm_block * cl2 * LAG_EXP
    return T_e * jnp.sum(LAG_W * integrand)


def _channel_c_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f0 = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(CF_EPS_NU, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(CF_EPS_NU, T_nu, q_nodes, f2_q)
    fm_block = _finite_mass_moment_blocking(CF_EPS_E, CF_EPS_NU, 'c', f0, f1, f2)
    integrand = CF_BASE_WEIGHT * (1.0 - f_e) * fm_block * CF_CL2_C
    return jnp.sum(integrand)


def _channel_d_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    eps_e = Q_DIM + LAG_X * T_e
    eps_nu = eps_e - Q_DIM
    mask = (eps_nu > 0) & (eps_e > 1.0)
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f_e = fermi_dirac_dimless(eps_e, T_e)
    f0 = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f2_q)
    cl2 = weak_correction_factor_jax(eps_e, 'd', True, True)
    fm_block = _finite_mass_moment_blocking(eps_e, eps_nu, 'd', f0, f1, f2)
    integrand = ps * f_e * fm_block * cl2 * LAG_EXP
    return T_e * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_e_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    eps_nu = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e = eps_nu - Q_DIM
    mask = eps_e > 1.0
    p_e = jnp.sqrt(jnp.maximum(eps_e ** 2 - 1.0, 1e-100))
    ps = eps_nu**2 * eps_e * p_e
    f0 = _interp_monopole_epsilon_M(eps_nu, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(eps_nu, T_nu, q_nodes, f2_q)
    f_e = fermi_dirac_dimless(eps_e, T_e)
    cl2 = weak_correction_factor_jax(eps_e, 'e', True, True)
    fm_occ = _finite_mass_moment_occupation(eps_e, eps_nu, 'e', f0, f1, f2)
    integrand = ps * fm_occ * (1.0 - f_e) * cl2 * LAG_EXP
    return T_nu * jnp.sum(jnp.where(mask, LAG_W * integrand, 0.0))


def _channel_f_live_cl3_moments(T_e, T_nu, q_nodes, f0_q, f1_q, f2_q, M0):
    f_e = fermi_dirac_dimless(CF_EPS_E, T_e)
    f0 = _interp_monopole_epsilon_M(CF_EPS_NU, T_nu, q_nodes, f0_q, M0)
    f1 = _interp_moment_epsilon_linear(CF_EPS_NU, T_nu, q_nodes, f1_q)
    f2 = _interp_moment_epsilon_linear(CF_EPS_NU, T_nu, q_nodes, f2_q)
    fm_occ = _finite_mass_moment_occupation(CF_EPS_E, CF_EPS_NU, 'f', f0, f1, f2)
    integrand = CF_BASE_WEIGHT * f_e * fm_occ * CF_CL2_F
    return jnp.sum(integrand)


@jax.jit
def compute_live_rates_from_monopoles_cl0_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL0 live weak kernel.

    This exists so the driver can lock CL0 into its own JAX compilation/cache
    bucket rather than relying on a dynamic correction-level branch.
    """
    lambda_np, lambda_pn = compute_live_born_rates_from_monopoles(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix=spline_matrix
    )
    return lambda_np, lambda_pn, _I0_CL0


@partial(jax.jit, static_argnames=("correction_level",))
def _compute_live_rates_from_monopoles_corrected_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    correction_level: int,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL1/CL2 live weak kernel.

    correction_level must be 1 or 2.  Keeping the level static here avoids
    tracer-driven branching inside the expensive corrected path.
    """
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    del spline_matrix
    M_nue = _prepare_logit_residual(q_nodes, f_nue_monopole)
    M_nuebar = _prepare_logit_residual(q_nodes, f_nuebar_monopole)

    I0 = _I0_CL1 if correction_level == 1 else _I0_CL2
    K = 1.0 / (tau_n * I0)
    I_a = _channel_a_live_corrected(T_e, T_nu, q_nodes, f_nue_monopole, M_nue, correction_level)
    I_b = _channel_b_live_corrected(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar, correction_level)
    I_c = _channel_c_live_corrected(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar, correction_level)
    I_d = _channel_d_live_corrected(T_e, T_nu, q_nodes, f_nue_monopole, M_nue, correction_level)
    I_e = _channel_e_live_corrected(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar, correction_level)
    I_f = _channel_f_live_corrected(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar, correction_level)
    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, I0


def _compute_live_rates_from_shared_monopole_corrected_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_monopole: jnp.ndarray,
    correction_level: int,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL1/CL2 live weak kernel for a shared monopole."""
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    del spline_matrix
    M_shared = _prepare_logit_residual(q_nodes, f_monopole)

    I0 = _I0_CL1 if correction_level == 1 else _I0_CL2
    K = 1.0 / (tau_n * I0)
    I_a = _channel_a_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    I_b = _channel_b_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    I_c = _channel_c_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    I_d = _channel_d_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    I_e = _channel_e_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    I_f = _channel_f_live_corrected(T_e, T_nu, q_nodes, f_monopole, M_shared, correction_level)
    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, I0


@jax.jit
def compute_live_rates_from_monopoles_cl1_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL1 live weak kernel."""
    return _compute_live_rates_from_monopoles_corrected_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, 1, spline_matrix
    )


@jax.jit
def compute_live_rates_from_shared_monopole_cl1_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized shared-monopole CL1 live weak kernel."""
    return _compute_live_rates_from_shared_monopole_corrected_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_monopole, 1, spline_matrix
    )


@jax.jit
def compute_live_rates_from_monopoles_cl2_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL2 live weak kernel."""
    return _compute_live_rates_from_monopoles_corrected_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, 2, spline_matrix
    )


@jax.jit
def compute_live_rates_from_shared_monopole_cl2_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized shared-monopole CL2 live weak kernel."""
    return _compute_live_rates_from_shared_monopole_corrected_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_monopole, 2, spline_matrix
    )


@jax.jit
def compute_live_rates_from_monopoles_cl3_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized CL3 live weak kernel.

    CL3 = CL2 (Coulomb + Sirlin) × FM (recoil + weak magnetism).
    Uses dedicated CL3 channel functions with the full correction chain
    composed inside the integrand, matching the SciPy reference contract.
    """
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    del spline_matrix
    M_nue = _prepare_logit_residual(q_nodes, f_nue_monopole)
    M_nuebar = _prepare_logit_residual(q_nodes, f_nuebar_monopole)

    K = 1.0 / (tau_n * _I0_CL3)
    I_a = _channel_a_live_cl3(T_e, T_nu, q_nodes, f_nue_monopole, M_nue)
    I_b = _channel_b_live_cl3(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_c = _channel_c_live_cl3(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_d = _channel_d_live_cl3(T_e, T_nu, q_nodes, f_nue_monopole, M_nue)
    I_e = _channel_e_live_cl3(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    I_f = _channel_f_live_cl3(T_e, T_nu, q_nodes, f_nuebar_monopole, M_nuebar)
    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, _I0_CL3


@jax.jit
def compute_live_rates_from_moments_cl3_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_0: jnp.ndarray,
    f_nue_1: jnp.ndarray,
    f_nue_2: jnp.ndarray,
    f_nuebar_0: jnp.ndarray,
    f_nuebar_1: jnp.ndarray,
    f_nuebar_2: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """CL3 live weak kernel coupled to ``l=0,1,2`` neutrino moments.

    The angular finite-mass kernel is projected as
    ``K0*f0 + K1*f1/3 + K2*f2/5`` for incoming neutrinos and the matching
    ``K0*(1-f0) - K1*f1/3 - K2*f2/5`` final-state blocking factor.  This is
    intended for principal-axis boosted-FD closures; it is not a replacement
    for a full angle-dependent collision solver.
    """
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    del spline_matrix
    M_nue_0 = _prepare_logit_residual(q_nodes, f_nue_0)
    M_nuebar_0 = _prepare_logit_residual(q_nodes, f_nuebar_0)

    K = 1.0 / (tau_n * _I0_CL3)
    I_a = _channel_a_live_cl3_moments(T_e, T_nu, q_nodes, f_nue_0, f_nue_1, f_nue_2, M_nue_0)
    I_b = _channel_b_live_cl3_moments(T_e, T_nu, q_nodes, f_nuebar_0, f_nuebar_1, f_nuebar_2, M_nuebar_0)
    I_c = _channel_c_live_cl3_moments(T_e, T_nu, q_nodes, f_nuebar_0, f_nuebar_1, f_nuebar_2, M_nuebar_0)
    I_d = _channel_d_live_cl3_moments(T_e, T_nu, q_nodes, f_nue_0, f_nue_1, f_nue_2, M_nue_0)
    I_e = _channel_e_live_cl3_moments(T_e, T_nu, q_nodes, f_nuebar_0, f_nuebar_1, f_nuebar_2, M_nuebar_0)
    I_f = _channel_f_live_cl3_moments(T_e, T_nu, q_nodes, f_nuebar_0, f_nuebar_1, f_nuebar_2, M_nuebar_0)
    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, _I0_CL3


@jax.jit
def compute_live_rates_from_shared_monopole_cl3_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Level-specialized shared-monopole CL3 live weak kernel."""
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV

    del spline_matrix
    M_shared = _prepare_logit_residual(q_nodes, f_monopole)

    K = 1.0 / (tau_n * _I0_CL3)
    I_a = _channel_a_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_b = _channel_b_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_c = _channel_c_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_d = _channel_d_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_e = _channel_e_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    I_f = _channel_f_live_cl3(T_e, T_nu, q_nodes, f_monopole, M_shared)
    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, _I0_CL3


def compute_live_rates_from_monopoles_level_specialized_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    correction_level: int = 0,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dispatch to a correction-level-specialized live weak kernel.

    This function is intentionally *not* jitted; it selects one of the already
    JIT-compiled level-specialized kernels so CL0/CL1/CL2/CL3 each occupy a
    stable cache bucket.
    """
    if correction_level == 0:
        return compute_live_rates_from_monopoles_cl0_jax(
            T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix
        )
    if correction_level == 1:
        return compute_live_rates_from_monopoles_cl1_jax(
            T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix
        )
    if correction_level == 2:
        return compute_live_rates_from_monopoles_cl2_jax(
            T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix
        )
    if correction_level == 3:
        return compute_live_rates_from_monopoles_cl3_jax(
            T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix
        )
    raise ValueError("JAX correction substrate currently supports correction_level ∈ {0,1,2,3}.")


def compute_live_rates_from_monopoles_cl012_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    correction_level: int = 0,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backwards-compatible CL0/1/2 live weak dispatch wrapper.

    New code should prefer `compute_live_rates_from_monopoles_level_specialized_jax`
    when it wants explicit cache separation by correction level.
    """
    return compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole,
        correction_level=correction_level, spline_matrix=spline_matrix,
    )


def compute_live_rates_with_budget_from_monopoles(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    correction_level: int = 0,
    spline_matrix: jnp.ndarray | None = None,
) -> LiveWeakBudgetJAX:
    """CL0/1/2/3 live weak budget on transported ν_e/ν̄_e monopoles.

    This keeps the same live-functional semantics as the SciPy reference while
    making the correction hierarchy explicit. The returned budget is a compact
    substrate object for future all-type weak kernels.
    """
    if correction_level not in (0, 1, 2, 3):
        raise ValueError("JAX correction substrate currently supports correction_level ∈ {0,1,2,3}.")

    lnp, lpn, I0 = compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma_MeV, T_nu_MeV, tau_n, q_nodes, f_nue_monopole, f_nuebar_monopole,
        correction_level=correction_level, spline_matrix=spline_matrix,
    )
    return LiveWeakBudgetJAX(
        lambda_np=float(lnp),
        lambda_pn=float(lpn),
        I0=float(I0),
        correction_level=int(correction_level),
        weak_mode=f"live_f0_cl{correction_level}_substrate",
    )
