"""
rabbit.jax.weak_live_fused — Fused 6-channel live weak rate kernel.

GPU-optimized: single JIT trace for all 6 n↔p channels.
Replaces 6 separate @jit channel functions with one fused kernel
that shares phase-space computations and batches interpolation.

Parity: matches weak_live_jax.py Born-level rates to <1e-12.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.jax.weak_jax import M_E_MEV, Q_DIM, fermi_dirac_dimless
from rabbit.jax.weak_live_jax import (
    _prepare_logit_residual,
    _interp_monopole_epsilon_M,
    _I0_CL0,
    _I0_CL1,
    _I0_CL2,
    _I0_CL3,
    LAG_X, LAG_W, LAG_EXP,
    LEG_X, LEG_W,
    CF_BASE_WEIGHT,
    CF_EPS_E,
    CF_EPS_NU,
    CF_CORR_CL1,
    CF_CORR_CL2,
    CF_CORR_CL3_C,
    CF_CORR_CL3_F,
)
from rabbit.jax.weak_corrections_jax import weak_correction_factor_jax
from rabbit.jax.weak_finite_mass_jax import finite_mass_scalar_correction_jax

# ═══════════════════════════════════════════════════════════════
# §1. Fused Born-level 6-channel kernel
# ═══════════════════════════════════════════════════════════════

@jax.jit
def fused_live_born_rates(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue: jnp.ndarray,
    f_nuebar: jnp.ndarray,
    spline_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute all 6 n↔p channels in a single fused JIT trace.

    Returns (lambda_np, lambda_pn) in [s⁻¹].
    """
    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV
    K = 1.0 / (tau_n * _I0_CL0)

    # ``spline_matrix`` is retained for backwards signature compatibility.
    # The live weak substrate now interpolates the logit residual δ(q), not
    # spline second derivatives, so the old matrix argument is intentionally
    # ignored by the fused kernel.
    del spline_matrix
    M_nue = _prepare_logit_residual(q_nodes, f_nue)
    M_nuebar = _prepare_logit_residual(q_nodes, f_nuebar)

    # ── Laguerre channels (a, b, d, e) — semi-infinite integrals ──

    # Channel a: ν_e + n → e⁻ + p  (Laguerre in ε_ν)
    eps_nu_a = LAG_X * T_nu
    eps_e_a = eps_nu_a + Q_DIM
    mask_a = eps_e_a > 1.0
    p_e_a = jnp.sqrt(jnp.maximum(eps_e_a**2 - 1.0, 1e-100))
    ps_a = eps_nu_a**2 * eps_e_a * p_e_a
    f_nu_a = _interp_monopole_epsilon_M(eps_nu_a, T_nu, q_nodes, f_nue, M_nue)
    f_e_a = fermi_dirac_dimless(eps_e_a, T_e)
    I_a = T_nu * jnp.sum(jnp.where(mask_a, LAG_W * ps_a * f_nu_a * (1.0 - f_e_a) * LAG_EXP, 0.0))

    # Channel b: e⁺ + n → ν̄_e + p  (Laguerre in ε_e from threshold)
    eps_e_b = 1.0 + LAG_X * T_e
    eps_nu_b = eps_e_b + Q_DIM
    p_e_b = jnp.sqrt(jnp.maximum(eps_e_b**2 - 1.0, 1e-100))
    ps_b = eps_nu_b**2 * eps_e_b * p_e_b
    f_e_b = fermi_dirac_dimless(eps_e_b, T_e)
    f_nubar_b = _interp_monopole_epsilon_M(eps_nu_b, T_nu, q_nodes, f_nuebar, M_nuebar)
    I_b = T_e * jnp.sum(LAG_W * ps_b * f_e_b * (1.0 - f_nubar_b) * LAG_EXP)

    # Channel d: ν̄_e + e⁻ + p → n  (Laguerre in ε_e from Q_DIM)
    eps_e_d = Q_DIM + LAG_X * T_e
    eps_nu_d = eps_e_d - Q_DIM
    mask_d = (eps_nu_d > 0.0) & (eps_e_d > 1.0)
    p_e_d = jnp.sqrt(jnp.maximum(eps_e_d**2 - 1.0, 1e-100))
    ps_d = eps_nu_d**2 * eps_e_d * p_e_d
    f_e_d = fermi_dirac_dimless(eps_e_d, T_e)
    f_nu_d = _interp_monopole_epsilon_M(eps_nu_d, T_nu, q_nodes, f_nue, M_nue)
    I_d = T_e * jnp.sum(jnp.where(mask_d, LAG_W * ps_d * f_e_d * (1.0 - f_nu_d) * LAG_EXP, 0.0))

    # Channel e: ν̄_e + p → e⁺ + n  (Laguerre in ε_ν from Q_DIM+1)
    eps_nu_e = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e_e = eps_nu_e - Q_DIM
    mask_e = eps_e_e > 1.0
    p_e_e = jnp.sqrt(jnp.maximum(eps_e_e**2 - 1.0, 1e-100))
    ps_e = eps_nu_e**2 * eps_e_e * p_e_e
    f_nu_e = _interp_monopole_epsilon_M(eps_nu_e, T_nu, q_nodes, f_nuebar, M_nuebar)
    f_e_ee = fermi_dirac_dimless(eps_e_e, T_e)
    I_e = T_nu * jnp.sum(jnp.where(mask_e, LAG_W * ps_e * f_nu_e * (1.0 - f_e_ee) * LAG_EXP, 0.0))

    # ── Legendre channels (c, f) — finite interval [1, Q_DIM] ──
    jac_leg = 0.5 * (Q_DIM - 1.0)
    eps_e_cf = jac_leg * (LEG_X + 1.0) + 1.0
    eps_nu_cf = Q_DIM - eps_e_cf
    mask_cf = (eps_nu_cf > 0.0) & (eps_e_cf > 1.0)
    p_e_cf = jnp.sqrt(jnp.maximum(eps_e_cf**2 - 1.0, 1e-100))
    ps_cf = eps_nu_cf**2 * eps_e_cf * p_e_cf
    f_e_cf = fermi_dirac_dimless(eps_e_cf, T_e)

    # Channel c uses f_nuebar (n→p neutron decay)
    f_nu_c = _interp_monopole_epsilon_M(eps_nu_cf, T_nu, q_nodes, f_nuebar, M_nuebar)
    I_c = jac_leg * jnp.sum(jnp.where(mask_cf, LEG_W * ps_cf * (1.0 - f_e_cf) * (1.0 - f_nu_c), 0.0))

    # Channel f uses f_nuebar (p→n proton conversion)
    f_nu_f = _interp_monopole_epsilon_M(eps_nu_cf, T_nu, q_nodes, f_nuebar, M_nuebar)
    I_f = jac_leg * jnp.sum(jnp.where(mask_cf, LEG_W * ps_cf * f_e_cf * f_nu_f, 0.0))

    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)

    return lambda_np, lambda_pn


# ═══════════════════════════════════════════════════════════════
# §2. Drop-in replacement API
# ═══════════════════════════════════════════════════════════════

def compute_live_born_rates_fused(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple:
    """Drop-in replacement for compute_live_born_rates_from_monopoles.

    Same signature, same physics, single JIT trace.
    """
    if spline_matrix is None:
        spline_matrix = jnp.empty((0, 0), dtype=jnp.asarray(q_nodes).dtype)
    return fused_live_born_rates(
        T_gamma_MeV, T_nu_MeV, tau_n,
        q_nodes, f_nue_monopole, f_nuebar_monopole, spline_matrix)


# ═══════════════════════════════════════════════════════════════
# §3. Fused correction-level-specialized CL0/CL1/CL2/CL3 kernels
# ═══════════════════════════════════════════════════════════════

def _i0_for_level(correction_level: int):
    if correction_level == 0:
        return _I0_CL0
    if correction_level == 1:
        return _I0_CL1
    if correction_level == 2:
        return _I0_CL2
    if correction_level == 3:
        return _I0_CL3
    raise ValueError("fused live weak backend supports correction_level in {0,1,2,3}.")


def _laguerre_correction(eps_e, eps_nu, channel: str, correction_level: int):
    if correction_level == 0:
        return 1.0
    corr = weak_correction_factor_jax(
        eps_e,
        channel,
        correction_level >= 1,
        correction_level >= 2,
    )
    if correction_level >= 3:
        corr = corr * finite_mass_scalar_correction_jax(eps_e, eps_nu, channel)
    return corr


def _bounded_c_correction(correction_level: int):
    if correction_level == 0:
        return 1.0
    if correction_level == 1:
        return CF_CORR_CL1
    if correction_level == 2:
        return CF_CORR_CL2
    if correction_level == 3:
        return CF_CORR_CL3_C
    raise ValueError("fused live weak backend supports correction_level in {0,1,2,3}.")


def _bounded_f_correction(correction_level: int):
    if correction_level == 0:
        return 1.0
    if correction_level == 1:
        return CF_CORR_CL1
    if correction_level == 2:
        return CF_CORR_CL2
    if correction_level == 3:
        return CF_CORR_CL3_F
    raise ValueError("fused live weak backend supports correction_level in {0,1,2,3}.")


@partial(jax.jit, static_argnames=("correction_level",))
def fused_live_rates_level(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue: jnp.ndarray,
    f_nuebar: jnp.ndarray,
    spline_matrix: jnp.ndarray,
    *,
    correction_level: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute CL0/CL1/CL2/CL3 live rates in one JAX trace.

    This keeps the same live-monopole functional as
    :mod:`rabbit.jax.weak_live_jax`, but avoids six separate channel calls per
    RHS evaluation.  The level is static so each correction level still lands
    in its own XLA cache bucket while staying inside the JAX/XLA runtime.
    """

    del spline_matrix
    cl = int(correction_level)
    I0 = _i0_for_level(cl)

    T_e = T_gamma_MeV / M_E_MEV
    T_nu = T_nu_MeV / M_E_MEV
    K = 1.0 / (tau_n * I0)

    M_nue = _prepare_logit_residual(q_nodes, f_nue)
    M_nuebar = _prepare_logit_residual(q_nodes, f_nuebar)

    # Channel a: ν_e + n -> e^- + p.
    eps_nu_a = LAG_X * T_nu
    eps_e_a = eps_nu_a + Q_DIM
    mask_a = eps_e_a > 1.0
    p_e_a = jnp.sqrt(jnp.maximum(eps_e_a**2 - 1.0, 1e-100))
    ps_a = eps_nu_a**2 * eps_e_a * p_e_a
    f_nu_a = _interp_monopole_epsilon_M(eps_nu_a, T_nu, q_nodes, f_nue, M_nue)
    f_e_a = fermi_dirac_dimless(eps_e_a, T_e)
    corr_a = _laguerre_correction(eps_e_a, eps_nu_a, "a", cl)
    I_a = T_nu * jnp.sum(
        jnp.where(
            mask_a,
            LAG_W * ps_a * f_nu_a * (1.0 - f_e_a) * corr_a * LAG_EXP,
            0.0,
        )
    )

    # Channel b: e^+ + n -> νbar_e + p.
    eps_e_b = 1.0 + LAG_X * T_e
    eps_nu_b = eps_e_b + Q_DIM
    p_e_b = jnp.sqrt(jnp.maximum(eps_e_b**2 - 1.0, 1e-100))
    ps_b = eps_nu_b**2 * eps_e_b * p_e_b
    f_e_b = fermi_dirac_dimless(eps_e_b, T_e)
    f_nubar_b = _interp_monopole_epsilon_M(eps_nu_b, T_nu, q_nodes, f_nuebar, M_nuebar)
    corr_b = _laguerre_correction(eps_e_b, eps_nu_b, "b", cl)
    I_b = T_e * jnp.sum(LAG_W * ps_b * f_e_b * (1.0 - f_nubar_b) * corr_b * LAG_EXP)

    # Bounded channels c/f on the finite interval [1, Q].
    f_e_cf = fermi_dirac_dimless(CF_EPS_E, T_e)
    f_nubar_cf = _interp_monopole_epsilon_M(
        CF_EPS_NU, T_nu, q_nodes, f_nuebar, M_nuebar
    )
    I_c = jnp.sum(
        CF_BASE_WEIGHT
        * (1.0 - f_e_cf)
        * (1.0 - f_nubar_cf)
        * _bounded_c_correction(cl)
    )
    I_f = jnp.sum(
        CF_BASE_WEIGHT
        * f_e_cf
        * f_nubar_cf
        * _bounded_f_correction(cl)
    )

    # Channel d: p + e^- -> n + ν_e.
    eps_e_d = Q_DIM + LAG_X * T_e
    eps_nu_d = eps_e_d - Q_DIM
    mask_d = (eps_nu_d > 0.0) & (eps_e_d > 1.0)
    p_e_d = jnp.sqrt(jnp.maximum(eps_e_d**2 - 1.0, 1e-100))
    ps_d = eps_nu_d**2 * eps_e_d * p_e_d
    f_e_d = fermi_dirac_dimless(eps_e_d, T_e)
    f_nu_d = _interp_monopole_epsilon_M(eps_nu_d, T_nu, q_nodes, f_nue, M_nue)
    corr_d = _laguerre_correction(eps_e_d, eps_nu_d, "d", cl)
    I_d = T_e * jnp.sum(
        jnp.where(
            mask_d,
            LAG_W * ps_d * f_e_d * (1.0 - f_nu_d) * corr_d * LAG_EXP,
            0.0,
        )
    )

    # Channel e: p + νbar_e -> n + e^+.
    eps_nu_e = (Q_DIM + 1.0) + LAG_X * T_nu
    eps_e_e = eps_nu_e - Q_DIM
    mask_e = eps_e_e > 1.0
    p_e_e = jnp.sqrt(jnp.maximum(eps_e_e**2 - 1.0, 1e-100))
    ps_e = eps_nu_e**2 * eps_e_e * p_e_e
    f_nubar_e = _interp_monopole_epsilon_M(eps_nu_e, T_nu, q_nodes, f_nuebar, M_nuebar)
    f_e_ee = fermi_dirac_dimless(eps_e_e, T_e)
    corr_e = _laguerre_correction(eps_e_e, eps_nu_e, "e", cl)
    I_e = T_nu * jnp.sum(
        jnp.where(
            mask_e,
            LAG_W * ps_e * f_nubar_e * (1.0 - f_e_ee) * corr_e * LAG_EXP,
            0.0,
        )
    )

    lambda_np = jnp.maximum(K * (I_a + I_b + I_c), 1.0 / tau_n)
    lambda_pn = jnp.maximum(K * (I_d + I_e + I_f), 0.0)
    return lambda_np, lambda_pn, I0


def compute_live_rates_from_monopoles_fused_jax(
    T_gamma_MeV: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    q_nodes: jnp.ndarray,
    f_nue_monopole: jnp.ndarray,
    f_nuebar_monopole: jnp.ndarray,
    *,
    correction_level: int = 0,
    spline_matrix: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if spline_matrix is None:
        spline_matrix = jnp.empty((0, 0), dtype=jnp.asarray(q_nodes).dtype)
    return fused_live_rates_level(
        T_gamma_MeV,
        T_nu_MeV,
        tau_n,
        q_nodes,
        f_nue_monopole,
        f_nuebar_monopole,
        spline_matrix,
        correction_level=int(correction_level),
    )


def _fused_level_wrapper(correction_level: int):
    def _kernel(
        T_gamma_MeV: jnp.ndarray,
        T_nu_MeV: jnp.ndarray,
        tau_n: jnp.ndarray,
        q_nodes: jnp.ndarray,
        f_nue_monopole: jnp.ndarray,
        f_nuebar_monopole: jnp.ndarray,
        spline_matrix: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return compute_live_rates_from_monopoles_fused_jax(
            T_gamma_MeV,
            T_nu_MeV,
            tau_n,
            q_nodes,
            f_nue_monopole,
            f_nuebar_monopole,
            correction_level=int(correction_level),
            spline_matrix=spline_matrix,
        )

    return jax.jit(_kernel)


compute_live_rates_from_monopoles_cl0_fused_jax = _fused_level_wrapper(0)
compute_live_rates_from_monopoles_cl1_fused_jax = _fused_level_wrapper(1)
compute_live_rates_from_monopoles_cl2_fused_jax = _fused_level_wrapper(2)
compute_live_rates_from_monopoles_cl3_fused_jax = _fused_level_wrapper(3)


def _fused_shared_level_wrapper(correction_level: int):
    def _kernel(
        T_gamma_MeV: jnp.ndarray,
        T_nu_MeV: jnp.ndarray,
        tau_n: jnp.ndarray,
        q_nodes: jnp.ndarray,
        f_monopole: jnp.ndarray,
        spline_matrix: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return compute_live_rates_from_monopoles_fused_jax(
            T_gamma_MeV,
            T_nu_MeV,
            tau_n,
            q_nodes,
            f_monopole,
            f_monopole,
            correction_level=int(correction_level),
            spline_matrix=spline_matrix,
        )

    return jax.jit(_kernel)


compute_live_rates_from_shared_monopole_cl0_fused_jax = _fused_shared_level_wrapper(0)
compute_live_rates_from_shared_monopole_cl1_fused_jax = _fused_shared_level_wrapper(1)
compute_live_rates_from_shared_monopole_cl2_fused_jax = _fused_shared_level_wrapper(2)
compute_live_rates_from_shared_monopole_cl3_fused_jax = _fused_shared_level_wrapper(3)
