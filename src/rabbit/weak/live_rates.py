"""
rabbit.weak.live_rates — Live weak rate distribution functional.

Computes weak n↔p rates as functionals of the neutrino monopole
distribution f₀(q_i) extracted from the transport hierarchy.

    λ_np = λ[f_{ν_e,0}, f_{ν̄_e,0}]
    λ_pn = λ[f_{ν_e,0}, f_{ν̄_e,0}]

This is the canonical rabbit path.  There are NO table fallbacks.  The monopole
distribution is fed directly into the 6-channel quadrature at each timestep;
staged augmented callers may additionally pass explicit CL3 angular correction
factors computed from their current distribution moments.

At Born level, ONLY the monopole f₀(E) contributes (R03: angular
orthogonality of V−A matrix element kills ℓ ≥ 1).  The integral
is 1D in energy, not 2D in energy × angle.

The comoving momentum q (Gauss–Laguerre nodes) relates to physical
neutrino energy by E_ν = q × T_ν.  In m_e units: ε_ν = q × T_ν/m_e.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import warnings
import numpy as np

# OPT-2: Module-level spline cache to avoid repeated construction
# when monopole distribution is unchanged between calls.
_spline_cache = {}  # key: (q_nodes.tobytes(), f_monopole.tobytes()) -> interp1d
_weak_kernel_cache = {}
_weak_static_channel_cache = {}

from rabbit.weak.channels import (
    channel_a, channel_b, channel_c,
    channel_d, channel_e, channel_f,
    fermi_dirac, get_I0_born,
    _M_ELECTRON_MEV, _Q_DIMLESS,
)
from rabbit.weak.quadrature import (
    WeakQuadrature, DEFAULT_WEAK_QUADRATURE, HIGHRES_WEAK_QUADRATURE,
)
from rabbit.config.grids import MomentumGrid
from rabbit.weak.corrections import (
    compute_I0_corrected, coulomb_correction_per_channel, radiative_correction_factor,
    compute_I0_cl3_corrected,
)
from rabbit.weak.finite_mass import finite_mass_scalar_correction


def _weak_quadrature_signature(quad: WeakQuadrature) -> tuple[int, int]:
    return int(quad.N_laguerre), int(quad.N_legendre)


def _build_static_channel_kernel_cache(q, correction_level, quad):
    key = (
        float(q),
        int(correction_level),
        *_weak_quadrature_signature(quad),
    )
    cached = _weak_static_channel_cache.get(key)
    if cached is not None:
        return cached

    lag_n, lag_w = quad.lag_nodes, quad.lag_weights
    leg_n, leg_w = quad.leg_nodes, quad.leg_weights
    exp_lag = np.exp(lag_n)

    if q > 1.0:
        eps_e_cf = 0.5 * (q - 1.0) * leg_n + 0.5 * (q + 1.0)
        jac_cf = 0.5 * (q - 1.0)
        eps_nu_cf = q - eps_e_cf
        mask_cf = (eps_nu_cf > 0.0) & (eps_e_cf > 1.0)
        if np.any(mask_cf):
            e_e_cf = eps_e_cf[mask_cf]
            e_nu_cf = eps_nu_cf[mask_cf]
            pe_cf = np.sqrt(e_e_cf**2 - 1.0)
            base_cf = jac_cf * leg_w[mask_cf] * (e_nu_cf**2 * e_e_cf * pe_cf)

            if correction_level > 0:
                corr_c = coulomb_correction_per_channel(e_e_cf, 'c')
                if correction_level >= 2:
                    corr_c = corr_c * radiative_correction_factor(e_e_cf)
            else:
                corr_c = 1.0
            if correction_level >= 3:
                corr_c = corr_c * finite_mass_scalar_correction(e_e_cf, e_nu_cf, 'c')
            pref_c_static = base_cf * corr_c

            if correction_level >= 3:
                corr_f = (coulomb_correction_per_channel(e_e_cf, 'f') if correction_level > 0 else 1.0)
                if correction_level >= 2:
                    corr_f = corr_f * radiative_correction_factor(e_e_cf)
                corr_f = corr_f * finite_mass_scalar_correction(e_e_cf, e_nu_cf, 'f')
            else:
                corr_f = corr_c
            pref_f_static = base_cf * corr_f
        else:
            e_e_cf = e_nu_cf = pref_c_static = pref_f_static = np.array([], dtype=float)
    else:
        e_e_cf = e_nu_cf = pref_c_static = pref_f_static = np.array([], dtype=float)

    cached = {
        'lag_nodes': lag_n,
        'lag_weights': lag_w,
        'exp_lag': exp_lag,
        'cf': (e_e_cf, e_nu_cf, pref_c_static, pref_f_static),
    }
    if len(_weak_static_channel_cache) > 32:
        _weak_static_channel_cache.clear()
    _weak_static_channel_cache[key] = cached
    return cached


def _build_channel_kernel_cache(T_e, T_nu, q, correction_level, quad):
    key = (
        float(T_e),
        float(T_nu),
        float(q),
        int(correction_level),
        *_weak_quadrature_signature(quad),
    )
    cached = _weak_kernel_cache.get(key)
    if cached is not None:
        return cached

    static = _build_static_channel_kernel_cache(q, correction_level, quad)
    lag_n = static['lag_nodes']
    lag_w = static['lag_weights']
    exp_lag = static['exp_lag']

    def _fd_e(eps):
        if T_e < 1e-30:
            return np.zeros_like(eps)
        arg = np.minimum(eps / T_e, 500.0)
        return 1.0 / (np.exp(arg) + 1.0)

    # Channel a
    eps_nu_a = lag_n * T_nu
    eps_e_a = eps_nu_a + q
    mask_a = eps_e_a > 1.0
    if np.any(mask_a):
        e_nu_a = eps_nu_a[mask_a]
        e_e_a = eps_e_a[mask_a]
        pe_a = np.sqrt(e_e_a**2 - 1.0)
        pref_a = lag_w[mask_a] * (e_nu_a**2 * e_e_a * pe_a) * exp_lag[mask_a]
        if correction_level > 0:
            pref_a = pref_a * coulomb_correction_per_channel(e_e_a, 'a')
        if correction_level >= 3:
            pref_a = pref_a * finite_mass_scalar_correction(e_e_a, e_nu_a, 'a')
        pref_a = T_nu * pref_a
        fd_e_a = _fd_e(e_e_a)
    else:
        e_nu_a = e_e_a = pref_a = fd_e_a = np.array([], dtype=float)

    # Channel b
    eps_e_b = 1.0 + lag_n * T_e
    eps_nu_b = eps_e_b + q
    pe_b = np.sqrt(eps_e_b**2 - 1.0)
    pref_b = lag_w * (eps_nu_b**2 * eps_e_b * pe_b) * exp_lag
    if correction_level > 0:
        pref_b = pref_b * coulomb_correction_per_channel(eps_e_b, 'b')
    if correction_level >= 3:
        pref_b = pref_b * finite_mass_scalar_correction(eps_e_b, eps_nu_b, 'b')
    pref_b = T_e * pref_b
    fd_e_b = _fd_e(eps_e_b)

    # Channels c and f share bounded kinematics and temperature-independent
    # correction factors.  Only electron blocking changes with T_e.
    e_e_cf, e_nu_cf, pref_c, pref_f = static['cf']
    fd_e_cf = _fd_e(e_e_cf) if pref_c.size else np.array([], dtype=float)

    # Channel d
    eps_e_d = q + lag_n * T_e
    eps_nu_d = eps_e_d - q
    mask_d = (eps_nu_d > 0.0) & (eps_e_d > 1.0)
    if np.any(mask_d):
        e_e_d = eps_e_d[mask_d]
        e_nu_d = eps_nu_d[mask_d]
        pe_d = np.sqrt(e_e_d**2 - 1.0)
        pref_d = lag_w[mask_d] * (e_nu_d**2 * e_e_d * pe_d) * exp_lag[mask_d]
        if correction_level > 0:
            pref_d = pref_d * coulomb_correction_per_channel(e_e_d, 'd')
        if correction_level >= 3:
            pref_d = pref_d * finite_mass_scalar_correction(e_e_d, e_nu_d, 'd')
        pref_d = T_e * pref_d
        fd_e_d = _fd_e(e_e_d)
    else:
        e_e_d = e_nu_d = pref_d = fd_e_d = np.array([], dtype=float)

    # Channel e
    eps_nu_e = (q + 1.0) + lag_n * T_nu
    eps_e_e = eps_nu_e - q
    mask_e = eps_e_e > 1.0
    if np.any(mask_e):
        e_nu_e = eps_nu_e[mask_e]
        e_e_e = eps_e_e[mask_e]
        pe_e = np.sqrt(e_e_e**2 - 1.0)
        pref_e = lag_w[mask_e] * (e_nu_e**2 * e_e_e * pe_e) * exp_lag[mask_e]
        if correction_level > 0:
            pref_e = pref_e * coulomb_correction_per_channel(e_e_e, 'e')
        if correction_level >= 3:
            pref_e = pref_e * finite_mass_scalar_correction(e_e_e, e_nu_e, 'e')
        pref_e = T_nu * pref_e
        fd_e_e = _fd_e(e_e_e)
    else:
        e_nu_e = e_e_e = pref_e = fd_e_e = np.array([], dtype=float)

    cached = {
        'a': (e_nu_a, pref_a, fd_e_a),
        'b': (eps_nu_b, pref_b, fd_e_b),
        'c': (e_nu_cf, pref_c, fd_e_cf),
        'd': (e_nu_d, pref_d, fd_e_d),
        'e': (e_nu_e, pref_e, fd_e_e),
        'f': (e_nu_cf, pref_f, fd_e_cf),
    }
    if len(_weak_kernel_cache) > 128:
        _weak_kernel_cache.clear()
    _weak_kernel_cache[key] = cached
    return cached


def _warn_if_out_of_bounds(label: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo < -1e-12 or hi > 1.0 + 1e-12:
        warnings.warn(
            f"{label}: monopole left physical FD range [0,1] (min={lo:.6e}, max={hi:.6e}). Interpolation applies only logit-domain numerical stabilization; physics claims should veto this run.",
            RuntimeWarning,
            stacklevel=2,
        )


# ═══════════════════════════════════════════════════════════════════════
# §1. Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LiveWeakResult:
    """Result of live weak rate evaluation.

    Attributes
    ----------
    lambda_np : float
        n → p total rate [s⁻¹].
    lambda_pn : float
        p → n total rate [s⁻¹].
    lambda_np_iso : float
        Isotropic (equilibrium FD) reference rate for n → p.
    lambda_pn_iso : float
        Isotropic reference rate for p → n.
    delta_np : float
        Fractional excess: (λ_np − λ_np_iso) / λ_np_iso.
    delta_pn : float
        Fractional excess for p → n.
    channels_np : tuple
        Per-channel breakdown (I_a, I_b, I_c) before normalization.
    channels_pn : tuple
        Per-channel breakdown (I_d, I_e, I_f).
    """
    lambda_np: float
    lambda_pn: float
    lambda_np_iso: float = 0.0
    lambda_pn_iso: float = 0.0
    delta_np: float = 0.0
    delta_pn: float = 0.0
    channels_np: tuple = (0.0, 0.0, 0.0)
    channels_pn: tuple = (0.0, 0.0, 0.0)

    @property
    def total_rate(self) -> float:
        return self.lambda_np + self.lambda_pn

    @property
    def equilibrium_Xn(self) -> float:
        total = self.total_rate
        if total > 0:
            return self.lambda_pn / total
        return 0.5


# ═══════════════════════════════════════════════════════════════════════
# §2. Distribution function builder
# ═══════════════════════════════════════════════════════════════════════

def _build_distribution_fn(
    f_monopole_q: np.ndarray,
    q_nodes: np.ndarray,
    T_nu_dimless: float,
    use_exact_fd: bool = False,
):
    """Build an interpolated f_ν(ε_ν) from the momentum-grid monopole.

    The hierarchy stores f(q_i) = f₀(q_i) × (1 + Ψ₀(q_i)) at
    Gauss–Laguerre nodes q_i.  The weak rate integrand needs f_ν(ε_ν)
    where ε_ν is energy in m_e units.

    Conversion: ε_ν = q × T_ν / m_e = q × T_nu_dimless.

    Parameters
    ----------
    f_monopole_q : ndarray, shape (N_q,)
        Monopole distribution at momentum grid nodes.
    q_nodes : ndarray, shape (N_q,)
        Gauss–Laguerre quadrature nodes (comoving momentum).
    T_nu_dimless : float
        Neutrino temperature in m_e units (T_ν / m_e).
    use_exact_fd : bool
        If True, evaluate equilibrium FD directly (collisionless fast path).
        This skips interpolation entirely — faster AND more accurate when
        the monopole equals the equilibrium distribution.

    Returns
    -------
    callable
        f_ν(ε_ν) → occupation number, accepting ndarray input.
    """
    if T_nu_dimless < 1e-30:
        return lambda eps: np.zeros_like(np.asarray(eps, dtype=float))

    # ── Fast path: exact FD evaluation (collisionless, Ψ₀ = 0) ──
    if use_exact_fd:
        def f_nu_exact(eps):
            eps = np.asarray(eps, dtype=float)
            arg = np.minimum(eps / T_nu_dimless, 500.0)
            return 1.0 / (np.exp(arg) + 1.0)
        return f_nu_exact

    # ── General path: interpolate δ(q)=logit(f(q))+q instead of f(q) directly ──
    _warn_if_out_of_bounds("_build_distribution_fn", f_monopole_q)
    # This preserves exact FD spectra for any N_q and avoids the severe low-N_q
    # oscillations of cubic interpolation on sparse Laguerre nodes.
    cache_key = (q_nodes.tobytes(), f_monopole_q.tobytes())
    if cache_key in _spline_cache:
        delta_q = _spline_cache[cache_key]
    else:
        eps_clip = 1e-14
        f_safe = np.clip(f_monopole_q, eps_clip, 1.0 - eps_clip)
        delta_q = np.log(f_safe / (1.0 - f_safe)) + q_nodes
        if len(_spline_cache) > 4:
            _spline_cache.clear()
        _spline_cache[cache_key] = delta_q

    inv_T = 1.0 / T_nu_dimless

    def f_nu(eps):
        eps = np.asarray(eps, dtype=float)
        q_eval = eps * inv_T
        delta_eval = np.interp(q_eval, q_nodes, delta_q, left=delta_q[0], right=0.0)
        result = 1.0 / (1.0 + np.exp(q_eval - delta_eval))
        return result

    return f_nu


# ═══════════════════════════════════════════════════════════════════════
# §3. Main entry point
# ═══════════════════════════════════════════════════════════════════════

def compute_live_weak_rates(
    f_nue_monopole: np.ndarray,
    f_nuebar_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_gamma_MeV: float,
    T_nu_MeV: Optional[float] = None,
    tau_n: float = 878.4,
    quad: WeakQuadrature = None,
    compute_iso_reference: bool = True,
    correction_level: int = 0,
    use_exact_fd: bool = False,
    sigma_plus_correction_factor: float = 1.0,
    lambda_np_correction_factor: float | None = None,
    lambda_pn_correction_factor: float | None = None,
) -> LiveWeakResult:
    """Compute weak n↔p rates from live neutrino monopole distributions.

    This is the canonical rabbit rate computation.  NO table
    fallback.  NO perturbative correction factor.

    Parameters
    ----------
    f_nue_monopole : ndarray, shape (N_q,)
        ν_e monopole distribution f₀(q_i)(1 + Ψ₀(q_i)) at each
        momentum grid node.  From transport projector.
    f_nuebar_monopole : ndarray, shape (N_q,)
        ν̄_e monopole distribution.
    q_nodes : ndarray, shape (N_q,)
        Gauss–Laguerre momentum grid nodes.
    T_gamma_MeV : float
        Photon temperature [MeV].
    T_nu_MeV : float or None
        Neutrino temperature [MeV].  If None, uses instantaneous
        decoupling: T_ν = T_γ × (4/11)^{1/3}.
    tau_n : float
        Neutron lifetime [s].
    quad : WeakQuadrature or None
        Quadrature specification.  Default: 32-pt Laguerre + Legendre.
    compute_iso_reference : bool
        If True, also compute the equilibrium FD reference rates
        for diagnostic comparison.
    correction_level : int
        0 = Born (default, backward compatible)
        1 = Born + Coulomb (Fermi function)
        2 = Born + Coulomb + Sirlin radiative
        3 = CL2 + finite-mass/recoil/weak-magnetism terms

    Returns
    -------
    LiveWeakResult
    """
    if correction_level not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported correction_level={correction_level}; expected 0,1,2,3")

    if quad is None:
        quad = DEFAULT_WEAK_QUADRATURE

    if T_gamma_MeV < 1e-4:
        return LiveWeakResult(
            lambda_np=1.0 / tau_n,
            lambda_pn=0.0,
        )

    if T_nu_MeV is None:
        T_nu_MeV = T_gamma_MeV * (4.0 / 11.0) ** (1.0 / 3.0)

    T_e = T_gamma_MeV / _M_ELECTRON_MEV
    T_nu = T_nu_MeV / _M_ELECTRON_MEV
    q = _Q_DIMLESS

    # ── Normalization: I₀ with same corrections ──
    enable_coulomb = correction_level >= 1
    enable_radiative = correction_level >= 2

    if correction_level >= 3:
        I0 = compute_I0_cl3_corrected()
    elif correction_level > 0:
        I0 = compute_I0_corrected(
            enable_coulomb=enable_coulomb,
            enable_radiative=enable_radiative)
    else:
        I0 = get_I0_born()

    # Build interpolated distribution functions from hierarchy monopoles
    # use_exact_fd=True: direct FD evaluation (collisionless, no interp overhead)
    f_nue_fn = _build_distribution_fn(f_nue_monopole, q_nodes, T_nu,
                                      use_exact_fd=use_exact_fd)
    f_nuebar_fn = _build_distribution_fn(f_nuebar_monopole, q_nodes, T_nu,
                                         use_exact_fd=use_exact_fd)

    lag_n, lag_w = quad.lag_nodes, quad.lag_weights
    kernel = _build_channel_kernel_cache(T_e, T_nu, q, correction_level, quad)

    def _evaluate_all_channels(f_nue_eval, f_nuebar_eval):
        # Channel (a): ν_e + n → p + e⁻
        e_nu, pref, fd_e = kernel['a']
        if pref.size:
            I_a = float(np.sum(pref * f_nue_eval(e_nu) * (1.0 - fd_e)))
        else:
            I_a = 0.0

        # Channel (b): e⁺ + n → p + ν̄_e
        e_nu, pref, fd_e = kernel['b']
        I_b = float(np.sum(pref * fd_e * (1.0 - f_nuebar_eval(e_nu))))

        # Channels (c) and (f) share the bounded kinematics
        e_nu_cf, pref_c, fd_e_cf = kernel['c']
        if pref_c.size:
            f_nubar_vals = f_nuebar_eval(e_nu_cf)
            I_c = float(np.sum(pref_c * (1.0 - fd_e_cf) * (1.0 - f_nubar_vals)))
            pref_f = kernel['f'][1]
            I_f = float(np.sum(pref_f * fd_e_cf * f_nubar_vals))
        else:
            I_c = 0.0
            I_f = 0.0

        # Channel (d): p + e⁻ → n + ν_e
        e_nu, pref, fd_e = kernel['d']
        if pref.size:
            I_d = float(np.sum(pref * fd_e * (1.0 - f_nue_eval(e_nu))))
        else:
            I_d = 0.0

        # Channel (e): p + ν̄_e → n + e⁺
        e_nu, pref, fd_e = kernel['e']
        if pref.size:
            I_e = float(np.sum(pref * f_nuebar_eval(e_nu) * (1.0 - fd_e)))
        else:
            I_e = 0.0

        return I_a, I_b, I_c, I_d, I_e, I_f

    I_a, I_b, I_c, I_d, I_e, I_f = _evaluate_all_channels(f_nue_fn, f_nuebar_fn)
    I_np = I_a + I_b + I_c
    I_pn = I_d + I_e + I_f

    lambda_np = max(I_np / (I0 * tau_n), 1.0 / tau_n)
    lambda_pn = max(I_pn / (I0 * tau_n), 0.0)

    # Leading-order CL3 angular K_2 correction.  The legacy scalar factor
    # preserves the original AP25 behavior; AP59 can pass species-specific
    # factors computed from explicit AP58 angular moment inputs.
    if sigma_plus_correction_factor != 1.0:
        lambda_np = lambda_np * float(sigma_plus_correction_factor)
        lambda_pn = lambda_pn * float(sigma_plus_correction_factor)
    if lambda_np_correction_factor is not None:
        factor_np = _validate_positive_rate_factor(
            lambda_np_correction_factor,
            "lambda_np_correction_factor",
        )
        lambda_np = lambda_np * factor_np
    if lambda_pn_correction_factor is not None:
        factor_pn = _validate_positive_rate_factor(
            lambda_pn_correction_factor,
            "lambda_pn_correction_factor",
        )
        lambda_pn = lambda_pn * factor_pn

    # ── Isotropic reference ──
    lambda_np_iso = 0.0
    lambda_pn_iso = 0.0
    delta_np = 0.0
    delta_pn = 0.0

    if compute_iso_reference:
        def f_fd_nue(eps):
            return fermi_dirac(eps, T_nu)

        def f_fd_nuebar(eps):
            return fermi_dirac(eps, T_nu)

        I_a_iso, I_b_iso, I_c_iso, I_d_iso, I_e_iso, I_f_iso = _evaluate_all_channels(f_fd_nue, f_fd_nuebar)
        I_np_iso = I_a_iso + I_b_iso + I_c_iso
        I_pn_iso = I_d_iso + I_e_iso + I_f_iso

        lambda_np_iso = max(I_np_iso / (I0 * tau_n), 1.0 / tau_n)
        lambda_pn_iso = max(I_pn_iso / (I0 * tau_n), 0.0)

        if lambda_np_iso > 1e-30:
            delta_np = (lambda_np - lambda_np_iso) / lambda_np_iso
        if lambda_pn_iso > 1e-30:
            delta_pn = (lambda_pn - lambda_pn_iso) / lambda_pn_iso

    return LiveWeakResult(
        lambda_np=lambda_np,
        lambda_pn=lambda_pn,
        lambda_np_iso=lambda_np_iso,
        lambda_pn_iso=lambda_pn_iso,
        delta_np=delta_np,
        delta_pn=delta_pn,
        channels_np=(I_a, I_b, I_c),
        channels_pn=(I_d, I_e, I_f),
    )


def _validate_positive_rate_factor(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return out
