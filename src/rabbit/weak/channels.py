"""
rabbit.weak.channels — Six weak n↔p channel integrands.

Channels (n→p, contributing to λ_np):
    (a) ν_e + n → p + e⁻         E_ν semi-infinite, ε_e = ε_ν + q
    (b) e⁺ + n → p + ν̄_e         E_e semi-infinite, ε_ν = ε_e + q
    (c) n → p + e⁻ + ν̄_e         bounded: ε_e ∈ [1, q], ε_ν = q − ε_e

Channels (p→n, contributing to λ_pn):
    (d) p + e⁻ → n + ν_e         E_e semi-infinite, ε_e > q
    (e) p + ν̄_e → n + e⁺         E_ν semi-infinite, ε_ν > q + 1
    (f) p + e⁻ + ν̄_e → n         bounded: same kinematics as (c), reverse

All energies are in units of m_e.  The dimensionless mass splitting is
q = Δ/m_e = Q_np/m_e ≈ 2.531.

Born-level normalization:
    K = 1 / (τ_n × I₀)

where I₀ is the zero-temperature neutron decay phase-space integral
(channel c at T=0):

    I₀ = ∫₁^q dε_e  ε_e (q − ε_e)² √(ε_e² − 1) ≈ 1.63609

so that λ_np → 1/τ_n at T → 0 (neutron decay).

The neutrino distribution f_ν(ε) is an ARBITRARY pointwise function —
not assumed to be Fermi–Dirac.  This is the distribution functional
approach required by P01 §4.

At Born level, ONLY the monopole component f₀(E) contributes (R03).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

# Physical constants in m_e units
_M_ELECTRON_MEV = 0.5109989500
_Q_NP_MEV = 1.29333236        # M_n - M_p
_Q_DIMLESS = _Q_NP_MEV / _M_ELECTRON_MEV  # ≈ 2.5310
_TAU_N = 878.4                 # neutron lifetime [s]

# Type alias for distribution functions: f(ε) → occupation number
DistributionFn = Callable[[np.ndarray], np.ndarray]


# ═══════════════════════════════════════════════════════════════════════
# §1. Fermi–Dirac reference distributions
# ═══════════════════════════════════════════════════════════════════════

def fermi_dirac(eps: np.ndarray, T_dimless: float) -> np.ndarray:
    """Standard Fermi–Dirac: f(ε, T) = 1/(exp(ε/T) + 1).

    Parameters
    ----------
    eps : ndarray
        Energy in m_e units.
    T_dimless : float
        Temperature in m_e units (T_MeV / m_e).
    """
    if T_dimless < 1e-30:
        return np.zeros_like(eps)
    arg = np.minimum(eps / T_dimless, 500.0)
    return 1.0 / (np.exp(arg) + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §2. Zero-temperature normalization integral I₀
# ═══════════════════════════════════════════════════════════════════════

def compute_I0_born(q: float = _Q_DIMLESS, n_leg: int = 64) -> float:
    """Neutron decay phase-space integral I₀ (Born, zero temperature).

    I₀ = ∫₁^q dε_e  ε_e (q − ε_e)² √(ε_e² − 1)

    Parameters
    ----------
    q : float
        Dimensionless mass splitting Q_np/m_e.
    n_leg : int
        Gauss–Legendre nodes for the bounded integral.

    Returns
    -------
    float
        I₀ ≈ 1.63609.
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_leg)

    # Map [-1, 1] → [1, q]
    eps_e = 0.5 * (q - 1.0) * x + 0.5 * (q + 1.0)
    jac = 0.5 * (q - 1.0)

    pe = np.sqrt(np.maximum(eps_e**2 - 1.0, 0.0))
    eps_nu = q - eps_e

    integrand = eps_e * eps_nu**2 * pe
    return float(jac * np.sum(w * integrand))


# Cached I₀
_I0_BORN_CACHE: Optional[float] = None


def get_I0_born() -> float:
    """Cached I₀ retrieval."""
    global _I0_BORN_CACHE
    if _I0_BORN_CACHE is None:
        _I0_BORN_CACHE = compute_I0_born()
    return _I0_BORN_CACHE


# ═══════════════════════════════════════════════════════════════════════
# §3. Channel integrands
# ═══════════════════════════════════════════════════════════════════════

# Each channel function returns the raw (un-normalized) integral value.
# The caller (live_rates.py) divides by (I₀ × τ_n) to get the rate.


def channel_a(
    f_nue: DistributionFn,
    T_e: float,
    nodes_lag: np.ndarray,
    weights_lag: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (a): ν_e + n → p + e⁻.

    Integration variable: ε_ν (Gauss–Laguerre, semi-infinite).
    Kinematics: ε_e = ε_ν + q.
    Distribution: f_ν(ε_ν) × (1 − f_e(ε_e)).
    """
    # Laguerre nodes are x_i; physical ε_ν = x_i × T_nu
    eps_nu = nodes_lag * T_nu
    eps_e = eps_nu + q

    mask = eps_e > 1.0
    if not np.any(mask):
        return 0.0

    e_nu = eps_nu[mask]
    e_e = eps_e[mask]
    w = weights_lag[mask]

    pe = np.sqrt(e_e**2 - 1.0)
    ps = e_nu**2 * e_e * pe

    # f_nue is the monopole distribution as a function of ε_ν
    f_nu_vals = f_nue(e_nu)
    f_e_vals = fermi_dirac(e_e, T_e)

    dist = f_nu_vals * (1.0 - f_e_vals)
    # Undo Laguerre weight: integrand × e^{x_i}
    integrand = ps * dist * np.exp(nodes_lag[mask])
    if correction_fn is not None:
        integrand *= correction_fn(e_e)

    return float(T_nu * np.sum(w * integrand))


def channel_b(
    f_nuebar: DistributionFn,
    T_e: float,
    nodes_lag: np.ndarray,
    weights_lag: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (b): e⁺ + n → p + ν̄_e.

    Integration variable: ε_e (Gauss–Laguerre from ε_e = 1).
    Kinematics: ε_ν = ε_e + q.
    Distribution: f_e(ε_e) × (1 − f_ν̄(ε_ν)).
    """
    eps_e = 1.0 + nodes_lag * T_e
    eps_nu = eps_e + q
    pe = np.sqrt(eps_e**2 - 1.0)
    ps = eps_nu**2 * eps_e * pe

    f_e_vals = fermi_dirac(eps_e, T_e)
    f_nu_vals = f_nuebar(eps_nu)

    dist = f_e_vals * (1.0 - f_nu_vals)
    integrand = ps * dist * np.exp(nodes_lag)
    if correction_fn is not None:
        integrand *= correction_fn(eps_e)

    return float(T_e * np.sum(weights_lag * integrand))


def channel_c(
    f_nuebar: DistributionFn,
    T_e: float,
    nodes_leg: np.ndarray,
    weights_leg: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (c): n → p + e⁻ + ν̄_e  (neutron decay, bounded).

    Integration variable: ε_e ∈ [1, q] (Gauss–Legendre).
    Kinematics: ε_ν = q − ε_e.
    Distribution: (1 − f_e(ε_e)) × (1 − f_ν̄(ε_ν)).
    """
    if q <= 1.0:
        return 0.0

    # Map [-1, 1] → [1, q]
    eps_e = 0.5 * (q - 1.0) * nodes_leg + 0.5 * (q + 1.0)
    jac = 0.5 * (q - 1.0)
    eps_nu = q - eps_e

    mask = (eps_nu > 0) & (eps_e > 1.0)
    if not np.any(mask):
        return 0.0

    e_e = eps_e[mask]
    e_nu = eps_nu[mask]
    w = weights_leg[mask]

    pe = np.sqrt(e_e**2 - 1.0)
    ps = e_nu**2 * e_e * pe

    f_e_vals = fermi_dirac(e_e, T_e)
    f_nu_vals = f_nuebar(e_nu)

    dist = (1.0 - f_e_vals) * (1.0 - f_nu_vals)
    integrand = ps * dist
    if correction_fn is not None:
        integrand *= correction_fn(e_e)

    return float(jac * np.sum(w * integrand))


def channel_d(
    f_nue: DistributionFn,
    T_e: float,
    nodes_lag: np.ndarray,
    weights_lag: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (d): p + e⁻ → n + ν_e.

    Integration variable: ε_e (Gauss–Laguerre from ε_e = q).
    Kinematics: ε_ν = ε_e − q.
    Distribution: f_e(ε_e) × (1 − f_ν(ε_ν)).
    """
    eps_e = q + nodes_lag * T_e
    eps_nu = eps_e - q

    mask = (eps_nu > 0) & (eps_e > 1.0)
    if not np.any(mask):
        return 0.0

    e_e = eps_e[mask]
    e_nu = eps_nu[mask]
    w = weights_lag[mask]

    pe = np.sqrt(e_e**2 - 1.0)
    ps = e_nu**2 * e_e * pe

    f_e_vals = fermi_dirac(e_e, T_e)
    f_nu_vals = f_nue(e_nu)

    dist = f_e_vals * (1.0 - f_nu_vals)
    integrand = ps * dist * np.exp(nodes_lag[mask])
    if correction_fn is not None:
        integrand *= correction_fn(e_e)

    return float(T_e * np.sum(w * integrand))


def channel_e(
    f_nuebar: DistributionFn,
    T_e: float,
    nodes_lag: np.ndarray,
    weights_lag: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (e): p + ν̄_e → n + e⁺.

    Integration variable: ε_ν (Gauss–Laguerre from ε_ν = q + 1).
    Kinematics: ε_e = ε_ν − q.
    Distribution: f_ν̄(ε_ν) × (1 − f_e(ε_e)).
    """
    eps_nu = (q + 1.0) + nodes_lag * T_nu
    eps_e = eps_nu - q

    mask = eps_e > 1.0
    if not np.any(mask):
        return 0.0

    e_nu = eps_nu[mask]
    e_e = eps_e[mask]
    w = weights_lag[mask]

    pe = np.sqrt(e_e**2 - 1.0)
    ps = e_nu**2 * e_e * pe

    f_nu_vals = f_nuebar(e_nu)
    f_e_vals = fermi_dirac(e_e, T_e)

    dist = f_nu_vals * (1.0 - f_e_vals)
    integrand = ps * dist * np.exp(nodes_lag[mask])
    if correction_fn is not None:
        integrand *= correction_fn(e_e)

    return float(T_nu * np.sum(w * integrand))


def channel_f(
    f_nuebar: DistributionFn,
    T_e: float,
    nodes_leg: np.ndarray,
    weights_leg: np.ndarray,
    T_nu: float,
    q: float = _Q_DIMLESS,
    correction_fn=None,
) -> float:
    """Channel (f): p + e⁻ + ν̄_e → n  (reverse neutron decay, bounded).

    Integration variable: ε_e ∈ [1, q] (Gauss–Legendre).
    Kinematics: ε_ν = q − ε_e (same as channel c).
    Distribution: f_e(ε_e) × f_ν̄(ε_ν).
    """
    if q <= 1.0:
        return 0.0

    eps_e = 0.5 * (q - 1.0) * nodes_leg + 0.5 * (q + 1.0)
    jac = 0.5 * (q - 1.0)
    eps_nu = q - eps_e

    mask = (eps_nu > 0) & (eps_e > 1.0)
    if not np.any(mask):
        return 0.0

    e_e = eps_e[mask]
    e_nu = eps_nu[mask]
    w = weights_leg[mask]

    pe = np.sqrt(e_e**2 - 1.0)
    ps = e_nu**2 * e_e * pe

    f_e_vals = fermi_dirac(e_e, T_e)
    f_nu_vals = f_nuebar(e_nu)

    dist = f_e_vals * f_nu_vals
    integrand = ps * dist
    if correction_fn is not None:
        integrand *= correction_fn(e_e)

    return float(jac * np.sum(w * integrand))
