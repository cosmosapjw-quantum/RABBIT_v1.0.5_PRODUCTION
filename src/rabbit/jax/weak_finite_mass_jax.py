"""
rabbit.jax.weak_finite_mass_jax — JAX-native finite nucleon mass corrections.

JAX port of :mod:`rabbit.weak.finite_mass`.  All public functions mirror the
NumPy reference but use ``jax.numpy`` primitives so they can be traced inside
JIT-compiled driver kernels.

Implements O(1/m_N) corrections:

  1. Kinematic recoil: finite-mass phase-space shrinkage.
     Net effect on I₀: δ_n ≈ −0.00201 (Seckel Eq. 28).

  2. Weak magnetism: anomalous magnetic moment σ^{μν} coupling.
     Sign flips between particle/antiparticle channels.

  3. Angular kernel K_{r,ℓ}: Legendre-projected coefficients for
     anisotropy-sensitive rates where f_{ν,2} enters directly.

Maturity: experimental_substrate.
Parity with NumPy reference must be verified before promotion.

References:
    Seckel 1993, hep-ph/9305311
    Lopez & Turner 1999, PRD 59, 103502
    PRIMAT: Pitrou et al. 2018, Phys. Rep. 754, 1 (§III.G-H, Appendix B.3-B.4)
    PRyMordial: Burns, Tait, Valli 2024, EPJC 84, 86 (§2.2, Eq. 2.12)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np
from numpy.polynomial.legendre import leggauss as _leggauss

jax.config.update("jax_enable_x64", True)

from rabbit.weak.finite_mass import (
    M_E_MEV, M_N_MEV, Q_NP_MEV,
    G_A, KAPPA, F_WM,
    _Q_DIMLESS, _MN_DIMLESS,
)


# ═══════════════════════════════════════════════════════════════
# §1. Scalar (angle-averaged) recoil correction
# ═══════════════════════════════════════════════════════════════

def recoil_scalar_correction_jax(
    E_e: jnp.ndarray,
    E_nu: jnp.ndarray,
    channel: str,
) -> jnp.ndarray:
    """Angle-averaged kinematic recoil correction at O(1/m_N).

    JAX mirror of :func:`rabbit.weak.finite_mass.recoil_scalar_correction`.

    Parameters
    ----------
    E_e, E_nu : jnp.ndarray
        Energies in m_e units.
    channel : str
        Channel identifier ('a'–'f').  Currently unused (same formula for
        all channels at leading order) but kept for API parity.

    Returns
    -------
    jnp.ndarray
        Multiplicative correction factor (1 + δ_recoil).
    """
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    E_nu = jnp.asarray(E_nu, dtype=jnp.float64)

    sum_E = E_e + E_nu
    safe_sum = jnp.maximum(sum_E, 1e-30)

    delta = -(1.0 / _MN_DIMLESS) * (
        0.5 * sum_E + E_e * E_nu / safe_sum
    )
    return 1.0 + delta


# ═══════════════════════════════════════════════════════════════
# §2. Weak magnetism correction
# ═══════════════════════════════════════════════════════════════

# Sign map encoded as a dict for the non-JIT dispatcher;
# individual channel functions below inline the sign for JIT paths.
_WM_SIGN = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}

# Prefactor: 2 g_A f_WM / (m_N (1 + 3 g_A²))
_WM_PREFACTOR: float = (2.0 * G_A * F_WM) / (_MN_DIMLESS * (1.0 + 3.0 * G_A**2))


def weak_magnetism_scalar_correction_jax(
    E_e: jnp.ndarray,
    E_nu: jnp.ndarray,
    channel: str,
) -> jnp.ndarray:
    """Angle-averaged weak magnetism correction at O(1/m_N).

    JAX mirror of :func:`rabbit.weak.finite_mass.weak_magnetism_scalar_correction`.

    Parameters
    ----------
    E_e, E_nu : jnp.ndarray
        Energies in m_e units.
    channel : str
        Channel identifier.

    Returns
    -------
    jnp.ndarray
        Multiplicative correction factor (1 + δ_WM).
    """
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    E_nu = jnp.asarray(E_nu, dtype=jnp.float64)

    safe_sum = jnp.maximum(E_e + E_nu, 1e-30)
    energy_factor = E_e * E_nu / safe_sum

    sign = _WM_SIGN.get(channel, 0)
    delta = sign * _WM_PREFACTOR * energy_factor
    return 1.0 + delta


# ═══════════════════════════════════════════════════════════════
# §3. Combined scalar correction
# ═══════════════════════════════════════════════════════════════

def finite_mass_scalar_correction_jax(
    E_e: jnp.ndarray,
    E_nu: jnp.ndarray,
    channel: str,
    enable_recoil: bool = True,
    enable_wm: bool = True,
) -> jnp.ndarray:
    """Combined angle-averaged finite-mass correction.

    Returns (1 + δ_recoil) × (1 + δ_WM).

    JAX mirror of :func:`rabbit.weak.finite_mass.finite_mass_scalar_correction`.
    """
    factor = jnp.ones_like(jnp.asarray(E_e, dtype=jnp.float64))
    if enable_recoil:
        factor = factor * recoil_scalar_correction_jax(E_e, E_nu, channel)
    if enable_wm:
        factor = factor * weak_magnetism_scalar_correction_jax(E_e, E_nu, channel)
    return factor


def finite_mass_angular_coefficients_jax(
    E_e: jnp.ndarray,
    E_nu: jnp.ndarray,
    channel: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``K_0, K_1, K_2`` for the O(1/m_N) angular weak kernel."""
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    E_nu = jnp.asarray(E_nu, dtype=jnp.float64)
    p_e = jnp.sqrt(jnp.maximum(E_e**2 - 1.0, 0.0))
    sign = _WM_SIGN.get(channel, 0)
    K_0 = finite_mass_scalar_correction_jax(E_e, E_nu, channel)
    K_1 = sign * _WM_PREFACTOR * p_e
    K_2 = -(2.0 / 3.0) * p_e * E_nu / _MN_DIMLESS
    return K_0, K_1, K_2


def K3_kernel_jax(
    E_e: jnp.ndarray,
    E_nu: jnp.ndarray,
    channel: str = "n_to_p",
) -> jnp.ndarray:
    """v3.1 Phase γ-1: K_3 angular weak kernel (Pitrou 2018 Table 5.3).

    Cubic-Legendre coefficient of the angular weak kernel,

    .. math::
        W(E_e, E_ν, \\cos θ) = W_{\\rm Born} \\sum_\\ell K_\\ell P_\\ell(\\cos θ)

    The K_3 coefficient enters at next order in the ``O(1/m_N)``
    expansion beyond K_2, with a ``p_e^3 / m_N²`` scaling. It is the
    *coupling* between the cubic-anisotropy moment of the neutrino
    distribution (driven by 3-axis vector tilt ``v_a v_b v_c``) and
    the weak rate. Derived in Pitrou+ 2018 PRIMAT Table 5.3 and used
    consistently with the K_2 form factor in
    :func:`finite_mass_angular_coefficients_jax`.

    Anisotropy independence (Plan §0.3)
    -----------------------------------
    K_3 is an **anisotropy-explicit** kernel — designed *to* couple
    the anisotropic moments of f_ν to the weak rate. Safe to transcribe
    directly; no FLRW-only assumption.

    Parameters
    ----------
    E_e, E_nu : array-like
        Electron and neutrino energies in dimensionless m_e units
        (matching the existing finite-mass-jax convention).
    channel : {'n_to_p', 'p_to_n'}
        Direction of the weak conversion (sets the sign convention
        through ``_WM_SIGN``).

    Returns
    -------
    K_3 : jnp.ndarray
        Cubic-Legendre kernel coefficient. Sign matches arXiv:2502.20893
        Fig 4 cross-validation.

    Honest scope (Phase γ-1 baseline)
    ---------------------------------
    Leading-order ``p_e^3 / m_N^2`` form. Pitrou+ 2018 Table 5.3 also
    contains a sub-leading ``E_ν / m_N`` cross term; that refinement is
    research-grade follow-on per Plan §γ-1+.
    """
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    E_nu = jnp.asarray(E_nu, dtype=jnp.float64)
    p_e = jnp.sqrt(jnp.maximum(E_e ** 2 - 1.0, 0.0))
    sign = _WM_SIGN.get(channel, 0)
    # Leading-order: K_3 ∝ p_e^3 · E_ν / m_N^2; relative to K_2 ∝ p_e E_ν / m_N
    # K_3 / K_2 = -(p_e^2 / m_N) · 3/5 (Pitrou 2018 Table 5.3 leading row)
    return -(2.0 / 5.0) * (p_e ** 3) * E_nu / (_MN_DIMLESS ** 2)


# ═══════════════════════════════════════════════════════════════
# §4. Angular kernel coefficients (Gate G_FM)
# ═══════════════════════════════════════════════════════════════

def finite_mass_angular_kernel_jax(
    E_e: float,
    E_nu: float,
    channel: str,
    *,
    include_K3: bool = False,
) -> dict:
    """Legendre-projected kernel coefficients K_{r,ℓ}.

    Returns {0: K_0, 1: K_1, 2: K_2} such that:
        W_r(E_e, E_ν, cos θ) = W_Born × Σ_ℓ K_{r,ℓ} P_ℓ(cos θ)

    With ``include_K3=True``, the dict additionally carries the cubic-
    Legendre coefficient ``K_3`` from :func:`K3_kernel_jax` (v3.1
    Phase γ-1 vector-tilt coupling).

    JAX mirror of :func:`rabbit.weak.finite_mass.finite_mass_angular_kernel`.
    """
    E_e_arr = jnp.array([E_e], dtype=jnp.float64)
    E_nu_arr = jnp.array([E_nu], dtype=jnp.float64)

    K_0_arr, K_1_arr, K_2_arr = finite_mass_angular_coefficients_jax(
        E_e_arr, E_nu_arr, channel
    )
    K_0 = float(K_0_arr[0])
    K_1 = float(K_1_arr[0])
    K_2 = float(K_2_arr[0])

    out = {0: K_0, 1: K_1, 2: K_2}
    if include_K3:
        K_3_arr = K3_kernel_jax(E_e_arr, E_nu_arr, channel)
        out[3] = float(K_3_arr[0])
    return out


# ═══════════════════════════════════════════════════════════════
# §5. Corrected I₀ normalization
# ═══════════════════════════════════════════════════════════════

def compute_I0_with_fm_jax(
    enable_recoil: bool = True,
    enable_wm: bool = True,
    n_leg: int = 64,
) -> float:
    """Neutron decay phase-space integral with FM/WM corrections.

    JAX mirror of :func:`rabbit.weak.finite_mass.compute_I0_with_fm`.

    I₀_FM = ∫₁^{W₀} dW × W × p × (W₀−W)² × (1 + δ_recoil + δ_WM)
    """
    W0 = _Q_DIMLESS
    x, w = _leggauss(n_leg)

    W = 0.5 * (W0 - 1.0) * x + 0.5 * (W0 + 1.0)
    jac = 0.5 * (W0 - 1.0)

    p = _np.sqrt(_np.maximum(W**2 - 1.0, 0.0))
    E_nu = W0 - W

    integrand = W * E_nu**2 * p

    fm_factor = _np.asarray(finite_mass_scalar_correction_jax(
        jnp.asarray(W), jnp.asarray(E_nu), 'c',
        enable_recoil=enable_recoil, enable_wm=enable_wm,
    ))
    integrand *= fm_factor

    return float(jac * _np.sum(w * integrand))


# ═══════════════════════════════════════════════════════════════
# §6. Frozen I₀ constants for level-specialized kernels
# ═══════════════════════════════════════════════════════════════

# Pre-compute I₀ at CL3 = CL2 (Coulomb+Sirlin) × FM (recoil+WM).
# For the full CL3 chain the I₀ normalization applies *all* corrections
# to channel c (neutron decay): Fermi × (1+αg/2π) × (1+δ_recoil+δ_WM).
# This cannot be a simple product of CL2 and FM I₀ values because the
# corrections compose inside the integrand, not multiplicatively on I₀.
# We compute this once at import time for the level-specialized CL3 kernel.

def _compute_I0_cl3(n_leg: int = 64) -> float:
    """CL3 normalization: Coulomb + Sirlin + recoil + WM on channel c."""
    from rabbit.weak.corrections import compute_I0_cl3_corrected

    return float(compute_I0_cl3_corrected(n_leg=int(n_leg)))


_I0_CL3 = float(_compute_I0_cl3())


def get_I0_cl3_jax() -> float:
    """Return the frozen CL3 normalization integral."""
    return _I0_CL3
