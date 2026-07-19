"""
rabbit.thermo.eos_photon_electron — Photon + e± equation of state.

Provides ρ(T), p(T), s(T), dρ/dT for the electromagnetic plasma (γ + e±)
with finite electron mass m_e and O(e³) QED correction.

The e±γ plasma is treated as a perfect fluid throughout BBN (§5 of the
equation sheet; Kn ~ 10⁻¹⁸, δY_p ~ 10⁻²¹).

Limits:
    T ≫ m_e:  ρ → (π²/30)(2 + 7/2) T⁴ = (11/4)(π²/30) T⁴
              g_* → 11/2 = 5.5
    T ≪ m_e:  ρ → (π²/30) T⁴ (photon only)
              g_* → 2

The O(e³) QED correction is MANDATORY (R04: shifts N_eff by ~ −0.001).
It arises from the thermal photon self-energy at O(α) and the Debye
screening plasma correction at O(α^{3/2}).

References:
    Heckler, Phys. Rev. D 49, 611 (1994) — O(e²)
    Bennett et al., JCAP 04, 073 (2020) — O(e³), numerical
    PRIMAT companion paper §II.C — implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.laguerre import laggauss


# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

_M_E: float = 0.5109989500       # electron mass [MeV]
_ALPHA: float = 1.0 / 137.035999084  # fine structure constant
_PI: float = np.pi
_PI2: float = np.pi**2

#: Photon energy density prefactor: ρ_γ = (π²/15) T⁴.
#: Note: (π²/15) = 2 × (π²/30), factor 2 for two photon polarizations.
_RHO_GAMMA_PREFACTOR: float = _PI2 / 15.0

#: Stefan–Boltzmann: ρ_γ = (π²/30) × g_γ × T⁴ with g_γ = 2.
_RHO_UNIT_PREFACTOR: float = _PI2 / 30.0

_QED_CORRECTION_FINITE_MU_SCALED: str = "finite_mu_scaled"
_QED_CORRECTION_EXACT_FINITE_MU_SCALAR: str = "exact_finite_mu_scalar"


# ═══════════════════════════════════════════════════════════════════════
# §2. Fermi–Dirac integrals for e± with finite mass
# ═══════════════════════════════════════════════════════════════════════

# Pre-computed Gauss–Laguerre nodes for e± integrals.
_N_QUAD: int = 64
_GL_NODES, _GL_WEIGHTS = laggauss(_N_QUAD)


def _validate_qed_correction_model(qed_correction_model: str) -> str:
    model = str(qed_correction_model)
    if model in {"scaled", _QED_CORRECTION_FINITE_MU_SCALED}:
        return _QED_CORRECTION_FINITE_MU_SCALED
    if model in {"exact_scalar", _QED_CORRECTION_EXACT_FINITE_MU_SCALAR}:
        return _QED_CORRECTION_EXACT_FINITE_MU_SCALAR
    raise ValueError(
        "qed_correction_model must be 'finite_mu_scaled' or 'exact_finite_mu_scalar'."
    )


def _Imn(m: int, n: int, x: float) -> float:
    """Generalized Fermi–Dirac integral I_{m,n}(x).

    I_{m,n}(x) = ∫₀^∞ du  (u² + 2ux)^{n/2} (u + x)^m / (e^{√(u²+2ux+x²)} + 1)

    where x = m_e/T and u is the Gauss–Laguerre integration variable.

    Key cases:
        I_{2,1}(x): energy density integral → ρ_{e±}/T⁴
        I_{0,3}(x): pressure integral → p_{e±}/T⁴
        I_{0,1}(x): number density integral
    """
    if x > 50.0:
        return 0.0

    result = 0.0
    for i in range(_N_QUAD):
        u = _GL_NODES[i]
        w = _GL_WEIGHTS[i]

        E = u + x
        p_sq = u * u + 2.0 * u * x
        if p_sq <= 0.0:
            continue
        p = np.sqrt(p_sq)

        # Distribution: exp(u) / (exp(E) + 1) = 1 / (exp(x) + exp(-u))
        exp_neg_u = np.exp(-u)
        exp_x = np.exp(min(x, 500.0))
        denom = exp_x + exp_neg_u
        if abs(denom) < 1e-300:
            continue
        dist = 1.0 / denom

        integrand = E**m * p**n * dist
        result += w * integrand

    return result


# Cache for efficiency (deterministic, idempotent)
_IMN_CACHE: dict = {}
_CHARGE_ASYMMETRY_SUSCEPTIBILITY_CACHE: dict = {}


def _Imn_cached(m: int, n: int, x: float) -> float:
    """Cached Fermi–Dirac integral."""
    key = (m, n, round(x, 12))
    if key not in _IMN_CACHE:
        _IMN_CACHE[key] = _Imn(m, n, x)
    return _IMN_CACHE[key]


def _finite_mass_fd_integral_with_mu(
    x: float,
    mu_over_T: float,
    *,
    energy_power: int,
    momentum_power: int,
) -> float:
    """Return a finite-mass FD integral for one signed-chemical-potential species."""

    if x > 50.0 and mu_over_T < x - 50.0:
        return 0.0
    shift = x - float(mu_over_T)
    result = 0.0
    exp_shift = np.exp(np.clip(shift, -500.0, 500.0))
    for i in range(_N_QUAD):
        u = _GL_NODES[i]
        w = _GL_WEIGHTS[i]
        E = u + x
        p_sq = u * u + 2.0 * u * x
        if p_sq <= 0.0:
            continue
        p = np.sqrt(p_sq)
        denom = exp_shift + np.exp(-u)
        if abs(denom) < 1e-300:
            continue
        result += w * E**energy_power * p**momentum_power / denom
    return result


def _electron_number_integral_with_mu(x: float, mu_over_T: float) -> float:
    """Return the finite-mass electron number integral for signed ``mu/T``."""

    return _finite_mass_fd_integral_with_mu(
        x,
        mu_over_T,
        energy_power=1,
        momentum_power=1,
    )


# ═══════════════════════════════════════════════════════════════════════
# §3. Core EOS functions (tree-level)
# ═══════════════════════════════════════════════════════════════════════

def rho_photon(T_MeV: float) -> float:
    """Photon energy density ρ_γ = (π²/15) T⁴ [MeV⁴]."""
    return _RHO_GAMMA_PREFACTOR * T_MeV**4


def rho_electron(T_MeV: float) -> float:
    """Electron + positron energy density ρ_{e±} [MeV⁴].

    ρ_{e±} = 2 × (T⁴/π²) × I_{2,1}(m_e/T)

    Factor 2 for particle + antiparticle.
    """
    x = _M_E / T_MeV
    if x > 50.0:
        return 0.0
    return 2.0 * T_MeV**4 / _PI2 * _Imn_cached(2, 1, x)


def pressure_electron(T_MeV: float) -> float:
    """Electron + positron pressure p_{e±} [MeV⁴].

    p_{e±} = 2 × (T⁴/(3π²)) × I_{0,3}(m_e/T)
    """
    x = _M_E / T_MeV
    if x > 50.0:
        return 0.0
    return 2.0 * T_MeV**4 / (3.0 * _PI2) * _Imn_cached(0, 3, x)


def electron_number_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Electron number density n_e- [MeV³] for finite mass and chemical potential."""

    T = float(T_MeV)
    mu = float(chemical_potential_MeV)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    x = _M_E / T
    integral = _electron_number_integral_with_mu(x, mu / T)
    return T**3 / _PI2 * integral


def positron_number_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Positron number density n_e+ [MeV³] for finite mass and electron μ_e."""

    return electron_number_density(T_MeV, -float(chemical_potential_MeV))


def electron_energy_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Electron energy density rho_e- [MeV^4] for finite mass and chemical potential."""

    T = float(T_MeV)
    mu = float(chemical_potential_MeV)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    x = _M_E / T
    integral = _finite_mass_fd_integral_with_mu(
        x,
        mu / T,
        energy_power=2,
        momentum_power=1,
    )
    return T**4 / _PI2 * integral


def positron_energy_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Positron energy density rho_e+ [MeV^4] for finite mass and electron mu_e."""

    return electron_energy_density(T_MeV, -float(chemical_potential_MeV))


def electron_positron_energy_density(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
) -> float:
    """Electron plus positron finite-mass energy density for signed electron mu_e."""

    mu = float(chemical_potential_MeV)
    return electron_energy_density(T_MeV, mu) + positron_energy_density(T_MeV, mu)


def electron_pressure_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Electron pressure p_e- [MeV^4] for finite mass and chemical potential."""

    T = float(T_MeV)
    mu = float(chemical_potential_MeV)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    x = _M_E / T
    integral = _finite_mass_fd_integral_with_mu(
        x,
        mu / T,
        energy_power=0,
        momentum_power=3,
    )
    return T**4 / (3.0 * _PI2) * integral


def positron_pressure_density(T_MeV: float, chemical_potential_MeV: float = 0.0) -> float:
    """Positron pressure p_e+ [MeV^4] for finite mass and electron mu_e."""

    return electron_pressure_density(T_MeV, -float(chemical_potential_MeV))


def electron_positron_pressure_density(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
) -> float:
    """Electron plus positron finite-mass pressure for signed electron mu_e."""

    mu = float(chemical_potential_MeV)
    return electron_pressure_density(T_MeV, mu) + positron_pressure_density(T_MeV, mu)


def electron_charge_asymmetry_density(T_MeV: float, chemical_potential_MeV: float) -> float:
    """Return n_e- - n_e+ [MeV³] for finite-mass Fermi-Dirac e±."""

    mu = float(chemical_potential_MeV)
    return electron_number_density(T_MeV, mu) - positron_number_density(T_MeV, mu)


def electron_charge_asymmetry_susceptibility(T_MeV: float) -> float:
    """Return d(n_e- - n_e+)/dmu_e at mu_e=0 [MeV^2]."""

    T = float(T_MeV)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    key = round(T, 12)
    if key in _CHARGE_ASYMMETRY_SUSCEPTIBILITY_CACHE:
        return _CHARGE_ASYMMETRY_SUSCEPTIBILITY_CACHE[key]

    x = _M_E / T
    exp_x = np.exp(np.clip(x, -500.0, 500.0))
    result = 0.0
    for i in range(_N_QUAD):
        u = _GL_NODES[i]
        w = _GL_WEIGHTS[i]
        E = u + x
        p_sq = u * u + 2.0 * u * x
        if p_sq <= 0.0:
            continue
        p = np.sqrt(p_sq)
        exp_neg_u = np.exp(-u)
        denom = exp_x + exp_neg_u
        if abs(denom) < 1.0e-300:
            continue
        result += w * E * p * exp_x / denom**2
    susceptibility = float(2.0 * T**2 / _PI2 * result)
    _CHARGE_ASYMMETRY_SUSCEPTIBILITY_CACHE[key] = susceptibility
    return susceptibility


def charge_neutral_electron_chemical_potential(
    T_MeV: float,
    positive_charge_density_MeV3: float,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-18,
    max_iterations: int = 160,
) -> float:
    """Solve n_e- - n_e+ = positive_charge_density for μ_e [MeV]."""

    T = float(T_MeV)
    target = float(positive_charge_density_MeV3)
    rtol = float(relative_tolerance)
    atol = float(absolute_tolerance)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    if not np.isfinite(target):
        raise ValueError("positive_charge_density_MeV3 must be finite.")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("relative_tolerance and absolute_tolerance must be positive.")
    if target == 0.0:
        return 0.0

    tolerance = max(atol, rtol * abs(target))

    def residual(mu: float) -> float:
        return electron_charge_asymmetry_density(T, mu) - target

    susceptibility = electron_charge_asymmetry_susceptibility(T)
    scale = max(T, _M_E)
    if susceptibility > 0.0:
        linear_mu = target / susceptibility
        if np.isfinite(linear_mu) and abs(linear_mu) <= 1.0e-6 * scale:
            return float(linear_mu)
        if np.isfinite(linear_mu) and abs(linear_mu) <= 5.0e-2 * scale:
            if abs(residual(linear_mu)) <= max(tolerance, 1.0e-8 * abs(target)):
                return float(linear_mu)

    lo = -scale
    hi = scale

    f_lo = residual(lo)
    f_hi = residual(hi)
    expansions = 0
    while f_lo > 0.0 and expansions < 32:
        lo *= 2.0
        f_lo = residual(lo)
        expansions += 1
    expansions = 0
    while f_hi < 0.0 and expansions < 32:
        hi *= 2.0
        f_hi = residual(hi)
        expansions += 1
    if f_lo > 0.0 or f_hi < 0.0:
        raise RuntimeError("failed to bracket charge-neutral electron chemical potential.")

    mid = 0.0
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= tolerance:
            return float(mid)
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    return float(mid)


def rho_plasma_tree(T_MeV: float) -> float:
    """Total plasma energy density (tree-level, no QED).

    ρ_plasma = ρ_γ + ρ_{e±}
    """
    return rho_photon(T_MeV) + rho_electron(T_MeV)


def pressure_plasma_tree(T_MeV: float) -> float:
    """Total plasma pressure (tree-level).

    p_plasma = ρ_γ/3 + p_{e±}
    """
    return rho_photon(T_MeV) / 3.0 + pressure_electron(T_MeV)


def entropy_density_plasma_tree(T_MeV: float) -> float:
    """Plasma entropy density s = (ρ + p)/T [MeV³].

    In the relativistic limit: s = (2π²/45) g_{*S} T³.
    """
    rho = rho_plasma_tree(T_MeV)
    p = pressure_plasma_tree(T_MeV)
    return (rho + p) / T_MeV


# ═══════════════════════════════════════════════════════════════════════
# §4. O(e³) QED correction (MANDATORY)
# ═══════════════════════════════════════════════════════════════════════

def _qed_delta_rho_from_electron_pair_energy(
    T_MeV: float,
    electron_pair_rho_MeV4: float,
) -> float:
    """Evaluate the existing isotropic QED correction for an e-/e+ bath."""

    T = float(T_MeV)
    rho_pair = float(electron_pair_rho_MeV4)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    if not np.isfinite(rho_pair) or rho_pair < 0.0:
        raise ValueError("electron_pair_rho_MeV4 must be finite and non-negative.")

    x = _M_E / T
    if x > 50.0:
        return 0.0

    # Relativistic e± energy density (massless, zero-mu limit).
    rho_rel_e = (7.0 / 4.0) * _RHO_GAMMA_PREFACTOR * T**4

    # Suppression/enhancement fraction for the current isotropic e-/e+ bath.
    f_e = max(rho_pair, 0.0) / max(rho_rel_e, 1e-100)

    # O(e²): thermal photon self-energy.
    delta_rho_e2 = (5.0 * _ALPHA / (4.0 * _PI)) * rho_rel_e * f_e

    # O(e³): Debye screening (plasma correction), following the same
    # approximate Bennett-style scaling used by the canonical zero-mu path.
    delta_rho_e3 = -0.20 * delta_rho_e2 * np.sqrt(_ALPHA / _PI)

    return delta_rho_e2 + delta_rho_e3


def qed_delta_rho(T_MeV: float) -> float:
    """QED correction to plasma energy density [MeV⁴].

    Includes O(e²) thermal photon self-energy and O(e³) Debye screening.

    O(e²): δρ ≈ (5α/4π) × ρ_{e±,rel} × f_e(x)
        where f_e(x) = ρ_{e±}(T) / ρ_{e±,rel}(T) accounts for
        Boltzmann suppression.

    O(e³): δρ ≈ −0.20 × δρ^{(e²)} × √(α/π)
        Debye screening correction (Bennett et al. 2020).

    Returns
    -------
    float
        δρ [MeV⁴].  Add to ρ_plasma_tree to get the corrected density.
    """
    return _qed_delta_rho_from_electron_pair_energy(T_MeV, rho_electron(T_MeV))


def qed_delta_rho_with_electron_mu(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
    *,
    qed_correction_model: str = _QED_CORRECTION_FINITE_MU_SCALED,
) -> float:
    """Finite-mu extension of the isotropic QED plasma correction.

    ``finite_mu_scaled`` keeps the repository's existing O(e²)+O(e³)
    isotropic QED formula but evaluates its finite-mass e-/e+ density factor
    on the signed electron chemical-potential bath.

    ``exact_finite_mu_scalar`` uses the scalar occupation-level finite-mu
    extension of :mod:`rabbit.thermo.qed_eos_exact`.  It is still an isotropic
    plasma-frame EOS, not an anisotropic/tensor QED response.
    """

    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    model = _validate_qed_correction_model(qed_correction_model)
    if model == _QED_CORRECTION_EXACT_FINITE_MU_SCALAR:
        from rabbit.thermo.qed_eos_exact import delta_rho_qed_exact_with_electron_mu

        return float(delta_rho_qed_exact_with_electron_mu(T_MeV, mu))
    if mu == 0.0:
        return qed_delta_rho(T_MeV)
    return _qed_delta_rho_from_electron_pair_energy(
        T_MeV,
        electron_positron_energy_density(T_MeV, mu),
    )


def qed_delta_pressure_with_electron_mu(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
    *,
    qed_correction_model: str = _QED_CORRECTION_FINITE_MU_SCALED,
) -> float:
    """QED pressure correction for the selected signed-mu scalar EOS model."""

    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    model = _validate_qed_correction_model(qed_correction_model)
    if model == _QED_CORRECTION_EXACT_FINITE_MU_SCALAR:
        from rabbit.thermo.qed_eos_exact import delta_P_qed_exact_with_electron_mu

        return float(delta_P_qed_exact_with_electron_mu(T_MeV, mu))
    return qed_delta_rho_with_electron_mu(
        T_MeV,
        mu,
        qed_correction_model=model,
    ) / 3.0


# ═══════════════════════════════════════════════════════════════════════
# §5. Full EOS with QED correction
# ═══════════════════════════════════════════════════════════════════════

def rho_plasma(T_MeV: float) -> float:
    """Total plasma energy density WITH O(e³) QED correction.

    ρ = ρ_γ + ρ_{e±} + δρ_QED

    This is the canonical EOS for rabbit.
    """
    return rho_plasma_tree(T_MeV) + qed_delta_rho(T_MeV)


def rho_plasma_with_electron_mu(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
    *,
    qed_correction_model: str = _QED_CORRECTION_FINITE_MU_SCALED,
) -> float:
    """Plasma energy density with finite-mass signed electron chemical potential.

    The e-/e+ tree-level term uses the supplied signed electron chemical
    potential.  The small scalar QED correction is selected by
    ``qed_correction_model``.
    """

    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    model = _validate_qed_correction_model(qed_correction_model)
    if mu == 0.0 and model == _QED_CORRECTION_FINITE_MU_SCALED:
        return rho_plasma(T_MeV)
    return (
        rho_photon(T_MeV)
        + electron_positron_energy_density(T_MeV, mu)
        + qed_delta_rho_with_electron_mu(T_MeV, mu, qed_correction_model=model)
    )


def pressure_plasma(T_MeV: float) -> float:
    """Total plasma pressure WITH QED correction.

    p ≈ p_tree + δρ_QED / 3  (relativistic relation for correction).
    """
    return pressure_plasma_tree(T_MeV) + qed_delta_rho(T_MeV) / 3.0


def pressure_plasma_with_electron_mu(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
    *,
    qed_correction_model: str = _QED_CORRECTION_FINITE_MU_SCALED,
) -> float:
    """Plasma pressure with finite-mass signed electron chemical potential."""

    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    model = _validate_qed_correction_model(qed_correction_model)
    if mu == 0.0 and model == _QED_CORRECTION_FINITE_MU_SCALED:
        return pressure_plasma(T_MeV)
    return (
        rho_photon(T_MeV) / 3.0
        + electron_positron_pressure_density(T_MeV, mu)
        + qed_delta_pressure_with_electron_mu(T_MeV, mu, qed_correction_model=model)
    )


def entropy_density_plasma(T_MeV: float) -> float:
    """Plasma entropy density with QED correction."""
    rho = rho_plasma(T_MeV)
    p = pressure_plasma(T_MeV)
    return (rho + p) / T_MeV


def drho_dT(T_MeV: float, dT: float = 1e-5) -> float:
    """Numerical derivative dρ/dT [MeV³].

    Uses central finite difference with relative step dT/T.
    """
    h = max(T_MeV * dT, 1e-10)
    return (rho_plasma(T_MeV + h) - rho_plasma(T_MeV - h)) / (2.0 * h)


def drho_dT_plasma_with_electron_mu(
    T_MeV: float,
    chemical_potential_MeV: float = 0.0,
    dT: float = 1e-5,
    *,
    qed_correction_model: str = _QED_CORRECTION_FINITE_MU_SCALED,
) -> float:
    """Fixed-mu numerical derivative of the signed-mu plasma energy density."""

    mu = float(chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential_MeV must be finite.")
    model = _validate_qed_correction_model(qed_correction_model)
    if mu == 0.0 and model == _QED_CORRECTION_FINITE_MU_SCALED:
        return drho_dT(T_MeV, dT=dT)
    h = max(float(T_MeV) * dT, 1e-10)
    return (
        rho_plasma_with_electron_mu(T_MeV + h, mu, qed_correction_model=model)
        - rho_plasma_with_electron_mu(T_MeV - h, mu, qed_correction_model=model)
    ) / (2.0 * h)


# ═══════════════════════════════════════════════════════════════════════
# §6. Entropy ratio and effective g_*
# ═══════════════════════════════════════════════════════════════════════

def entropy_ratio_S(T_MeV: float) -> float:
    """Entropy ratio S(T) = s_plasma / s_γ.

    S → 11/4 = 2.75  for T ≫ m_e
    S → 1            for T ≪ m_e

    Includes QED correction.
    """
    s_gamma = (4.0 / 3.0) * rho_photon(T_MeV) / T_MeV
    s_plasma = entropy_density_plasma(T_MeV)
    return s_plasma / max(s_gamma, 1e-100)


def g_star_rho(T_MeV: float) -> float:
    """Effective g_* for energy density.

    g_* = ρ_plasma / (π²/30 × T⁴)
    """
    rho = rho_plasma(T_MeV)
    return rho / (_RHO_UNIT_PREFACTOR * T_MeV**4)


def g_star_s(T_MeV: float) -> float:
    """Effective g_{*S} for entropy.

    From s = (2π²/45) g_{*S} T³:  g_{*S} = 45/(2π²) × s/T³
    """
    s = entropy_density_plasma(T_MeV)
    return (45.0 / (2.0 * _PI2)) * s / max(T_MeV**3, 1e-100)
