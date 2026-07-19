"""
rabbit.weak.corrections — Beyond-Born weak rate corrections.

Implements the first two correction levels of the five-level hierarchy:

    Level 1: Born (existing, in channels.py)
    Level 2: Coulomb (this module) — multiplicative Fermi function F(Z, E_e)
    Level 3: Radiative (this module) — multiplicative Sirlin g-function
    Level 4: Finite nucleon mass (future, finite_mass.py)
    Level 5: Thermal QED (future, thermal_correction.py)

Each correction is STRICTLY multiplicative on the Born integrand,
so the existing Teff-corrected distribution function flows through
unchanged.  The correction factors enter INSIDE the channel integrands.

Sign/charge convention per channel:
    The Fermi function F(Z, E_e) depends on whether the charged lepton
    interacting with the proton is an electron (Z=+1, attractive) or
    positron (Z=-1, repulsive):
        (a) ν_e + n → p + e⁻    : electron final state → F(Z=+1)
        (b) e⁺ + n → p + ν̄_e   : positron initial state → F(Z=-1)
        (c) n → p + e⁻ + ν̄_e   : electron final state → F(Z=+1)
        (d) p + e⁻ → n + ν_e   : electron initial state → F(Z=+1)
        (e) p + ν̄_e → n + e⁺   : positron final state → F(Z=-1)
        (f) p + e⁻ + ν̄_e → n   : electron initial state → F(Z=+1)

References:
    Fermi function: Hayen et al. 2018, RMP 90, 015008
    Sirlin g-function: Sirlin, Phys. Rev. 164, 1767 (1967)
    CMS resummation: Czarnecki, Marciano, Sirlin, PRD 70, 093006 (2004)
    PRIMAT: Pitrou et al. 2018, Phys. Rep. 754, 1 (§III.E, Appendix B)
    PRyMordial: Burns, Tait, Valli 2024, EPJC 84, 86 (§2.2)
"""
from __future__ import annotations

import numpy as np
from scipy.special import spence  # Li₂(z) = -∫₀ᶻ ln(1-t)/t dt
from rabbit.weak.finite_mass import finite_mass_scalar_correction


# ═══════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════

ALPHA: float = 1.0 / 137.035999084       # fine structure constant
M_ELECTRON_MEV: float = 0.5109989500     # electron mass [MeV]
M_PROTON_MEV: float = 938.272088         # proton mass [MeV]
Q_NP_MEV: float = 1.29333236            # n-p mass splitting [MeV]
M_P_OVER_M_E: float = M_PROTON_MEV / M_ELECTRON_MEV  # ≈ 1836.15

# Dimensionless endpoint for neutron decay: W₀ = Q_np/m_e
# (maximum total electron energy at the endpoint, in m_e units)
# Energy conservation: E_e + E_ν = Q_np, so E_e_max = Q_np when E_ν = 0.
# W₀ = E_e_max / m_e = Q_np / m_e ≈ 2.531
# Integration range for channel (c): W ∈ [1, W₀]
W0_NEUTRON_DECAY: float = Q_NP_MEV / M_ELECTRON_MEV  # ≈ 2.531


# ═══════════════════════════════════════════════════════════════
# §2. Coulomb correction: Sommerfeld Fermi function
# ═══════════════════════════════════════════════════════════════

def fermi_sommerfeld(E_e: np.ndarray, Z: int = 1) -> np.ndarray:
    """Sommerfeld Coulomb correction factor F(Z, E_e).

    F(η) = 2πη / (1 − e^{−2πη})

    where η = ±αZE_e/p_e (Sommerfeld parameter):
        η > 0 for electrons (attractive, F > 1)
        η < 0 for positrons (repulsive, F < 1)

    Parameters
    ----------
    E_e : ndarray
        Electron/positron energy in m_e units (W = E_e/m_e, so E_e ≥ 1).
    Z : int
        Nuclear charge.  +1 for proton (e⁻), -1 for positrons (e⁺).

    Returns
    -------
    ndarray
        Fermi function F(Z, E_e).  Returns 1 where E_e < 1 (below threshold).

    Notes
    -----
    Accuracy: ~10⁻⁴ for Z=1 at BBN energies (validated against PRIMAT).
    The full relativistic formula with Γ(γ₀+iη) gives sub-10⁻⁵ agreement.
    """
    E_e = np.asarray(E_e, dtype=float)
    result = np.ones_like(E_e)

    mask = E_e > 1.0 + 1e-15
    if not np.any(mask):
        return result

    W = E_e[mask]
    p = np.sqrt(W**2 - 1.0)
    eta = ALPHA * Z * W / p

    # Sommerfeld formula: 2πη / (1 - e^{-2πη})
    arg = -2.0 * np.pi * eta
    # Numerically stable: for large |η|, avoid overflow
    safe_arg = np.clip(arg, -500, 500)
    denom = 1.0 - np.exp(safe_arg)

    # Handle η → 0 limit: F → 1
    small = np.abs(eta) < 1e-10
    F = np.where(small, 1.0, 2.0 * np.pi * eta / denom)

    result[mask] = F
    return result


def coulomb_correction_per_channel(E_e: np.ndarray, channel: str) -> np.ndarray:
    """Coulomb correction factor for a specific weak channel.

    Parameters
    ----------
    E_e : ndarray
        Lepton energy in m_e units.
    channel : str
        One of 'a', 'b', 'c', 'd', 'e', 'f'.

    Returns
    -------
    ndarray
        Multiplicative correction factor.

    Channel assignments (PRIMAT convention):
        (a) ν_e + n → p + e⁻    : electron, F(Z=+1) > 1
        (b) e⁺ + n → p + ν̄_e   : positron, F(Z=-1) < 1
        (c) n → p + e⁻ + ν̄_e   : electron, F(Z=+1) > 1
        (d) p + e⁻ → n + ν_e   : electron, F(Z=+1) > 1
        (e) p + ν̄_e → n + e⁺   : positron, F(Z=-1) < 1
        (f) p + e⁻ + ν̄_e → n   : electron, F(Z=+1) > 1
    """
    # Electron channels: Z = +1 (attractive Coulomb with proton)
    # Positron channels: Z = -1 (repulsive)
    if channel in ('a', 'c', 'd', 'f'):
        return fermi_sommerfeld(E_e, Z=+1)
    elif channel in ('b', 'e'):
        return fermi_sommerfeld(E_e, Z=-1)
    else:
        raise ValueError(f"Unknown channel: {channel}")


# ═══════════════════════════════════════════════════════════════
# §3. Sirlin outer radiative correction: g-function
# ═══════════════════════════════════════════════════════════════

def _dilogarithm(x: np.ndarray) -> np.ndarray:
    """Compute Li₂(x) = -∫₀ˣ ln(1-t)/t dt.

    scipy.special.spence computes the Spence function S(x) = -Li₂(1-x).
    So Li₂(x) = S(1-x) = spence(1-x).

    Wait — scipy convention: spence(x) = ∫₁ˣ ln(t)/(1-t) dt = Li₂(1-x).
    So Li₂(x) = spence(1-x).
    """
    return spence(np.clip(1.0 - x, 0.0, 2.0))


def sirlin_g_function(W: np.ndarray, W0: float = W0_NEUTRON_DECAY) -> np.ndarray:
    """Sirlin outer radiative correction g(W₀, W).

    The Born integrand is multiplied by (1 + α/(2π) × g(W₀, W)).

    Formula (Sirlin 1967, Eq. 20; PRIMAT Appendix B.2):

        g(W₀,W) = 3 ln(m_p/m_e) − 3/4
                   + 4(L/β − 1)[(W₀−W)/(3W) − 3/2 + ln(2(W₀−W))]
                   + (L/β)[2(1+β²) + (W₀−W)²/(6W²) − 4L]
                   + (4/β) S(2β/(1+β))

    where:
        β = p/W = √(1 − 1/W²)    (electron velocity)
        L = arctanh(β)             (= ½ ln[(1+β)/(1−β)])
        S(x) = -Li₂(x)            (Spence function)

    Parameters
    ----------
    W : ndarray
        Total electron energy in m_e units (W ≥ 1).
    W0 : float
        Endpoint energy in m_e units.

    Returns
    -------
    ndarray
        g(W₀, W) at each energy point.
    """
    W = np.asarray(W, dtype=float)
    g = np.zeros_like(W)

    mask = (W > 1.0 + 1e-10) & (W < W0 - 1e-10)
    if not np.any(mask):
        return g

    Wm = W[mask]
    beta = np.sqrt(1.0 - 1.0 / Wm**2)
    L = np.arctanh(np.clip(beta, -0.99999, 0.99999))

    delta_W = W0 - Wm  # > 0 inside the phase space

    # Term 1: constant
    t1 = 3.0 * np.log(M_P_OVER_M_E) - 0.75

    # Term 2: 4(L/β - 1) × [...]
    L_over_beta = L / np.clip(beta, 1e-15, None)
    bracket2 = delta_W / (3.0 * Wm) - 1.5 + np.log(np.clip(2.0 * delta_W, 1e-30, None))
    t2 = 4.0 * (L_over_beta - 1.0) * bracket2

    # Term 3: (L/β) × [2(1+β²) + (W₀-W)²/(6W²) - 4L]
    t3 = L_over_beta * (2.0 * (1.0 + beta**2) + delta_W**2 / (6.0 * Wm**2) - 4.0 * L)

    # Term 4: (4/β) × S(2β/(1+β)) where S = -Li₂
    x_arg = 2.0 * beta / (1.0 + beta)
    S_val = -_dilogarithm(x_arg)  # S(x) = -Li₂(x)
    t4 = (4.0 / np.clip(beta, 1e-15, None)) * S_val

    g[mask] = t1 + t2 + t3 + t4
    return g


def radiative_correction_factor(E_e: np.ndarray, W0: float = W0_NEUTRON_DECAY) -> np.ndarray:
    """Multiplicative radiative correction factor (1 + α/(2π) × g).

    Parameters
    ----------
    E_e : ndarray
        Electron energy in m_e units (W = E_e).
    W0 : float
        Channel endpoint in m_e units.

    Returns
    -------
    ndarray
        Factor by which to multiply the Born integrand.
    """
    g = sirlin_g_function(E_e, W0)
    return 1.0 + (ALPHA / (2.0 * np.pi)) * g


# ═══════════════════════════════════════════════════════════════
# §4. Combined correction factor
# ═══════════════════════════════════════════════════════════════

def weak_correction_factor(
    E_e: np.ndarray,
    channel: str,
    enable_coulomb: bool = True,
    enable_radiative: bool = True,
) -> np.ndarray:
    """Combined multiplicative correction factor for a weak channel.

    Returns F(Z, E_e) × (1 + α/(2π) × g(W₀, W))  for bounded channels,
    or     F(Z, E_e)                                 for semi-infinite channels.

    The Sirlin g-function is applied ONLY to bounded channels (c, f)
    that share the neutron-decay kinematics with endpoint W₀ = Q_np/m_e.
    For semi-infinite channels (a, b, d, e), the dominant radiative
    correction (3 ln(m_p/m_e)) cancels between the channel integrand
    and the I₀ normalization, leaving only the Coulomb factor.

    Parameters
    ----------
    E_e : ndarray
        Lepton energy in m_e units.
    channel : str
        Weak channel identifier ('a'–'f').
    enable_coulomb : bool
        Apply Coulomb (Fermi function) correction.
    enable_radiative : bool
        Apply Sirlin outer radiative correction (bounded channels only).

    Returns
    -------
    ndarray
        Combined multiplicative factor (1.0 when both disabled).
    """
    factor = np.ones_like(np.asarray(E_e, dtype=float))

    if enable_coulomb:
        factor *= coulomb_correction_per_channel(E_e, channel)

    if enable_radiative:
        # Sirlin g only for bounded channels c (neutron decay) and f (reverse)
        # These have the well-defined endpoint W₀ = Q_np/m_e ≈ 2.531.
        # Semi-infinite channels (a,b,d,e): radiative correction is absorbed
        # into the I₀ normalization via the measured τ_n.
        if channel in ('c', 'f'):
            factor *= radiative_correction_factor(E_e, W0_NEUTRON_DECAY)

    return factor

    return factor


# ═══════════════════════════════════════════════════════════════
# §5. Correction to I₀ normalization
# ═══════════════════════════════════════════════════════════════

def compute_I0_corrected(
    enable_coulomb: bool = True,
    enable_radiative: bool = True,
    n_leg: int = 64,
    _cache: dict = {},
) -> float:
    """Neutron decay phase-space integral with corrections.

    I₀_corr = ∫₁^{W₀} dW × W × p × (W₀-W)² × F(+1,W) × (1+α/2π g)

    The normalization K = 1/(τ_n × I₀_corr) absorbs all corrections
    into the rate normalization, so that λ_np → 1/τ_n at T→0.

    This is ESSENTIAL: the corrections must be included consistently
    in BOTH the normalization AND the channel integrands.

    OPT-3: Result is T-independent and cached after first computation.
    """
    key = (enable_coulomb, enable_radiative, n_leg)
    if key in _cache:
        return _cache[key]

    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_leg)

    W0 = W0_NEUTRON_DECAY
    # Map [-1,1] → [1, W₀]
    W = 0.5 * (W0 - 1.0) * x + 0.5 * (W0 + 1.0)
    jac = 0.5 * (W0 - 1.0)

    p = np.sqrt(np.maximum(W**2 - 1.0, 0.0))
    E_nu = W0 - W

    integrand = W * E_nu**2 * p

    # Apply corrections
    corr = weak_correction_factor(W, 'c',
                                   enable_coulomb=enable_coulomb,
                                   enable_radiative=enable_radiative)
    integrand *= corr

    result = float(jac * np.sum(w * integrand))
    _cache[key] = result
    return result


def compute_I0_cl3_corrected(
    n_leg: int = 64,
    _cache: dict = {},
) -> float:
    """CL3 normalization: Coulomb + Sirlin + recoil + weak magnetism."""
    key = ("cl3", n_leg)
    if key in _cache:
        return _cache[key]

    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n_leg)

    W0 = W0_NEUTRON_DECAY
    W = 0.5 * (W0 - 1.0) * x + 0.5 * (W0 + 1.0)
    jac = 0.5 * (W0 - 1.0)
    p = np.sqrt(np.maximum(W**2 - 1.0, 0.0))
    E_nu = W0 - W

    integrand = W * E_nu**2 * p
    cl2 = weak_correction_factor(W, 'c', enable_coulomb=True, enable_radiative=True)
    fm = finite_mass_scalar_correction(W, E_nu, 'c', enable_recoil=True, enable_wm=True)
    integrand *= cl2 * fm

    result = float(jac * np.sum(w * integrand))
    _cache[key] = result
    return result


# ═══════════════════════════════════════════════════════════════
# §6. Diagnostics
# ═══════════════════════════════════════════════════════════════

def correction_summary(T_MeV: float = 1.0):
    """Print a diagnostic summary of correction magnitudes.

    Parameters
    ----------
    T_MeV : float
        Temperature for evaluating typical correction magnitudes.
    """
    T_dimless = T_MeV / M_ELECTRON_MEV
    E_e = np.linspace(1.01, 10.0, 100)

    F_plus = fermi_sommerfeld(E_e, Z=+1)
    F_minus = fermi_sommerfeld(E_e, Z=-1)
    g_vals = sirlin_g_function(E_e)
    rad_factor = radiative_correction_factor(E_e)

    I0_born = compute_I0_corrected(False, False)
    I0_coul = compute_I0_corrected(True, False)
    I0_rad = compute_I0_corrected(False, True)
    I0_both = compute_I0_corrected(True, True)

    return {
        'I0_born': I0_born,
        'I0_coulomb': I0_coul,
        'I0_radiative': I0_rad,
        'I0_both': I0_both,
        'delta_I0_coulomb': (I0_coul - I0_born) / I0_born,
        'delta_I0_radiative': (I0_rad - I0_born) / I0_born,
        'delta_I0_total': (I0_both - I0_born) / I0_born,
        'F_electron_mean': np.mean(F_plus),
        'F_positron_mean': np.mean(F_minus),
        'g_mean': np.mean(g_vals),
        'radiative_factor_mean': np.mean(rad_factor),
    }
