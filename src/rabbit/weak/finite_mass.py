"""
rabbit.weak.finite_mass — Finite nucleon mass and weak magnetism corrections.

Implements O(1/m_N) corrections to the weak n↔p rate integrands:

    1. Kinematic recoil: finite-mass energy-momentum conservation
       modifies the Born phase space. Net effect: reduces rates by ~1-2%.
       ΔY_p ≈ −0.0016 to −0.0018.

    2. Weak magnetism: anomalous magnetic moment σ^{μν} coupling
       introduces angular dependence in |M|². Sign flips between
       particle and antiparticle channels.
       ΔY_p ≈ −0.0005 to −0.0007.

ANISOTROPY SIGNIFICANCE (Gate G_FM):
    At Born level, the kernel W_r(q) is angle-independent → only f_{ν,0}
    enters weak rates.

    At O(1/m_N), the kernel gains angular dependence:
        W_r(q, cos θ_{eν}) = W_{r,0}(q) + W_{r,1}(q) cos θ + W_{r,2}(q) cos²θ

    This means f_{ν,2} enters the rate DIRECTLY for the first time:
        λ_r = Σ_ℓ ∫ K_{r,ℓ}(q) f_{ν,ℓ}(q) dq

    For Paper I (orthogonal, Σ_H ~ 0.1):
        |direct f₂ effect| ~ O(E/m_N × Ψ₂) ~ 2×10⁻⁴
    This is subdominant to Channel 1 (O(Σ²) ~ 10⁻²) but not negligible
    for FLRW precision targets.

Formulation:
    We follow the Seckel (1993, hep-ph/9305311) / Lopez & Turner (1999)
    formalism as systematized by PRIMAT (Pitrou et al. 2018, §III.G-H).

    The correction is expressed as a multiplicative factor on the Born integrand:
        dΓ_corrected = dΓ_Born × (1 + δ_recoil + δ_WM)

    For angle-averaged (scalar) rates:
        ⟨1 + δ⟩ = 1 + δ_scalar(E_e, E_ν, channel)

    For angle-resolved (projected kernel) rates:
        1 + δ(θ) = 1 + δ_0 + δ_1 cos θ + δ_2 cos²θ

References:
    Seckel 1993, hep-ph/9305311
    Lopez & Turner 1999, PRD 59, 103502
    PRIMAT: Pitrou et al. 2018, Phys. Rep. 754, 1 (§III.G-H, Appendix B.3-B.4)
    PRyMordial: Burns, Tait, Valli 2024, EPJC 84, 86 (§2.2, Eq. 2.12)
"""
from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════

M_E_MEV: float = 0.5109989500       # electron mass [MeV]
M_N_MEV: float = 939.565420         # average nucleon mass [MeV]
Q_NP_MEV: float = 1.29333236        # n-p mass splitting [MeV]

# Weak coupling constants
G_A: float = 1.2754                  # axial coupling constant |g_A|
KAPPA: float = 3.706                 # isovector anomalous magnetic moment
                                      # κ = μ_p − μ_n − 1 = 2.793−(−1.913)−1
F_WM: float = KAPPA / 2.0           # weak magnetism parameter f_WM = κ/2 = 1.853

# Derived
_Q_DIMLESS: float = Q_NP_MEV / M_E_MEV  # ≈ 2.531
_MN_DIMLESS: float = M_N_MEV / M_E_MEV  # ≈ 1838.7


# ═══════════════════════════════════════════════════════════════
# §2. Scalar (angle-averaged) recoil correction
# ═══════════════════════════════════════════════════════════════

def recoil_scalar_correction(E_e: np.ndarray, E_nu: np.ndarray,
                              channel: str) -> np.ndarray:
    """Angle-averaged kinematic recoil correction at O(1/m_N).

    This is the scalar (ℓ=0) part of the finite-mass correction.
    It modifies the Born integrand multiplicatively:
        dΓ → dΓ × (1 + δ_recoil)

    The leading correction from finite-mass phase space (Seckel 1993,
    Eq. 12; Lopez & Turner 1999) involves the energy transfer to
    the nucleon and the modified neutrino energy.

    For 2-body channels, the angle-averaged leading recoil correction is:
        δ_recoil ≈ −(E_ν + E_e)/(2 m_N) × C_channel

    where C_channel accounts for the specific kinematics.

    Parameters
    ----------
    E_e : ndarray
        Electron/positron energy in m_e units.
    E_nu : ndarray
        Neutrino energy in m_e units.
    channel : str
        Channel identifier ('a'–'f').

    Returns
    -------
    ndarray
        Multiplicative correction factor (1 + δ_recoil).
    """
    E_e = np.asarray(E_e, dtype=float)
    E_nu = np.asarray(E_nu, dtype=float)

    # Leading recoil: phase-space shrinkage from finite nucleon mass.
    # The corrected matrix element × phase space, after angle averaging,
    # gives a factor (Seckel Eq. 12, Lopez correction):
    #
    # For n→p channels (a,b,c):
    #   δ_recoil ≈ -(1/m_N)[E_ν + E_e - (E_ν - E_e)²/(2(E_ν + E_e))]
    #            = -(1/m_N)[(E_ν + E_e)/2 + E_ν E_e/(E_ν + E_e)]
    #
    # For p→n channels (d,e,f): same form with Q → -Q in kinematics

    sum_E = E_e + E_nu
    safe_sum = np.maximum(sum_E, 1e-30)

    # Phase-space recoil: leading O(1/m_N) correction
    # From Seckel Eq. 12 (angle-averaged):
    delta = -(1.0 / _MN_DIMLESS) * (
        0.5 * sum_E + E_e * E_nu / safe_sum
    )

    return 1.0 + delta


# ═══════════════════════════════════════════════════════════════
# §3. Weak magnetism correction
# ═══════════════════════════════════════════════════════════════

def weak_magnetism_scalar_correction(E_e: np.ndarray, E_nu: np.ndarray,
                                      channel: str) -> np.ndarray:
    """Angle-averaged weak magnetism correction at O(1/m_N).

    Weak magnetism arises from the anomalous magnetic moment of the nucleon,
    adding a σ^{μν} tensor coupling to the standard V-A current.

    The correction has a SIGN FLIP between particle and antiparticle channels:
        - n→p with e⁻ (channels a, c): δ_WM > 0 (enhancement)
        - n→p with e⁺ (channel b): δ_WM < 0 (suppression)
        - p→n with e⁻ (channels d, f): δ_WM < 0 (suppression)
        - p→n with e⁺ (channel e): δ_WM > 0 (enhancement)

    The net effect after summing all channels is a small NEGATIVE
    correction to Y_p (ΔY_p ≈ −0.0005 to −0.0007).

    From PRIMAT §III.H and PRyMordial Eq. 2.12:
        δ_WM = sign × (2 g_A f_WM) / (m_N (1+3g_A²)) × (E_e E_ν / (E_e + E_ν))

    The sign convention follows Horowitz (2002) / PRIMAT:
        sign = +1 for electron channels (a,c,d,f) where WM enhances n→p
        sign = -1 for positron channels (b,e)

    Parameters
    ----------
    E_e, E_nu : ndarray
        Energies in m_e units.
    channel : str
        Channel identifier.

    Returns
    -------
    ndarray
        Multiplicative correction factor (1 + δ_WM).
    """
    E_e = np.asarray(E_e, dtype=float)
    E_nu = np.asarray(E_nu, dtype=float)

    # WM prefactor: 2 g_A f_WM / (m_N (1+3g_A²))
    prefactor = (2.0 * G_A * F_WM) / (_MN_DIMLESS * (1.0 + 3.0 * G_A**2))

    # Energy factor (angle-averaged): E_e × E_ν / (E_e + E_ν)
    safe_sum = np.maximum(E_e + E_nu, 1e-30)
    energy_factor = E_e * E_nu / safe_sum

    # Sign convention per channel
    # From PRIMAT: WM shifts n→p rates and p→n rates in opposite directions.
    # Within n→p: electron (a,c) enhanced, positron (b) suppressed.
    # Within p→n: electron (d,f) suppressed, positron (e) enhanced.
    # Net: n→p gets a bit smaller, p→n gets a bit smaller → Y_p decreases.
    sign_map = {
        'a': +1,  # ν_e + n → p + e⁻ (electron final state)
        'b': -1,  # e⁺ + n → p + ν̄_e (positron initial state)
        'c': +1,  # n → p + e⁻ + ν̄_e (electron final state)
        'd': -1,  # p + e⁻ → n + ν_e (reversed from a)
        'e': +1,  # p + ν̄_e → n + e⁺ (reversed from b)
        'f': -1,  # p + e⁻ + ν̄_e → n (reversed from c)
    }
    sign = sign_map.get(channel, 0)

    delta = sign * prefactor * energy_factor
    return 1.0 + delta


# ═══════════════════════════════════════════════════════════════
# §4. Combined scalar correction
# ═══════════════════════════════════════════════════════════════

def finite_mass_scalar_correction(E_e: np.ndarray, E_nu: np.ndarray,
                                   channel: str,
                                   enable_recoil: bool = True,
                                   enable_wm: bool = True) -> np.ndarray:
    """Combined angle-averaged finite-mass correction factor.

    Returns (1 + δ_recoil) × (1 + δ_WM) ≈ 1 + δ_recoil + δ_WM
    at leading order in 1/m_N.

    Parameters
    ----------
    E_e, E_nu : ndarray
        Energies in m_e units.
    channel : str
        Channel identifier.
    enable_recoil, enable_wm : bool
        Toggle individual corrections.

    Returns
    -------
    ndarray
        Multiplicative correction factor.
    """
    factor = np.ones_like(np.asarray(E_e, dtype=float))

    if enable_recoil:
        factor *= recoil_scalar_correction(E_e, E_nu, channel)

    if enable_wm:
        factor *= weak_magnetism_scalar_correction(E_e, E_nu, channel)

    return factor


# ═══════════════════════════════════════════════════════════════
# §5. Angular kernel coefficients (Gate G_FM)
# ═══════════════════════════════════════════════════════════════

def finite_mass_angular_kernel(E_e: float, E_nu: float,
                                channel: str) -> dict:
    """Legendre-projected kernel coefficients for FM/WM correction.

    Returns K_{r,ℓ} coefficients such that:
        W_r(E_e, E_ν, cos θ) = W_Born × Σ_ℓ K_{r,ℓ} P_ℓ(cos θ)

    For the finite-mass correction at O(1/m_N):
        K_{r,0} = 1 + δ_scalar(E_e, E_ν)  (scalar part)
        K_{r,1} = b₁(E_e, E_ν) / m_N      (dipole: WM cos θ term)
        K_{r,2} = c₂(E_e, E_ν) / m_N      (quadrupole: recoil cos²θ)

    The ℓ=1 coefficient comes from weak magnetism (σ^{μν} p_ν term).
    The ℓ=2 coefficient comes from recoil kinematics (p_e · p_ν term).

    Parameters
    ----------
    E_e, E_nu : float
        Energies in m_e units.
    channel : str
        Channel identifier.

    Returns
    -------
    dict : {0: K_0, 1: K_1, 2: K_2}
        Legendre coefficients.
    """
    E_e_arr = np.array([E_e])
    E_nu_arr = np.array([E_nu])

    # ℓ=0: scalar (angle-averaged) correction
    K_0 = float(finite_mass_scalar_correction(E_e_arr, E_nu_arr, channel)[0])

    # ℓ=1: dipole (weak magnetism angular term)
    # The WM correction has a cos θ_{eν} dependence from the
    # (p_e × p_ν) term in the σ^{μν} coupling.
    # Projected: K_1 = (3/2) ∫ P₁(μ) δ_WM(μ) dμ
    # For the leading WM angular term:
    p_e = np.sqrt(max(E_e**2 - 1.0, 0.0))
    wm_angular_prefactor = (2.0 * G_A * F_WM) / (_MN_DIMLESS * (1.0 + 3.0 * G_A**2))

    sign_map = {'a': +1, 'b': -1, 'c': +1, 'd': -1, 'e': +1, 'f': -1}
    sign = sign_map.get(channel, 0)

    # The angular part of WM: δ_WM^{angular}(cos θ) ~ sign × prefactor × p_e × cos θ
    # This is O(1/m_N) — no additional m_N division needed.
    K_1 = sign * wm_angular_prefactor * p_e

    # ℓ=2: quadrupole (recoil p_e·p_ν cos²θ term)
    # From finite-mass kinematics at O(1/m_N):
    # The cos²θ correction from (p_e·p_ν)² phase space modification
    K_2 = -(2.0 / 3.0) * p_e * E_nu / (_MN_DIMLESS)

    return {0: K_0, 1: K_1, 2: K_2}


# ═══════════════════════════════════════════════════════════════
# §6. Corrected I₀ normalization
# ═══════════════════════════════════════════════════════════════

def compute_I0_with_fm(enable_recoil: bool = True,
                        enable_wm: bool = True,
                        n_leg: int = 64) -> float:
    """Neutron decay phase-space integral with FM/WM corrections.

    I₀_FM = ∫₁^{W₀} dW × W × p × (W₀−W)² × (1+δ_recoil+δ_WM)

    Must be used with the SAME corrections in the channel integrands
    to ensure λ_np → 1/τ_n at T→0.

    Note: At T=0, only channel (c) contributes (neutron decay).
    The recoil correction to I₀ is δ_n ≈ −0.00201 (Seckel Eq. 28).
    """
    from numpy.polynomial.legendre import leggauss

    W0 = _Q_DIMLESS  # endpoint
    x, w = leggauss(n_leg)

    W = 0.5 * (W0 - 1.0) * x + 0.5 * (W0 + 1.0)
    jac = 0.5 * (W0 - 1.0)

    p = np.sqrt(np.maximum(W**2 - 1.0, 0.0))
    E_nu = W0 - W  # in m_e units

    integrand = W * E_nu**2 * p

    # Apply FM/WM correction (channel c: neutron decay)
    fm_factor = finite_mass_scalar_correction(W, E_nu, 'c',
                                               enable_recoil=enable_recoil,
                                               enable_wm=enable_wm)
    integrand *= fm_factor

    return float(jac * np.sum(w * integrand))


# ═══════════════════════════════════════════════════════════════
# §7. Diagnostics
# ═══════════════════════════════════════════════════════════════

def fm_correction_summary():
    """Diagnostic summary of FM/WM correction magnitudes."""
    from weak_corrections import compute_I0_corrected

    I0_born = compute_I0_corrected(False, False)
    I0_coul_sir = compute_I0_corrected(True, True)
    I0_fm_recoil = compute_I0_with_fm(True, False)
    I0_fm_wm = compute_I0_with_fm(False, True)
    I0_fm_both = compute_I0_with_fm(True, True)

    return {
        'I0_born': I0_born,
        'I0_coulomb_sirlin': I0_coul_sir,
        'I0_fm_recoil_only': I0_fm_recoil,
        'I0_fm_wm_only': I0_fm_wm,
        'I0_fm_both': I0_fm_both,
        'delta_recoil': (I0_fm_recoil - I0_born) / I0_born,
        'delta_wm': (I0_fm_wm - I0_born) / I0_born,
        'delta_fm_total': (I0_fm_both - I0_born) / I0_born,
    }
