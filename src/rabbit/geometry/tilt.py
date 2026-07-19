"""
rabbit.geometry.tilt — Tilt evolution and frame corrections.

Tilt = bulk velocity of matter relative to geometry frame.
u^a = Γ(e₀^a + v^α eα^a),  Γ = 1/√(1-v²)

CRITICAL: Tilt GROWS during radiation era (γ=4/3).
  FLRW eigenvalue: λ = 7 − 9γ/2 = +1
  v ∝ a ∝ exp(N) → BBN requires v₀ ≲ 10⁻⁵

References: Hewitt, Bridson, Wainwright (2001) GRG 33, 65;
            Wainwright & Ellis (1997) Ch. 7
"""
import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# §1. Tilt state
# ═══════════════════════════════════════════════════════════════

@dataclass
class TiltState:
    """Current tilt state."""
    v: np.ndarray       # velocity components (1D or 3D)
    v_sq: float         # |v|²
    Gamma: float        # 1/√(1-v²)

    @classmethod
    def from_velocity(cls, v, scalar=False):
        if scalar:
            v = np.array([v])
        v = np.atleast_1d(np.asarray(v, dtype=float))
        v_sq = float(np.sum(v**2))
        if v_sq >= 1.0:
            raise ValueError(f"Superluminal: v²={v_sq}")
        return cls(v=v, v_sq=v_sq, Gamma=1.0/np.sqrt(1.0 - v_sq))

    @classmethod
    def zero(cls, ndim=1):
        return cls(v=np.zeros(ndim), v_sq=0.0, Gamma=1.0)

    @property
    def speed(self):
        return np.sqrt(self.v_sq)


# ═══════════════════════════════════════════════════════════════
# §2. Tilt stability
# ═══════════════════════════════════════════════════════════════

GAMMA_CRIT = 14.0 / 9.0  # ≈ 1.5556


def tilt_eigenvalue_flrw(gamma=4.0/3.0):
    """Linearized eigenvalue at FLRW: λ = 7 − 9γ/2.

    γ=4/3 (radiation): λ = +1 → GROWS
    γ=1 (dust): λ = +5/2 → GROWS faster
    γ=14/9: λ = 0 → marginal
    """
    return 7.0 - 4.5 * gamma


# ═══════════════════════════════════════════════════════════════
# §3. Tilt RHS
# ═══════════════════════════════════════════════════════════════

def compute_tilt_rhs_flrw(v_scalar, q=1.0, gamma=4.0/3.0):
    """RADIATION-ONLY diagnostic: scalar tilt evolution at FLRW (Σ=0).

    WARNING: This function is validated only for γ=4/3 (radiation).
    For other γ values, the full coupled system (compute_tilt_rhs_typeI)
    must be used. Do NOT use for dust (γ=1) or stiff (γ=2) matter.

    dv/dN = v × (1-v²)/G × [(2-q) − (3γ-4)]

    At FLRW radiation: q=1, (3γ-4)=0 for γ=4/3:
      dv/dN = v × (1-v²)/G × 1 ≈ v  (small v, eigenvalue +1)
    """
    v_sq = v_scalar**2
    if v_sq >= 0.99:
        return 0.0  # cap at v→1
    G = 1.0 + (gamma - 1.0) * v_sq
    eos_term = 3.0 * gamma - 4.0  # = 0 for radiation
    bracket = (2.0 - q) - eos_term
    return v_scalar * (1.0 - v_sq) / G * bracket


def compute_tilt_rhs_principal_axis(
    v_scalar,
    Sigma_plus,
    Sigma_minus,
    q,
    axis=3,
    gamma=4.0/3.0,
):
    """Scalar tilt RHS for velocity aligned with principal axis 1, 2, or 3.

    This is the diagonal-state analogue of the full vector equation.  Mixed
    axis tilt is not represented here because it would source off-diagonal
    anisotropic stress outside the current Wainwright-Hsu state vector.
    """
    if axis not in (1, 2, 3):
        raise ValueError(f"tilt principal axis must be 1, 2, or 3; got {axis!r}.")
    v_sq = float(v_scalar) ** 2
    if v_sq >= 0.99:
        return 0.0
    G = 1.0 + (float(gamma) - 1.0) * v_sq
    s3 = np.sqrt(3.0)
    lam = {
        1: -float(Sigma_plus) - s3 * float(Sigma_minus),
        2: -float(Sigma_plus) + s3 * float(Sigma_minus),
        3: 2.0 * float(Sigma_plus),
    }[axis]
    eos_term = 3.0 * float(gamma) - 4.0
    friction = (2.0 - float(q)) * (1.0 - (float(gamma) - 1.0) * v_sq) / G
    return float(v_scalar) * (1.0 - v_sq) / G * (friction + lam - eos_term)


def compute_tilt_rhs_typeI(tilt, Sigma_plus, Sigma_minus, q, gamma=4.0/3.0):
    """Full tilt evolution for Type I.

    dv_α/dN = v_α(1-v²)/G × [(2-q)(1-(γ-1)v²)/G + λ_α − (3γ-4)]

    where λ_α are shear eigenvalues:
      λ₁ = -Σ₊ - √3 Σ₋
      λ₂ = -Σ₊ + √3 Σ₋
      λ₃ = 2Σ₊
    """
    v = tilt.v
    v_sq = tilt.v_sq
    G = 1.0 + (gamma - 1.0) * v_sq

    if v_sq >= 0.99:
        return np.zeros_like(v)

    # Shear eigenvalues
    s3 = np.sqrt(3.0)
    lam = np.array([
        -Sigma_plus - s3 * Sigma_minus,
        -Sigma_plus + s3 * Sigma_minus,
        2.0 * Sigma_plus,
    ])

    prefactor = (1.0 - v_sq) / G
    eos_term = 3.0 * gamma - 4.0
    friction = (2.0 - q) * (1.0 - (gamma - 1.0) * v_sq) / G

    dv = np.zeros(min(len(v), 3))
    for a in range(len(dv)):
        dv[a] = v[a] * prefactor * (friction + lam[a] - eos_term)

    return dv


# ═══════════════════════════════════════════════════════════════
# §4. Frame corrections for BBN
# ═══════════════════════════════════════════════════════════════

def tilt_hubble_correction(Omega_orth, v_sq, gamma=4.0/3.0):
    """Ω_tilt from tilted Friedmann constraint.

    Ω_tilt = Ω_orth × G / (G + (2-γ)v²)

    For radiation (γ=4/3): correction = (1+v²/3)/(1+v²) ≈ 1 - 2v²/3
    """
    G = 1.0 + (gamma - 1.0) * v_sq
    denom = G + (2.0 - gamma) * v_sq
    if denom < 1e-30:
        return Omega_orth
    return Omega_orth * G / denom


def tilted_normal_energy_density_factor(v_sq, gamma=4.0/3.0):
    """Normal-frame T00 factor for a tilted perfect fluid.

    For ``p=(gamma-1) rho`` and ``u^a=Gamma(n^a+v^a)``, the normal-frame
    energy density is ``rho_n = rho * (1 + gamma * Gamma^2 * v^2)``.
    """
    gamma_sq = 1.0 / max(1.0 - float(v_sq), 1e-30)
    return 1.0 + float(gamma) * gamma_sq * float(v_sq)


def tilt_hubble_stress_energy_factor(v_sq, gamma=4.0/3.0):
    """H/H_orthogonal from tilted-fluid normal-frame energy density."""
    return np.sqrt(tilted_normal_energy_density_factor(v_sq, gamma=gamma))


def tilt_anisotropic_stress_scalar(v_sq, Omega_tilted, gamma=4.0/3.0):
    """Perfect-fluid ``(Π_+, Π_-)`` from scalar tilt along the 3-axis."""
    gamma_sq = 1.0 / max(1.0 - float(v_sq), 1e-30)
    pi_plus = float(gamma) * float(Omega_tilted) * gamma_sq * float(v_sq) / 3.0
    return pi_plus, 0.0


def tilt_anisotropic_stress_principal_axis(
    v_sq,
    Omega_tilted,
    axis=3,
    gamma=4.0/3.0,
):
    """Perfect-fluid ``(Π_+, Π_-)`` for principal-axis tilt."""
    if axis not in (1, 2, 3):
        raise ValueError(f"tilt principal axis must be 1, 2, or 3; got {axis!r}.")
    gamma_sq = 1.0 / max(1.0 - float(v_sq), 1e-30)
    c = float(gamma) * float(Omega_tilted) * gamma_sq * float(v_sq)
    if axis == 1:
        return -c / 6.0, -c / (2.0 * np.sqrt(3.0))
    if axis == 2:
        return -c / 6.0, c / (2.0 * np.sqrt(3.0))
    return c / 3.0, 0.0


def boosted_fd_legendre_moments(q_nodes, v, n_mu=16):
    """Return plasma-frame ``l=0,1,2`` moments of a boosted FD spectrum."""
    q_nodes = np.asarray(q_nodes, dtype=float)
    v = float(v)
    if abs(v) >= 1.0:
        raise ValueError(f"boosted FD moments require |v| < 1, got v={v}.")
    mu, w = np.polynomial.legendre.leggauss(int(n_mu))
    gamma_lorentz = 1.0 / np.sqrt(max(1.0 - v * v, 1e-30))
    q_geo = gamma_lorentz * q_nodes[:, None] * (1.0 + v * mu[None, :])
    f_mu = 1.0 / (np.exp(np.clip(q_geo, 0.0, 500.0)) + 1.0)
    p1 = mu[None, :]
    p2 = 0.5 * (3.0 * mu[None, :] ** 2 - 1.0)
    f0 = 0.5 * np.sum(w[None, :] * f_mu, axis=1)
    f1 = 1.5 * np.sum(w[None, :] * f_mu * p1, axis=1)
    f2 = 2.5 * np.sum(w[None, :] * f_mu * p2, axis=1)
    return np.clip(f0, 0.0, 1.0), f1, f2


def tilt_temperature(T_gamma, v_sq):
    """Fluid-frame temperature: T_fluid = T_γ × Γ."""
    return T_gamma / np.sqrt(max(1.0 - v_sq, 1e-30))


def tilt_dipole_parameter(v_sq):
    """T_d = (2/3)|v| — dipole Teff parameter."""
    return (2.0 / 3.0) * np.sqrt(max(v_sq, 0.0))


def tilt_to_delta_neff(v_sq):
    """Leading-order ΔN_eff equivalent: ΔN_eff ≈ 1.26 × v²."""
    N_eff = 3.044
    g = 1.0 + 7.0 * N_eff / 43.0
    return (43.0 / 7.0) * g * v_sq / 6.0
