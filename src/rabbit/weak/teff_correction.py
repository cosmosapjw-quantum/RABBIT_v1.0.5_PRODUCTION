"""
rabbit.weak.teff_correction — Teff spectral hardening / closure reconstruction for weak rates.

Implements the second-order spectral hardening effect identified in
Paper I (Section III.B): when the neutrino distribution has angular
anisotropy parameterized by a direction-dependent effective temperature
Θ(e), the angle-averaged spectrum is NOT the isotropic equilibrium f₀(q).

The correction arises from Jensen's inequality applied to the nonlinear
composition f_FD(q/Θ(e)). For a quadrupole deformation Θ(μ) = 1 + T₂ P₂(μ),
the angle-averaged distribution has a spectral hardening correction:

    ⟨f(q,μ)⟩ - f₀(q) = ½ Σ₂ × f₀(1-f₀) × [q²(1-2f₀) - 2q]

where Σ₂ = T₂²/5 is the angular variance of the Teff deformation.

Physical origin (Paper I, Eq. 23 and surrounding discussion):
    The y=2 crossover separates soft-particle deficit (y<2) from
    hard-particle excess (y>2). The angle-averaged spectrum is spectrally
    harder than the isotropic equilibrium — structurally analogous to
    the thermal Sunyaev–Zel'dovich y-distortion, with the anisotropy
    variance Σ₂ playing the role of the Comptonization parameter.

Connection to BBN:
    In Type I Bianchi cosmology, shear drives the neutrino transport
    quadrupole Ψ₂, creating an angular anisotropy. The reduced
    quadrupole π̃ from the transport projectors determines the Teff
    angular deformation T₂ = π̃/4 (from the ρ ∝ Θ⁴ matching
    of Paper I, Section III.B). The spectral hardening modifies
    the effective neutrino monopole entering the weak n↔p rates.

    For Σ_H = 0.01 and a few radiation-dominated e-folds:
        π̃ ≈ Ψ₂ ≈ 0.05, T₂ ≈ 0.012, Σ₂ ≈ 3×10⁻⁵
        δλ/λ ≈ Σ₂ × O(1) ≈ few × 10⁻⁵

    This is the SECOND physics channel of Bianchi-BBN coupling:
        Channel 1: Σ → H → yields  (geometry, already implemented)
        Channel 2: Σ → Ψ₂ → π̃ → Teff hardening → δf₀ → δλ → yields

References:
    Paper I: Sections III.B (spectral hardening), V (tangency diagnostic),
             VI.C (entropy-projection inequality)
"""
from __future__ import annotations
from functools import lru_cache

import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss


def _warn_fd_bounds(label: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo < -1e-12 or hi > 1.0 + 1e-12:
        warnings.warn(
            f"{label}: distribution left the physical FD range [0,1] (min={lo:.6e}, max={hi:.6e}). Result is diagnostic-only unless vetoed.",
            RuntimeWarning,
            stacklevel=2,
        )


# ═══════════════════════════════════════════════════════════════════════
# §1. Teff ↔ transport mapping


@lru_cache(maxsize=8)
def _leggauss_cached(N_mu: int):
    """Cache Gauss-Legendre nodes/weights for repeated exact Teff calls."""
    return leggauss(N_mu)


def _interp_logit_residual_batch(q_nodes: np.ndarray, delta_q: np.ndarray, q_eval: np.ndarray) -> np.ndarray:
    """Vectorized piecewise-linear interpolation of the logit residual.

    Parameters
    ----------
    q_nodes : ndarray, shape (N_q,)
        Monotone Laguerre nodes.
    delta_q : ndarray, shape (N_q,)
        Logit residual values at the nodes.
    q_eval : ndarray, shape (..., N_q)
        Evaluation points.
    """
    idx = np.searchsorted(q_nodes, q_eval, side='right') - 1
    idx = np.clip(idx, 0, q_nodes.size - 2)

    x0 = q_nodes[idx]
    x1 = q_nodes[idx + 1]
    h = np.maximum(x1 - x0, 1e-30)
    t = (q_eval - x0) / h

    delta_eval = (1.0 - t) * delta_q[idx] + t * delta_q[idx + 1]
    delta_eval = np.where(q_eval <= q_nodes[0], delta_q[0], delta_eval)
    delta_eval = np.where(q_eval >= q_nodes[-1], 0.0, delta_eval)
    return delta_eval

# ═══════════════════════════════════════════════════════════════════════

def pi_tilde_to_teff_quadrupole(pi_tilde: float) -> float:
    """Convert transport reduced quadrupole π̃ to Teff deformation T₂.

    The mapping uses energy-density matching (Paper I, Sec. III.B):
        ρ(μ) ∝ Θ(μ)⁴ ≈ 1 + 4T₂P₂(μ) + ...

    In the PSTF transport:
        ρ(μ) ∝ 1 + π̃ P₂(μ)

    Matching at linear order: 4T₂ = π̃  →  T₂ = π̃/4.
    """
    return pi_tilde / 4.0


def teff_angular_variance(T2: float) -> float:
    """Angular variance Σ₂ = ⟨θ²⟩ of the Teff deformation.

    For θ(μ) = T₂ P₂(μ) with ⟨θ⟩ = 0 (gauge condition):
        Σ₂ = T₂² × ⟨P₂²⟩ = T₂² × 1/5
    """
    return T2**2 / 5.0


def teff_quadrupole_theta_bounds(pi_tilde: float) -> tuple[float, float]:
    """Return exact min/max of Θ(μ)=1+(π̃/4)P₂(μ).

    For P₂(μ)=½(3μ²-1), the extrema are reached at μ=0 and |μ|=1.
    With T₂ = π̃/4:

        Θ(μ=0) = 1 - T₂/2
        Θ(|μ|=1) = 1 + T₂

    The minimum is therefore piecewise in the sign of T₂. This is the
    exact realizability certificate for the quadrupole-only Teff ansatz.
    """
    T2 = pi_tilde_to_teff_quadrupole(pi_tilde)
    theta_axis = 1.0 + T2
    theta_equator = 1.0 - 0.5 * T2
    theta_min = min(theta_axis, theta_equator)
    theta_max = max(theta_axis, theta_equator)
    return float(theta_min), float(theta_max)


def fd_bounds_summary(arr: np.ndarray) -> dict[str, float | bool]:
    """Summarize FD admissibility of a sampled distribution array."""
    arr = np.asarray(arr, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return {
        'min': lo,
        'max': hi,
        'is_fd_admissible': bool(lo >= -1e-12 and hi <= 1.0 + 1e-12),
    }


# ═══════════════════════════════════════════════════════════════════════
# §2. Perturbative spectral hardening (Paper I, Sec. III.B)
# ═══════════════════════════════════════════════════════════════════════

def spectral_hardening_fd(q: np.ndarray, Sigma2: float) -> np.ndarray:
    """Second-order Teff spectral hardening correction for Fermi-Dirac.

    δf(q) = ½ Σ₂ × f₀(1-f₀) × [q²(1-2f₀) - 2q]

    where f₀(q) = 1/(e^q + 1) is the equilibrium FD distribution.

    Derived from the angle average of f_FD(q/Θ(μ)) expanded to O(θ²)
    (Paper I, Section III.B, adapted from the MB generating function
    Eq. 23 to FD statistics).

    The correction changes sign at q ≈ 2 (soft deficit / hard excess).
    """
    if abs(Sigma2) < 1e-30:
        return np.zeros_like(q)

    f0 = 1.0 / (np.exp(np.clip(q, -500, 500)) + 1.0)
    f0_1mf0 = f0 * (1.0 - f0)  # = e^q / (e^q + 1)²
    one_minus_2f0 = 1.0 - 2.0 * f0  # = (e^q - 1)/(e^q + 1) = tanh(q/2)

    bracket = q**2 * one_minus_2f0 - 2.0 * q
    return 0.5 * Sigma2 * f0_1mf0 * bracket


def spectral_hardening_mb(q: np.ndarray, Sigma2: float) -> np.ndarray:
    """Second-order Teff spectral hardening for Maxwell-Boltzmann.

    δf(q) = ½ Σ₂ × f₀ × q(q - 2)

    From Paper I, Eq. 23: f̄₍₂₎ = e⁻ʸ[1 + ½Σ₂ y(y-2)].
    """
    if abs(Sigma2) < 1e-30:
        return np.zeros_like(q)

    f0 = np.exp(-np.clip(q, -500, 500))
    return 0.5 * Sigma2 * f0 * q * (q - 2.0)


# ═══════════════════════════════════════════════════════════════════════
# §3. Exact angular quadrature (nonperturbative)
# ═══════════════════════════════════════════════════════════════════════

def teff_corrected_monopole_exact(
    f0: np.ndarray,
    q_nodes: np.ndarray,
    pi_tilde: float,
    N_mu: int = 16,
) -> np.ndarray:
    """Compute the Teff-closure reconstructed monopole by angular quadrature.

    This routine is *not* the exact Boltzmann-evolved correction for an
    arbitrary nonthermal input monopole. Instead it assumes the Teff closure
    ansatz

        f(q, μ) = f_iso(q / Θ(μ)),

    where ``f_iso`` is reconstructed from the supplied isotropic monopole
    samples ``f0(q_nodes)`` and

        Θ(μ) = 1 + (π̃/4) P₂(μ).

    The returned monopole is therefore the angular average of the Teff closure
    built from the supplied baseline shape. For equilibrium FD input this
    reduces to the original FD-based construction.

    Returns ``f0`` unchanged when ``pi_tilde = 0``. If the Teff deformation
    violates realizability (Θ ≤ 0 for some μ), we fall back to the small-
    anisotropy perturbative spectral-hardening correction around the supplied
    baseline.
    """
    if abs(pi_tilde) < 1e-15:
        return f0.copy()

    q_nodes = np.asarray(q_nodes, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    T2 = pi_tilde_to_teff_quadrupole(pi_tilde)

    # Gauss-Legendre quadrature on [-1, 1] (cached because exact Teff calls
    # happen at every RHS evaluation in the SciPy reference path).
    mu_nodes, mu_weights = _leggauss_cached(int(N_mu))

    # Reconstruct the isotropic baseline through the logit residual
    #     δ(q) = logit(f(q)) + q,
    # so exact FD input has δ=0 for any N_q. This mirrors the stabilized
    # live-weak path and removes the remaining low-N_q interpolation artifacts
    # from the Teff closure itself.
    eps_clip = 1e-14
    f_safe = np.clip(f0, eps_clip, 1.0 - eps_clip)
    delta_q = np.log(f_safe / (1.0 - f_safe)) + q_nodes

    P2_mu = 0.5 * (3.0 * mu_nodes**2 - 1.0)
    Theta = 1.0 + T2 * P2_mu

    if np.any(Theta <= 0.0):
        Sigma2 = teff_angular_variance(T2)
        out = f0 + spectral_hardening_fd(q_nodes, Sigma2)
        _warn_fd_bounds("teff_corrected_monopole_exact[fallback]", out)
        return out

    q_scaled = q_nodes[None, :] / Theta[:, None]
    delta_eval = _interp_logit_residual_batch(q_nodes, delta_q, q_scaled)
    expo = np.clip(q_scaled - delta_eval, -500.0, 500.0)
    f_at_mu = 1.0 / (1.0 + np.exp(expo))
    f_corrected = 0.5 * np.sum(mu_weights[:, None] * f_at_mu, axis=0)

    _warn_fd_bounds("teff_corrected_monopole_exact", f_corrected)
    return f_corrected


# ═══════════════════════════════════════════════════════════════════════
# §4. Teff tangency diagnostic D≥2 (Paper I, Sec. V)
# ═══════════════════════════════════════════════════════════════════════

def teff_tangency_diagnostic(
    f_monopole: np.ndarray,
    q_nodes: np.ndarray,
    pi_tilde: float,
    q_weights: np.ndarray | None = None,
) -> float:
    """Compute the Teff tangency diagnostic Dres≥2 (Paper I, Theorem 7).

    Measures the n≥2 Laguerre energy-mode content of the state residual
    between the actual monopole f₀ and its best-fit Teff representative.

    For the collisionless case (f₀ = FD exactly), this measures how
    well the PSTF distribution matches the Teff family, which is a
    proxy for the magnitude of the spectral hardening correction.

    A small D_res indicates the Teff approximation is adequate;
    a large D_res signals the need for spectral-shape upgrades
    (Paper I, Algorithm 1 — diagnose-to-switch protocol).

    Returns
    -------
    float
        D_res≥2 (dimensionless). Typical values: 0 for exactly Teff
        states, O(Σ₂) for weakly anisotropic states.
    """
    if abs(pi_tilde) < 1e-15:
        return 0.0

    # Compute the Teff-closure reconstructed monopole
    f_teff = teff_corrected_monopole_exact(f_monopole, q_nodes, pi_tilde)

    # State residual: deviation of actual f from Teff representative
    # In the collisionless limit, f_monopole = f_FD(q), and
    # f_teff = ⟨f_FD(q/Θ)⟩ ≠ f_FD(q) due to spectral hardening.
    # The residual measures this mismatch.
    residual = f_monopole - f_teff

    # Entropy-normalized norm: weight by 1/[f₀(1-f₀)]
    f0 = np.clip(f_monopole, 1e-30, 1.0 - 1e-30)
    entropy_weight = 1.0 / (f0 * (1.0 - f0))

    # Use the actual standard-Laguerre quadrature rule.  Raw weights w_i
    # approximate ∫ e^{-q} g(q) dq, so for ∫ h(q)/q dq we need
    #     Σ_i w_i e^{q_i} h(q_i) / q_i.
    q = np.clip(q_nodes, 1e-30, None)
    if q_weights is None:
        from numpy.polynomial.laguerre import laggauss
        _, q_weights = laggauss(len(q_nodes))
    q_weights = np.asarray(q_weights, dtype=float)
    w_diag = q_weights * np.exp(q) / q

    # Remove the affine (n=0, n=1) part by regression
    # (Paper I, Proposition 5: regression-based extraction)
    G_norm = residual * entropy_weight  # entropy-normalized residual

    # Weighted regression: fit G_norm = a + b*q in ⟨·,·⟩* norm
    # Design matrix: [1, q]
    W = w_diag
    S0 = np.sum(W)
    S1 = np.sum(W * q)
    Sq = np.sum(W * q**2)
    Sg = np.sum(W * G_norm)
    Sgq = np.sum(W * q * G_norm)

    denom = S0 * Sq - S1**2
    if abs(denom) < 1e-50:
        return 0.0

    a_fit = (Sq * Sg - S1 * Sgq) / denom
    b_fit = (S0 * Sgq - S1 * Sg) / denom

    # Residual after removing affine part
    resid_n2plus = G_norm - a_fit - b_fit * q

    # Diagnostic: L² norm in Laguerre inner product
    D_sq = np.sum(W * resid_n2plus**2)
    return np.sqrt(max(D_sq, 0.0))


# ═══════════════════════════════════════════════════════════════════════
# §5. Summary diagnostic for the canonical driver
# ═══════════════════════════════════════════════════════════════════════

def compute_teff_weak_correction(
    f_nue_monopole: np.ndarray,
    f_nuebar_monopole: np.ndarray,
    q_nodes: np.ndarray,
    pi_tilde_nue: float,
    pi_tilde_nuebar: float,
    method: str = 'exact',
    N_mu: int = 16,
):
    """Compute Teff-corrected monopoles for both ν_e and ν̄_e.

    This is the top-level function called by the canonical driver
    to wire the transport quadrupole into the weak rate calculation.

    Parameters
    ----------
    f_nue_monopole, f_nuebar_monopole : ndarray
        Monopole distributions from transport projectors.
    q_nodes : ndarray
        Momentum grid nodes.
    pi_tilde_nue, pi_tilde_nuebar : float
        Reduced quadrupoles from transport (species 0, 1).
    method : str
        'exact' for angular quadrature, 'perturbative' for O(Σ₂).
    N_mu : int
        Angular quadrature points (for 'exact' method).

    Returns
    -------
    tuple of (ndarray, ndarray, dict)
        (f_nue_corrected, f_nuebar_corrected, diagnostics)
    """
    if method == 'exact':
        f_nue_corr = teff_corrected_monopole_exact(
            f_nue_monopole, q_nodes, pi_tilde_nue, N_mu)
        f_nuebar_corr = teff_corrected_monopole_exact(
            f_nuebar_monopole, q_nodes, pi_tilde_nuebar, N_mu)
    elif method == 'perturbative':
        T2_nue = pi_tilde_to_teff_quadrupole(pi_tilde_nue)
        T2_nuebar = pi_tilde_to_teff_quadrupole(pi_tilde_nuebar)
        Sigma2_nue = teff_angular_variance(T2_nue)
        Sigma2_nuebar = teff_angular_variance(T2_nuebar)
        f_nue_corr = f_nue_monopole + spectral_hardening_fd(
            q_nodes, Sigma2_nue)
        f_nuebar_corr = f_nuebar_monopole + spectral_hardening_fd(
            q_nodes, Sigma2_nuebar)
    else:
        raise ValueError(f"Unknown method: {method}")

    _warn_fd_bounds("compute_teff_weak_correction[nu_e]", f_nue_corr)
    _warn_fd_bounds("compute_teff_weak_correction[nu_ebar]", f_nuebar_corr)

    # Diagnostics
    Sigma2_nue = teff_angular_variance(pi_tilde_to_teff_quadrupole(pi_tilde_nue))
    Sigma2_nuebar = teff_angular_variance(pi_tilde_to_teff_quadrupole(pi_tilde_nuebar))
    delta_nue = np.max(np.abs(f_nue_corr - f_nue_monopole))
    delta_nuebar = np.max(np.abs(f_nuebar_corr - f_nuebar_monopole))

    diag = {
        'Sigma2_nue': Sigma2_nue,
        'Sigma2_nuebar': Sigma2_nuebar,
        'max_delta_f_nue': delta_nue,
        'max_delta_f_nuebar': delta_nuebar,
        'pi_tilde_nue': pi_tilde_nue,
        'pi_tilde_nuebar': pi_tilde_nuebar,
    }

    return f_nue_corr, f_nuebar_corr, diag
