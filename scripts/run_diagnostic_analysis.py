#!/usr/bin/env python3
"""
Tangency diagnostic D≥2 computation for Bianchi Type I BBN.

Computes the Gorban-Karlin tangency diagnostic from the Teff paper
(Park, Cheoun, Park 2026) at each timestep during BBN evolution,
measuring when the per-ray equilibrium ansatz f = Φ(y/Θ) breaks down.

The diagnostic is:
  D≥2 = ||G_ξ - (a + by)||_*
where G_ξ = C[f] / [f(1+ξf)] is the entropy-normalized collision field,
and the norm is in the generalized Laguerre inner product.

Physics: In Bianchi I, the neutrino distribution evolves via
  L[f] = C[f]
where L includes the shear-driven angular redistribution and C includes
ν-e scattering (the dominant incomplete-decoupling process).
"""
import sys, os, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, roots_laguerre
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_genlaguerre

from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1
from rabbit.thermo.eos_photon_electron import _RHO_GAMMA_PREFACTOR
from rabbit.geometry.typeI import compute_typeI_geometry_rhs

OUT_DIR = Path(__file__).parent.parent / "diagnostic_outputs"
OUT_DIR.mkdir(exist_ok=True)

# Constants
TAU_N = 879.6; ETA = 6.104e-10; N_EFF = 3.044; F_NU = 0.40520
MEV_TO_S = 1.0 / 6.58212e-22; G_N = 6.70883e-45
G_F = 1.1663787e-5  # Fermi constant in GeV^-2
ME = 0.511  # electron mass in MeV
N_MU = 12  # angular rays

# ═══════════════════════════════════════════════════════════════
# §1. Laguerre basis and diagnostic extraction
# ═══════════════════════════════════════════════════════════════

def laguerre_minus1(n, y):
    """Generalized Laguerre L^{(-1)}_n(y)."""
    return eval_genlaguerre(n, -1, y)

def compute_diagnostic_D2(G_y, y_grid, y_weights):
    """Compute D≥2 by weighted affine regression.

    G_y: entropy-normalized collision field at fixed direction, shape (N_y,)
    y_grid: energy grid points
    y_weights: quadrature weights w(y) = (1/y) exp(-y)

    Returns d≥2 (scalar) and c_n coefficients.
    """
    # Weighted least squares: minimize ||G - a - b*y||^2_*
    W = y_weights  # w_i = (1/y_i) exp(-y_i) * dy_i
    A_mat = np.column_stack([np.ones_like(y_grid), y_grid])  # [1, y]
    WA = A_mat * W[:, None]
    coeffs = np.linalg.lstsq(WA.T @ A_mat, WA.T @ G_y, rcond=None)[0]
    residual = G_y - coeffs[0] - coeffs[1] * y_grid
    d2_sq = np.sum(W * residual**2)

    # Extract individual Laguerre coefficients c_n for n=2,3,4
    c_n = {}
    for n in [2, 3, 4]:
        Ln = laguerre_minus1(n, y_grid)
        c_n[n] = n * np.sum(W * residual * Ln)

    return np.sqrt(max(d2_sq, 0)), c_n, coeffs


# ═══════════════════════════════════════════════════════════════
# §2. Collision operators in Bianchi I
# ═══════════════════════════════════════════════════════════════

def nu_e_scattering_rate(T_MeV, y):
    """ν-e scattering rate Γ_νe(E) ≈ G_F^2 T^4 E (for ν_e).

    More precisely: Γ ≈ (G_F^2 / π) (g_L^2 + g_R^2) T^4 E
    where g_L = 1/2 + sin^2θ_W ≈ 0.73, g_R = sin^2θ_W ≈ 0.23
    """
    g_L = 0.5 + 0.2312  # for ν_e
    g_R = 0.2312
    E = y * T_MeV  # energy in MeV
    # Rate in MeV: Γ = (G_F^2/π) * (g_L^2 + g_R^2) * T^4 * E
    # G_F in GeV^-2 = 1e-10 MeV^-2
    GF_MeV = G_F * 1e-10  # MeV^-2
    rate = (GF_MeV**2 / np.pi) * (g_L**2 + g_R**2) * T_MeV**4 * E
    return rate

def hubble_rate(T_MeV):
    """Hubble rate H(T) in MeV."""
    rho = _RHO_GAMMA_PREFACTOR * T_MeV**4 * (1 + N_EFF * 7/8)
    return np.sqrt(max(8*np.pi*G_N/3 * rho, 0))

def collision_field_nue_scattering(f_theta, Theta, y_grid, mu_dir, T_MeV, T_nu):
    """Approximate ν-e scattering collision field for a Teff state.

    C[f] ≈ -Γ(E) * (f - f_eq) for isoenergetic approximation
    More precisely for the Fokker-Planck limit:
    C[f] = Γ(y) * [<f>_Ω - f]  (angular relaxation at fixed energy)

    For a Teff state f = Φ(y/Θ(μ)), the isotropic average is
    <f>_Ω = (1/2) ∫ Φ(y/Θ(μ')) dμ'
    """
    f_iso = np.mean(f_theta)  # isotropic average (simple approx)
    C = nu_e_scattering_rate(T_nu, y_grid) * (f_iso - f_theta)
    return C

def entropy_normalize(C, f, xi=-1):
    """Compute G_ξ = C / [f(1 + ξf)]."""
    denom = f * (1 + xi * f)
    denom = np.maximum(denom, 1e-30)
    return C / denom


# ═══════════════════════════════════════════════════════════════
# §3. BBN evolution with diagnostic tracking
# ═══════════════════════════════════════════════════════════════

def run_diagnostic_evolution(Sigma_H, T_start=10.0, T_end=0.1, N_steps=200):
    """Evolve Bianchi I and compute D≥2 at each step."""

    mu0, w0 = leggauss(N_MU)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)

    # Energy grid for diagnostic (Laguerre-matched)
    N_y = 40
    y_grid = np.linspace(0.1, 15.0, N_y)
    y_weights = (1.0 / y_grid) * np.exp(-y_grid) * (y_grid[1] - y_grid[0])

    # Temperature grid (log-spaced)
    T_vals = np.logspace(np.log10(T_start), np.log10(T_end), N_steps)

    # Shear evolution: Σ(N) = Σ_H * exp(-N/2) * cos(ω*N)
    omega = 1.023

    results = {
        'Sigma_H': Sigma_H,
        'T': [], 'N_efold': [],
        'Sigma': [], 'D2_max': [], 'D2_mean': [],
        'Gamma_over_H': [], 'c2_max': [], 'c3_max': [],
        'T_dec': None,  # decoupling temperature
    }

    N0 = 0  # e-fold at T_start

    for i, T in enumerate(T_vals):
        N_e = np.log(T_start / T)  # e-folds since start

        # Shear at this epoch
        Sigma = Sigma_H * np.exp(-N_e / 2) * np.cos(omega * N_e)

        # Neutrino temperature
        T_nu = T_nu_from_T_gamma_tier1(T)

        # Hubble rate
        H = hubble_rate(T) * MEV_TO_S  # in s^-1... actually in MeV
        H_mev = hubble_rate(T)

        # Accumulated shear integral for each ray
        # Simple estimate: I_j ≈ Sigma * P2(μ_j) * ΔN (cumulative)
        S_accum = Sigma_H * 2 * (1 - np.exp(-N_e/2))  # ∫Σ dN approx

        # Ray directions (evolved)
        X_evolved = X0 * np.exp(6 * S_accum)
        mu_j = np.sign(mu0) * np.sqrt(np.minimum(X_evolved / (1 + X_evolved), 1-1e-15))
        P2_j = eval_legendre(2, mu_j)

        # Energy shift integrals per ray
        I_j = Sigma * P2_j * N_e * 0.5  # rough cumulative estimate

        # Effective temperature per ray: Θ_j ≈ exp(-2I_j)
        Theta_j = np.exp(-2 * I_j)
        Theta_j = np.maximum(Theta_j, 0.01)

        # Compute D≥2 at several representative directions
        D2_per_dir = []
        c2_per_dir = []
        c3_per_dir = []

        for j_dir in range(N_MU):
            Theta_dir = Theta_j[j_dir]

            # Distribution at this direction
            f_dir = 1.0 / (np.exp(y_grid / Theta_dir) + 1)  # FD

            # Isotropic average (angle-averaged over all rays)
            f_iso = np.zeros(N_y)
            for k in range(N_MU):
                f_iso += w0[k] * 1.0 / (np.exp(y_grid / Theta_j[k]) + 1)
            f_iso *= 0.5

            # Collision field: C ≈ Γ(y) * (f_iso - f_dir)
            Gamma_y = nu_e_scattering_rate(T_nu, y_grid)
            C_dir = Gamma_y * (f_iso - f_dir)

            # Entropy-normalized field
            G_xi = entropy_normalize(C_dir, f_dir, xi=-1)

            # Replace NaN/Inf
            G_xi = np.nan_to_num(G_xi, nan=0, posinf=0, neginf=0)

            # Compute diagnostic
            d2, c_n, _ = compute_diagnostic_D2(G_xi, y_grid, y_weights)
            D2_per_dir.append(d2)
            c2_per_dir.append(abs(c_n.get(2, 0)))
            c3_per_dir.append(abs(c_n.get(3, 0)))

        D2_max = max(D2_per_dir)
        D2_mean = np.mean(D2_per_dir)

        # Scattering rate / Hubble (at thermal energy y~3)
        Gamma_thermal = nu_e_scattering_rate(T_nu, 3.0)
        ratio = Gamma_thermal / max(H_mev, 1e-30)

        # Decoupling: Γ/H = 1
        if results['T_dec'] is None and ratio < 1.0:
            results['T_dec'] = float(T)

        results['T'].append(float(T))
        results['N_efold'].append(float(N_e))
        results['Sigma'].append(float(Sigma))
        results['D2_max'].append(float(D2_max))
        results['D2_mean'].append(float(D2_mean))
        results['Gamma_over_H'].append(float(ratio))
        results['c2_max'].append(float(max(c2_per_dir)))
        results['c3_max'].append(float(max(c3_per_dir)))

    return results


def main():
    print("="*60)
    print("  Tangency diagnostic D≥2 in Bianchi Type I")
    print("="*60)

    sigma_vals = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
    all_results = {}

    for Sig in sigma_vals:
        print(f"\n  Σ_H = {Sig}...", end=' ', flush=True)
        t0 = time.time()
        res = run_diagnostic_evolution(Sig)
        print(f"T_dec = {res['T_dec']:.3f} MeV  ({time.time()-t0:.1f}s)")
        all_results[f'Sigma_{Sig}'] = res

    # Save
    with open(OUT_DIR / 'diagnostic_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print decoupling temperatures
    print(f"\n{'─'*60}")
    print(f"  {'Σ_H':>6s}  {'T_dec [MeV]':>12s}  {'D2_max at T=2MeV':>16s}")
    print(f"{'─'*60}")
    for Sig in sigma_vals:
        res = all_results[f'Sigma_{Sig}']
        # Find D2 near T=2 MeV
        idx_2 = min(range(len(res['T'])), key=lambda i: abs(res['T'][i] - 2.0))
        d2_at_2 = res['D2_max'][idx_2]
        print(f"  {Sig:6.2f}  {res['T_dec']:12.3f}  {d2_at_2:16.4e}")

    return all_results


if __name__ == "__main__":
    results = main()
