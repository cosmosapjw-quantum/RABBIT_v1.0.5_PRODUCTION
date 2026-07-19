#!/usr/bin/env python3
"""
RABBIT Ablation Study: Teff (ℓ, n) decomposition.

Compares the Channel 2 (spectral distortion) signal across different
angular resolution (ℓ) and spectral expansion order (n) against the
fully nonperturbative characteristic ray monopole.

Grid:
  ℓ = 2    : linearised PSTF hierarchy (q-independent Ψ₂)
  ℓ = ∞    : characteristic rays (exact angular resolution)

  n = 0    : equilibrium FD monopole (no spectral correction)
  n = 2    : perturbative Teff spectral hardening (O(T₂²))
  n = full : exact numerical angle-average of f_FD(q/Θⱼ)

For each (ℓ, n) at each Σ_H:
  Signal = Yp(ℓ, n) - Yp(ℓ, n=0)     [Channel 2 contribution]
  Ref    = Yp(ℓ=∞, n=full) - Yp(ℓ=∞, n=0)  [exact Ch2]
  Loss   = 1 - Signal/Ref             [fraction of Ch2 lost]

Usage:
  python3 scripts/run_ablation_study.py
  python3 scripts/run_ablation_study.py --sigma 0.1 0.3 0.5
  python3 scripts/run_ablation_study.py --fast  # only Σ=0.1, 0.3
"""
import sys, os, time, json, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, roots_laguerre
from numpy.polynomial.legendre import leggauss

from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1
from rabbit.thermo.eos_photon_electron import _RHO_GAMMA_PREFACTOR
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.network.abundances_standard import (
    abundance_rhs_phase1, abundance_rhs_phase2, phase1_to_phase2, N_SPECIES)

# ── Constants ──
TAU_N = 879.6; ETA = 6.104e-10; N_EFF = 3.044; F_NU = 0.40520
MEV_TO_S = 1.0 / 6.58212e-22; G_N = 6.70883e-45
B_Q = 8.0 / 15.0; D_DAMP = 4.0

OUT_DIR = Path(__file__).parent.parent / "ablation_outputs"


def hubble(Tg, Tnu, S2):
    rho = _RHO_GAMMA_PREFACTOR * Tg**4 * (1 + N_EFF * 7/8 * (Tnu/Tg)**4)
    return np.sqrt(max(8*np.pi*G_N/3 * rho / max(1-S2, 1e-10), 0))


def setup_grids(N_mu=12):
    mu0, w0 = leggauss(N_mu)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)
    q_gl, q_wgl = roots_laguerre(20)
    f0_gl = 1.0 / (np.exp(np.minimum(q_gl, 500)) + 1)
    return mu0.astype(np.float64), w0.astype(np.float64), X0.astype(np.float64), \
           q_gl.astype(np.float64), q_wgl.astype(np.float64), f0_gl.astype(np.float64)


def mu_signed(mu0, X0, S):
    X = X0 * np.exp(6 * S)
    return np.sign(mu0) * np.sqrt(np.minimum(X / (1 + X), 1 - 1e-15))


# ═══════════════════════════════════════════════════════════════════════
# Monopole extraction variants
# ═══════════════════════════════════════════════════════════════════════

def monopole_exact(I, J, w0, q_gl):
    """(ℓ=∞, n=full): Exact angle-average of f_FD(q/Θ_j)."""
    alpha = np.exp(2 * I)
    qa = q_gl[:, None] * alpha[None, :]
    f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)
    return 0.5 * f_vals @ (w0 * J)


def monopole_equilibrium(q_gl):
    """(any ℓ, n=0): Pure equilibrium FD."""
    return 1.0 / (np.exp(np.minimum(q_gl, 500)) + 1)


def monopole_teff_perturbative(I, J, w0, q_gl, n_order=2):
    """(ℓ=∞, n=2): Perturbative Teff spectral hardening.

    Uses the O(T₂²) correction from Jensen's inequality:
      δf/f₀ = Σ₂ × f₀(1-f₀) × [q²(1-2f₀) - 2q]
    where Σ₂ = variance of Θ_j around 1.
    """
    f0 = monopole_equilibrium(q_gl)
    if n_order == 0:
        return f0.copy()

    # Compute angular variance of Θ from ray data
    Theta = np.exp(-2 * I)
    Theta_mean = 0.5 * np.sum(w0 * J * Theta)

    # T₂ from quadrupole projection
    mu_eff = np.sqrt(np.arange(len(I)) / max(len(I)-1, 1))  # approximate
    # Actually use the Θ variance directly
    Sigma2 = 0.5 * np.sum(w0 * J * (Theta/Theta_mean - 1)**2)

    if n_order >= 2 and abs(Sigma2) > 1e-30:
        # O(T₂²) spectral hardening
        q = q_gl
        correction = Sigma2 * f0 * (1-f0) * (q**2 * (1-2*f0) - 2*q)
        return np.clip(f0 + correction, 0, 1)
    return f0.copy()


def monopole_teff_exact_integral(pi_tilde, q_gl, N_mu_int=32):
    """(ℓ=2, n=full): Exact numerical Teff integral with Ψ₂-based deformation.

    Θ(μ) = 1 + T₂ P₂(μ), T₂ = π̃/4.
    f̃₀(q) = ½∫₋₁¹ f_FD(q/Θ(μ)) dμ
    """
    T2 = pi_tilde / 4.0
    if abs(T2) < 1e-15:
        return monopole_equilibrium(q_gl)

    mu_int, w_int = leggauss(N_mu_int)
    P2_int = eval_legendre(2, mu_int)
    Theta_int = 1.0 + T2 * P2_int  # (N_mu_int,)

    # Check for singularity
    if np.any(Theta_int <= 0):
        # Teff singularity: fall back to safe clipping
        Theta_int = np.maximum(Theta_int, 0.01)

    qa = q_gl[:, None] / Theta_int[None, :]  # (N_q, N_mu_int)
    f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)
    return 0.5 * f_vals @ w_int


def monopole_teff_perturbative_from_pi(pi_tilde, q_gl):
    """(ℓ=2, n=2): Perturbative Teff from linearised Ψ₂."""
    f0 = monopole_equilibrium(q_gl)
    T2 = pi_tilde / 4.0
    Sigma2 = T2**2 / 5.0
    if abs(Sigma2) < 1e-30:
        return f0.copy()
    q = q_gl
    correction = Sigma2 * f0 * (1-f0) * (q**2 * (1-2*f0) - 2*q)
    return np.clip(f0 + correction, 0, 1)


# ═══════════════════════════════════════════════════════════════════════
# BBN solver
# ═══════════════════════════════════════════════════════════════════════

def run_bbn(Sigma_H, mode='characteristic', monopole_mode='exact',
            N_mu=12, T_start=10.0, T_handoff=0.08, T_end=0.005,
            rtol=1e-8, atol=1e-10, CL=0):
    """Run a single BBN computation.

    Parameters
    ----------
    mode : 'characteristic' or 'linearised'
    monopole_mode : 'exact', 'equilibrium', 'teff_n2', 'teff_full'
    """
    mu0, w0, X0, q_gl, q_wgl, f0_gl = setup_grids(N_mu)

    is_char = (mode == 'characteristic')
    n_I = N_mu if is_char else 0
    n_J = N_mu if is_char else 0

    # Index layout
    I_SP = 0
    if is_char:
        I_I = 1; I_J = 1+n_I; I_S = 1+n_I+n_J; I_TG = I_S+1
    else:
        I_PSI = 1; I_TG = 2
    I_NET = I_TG + 1

    def rhs(N, y, phase):
        Sp = y[I_SP]
        Tg = y[I_TG]
        n_net = 2 if phase == 1 else N_SPECIES
        X = y[I_NET:I_NET+n_net]
        if Tg < 1e-6:
            return np.zeros(len(y))
        S2 = Sp**2

        if is_char:
            I_vals = y[I_I:I_I+n_I]
            J_vals = y[I_J:I_J+n_J]
            S_val = y[I_S]
            mu = mu_signed(mu0, X0, S_val)
            P2_mu = eval_legendre(2, mu)

            # Stress (always exact for characteristic)
            Pi = F_NU * np.sum(w0 * J_vals * P2_mu * np.exp(-8 * I_vals))

            # Transport RHS
            dI = Sp * P2_mu
            dJ = 3 * Sp * (1 - 3*mu**2) * J_vals
            dS = Sp

            # Monopole for weak rates (the ablation variable)
            if monopole_mode == 'exact':
                f_mono = monopole_exact(I_vals, J_vals, w0, q_gl)
            elif monopole_mode == 'equilibrium':
                f_mono = monopole_equilibrium(q_gl)
            elif monopole_mode == 'teff_n2':
                f_mono = monopole_teff_perturbative(I_vals, J_vals, w0, q_gl, n_order=2)
            else:
                f_mono = monopole_exact(I_vals, J_vals, w0, q_gl)
        else:
            Psi2 = y[I_PSI]
            Pi = 6 * F_NU * Psi2
            pi_tilde = Psi2  # in linearised regime

            if monopole_mode == 'equilibrium':
                f_mono = monopole_equilibrium(q_gl)
            elif monopole_mode == 'teff_n2':
                f_mono = monopole_teff_perturbative_from_pi(pi_tilde, q_gl)
            elif monopole_mode == 'teff_full':
                f_mono = monopole_teff_exact_integral(pi_tilde, q_gl)
            else:
                f_mono = monopole_equilibrium(q_gl)

        f_mono = np.clip(f_mono, 0, 1)

        # Geometry
        Om = max(0, 1-S2)
        dSp, _ = compute_typeI_geometry_rhs(Sp, 0., Pi, 0., Om)

        # Thermo + Hubble
        dTg = dT_gamma_dN_tier1(Tg)
        Tnu = T_nu_from_T_gamma_tier1(Tg)
        H = hubble(Tg, Tnu, S2) * MEV_TO_S

        # Weak rates
        weak = compute_live_weak_rates(f_mono, f_mono, q_gl, Tg, Tnu, TAU_N,
                                        compute_iso_reference=False, correction_level=CL)

        # Network
        if phase == 1:
            dX = np.zeros(n_net)
            dX[0] = abundance_rhs_phase1(X[0], weak.lambda_np, weak.lambda_pn)/max(H,1e-100)
            dX[1] = -dX[0]
        else:
            dX = abundance_rhs_phase2(X, Tg, ETA, weak.lambda_np, weak.lambda_pn,
                                       n_reactions=12)/max(H,1e-100)

        dy = np.zeros(len(y))
        dy[I_SP] = dSp
        if is_char:
            dy[I_I:I_I+n_I] = dI
            dy[I_J:I_J+n_J] = dJ
            dy[I_S] = dS
        else:
            dy[I_PSI] = -B_Q * Sp - D_DAMP * Psi2
        dy[I_TG] = dTg
        dy[I_NET:I_NET+n_net] = dX
        return dy

    # Initial conditions
    Tnu0 = T_nu_from_T_gamma_tier1(T_start)
    w0_wr = compute_live_weak_rates(f0_gl, f0_gl, q_gl, T_start, Tnu0, TAU_N,
                                     compute_iso_reference=False, correction_level=CL)
    Xn_eq = float(w0_wr.lambda_pn / max(w0_wr.lambda_np + w0_wr.lambda_pn, 1e-100))

    n_total_p1 = I_NET + 2
    y0 = np.zeros(n_total_p1)
    y0[I_SP] = Sigma_H
    if is_char:
        y0[I_J:I_J+n_J] = 1.0
    y0[I_TG] = T_start
    y0[I_NET] = Xn_eq; y0[I_NET+1] = 1-Xn_eq

    N_start = np.log(T_start / T_handoff)  # N increases as T decreases

    # Phase 1: weak freeze-out
    sol1 = solve_ivp(lambda N, y: rhs(N, y, 1), [0, N_start], y0,
                     method='Radau', rtol=rtol, atol=atol, max_step=0.5)
    if not sol1.success:
        return None

    # Phase 2: nuclear burning
    y1_end = sol1.y[:, -1]
    Tg_ho = y1_end[I_TG]
    Xn_ho = y1_end[I_NET]
    X_nse = phase1_to_phase2(Xn_ho)

    n_total_p2 = I_NET + N_SPECIES
    y2_0 = np.zeros(n_total_p2)
    y2_0[:I_NET] = y1_end[:I_NET]
    y2_0[I_NET:I_NET+N_SPECIES] = X_nse

    N_end = N_start + np.log(T_handoff / T_end)
    sol2 = solve_ivp(lambda N, y: rhs(N, y, 2), [N_start, N_end], y2_0,
                     method='Radau', rtol=rtol, atol=atol, max_step=0.5)
    if not sol2.success:
        return None

    X_final = sol2.y[I_NET:I_NET+N_SPECIES, -1]
    Yp = float(X_final[5])  # He-4 mass fraction (already Yp in PRIMAT convention)
    return Yp


# ═══════════════════════════════════════════════════════════════════════
# Main ablation study
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Teff (ℓ, n) ablation study")
    parser.add_argument('--sigma', nargs='+', type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument('--fast', action='store_true', help='Quick run: Σ=0.1, 0.3 only')
    parser.add_argument('--cl', type=int, default=0, help='Correction level (0=Born)')
    args = parser.parse_args()

    if args.fast:
        sigma_values = [0.1, 0.3]
    else:
        sigma_values = args.sigma

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  RABBIT Ablation Study: Teff (ℓ, n) decomposition")
    print(f"  Σ_H values: {sigma_values}")
    print(f"  CL = {args.cl}")
    print("=" * 70)

    # Define the (ℓ, n) configurations
    configs = [
        # (label, mode, monopole_mode, ℓ_label, n_label)
        ('ℓ=∞, n=full', 'characteristic', 'exact',        '∞', 'full'),
        ('ℓ=∞, n=2',    'characteristic', 'teff_n2',      '∞', '2'),
        ('ℓ=∞, n=0',    'characteristic', 'equilibrium',  '∞', '0'),
        ('ℓ=2, n=full',  'linearised',    'teff_full',    '2', 'full'),
        ('ℓ=2, n=2',     'linearised',    'teff_n2',      '2', '2'),
        ('ℓ=2, n=0',     'linearised',    'equilibrium',  '2', '0'),
    ]

    # FLRW reference
    print("\n[FLRW] Running Σ=0 reference...", end=' ', flush=True)
    t0 = time.time()
    Yp_flrw = run_bbn(0.0, mode='characteristic', monopole_mode='exact', CL=args.cl)
    print(f"Yp = {Yp_flrw:.8f}  ({time.time()-t0:.1f}s)")

    results = {'flrw': Yp_flrw, 'sigma_values': sigma_values, 'CL': args.cl}

    for Sig in sigma_values:
        print(f"\n{'─'*60}")
        print(f"  Σ_H = {Sig}")
        print(f"{'─'*60}")

        sig_results = {}
        for label, mode, mono_mode, ell_l, n_l in configs:
            print(f"  [{label:15s}] ", end='', flush=True)
            t0 = time.time()
            Yp = run_bbn(Sig, mode=mode, monopole_mode=mono_mode, CL=args.cl)
            elapsed = time.time() - t0
            if Yp is None:
                print("FAILED")
                sig_results[label] = None
                continue
            dYp = Yp - Yp_flrw
            print(f"Yp = {Yp:.8f}  ΔYp = {dYp:+.6e}  ({elapsed:.1f}s)")
            sig_results[label] = {'Yp': Yp, 'dYp': dYp}

        # Compute Channel 2 signal and loss
        ref_full = sig_results.get('ℓ=∞, n=full', {})
        ref_eq = sig_results.get('ℓ=∞, n=0', {})
        if ref_full and ref_eq and ref_full.get('Yp') and ref_eq.get('Yp'):
            ch2_ref = ref_full['Yp'] - ref_eq['Yp']
            print(f"\n  Channel 2 reference (ℓ=∞): {ch2_ref:+.6e}")
            print(f"  Total ΔYp (ℓ=∞, n=full):   {ref_full['dYp']:+.6e}")

            print(f"\n  {'Config':20s} {'Signal':>12s} {'Loss':>8s}")
            print(f"  {'─'*44}")
            for label, _, _, ell_l, n_l in configs:
                r = sig_results.get(label)
                if r and r.get('Yp') and ref_eq.get('Yp'):
                    signal = r['Yp'] - ref_eq['Yp']
                    if abs(ch2_ref) > 1e-15:
                        loss = (1 - signal/ch2_ref) * 100
                    else:
                        loss = 0.0
                    print(f"  {label:20s} {signal:+12.6e} {loss:7.1f}%")
                    r['ch2_signal'] = signal
                    r['ch2_loss_pct'] = loss

        results[f'Sigma_{Sig}'] = sig_results

    # Summary table
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY: Channel 2 signal loss [%]")
    print(f"{'='*70}")
    print(f"  {'Σ_H':>6s}", end='')
    for label, _, _, _, _ in configs:
        short = label.replace('ℓ=', '').replace(', n=', '/n')
        print(f"  {short:>10s}", end='')
    print()

    for Sig in sigma_values:
        key = f'Sigma_{Sig}'
        if key not in results:
            continue
        sr = results[key]
        print(f"  {Sig:6.2f}", end='')
        for label, _, _, _, _ in configs:
            r = sr.get(label, {})
            loss = r.get('ch2_loss_pct', float('nan'))
            print(f"  {loss:9.1f}%", end='')
        print()

    # Save
    out_path = OUT_DIR / "ablation_results.json"
    # Convert to serializable
    ser = {}
    for k, v in results.items():
        if isinstance(v, dict):
            ser[k] = {kk: {kkk: float(vvv) if isinstance(vvv, (float, np.floating)) else vvv
                          for kkk, vvv in vv.items()} if isinstance(vv, dict) else v
                     for kk, vv in v.items()}
        else:
            ser[k] = float(v) if isinstance(v, (float, np.floating)) else v

    with open(out_path, 'w') as f:
        json.dump(ser, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
