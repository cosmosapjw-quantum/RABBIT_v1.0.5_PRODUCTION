#!/usr/bin/env python3
"""
RABBIT Ablation Study: Teff (ℓ, n) truncation vs exact ray-based transport.

Performs a genuine numerical computation comparing the Channel-2 (spectral
distortion) signal at different angular (ℓ) and spectral (n) truncation
levels of the Teff expansion against the fully nonperturbative characteristic
ray solution.

Physics:
    Exact monopole (ray-based, no Teff):
        f̃₀(q) = (1/2) Σⱼ wⱼ Jⱼ f_FD(q/Θⱼ),  Θⱼ = exp(-2Iⱼ)

    Teff ℓ-truncation:
        Θ(μ) = Θ̄ [1 + c₂P₂(μ) + c₄P₄(μ) + ...]
        ℓ=0: Θ = Θ̄ (monopole only)
        ℓ=2: Θ(μ) = Θ̄(1 + c₂P₂)
        ℓ=4: Θ(μ) = Θ̄(1 + c₂P₂ + c₄P₄)
        ℓ=∞: use all Θⱼ directly (exact angular quadrature)

    Teff n-truncation (spectral expansion):
        n=0: f̃₀(q) = f_FD(q) (no spectral correction)
        n=1: first-order ∝ c₂ (vanishes by symmetry for even P₂)
        n=2: second-order ∝ c₂² (Jensen spectral hardening)
        n=full: exact numerical ½∫dμ f_FD(q/Θ(μ))

    Signal metric:
        ΔΛ = |Λ[f̃₀] - Λ[f_FD]|  (weak rate shift)
        where Λ[f] = ∫ dε_e ε_e p_e ε_ν² f(ε_ν) [1-f_e(ε_e)]

Usage:
    python3 scripts/run_teff_ablation_study.py [--sigma 0.1 0.3 0.5] [--outdir figures/]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
import os, sys, argparse

# ═══════════════════════════════════════════════════════════════════
# §1. Characteristic ray transport (simplified self-consistent)
# ═══════════════════════════════════════════════════════════════════

def fermi_dirac(x):
    """f_FD(x) = 1/(e^x + 1), overflow-safe."""
    return 1.0 / (np.exp(np.minimum(x, 500.0)) + 1.0)


def evolve_characteristic_rays(Sigma_H, N_mu=24, N_steps=2000, N_total=6.0):
    """Evolve characteristic rays through coupled shear-transport system.

    Solves the self-consistent system:
        dΣ/dN = -Σ + Π(I, J, μ)
        dI_j/dN = Σ P₂(μ_j)
        dJ_j/dN = 3Σ(1-3μ_j²)J_j
        dS/dN = Σ

    Returns the final ray state (I, J, mu, w, S).
    """
    mu0, w0 = leggauss(N_mu)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)
    signs = np.sign(mu0)
    f_nu = 0.4052

    Sigma = Sigma_H
    I = np.zeros(N_mu)
    J = np.ones(N_mu)
    S = 0.0
    dN = N_total / N_steps

    for _ in range(N_steps):
        # Current directions
        X = X0 * np.exp(6 * S)
        mu = signs * np.sqrt(np.minimum(X / (1 + X), 1 - 1e-15))
        P2 = eval_legendre(2, mu)

        # Anisotropic stress
        Pi = f_nu * np.sum(w0 * J * P2 * np.exp(-8 * I))

        # Euler step (sufficient for this diagnostic)
        q_dec = 1 + Sigma**2
        dSigma = (-(2 - q_dec) * Sigma + Pi) * dN
        dI = Sigma * P2 * dN
        dJ = 3 * Sigma * (1 - 3 * mu**2) * J * dN
        dS = Sigma * dN

        Sigma += dSigma
        I += dI
        J += dJ
        S += dS

    # Final directions
    X = X0 * np.exp(6 * S)
    mu = signs * np.sqrt(np.minimum(X / (1 + X), 1 - 1e-15))

    return I, J, mu, w0, S, Sigma


# ═══════════════════════════════════════════════════════════════════
# §2. Monopole computation at various (ℓ, n) truncation levels
# ═══════════════════════════════════════════════════════════════════

def exact_ray_monopole(q, I, J, w):
    """Fully nonperturbative monopole: f̃₀(q) = ½Σ wⱼJⱼ f_FD(q/Θⱼ)."""
    Theta = np.exp(-2 * I)  # (N_mu,)
    qa = q[:, None] / Theta[None, :]  # (N_q, N_mu)
    f_vals = fermi_dirac(qa)
    return 0.5 * f_vals @ (w * J)


def teff_monopole(q, I, J, mu, w, ell_max, n_order):
    """Teff-approximated monopole at truncation (ℓ_max, n_order).

    ℓ_max: angular truncation of the Θ(μ) expansion
    n_order: spectral expansion order (0, 1, 2, or 'full')
    """
    Theta_j = np.exp(-2 * I)  # per-ray effective temperatures

    # ── Step 1: Angular truncation of Θ(μ) ──
    if ell_max == 0:
        # Monopole only: Θ̄ = ⟨Θ⟩
        Theta_bar = 0.5 * np.sum(w * J * Theta_j)
        Theta_func = np.full_like(mu, Theta_bar)
    elif ell_max == 2:
        # Monopole + quadrupole
        Theta_bar = 0.5 * np.sum(w * J * Theta_j)
        P2_mu = eval_legendre(2, mu)
        c2 = (5.0 / 2.0) * np.sum(w * J * Theta_j * P2_mu) / (2 * Theta_bar)
        Theta_func = Theta_bar * (1 + c2 * P2_mu)
    elif ell_max == 4:
        # Monopole + quadrupole + hexadecapole
        Theta_bar = 0.5 * np.sum(w * J * Theta_j)
        P2_mu = eval_legendre(2, mu)
        P4_mu = eval_legendre(4, mu)
        c2 = (5.0 / 2.0) * np.sum(w * J * Theta_j * P2_mu) / (2 * Theta_bar)
        c4 = (9.0 / 2.0) * np.sum(w * J * Theta_j * P4_mu) / (2 * Theta_bar)
        Theta_func = Theta_bar * (1 + c2 * P2_mu + c4 * P4_mu)
    elif ell_max >= 100:  # "infinity"
        # Use full per-ray Θ_j (no angular truncation)
        Theta_func = Theta_j
    else:
        raise ValueError(f"ell_max={ell_max} not supported")

    # ── Step 2: Spectral expansion ──
    if n_order == 0:
        # No spectral correction: just equilibrium
        return fermi_dirac(q)

    elif n_order == 1:
        # First-order: ⟨f(q/Θ)⟩ ≈ f(q) + ⟨δΘ/Θ⟩ × q f'(q)
        # For even-parity Θ(μ), ⟨δΘ⟩ = 0, so n=1 = n=0
        return fermi_dirac(q)

    elif n_order == 2:
        # Second-order Jensen spectral hardening
        f0 = fermi_dirac(q)
        df = -f0 * (1 - f0)  # f'(q) = -f(1-f)
        d2f = f0 * (1 - f0) * (1 - 2*f0)  # f''(q)

        # Variance of Θ
        Theta_bar = 0.5 * np.sum(w * J * Theta_func)
        Sigma2 = 0.5 * np.sum(w * J * ((Theta_func/Theta_bar - 1)**2))

        # Second-order correction: δf = ½ Σ₂ × [q² f''(q) + 2q f'(q)]
        correction = 0.5 * Sigma2 * (q**2 * d2f + 2 * q * df)
        return f0 + correction

    elif n_order == 'full' or n_order >= 100:
        # Exact numerical integration: ½ Σⱼ wⱼ Jⱼ f_FD(q/Θⱼ)
        qa = q[:, None] / np.maximum(Theta_func[None, :], 1e-30)
        f_vals = fermi_dirac(qa)
        return 0.5 * f_vals @ (w * J)

    else:
        raise ValueError(f"n_order={n_order}")


# ═══════════════════════════════════════════════════════════════════
# §3. Weak rate integral (simplified Channel 2 diagnostic)
# ═══════════════════════════════════════════════════════════════════

def weak_rate_integral(f_nu_mono, q_nodes, q_weights):
    """Simplified weak rate integral ∝ ∫ q³ f(q) dq (energy-weighted).

    This captures the Channel-2 physics: the spectral distortion of
    the monopole modifies the effective neutrino energy density seen
    by the weak rates.
    """
    integrand = q_weights * np.exp(q_nodes) * q_nodes**3 * f_nu_mono
    return np.sum(integrand)


def channel2_signal(f_mono, f_eq, q_nodes, q_weights):
    """Channel-2 signal: fractional shift in the weak rate integral."""
    Lambda_mono = weak_rate_integral(f_mono, q_nodes, q_weights)
    Lambda_eq = weak_rate_integral(f_eq, q_nodes, q_weights)
    return (Lambda_mono - Lambda_eq) / Lambda_eq


# ═══════════════════════════════════════════════════════════════════
# §4. Full ablation computation
# ═══════════════════════════════════════════════════════════════════

def run_ablation(Sigma_H, N_mu=24, N_q=80):
    """Run the full (ℓ, n) ablation study at a given Σ_H."""
    from numpy.polynomial.laguerre import laggauss

    print(f"  Σ_H = {Sigma_H}: evolving rays...", end=" ", flush=True)
    I, J, mu, w, S, Sigma_final = evolve_characteristic_rays(
        Sigma_H, N_mu=N_mu, N_steps=3000, N_total=8.0)
    print(f"Σ_final={Sigma_final:.6f}, max|I|={np.max(np.abs(I)):.6f}")

    # Momentum grid (Gauss-Laguerre)
    q_nodes, q_weights_raw = laggauss(N_q)
    q_weights = q_weights_raw  # for ∫ g(q) e^{-q} dq ≈ Σ wᵢ g(qᵢ)
    f_eq = fermi_dirac(q_nodes)

    # Reference: exact ray monopole (fully nonperturbative)
    f_exact = exact_ray_monopole(q_nodes, I, J, w)
    signal_exact = channel2_signal(f_exact, f_eq, q_nodes, q_weights)

    # Ablation grid: (ℓ, n) combinations
    ell_values = [0, 2, 4, 999]  # 999 = ∞
    n_values = [0, 1, 2, 'full']
    ell_labels = ['$\\ell=0$', '$\\ell=2$', '$\\ell=4$', '$\\ell=\\infty$']
    n_labels = ['$n=0$', '$n=1$', '$n=2$', '$n=$full']

    results = np.zeros((len(ell_values), len(n_values)))
    signals = np.zeros((len(ell_values), len(n_values)))
    monopoles = {}

    for i, ell in enumerate(ell_values):
        for j, n in enumerate(n_values):
            f_approx = teff_monopole(q_nodes, I, J, mu, w, ell, n)
            sig = channel2_signal(f_approx, f_eq, q_nodes, q_weights)
            signals[i, j] = sig

            # Signal loss relative to exact
            if abs(signal_exact) > 1e-30:
                loss = 1.0 - sig / signal_exact
                results[i, j] = loss * 100  # percentage
            else:
                results[i, j] = 0.0

            # Store selected monopoles for plotting
            key = (ell_labels[i], n_labels[j])
            monopoles[key] = f_approx

    return {
        'Sigma_H': Sigma_H,
        'signal_exact': signal_exact,
        'signals': signals,
        'loss_pct': results,
        'ell_labels': ell_labels,
        'n_labels': n_labels,
        'ell_values': ell_values,
        'n_values': n_values,
        'q_nodes': q_nodes,
        'f_exact': f_exact,
        'f_eq': f_eq,
        'monopoles': monopoles,
        'I': I, 'J': J, 'mu': mu, 'w': w,
    }


# ═══════════════════════════════════════════════════════════════════
# §5. Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_ablation_heatmap(results_list, outdir):
    """Heatmap of signal loss for each (ℓ, n)."""
    n_sigma = len(results_list)
    fig, axes = plt.subplots(1, n_sigma, figsize=(6*n_sigma, 5))
    if n_sigma == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        data = res['loss_pct']
        im = ax.imshow(data, cmap='RdYlGn_r', vmin=-5, vmax=105, aspect='auto')
        ax.set_xticks(range(len(res['n_labels'])))
        ax.set_xticklabels(res['n_labels'], fontsize=11)
        ax.set_yticks(range(len(res['ell_labels'])))
        ax.set_yticklabels(res['ell_labels'], fontsize=11)
        ax.set_xlabel('Spectral order $n$', fontsize=12)
        ax.set_ylabel('Angular resolution $\\ell$', fontsize=12)
        ax.set_title(f'$\\Sigma_H = {res["Sigma_H"]}$\n'
                     f'Exact signal: {res["signal_exact"]:.2e}', fontsize=11)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = 'white' if val > 50 else 'black'
                txt = f'{val:.1f}%' if abs(val) < 999 else 'N/A'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes[-1], label='Channel-2 signal loss [%]', shrink=0.8)
    fig.suptitle('Teff $(\\ell, n)$ Ablation: Signal Loss vs Exact Ray Transport',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(outdir, 'fig_ablation_computed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def plot_monopole_comparison(res, outdir):
    """Compare monopole distortions at different (ℓ, n) levels."""
    q = res['q_nodes']
    f_eq = res['f_eq']
    f_exact = res['f_exact']
    mask = (q > 0.3) & (q < 15)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel a: monopole distortion δf/f_eq
    ax = axes[0]
    delta_exact = (f_exact - f_eq) / np.maximum(f_eq, 1e-30)
    ax.plot(q[mask], delta_exact[mask] * 1e4, 'k-', lw=2.5, label='Exact (ray)')

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for idx, n in enumerate([0, 2, 'full']):
        n_label = f'$n={n}$' if isinstance(n, int) else '$n=$full'
        key = ('$\\ell=\\infty$', n_label)
        if key in res['monopoles']:
            f_approx = res['monopoles'][key]
            delta = (f_approx - f_eq) / np.maximum(f_eq, 1e-30)
            ax.plot(q[mask], delta[mask] * 1e4, '--', lw=1.5,
                    color=colors[idx], label=f'$\\ell=\\infty$, {n_label}')

    ax.set_xlabel('$q = p/T_\\nu$', fontsize=12)
    ax.set_ylabel('$\\delta f / f_{\\rm eq}$ [$\\times 10^{-4}$]', fontsize=12)
    ax.set_title(f'(a) Monopole distortion, $\\Sigma_H={res["Sigma_H"]}$')
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)

    # Panel b: ℓ-dependence at n=full
    ax = axes[1]
    ax.plot(q[mask], delta_exact[mask] * 1e4, 'k-', lw=2.5, label='Exact (ray)')
    ell_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    for idx, (ell_label, ell_val) in enumerate(zip(res['ell_labels'], res['ell_values'])):
        key = (ell_label, '$n=$full')
        if key in res['monopoles']:
            f_approx = res['monopoles'][key]
            delta = (f_approx - f_eq) / np.maximum(f_eq, 1e-30)
            ax.plot(q[mask], delta[mask] * 1e4, '--', lw=1.5,
                    color=ell_colors[idx], label=f'{ell_label}, $n=$full')

    ax.set_xlabel('$q = p/T_\\nu$', fontsize=12)
    ax.set_ylabel('$\\delta f / f_{\\rm eq}$ [$\\times 10^{-4}$]', fontsize=12)
    ax.set_title(f'(b) Angular resolution at $n=$full')
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)

    # Panel c: signal recovery fraction vs Sigma_H (if multiple)
    ax = axes[2]
    # Show the ℓ=2 n=2 case (standard PSTF+Teff) vs exact
    key_std = ('$\\ell=2$', '$n=2$')
    key_inf_full = ('$\\ell=\\infty$', '$n=$full')
    if key_std in res['monopoles'] and key_inf_full in res['monopoles']:
        f_std = res['monopoles'][key_std]
        f_inf = res['monopoles'][key_inf_full]
        delta_std = (f_std - f_eq) / np.maximum(f_eq, 1e-30)
        delta_inf = (f_inf - f_eq) / np.maximum(f_eq, 1e-30)

        ax.plot(q[mask], delta_exact[mask] * 1e4, 'k-', lw=2.5, label='Exact ray')
        ax.plot(q[mask], delta_inf[mask] * 1e4, 'b--', lw=1.5,
                label='$\\ell=\\infty$, $n=$full')
        ax.plot(q[mask], delta_std[mask] * 1e4, 'r:', lw=2,
                label='$\\ell=2$, $n=2$ (std Teff)')
        ax.fill_between(q[mask],
                        delta_std[mask] * 1e4, delta_exact[mask] * 1e4,
                        alpha=0.15, color='red', label='Teff truncation error')

    ax.set_xlabel('$q = p/T_\\nu$', fontsize=12)
    ax.set_ylabel('$\\delta f / f_{\\rm eq}$ [$\\times 10^{-4}$]', fontsize=12)
    ax.set_title('(c) Standard Teff vs exact ray')
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, 'fig_ablation_monopoles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def plot_sigma_scan(all_results, outdir):
    """Signal loss vs Σ_H for selected (ℓ, n) configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sigmas = [r['Sigma_H'] for r in all_results]
    exact_signals = [r['signal_exact'] for r in all_results]

    # Panel a: exact signal vs Sigma_H
    ax = axes[0]
    ax.semilogy(sigmas, np.abs(exact_signals), 'ko-', lw=2, ms=8)
    ax.set_xlabel('$\\Sigma_H$', fontsize=12)
    ax.set_ylabel('|Channel-2 signal|', fontsize=12)
    ax.set_title('(a) Exact Channel-2 signal magnitude')
    ax.grid(True, alpha=0.3)

    # Panel b: signal loss for selected configs
    ax = axes[1]
    configs = [
        (0, 0, '$\\ell=0, n=0$', 'red', 'o'),
        (1, 2, '$\\ell=2, n=2$ (std)', 'blue', 's'),
        (2, 2, '$\\ell=4, n=2$', 'green', '^'),
        (3, 3, '$\\ell=\\infty, n=$full', 'purple', 'D'),
    ]
    for i_ell, i_n, label, color, marker in configs:
        losses = [r['loss_pct'][i_ell, i_n] for r in all_results]
        ax.plot(sigmas, losses, f'{marker}-', color=color, lw=1.5,
                ms=7, label=label)

    ax.set_xlabel('$\\Sigma_H$', fontsize=12)
    ax.set_ylabel('Signal loss [%]', fontsize=12)
    ax.set_title('(b) Truncation error vs anisotropy')
    ax.legend(fontsize=9)
    ax.set_ylim(-10, 110)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(100, color='k', lw=0.5, ls=':')

    fig.tight_layout()
    path = os.path.join(outdir, 'fig_ablation_sigma_scan.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# §6. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Teff ablation study')
    parser.add_argument('--sigma', nargs='+', type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.5],
                        help='Sigma_H values to scan')
    parser.add_argument('--outdir', default='figures',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  RABBIT Teff Ablation Study")
    print("  Exact ray transport vs Teff (ℓ, n) truncation")
    print("=" * 60)

    all_results = []
    for sigma in args.sigma:
        res = run_ablation(sigma, N_mu=24, N_q=60)
        all_results.append(res)

        print(f"\n  Σ_H = {sigma}: signal loss matrix [%]:")
        print(f"  {'':>12s}", end="")
        for nl in res['n_labels']:
            print(f"  {nl:>10s}", end="")
        print()
        for i, el in enumerate(res['ell_labels']):
            print(f"  {el:>12s}", end="")
            for j in range(len(res['n_labels'])):
                print(f"  {res['loss_pct'][i,j]:>9.1f}%", end="")
            print()

    print("\n" + "=" * 60)
    print("  Generating figures...")
    print("=" * 60)

    plot_ablation_heatmap(all_results, args.outdir)
    # Plot monopole comparison for Σ_H = 0.3 (or closest)
    idx_03 = min(range(len(args.sigma)),
                 key=lambda i: abs(args.sigma[i] - 0.3))
    plot_monopole_comparison(all_results[idx_03], args.outdir)
    plot_sigma_scan(all_results, args.outdir)

    print("\n" + "=" * 60)
    print("  Done. Figures saved to", args.outdir)
    print("=" * 60)


if __name__ == "__main__":
    main()
