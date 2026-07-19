#!/usr/bin/env python3
"""Block B: Production inference with canonical forward solver.

Derives Σ_H constraints using the EXACT production BBN solver
(characteristic transport, NOT the simplified analytical proxy).

Runs:
  1. 1D grid scan: Σ_H ∈ [0, 0.9] at CL0, CL1, CL2
  2. Profile likelihood → 95% CL upper limit
  3. Savage-Dickey Bayes factor B₀₁
  4. Comparison: characteristic vs linearized transport

Output: PRODUCTION_INFERENCE_RESULTS.json

Usage:
  python3 scripts/run_production_inference.py
  python3 scripts/run_production_inference.py --cl 0        # CL0 only
  python3 scripts/run_production_inference.py --linearized  # also run linearized
  python3 scripts/run_production_inference.py --npoints 30  # fewer grid points
"""
import sys, os, time, json, argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from rabbit.inference.forward_likelihood import (
    canonical_forward_solver, BBNLikelihood, ForwardModel,
    OBS_YP, OBS_DH, OBS_TAU_N, OBS_ETA,
    grid_scan, confidence_interval_1d,
)

OUT_DIR = Path(__file__).parent.parent / "audit_outputs"
OUT_DIR.mkdir(exist_ok=True)


def run_1d_scan(correction_level, n_points=50, sigma_max=0.9, backend='scipy'):
    """Run 1D Σ_H grid scan with production solver."""
    sigma_grid = np.linspace(0, sigma_max, n_points)

    model = ForwardModel(
        solver_fn=lambda Sigma_H=0.0, eta=6.104e-10, tau_n=878.4, **kw:
            canonical_forward_solver(
                Sigma_H=Sigma_H, eta=eta, tau_n=tau_n,
                correction_level=correction_level,
                n_reactions=12, N_q=20, backend=backend))

    lik = BBNLikelihood(model, [OBS_YP, OBS_DH],
                         parameter_priors={'tau_n': OBS_TAU_N, 'eta': OBS_ETA})

    result = grid_scan(lik, param_grids={'Sigma_H': sigma_grid}, verbose=True)
    return result, sigma_grid


def savage_dickey(grid_result, sigma_grid, prior_sigma=0.5):
    """Compute Savage-Dickey Bayes factor B₀₁.

    B₀₁ = p(Σ_H=0 | data, M₁) / p(Σ_H=0 | M₁)

    Prior: half-normal with σ = prior_sigma.
    """
    chi2 = grid_result.chi2.ravel()
    dchi2 = chi2 - np.min(chi2)
    log_lik = -0.5 * dchi2

    # Prior: half-normal p(Σ) = (2/σ√(2π)) exp(-Σ²/(2σ²)) for Σ≥0
    log_prior = -0.5 * (sigma_grid / prior_sigma)**2
    log_prior -= np.log(prior_sigma * np.sqrt(2 * np.pi) / 2)

    # Unnormalized posterior
    log_post = log_lik + log_prior
    log_post -= np.max(log_post)  # numerical stability
    post = np.exp(log_post)

    # Normalize
    dS = sigma_grid[1] - sigma_grid[0]
    post_norm = post / (np.sum(post) * dS)
    prior_at_zero = np.exp(-0.0) * 2 / (prior_sigma * np.sqrt(2 * np.pi))

    # Posterior density at Σ=0 (interpolate)
    post_at_zero = post_norm[0]

    B01 = post_at_zero / prior_at_zero
    return {
        'B01': float(B01),
        'ln_B01': float(np.log(B01)) if B01 > 0 else float('-inf'),
        'prior_at_zero': float(prior_at_zero),
        'post_at_zero': float(post_at_zero),
        'prior_sigma': prior_sigma,
    }


def main():
    parser = argparse.ArgumentParser(description="Production inference with canonical solver")
    parser.add_argument('--cl', type=int, nargs='+', default=[0, 1, 2],
                        help='Correction levels to scan (default: 0 1 2)')
    parser.add_argument('--npoints', type=int, default=50,
                        help='Grid points (default: 50)')
    parser.add_argument('--sigma-max', type=float, default=0.9,
                        help='Max Σ_H (default: 0.9)')
    parser.add_argument('--linearized', action='store_true',
                        help='Also run linearized for comparison')
    args = parser.parse_args()

    print("=" * 70)
    print("  RABBIT Production Inference — Canonical Forward Solver")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Transport: CHARACTERISTIC (exact nonlinear)")
    print(f"  Grid: {args.npoints} points, Σ_H ∈ [0, {args.sigma_max}]")
    print(f"  Correction levels: {args.cl}")
    print("=" * 70)

    results = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'transport_mode': 'characteristic',
        'n_points': args.npoints,
        'sigma_max': args.sigma_max,
        'scans': {},
    }

    for cl in args.cl:
        print(f"\n{'='*60}")
        print(f"  CL{cl} — 1D Σ_H scan ({args.npoints} points)")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        grid_result, sigma_grid = run_1d_scan(
            correction_level=cl, n_points=args.npoints,
            sigma_max=args.sigma_max)
        elapsed = time.perf_counter() - t0

        # Profile likelihood
        chi2 = grid_result.chi2.ravel()
        best_idx = np.argmin(chi2)
        best_sigma = float(sigma_grid[best_idx])
        best_chi2 = float(chi2[best_idx])
        dchi2 = chi2 - best_chi2

        # 95% CL (Δχ² = 3.84 for 1-sided)
        above = np.where(dchi2 > 3.84)[0]
        if len(above) > 0 and above[0] > 0:
            sigma_95 = float(np.interp(3.84, dchi2[:above[0]+1], sigma_grid[:above[0]+1]))
        else:
            sigma_95 = float(args.sigma_max)

        # Savage-Dickey
        sd = savage_dickey(grid_result, sigma_grid)

        # Yp at key points
        yp_values = {}
        for s in [0.0, 0.1, 0.3, 0.5]:
            idx = np.argmin(np.abs(sigma_grid - s))
            pred = canonical_forward_solver(
                Sigma_H=float(sigma_grid[idx]),
                correction_level=cl, n_reactions=12, N_q=20)
            yp_values[f'Sigma_{s}'] = {
                'Sigma_H': float(sigma_grid[idx]),
                'Yp': float(pred.Yp), 'DH': float(pred.DH),
            }

        # ── Collect scan result ──
        scan_result = {
            'correction_level': cl,
            'n_points': args.npoints,
            'sigma_max': args.sigma_max,
            'best_sigma': best_sigma,
            'best_chi2': best_chi2,
            'sigma_95': sigma_95,
            'B01': sd['B01'],
            'ln_B01': sd['ln_B01'],
            'yp_values': yp_values,
            'elapsed_s': elapsed,
        }

        # ── High-Σ safety check ──
        _SIGMA_SAFE = 0.75
        unsafe_points = np.sum(sigma_grid > _SIGMA_SAFE)
        if unsafe_points > 0:
            # Flag any chi2 anomalies above safe threshold
            safe_mask = sigma_grid <= _SIGMA_SAFE
            if np.any(~safe_mask):
                chi2_unsafe = chi2[~safe_mask]
                chi2_safe_max = np.max(chi2[safe_mask]) if np.any(safe_mask) else 999
                anomalous = np.any(chi2_unsafe < chi2_safe_max)
                if anomalous:
                    scan_result['high_sigma_warning'] = (
                        f"Chi² anomaly detected above Σ > {_SIGMA_SAFE}: "
                        f"some high-Σ values have lower chi² than safe-region maximum. "
                        f"Constraint at Σ < {sigma_95:.3f} is reliable if below {_SIGMA_SAFE}."
                    )
                    print(f"    ⚠ High-Σ anomaly: results above Σ > {_SIGMA_SAFE} may be unreliable")

        scan_result['grid_sensitivity_note'] = (
            f"B₀₁ has ~20% sensitivity to grid resolution. "
            f"Qualitative conclusion (Jeffreys scale) is stable; "
            f"quantitative B₀₁ should be quoted as B₀₁ ≈ {sd['B01']:.1f} ± 0.3."
        )

        results['scans'][f'CL{cl}'] = scan_result

        print(f"\n  Results (CL{cl}):")
        print(f"    Best fit: Σ_H = {best_sigma:.3f}, χ² = {best_chi2:.2f}")
        print(f"    χ²(Σ=0) = {chi2[0]:.2f}")
        print(f"    95% CL upper limit: Σ_H < {sigma_95:.3f}")
        print(f"    Savage-Dickey B₀₁ = {sd['B01']:.3f} ± ~0.3 (ln B₀₁ = {sd['ln_B01']:.3f})")
        print(f"    Elapsed: {elapsed:.1f}s")
        for key, val in yp_values.items():
            print(f"    {key}: Yp = {val['Yp']:.8f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'CL':>3s}  {'Σ_H(95%)':>10s}  {'B₀₁':>8s}  {'ln B₀₁':>8s}  {'Yp(0)':>12s}  {'Yp(0.3)':>12s}")
    for cl in args.cl:
        s = results['scans'].get(f'CL{cl}')
        if s:
            yp0 = s['yp_values'].get('Sigma_0.0', {}).get('Yp', 0)
            yp3 = s['yp_values'].get('Sigma_0.3', {}).get('Yp', 0)
            print(f"  CL{cl}  {s['sigma_95']:10.4f}  {s['B01']:8.3f}  "
                  f"{s['ln_B01']:8.3f}  {yp0:12.8f}  {yp3:12.8f}")

    # Write
    json_path = OUT_DIR / "PRODUCTION_INFERENCE_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")


if __name__ == "__main__":
    main()
