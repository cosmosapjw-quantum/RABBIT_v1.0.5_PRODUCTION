#!/usr/bin/env python3
"""Block E: Verification infrastructure strengthening.

Tests:
  E1. Tolerance sensitivity sweep (rtol: 1e-6 → 1e-10)
  E2. Characteristic vs linearized regression table
  E3. AD gradient finite-difference cross-check (if JAX available)
  E4. Transport mode consistency at multiple Σ

Output: VERIFICATION_INFRASTRUCTURE_RESULTS.json
"""
import sys, os, time, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
from rabbit.config.transport_mode import TransportMode
from rabbit.config.solver_config import SolverConfig, SolverMethod

OUT_DIR = Path(__file__).parent.parent / "audit_outputs"
OUT_DIR.mkdir(exist_ok=True)


def test_e1_tolerance():
    """E1: Tolerance sensitivity at Σ=0.3."""
    print("E1. Tolerance sensitivity (Σ=0.3, characteristic):")
    results = {}
    for rtol in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        solver = SolverConfig(method=SolverMethod.RADAU, rtol=rtol, atol=rtol*1e-2, max_step=0.5)
        r = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.3, transport_mode=TransportMode.CHARACTERISTIC,
            N_q=20, solver=solver))
        results[str(rtol)] = r.observables.Yp
        print(f"  rtol={rtol:.0e}: Yp={r.observables.Yp:.10f}  ({r.wall_time_s:.1f}s)")

    # Convergence check
    vals = list(results.values())
    diffs = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]
    print(f"  Successive diffs: {[f'{d:.1e}' for d in diffs]}")
    converged = diffs[-1] < 1e-8
    print(f"  Converged at rtol=1e-8: {'YES' if converged else 'NO'} (last diff={diffs[-1]:.1e})")
    return results, converged


def test_e2_regression_table():
    """E2: Characteristic vs linearized regression at multiple Σ and CL."""
    print("\nE2. Regression table (characteristic vs linearized):")
    print(f"  {'Σ':>5s}  {'CL':>3s}  {'Yp_lin':>12s}  {'Yp_char':>12s}  {'ΔYp':>12s}")

    table = []
    for sigma in [0.0, 0.1, 0.3, 0.5]:
        for cl in [0, 2]:
            r_l = run_full_coupled_typeI(FullCoupledConfig(
                Sigma_H_plus=sigma, correction_level=cl, N_q=20,
                transport_mode=TransportMode.LINEARIZED_PSTF))
            r_c = run_full_coupled_typeI(FullCoupledConfig(
                Sigma_H_plus=sigma, correction_level=cl, N_q=20,
                transport_mode=TransportMode.CHARACTERISTIC))
            dYp = r_c.observables.Yp - r_l.observables.Yp
            print(f"  {sigma:5.1f}  CL{cl}  {r_l.observables.Yp:12.8f}  {r_c.observables.Yp:12.8f}  {dYp:+12.4e}")
            table.append({
                'Sigma_H': sigma, 'CL': cl,
                'Yp_linearized': r_l.observables.Yp,
                'Yp_characteristic': r_c.observables.Yp,
                'delta_Yp': dYp,
            })
    return table


def test_e3_ad_gradient():
    """E3: AD gradient vs finite-difference (JAX, if available)."""
    print("\nE3. AD gradient check:")
    try:
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        os.environ.setdefault('JAX_ENABLE_X64', '1')
        import jax; jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        # Finite-difference gradient using SciPy production driver
        h = 0.001
        r_plus = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.1 + h, correction_level=0, N_q=20,
            transport_mode=TransportMode.CHARACTERISTIC))
        r_minus = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.1 - h, correction_level=0, N_q=20,
            transport_mode=TransportMode.CHARACTERISTIC))
        grad_fd = (r_plus.observables.Yp - r_minus.observables.Yp) / (2*h)

        print(f"  FD gradient dYp/dΣ at Σ=0.1: {grad_fd:.6e}")
        print(f"  (Yp(0.101)={r_plus.observables.Yp:.8f}, Yp(0.099)={r_minus.observables.Yp:.8f})")

        # JAX AD gradient would go here when JAX driver supports characteristic
        print(f"  JAX AD gradient: deferred (JAX driver characteristic integration = Block A.3)")

        return {'fd_gradient': grad_fd, 'Sigma': 0.1, 'h': h}

    except ImportError:
        print("  SKIP: JAX not available")
        return None


def test_e4_mode_consistency():
    """E4: Both modes produce same FLRW baseline across CL levels."""
    print("\nE4. Transport mode consistency (FLRW):")
    all_pass = True
    for cl in [0, 1, 2]:
        r_l = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.0, correction_level=cl, N_q=20,
            transport_mode=TransportMode.LINEARIZED_PSTF))
        r_c = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.0, correction_level=cl, N_q=20,
            transport_mode=TransportMode.CHARACTERISTIC))
        dYp = abs(r_c.observables.Yp - r_l.observables.Yp)
        ok = dYp < 1e-8
        all_pass &= ok
        print(f"  CL{cl}: |ΔYp| = {dYp:.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def main():
    print("=" * 70)
    print("  Block E: Verification Infrastructure")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    results = {'generated': datetime.now(timezone.utc).isoformat()}

    tol_results, tol_ok = test_e1_tolerance()
    results['E1_tolerance'] = {'data': tol_results, 'converged': tol_ok}

    table = test_e2_regression_table()
    results['E2_regression'] = table

    ad = test_e3_ad_gradient()
    results['E3_ad_gradient'] = ad

    mode_ok = test_e4_mode_consistency()
    results['E4_mode_consistency'] = mode_ok

    # Write
    json_path = OUT_DIR / "VERIFICATION_INFRASTRUCTURE_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")


if __name__ == "__main__":
    main()
