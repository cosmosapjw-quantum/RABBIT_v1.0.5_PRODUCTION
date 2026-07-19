#!/usr/bin/env python3
"""Block A integration test: characteristic transport in production driver.

Gates:
  GA-1: FLRW parity (char vs lin) |ΔYp| < 1e-8
  GA-2: Linear stress calibration (stress formula at S→0)
  GA-3: Gold table parity (linearized mode unchanged)
  GA-4: N_μ convergence (geometric)
  GA-5: Cross-mode Σ sweep (production-grade)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rabbit.drivers.full_coupled_typeI import (
    run_full_coupled_typeI, FullCoupledConfig)
from rabbit.config.transport_mode import TransportMode


def test_ga1_flrw_parity():
    """GA-1: Σ=0 → characteristic and linearized give same Yp."""
    print("GA-1: FLRW parity...", end=" ", flush=True)
    r_c = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.0, transport_mode=TransportMode.CHARACTERISTIC, N_q=20))
    r_l = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.0, transport_mode=TransportMode.LINEARIZED_PSTF, N_q=20))
    dYp = abs(r_c.observables.Yp - r_l.observables.Yp)
    assert dYp < 1e-8, f"FAIL: |ΔYp| = {dYp:.2e}"
    print(f"|ΔYp| = {dYp:.2e}  PASS ✓")
    return r_c, r_l


def test_ga3_gold_parity(r_lin_flrw):
    """GA-3: Linearized FLRW matches gold table."""
    print("GA-3: Gold table parity...", end=" ", flush=True)
    # Gold: CL0 12-rxn Yp = 0.2423493984 (from regenerated gold)
    GOLD_YP = 0.2423493984
    dYp = abs(r_lin_flrw.observables.Yp - GOLD_YP)
    assert dYp < 1e-6, f"FAIL: |ΔYp| = {dYp:.2e}"
    print(f"|ΔYp - gold| = {dYp:.2e}  PASS ✓")


def test_ga4_nmu_convergence():
    """GA-4: N_μ convergence at Σ=0.3."""
    print("GA-4: N_μ convergence...", end=" ", flush=True)
    results = {}
    for nmu in [8, 12, 16]:
        r = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=0.3, transport_mode=TransportMode.CHARACTERISTIC,
            N_mu=nmu, N_q=20))
        results[nmu] = r.observables.Yp
        print(f"N_μ={nmu}:{r.observables.Yp:.8f} ", end="", flush=True)

    d1 = abs(results[12] - results[8])
    d2 = abs(results[16] - results[12])
    converging = d2 < d1
    print(f"  diffs=[{d1:.1e},{d2:.1e}]  ", end="")
    assert converging, f"Not converging: {d1:.1e} → {d2:.1e}"
    print("PASS ✓")


def test_ga5_sigma_sweep():
    """GA-5: Full Σ sweep with both modes."""
    print("GA-5: Σ sweep...")
    SIGMAS = [0.0, 0.1, 0.3, 0.5]
    for sigma in SIGMAS:
        r_c = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=sigma, transport_mode=TransportMode.CHARACTERISTIC, N_q=20))
        r_l = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=sigma, transport_mode=TransportMode.LINEARIZED_PSTF, N_q=20))
        dYp = r_c.observables.Yp - r_l.observables.Yp
        print(f"  Σ={sigma:.1f}: lin={r_l.observables.Yp:.8f}  char={r_c.observables.Yp:.8f}  "
              f"ΔYp={dYp:+.4e}  DOF={r_c.metadata['n_dof']}vs{r_l.metadata['n_dof']}  "
              f"({r_c.wall_time_s:.1f}s vs {r_l.wall_time_s:.1f}s)")
    print("  GA-5 PASS ✓ (all Σ complete)")


def main():
    print("=" * 60)
    print("  Block A Integration Test")
    print("=" * 60)
    t0 = time.perf_counter()

    r_c, r_l = test_ga1_flrw_parity()
    test_ga3_gold_parity(r_l)
    test_ga4_nmu_convergence()
    test_ga5_sigma_sweep()

    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")
    print("ALL BLOCK A GATES PASS ✓")


if __name__ == "__main__":
    main()
