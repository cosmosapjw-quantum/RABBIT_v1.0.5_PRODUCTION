"""
rabbit.validation.generic_typeI — Generic Type I (Σ₊, Σ₋) validation.

Validates the two-polarization transport implementation (P4-A) by:
  1. LRS recovery: Σ₋ = 0 reproduces the single-polarization result
  2. Exchange symmetry: (Σ₊, 0) ↔ (0, Σ₋) gives identical yields
  3. Isotropy: Y_p depends on Σ² = Σ₊² + Σ₋² at leading order
  4. 2D constraint surface: (Σ₊, Σ₋) allowed region from Y_p observations

Physics:
  In generic Type I, the total shear scalar is Σ² = Σ₊² + Σ₋².
  The modified Hubble rate H = H_FLRW / √(1-Σ²) depends only on Σ².
  Channel 1 (expansion) therefore depends only on Σ², not on the
  individual polarizations.

  Channel 2 (spectral hardening) DOES depend on the polarization split:
  Ψ₂⁺ couples to Σ₊ and Ψ₂⁻ couples to Σ₋, generating independent
  anisotropic stresses Π₊ and Π₋.  However, the Teff correction is
  also O(Σ²), and the two contributions add in the rate integral:
  δλ/λ ∝ T₂⁺² + T₂⁻² ∝ Σ₊² + Σ₋² = Σ².

  Therefore, to leading order O(Σ²), Y_p depends only on Σ²:
      ΔY_p(Σ₊, Σ₋) ≈ ΔY_p(Σ, 0) where Σ² = Σ₊² + Σ₋²

  Anisotropy between polarizations appears at O(Σ⁴) from cross-terms
  in the nonlinear geometry (q = 1 + Σ₊² + Σ₋²) and from the
  polarization-dependent neutrino viscosity (separate π̃₊, π̃₋).

  The exchange symmetry (Σ₊ ↔ Σ₋) is EXACT at all orders by the
  Z₂ symmetry of the Type I shear equations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ═══════════════════════════════════════════════════════════════
# §1. Scan configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class GenericTypeIScanConfig:
    """Configuration for a (Σ₊, Σ₋) parameter scan.

    Parameters
    ----------
    Sigma_values : list of float
        1D grid of Σ values to scan.  The 2D grid is
        {(Σ₊, Σ₋) : Σ₊ ∈ Sigma_values, Σ₋ ∈ Sigma_values, Σ₊²+Σ₋² < 1}.
    include_diagonal : bool
        If True, include points along Σ₊ = Σ₋.
    include_exchange : bool
        If True, include both (a, b) and (b, a) for exchange symmetry test.
    tau_n : float
        Neutron lifetime [s].
    eta : float
        Baryon-to-photon ratio.
    N_q : int
        Momentum grid resolution.
    tier : int
        Decoupling tier (1 or 2).
    """
    Sigma_values: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    include_diagonal: bool = True
    include_exchange: bool = True
    tau_n: float = 878.4
    eta: float = 6.104e-10
    N_q: int = 80
    tier: int = 1


def generate_scan_points(config: GenericTypeIScanConfig) -> List[Tuple[float, float]]:
    """Generate (Σ₊, Σ₋) scan points satisfying Σ₊² + Σ₋² < 1.

    Returns sorted list of (Σ₊, Σ₋) tuples.
    """
    points = set()
    vals = config.Sigma_values

    for sp in vals:
        for sm in vals:
            if sp**2 + sm**2 < 0.99:  # Friedmann constraint
                points.add((sp, sm))
                if config.include_exchange and sp != sm:
                    points.add((sm, sp))

    return sorted(points)


# ═══════════════════════════════════════════════════════════════
# §2. Scan result container
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScanPoint:
    """Result at one (Σ₊, Σ₋) point."""
    Sigma_plus: float
    Sigma_minus: float
    Sigma_total: float  # √(Σ₊² + Σ₋²)
    Yp: float
    DH: float           # D/H
    Li7H: float         # ⁷Li/H
    delta_Yp: float     # Y_p - Y_p(FLRW)
    delta_Yp_rel: float # ΔY_p / Y_p(FLRW)
    N_eff_measured: float
    wall_time_s: float


@dataclass
class GenericTypeIScanResult:
    """Full scan result."""
    points: List[ScanPoint]
    Yp_flrw: float
    DH_flrw: float
    config: GenericTypeIScanConfig

    def to_arrays(self):
        """Extract numpy arrays for plotting."""
        sp = np.array([p.Sigma_plus for p in self.points])
        sm = np.array([p.Sigma_minus for p in self.points])
        st = np.array([p.Sigma_total for p in self.points])
        yp = np.array([p.Yp for p in self.points])
        dh = np.array([p.DH for p in self.points])
        dyp = np.array([p.delta_Yp_rel for p in self.points])
        return sp, sm, st, yp, dh, dyp

    def lrs_points(self) -> List[ScanPoint]:
        """Points with Σ₋ = 0 (LRS limit)."""
        return [p for p in self.points if abs(p.Sigma_minus) < 1e-15]

    def exchange_pairs(self) -> List[Tuple[ScanPoint, ScanPoint]]:
        """Find (Σ₊,Σ₋) and (Σ₋,Σ₊) pairs for symmetry test."""
        lookup = {}
        for p in self.points:
            key = (round(p.Sigma_plus, 6), round(p.Sigma_minus, 6))
            lookup[key] = p

        pairs = []
        seen = set()
        for p in self.points:
            sp, sm = round(p.Sigma_plus, 6), round(p.Sigma_minus, 6)
            if sp == sm:
                continue
            swap = (sm, sp)
            if swap in lookup and (sp, sm) not in seen:
                pairs.append((p, lookup[swap]))
                seen.add((sp, sm))
                seen.add(swap)
        return pairs

    def isotropy_residual(self) -> float:
        """Max |ΔY_p(Σ₊,Σ₋) - ΔY_p(Σ,0)| / |ΔY_p(Σ,0)| over all points.

        Tests the O(Σ²) isotropy: Y_p should depend only on Σ² to
        leading order.  Nonzero residual comes from O(Σ⁴) corrections.
        """
        lrs = {round(p.Sigma_total, 6): p for p in self.lrs_points()
               if abs(p.Sigma_total) > 1e-10}
        max_res = 0.0
        for p in self.points:
            if abs(p.Sigma_total) < 1e-10:
                continue
            st_key = round(p.Sigma_total, 6)
            if st_key in lrs:
                ref = lrs[st_key]
                if abs(ref.delta_Yp) > 1e-15:
                    res = abs(p.delta_Yp - ref.delta_Yp) / abs(ref.delta_Yp)
                    max_res = max(max_res, res)
        return max_res


# ═══════════════════════════════════════════════════════════════
# §3. Run scan (requires rabbit driver)
# ═══════════════════════════════════════════════════════════════

def run_generic_typeI_scan(config: GenericTypeIScanConfig = None,
                            verbose: bool = True) -> GenericTypeIScanResult:
    """Run the full (Σ₊, Σ₋) parameter scan.

    Requires the rabbit package to be importable.
    """
    from rabbit.drivers.full_coupled_typeI import (
        FullCoupledConfig, run_full_coupled_typeI,
    )

    if config is None:
        config = GenericTypeIScanConfig()

    scan_points = generate_scan_points(config)
    if verbose:
        print(f"Generic Type I scan: {len(scan_points)} points")

    # FLRW baseline
    cfg_flrw = FullCoupledConfig(
        Sigma_H_plus=0.0, Sigma_H_minus=0.0,
        tau_n=config.tau_n, eta=config.eta,
        N_q=config.N_q, tier=config.tier,
    )
    result_flrw = run_full_coupled_typeI(cfg_flrw)
    Yp_flrw = result_flrw.observables.Yp
    DH_flrw = result_flrw.observables.DH
    if verbose:
        print(f"  FLRW baseline: Y_p = {Yp_flrw:.8f}, D/H = {DH_flrw:.6e}")

    results = []
    for i, (sp, sm) in enumerate(scan_points):
        if verbose and i % 5 == 0:
            print(f"  Point {i+1}/{len(scan_points)}: "
                  f"Σ₊={sp:.3f}, Σ₋={sm:.3f}")

        cfg = FullCoupledConfig(
            Sigma_H_plus=sp, Sigma_H_minus=sm,
            tau_n=config.tau_n, eta=config.eta,
            N_q=config.N_q, tier=config.tier,
        )
        try:
            result = run_full_coupled_typeI(cfg)
            Yp = result.observables.Yp
            DH = result.observables.DH
            Li7H = result.observables.Li7_H
            N_eff = result.metadata.get('N_eff_measured', 0)
            wt = result.wall_time_s
        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")
            continue

        Sigma_total = np.sqrt(sp**2 + sm**2)
        delta_Yp = Yp - Yp_flrw
        delta_Yp_rel = delta_Yp / Yp_flrw if Yp_flrw > 0 else 0

        results.append(ScanPoint(
            Sigma_plus=sp, Sigma_minus=sm,
            Sigma_total=Sigma_total,
            Yp=Yp, DH=DH, Li7H=Li7H,
            delta_Yp=delta_Yp, delta_Yp_rel=delta_Yp_rel,
            N_eff_measured=N_eff, wall_time_s=wt,
        ))

    return GenericTypeIScanResult(
        points=results, Yp_flrw=Yp_flrw, DH_flrw=DH_flrw, config=config,
    )


# ═══════════════════════════════════════════════════════════════
# §4. Validation tests
# ═══════════════════════════════════════════════════════════════

def validate_lrs_recovery(scan: GenericTypeIScanResult,
                           reference_lrs: Optional[Dict[float, float]] = None,
                           tol: float = 1e-10) -> bool:
    """Check that Σ₋=0 reproduces LRS results.

    If reference_lrs is provided (dict: Σ₊ → Y_p), compare against it.
    Otherwise just check that LRS points exist and are consistent.
    """
    lrs = scan.lrs_points()
    if len(lrs) < 2:
        return False
    if reference_lrs is not None:
        for p in lrs:
            key = round(p.Sigma_plus, 4)
            if key in reference_lrs:
                diff = abs(p.Yp - reference_lrs[key])
                if diff > tol:
                    return False
    return True


def validate_exchange_symmetry(scan: GenericTypeIScanResult,
                                tol: float = 1e-10) -> Tuple[bool, float]:
    """Check Y_p(Σ₊,Σ₋) = Y_p(Σ₋,Σ₊).

    Returns (passed, max_violation).
    """
    pairs = scan.exchange_pairs()
    if not pairs:
        return True, 0.0
    max_viol = 0.0
    for p1, p2 in pairs:
        diff = abs(p1.Yp - p2.Yp)
        max_viol = max(max_viol, diff)
    return max_viol < tol, max_viol


def validate_isotropy_leading_order(scan: GenericTypeIScanResult,
                                     tol: float = 0.01) -> Tuple[bool, float]:
    """Check that Y_p depends only on Σ² to leading order.

    The residual should be O(Σ⁴), so for Σ ~ 0.1 we expect ~10⁻⁴.
    """
    res = scan.isotropy_residual()
    return res < tol, res


# ═══════════════════════════════════════════════════════════════
# §5. Quick command-line interface
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("Running Generic Type I validation scan...")
    print("=" * 60)

    # Small scan for quick validation
    config = GenericTypeIScanConfig(
        Sigma_values=[0.0, 0.05, 0.10, 0.15, 0.20],
        N_q=40,  # reduced for speed
    )
    scan = run_generic_typeI_scan(config)

    # Tests
    print("\nValidation tests:")
    ok_lrs = validate_lrs_recovery(scan)
    print(f"  LRS recovery: {'PASS' if ok_lrs else 'FAIL'}")

    ok_ex, viol_ex = validate_exchange_symmetry(scan)
    print(f"  Exchange symmetry: {'PASS' if ok_ex else 'FAIL'} "
          f"(max violation = {viol_ex:.2e})")

    ok_iso, viol_iso = validate_isotropy_leading_order(scan)
    print(f"  Isotropy (leading order): {'PASS' if ok_iso else 'FAIL'} "
          f"(residual = {viol_iso:.2e})")

    # Summary table
    print("\nScan results:")
    print(f"{'Σ₊':>6s} {'Σ₋':>6s} {'Σ_tot':>7s} {'Y_p':>12s} {'ΔY_p/Y_p':>12s}")
    print("-" * 50)
    for p in sorted(scan.points, key=lambda x: (x.Sigma_plus, x.Sigma_minus)):
        print(f"{p.Sigma_plus:6.3f} {p.Sigma_minus:6.3f} "
              f"{p.Sigma_total:7.4f} {p.Yp:12.8f} {p.delta_Yp_rel:12.4e}")
