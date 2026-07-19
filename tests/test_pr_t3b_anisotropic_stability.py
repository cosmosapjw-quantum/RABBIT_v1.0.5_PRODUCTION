"""PR-T3B-PF #12: anisotropic ``N_eff`` stability sweep.

The PR-T3D phase prompt §5 requires that tier-3 ``N_eff`` move
``< 1e-3`` across ``Σ_H ∈ {0, 0.1, 0.3}`` once the canonical
surface is in place.  This module measures the *current*
preflight behaviour of the converged ``projected_physical_preflight``
AP-form mode so the canonical PR-T3B work has a clear baseline:

* ``Σ_H = 0.00``: ``N_eff = 3.030738``  (FLRW reference)
* ``Σ_H = 0.05``: ``N_eff = 2.753109``  (drops by ``-0.28``)
* ``Σ_H = 0.10``: ``N_eff = 2.487129``  (drops by ``-0.54``)
* ``Σ_H = 0.30``: solver fails (max_steps exceeded)

The N_eff drop with anisotropy is structural: the AP-form
collision wrapper does not properly equilibrate the bank state
when shear-driven anisotropic transport pushes the per-ray
distributions away from the symmetric baseline.  This is the
**second** calibration signal canonical PR-T3B work needs: even
if the ``Σ=0`` gap to Mangano (``~0.013``) is closed via an
AP/IMEX hybrid, the anisotropic regime requires additional work
(probably species-dependent damping coefficients tuned against a
linearised-PSTF reference).

The locked test bundle is intentionally **loose** — it captures
the current state with a wide envelope so any further drift
flags a regression while leaving room for canonical-track
improvements.  Tightening to the canonical ``< 1e-3`` target is
deferred to PR-T3B canonical.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def projected_physical_anisotropic_results() -> dict:
    """Per-Σ_H ``run_full_boltzmann_jax`` results on the bounded
    grid ``(N_mu=4, N_q=6)``.  Σ_H = 0.30 is excluded because the
    preflight solver budget is exceeded; smaller values are
    locked individually below."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    results: dict = {}
    for sigma in (0.0, 0.05, 0.10):
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=sigma,
            N_mu=4,
            N_q=6,
            correction_level=0,
            n_reactions=12,
            collision_mode="projected_physical_preflight",
            thermo_tier=2,
            rtol=1e-6,
            atol=1e-8,
            max_steps=512,
            event_refine_steps=12,
        )
        r = run_full_boltzmann_jax(cfg)
        assert r.success, (
            f"projected_physical anisotropic run failed at Σ_H={sigma}"
        )
        results[sigma] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
        }
    return results


# Locked per-σ baselines from the August-2025 measurement.  Loose
# envelopes (~5%) reflect the current preflight imperfection; the
# canonical PR-T3B target is < 1e-3 spread across this sweep.
_LOCKED_PROJECTED_PHYSICAL = {
    0.0: (0.2415, 3.0307),
    0.05: (0.2385, 2.7531),
    0.10: (0.2352, 2.4871),
}
_TOL_YP = 5e-3
_TOL_N_EFF = 5e-2


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.10])
def test_projected_physical_anisotropic_n_eff_baseline(
    projected_physical_anisotropic_results: dict, sigma: float
) -> None:
    """Per-σ_H ``N_eff`` baseline lock.  Loose envelopes (``5e-2``)
    so the test is not flaky across float reduction noise; the
    canonical PR-T3B target is ``< 1e-3`` across this sweep."""
    measured = projected_physical_anisotropic_results[sigma]["N_eff"]
    expected = _LOCKED_PROJECTED_PHYSICAL[sigma][1]
    assert abs(measured - expected) < _TOL_N_EFF, (
        f"projected_physical at Σ_H={sigma}: N_eff = {measured:.6f} "
        f"drifted from baseline {expected:.4f} ± {_TOL_N_EFF}"
    )


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.10])
def test_projected_physical_anisotropic_yp_baseline(
    projected_physical_anisotropic_results: dict, sigma: float
) -> None:
    """Per-σ_H ``Y_p`` baseline lock."""
    measured = projected_physical_anisotropic_results[sigma]["Yp"]
    expected = _LOCKED_PROJECTED_PHYSICAL[sigma][0]
    assert abs(measured - expected) < _TOL_YP, (
        f"projected_physical at Σ_H={sigma}: Y_p = {measured:.6f} "
        f"drifted from baseline {expected:.4f} ± {_TOL_YP}"
    )


def test_anisotropic_n_eff_drops_monotonically_with_sigma(
    projected_physical_anisotropic_results: dict,
) -> None:
    """The N_eff diagnostic decreases monotonically with σ_H over
    the bounded sweep.  This is the **physically wrong** direction
    on the production-grade canonical surface (incomplete
    decoupling should be roughly σ-independent), but it captures
    the current preflight behaviour: AP-form damping does not
    track the shear-driven transport asymmetry, so T_νₑ drifts
    away from T_νₓ in a way the diagnostic conflates with reduced
    heating.  Locking the monotonic drop ensures the canonical
    work knows it must REVERSE this trend."""
    n_effs = [
        projected_physical_anisotropic_results[s]["N_eff"]
        for s in (0.0, 0.05, 0.10)
    ]
    # n_eff[0.0] > n_eff[0.05] > n_eff[0.10]
    assert n_effs[0] > n_effs[1] > n_effs[2], (
        f"projected_physical anisotropic ordering violated: {n_effs}"
    )


def test_anisotropic_n_eff_spread_documents_canonical_gap(
    projected_physical_anisotropic_results: dict,
) -> None:
    """The N_eff spread across ``Σ_H ∈ {0, 0.05, 0.10}`` is
    currently ``~0.54`` — far above the canonical target of
    ``< 1e-3``.  This locks the gap so canonical PR-T3B has a
    measurable destination."""
    n_effs = [
        projected_physical_anisotropic_results[s]["N_eff"]
        for s in (0.0, 0.05, 0.10)
    ]
    spread = max(n_effs) - min(n_effs)
    # Lock at 0.4 < spread < 0.7 to flag any further degradation
    # without being so tight it flakes on small numerical noise.
    assert 0.4 < spread < 0.7, (
        f"projected_physical anisotropic spread drifted: {spread:.4f}"
    )
    # Also confirm the spread is much larger than canonical target.
    canonical_n_eff_tol = 1e-3
    assert spread > 100 * canonical_n_eff_tol, (
        f"spread {spread:.4f} unexpectedly close to canonical target "
        f"{canonical_n_eff_tol:g}: this would mean the gap closed by "
        "accident -- verify the canonical work has actually landed."
    )


# ---------------------------------------------------------------------------
# AP-form variant comparison: spectral_relaxation vs projected_physical
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spectral_relaxation_anisotropic_results() -> dict:
    """Per-σ_H ``spectral_relaxation_preflight`` results on the same
    bounded grid.  Captured separately so the cross-mode comparison
    test can reason about both AP-form variants without re-running
    the projected_physical fixture."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    results: dict = {}
    for sigma in (0.0, 0.05, 0.10):
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=sigma,
            N_mu=4,
            N_q=6,
            correction_level=0,
            n_reactions=12,
            collision_mode="spectral_relaxation_preflight",
            thermo_tier=2,
            rtol=1e-6,
            atol=1e-8,
            max_steps=512,
            event_refine_steps=12,
        )
        r = run_full_boltzmann_jax(cfg)
        assert r.success, (
            f"spectral_relaxation anisotropic run failed at Σ_H={sigma}"
        )
        results[sigma] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
        }
    return results


def test_spectral_relaxation_anisotropy_robust(
    spectral_relaxation_anisotropic_results: dict,
) -> None:
    """``spectral_relaxation_preflight`` is **anisotropy-robust** —
    its N_eff is invariant to Σ_H to floating-point reduction
    order.

    Measured baselines:

    * ``Σ_H = 0.00``: ``N_eff = 3.034495``
    * ``Σ_H = 0.05``: ``N_eff = 3.034520``
    * ``Σ_H = 0.10``: ``N_eff = 3.034562``

    The spread is ``~7e-5``, **already passing the canonical
    PR-T3D §5 stability gate** of ``< 1e-3``.  This is the
    **decisive** finding: the AP-form damping closure
    (spectral_relaxation) handles shear-driven transport
    asymmetry correctly, while the AP-form source closure
    (projected_physical) does not.

    For canonical PR-T3B work this identifies
    ``spectral_relaxation`` as the better AP-form starting point
    for anisotropic regimes.
    """
    n_effs = [
        spectral_relaxation_anisotropic_results[s]["N_eff"]
        for s in (0.0, 0.05, 0.10)
    ]
    spread = max(n_effs) - min(n_effs)
    # Already passes the canonical < 1e-3 gate.  Lock at 5e-4 to
    # leave headroom for cross-platform float reduction noise.
    assert spread < 5e-4, (
        f"spectral_relaxation anisotropic spread widened beyond "
        f"the canonical < 1e-3 gate: {spread:.6f}"
    )


def test_spectral_relaxation_strictly_more_robust_than_projected_physical(
    spectral_relaxation_anisotropic_results: dict,
    projected_physical_anisotropic_results: dict,
) -> None:
    """Cross-mode comparison: ``spectral_relaxation`` is at least
    **three orders of magnitude** more anisotropy-robust than
    ``projected_physical``.  This is the central calibration
    signal for picking the canonical AP-form variant."""
    sp_neffs = [
        spectral_relaxation_anisotropic_results[s]["N_eff"]
        for s in (0.0, 0.05, 0.10)
    ]
    pp_neffs = [
        projected_physical_anisotropic_results[s]["N_eff"]
        for s in (0.0, 0.05, 0.10)
    ]
    sp_spread = max(sp_neffs) - min(sp_neffs)
    pp_spread = max(pp_neffs) - min(pp_neffs)
    ratio = pp_spread / max(sp_spread, 1e-300)
    assert ratio > 1000, (
        f"spectral_relaxation should be > 1000x more robust than "
        f"projected_physical, got ratio {ratio:.2e} "
        f"(sp_spread={sp_spread:.6f}, pp_spread={pp_spread:.6f})"
    )


def test_spectral_relaxation_grid_4_6_baseline_lock() -> None:
    """``spectral_relaxation_preflight`` is anisotropy-robust at the
    bounded ``(N_mu=4, N_q=6)`` grid but **fails** at larger grids
    even with extra ``max_steps`` budget.  This is a separate
    canonical blocker beyond the FLRW gap and the
    projected_physical anisotropy issue.

    Empirical sweep:

    * ``(4, 6)``  max_steps=256  → ``N_eff = 3.034495`` (pass)
    * ``(4, 6)``  max_steps=1024 → ``N_eff = 3.034495`` (pass; same value)
    * ``(8, 12)`` max_steps=1024 → FAILED
    * ``(8, 12)`` max_steps=2048 → FAILED
    * ``(12, 20)`` max_steps=2048 → FAILED

    The failure mode is not a budget issue (doubling max_steps
    does not help); something else in the spectral_relaxation
    closure exceeds an internal tolerance/conditioning limit at
    larger grids.  Locking the smallest-grid baseline so canonical
    PR-T3B work can compare any future fix.
    """
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="spectral_relaxation_preflight",
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=1024,
        event_refine_steps=12,
    )
    r = run_full_boltzmann_jax(cfg)
    assert r.success
    n_eff = float(r.metadata["N_eff_measured"])
    # Locked at 3.034495 +/- 1e-4
    assert abs(n_eff - 3.034495) < 1e-4, (
        f"spectral_relaxation (4,6) baseline drifted: {n_eff:.6f}"
    )
