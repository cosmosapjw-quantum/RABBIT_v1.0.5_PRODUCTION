"""PR-T3B-PF #10: cross-mode FLRW ``N_eff`` diagnostic.

Compares the FLRW (``Σ=0``) ``N_eff`` produced by each available
collision-mode preflight in
``rabbit.jax.driver_typeI_full_boltzmann``:

* ``collisionless`` (tier-1): no collision source (sanity floor).
* ``spectral_relaxation_preflight`` (tier-2): AP-form relaxation
  with spectral damping.
* ``projected_physical_preflight`` (tier-2): AP-form
  state-dependent projected source + damping.
* ``jax_kernel_preflight`` (tier-2): full Hannestad-Madsen ν-e +
  pair JAX kernel (the structural target for canonical PR-T3B
  but currently wrong-sign due to the deferred T-rescaling fix).

The locked relative ordering at the bounded preflight grid
``(N_mu, N_q) = (4, 6)``:

    spectral_relaxation_preflight   → N_eff ≈ 3.0345  (closest to Mangano 3.044)
    projected_physical_preflight    → N_eff ≈ 3.0307
    collisionless                   → N_eff ≈ 3.0107  (no collision sanity floor)
    jax_kernel_preflight            → N_eff ≈ 2.9934  (anti-heating bug from T_e=T_ν approx)

The AP-form modes already lie within ``0.014`` of Mangano 2005's
``N_eff = 3.044`` — much closer than the full-kernel mode.  This
is the calibration signal future canonical PR-T3B work needs to
target: the AP-form gives ``+0.034``, the canonical (full kernel
+ correct T-rescaling) should give ``+0.044``, so the residual
gap is ``~0.01``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def cross_mode_results() -> dict:
    """Runs the FLRW solve for every available collision mode once
    and shares the outputs across the parametrized comparisons."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    results: dict = {}
    cases = [
        ("collisionless", 1),
        ("spectral_relaxation_preflight", 2),
        ("projected_physical_preflight", 2),
        ("jax_kernel_preflight", 2),
        ("ap_unified_preflight", 2),
    ]
    for mode, tier in cases:
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=0.0,
            N_mu=4,
            N_q=6,
            correction_level=0,
            n_reactions=12,
            collision_mode=mode,
            thermo_tier=tier,
            rtol=1e-6,
            atol=1e-8,
            max_steps=256,
            event_refine_steps=12,
        )
        r = run_full_boltzmann_jax(cfg)
        assert r.success, f"FLRW {mode} run failed"
        results[mode] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
        }
    return results


# Locked baselines (tied to Aug-2025 measurement at the cubic-spline
# swap state, extended Apr-2026 with ap_unified canonical milestone).
# These are MEASURED values, not target values.
_LOCKED = {
    "collisionless": (0.2424, 3.0107),
    "spectral_relaxation_preflight": (0.2414, 3.0345),
    "projected_physical_preflight": (0.2415, 3.0307),
    "jax_kernel_preflight": (0.2417, 2.9934),
    "ap_unified_preflight": (0.2414, 3.0345),
}
_TOLERANCE_YP = 5e-4
_TOLERANCE_NEFF = 5e-3


@pytest.mark.parametrize(
    "mode",
    [
        "collisionless",
        "spectral_relaxation_preflight",
        "projected_physical_preflight",
        "jax_kernel_preflight",
        "ap_unified_preflight",
    ],
)
def test_per_mode_flrw_neff_baseline(cross_mode_results: dict, mode: str) -> None:
    """Each mode's FLRW ``N_eff`` is locked at its measured baseline
    (with a generous ``±5e-3`` envelope so cross-platform float
    reduction noise does not flake the test)."""
    measured = cross_mode_results[mode]["N_eff"]
    expected = _LOCKED[mode][1]
    assert abs(measured - expected) < _TOLERANCE_NEFF, (
        f"{mode} FLRW N_eff drifted: measured {measured:.6f}, "
        f"expected {expected:.4f} ± {_TOLERANCE_NEFF}"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "collisionless",
        "spectral_relaxation_preflight",
        "projected_physical_preflight",
        "jax_kernel_preflight",
        "ap_unified_preflight",
    ],
)
def test_per_mode_flrw_yp_baseline(cross_mode_results: dict, mode: str) -> None:
    """Each mode's FLRW ``Y_p`` is locked at its measured baseline."""
    measured = cross_mode_results[mode]["Yp"]
    expected = _LOCKED[mode][0]
    assert abs(measured - expected) < _TOLERANCE_YP, (
        f"{mode} FLRW Y_p drifted: measured {measured:.6f}, "
        f"expected {expected:.4f} ± {_TOLERANCE_YP}"
    )


def test_ap_unified_matches_spectral_at_flrw(cross_mode_results: dict) -> None:
    """Canonical milestone: ``ap_unified_preflight`` reaches
    spectral_relaxation's FLRW fidelity (gap ``~0.0095`` to
    Mangano) without spectral's grid-breaking source_scale clip."""
    sp = cross_mode_results["spectral_relaxation_preflight"]["N_eff"]
    apu = cross_mode_results["ap_unified_preflight"]["N_eff"]
    # ap_unified should match spectral at FLRW within ~5e-4
    assert abs(sp - apu) < 5e-4, (
        f"ap_unified FLRW {apu:.6f} drifted from spectral {sp:.6f}"
    )


def test_ap_form_modes_closer_to_mangano_than_jax_kernel(
    cross_mode_results: dict,
) -> None:
    """Documents the central PR-T3B finding: the AP-form preflight
    modes (spectral_relaxation, projected_physical) currently
    produce ``N_eff`` closer to Mangano 2005's ``3.044`` than the
    full Hannestad-Madsen JAX kernel mode.  This is because the
    JAX-kernel mode treats ``T_e = T_ν`` inside the integrand
    (the q-grid remap fix from PR-T3B-PF #6 / #8 produces a stiff
    ∂C/∂T manifold that Rodas5P cannot handle without IMEX or AP
    splitting).  The AP-form modes capture the equilibrating
    physics directly via the relaxation factor.

    Locking the relative ordering ensures any future change keeps
    the calibration signal pointing in the right direction."""
    MANGANO = 3.044
    spectral_gap = abs(cross_mode_results["spectral_relaxation_preflight"]["N_eff"] - MANGANO)
    projected_gap = abs(cross_mode_results["projected_physical_preflight"]["N_eff"] - MANGANO)
    kernel_gap = abs(cross_mode_results["jax_kernel_preflight"]["N_eff"] - MANGANO)

    assert spectral_gap < kernel_gap, (
        f"spectral_relaxation gap to Mangano ({spectral_gap:.4f}) "
        f"should be less than jax_kernel gap ({kernel_gap:.4f})"
    )
    assert projected_gap < kernel_gap, (
        f"projected_physical gap to Mangano ({projected_gap:.4f}) "
        f"should be less than jax_kernel gap ({kernel_gap:.4f})"
    )

    # Both AP-form modes should be within ~0.02 of Mangano (loose
    # preflight bound; canonical target is < 0.005).
    assert spectral_gap < 0.02, (
        f"spectral_relaxation gap to Mangano widened: {spectral_gap:.4f}"
    )
    assert projected_gap < 0.02, (
        f"projected_physical gap to Mangano widened: {projected_gap:.4f}"
    )


def test_collisionless_is_tier1_baseline(cross_mode_results: dict) -> None:
    """Collisionless (tier-1, no collision source) sits between the
    AP-form modes and the kernel mode in the FLRW ``N_eff``
    spectrum at the current preflight grid.  This is a sanity
    check: with no collision source, the system follows the
    standard adiabatic ``T_ν / T_γ = (4/11)^{1/3}`` ratio after
    e+ e- annihilation, giving ``N_eff ≈ 3.0`` modulo the bounded
    grid effects."""
    n_eff = cross_mode_results["collisionless"]["N_eff"]
    assert 3.00 < n_eff < 3.02, (
        f"collisionless N_eff baseline drifted: {n_eff:.6f}"
    )
