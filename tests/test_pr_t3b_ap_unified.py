"""PR-T3B canonical #1: AP-form unification mode (``ap_unified_preflight``).

The canonical mode combines THREE mechanisms:

1. Full Fermi-Dirac ``psi_target = (f_target - f_eq) / f_eq``
   from ``spectral_relaxation_preflight`` (anisotropy-aware
   target shape).
2. ``c_psi = gamma_over_H * (psi_target - psi_actual)``
   structure decomposed into ``source_raw`` and ``damping_raw``
   per spectral_relaxation, with the damping projection that
   subtracts the plasma-coupled component (preserves energy
   neutrality of the damping mechanism).
3. Soft total-rate enforcement on ``source_raw`` (scales it so
   the plasma energy moment matches the canonical Mangano
   rate).  Uses a softer clip ``[0.1, 5]`` and a 1% activation
   floor to avoid the ill-conditioning that breaks
   ``spectral_relaxation`` at grids above ``(4, 6)``.

**Canonical milestone reached**: this combination achieves
**both** anisotropy stability (spread ``< 1e-3`` across the
``Σ_H`` sweep, passing the canonical PR-T3D §5 gate) **and**
grid scaling (spread ``< 1e-4`` across the ``(N_mu, N_q)``
sweep).  FLRW gap to Mangano remains ``~0.0095`` — the
documented AP-form model approximation limit.

Measurements at the bounded preflight surface:

* FLRW ``Σ_H = 0``: ``N_eff ≈ 3.0345`` (gap ``+0.0095``)
* Grid scaling: fully converged across
  ``(4,6) / (8,12) / (12,20)`` with spread ``< 1e-4``.
* Anisotropy: spread ``< 1e-3`` across
  ``Σ_H ∈ {0, 0.05, 0.10}`` (passes canonical gate).
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def ap_unified_results() -> dict:
    """Per-grid + per-σ ``ap_unified_preflight`` FLRW + anisotropic
    measurements."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    results: dict = {}
    cases = [
        ("flrw_4_6", 0.0, 4, 6),
        ("flrw_8_12", 0.0, 8, 12),
        ("flrw_12_20", 0.0, 12, 20),
        ("aniso_05", 0.05, 4, 6),
        ("aniso_10", 0.10, 4, 6),
    ]
    for label, sigma, N_mu, N_q in cases:
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=sigma,
            N_mu=N_mu,
            N_q=N_q,
            correction_level=0,
            n_reactions=12,
            collision_mode="ap_unified_preflight",
            thermo_tier=2,
            rtol=1e-6,
            atol=1e-8,
            max_steps=512,
            event_refine_steps=12,
        )
        r = run_full_boltzmann_jax(cfg)
        assert r.success, f"ap_unified failed at {label}"
        results[label] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
            "metadata": r.metadata,
        }
    return results


def test_ap_unified_flrw_baseline(ap_unified_results: dict) -> None:
    """FLRW ``Σ_H = 0`` ``N_eff`` baseline locked at ``3.0345 ± 5e-3``.

    Matches ``spectral_relaxation`` (3.0345) at FLRW: the soft
    rate enforcement recovers the same fidelity as spectral's
    aggressive scaling, while the softer clip avoids the
    ill-conditioning that breaks spectral above ``(4, 6)``.
    """
    n_eff = ap_unified_results["flrw_4_6"]["N_eff"]
    assert abs(n_eff - 3.0345) < 5e-3, (
        f"ap_unified FLRW N_eff drifted: {n_eff:.6f}"
    )


def test_ap_unified_grid_converged(ap_unified_results: dict) -> None:
    """The mode is grid-converged like ``projected_physical``.

    Locked at spread ``< 1e-4`` across
    ``(N_mu, N_q) ∈ {(4, 6), (8, 12), (12, 20)}``.
    """
    n_effs = [
        ap_unified_results[label]["N_eff"]
        for label in ("flrw_4_6", "flrw_8_12", "flrw_12_20")
    ]
    spread = max(n_effs) - min(n_effs)
    assert spread < 1e-4, (
        f"ap_unified grid spread widened: {spread:.6f}"
    )


def test_ap_unified_anisotropy_robust(ap_unified_results: dict) -> None:
    """**Canonical milestone**: ``ap_unified`` is anisotropy-robust
    to ``< 1e-3`` spread across ``Σ_H ∈ {0, 0.05, 0.10}``,
    passing the canonical PR-T3D §5 stability gate.

    This is the central canonical achievement of the AP-form
    unification: the rate enforcement (sigma-independent total
    rate by construction) + damping projection (energy-neutral
    damping) combination recovers spectral_relaxation's
    anisotropy stability inside the ``ap_unified`` mode without
    spectral's grid-breaking source_scale clip.
    """
    n_eff_0 = ap_unified_results["flrw_4_6"]["N_eff"]
    n_eff_05 = ap_unified_results["aniso_05"]["N_eff"]
    n_eff_10 = ap_unified_results["aniso_10"]["N_eff"]
    n_effs = [n_eff_0, n_eff_05, n_eff_10]
    spread = max(n_effs) - min(n_effs)
    # Canonical PR-T3D §5 gate.  Locked at 5e-4 (measured ~7e-5)
    # to leave room for cross-platform float reduction noise.
    assert spread < 5e-4, (
        f"ap_unified anisotropy spread widened beyond canonical "
        f"PR-T3D §5 gate: {spread:.6f}"
    )


def test_ap_unified_metadata_contracts(ap_unified_results: dict) -> None:
    """``ap_unified_preflight`` carries the canonical ``ap_unified``
    contract strings."""
    from rabbit.thermo.nudec_tables import _C_RATE

    md = ap_unified_results["flrw_4_6"]["metadata"]
    assert md["collision_scope_contract"] == "ap_unified_preflight_v1"
    assert md["jacobian_payload_contract"] == "ap_unified_transport_plus_active_plus_3T_low_rank_v1"
    assert md["thermo_tier"] == 2
    assert md["mangano_energy_transfer_C_rate"] == pytest.approx(_C_RATE)
    assert md["collision_rate_prefactor_source"] == "rabbit.thermo.nudec_tables._C_RATE"
    assert md["rate_enforcement_sign_contract"] == (
        "positive_scale_requires_matching_energy_moment_sign_v1"
    )


def test_ap_unified_matches_spectral_at_flrw(
    ap_unified_results: dict,
) -> None:
    """``ap_unified`` recovers ``spectral_relaxation``'s FLRW
    fidelity (gap ``~0.0095``) without spectral's grid-breaking
    source_scale clip."""
    n_eff_unified = ap_unified_results["flrw_4_6"]["N_eff"]
    SPECTRAL_FLRW = 3.034495
    # Should match spectral at FLRW within ~5e-4.
    assert abs(n_eff_unified - SPECTRAL_FLRW) < 5e-4, (
        f"ap_unified FLRW N_eff {n_eff_unified:.6f} drifted from "
        f"spectral_relaxation baseline {SPECTRAL_FLRW:.6f}"
    )
