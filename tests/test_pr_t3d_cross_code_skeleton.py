"""PR-T3D-PF cross-code skeleton: structure-validate the cross-code
fixture and report the gap between the current
``jax_kernel_preflight`` FLRW baseline and the published Mangano /
LASAGNA / FortEPiaNO / PRIMAT-AC2024 reference values.

This is a **diagnostic / reporting** test, not the canonical parity
gate.  The phase-prompt PR-T3D acceptance is
``|ΔY_p| < 5e-4`` and ``|N_eff - 3.044| < 0.005`` — neither holds
on the current preflight surface (FLRW N_eff baseline ≈ 2.993).
The test passes today by enforcing only loose preflight bounds and
records the measured gap so future canonical work has a baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tier3_cross_code.json"


@pytest.fixture(scope="module")
def cross_code_targets() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Fixture structural integrity
# ---------------------------------------------------------------------------


def test_cross_code_fixture_has_required_codes(cross_code_targets: dict) -> None:
    """Every fixture key the canonical PR-T3D parity tests will need
    must be present and well-formed."""
    required_codes = ("lasagna", "fortepiano", "primat_ac2024", "mangano_2005")
    for code in required_codes:
        assert code in cross_code_targets, (
            f"cross-code fixture missing required entry: {code!r}"
        )
        entry = cross_code_targets[code]
        assert "N_eff" in entry, f"{code}: missing N_eff"
        assert "reference" in entry, f"{code}: missing reference"
        # Y_p may be null for mangano (not quoted at same precision).
        assert "Y_p" in entry, f"{code}: missing Y_p key"


def test_cross_code_fixture_metadata_fields(cross_code_targets: dict) -> None:
    """The ``_metadata`` block carries the canonical-tolerance target
    and the preflight status note used by the audit docs."""
    md = cross_code_targets["_metadata"]
    assert md["fixture_format_version"] == "v1"
    tol = md["tolerance_target_canonical"]
    assert abs(tol["Y_p_abs"] - 5e-4) < 1e-12
    assert abs(tol["N_eff_abs"] - 5e-3) < 1e-12
    # Preflight status string must exist and reference the audit docs
    # so the gap is traceable from the fixture alone.
    assert "preflight_status" in md
    assert "audit" in md["preflight_status"].lower()


def test_cross_code_fixture_preflight_measurements_present(
    cross_code_targets: dict,
) -> None:
    """The ``_metadata.preflight_measurements`` block records the
    measured FLRW values from the current preflight surface so
    the canonical PR-T3D work has a reproducible baseline next to
    the published cross-code targets."""
    md = cross_code_targets["_metadata"]
    pm = md["preflight_measurements"]
    for mode in (
        "spectral_relaxation_preflight",
        "projected_physical_preflight",
        "jax_kernel_preflight",
        "jax_kernel_remap_preconditioned_preflight",
        "collisionless_tier1",
    ):
        assert mode in pm, (
            f"preflight_measurements missing mode: {mode!r}"
        )
        # projected_physical carries grid-resolved entries; the others
        # carry a single ``N_eff`` field.
        if mode == "projected_physical_preflight":
            for grid_key in ("_grid_4_6", "_grid_8_12", "_grid_12_20"):
                assert grid_key in pm[mode]
                assert "N_eff" in pm[mode][grid_key]
        elif mode == "jax_kernel_remap_preconditioned_preflight":
            for grid_key in ("_grid_4_6", "_grid_4_8", "_grid_4_10"):
                assert grid_key in pm[mode]
                assert "N_eff" in pm[mode][grid_key]
        else:
            assert "N_eff" in pm[mode]


def test_cross_code_fixture_ap_unified_milestone_block_present(
    cross_code_targets: dict,
) -> None:
    """The ``ap_unified_preflight`` block records the PR-T3B canonical
    milestone numbers (the candidate now publicly dispatchable as
    ``backend='jax_ap_unified_tier3'``).  Locks: 3 grid entries, an
    anisotropic sweep, the documented Mangano gap, and the gate-passed
    flag so a registry-driven canonical readiness check can read it
    directly."""
    pm = cross_code_targets["_metadata"]["preflight_measurements"]
    assert "ap_unified_preflight" in pm
    apu = pm["ap_unified_preflight"]
    for gk in ("_grid_4_6", "_grid_8_12", "_grid_12_20"):
        assert gk in apu and "N_eff" in apu[gk] and "Y_p" in apu[gk]
    assert apu["_grid_spread_n_eff"] < 1e-4
    sweep = apu["_anisotropic_sweep_4_6"]
    for sk in ("_sigma_0p00", "_sigma_0p05", "_sigma_0p10"):
        assert sk in sweep
    assert sweep["_measured_n_eff_spread"] < 1e-3
    assert sweep["_canonical_target_passed"] is True
    assert abs(apu["_flrw_gap_to_mangano_2005"] - 0.0095) < 1e-6
    assert "ap_form_model" in apu["_flrw_gap_origin"].lower()


def test_cross_code_fixture_ap_unified_grid_consistency(
    cross_code_targets: dict,
) -> None:
    """``ap_unified_preflight`` grid-resolved entries cluster within
    ``< 1e-4`` (the PR-T3B canonical-milestone grid envelope)."""
    apu = cross_code_targets["_metadata"]["preflight_measurements"][
        "ap_unified_preflight"
    ]
    n_effs = [apu[gk]["N_eff"] for gk in ("_grid_4_6", "_grid_8_12", "_grid_12_20")]
    spread = max(n_effs) - min(n_effs)
    assert spread < 1e-4, (
        f"ap_unified_preflight grid spread widened in fixture: {spread:.6e}"
    )


def test_cross_code_fixture_ap_unified_cl_ladder_block(
    cross_code_targets: dict,
) -> None:
    """The ``_cl_ladder_flrw_4_6`` block records the PR-T3 full CL
    envelope (CL0-CL3) for the publicly dispatchable AP-unified
    candidate.  Locks Y_p ordering, N_eff CL-invariance, and the
    Y_p gap to LASAGNA / PRIMAT-AC2024 at CL2 / CL3."""
    apu = cross_code_targets["_metadata"]["preflight_measurements"][
        "ap_unified_preflight"
    ]
    assert "_cl_ladder_flrw_4_6" in apu
    cl_ladder = apu["_cl_ladder_flrw_4_6"]
    cl_keys = ("_cl0", "_cl1", "_cl2", "_cl3")
    for k in cl_keys:
        assert k in cl_ladder
        assert "Y_p" in cl_ladder[k] and "N_eff" in cl_ladder[k]
    # Y_p must increase monotonically with CL.
    yp_ladder = [cl_ladder[k]["Y_p"] for k in cl_keys]
    assert all(yp_ladder[i] < yp_ladder[i + 1] for i in range(3))
    assert cl_ladder["_yp_ordering_passed"] is True
    # N_eff stays within the locked CL-invariant spread.
    n_ladder = [cl_ladder[k]["N_eff"] for k in cl_keys]
    n_spread = max(n_ladder) - min(n_ladder)
    assert n_spread <= cl_ladder["_n_eff_cl_invariant_spread"] * 5  # 5x headroom
    # CL2 should land within 1e-3 of LASAGNA Y_p; CL3 within 1e-3 of PRIMAT.
    assert abs(cl_ladder["_yp_cl2_vs_lasagna_gap"]) < 1e-3
    assert abs(cl_ladder["_yp_cl3_vs_primat_ac2024_gap"]) < 1e-3


def test_cross_code_fixture_projected_physical_grid_consistency(
    cross_code_targets: dict,
) -> None:
    """The ``projected_physical_preflight`` grid-resolved entries
    cluster within ``< 1e-4`` (the locked grid-convergence
    envelope from PR-T3B-PF #11)."""
    pm = cross_code_targets["_metadata"]["preflight_measurements"][
        "projected_physical_preflight"
    ]
    n_effs = [
        pm["_grid_4_6"]["N_eff"],
        pm["_grid_8_12"]["N_eff"],
        pm["_grid_12_20"]["N_eff"],
    ]
    spread = max(n_effs) - min(n_effs)
    assert spread < 1e-4, (
        f"projected_physical preflight measurements drifted across grids: "
        f"spread = {spread:.6f}"
    )


def test_cross_code_fixture_remap_preconditioned_boundary_recorded(
    cross_code_targets: dict,
) -> None:
    """The remapped-kernel Jacobian preconditioner is an executable
    stiffness bridge, not a canonical grid-converged physics surface."""
    pm = cross_code_targets["_metadata"]["preflight_measurements"][
        "jax_kernel_remap_preconditioned_preflight"
    ]
    n_eff_values = [
        pm["_grid_4_6"]["N_eff"],
        pm["_grid_4_8"]["N_eff"],
        pm["_grid_4_10"]["N_eff"],
    ]
    assert n_eff_values[0] > n_eff_values[1] > n_eff_values[2]
    assert pm["_grid_spread_n_eff"] == pytest.approx(
        n_eff_values[0] - n_eff_values[-1], rel=0.0, abs=1e-10
    )
    assert pm["_grid_spread_n_eff"] > 1.5e-2

    tol = pm["_tolerance_envelope_4_6"]
    assert tol["_production_to_tight_n_eff_gap"] < 2.0e-3
    assert tol["_loose_to_tight_n_eff_gap"] < 5.0e-3

    note = pm["_note"].lower()
    assert "not a canonical calibration" in note
    assert "not grid-converged" in note


def test_cross_code_fixture_anisotropic_sweep_present(
    cross_code_targets: dict,
) -> None:
    """The ``anisotropic_sweep_projected_physical`` block records
    the per-σ_H FLRW measurements + the spread that documents the
    canonical PR-T3D §5 stability gate."""
    pm = cross_code_targets["_metadata"]["preflight_measurements"]
    assert "anisotropic_sweep_projected_physical" in pm
    sweep = pm["anisotropic_sweep_projected_physical"]
    for sigma_key in ("_sigma_0p00", "_sigma_0p05", "_sigma_0p10"):
        assert sigma_key in sweep
        assert "N_eff" in sweep[sigma_key]
        assert "Y_p" in sweep[sigma_key]
    # The 0.30 entry is a string (fail marker) not a dict.
    assert "_sigma_0p30" in sweep
    assert "FAILED" in str(sweep["_sigma_0p30"])
    # Spread is locked at the measured value with bounded envelope.
    measured_spread = sweep["_measured_n_eff_spread"]
    assert 0.4 < measured_spread < 0.7
    # And the canonical-target string references the §5 gate so a
    # reader of the fixture knows what to aim for.
    target = sweep["_canonical_target"].lower()
    assert "1e-3" in target or "5" in target or "phase prompt" in target


def test_cross_code_fixture_neff_consistency(cross_code_targets: dict) -> None:
    """All cross-code N_eff entries cluster within the canonical
    tolerance of each other (3.043 vs 3.044 vs 3.044)."""
    neffs = [
        cross_code_targets["lasagna"]["N_eff"],
        cross_code_targets["fortepiano"]["N_eff"],
        cross_code_targets["primat_ac2024"]["N_eff"],
        cross_code_targets["mangano_2005"]["N_eff"],
    ]
    spread = max(neffs) - min(neffs)
    assert spread < 5e-3, (
        f"cross-code N_eff spread exceeds 5e-3: {spread:.4f}"
    )


# ---------------------------------------------------------------------------
# Diagnostic: report the gap from the current preflight FLRW baseline
# ---------------------------------------------------------------------------


# Preflight bounds tightened to within ~15% of the measured gap so any
# future drift is flagged early.  Canonical bounds (5e-4 and 5e-3) remain
# a separate, deferred target.
_N_EFF_PREFLIGHT_BOUND = 6.0e-2  # measured 5.1e-2 vs all 4 codes
_Y_P_PREFLIGHT_BOUND = 6.0e-3    # measured 5.3e-3 vs LASAGNA/FortEPiaNO/PRIMAT

# Canonical PR-T3D targets, recorded as constants for the audit doc.
_CANONICAL_N_EFF_TOL = 5.0e-3
_CANONICAL_Y_P_TOL = 5.0e-4


@pytest.fixture(scope="module")
def flrw_jax_kernel_result():
    """One FLRW (Σ=0) ``jax_kernel_preflight`` solve, shared across all
    per-cross-code gap tests so the comparison runs in a single CPU
    pass."""
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
        collision_mode="jax_kernel_preflight",
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=256,
        event_refine_steps=12,
    )
    r = run_full_boltzmann_jax(cfg)
    assert r.success
    return {
        "Yp": float(r.Yp),
        "N_eff": float(r.metadata["N_eff_measured"]),
    }


@pytest.mark.parametrize(
    "code", ["lasagna", "fortepiano", "primat_ac2024", "mangano_2005"]
)
def test_flrw_n_eff_gap_to_cross_code(
    flrw_jax_kernel_result: dict,
    cross_code_targets: dict,
    code: str,
) -> None:
    """Per-cross-code N_eff gap diagnostic.

    Locks the FLRW (``Σ=0``) ``N_eff`` measured by the
    ``jax_kernel_preflight`` runtime mode within
    ``< 6e-2`` of the published cross-code reference for
    ``code in {LASAGNA, FortEPiaNO, PRIMAT-AC2024, Mangano 2005}``.

    Measured FLRW baseline: ``N_eff ≈ 2.993427``; gap to all four
    cross-codes is ``~5.1e-2``.  The canonical PR-T3D target is
    ``< 5e-3`` (recorded separately so the audit trail captures
    both the current preflight bound and the canonical destination).
    """
    target = cross_code_targets[code]
    N_eff = flrw_jax_kernel_result["N_eff"]
    n_gap = abs(N_eff - target["N_eff"])
    assert n_gap < _N_EFF_PREFLIGHT_BOUND, (
        f"FLRW N_eff gap to {code} widened beyond preflight bound: "
        f"|N_eff - target| = {n_gap:.4f} > {_N_EFF_PREFLIGHT_BOUND} "
        f"(canonical target < {_CANONICAL_N_EFF_TOL:g})"
    )


@pytest.mark.parametrize(
    "code", ["lasagna", "fortepiano", "primat_ac2024"]  # mangano omits Y_p
)
def test_flrw_yp_gap_to_cross_code(
    flrw_jax_kernel_result: dict,
    cross_code_targets: dict,
    code: str,
) -> None:
    """Per-cross-code Y_p gap diagnostic.

    Locks the FLRW (``Σ=0``) ``Y_p`` measured by the
    ``jax_kernel_preflight`` runtime mode within
    ``< 6e-3`` of the published cross-code reference for
    ``code in {LASAGNA, FortEPiaNO, PRIMAT-AC2024}``.  Mangano 2005
    does not quote ``Y_p`` to the same precision and is omitted.

    Measured FLRW baseline: ``Y_p ≈ 0.2417042168``; gap to all
    three cross-codes is ``~5.3e-3``.  The canonical PR-T3D target
    is ``< 5e-4``.
    """
    target = cross_code_targets[code]
    assert target["Y_p"] is not None, (
        f"cross-code fixture {code} unexpectedly omits Y_p"
    )
    Yp = flrw_jax_kernel_result["Yp"]
    y_gap = abs(Yp - target["Y_p"])
    assert y_gap < _Y_P_PREFLIGHT_BOUND, (
        f"FLRW Y_p gap to {code} widened beyond preflight bound: "
        f"|Y_p - target| = {y_gap:.5f} > {_Y_P_PREFLIGHT_BOUND} "
        f"(canonical target < {_CANONICAL_Y_P_TOL:g})"
    )


def test_flrw_baseline_measurement_invariant(
    flrw_jax_kernel_result: dict,
) -> None:
    """Lock the absolute FLRW measurement values used by the per-code
    diagnostics above.  This guards against measurement drift even
    if every cross-code reference were to shift in lockstep."""
    Yp = flrw_jax_kernel_result["Yp"]
    N_eff = flrw_jax_kernel_result["N_eff"]
    # Measured: Yp = 0.2417042168, N_eff = 2.993427
    assert 0.241 < Yp < 0.243, f"FLRW Y_p baseline drifted: {Yp:.10f}"
    assert 2.97 < N_eff < 3.02, f"FLRW N_eff baseline drifted: {N_eff:.6f}"
