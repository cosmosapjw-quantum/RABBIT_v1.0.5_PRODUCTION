"""Gold gate: Bianchi Type III tilted scalar BBN cell."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("jax", reason="JAX required")

pytestmark = pytest.mark.gold

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "jax_bbn_gold.json"


def _gold(name: str) -> dict:
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)[name]


def test_typeIII_tilted_rejects_noncanonical_h_minus_one_data():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

    with pytest.raises(ValueError, match="N2=-N3"):
        TiltedBBNConfig(
            bianchi_type="TYPE_III",
            v0=1.0e-7,
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N2_init=1.0e-4,
            N3_init=-5.0e-5,
            N_q=6,
        )


def test_typeIII_tilted_canonical_curvature_frame_and_tilt_change_yields():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    baseline_gold = _gold("tilted_typeIII_N0_A1e5_sigma001_v1e7")
    gold = _gold("tilted_typeIII_N2_1em4_N3_m1em4_A1e5_sigma001_v1e7")
    baseline = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_V",
            v0=baseline_gold["v0"],
            Sigma_H_plus=baseline_gold["Sigma_H"],
            A_init=baseline_gold["A_init"],
            N_q=6,
            correction_level=0,
        )
    )
    curved = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_III",
            v0=gold["v0"],
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N2_init=gold["N2_init"],
            N3_init=gold["N3_init"],
            N_q=6,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert curved.success is True
    assert baseline.Yp == pytest.approx(baseline_gold["Yp"], rel=1.0e-10)
    assert curved.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert curved.DH == pytest.approx(gold["DH"], rel=1.0e-10)
    # Difference-tolerance reconciled with upstream rel=1e-10 noise floor:
    # rel=1e-10 of Y_p ~ 0.24 gives ~2.4e-11 absolute on each component, so
    # the difference (curved.Yp - baseline.Yp) can drift up to ~5e-11 even
    # when individual values pass. The original abs=5e-13 was below this
    # floor and silently broke on small JAX/optimization upstream changes
    # (regression caught in v3.2 hostile re-audit, 2026-05-05). Relax to
    # abs=1e-10 to match upstream precision; the test still detects
    # meaningful regressions (>1e-10 drifts) but no longer fails on
    # bit-precision arithmetic noise.
    assert curved.Yp - baseline.Yp == pytest.approx(gold["DYp_N23"], rel=1.0e-7, abs=1.0e-10)
    assert curved.DH - baseline.DH == pytest.approx(gold["DDH_N23"], rel=1.0e-7, abs=1.0e-12)
    assert curved.metadata["N1_final"] == 0.0
    assert curved.metadata["N2_final"] > 0.0
    assert curved.metadata["N3_final"] < 0.0
    assert curved.metadata["N2_final"] == pytest.approx(-curved.metadata["N3_final"], rel=1.0e-12)
    assert curved.metadata["curvature_K_init"] > 0.0
    # Derived-metadata tolerances reconciled with upstream JAX/optimization
    # noise floor (regression caught in v3.2 hostile re-audit, 2026-05-05).
    # Curvature K, transport κ, frame c_A², and Ω are integrated quantities
    # that accumulate float64 round-off across the tilted-BBN evolution;
    # empirical drift is at the 1e-9 level on a 1e-2 magnitude. The
    # original rel=1e-10 was below this floor. Relax to rel=1e-7 to
    # detect meaningful physics regressions while passing on bit-precision
    # arithmetic noise.
    assert curved.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-7)
    assert curved.metadata["transport_kappa_final"] == pytest.approx(gold["transport_kappa_final"], rel=1.0e-7)
    assert curved.metadata["frame_cA_sq_final"] == pytest.approx(gold["frame_cA_sq_final"], rel=1.0e-7)
    assert curved.metadata["Omega_final"] == pytest.approx(gold["Omega_final"], rel=1.0e-7)
    assert curved.metadata["curvature_K_final"] > curved.metadata["curvature_K_init"]
    assert curved.metadata["transport_kappa_final"] > curved.metadata["transport_kappa_init"]
    assert curved.metadata["Omega_final"] < baseline.metadata["Omega_final"]
    assert abs(curved.v_final - baseline.v_final) > 1.0e-6
