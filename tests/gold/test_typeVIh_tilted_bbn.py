"""Gold gate: Bianchi Type VI_h tilted scalar h-locked BBN cell."""
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


def test_typeVIh_tilted_requires_explicit_h():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

    with pytest.raises(ValueError, match="requires an explicit h"):
        TiltedBBNConfig(
            bianchi_type="TYPE_VIH",
            v0=1.0e-7,
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N2_init=1.0e-4,
            N3_init=-2.0e-4,
            N_q=6,
        )


def test_typeVIh_tilted_rejects_wrong_h_relation():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

    with pytest.raises(ValueError, match="N3=h\\*N2"):
        TiltedBBNConfig(
            bianchi_type="TYPE_VIH",
            h=-2.0,
            v0=1.0e-7,
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N2_init=1.0e-4,
            N3_init=-1.0e-4,
            N_q=6,
        )


def test_typeVIh_tilted_h_locked_curvature_frame_and_tilt_change_yields():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    baseline_gold = _gold("tilted_typeVIh_hm2_N0_A1e5_sigma001_v1e7")
    gold = _gold("tilted_typeVIh_hm2_N2_1em4_N3_m2em4_A1e5_sigma001_v1e7")
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
            bianchi_type="TYPE_VIH",
            h=gold["h"],
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
    assert baseline.Yp == pytest.approx(baseline_gold["Yp"], rel=1.0e-6)
    assert curved.Yp == pytest.approx(gold["Yp"], rel=1.0e-6)
    assert curved.DH == pytest.approx(gold["DH"], rel=1.0e-6)
    assert curved.Yp - baseline.Yp == pytest.approx(gold["DYp_N23"], rel=1.0e-6, abs=1.0e-9)
    assert curved.DH - baseline.DH == pytest.approx(gold["DDH_N23"], rel=1.0e-6, abs=1.0e-10)
    assert curved.metadata["h_parameter"] == pytest.approx(gold["h"])
    assert curved.metadata["c_factor"] == pytest.approx(gold["c_factor"])
    assert curved.metadata["N1_final"] == 0.0
    assert curved.metadata["N2_final"] > 0.0
    assert curved.metadata["N3_final"] < 0.0
    assert curved.metadata["N3_final"] == pytest.approx(gold["h"] * curved.metadata["N2_final"], rel=1.0e-12)
    assert curved.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-6)
    assert curved.metadata["transport_kappa_final"] == pytest.approx(gold["transport_kappa_final"], rel=1.0e-6)
    assert curved.metadata["frame_cA_sq_final"] == pytest.approx(gold["frame_cA_sq_final"], rel=1.0e-6)
    assert curved.metadata["Omega_final"] == pytest.approx(gold["Omega_final"], rel=1.0e-6)
    assert curved.metadata["Omega_final"] < baseline.metadata["Omega_final"]
    assert abs(curved.v_final - baseline.v_final) > 1.0e-6
