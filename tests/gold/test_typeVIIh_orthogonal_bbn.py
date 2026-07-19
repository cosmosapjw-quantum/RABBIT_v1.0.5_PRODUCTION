"""Gold gate: Bianchi Type VII_h orthogonal h-locked BBN cell."""
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


def test_typeVIIh_orthogonal_requires_explicit_h():
    from rabbit.jax.driver_classB import JAXClassBConfig

    with pytest.raises(ValueError, match="requires an explicit h"):
        JAXClassBConfig(
            bianchi_type="TYPE_VIIH",
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N2_init=5.0e-4,
            N3_init=2.5e-4,
            N_q=6,
        )


def test_typeVIIh_orthogonal_rejects_wrong_h_relation():
    from rabbit.jax.driver_classB import JAXClassBConfig

    with pytest.raises(ValueError, match="N₃=h\\*N₂"):
        JAXClassBConfig(
            bianchi_type="TYPE_VIIH",
            h=0.5,
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N2_init=5.0e-4,
            N3_init=1.0e-4,
            N_q=6,
        )


def test_typeVIIh_orthogonal_h_locked_curvature_and_frame_change_yields():
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax

    baseline_gold = _gold("classB_typeVIIh_h0p5_N0_A1e5_sigma001")
    gold = _gold("classB_typeVIIh_h0p5_N2_5em4_N3_2p5em4_A1e5_sigma001")
    baseline = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_V",
            Sigma_H_plus=baseline_gold["Sigma_H"],
            A_init=baseline_gold["A_init"],
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )
    curved = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_VIIH",
            h=gold["h"],
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N2_init=gold["N2_init"],
            N3_init=gold["N3_init"],
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert curved.success is True
    assert baseline.Yp == pytest.approx(baseline_gold["Yp"], rel=1.0e-10)
    assert curved.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert curved.DH == pytest.approx(gold["DH"], rel=1.0e-10)
    assert curved.Yp - baseline.Yp == pytest.approx(gold["DYp_N23"], rel=1.0e-7, abs=5.0e-13)
    assert curved.DH - baseline.DH == pytest.approx(gold["DDH_N23"], rel=1.0e-7, abs=5.0e-15)
    assert curved.metadata["h_parameter"] == pytest.approx(gold["h"])
    assert curved.metadata["c_factor"] == pytest.approx(gold["c_factor"])
    assert curved.metadata["N1_final"] == 0.0
    assert curved.metadata["N2_final"] > 0.0
    assert curved.metadata["N3_final"] > 0.0
    assert curved.metadata["N3_final"] == pytest.approx(gold["h"] * curved.metadata["N2_final"], rel=1.0e-12)
    assert curved.metadata["curvature_K_init"] == pytest.approx(gold["curvature_K_init"], rel=1.0e-10)
    assert curved.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-10)
    assert curved.metadata["transport_kappa_final"] == pytest.approx(gold["transport_kappa_final"], rel=1.0e-10)
    assert curved.metadata["frame_cA_sq_final"] == pytest.approx(gold["frame_cA_sq_final"], rel=1.0e-10)
    assert curved.metadata["Omega_final"] == pytest.approx(gold["Omega_final"], rel=1.0e-10)
    assert curved.metadata["Omega_final"] < baseline.metadata["Omega_final"]
