"""Gold gate: Bianchi Type VII0 orthogonal BBN cell."""
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


def test_typeVII0_orthogonal_zero_structure_recovers_typeI_jax_cell():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    gold = _gold("classA_typeVII0_N0_sigma003")
    type_i = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_I",
            Sigma_H_plus=gold["Sigma_H"],
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )
    type_vii0_zero = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VII0",
            Sigma_H_plus=gold["Sigma_H"],
            N2_init=0.0,
            N3_init=0.0,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )

    assert type_i.success is True
    assert type_vii0_zero.success is True
    assert type_vii0_zero.Yp == pytest.approx(type_i.Yp, abs=1.0e-14)
    assert type_vii0_zero.DH == pytest.approx(type_i.DH, rel=1.0e-14)
    assert type_vii0_zero.Yp == pytest.approx(gold["Yp"], rel=1.0e-6)
    assert type_vii0_zero.metadata["curvature_K_final"] == 0.0
    assert type_vii0_zero.metadata["transport_kappa_final"] == 0.0


def test_typeVII0_orthogonal_same_sign_curvature_changes_yields():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    baseline_gold = _gold("classA_typeVII0_N0_sigma003")
    gold = _gold("classA_typeVII0_N2_5e3_N3_3e3_sigma003")
    baseline = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VII0",
            Sigma_H_plus=baseline_gold["Sigma_H"],
            N2_init=0.0,
            N3_init=0.0,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )
    curved = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VII0",
            Sigma_H_plus=gold["Sigma_H"],
            N2_init=gold["N2_init"],
            N3_init=gold["N3_init"],
            N_q=20,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert curved.success is True
    assert curved.Yp == pytest.approx(gold["Yp"], rel=1.0e-6)
    assert curved.DH == pytest.approx(gold["DH"], rel=1.0e-6)
    assert curved.Yp - baseline.Yp == pytest.approx(gold["DYp_N23"], rel=1.0e-6)
    assert curved.metadata["N1_final"] == 0.0
    assert curved.metadata["N2_final"] > gold["N2_init"]
    assert curved.metadata["N3_final"] > gold["N3_init"]
    assert curved.metadata["N2_final"] != pytest.approx(curved.metadata["N3_final"])
    assert curved.metadata["curvature_K_init"] > 0.0
    assert curved.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-6)
    assert curved.metadata["transport_kappa_final"] == pytest.approx(gold["transport_kappa_final"], rel=1.0e-6)
    assert curved.metadata["transport_kappa_final"] > curved.metadata["transport_kappa_init"]


def test_typeVII0_orthogonal_rejects_opposite_sign_structure_constants():
    from rabbit.jax.driver_classA import JAXClassAConfig

    with pytest.raises(ValueError, match="N₂N₃≥0"):
        JAXClassAConfig(
            bianchi_type="TYPE_VII0",
            Sigma_H_plus=0.03,
            N2_init=0.005,
            N3_init=-0.003,
            N_q=20,
        )
