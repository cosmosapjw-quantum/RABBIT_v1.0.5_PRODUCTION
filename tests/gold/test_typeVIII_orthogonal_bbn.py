"""Gold gate: Bianchi Type VIII orthogonal BBN cell."""
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


def test_typeVIII_orthogonal_zero_structure_recovers_typeI_jax_cell():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    gold = _gold("classA_typeVIII_N0_sigma001")
    type_i = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_I",
            Sigma_H_plus=gold["Sigma_H"],
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )
    type_viii_zero = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VIII",
            Sigma_H_plus=gold["Sigma_H"],
            N1_init=0.0,
            N2_init=0.0,
            N3_init=0.0,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )

    assert type_i.success is True
    assert type_viii_zero.success is True
    assert type_viii_zero.Yp == pytest.approx(type_i.Yp, abs=1.0e-14)
    assert type_viii_zero.DH == pytest.approx(type_i.DH, rel=1.0e-14)
    assert type_viii_zero.Yp == pytest.approx(gold["Yp"], rel=1.0e-6)
    assert type_viii_zero.metadata["curvature_K_final"] == 0.0
    assert type_viii_zero.metadata["transport_kappa_final"] == 0.0


def test_typeVIII_orthogonal_mixed_sign_curvature_changes_yields():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    baseline_gold = _gold("classA_typeVIII_N0_sigma001")
    gold = _gold("classA_typeVIII_N1_m1em4_N2_1em4_N3_1em4_sigma001")
    baseline = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VIII",
            Sigma_H_plus=baseline_gold["Sigma_H"],
            N1_init=0.0,
            N2_init=0.0,
            N3_init=0.0,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )
    curved = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_VIII",
            Sigma_H_plus=gold["Sigma_H"],
            N1_init=gold["N1_init"],
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
    assert curved.Yp - baseline.Yp == pytest.approx(gold["DYp_N123"], rel=1.0e-6, abs=1.0e-9)
    assert curved.DH - baseline.DH == pytest.approx(gold["DDH_N123"], rel=1.0e-6, abs=1.0e-10)
    assert curved.metadata["N1_final"] < 0.0
    assert curved.metadata["N2_final"] > 0.0
    assert curved.metadata["N3_final"] > 0.0
    assert curved.metadata["curvature_K_init"] > 0.0
    assert curved.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-6)
    assert curved.metadata["transport_kappa_final"] == pytest.approx(gold["transport_kappa_final"], rel=1.0e-6)
    assert curved.metadata["Omega_final"] == pytest.approx(gold["Omega_final"], rel=1.0e-6)
    assert curved.metadata["curvature_K_final"] > curved.metadata["curvature_K_init"]
    assert curved.metadata["transport_kappa_final"] > curved.metadata["transport_kappa_init"]
    assert curved.metadata["Omega_final"] > 0.95


def test_typeVIII_orthogonal_rejects_same_sign_structure_constants():
    from rabbit.jax.driver_classA import JAXClassAConfig

    with pytest.raises(ValueError, match="mixed-sign"):
        JAXClassAConfig(
            bianchi_type="TYPE_VIII",
            Sigma_H_plus=0.01,
            N1_init=0.001,
            N2_init=0.001,
            N3_init=0.001,
            N_q=20,
        )
