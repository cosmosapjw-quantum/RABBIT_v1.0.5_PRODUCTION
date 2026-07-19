"""Gold gate: Bianchi Type IV orthogonal BBN cell."""
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


def test_typeIV_orthogonal_N1_zero_recovers_typeV_frame_cell():
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax

    gold = _gold("classB_typeIV_N10_A1e5_sigma001")
    type_v = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_V",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )
    type_iv_n1_zero = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_IV",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N1_init=0.0,
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )

    assert type_v.success is True
    assert type_iv_n1_zero.success is True
    assert type_iv_n1_zero.Yp == pytest.approx(type_v.Yp, abs=1.0e-14)
    assert type_iv_n1_zero.DH == pytest.approx(type_v.DH, rel=1.0e-14)
    assert type_iv_n1_zero.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert type_iv_n1_zero.metadata["curvature_K_final"] == 0.0
    assert type_iv_n1_zero.metadata["transport_kappa_final"] == 0.0


def test_typeIV_orthogonal_A_and_N1_couple_into_yields():
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax

    baseline_gold = _gold("classB_typeIV_N10_A1e5_sigma001")
    gold = _gold("classB_typeIV_N1e3_A1e5_sigma001")
    baseline = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_IV",
            Sigma_H_plus=baseline_gold["Sigma_H"],
            A_init=baseline_gold["A_init"],
            N1_init=0.0,
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )
    coupled = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_IV",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N1_init=gold["N1_init"],
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert coupled.success is True
    assert coupled.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert coupled.DH == pytest.approx(gold["DH"], rel=1.0e-10)
    assert coupled.Yp - baseline.Yp == pytest.approx(gold["DYp_N1"], rel=1.0e-8)
    assert coupled.metadata["A_final"] > coupled.metadata["A_init"]
    assert coupled.metadata["N1_final"] > coupled.metadata["N1_init"]
    assert coupled.metadata["curvature_K_final"] == pytest.approx(gold["curvature_K_final"], rel=1.0e-10)
    assert coupled.metadata["curvature_K_final"] > coupled.metadata["curvature_K_init"]
    assert coupled.metadata["transport_kappa_final"] > coupled.metadata["transport_kappa_init"]
    assert coupled.metadata["Omega_final"] < baseline.metadata["Omega_final"]


def test_typeIV_orthogonal_rejects_inactive_structure_constants():
    from rabbit.jax.driver_classB import JAXClassBConfig

    with pytest.raises(ValueError, match="Type IV requires"):
        JAXClassBConfig(
            bianchi_type="TYPE_IV",
            Sigma_H_plus=0.01,
            A_init=1.0e-5,
            N1_init=1.0e-3,
            N2_init=1.0e-3,
        )
