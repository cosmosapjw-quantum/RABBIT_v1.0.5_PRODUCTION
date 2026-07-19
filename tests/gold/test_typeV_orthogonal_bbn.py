"""Gold gate: Bianchi Type V orthogonal BBN cell."""
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


def test_typeV_orthogonal_zero_frame_variable_is_flat_classB_limit():
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax

    gold = _gold("classB_typeV_A0_sigma002")
    result = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_V",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=0.0,
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )

    assert result.success is True
    assert result.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert result.DH == pytest.approx(gold["DH"], rel=1.0e-10)
    assert result.metadata["A_final"] == 0.0
    assert result.metadata["frame_cA_sq_final"] == 0.0
    assert result.metadata["curvature_K_final"] == 0.0


def test_typeV_orthogonal_frame_variable_evolves_and_changes_yields():
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax

    gold = _gold("classB_typeV_A1e4")
    baseline = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_V",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=0.0,
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )
    framed = run_classB_jax(
        JAXClassBConfig(
            bianchi_type="TYPE_V",
            Sigma_H_plus=gold["Sigma_H"],
            A_init=gold["A_init"],
            N_q=6,
            n_ell=2,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert framed.success is True
    assert framed.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
    assert framed.DH == pytest.approx(gold["DH"], rel=1.0e-10)
    assert framed.metadata["A_final"] > framed.metadata["A_init"]
    assert framed.metadata["frame_cA_sq_final"] > framed.metadata["frame_cA_sq_init"]
    assert framed.metadata["Omega_final"] < baseline.metadata["Omega_final"]
    assert framed.Yp > baseline.Yp
    assert framed.Yp - baseline.Yp == pytest.approx(gold["DYp_A"], rel=5.0e-8)
    assert framed.metadata["curvature_K_final"] == 0.0


def test_typeV_orthogonal_public_inference_endpoint_is_retired():
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend="jax_classB", Sigma_H=0.02)
