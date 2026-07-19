"""Gold gate: Bianchi Type V tilted scalar BBN cell."""
from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="JAX required")

pytestmark = pytest.mark.gold


def test_typeV_tilted_zero_frame_variable_is_self_consistent_null():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    result = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_V",
            v0=1.0e-7,
            Sigma_H_plus=0.02,
            A_init=0.0,
            N_q=6,
            correction_level=0,
        )
    )

    assert result.success is True
    assert result.metadata["canonical_bianchi_type"] == "TYPE_V"
    assert result.metadata["A_final"] == 0.0
    assert result.metadata["frame_cA_sq_final"] == 0.0
    assert result.metadata["curvature_K_final"] == 0.0
    assert abs(result.v_final) > 1.0e-7
    assert 0.20 < result.Yp < 0.30


def test_typeV_tilted_frame_variable_evolves_and_changes_yields():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    baseline = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_V",
            v0=1.0e-7,
            Sigma_H_plus=0.02,
            A_init=0.0,
            N_q=6,
            correction_level=0,
        )
    )
    framed = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_V",
            v0=1.0e-7,
            Sigma_H_plus=0.02,
            A_init=1.0e-4,
            N_q=6,
            correction_level=0,
        )
    )

    assert baseline.success is True
    assert framed.success is True
    assert framed.metadata["transport_mode"] == "tilted_kappa_cascade_lmax2"
    assert framed.metadata["A_final"] > framed.metadata["A_init"]
    assert framed.metadata["frame_cA_sq_final"] > framed.metadata["frame_cA_sq_init"]
    assert framed.metadata["Omega_final"] < baseline.metadata["Omega_final"]
    assert framed.Yp > baseline.Yp
    assert framed.Yp - baseline.Yp == pytest.approx(1.7845171579254102e-5, rel=1.0e-6)
    assert framed.metadata["curvature_K_final"] == 0.0
    assert abs(framed.v_final) > abs(baseline.v_final)


def test_typeV_tilted_rejects_inactive_structure_constants():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

    with pytest.raises(ValueError, match="inactive tilted structure constants"):
        TiltedBBNConfig(
            bianchi_type="TYPE_V",
            v0=1.0e-7,
            Sigma_H_plus=0.02,
            A_init=1.0e-4,
            N1_init=1.0e-3,
        )
