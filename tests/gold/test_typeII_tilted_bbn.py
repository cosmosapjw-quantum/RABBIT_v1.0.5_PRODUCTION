"""Gold gate: Bianchi Type II tilted scalar BBN cell."""
from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="JAX required")

pytestmark = pytest.mark.gold


def test_typeII_tilted_zero_curvature_recovers_typeI_tilted_cell():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    type_i = run_tilted_bbn(
        TiltedBBNConfig(bianchi_type="TYPE_I", v0=1.0e-7, Sigma_H_plus=0.05)
    )
    type_ii_zero = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_II",
            v0=1.0e-7,
            Sigma_H_plus=0.05,
            N1_init=0.0,
        )
    )

    assert type_i.success is True
    assert type_ii_zero.success is True
    assert type_ii_zero.Yp == pytest.approx(type_i.Yp, abs=1.0e-12)
    assert type_ii_zero.DH == pytest.approx(type_i.DH, rel=1.0e-12)
    assert type_ii_zero.v_final == pytest.approx(type_i.v_final, rel=1.0e-12)
    assert type_ii_zero.metadata["curvature_K_final"] == 0.0


def test_typeII_tilted_curvature_and_tilt_enter_forward_model():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    flat_tilted = run_tilted_bbn(
        TiltedBBNConfig(bianchi_type="TYPE_I", v0=1.0e-7, Sigma_H_plus=0.05)
    )
    curved_tilted = run_tilted_bbn(
        TiltedBBNConfig(
            bianchi_type="TYPE_II",
            v0=1.0e-7,
            Sigma_H_plus=0.05,
            N1_init=0.1,
        )
    )

    assert curved_tilted.success is True
    assert curved_tilted.metadata["transport_mode"] == "tilted_kappa_cascade_lmax2"
    assert curved_tilted.metadata["canonical_bianchi_type"] == "TYPE_II"
    assert curved_tilted.metadata["curvature_K_init"] > 0.0
    assert curved_tilted.metadata["curvature_K_final"] > curved_tilted.metadata["curvature_K_init"]
    assert curved_tilted.metadata["N1_final"] != pytest.approx(0.1)
    assert curved_tilted.metadata["Omega_final"] > 0.0
    assert abs(curved_tilted.v_final) > 1.0e-7
    assert curved_tilted.Yp > flat_tilted.Yp
    assert abs(curved_tilted.Yp - flat_tilted.Yp) > 1.0e-3
    assert 0.20 < curved_tilted.Yp < 0.30
    assert 1.0e-6 < curved_tilted.DH < 1.0e-3


def test_typeII_tilted_rejects_inactive_structure_constants():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig

    with pytest.raises(ValueError, match="inactive tilted structure constants"):
        TiltedBBNConfig(
            bianchi_type="TYPE_II",
            v0=1.0e-7,
            Sigma_H_plus=0.05,
            N1_init=0.1,
            N2_init=0.1,
        )
