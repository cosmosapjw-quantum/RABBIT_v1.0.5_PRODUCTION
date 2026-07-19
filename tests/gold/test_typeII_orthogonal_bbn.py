"""Gold gate: Bianchi Type II orthogonal BBN cell."""
from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="JAX required")

pytestmark = pytest.mark.gold


def test_typeII_orthogonal_zero_curvature_recovers_typeI_jax_cell():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    type_i = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_I",
            Sigma_H_plus=0.05,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )
    type_ii_zero = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_II",
            Sigma_H_plus=0.05,
            N1_init=0.0,
            N_q=20,
            correction_level=0,
            transport_mode="linearized",
        )
    )

    assert type_i.success is True
    assert type_ii_zero.success is True
    assert type_ii_zero.Yp == pytest.approx(type_i.Yp, abs=1.0e-12)
    assert type_ii_zero.DH == pytest.approx(type_i.DH, rel=1.0e-12)
    assert type_ii_zero.metadata["curvature_K_final"] == 0.0


def test_typeII_orthogonal_curvature_evolves_and_changes_yields():
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax

    weak = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_II",
            Sigma_H_plus=0.05,
            N1_init=0.1,
            N_q=20,
            correction_level=0,
        )
    )
    strong = run_classA_jax(
        JAXClassAConfig(
            bianchi_type="TYPE_II",
            Sigma_H_plus=0.05,
            N1_init=0.2,
            N_q=20,
            correction_level=0,
        )
    )

    assert weak.success is True
    assert strong.success is True
    assert weak.metadata["transport_mode"] == "kappa_cascade_lmax2"
    assert weak.metadata["production_authority"] == "candidate_classA_curved_transport"
    assert weak.metadata["curvature_K_init"] > 0.0
    assert weak.metadata["curvature_K_final"] > weak.metadata["curvature_K_init"]
    assert weak.metadata["N1_final"] != pytest.approx(0.1)
    assert strong.metadata["curvature_K_init"] > weak.metadata["curvature_K_init"]
    assert strong.Yp > weak.Yp
    assert abs(strong.Yp - weak.Yp) > 1.0e-4
    assert 0.20 < weak.Yp < 0.30
    assert 1.0e-6 < weak.DH < 1.0e-3
