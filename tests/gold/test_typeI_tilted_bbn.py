"""Gold gate: Bianchi Type I tilted scalar BBN cell."""
from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="JAX required")

pytestmark = pytest.mark.gold


def test_typeI_tilted_v0_zero_recovers_untilted_scalar_cell():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    untilted = run_tilted_bbn(
        TiltedBBNConfig(bianchi_type="TYPE_I", v0=0.0, Sigma_H_plus=0.0)
    )

    assert untilted.success is True
    assert untilted.v_final == 0.0
    assert 0.240 < untilted.Yp < 0.245
    assert untilted.metadata["bianchi_type"] == "TYPE_I"


def test_typeI_tilted_small_v0_is_physical_and_grows():
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

    baseline = run_tilted_bbn(
        TiltedBBNConfig(bianchi_type="TYPE_I", v0=0.0, Sigma_H_plus=0.0)
    )
    tilted = run_tilted_bbn(
        TiltedBBNConfig(bianchi_type="TYPE_I", v0=1.0e-7, Sigma_H_plus=0.05)
    )

    assert baseline.success is True
    assert tilted.success is True
    assert abs(tilted.v_final) > 1.0e-7
    assert 0.20 < tilted.Yp < 0.30
    assert tilted.Yp >= baseline.Yp - 1.0e-6
