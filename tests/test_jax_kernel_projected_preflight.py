from __future__ import annotations

import pytest

pytest.importorskip("jax")


@pytest.mark.production
@pytest.mark.slow
@pytest.mark.gold
def test_jax_kernel_projected_preflight_flrw_reheating_window():
    """Full ODE smoke for the remapped kernel shape + Mangano moment projection.

    This is not the public canonical tier-3 dispatch.  It verifies that the
    newly wired temperature-frame remap can be carried through the full
    full-Boltzmann driver when the stiff kernel energy moment is projected to
    the known Mangano total-rate source.
    """
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        collision_mode="jax_kernel_projected_preflight",
        thermo_tier=2,
        rtol=1.0e-5,
        atol=1.0e-7,
        max_steps=2000,
        event_refine_steps=12,
    )
    result = run_full_boltzmann_jax(cfg)
    assert result.success, result.metadata
    assert result.metadata["collision_scope_contract"] == (
        "jax_kernel_temperature_remap_mangano_projected_v1"
    )
    assert 3.02 < result.metadata["N_eff_measured"] < 3.045
    assert result.metadata["N_eff_measured"] > 3.010
    assert 0.23 < result.Yp < 0.25
