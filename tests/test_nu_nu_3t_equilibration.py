"""Energy-conserving diagonal nu-nu equilibration in the 3T tier-3 path."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.nudec_coupled_jax import (
    hubble_3T_jax,
    nu_nu_temperature_equilibration_sources_jax,
)
from rabbit.thermo.nudec_coupled import nu_nu_temperature_equilibration_sources


jax.config.update("jax_enable_x64", True)


@pytest.mark.production
def test_nu_nu_3t_source_vanishes_in_equal_temperature_limit() -> None:
    H = hubble_3T_jax(jnp.asarray(2.0), jnp.asarray(1.7), jnp.asarray(1.7), jnp.asarray(0.0))
    dQ_e, dQ_x = nu_nu_temperature_equilibration_sources_jax(
        jnp.asarray(1.7),
        jnp.asarray(1.7),
        H,
    )
    assert float(dQ_e) == pytest.approx(0.0, abs=1e-30)
    assert float(dQ_x) == pytest.approx(0.0, abs=1e-30)


@pytest.mark.production
def test_nu_nu_3t_source_conserves_neutrino_energy_and_relaxes_temperature_split() -> None:
    H = hubble_3T_jax(jnp.asarray(2.0), jnp.asarray(1.85), jnp.asarray(1.55), jnp.asarray(0.0))
    dQ_e, dQ_x = nu_nu_temperature_equilibration_sources_jax(
        jnp.asarray(1.85),
        jnp.asarray(1.55),
        H,
    )

    # T_nue > T_nux: diagonal nu-nu scattering moves energy from the
    # electron-neutrino pair into the heavy-flavour bank, without touching
    # the electromagnetic plasma.
    assert float(dQ_e) < 0.0
    assert float(dQ_x) > 0.0
    assert float(dQ_e + dQ_x) == pytest.approx(0.0, abs=1e-27)


@pytest.mark.production
def test_nu_nu_3t_numpy_jax_parity() -> None:
    T_e = 1.83
    T_x = 1.61
    H = float(hubble_3T_jax(jnp.asarray(2.0), jnp.asarray(T_e), jnp.asarray(T_x), jnp.asarray(0.0)))

    jax_src = np.asarray(
        nu_nu_temperature_equilibration_sources_jax(
            jnp.asarray(T_e),
            jnp.asarray(T_x),
            jnp.asarray(H),
        )
    )
    np_src = np.asarray(nu_nu_temperature_equilibration_sources(T_e, T_x, H))
    np.testing.assert_allclose(jax_src, np_src, rtol=1e-12, atol=1e-30)


@pytest.mark.slow
def test_ap_unified_nu_nu_preflight_runs_and_reduces_final_temperature_split() -> None:
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    common = dict(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=6,
        correction_level=0,
        n_reactions=12,
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=512,
        event_refine_steps=12,
    )
    baseline = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(collision_mode="ap_unified_preflight", **common)
    )
    with_nunu = run_full_boltzmann_jax(
        JAXFullBoltzmannConfig(collision_mode="ap_unified_nu_nu_preflight", **common)
    )

    assert baseline.success
    assert with_nunu.success
    assert with_nunu.metadata["collision_scope_contract"] == (
        "ap_unified_plus_energy_conserving_nu_nu_3T_preflight_v1"
    )
    assert with_nunu.metadata["nu_nu_equilibration_enabled"] is True

    base_split = abs(baseline.metadata["T_nu_e_final"] - baseline.metadata["T_nu_x_final"])
    nunu_split = abs(with_nunu.metadata["T_nu_e_final"] - with_nunu.metadata["T_nu_x_final"])
    assert nunu_split < 0.75 * base_split

    # ν-ν equilibration changes the flavour-temperature history but not by
    # inserting an external plasma-energy source.  At the bounded grid it
    # modestly raises the AP-unified FLRW N_eff while staying below the
    # Mangano benchmark.
    assert with_nunu.metadata["N_eff_measured"] > baseline.metadata["N_eff_measured"]
    assert with_nunu.metadata["N_eff_measured"] < 3.044
