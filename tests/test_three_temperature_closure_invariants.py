from __future__ import annotations

import numpy as np
import pytest

from rabbit.jax.nudec_coupled_jax import N_eff_from_3T_jax
from rabbit.thermo.nudec_coupled import (
    N_eff_from_3T,
    _drho_nu_pair_dT,
    asymptotic_N_eff_3T_payload,
    coupled_3T_rhs_from_collision_moments,
)


def test_python_and_jax_neff_3t_definitions_match() -> None:
    states = (
        (0.9, 0.9, 0.9),
        (0.05, 0.0362, 0.0362),
        (0.009, 0.00645, 0.00640),
    )

    for T_gamma, T_nu_e, T_nu_x in states:
        expected = N_eff_from_3T(T_gamma, T_nu_e, T_nu_x)
        measured = float(N_eff_from_3T_jax(T_gamma, T_nu_e, T_nu_x))
        assert measured == pytest.approx(expected, rel=2.0e-7, abs=1.0e-10)


def test_neff_3t_degenerate_temperature_is_unavailable_not_plausible() -> None:
    measured = N_eff_from_3T(0.0, 0.0, 0.0)
    measured_jax = float(N_eff_from_3T_jax(0.0, 0.0, 0.0))

    assert np.isnan(measured)
    assert np.isnan(measured_jax)


def test_positive_nux_bank_heat_source_warms_nux_relative_to_free_streaming() -> None:
    T_gamma = 0.8
    T_nu_e = 0.7
    T_nu_x = 0.6
    d2 = 2.0 * _drho_nu_pair_dT(T_nu_x)
    dQ_nux_bank_N = 0.25 * d2 * T_nu_x

    baseline = coupled_3T_rhs_from_collision_moments(
        T_gamma,
        T_nu_e,
        T_nu_x,
        dQ_nue_pair_N=0.0,
        dQ_nux_bank_N=0.0,
        plasma_drho_dT_MeV3=1.0,
        plasma_dT_base_dN=0.0,
    )
    heated = coupled_3T_rhs_from_collision_moments(
        T_gamma,
        T_nu_e,
        T_nu_x,
        dQ_nue_pair_N=0.0,
        dQ_nux_bank_N=dQ_nux_bank_N,
        plasma_drho_dT_MeV3=1.0,
        plasma_dT_base_dN=0.0,
    )

    assert heated[0] < baseline[0]
    assert heated[1] == pytest.approx(baseline[1])
    assert heated[2] > baseline[2]
    assert heated[2] == pytest.approx(-0.75 * T_nu_x)


def test_equal_temperature_nunu_equilibration_source_vanishes() -> None:
    baseline = coupled_3T_rhs_from_collision_moments(
        0.8,
        0.7,
        0.7,
        dQ_nue_pair_N=0.0,
        dQ_nux_bank_N=0.0,
        enable_nu_nu_equilibration=False,
        plasma_drho_dT_MeV3=1.0,
        plasma_dT_base_dN=0.0,
    )
    with_nunu = coupled_3T_rhs_from_collision_moments(
        0.8,
        0.7,
        0.7,
        dQ_nue_pair_N=0.0,
        dQ_nux_bank_N=0.0,
        enable_nu_nu_equilibration=True,
        plasma_drho_dT_MeV3=1.0,
        plasma_dT_base_dN=0.0,
    )

    np.testing.assert_allclose(with_nunu, baseline, rtol=0.0, atol=1.0e-14)


def test_asymptotic_neff_tail_reports_clean_no_qke_band() -> None:
    payload = asymptotic_N_eff_3T_payload(3.0, 3.0, 3.0)

    assert payload["available"] is True
    assert payload["tail_integrated"] is True
    assert payload["N_eff_3T_in_span"] > 10.0
    assert 3.00 <= payload["N_eff_3T_asymptotic"] <= 3.06
    assert payload["N_eff_3T_asymptotic_floor_band_passed"] is True
    assert payload["N_eff_3T_claim_status"] == "diagnostic_proxy_not_physical_N_eff"
    assert payload["N_eff_3T_asymptotic_readout_scope"] == (
        "thermo_only_tail_no_network_no_phase2"
    )


def test_asymptotic_neff_tail_preserves_bridge_excess_signal() -> None:
    payload = asymptotic_N_eff_3T_payload(
        0.08735709565954111,
        0.06572257387709174,
        0.06567977286950767,
    )

    assert payload["available"] is True
    assert payload["N_eff_3T_in_span"] == pytest.approx(3.6966604217611163)
    assert payload["N_eff_3T_asymptotic"] > 3.06
    assert payload["N_eff_3T_asymptotic_floor_band_passed"] is False
    assert payload["N_eff_3T_asymptotic_delta_from_expected"] > 0.07


def test_asymptotic_neff_tail_marks_unreached_stop_unavailable() -> None:
    payload = asymptotic_N_eff_3T_payload(3.0, 3.0, 3.0, max_tail_N=1.0e-4)

    assert payload["available"] is False
    assert str(payload["unavailable_reason"]).startswith("tail_stop_not_reached:")
    assert payload["N_eff_3T_in_span"] > 10.0
    assert payload["N_eff_3T_asymptotic"] is None
    assert payload["N_eff_3T_asymptotic_floor_band_passed"] is None
