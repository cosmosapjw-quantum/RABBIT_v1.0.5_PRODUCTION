from __future__ import annotations

import math

import pytest

from rabbit.thermo.eos_photon_electron import (
    drho_dT_plasma_with_electron_mu,
    electron_positron_energy_density,
    electron_positron_pressure_density,
    pressure_plasma_with_electron_mu,
    qed_delta_pressure_with_electron_mu,
    qed_delta_rho_with_electron_mu,
    rho_photon,
    rho_plasma_with_electron_mu,
)
from rabbit.thermo.nudec_coupled import coupled_3T_rhs, hubble_3T
from rabbit.thermo.qed_eos_exact import (
    delta_P_qed_exact,
    delta_P_qed_exact_with_electron_mu,
    delta_rho_qed_exact,
    delta_rho_qed_exact_with_electron_mu,
    qed_correction_exact,
)


def test_exact_qed_finite_mu_scalar_occupation_reduces_to_zero_mu() -> None:
    T = 0.8
    mu = 0.2

    zero = qed_correction_exact(T, chemical_potential_MeV=0.0)
    finite_mu = qed_correction_exact(T, chemical_potential_MeV=mu)

    assert zero["rho_total"] == pytest.approx(delta_rho_qed_exact(T))
    assert zero["P_total"] == pytest.approx(delta_P_qed_exact(T))
    assert delta_rho_qed_exact_with_electron_mu(T, 0.0) == pytest.approx(delta_rho_qed_exact(T))
    assert delta_P_qed_exact_with_electron_mu(T, 0.0) == pytest.approx(delta_P_qed_exact(T))
    assert delta_rho_qed_exact_with_electron_mu(T, mu) == pytest.approx(
        delta_rho_qed_exact_with_electron_mu(T, -mu)
    )
    assert delta_P_qed_exact_with_electron_mu(T, mu) == pytest.approx(
        delta_P_qed_exact_with_electron_mu(T, -mu)
    )
    assert math.isfinite(finite_mu["rho_total"])
    assert math.isfinite(finite_mu["P_total"])
    assert finite_mu["rho_total"] != pytest.approx(zero["rho_total"])


def test_signed_mu_plasma_eos_exact_scalar_qed_mode() -> None:
    T = 0.8
    mu = 0.2
    model = "exact_finite_mu_scalar"

    expected_rho = (
        rho_photon(T)
        + electron_positron_energy_density(T, mu)
        + delta_rho_qed_exact_with_electron_mu(T, mu)
    )
    expected_pressure = (
        rho_photon(T) / 3.0
        + electron_positron_pressure_density(T, mu)
        + delta_P_qed_exact_with_electron_mu(T, mu)
    )

    assert qed_delta_rho_with_electron_mu(T, mu, qed_correction_model=model) == pytest.approx(
        delta_rho_qed_exact_with_electron_mu(T, mu)
    )
    assert qed_delta_pressure_with_electron_mu(T, mu, qed_correction_model=model) == pytest.approx(
        delta_P_qed_exact_with_electron_mu(T, mu)
    )
    assert rho_plasma_with_electron_mu(T, mu, qed_correction_model=model) == pytest.approx(
        expected_rho
    )
    assert pressure_plasma_with_electron_mu(T, mu, qed_correction_model=model) == pytest.approx(
        expected_pressure
    )
    assert drho_dT_plasma_with_electron_mu(T, mu, qed_correction_model=model) > 0.0
    assert rho_plasma_with_electron_mu(T, mu, qed_correction_model=model) != pytest.approx(
        rho_plasma_with_electron_mu(T, mu)
    )


def test_coupled_3t_thermo_accepts_exact_scalar_qed_mode() -> None:
    model = "exact_finite_mu_scalar"
    H = hubble_3T(
        0.8,
        0.79,
        0.78,
        Sigma_sq=0.01,
        electron_chemical_potential_MeV=0.2,
        qed_correction_model=model,
    )
    rhs = coupled_3T_rhs(
        0.8,
        0.79,
        0.78,
        H_MeV=H,
        electron_chemical_potential_MeV=0.2,
        qed_correction_model=model,
    )

    assert H > 0.0
    assert all(math.isfinite(float(value)) for value in rhs)


def test_exact_scalar_qed_mode_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="qed_correction_model"):
        qed_delta_rho_with_electron_mu(0.8, 0.2, qed_correction_model="bad")
