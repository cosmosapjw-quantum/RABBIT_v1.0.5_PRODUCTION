from __future__ import annotations

import pytest
import numpy as np

import rabbit.thermo.eos_photon_electron as eos
from rabbit.thermo.eos_photon_electron import (
    charge_neutral_electron_chemical_potential,
    drho_dT,
    drho_dT_plasma_with_electron_mu,
    electron_charge_asymmetry_density,
    electron_charge_asymmetry_susceptibility,
    electron_energy_density,
    electron_number_density,
    electron_pressure_density,
    electron_positron_energy_density,
    electron_positron_pressure_density,
    positron_energy_density,
    positron_number_density,
    positron_pressure_density,
    pressure_electron,
    pressure_plasma,
    pressure_plasma_with_electron_mu,
    rho_photon,
    rho_electron,
    rho_plasma,
    rho_plasma_with_electron_mu,
)


def test_finite_mass_electron_number_density_splits_with_mu() -> None:
    T = 0.8

    assert electron_number_density(T, 0.0) == pytest.approx(positron_number_density(T, 0.0))
    assert electron_charge_asymmetry_density(T, 0.1) > 0.0
    assert electron_charge_asymmetry_density(T, -0.1) < 0.0
    assert electron_number_density(T, 0.1) > positron_number_density(T, 0.1)


def test_finite_mass_electron_number_density_uses_single_charge_spin_degeneracy() -> None:
    T = 50.0
    zeta3 = 1.202056903
    expected_massless_single_charge = 3.0 * zeta3 / (2.0 * np.pi**2) * T**3

    assert electron_number_density(T, 0.0) == pytest.approx(
        expected_massless_single_charge,
        rel=1.0e-4,
    )


def test_finite_mass_electron_positron_energy_and_pressure_split_with_mu() -> None:
    T = 0.8

    assert electron_energy_density(T, 0.0) == pytest.approx(positron_energy_density(T, 0.0))
    assert electron_pressure_density(T, 0.0) == pytest.approx(positron_pressure_density(T, 0.0))
    assert electron_positron_energy_density(T, 0.0) == pytest.approx(rho_electron(T))
    assert electron_positron_pressure_density(T, 0.0) == pytest.approx(pressure_electron(T))
    assert electron_energy_density(T, 0.1) > positron_energy_density(T, 0.1)
    assert electron_pressure_density(T, 0.1) > positron_pressure_density(T, 0.1)


def test_signed_mu_plasma_eos_reduces_to_canonical_at_zero_mu() -> None:
    T = 0.8

    assert rho_plasma_with_electron_mu(T, 0.0) == pytest.approx(rho_plasma(T))
    assert pressure_plasma_with_electron_mu(T, 0.0) == pytest.approx(pressure_plasma(T))
    assert drho_dT_plasma_with_electron_mu(T, 0.0) == pytest.approx(drho_dT(T))
    assert rho_plasma_with_electron_mu(T, 0.2) == pytest.approx(
        rho_plasma_with_electron_mu(T, -0.2)
    )
    assert pressure_plasma_with_electron_mu(T, 0.2) == pytest.approx(
        pressure_plasma_with_electron_mu(T, -0.2)
    )
    assert rho_plasma_with_electron_mu(T, 0.2) > rho_plasma(T)
    assert pressure_plasma_with_electron_mu(T, 0.2) > pressure_plasma(T)


def test_signed_mu_qed_correction_tracks_finite_mu_electron_bath() -> None:
    T = 0.8
    mu = 0.2
    qed_mu = getattr(eos, "qed_delta_rho_with_electron_mu", None)

    assert callable(qed_mu)
    assert qed_mu(T, 0.0) == pytest.approx(eos.qed_delta_rho(T))
    assert qed_mu(T, mu) == pytest.approx(qed_mu(T, -mu))
    assert qed_mu(T, mu) > eos.qed_delta_rho(T)


def test_signed_mu_plasma_eos_uses_finite_mu_qed_correction() -> None:
    T = 0.8
    mu = 0.2
    qed_mu = getattr(eos, "qed_delta_rho_with_electron_mu", None)

    assert callable(qed_mu)
    expected_rho = rho_photon(T) + electron_positron_energy_density(T, mu) + qed_mu(T, mu)
    expected_pressure = (
        rho_photon(T) / 3.0
        + electron_positron_pressure_density(T, mu)
        + qed_mu(T, mu) / 3.0
    )
    zero_mu_qed_rho = rho_photon(T) + electron_positron_energy_density(T, mu) + eos.qed_delta_rho(T)

    assert rho_plasma_with_electron_mu(T, mu) == pytest.approx(expected_rho)
    assert pressure_plasma_with_electron_mu(T, mu) == pytest.approx(expected_pressure)
    assert rho_plasma_with_electron_mu(T, mu) > zero_mu_qed_rho


def test_charge_neutral_electron_chemical_potential_solves_target_density() -> None:
    T = 0.8
    target = 1.0e-10

    mu_e = charge_neutral_electron_chemical_potential(T, target)

    assert mu_e > 0.0
    assert electron_charge_asymmetry_density(T, mu_e) == pytest.approx(target, rel=1.0e-6, abs=1.0e-18)


def test_charge_asymmetry_susceptibility_matches_finite_difference() -> None:
    T = 0.8
    step = 1.0e-5

    finite_difference = (
        electron_charge_asymmetry_density(T, step)
        - electron_charge_asymmetry_density(T, -step)
    ) / (2.0 * step)

    assert electron_charge_asymmetry_susceptibility(T) == pytest.approx(
        finite_difference,
        rel=1.0e-7,
    )


def test_charge_neutral_mu_uses_linear_response_for_bbn_scale_targets(monkeypatch) -> None:
    T = 0.8
    target = 1.0e-12
    original = eos.electron_charge_asymmetry_density
    calls = {"count": 0}

    def counted_asymmetry(T_MeV: float, chemical_potential_MeV: float) -> float:
        calls["count"] += 1
        return original(T_MeV, chemical_potential_MeV)

    monkeypatch.setattr(eos, "electron_charge_asymmetry_density", counted_asymmetry)

    mu_e = eos.charge_neutral_electron_chemical_potential(T, target)

    assert calls["count"] == 0
    assert mu_e == pytest.approx(
        target / electron_charge_asymmetry_susceptibility(T),
        rel=1.0e-12,
    )
    assert original(T, mu_e) == pytest.approx(target, rel=1.0e-4, abs=1.0e-24)


def test_charge_neutral_electron_chemical_potential_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="T_MeV"):
        charge_neutral_electron_chemical_potential(0.0, 1.0e-10)
    with pytest.raises(ValueError, match="positive_charge_density"):
        charge_neutral_electron_chemical_potential(0.8, float("nan"))
    with pytest.raises(ValueError, match="chemical_potential_MeV"):
        electron_energy_density(0.8, float("nan"))
