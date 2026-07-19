"""Frozen JAX thermo-provider component parity checks.

These tests intentionally exercise no JAX forward endpoint.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("jax", reason="JAX required")

from rabbit.jax.thermo_provider_jax import Tier1ThermoProvider, Tier2ThermoProvider
from rabbit.thermo.incomplete_decoupling import (
    N_eff_from_T_ratio,
    T_nu_from_T_gamma_tier1,
    dT_gamma_dN_tier1,
)
from rabbit.thermo.nudec_coupled import N_eff_from_3T, coupled_3T_rhs, hubble_3T


@pytest.mark.parametrize("T_gamma", [10.0, 3.0, 1.0, 0.5, 0.1])
def test_tier1_provider_matches_current_flrw_thermo_baseline(T_gamma):
    result = Tier1ThermoProvider(N_eff=3.044)(T_gamma=T_gamma, Sigma_sq=0.0)
    T_nu_ref = T_nu_from_T_gamma_tier1(T_gamma)
    assert math.isclose(result.T_nu_for_weak, T_nu_ref, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(result.T_nu_e, T_nu_ref, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(result.T_nu_x, T_nu_ref, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(
        result.dT_gamma_dN,
        dT_gamma_dN_tier1(T_gamma),
        rel_tol=1e-8,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result.N_eff_effective,
        N_eff_from_T_ratio(T_nu_ref, T_gamma),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert result.tier == 1


@pytest.mark.parametrize(
    "T_gamma,T_nu_e,T_nu_x",
    [(3.0, 3.0, 3.0), (1.5, 1.45, 1.45), (1.0, 0.95, 0.95), (0.5, 0.45, 0.44)],
)
def test_tier2_provider_matches_minimal_3t_reference_subset_flrw(
    T_gamma, T_nu_e, T_nu_x
):
    result = Tier2ThermoProvider()(
        T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, Sigma_sq=0.0
    )
    H_ref = hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=0.0)
    dTg_ref, dTne_ref, dTnx_ref = coupled_3T_rhs(
        T_gamma, T_nu_e, T_nu_x, H_MeV=H_ref
    )
    assert math.isclose(result.hubble, H_ref, rel_tol=2e-7, abs_tol=0.0)
    assert math.isclose(result.dT_gamma_dN, dTg_ref, rel_tol=2e-6, abs_tol=1e-11)
    assert math.isclose(result.dT_nu_e_dN, dTne_ref, rel_tol=2e-6, abs_tol=1e-11)
    assert math.isclose(result.dT_nu_x_dN, dTnx_ref, rel_tol=2e-6, abs_tol=1e-11)
    assert math.isclose(
        result.N_eff_effective,
        N_eff_from_3T(T_gamma, T_nu_e, T_nu_x),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert result.tier == 2
