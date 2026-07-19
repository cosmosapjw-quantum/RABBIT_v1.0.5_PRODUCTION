"""Null-limit locks for the surviving SciPy forward surface and F06 fences."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
from rabbit.inference.forward_likelihood import canonical_forward_solver


FIXTURE = Path(__file__).parent / "fixtures" / "flrw_gold_v861.json"
_SMOKE = dict(N_q=20, n_reactions=12, correction_level=0)


def _gold_cl0_12():
    with FIXTURE.open() as stream:
        gold = json.load(stream)
    for entry in gold["entries"]:
        if entry["n_rxn"] == 12 and entry["CL"] == 0:
            return entry
    raise ValueError("Gold CL0/12 not found")


class TestIsotropicRecovery:
    def test_sigma_zero_matches_gold(self):
        gold = _gold_cl0_12()
        result = run_full_coupled_typeI(
            FullCoupledConfig(
                Sigma_H_plus=0.0,
                Sigma_H_minus=0.0,
                enable_teff=False,
                **_SMOKE,
            )
        )
        assert abs(result.observables.Yp - gold["Yp"]) < 1e-6
        assert abs(result.observables.DH - gold["DH"]) / gold["DH"] < 0.005

    def test_sigma_zero_both_polarizations_is_physical(self):
        result = run_full_coupled_typeI(
            FullCoupledConfig(
                Sigma_H_plus=0.0,
                Sigma_H_minus=0.0,
                enable_teff=False,
                **_SMOKE,
            )
        )
        assert result.observables.Yp > 0.24


class TestTeffRecovery:
    def test_scipy_public_surface_rejects_deprecated_teff(self):
        result = canonical_forward_solver(
            Sigma_H=0.0, N_q=20, backend="scipy", enable_teff=False
        )
        assert result.success is True
        with pytest.raises(ValueError, match="enable_teff=True is deprecated legacy"):
            canonical_forward_solver(
                Sigma_H=0.0, N_q=20, backend="scipy", enable_teff=True
            )


class TestRetiredPublicBianchiNullSurfaces:
    @pytest.mark.parametrize(
        "backend",
        ["jax_classA", "jax_classB", "jax_tilted", "jax_tilted_full_coupled"],
    )
    def test_backend_name_is_rejected_before_geometry_kwargs(self, backend):
        with pytest.raises(ValueError, match="retired from the public forward surface"):
            canonical_forward_solver(Sigma_H=0.0, backend=backend)

    def test_geometry_and_tilt_kwargs_are_absent_from_public_signature(self):
        params = inspect.signature(canonical_forward_solver).parameters
        for name in (
            "bianchi_type",
            "Sigma_H_minus",
            "N1_init",
            "N2_init",
            "N3_init",
            "A_init",
            "h",
            "v0",
        ):
            assert name not in params


class TestAnisotropicDirection:
    def test_shear_increases_yp(self):
        baseline = run_full_coupled_typeI(
            FullCoupledConfig(Sigma_H_plus=0.0, enable_teff=False, **_SMOKE)
        )
        sheared = run_full_coupled_typeI(
            FullCoupledConfig(Sigma_H_plus=0.1, enable_teff=False, **_SMOKE)
        )
        assert sheared.observables.Yp > baseline.observables.Yp

    def test_shear_sensitivity_formula(self):
        baseline = run_full_coupled_typeI(
            FullCoupledConfig(Sigma_H_plus=0.0, enable_teff=False, **_SMOKE)
        )
        sheared = run_full_coupled_typeI(
            FullCoupledConfig(Sigma_H_plus=0.1, enable_teff=False, **_SMOKE)
        )
        delta_yp = sheared.observables.Yp - baseline.observables.Yp
        expected = 0.0042 * np.log(1.0 / (1.0 - 0.1**2))
        assert 0.2 < delta_yp / expected < 5.0
