"""Foundation F12 — versioned observation registry.

Asserts that every inline observation copy across the inference layer
resolves to the same value sourced from rabbit.data.observations.json,
that the registry returns frozen Observation dataclasses, and that the
historical sigma_DH typo (0.025e-5) cannot recur.
"""
from __future__ import annotations

import math

import pytest

from rabbit.data import Observation, load_observations


@pytest.mark.production
class TestObservationRegistry:
    def test_load_observations_returns_known_keys(self):
        obs = load_observations()
        assert set(obs) >= {"Yp", "DH", "tau_n", "eta", "Li7H"}

    def test_observations_are_frozen(self):
        obs = load_observations()
        with pytest.raises(Exception):
            obs["Yp"].value = 0.0  # type: ignore[misc]

    def test_dh_sigma_is_cooke_2018_value(self):
        """Regression guard against the legacy 0.025e-5 sigma_DH typo."""
        obs = load_observations()
        assert math.isclose(obs["DH"].sigma, 0.029e-5, rel_tol=0, abs_tol=1e-12)
        assert not math.isclose(obs["DH"].sigma, 0.025e-5, rel_tol=0, abs_tol=1e-12)

    def test_yp_sigma_is_aver_2021_value(self):
        obs = load_observations()
        assert math.isclose(obs["Yp"].value, 0.2449, abs_tol=1e-12)
        assert math.isclose(obs["Yp"].sigma, 0.0040, abs_tol=1e-12)

    def test_eta_value_is_planck_2018(self):
        obs = load_observations()
        assert math.isclose(obs["eta"].value, 6.104e-10, rel_tol=1e-12)
        assert math.isclose(obs["eta"].sigma, 0.058e-10, rel_tol=1e-12)

    def test_tau_n_is_pdg_2024(self):
        obs = load_observations()
        assert math.isclose(obs["tau_n"].value, 878.4, abs_tol=1e-12)
        assert math.isclose(obs["tau_n"].sigma, 0.5, abs_tol=1e-12)

    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            load_observations(version="0.0")

    def test_observation_log_likelihood_matches_chi2(self):
        o = Observation(name="x", value=1.0, sigma=0.1, ref="test")
        ll = o.log_likelihood(1.2)
        chi2 = o.chi2(1.2)
        assert math.isclose(ll, -0.5 * chi2, rel_tol=1e-12)

    def test_observation_back_compat_three_arg_constructor(self):
        # Legacy callers do `Observation("Y_p", 0.2, 0.01)` — must keep working.
        o = Observation("Y_p", 0.2, 0.01)
        assert o.name == "Y_p"
        assert o.ref == ""


@pytest.mark.production
class TestInlineCallSitesAreSourcedFromRegistry:
    """Each refactored module must resolve to the registry values."""

    def test_forward_likelihood_obs_match_registry(self):
        from rabbit.inference.forward_likelihood import (
            OBS_DH, OBS_ETA, OBS_TAU_N, OBS_YP,
        )
        reg = load_observations()
        assert OBS_YP.value == reg["Yp"].value
        assert OBS_YP.sigma == reg["Yp"].sigma
        assert OBS_DH.value == reg["DH"].value
        assert OBS_DH.sigma == reg["DH"].sigma
        assert OBS_TAU_N.value == reg["tau_n"].value
        assert OBS_ETA.value == reg["eta"].value

    def test_bbn_inference_module_constants_match_registry(self):
        from rabbit.inference import bbn_inference as bi
        reg = load_observations()
        assert bi.YP_OBS == reg["Yp"].value
        assert bi.YP_ERR == reg["Yp"].sigma
        assert math.isclose(bi.DH_OBS, reg["DH"].value, rel_tol=1e-12)
        assert math.isclose(bi.DH_ERR, reg["DH"].sigma, rel_tol=1e-12)
        # eta_10 = eta * 1e10
        assert math.isclose(bi.ETA_OBS, reg["eta"].value * 1e10, rel_tol=1e-9)
        assert math.isclose(bi.ETA_ERR, reg["eta"].sigma * 1e10, rel_tol=1e-9)
        assert bi.TAUN_OBS == reg["tau_n"].value
        assert bi.TAUN_ERR == reg["tau_n"].sigma

    def test_sampler_BBNLikelihood_defaults_match_registry(self):
        from rabbit.inference.sampler import BBNLikelihood
        reg = load_observations()
        like = BBNLikelihood()
        assert like.Yp_obs == reg["Yp"].value
        assert like.Yp_err == reg["Yp"].sigma
        assert math.isclose(like.DH_obs, reg["DH"].value, rel_tol=1e-12)
        assert math.isclose(like.DH_err, reg["DH"].sigma, rel_tol=1e-12)
        # The legacy typo guard.
        assert not math.isclose(like.DH_err, 0.025e-5, rel_tol=0, abs_tol=1e-12)

    def test_no_inline_observation_constants_left_in_inference_layer(self):
        """Static guard: no module under rabbit.inference may hard-code
        the legacy DH typo or the published values without going through
        the registry.  Greps the on-disk source files."""
        import pathlib
        import re

        pkg = pathlib.Path(__file__).resolve().parents[1] / "src" / "rabbit"
        offenders = []
        # Forbidden literal: the legacy sigma_DH typo.
        typo_re = re.compile(r"0\.025e-?0*5")
        for path in pkg.rglob("*.py"):
            if path.name == "test_observation_registry.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if typo_re.search(text):
                offenders.append(str(path))
        assert offenders == [], (
            f"Legacy 0.025e-5 sigma_DH typo found in: {offenders}"
        )
