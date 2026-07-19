"""Regression locks for hostile-audit hardening fixes."""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.data import Observation
from rabbit.inference.forward_likelihood import (
    BBNLikelihood,
    BBNPrediction,
    _surface_scope_metadata,
    canonical_forward_solver,
)


class _FakeForwardModel:
    default_params = {}

    def __init__(self, pred: BBNPrediction):
        self._pred = pred

    def predict(self, **params):
        return self._pred


def test_default_likelihood_scores_registry_observation_names():
    """Registry names are Yp/DH; they must not silently score zero terms."""
    pred = BBNPrediction(Yp=999.0, DH=999.0)
    like = BBNLikelihood(_FakeForwardModel(pred))

    ll = like.log_likelihood()

    assert np.isfinite(ll)
    assert ll < -1.0e12


def test_likelihood_accepts_legacy_and_registry_observable_aliases():
    pred = BBNPrediction(Yp=0.25, DH=2.5e-5)
    observations = [
        Observation("Y_p", 0.25, 1.0e-3),
        Observation("D/H", 2.5e-5, 1.0e-7),
        Observation("Yp", 0.25, 1.0e-3),
        Observation("DH", 2.5e-5, 1.0e-7),
    ]
    like = BBNLikelihood(_FakeForwardModel(pred), observations=observations)

    assert like.log_likelihood() == pytest.approx(0.0)


def test_likelihood_rejects_unknown_observable_name():
    pred = BBNPrediction(Yp=0.25, DH=2.5e-5)
    like = BBNLikelihood(
        _FakeForwardModel(pred),
        observations=[Observation("Li7H", 1.0e-10, 1.0e-11)],
    )

    with pytest.raises(ValueError, match="Unsupported BBN observation name"):
        like.log_likelihood()


def test_sampler_likelihood_uses_shared_observable_aliases():
    from rabbit.inference.sampler import BBNLikelihood as SamplerLikelihood

    def solver(params):
        return {"Yp": 0.25, "DH": 2.5e-5}

    like = SamplerLikelihood(
        solver_fn=solver,
        observations=[
            Observation("Y_p", 0.25, 1.0e-3),
            Observation("D/H", 2.5e-5, 1.0e-7),
            Observation("Yp", 0.25, 1.0e-3),
            Observation("DH", 2.5e-5, 1.0e-7),
        ],
    )

    assert like.log_likelihood({}) == pytest.approx(0.0)


def test_public_scipy_tolerances_reach_effective_solver_metadata():
    pred = canonical_forward_solver(
        Sigma_H=0.05,
        backend="scipy",
        N_q=6,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    assert pred.success
    assert pred.metadata["rtol"] == pytest.approx(1.0e-6)
    assert pred.metadata["atol"] == pytest.approx(1.0e-8)
    assert pred.metadata["solver_method_effective"] == "BDF"
    assert pred.metadata["solver_method_requested"] == "BDF"


def test_canonical_surface_metadata_is_narrowly_typeI_characteristic():
    scipy_meta = _surface_scope_metadata(
        CAPABILITY_BY_BACKEND["scipy"],
        transport_mode="characteristic",
        production_authority="raw_characteristic",
    )
    classa_meta = _surface_scope_metadata(
        CAPABILITY_BY_KEY["jax_classA_driver"],
        transport_mode="kappa_cascade_lmax2",
        production_authority="candidate_classA_curved_transport",
    )

    assert scipy_meta["canonical_surface"] == "typeI_characteristic"
    assert classa_meta["canonical_surface"] == "none"


def test_external_cross_code_module_help_runs():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "rabbit.external.run_cross_code", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--code" in out.stdout


def test_makefile_exposes_sync_counts_target():
    from pathlib import Path

    text = Path("Makefile").read_text()
    assert "sync-counts:" in text
    assert "scripts/sync_test_counts.py" in text


def test_alterbbn_docs_do_not_reference_missing_dockerfile_recipe():
    from pathlib import Path

    checked = [
        Path("src/rabbit/external/alterbbn.py"),
        Path("tests/test_cross_code_live.py"),
    ]
    for path in checked:
        text = path.read_text()
        assert "scripts/external/alterbbn.Dockerfile" not in text
        assert "to be added" not in text


def test_typeI_examples_exist_and_are_scoped():
    from pathlib import Path

    forward = Path("examples/typeI_forward_core.py")
    null_grid = Path("examples/typeI_inference_null_grid.py")
    assert forward.exists()
    assert null_grid.exists()
    text = forward.read_text() + "\n" + null_grid.read_text()
    assert "enable_teff" not in text
    assert "QKE calculation" not in text
    assert "Bayes factor" not in text
