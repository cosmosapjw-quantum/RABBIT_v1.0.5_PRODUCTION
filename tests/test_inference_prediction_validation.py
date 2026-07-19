"""Fail-closed prediction validation shared by every inference boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from rabbit.data import Observation
from rabbit.inference import observables as observable_helpers
from rabbit.inference.forward_likelihood import (
    BBNLikelihood,
    BBNPrediction,
    ForwardModel,
)
from rabbit.inference import sampler


_YP_OBS = Observation("Yp", 0.245, 0.004)
_DH_OBS = Observation("DH", 2.5e-5, 3e-7)


@pytest.mark.parametrize(
    "prediction",
    [
        BBNPrediction(
            Yp=0.245,
            DH=2.5e-5,
            metadata={"surrogate": True},
        ),
        {
            "Yp": 0.245,
            "DH": 2.5e-5,
            "success": True,
            "metadata": {"surrogate": True},
        },
        {
            "Yp": 0.245,
            "DH": 2.5e-5,
            "success": True,
            "metadata": {"surrogate": np.array(True)},
        },
        {
            "Yp": 0.245,
            "DH": 2.5e-5,
            "success": True,
            "metadata": {"surrogate": 1},
        },
    ],
)
def test_scalar_validator_rejects_surrogate_objects_and_mappings(prediction):
    with pytest.raises(ValueError):
        observable_helpers.validate_prediction_for_inference(prediction)


@pytest.mark.parametrize(
    "prediction",
    [
        BBNPrediction(Yp=0.245, DH=2.5e-5, success=False),
        BBNPrediction(Yp=np.nan, DH=2.5e-5),
        BBNPrediction(Yp=0.245, DH=np.inf),
        {"DH": 2.5e-5, "success": True},
        {"Yp": 0.245, "success": True},
        {"Yp": np.nan, "DH": 2.5e-5, "success": True},
        {"Yp": 0.245, "DH": -np.inf, "success": True},
        {"Yp": 0.245, "DH": 2.5e-5, "success": False},
        {"Yp": 0.245, "DH": 2.5e-5, "metadata": {}},
    ],
)
def test_scalar_validator_returns_false_for_failed_missing_or_nonfinite_predictions(
    prediction,
):
    assert observable_helpers.validate_prediction_for_inference(prediction) is False


@pytest.mark.parametrize("success", [1, 0, np.nan, np.inf, "yes"])
def test_scalar_validator_rejects_nonboolean_success(success):
    prediction = {
        "Yp": 0.245,
        "DH": 2.5e-5,
        "success": success,
        "metadata": {},
    }

    assert observable_helpers.validate_prediction_for_inference(prediction) is False


@pytest.mark.parametrize("value", [None, np.ma.masked, np.ma.array(0.245, mask=True)])
def test_scalar_validator_rejects_missing_or_masked_required_value(value):
    prediction = BBNPrediction(Yp=value, DH=2.5e-5, success=True)
    assert observable_helpers.validate_prediction_for_inference(prediction) is False

    model = _model_returning(prediction)
    assert model.log_likelihood(_YP_OBS) == -np.inf
    assert BBNLikelihood(model, observations=[_YP_OBS]).log_likelihood() == -np.inf


def test_validator_requires_explicit_object_success_and_mapping_metadata():
    missing_success = SimpleNamespace(Yp=0.245, DH=2.5e-5, metadata={})
    assert observable_helpers.validate_prediction_for_inference(missing_success) is False

    malformed_metadata = {
        "Yp": 0.245,
        "DH": 2.5e-5,
        "success": True,
        "metadata": "surrogate=false",
    }
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        observable_helpers.validate_prediction_for_inference(malformed_metadata)


@pytest.mark.parametrize(
    "prediction",
    [
        BBNPrediction(Yp=0.245, DH=2.5e-5),
        {"Yp": 0.245, "DH": 2.5e-5, "success": True, "metadata": {}},
    ],
)
def test_scalar_validator_accepts_valid_predictions(prediction):
    assert observable_helpers.validate_prediction_for_inference(prediction) is True


def test_validator_preserves_public_alias_boundary_and_allows_explicit_raw_field():
    prediction = {
        "Yp": 0.245,
        "DH": 2.5e-5,
        "Li7H": 5e-10,
        "success": True,
        "metadata": {},
    }

    with pytest.raises(ValueError, match="Unsupported BBN observation"):
        observable_helpers.validate_prediction_for_inference(
            prediction,
            observation_names=("Li7H",),
        )

    assert observable_helpers.validate_prediction_for_inference(
        prediction,
        extra_prediction_fields=("Li7H",),
    ) is True


def _model_returning(prediction: BBNPrediction) -> ForwardModel:
    return ForwardModel(lambda **_params: prediction)


def test_forward_and_primary_likelihood_reject_surrogate_predictions():
    pred = BBNPrediction(
        Yp=0.245,
        DH=2.5e-5,
        metadata={"surrogate": True},
    )
    model = _model_returning(pred)

    with pytest.raises(ValueError):
        model.log_likelihood(_YP_OBS)
    with pytest.raises(ValueError):
        BBNLikelihood(model, observations=[_YP_OBS]).log_likelihood()


@pytest.mark.parametrize(
    ("prediction", "observation"),
    [
        (BBNPrediction(Yp=0.245, DH=2.5e-5, success=False), _YP_OBS),
        (BBNPrediction(Yp=np.nan, DH=2.5e-5), _YP_OBS),
        (BBNPrediction(Yp=0.245, DH=np.inf), _DH_OBS),
    ],
)
def test_forward_and_primary_likelihood_return_negative_infinity_for_invalid_predictions(
    prediction,
    observation,
):
    model = _model_returning(prediction)

    assert model.log_likelihood(observation) == -np.inf
    assert BBNLikelihood(model, observations=[observation]).log_likelihood() == -np.inf


def test_scalar_likelihood_validates_only_the_scored_observation():
    model = _model_returning(BBNPrediction(Yp=0.245, DH=np.inf))

    assert model.log_likelihood(_YP_OBS) == 0.0
    assert BBNLikelihood(model, observations=[_YP_OBS]).log_likelihood() == 0.0


def test_valid_nonzero_likelihood_bit_pattern_is_unchanged(monkeypatch):
    import rabbit.inference.forward_likelihood as forward_likelihood

    observation = Observation("Yp", 0.245, 0.004)
    prediction_value = 0.24623456789
    expected_bits = np.uint64(0xBFA862F35E325D5A)
    prediction = BBNPrediction(Yp=prediction_value, DH=2.5e-5)
    model = _model_returning(prediction)

    scalar_scores = (
        model.log_likelihood(observation),
        BBNLikelihood(model, observations=[observation]).log_likelihood(),
        sampler.BBNLikelihood(
            solver_fn=lambda _params: {
                "Yp": prediction_value,
                "DH": 2.5e-5,
            },
            observations=[observation],
        ).log_likelihood({}),
    )
    for score in scalar_scores:
        assert np.float64(score).view(np.uint64) == expected_bits

    monkeypatch.setattr(
        forward_likelihood,
        "canonical_batch_forward_solver",
        lambda *_args, **_kwargs: {
            "Yp": np.array([prediction_value, prediction_value]),
            "success": np.array([True, True]),
            "metadata": {},
        },
    )
    batch = forward_likelihood.canonical_batch_log_likelihood(
        np.array([0.0, 1e-4]),
        observations=[observation],
    )
    np.testing.assert_array_equal(
        np.asarray(batch["log_likelihood"], dtype=np.float64).view(np.uint64),
        np.array([expected_bits, expected_bits], dtype=np.uint64),
    )


def test_sampler_adapter_preserves_validation_fields_and_fails_closed():
    failed = BBNPrediction(
        Yp=0.245,
        DH=2.5e-5,
        success=False,
        metadata={"failure_reason": "test-only"},
    )

    mapped = sampler._prediction_to_observable_dict(failed)

    assert mapped["success"] is False
    assert mapped["metadata"] == {"failure_reason": "test-only"}
    assert sampler.BBNLikelihood(solver_fn=lambda _params: failed).log_likelihood({}) == -np.inf

    surrogate = BBNPrediction(
        Yp=0.245,
        DH=2.5e-5,
        metadata={"surrogate": True},
    )
    with pytest.raises(ValueError):
        sampler.BBNLikelihood(solver_fn=lambda _params: surrogate).log_likelihood({})

    mapping_metadata_surrogate = BBNPrediction(
        Yp=0.245,
        DH=2.5e-5,
        metadata=MappingProxyType({"surrogate": True}),
    )
    with pytest.raises(ValueError, match="SURROGATE"):
        sampler._prediction_to_observable_dict(mapping_metadata_surrogate)
    with pytest.raises(ValueError, match="SURROGATE"):
        sampler.BBNLikelihood(
            solver_fn=lambda _params: mapping_metadata_surrogate
        ).log_likelihood({})

    missing_success = {"Yp": 0.245, "DH": 2.5e-5, "metadata": {}}
    normalized = sampler._prediction_to_observable_dict(missing_success)
    assert normalized["success"] is True
    assert np.isfinite(
        sampler.BBNLikelihood(
            solver_fn=lambda _params: missing_success
        ).log_likelihood({})
    )


@pytest.mark.parametrize(
    "invalid_value",
    [None, "not-a-number", [0.245, 0.246], 0.245 + 0.1j, True],
)
def test_sampler_object_adapter_defers_invalid_payloads_to_validator(invalid_value):
    invalid_yp = BBNPrediction(Yp=invalid_value, DH=2.5e-5)
    assert (
        sampler.BBNLikelihood(
            solver_fn=lambda _params: invalid_yp
        ).log_likelihood({})
        == -np.inf
    )

    invalid_li7 = BBNPrediction(
        Yp=0.245,
        DH=2.5e-5,
        metadata={"Li7H": invalid_value},
    )
    assert (
        sampler.BBNLikelihood(
            solver_fn=lambda _params: invalid_li7,
            use_Li7=True,
        ).log_likelihood({})
        == -np.inf
    )


def test_sampler_rejects_grid_emulator_and_invalid_li7():
    class _GridEmulatorStub:
        def predict_all(self, _params):
            raise AssertionError("known surrogate must be rejected before interpolation")

    with pytest.raises(ValueError, match="SURROGATE"):
        sampler.BBNLikelihood(emulator=_GridEmulatorStub()).log_likelihood({})

    invalid_li7 = {
        "Yp": 0.245,
        "DH": 2.5e-5,
        "Li7H": np.nan,
        "success": True,
        "metadata": {},
    }
    likelihood = sampler.BBNLikelihood(
        solver_fn=lambda _params: invalid_li7,
        use_Li7=True,
    )
    assert likelihood.log_likelihood({}) == -np.inf


def test_grid_emulator_provenance_survives_bound_and_wrapped_solver_routes():
    config = sampler.GridEmulatorConfig(
        param_ranges={"eta": (6.0e-10, 6.2e-10, 2)},
        observables=("Yp", "DH"),
    )
    emulator = sampler.GridEmulator(
        config,
        {"eta": np.array([6.0e-10, 6.2e-10])},
        {
            "Yp": np.array([0.245, 0.246]),
            "DH": np.array([2.5e-5, 2.4e-5]),
        },
    )

    prediction = emulator.predict_all({"eta": 6.1e-10})
    assert prediction["success"] is True
    assert prediction["metadata"] == {
        "surrogate": True,
        "surrogate_kind": "grid_emulator",
    }

    solver_routes = (
        emulator.predict_all,
        lambda params: emulator.predict_all(params),
    )
    for solver_fn in solver_routes:
        with pytest.raises(ValueError, match="SURROGATE"):
            sampler.BBNLikelihood(solver_fn=solver_fn).log_likelihood(
                {"eta": 6.1e-10}
            )


def test_sampler_rejects_known_surrogate_before_observable_access():
    class GuardedSurrogateMapping(Mapping):
        def __iter__(self):
            return iter(("metadata", "Yp", "DH"))

        def __len__(self):
            return 3

        def __getitem__(self, key):
            if key == "metadata":
                return {"surrogate": True}
            raise AssertionError(f"surrogate observable {key} was accessed")

    with pytest.raises(ValueError, match="SURROGATE"):
        sampler._prediction_to_observable_dict(GuardedSurrogateMapping())


def test_batch_likelihood_masks_failed_and_nonfinite_rows(monkeypatch):
    import rabbit.inference.forward_likelihood as forward_likelihood

    prediction = {
        "Yp": np.array([0.245, np.nan, 0.245, 0.245]),
        "DH": np.array([2.5e-5, 2.5e-5, np.inf, 2.5e-5]),
        "success": np.array([True, True, True, False]),
        "metadata": {},
    }
    expected = np.array([True, False, False, False])

    np.testing.assert_array_equal(
        observable_helpers.prediction_valid_mask_for_inference(prediction),
        expected,
    )
    monkeypatch.setattr(
        forward_likelihood,
        "canonical_batch_forward_solver",
        lambda *_args, **_kwargs: prediction,
    )

    result = forward_likelihood.canonical_batch_log_likelihood([0.0] * 4)

    np.testing.assert_array_equal(np.asarray(result["valid_mask"]), expected)
    assert np.isfinite(np.asarray(result["log_likelihood"])[0])
    assert np.all(np.isneginf(np.asarray(result["log_likelihood"])[1:]))


def test_batch_likelihood_sanitizes_failed_extreme_payload_before_arithmetic(
    monkeypatch,
):
    import rabbit.inference.forward_likelihood as forward_likelihood

    prediction = {
        "Yp": np.array([0.245, np.finfo(np.float64).max]),
        "DH": np.array([2.5e-5, np.finfo(np.float64).max]),
        "success": np.array([True, False]),
        "metadata": {},
    }
    monkeypatch.setattr(
        forward_likelihood,
        "canonical_batch_forward_solver",
        lambda *_args, **_kwargs: prediction,
    )

    result = forward_likelihood.canonical_batch_log_likelihood([0.0, 1e-4])
    scores = np.asarray(result["log_likelihood"])

    assert np.isfinite(scores[0])
    assert np.isneginf(scores[1])
    assert not np.any(np.isnan(scores))


def test_batch_validator_rejects_nonboolean_success_without_casting():
    prediction = {
        "Yp": np.array([0.245, 0.245]),
        "DH": np.array([2.5e-5, 2.5e-5]),
        "success": np.array([1.0, np.nan]),
        "metadata": {},
    }

    np.testing.assert_array_equal(
        observable_helpers.prediction_valid_mask_for_inference(prediction),
        np.array([False, False]),
    )


def test_batch_validator_preserves_masked_array_invalidity():
    prediction = {
        "Yp": np.ma.array([0.245, 0.246], mask=[False, True]),
        "DH": np.array([2.5e-5, 2.5e-5]),
        "success": np.array([True, True]),
        "metadata": {},
    }

    np.testing.assert_array_equal(
        observable_helpers.prediction_valid_mask_for_inference(prediction),
        np.array([True, False]),
    )


@pytest.mark.parametrize(
    ("success", "expected"),
    [
        (True, np.array([True, True])),
        (np.array([True, False]), np.array([True, False])),
    ],
    ids=["scalar-status-broadcast", "exact-status-mask"],
)
def test_batch_validator_expected_shape_accepts_only_scalar_or_exact_status(
    success,
    expected,
):
    prediction = {
        "Yp": np.array([0.245, 0.246]),
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": success,
        "metadata": {},
    }

    valid = observable_helpers.prediction_valid_mask_for_inference(
        prediction,
        expected_shape=(2,),
    )

    assert np.asarray(valid).shape == (2,)
    np.testing.assert_array_equal(valid, expected)


@pytest.mark.parametrize(
    "success",
    [np.array([True]), np.ones((2, 1), dtype=bool)],
    ids=["singleton", "cross-shape"],
)
def test_batch_validator_expected_shape_rejects_malformed_status(success):
    prediction = {
        "Yp": np.array([0.245, 0.246]),
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": success,
        "metadata": {},
    }

    valid = observable_helpers.prediction_valid_mask_for_inference(
        prediction,
        expected_shape=(2,),
    )

    assert np.asarray(valid).shape == (2,)
    np.testing.assert_array_equal(valid, np.array([False, False]))


@pytest.mark.parametrize(
    "malformed_yp",
    [0.245, np.array([0.245]), np.array([[0.245], [0.246]])],
    ids=["scalar", "singleton", "cross-shape"],
)
def test_batch_validator_expected_shape_rejects_malformed_observable(
    malformed_yp,
):
    prediction = {
        "Yp": malformed_yp,
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": np.array([True, False]),
        "metadata": {},
    }

    valid = observable_helpers.prediction_valid_mask_for_inference(
        prediction,
        expected_shape=(2,),
    )

    assert np.asarray(valid).shape == (2,)
    np.testing.assert_array_equal(valid, np.array([False, False]))


@pytest.mark.parametrize("three_dimensional", [False, True])
@pytest.mark.parametrize("missing_key", ["Yp", "DH"])
def test_batch_likelihood_returns_negative_infinity_for_missing_observable(
    monkeypatch,
    three_dimensional,
    missing_key,
):
    import rabbit.inference.forward_likelihood as forward_likelihood

    prediction = {
        "Yp": np.array([0.245, 0.246]),
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": np.array([True, True]),
        "metadata": {},
    }
    prediction.pop(missing_key)

    if three_dimensional:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver_3d",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood_3d(
            np.array([[0.0, 6.1e-10, 878.4], [1e-4, 6.1e-10, 878.4]])
        )
    else:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood(
            np.array([0.0, 1e-4])
        )

    assert np.all(np.isneginf(np.asarray(result["log_likelihood"])))


@pytest.mark.parametrize("three_dimensional", [False, True])
@pytest.mark.parametrize(
    "invalid_value",
    [None, "not-a-number", object(), 0.245 + 0.1j, True],
)
def test_batch_likelihood_safely_rejects_uncoercible_required_value(
    monkeypatch,
    three_dimensional,
    invalid_value,
):
    import rabbit.inference.forward_likelihood as forward_likelihood

    prediction = {
        "Yp": invalid_value,
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": np.array([True, True]),
        "metadata": {},
    }
    if three_dimensional:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver_3d",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood_3d(
            np.array([[0.0, 6.1e-10, 878.4], [1e-4, 6.1e-10, 878.4]])
        )
    else:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood(
            np.array([0.0, 1e-4])
        )

    assert np.all(np.isneginf(np.asarray(result["log_likelihood"])))


@pytest.mark.parametrize("three_dimensional", [False, True])
def test_batch_likelihood_broadcasts_scalar_boolean_success_without_drift(
    monkeypatch,
    three_dimensional,
):
    import rabbit.inference.forward_likelihood as forward_likelihood

    observation = Observation("Yp", 0.245, 0.004)
    prediction = {
        "Yp": np.array([0.24623456789, 0.247654321]),
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": True,
        "metadata": {},
    }
    if three_dimensional:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver_3d",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood_3d(
            np.array([[0.0, 6.1e-10, 878.4], [1e-4, 6.1e-10, 878.4]]),
            observations=[observation],
        )
    else:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood(
            np.array([0.0, 1e-4]),
            observations=[observation],
        )

    expected_bits = np.array(
        [0xBFA862F35E325D5A, 0xBFCC2E8292A8D3AA],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(
        np.asarray(result["log_likelihood"], dtype=np.float64).view(np.uint64),
        expected_bits,
    )


@pytest.mark.parametrize("three_dimensional", [False, True])
@pytest.mark.parametrize(
    "malformed_value",
    [0.245, np.array([0.245]), np.array([0.245, 0.246, 0.247])],
    ids=["scalar", "singleton", "wrong-length"],
)
def test_batch_likelihood_rejects_observable_shape_broadcast(
    monkeypatch,
    three_dimensional,
    malformed_value,
):
    """One solver value must never masquerade as multiple batch rows."""
    import rabbit.inference.forward_likelihood as forward_likelihood

    prediction = {
        "Yp": malformed_value,
        "DH": np.array([2.5e-5, 2.4e-5]),
        "success": True,
        "metadata": {},
    }
    if three_dimensional:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver_3d",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood_3d(
            np.array([[0.0, 6.1e-10, 878.4], [1e-4, 6.1e-10, 878.4]])
        )
    else:
        monkeypatch.setattr(
            forward_likelihood,
            "canonical_batch_forward_solver",
            lambda *_args, **_kwargs: prediction,
        )
        result = forward_likelihood.canonical_batch_log_likelihood(
            np.array([0.0, 1e-4])
        )

    np.testing.assert_array_equal(
        np.asarray(result["valid_mask"]), np.array([False, False])
    )
    assert np.all(np.isneginf(np.asarray(result["log_likelihood"])))


def test_legacy_likelihood_factories_reject_successful_nonfinite_solve(monkeypatch):
    import rabbit.inference.bbn_inference as bbn_inference

    monkeypatch.setattr(
        bbn_inference,
        "_scipy_forward_solve",
        lambda *_args, **_kwargs: {
            "Yp": np.nan,
            "DH": 2.5e-5,
            "success": True,
        },
    )

    loglike = bbn_inference.make_log_likelihood(backend="scipy")
    sigma_loglike = bbn_inference.make_log_likelihood_sigma(backend="scipy")

    assert loglike(np.array([6.104, 878.4])) == -np.inf
    assert sigma_loglike(np.array([0.0])) == -np.inf


def test_legacy_joint_posterior_rejects_successful_nonfinite_solve(monkeypatch):
    import jax.numpy as jnp
    import rabbit.inference.bbn_inference as bbn_inference
    from rabbit.inference.joint_3d_inference import (
        JointInference3DConfig,
        make_log_posterior_3d,
    )

    monkeypatch.setattr(
        bbn_inference,
        "_scipy_forward_solve",
        lambda *_args, **_kwargs: {
            "Yp": np.nan,
            "DH": 2.5e-5,
            "success": True,
        },
    )
    log_posterior = make_log_posterior_3d(
        JointInference3DConfig(
            backend="scipy",
            correction_level=0,
            N_q=6,
        )
    )

    assert float(log_posterior(jnp.array([0.0, 6.104, 878.4]))) == -np.inf


def test_bbn_jax_sampler_wrappers_fail_before_solver_or_engine(monkeypatch):
    """All five convenience wrappers expose one frozen B-05 boundary."""
    import rabbit.inference.bbn_inference as bbn_inference
    import rabbit.inference.forward_likelihood as forward_likelihood
    import rabbit.inference.jax_nested as jax_nested
    import rabbit.inference.jax_nuts as jax_nuts
    import rabbit.inference.joint_3d_inference as joint_3d

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("wrapper crossed the frozen B-05 boundary")

    monkeypatch.setattr(forward_likelihood, "canonical_forward_solver", forbidden_call)
    monkeypatch.setattr(bbn_inference, "_jax_forward_solve", forbidden_call)
    monkeypatch.setattr(jax_nuts, "run_nuts", forbidden_call)
    monkeypatch.setattr(jax_nested, "run_nss", forbidden_call)
    monkeypatch.setattr(joint_3d, "make_log_posterior_3d", forbidden_call)

    calls = (
        lambda: jax_nuts.run_bbn_nuts(num_warmup=1, num_samples=1),
        lambda: jax_nested.run_bbn_nss(n_live=2, max_iterations=1),
        lambda: bbn_inference.run_bbn_nuts_production(),
        lambda: bbn_inference.run_bbn_nss_production(),
        lambda: joint_3d.run_bbn_nuts_3d(),
    )
    expected = observable_helpers.BBN_JAX_SAMPLER_UNAVAILABLE
    for call in calls:
        with pytest.raises(RuntimeError) as exc_info:
            call()
        assert str(exc_info.value) == expected


def test_retired_nuts_3d_script_fails_without_creating_artifact(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_nuts_3d_demo.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path / "forbidden.npz")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == observable_helpers.BBN_JAX_SAMPLER_UNAVAILABLE
    assert not (tmp_path / "forbidden.npz").exists()
