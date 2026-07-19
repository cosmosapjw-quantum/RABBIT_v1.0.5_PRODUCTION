"""F06 public-Bianchi API retirement lock.

Alternate-geometry source modules remain available for component research;
their initial-data coordinates no longer belong to the public forward API.
"""
from __future__ import annotations

import inspect

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize(
    "backend",
    ["jax_classA", "jax_classB", "jax_tilted", "jax_tilted_full_coupled"],
)
def test_public_bianchi_backend_name_is_retired(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend=backend)


def test_public_forward_signature_has_no_bianchi_initial_data_coordinates():
    params = inspect.signature(canonical_forward_solver).parameters
    retired = {
        "bianchi_type",
        "Sigma_H_minus",
        "N1_init",
        "N2_init",
        "N3_init",
        "structure_scale",
        "A_init",
        "frame_scale",
        "h",
        "v0",
        "tilt_axis",
        "tilt_stress_feedback",
        "tilt_weak_rate_boost",
    }
    assert retired.isdisjoint(params)


def test_alternate_geometry_metadata_is_preserved_but_non_dispatchable():
    retained = {
        "jax_classA_geometry",
        "jax_classA_driver",
        "jax_classA_characteristic",
        "jax_classB_driver",
        "jax_tilted_bbn",
        "jax_tilted_full_coupled",
    }
    assert retained <= set(CAPABILITY_BY_KEY)
    assert set(CAPABILITY_BY_BACKEND) == {"scipy", "auto"}
