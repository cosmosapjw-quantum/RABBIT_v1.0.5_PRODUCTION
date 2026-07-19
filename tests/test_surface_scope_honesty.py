import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_public_capabilities_expose_scope_contracts():
    for backend in ["scipy", "auto"]:
        cap = CAPABILITY_BY_BACKEND[backend]
        assert cap.readiness_scope_contract
        assert cap.transport_scope_contract
        assert cap.thermo_scope_contract
        assert cap.collision_scope_contract


def test_scipy_surface_scope_metadata_core():
    pred = canonical_forward_solver(Sigma_H=0.0, backend="scipy", correction_level=0, N_q=6)
    assert pred.success
    assert pred.metadata["runtime_surface_contract"] == "scipy_typeI_core_reference_runtime_v1"
    assert pred.metadata["transport_scope_contract"] == "characteristic_typeI_reference_v1"
    assert pred.metadata["thermo_scope_contract"] == "tier1_baseline_thermo_v1"


def test_scipy_surface_scope_metadata_per_species():
    pred = canonical_forward_solver(
        Sigma_H=0.03,
        backend="scipy",
        correction_level=3,
        N_q=6,
        tier=2,
        enable_collisions=True,
    )
    assert pred.success
    assert pred.metadata["transport_species_mode"] == "per_species"
    assert pred.metadata["runtime_surface_contract"] == "scipy_characteristic_decoupling_backbone_runtime_v1"
    assert pred.metadata["collision_scope_contract"] == "anisotropic_residual_relaxation_v1"
    assert pred.metadata["collision_closure_mode"] == "anisotropic_residual_relaxation_v1"
    assert pred.metadata["residual_rate_calibration_mode"] == "spectrum+blocking+mismatch+distortion_v1"
    assert pred.metadata["thermo_exchange_mode"] == "isotropic_decoupling_backbone_v1"
    assert pred.metadata["weak_background_mode"] == "isotropic_decoupling_backbone_v1"
    assert pred.metadata["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"


@pytest.mark.parametrize("backend", ["jax", "jax_advanced", "jax_characteristic"])
def test_retired_jax_forward_names_are_not_scope_surfaces(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend=backend, N_q=6)
