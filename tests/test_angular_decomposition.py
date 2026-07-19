from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MultipoleSpec
from rabbit.transport.angular_decomposition import (
    AngularDecompositionSpec,
    AngularMode,
    CollisionMomentResult,
    CollisionQuadratureSpec,
    EllMaxConvergenceSpec,
    FIXED_NON_LRS_S2_THREE_MODE_PROJECTION_CONTRACT,
    active_lrs_modes,
    active_non_lrs_diagonal_modes,
)


def test_lrs_modes_are_even_axisymmetric() -> None:
    modes = active_lrs_modes(6)

    assert [mode.ell for mode in modes] == [0, 2, 4, 6]
    assert all(mode.m == 0 for mode in modes)
    assert all(mode.parity == "axisymmetric" for mode in modes)


def test_non_lrs_diagonal_modes_use_even_cosine_real_basis() -> None:
    modes = active_non_lrs_diagonal_modes(4)

    assert modes == (
        AngularMode(0, 0, "axisymmetric"),
        AngularMode(2, 0, "axisymmetric"),
        AngularMode(2, 2, "cos"),
        AngularMode(4, 0, "axisymmetric"),
        AngularMode(4, 2, "cos"),
        AngularMode(4, 4, "cos"),
    )


def test_non_lrs_diagnostic_sine_partners_are_explicit_opt_in() -> None:
    modes = active_non_lrs_diagonal_modes(2, include_sine=True)

    assert AngularMode(2, 2, "cos") in modes
    assert AngularMode(2, 2, "sin") in modes


def test_angular_mode_rejects_invalid_real_basis_labels() -> None:
    with pytest.raises(ValueError, match="m cannot exceed ell"):
        AngularMode(ell=2, m=3, parity="cos")

    with pytest.raises(ValueError, match="m=0 modes"):
        AngularMode(ell=2, m=0, parity="cos")

    with pytest.raises(ValueError, match="m>0 real modes"):
        AngularMode(ell=2, m=2, parity="axisymmetric")


def test_angular_decomposition_spec_builders_lock_geometry_contracts() -> None:
    lrs = AngularDecompositionSpec.lrs(4)
    non_lrs = AngularDecompositionSpec.non_lrs_diagonal(4)

    assert lrs.geometry == "lrs"
    assert lrs.representation == "axisymmetric_legendre"
    assert lrs.mode_count == 3
    assert non_lrs.geometry == "non_lrs"
    assert non_lrs.mode_count == 6


def test_fixed_non_lrs_s2_three_mode_spec_is_explicitly_labeled() -> None:
    spec = AngularDecompositionSpec.fixed_non_lrs_s2_three_mode()

    assert spec.geometry == "non_lrs"
    assert spec.representation == "sn_pstf_projection"
    assert spec.ell_max == 2
    assert spec.active_modes == (
        AngularMode(0, 0, "axisymmetric"),
        AngularMode(2, 0, "axisymmetric"),
        AngularMode(2, 2, "cos"),
    )
    assert spec.closure == "projected_tail"
    assert spec.mode_count == 3
    assert spec.contract == FIXED_NON_LRS_S2_THREE_MODE_PROJECTION_CONTRACT


def test_augmented_spec_does_not_relax_legacy_typeI_multipole_contract() -> None:
    augmented = AngularDecompositionSpec.lrs(8)

    assert augmented.ell_max == 8
    assert MultipoleSpec().ell_max == 2
    with pytest.raises(ValueError, match="must be exactly 2"):
        MultipoleSpec(ell_max=4)


def test_angular_decomposition_rejects_odd_ladder() -> None:
    with pytest.raises(ValueError, match="even ell_max"):
        AngularDecompositionSpec.lrs(3)


def test_lrs_spec_rejects_non_axisymmetric_modes() -> None:
    with pytest.raises(ValueError, match="LRS angular decomposition"):
        AngularDecompositionSpec(
            geometry="lrs",
            representation="axisymmetric_legendre",
            ell_max=2,
            active_modes=(AngularMode(2, 2, "cos"),),
        )


def test_ell_max_convergence_spec_requires_increasing_even_values() -> None:
    spec = EllMaxConvergenceSpec((2, 4, 6, 8))

    assert spec.ell_values == (2, 4, 6, 8)

    with pytest.raises(ValueError, match="strictly increasing"):
        EllMaxConvergenceSpec((2, 6, 4))

    with pytest.raises(ValueError, match="even ell_max"):
        EllMaxConvergenceSpec((2, 3, 4))


def test_collision_quadrature_spec_is_deterministic_and_shape_checked() -> None:
    spec = CollisionQuadratureSpec(
        q_nodes=[0.5, 1.5],
        q_weights=[0.25, 0.75],
        nue_y2_nodes=[0.1],
        nue_y2_weights=[1.0],
        nue_y3_nodes=[0.2, 0.8],
        nue_y3_weights=[0.4, 0.6],
        pair_y2_nodes=[0.3],
        pair_y2_weights=[1.0],
        pair_leg_nodes=[-0.5, 0.5],
        pair_leg_weights=[1.0, 1.0],
    )
    same = CollisionQuadratureSpec(
        q_nodes=(0.5, 1.5),
        q_weights=(0.25, 0.75),
        nue_y2_nodes=(0.1,),
        nue_y2_weights=(1.0,),
        nue_y3_nodes=(0.2, 0.8),
        nue_y3_weights=(0.4, 0.6),
        pair_y2_nodes=(0.3,),
        pair_y2_weights=(1.0,),
        pair_leg_nodes=(-0.5, 0.5),
        pair_leg_weights=(1.0, 1.0),
    )

    assert spec == same
    assert np.allclose(spec.arrays()["q_nodes"], [0.5, 1.5])

    with pytest.raises(ValueError, match="equal length"):
        CollisionQuadratureSpec(
            q_nodes=[0.5, 1.5],
            q_weights=[1.0],
            nue_y2_nodes=[0.1],
            nue_y2_weights=[1.0],
            nue_y3_nodes=[0.2],
            nue_y3_weights=[1.0],
            pair_y2_nodes=[0.3],
            pair_y2_weights=[1.0],
            pair_leg_nodes=[0.0],
            pair_leg_weights=[1.0],
        )


def test_collision_moment_result_normalizes_payload() -> None:
    result = CollisionMomentResult(
        C=[[0.0, 1.0]],
        energy_transfer_by_species={"nue": 1},
        detailed_balance_residual=0.0,
        number_residual=1.0e-12,
        quadrature_contract="fixed_gauss_sn_v1",
    )

    assert result.C.shape == (1, 2)
    assert result.energy_transfer_by_species == {"nue": 1.0}
