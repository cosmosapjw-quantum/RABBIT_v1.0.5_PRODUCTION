from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import rabbit.collisions.pstf_contractions as pstf_contractions
from rabbit.collisions.deterministic_reference import COLLISION_STATISTICAL_MONOMIALS
from rabbit.collisions.pstf_contractions import (
    PSTFBilinearMatrixElementTerm,
    PSTFMassMatrixElementTerm,
    PSTFMassQuarticMatrixElementTerm,
    PSTFRadialInvariantPrefactorConfig,
    build_local_pstf_statistical_kernel_table,
    build_pstf_channel_kernel_table,
    build_pstf_radial_channel_kernel_grid,
    build_pstf_radial_channel_kernel_grid_batch,
    build_universal_pstf_geometric_kernel_table,
    contract_local_pstf_statistical_kernel,
    contract_pstf_channel_radial_grid,
    contract_pstf_radial_channel_kernel_grid,
    contract_pstf_radial_channel_kernel_grid_batch,
    read_pstf_radial_channel_kernel_grid_npz,
    write_pstf_radial_channel_kernel_grid_npz,
)
from rabbit.transport.augmented_pstf_distribution import fermi_dirac_from_logit
from rabbit.transport.augmented_typeI_nonlrs_collisionless import build_non_lrs_s2_grid


def _lrs_basis(n_mu: int = 18) -> tuple[np.ndarray, np.ndarray]:
    mu, weights = leggauss(n_mu)
    p0 = np.ones_like(mu)
    p1 = mu
    p2 = 0.5 * (3.0 * mu**2 - 1.0)
    return np.vstack([p0, p1, p2]), weights


def _project_modes(source: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = float(np.sum(weights))
    gram = np.einsum("ma,na,a->mn", basis, basis, weights) / norm
    rhs = np.einsum("a,ma,a->m", source, basis, weights) / norm
    return np.linalg.solve(gram, rhs)


def test_bd175_pstf_contraction_cache_keys_reuse_call_local_array_memo() -> None:
    values = np.arange(12.0, dtype=float).reshape(3, 4)
    memo: dict[int, tuple[np.ndarray, tuple[object, ...]]] = {}

    first = pstf_contractions._cache_array_key(values, memo=memo)
    second = pstf_contractions._cache_array_key(values, memo=memo)
    same_content_copy = pstf_contractions._cache_array_key(values.copy(), memo=memo)

    assert first == second
    assert first is second
    assert first == same_content_copy
    assert first is not same_content_copy
    assert len(memo) == 2

    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    prefix_memo: dict[int, tuple[np.ndarray, tuple[object, ...]]] = {}
    prefix = pstf_contractions._geometric_table_cache_prefix(
        (basis, basis, basis, basis),
        (weights, weights, weights, weights),
        (directions, directions, directions, directions),
        mu_pairs=((1, 2),),
        mumu_pairs=(((1, 2), (3, 4)),),
        memo=prefix_memo,
    )
    prefix_again = pstf_contractions._geometric_table_cache_prefix(
        (basis, basis, basis, basis),
        (weights, weights, weights, weights),
        (directions, directions, directions, directions),
        mu_pairs=((1, 2),),
        mumu_pairs=(((1, 2), (3, 4)),),
        memo=prefix_memo,
    )

    assert prefix == prefix_again
    assert len(prefix_memo) == 3

    grid_memo: dict[int, tuple[np.ndarray, tuple[object, ...]]] = {}
    geometric_cache: dict[object, object] = {}
    delta = np.ones((2, 2, 2, 2), dtype=float)
    radial_kwargs = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=delta,
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5]),
        p3_energies=np.array([1.0]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([1.9]),
        p2_momenta=np.array([2.4]),
        p3_momenta=np.array([0.8]),
        p4_momentum_grid=np.array([1.8, 2.7, 3.6]),
        q2_weights=np.array([0.7]),
        q3_weights=np.array([1.1]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
        geometric_table_cache=geometric_cache,
        array_key_memo=grid_memo,
    )
    build_pstf_radial_channel_kernel_grid(**radial_kwargs)
    grid_memo_size = len(grid_memo)
    build_pstf_radial_channel_kernel_grid(**radial_kwargs)

    assert grid_memo_size == 4
    assert len(grid_memo) == grid_memo_size


def _manual_geometric_tensor(
    *,
    projection: np.ndarray,
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    directions_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    delta_weights: np.ndarray,
    monomial: str,
    q_pairs: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    digits = tuple(int(char) for char in monomial)
    shape = (projection.shape[0],) + tuple(basis_by_particle[idx - 1].shape[0] for idx in digits)
    out = np.zeros(shape, dtype=float)
    for a1 in range(delta_weights.shape[0]):
        for a2 in range(delta_weights.shape[1]):
            for a3 in range(delta_weights.shape[2]):
                for a4 in range(delta_weights.shape[3]):
                    angle_indices = (a1, a2, a3, a4)
                    coeff = float(delta_weights[angle_indices])
                    coeff *= float(weights_by_particle[1][a2])
                    coeff *= float(weights_by_particle[2][a3])
                    coeff *= float(weights_by_particle[3][a4])
                    for i, j in q_pairs:
                        coeff *= float(
                            np.dot(
                                directions_by_particle[i - 1][angle_indices[i - 1]],
                                directions_by_particle[j - 1][angle_indices[j - 1]],
                            )
                        )
                    factors = [projection[:, a1]]
                    factors.extend(
                        basis_by_particle[idx - 1][:, angle_indices[idx - 1]]
                        for idx in digits
                    )
                    term = factors[0]
                    for factor in factors[1:]:
                        term = np.multiply.outer(term, factor)
                    out += coeff * term
    return out


def test_universal_pstf_geometric_kernel_table_matches_direct_quadrature() -> None:
    directions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    basis = np.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    weights = np.asarray([1.0, 1.0])
    basis_by_particle = (basis, basis, basis, basis)
    weights_by_particle = (weights, weights, weights, weights)
    directions_by_particle = (directions, directions, directions, directions)
    delta_weights = np.ones((2, 2, 2, 2), dtype=float)

    table = build_universal_pstf_geometric_kernel_table(
        basis_by_particle,
        weights_by_particle,
        directions_by_particle,
        delta_weights,
        mu_pairs=((1, 2),),
        mumu_pairs=(((1, 2), (3, 4)),),
    )

    assert table.geometric_contract == "universal_pstf_geometric_kernels_v1"
    assert tuple(table.G0) == COLLISION_STATISTICAL_MONOMIALS
    assert table.G0["34"] == pytest.approx(
        _manual_geometric_tensor(
            projection=table.output_projection_matrix,
            basis_by_particle=basis_by_particle,
            weights_by_particle=weights_by_particle,
            directions_by_particle=directions_by_particle,
            delta_weights=delta_weights,
            monomial="34",
        ),
        rel=0.0,
        abs=1.0e-15,
    )
    assert table.G_mu[(1, 2)]["34"] == pytest.approx(
        _manual_geometric_tensor(
            projection=table.output_projection_matrix,
            basis_by_particle=basis_by_particle,
            weights_by_particle=weights_by_particle,
            directions_by_particle=directions_by_particle,
            delta_weights=delta_weights,
            monomial="34",
            q_pairs=((1, 2),),
        ),
        rel=0.0,
        abs=1.0e-15,
    )
    assert table.G_mumu[((1, 2), (3, 4))]["123"] == pytest.approx(
        _manual_geometric_tensor(
            projection=table.output_projection_matrix,
            basis_by_particle=basis_by_particle,
            weights_by_particle=weights_by_particle,
            directions_by_particle=directions_by_particle,
            delta_weights=delta_weights,
            monomial="123",
            q_pairs=((1, 2), (3, 4)),
        ),
        rel=0.0,
        abs=1.0e-15,
    )


def test_universal_pstf_geometric_kernel_table_rejects_bad_delta_shape() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])

    with pytest.raises(ValueError, match="momentum_delta_weights"):
        build_universal_pstf_geometric_kernel_table(
            (basis, basis, basis, basis),
            (weights, weights, weights, weights),
            (directions, directions, directions, directions),
            np.ones((2, 2, 2), dtype=float),
        )


def test_pstf_channel_kernel_table_combines_geometric_kernels_with_pi_descriptors() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    geometric = build_universal_pstf_geometric_kernel_table(
        (basis, basis, basis, basis),
        (weights, weights, weights, weights),
        (directions, directions, directions, directions),
        np.ones((2, 2, 2, 2), dtype=float),
        mu_pairs=((1, 2), (3, 4)),
        mumu_pairs=(((1, 2), (3, 4)),),
    )
    energies = (2.0, 3.0, 5.0, 7.0)
    momenta = (0.5, 0.7, 1.1, 1.3)
    electron_mass_momentum = 0.511

    channel = build_pstf_channel_kernel_table(
        geometric,
        energies=energies,
        momenta=momenta,
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=2.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
        mass_terms=(
            PSTFMassMatrixElementTerm(
                zeta=0.5,
                pair=(1, 2),
            ),
        ),
        mass_quartic_terms=(PSTFMassQuarticMatrixElementTerm(zeta=-0.125),),
        electron_mass_momentum=electron_mass_momentum,
        coupling_prefactor=1.25,
        radial_prefactor=0.75,
    )

    G0 = geometric.G0["34"]
    G12 = geometric.G_mu[(1, 2)]["34"]
    G34 = geometric.G_mu[(3, 4)]["34"]
    G12_34 = geometric.G_mumu[((1, 2), (3, 4))]["34"]
    D12 = energies[0] * energies[1] * G0 - momenta[0] * momenta[1] * G12
    D12_34 = (
        energies[0] * energies[1] * energies[2] * energies[3] * G0
        - energies[0] * energies[1] * momenta[2] * momenta[3] * G34
        - momenta[0] * momenta[1] * energies[2] * energies[3] * G12
        + momenta[0] * momenta[1] * momenta[2] * momenta[3] * G12_34
    )
    expected = 0.75 * 1.25 * (
        2.0 * D12_34
        + 0.5 * electron_mass_momentum**2 * D12
        - 0.125 * electron_mass_momentum**4 * G0
    )

    assert channel.channel_contract == "pstf_channel_kernel_from_geometric_table_v1"
    assert [term.zeta for term in channel.mass_quartic_terms] == pytest.approx([-0.125])
    assert channel.K_by_monomial["34"] == pytest.approx(expected, rel=0.0, abs=1.0e-15)
    assert tuple(channel.K_by_monomial) == COLLISION_STATISTICAL_MONOMIALS


def test_pstf_channel_kernel_table_requires_requested_geometric_mu_products() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    geometric = build_universal_pstf_geometric_kernel_table(
        (basis, basis, basis, basis),
        (weights, weights, weights, weights),
        (directions, directions, directions, directions),
        np.ones((2, 2, 2, 2), dtype=float),
        mu_pairs=((1, 2),),
    )

    with pytest.raises(ValueError, match="G_mumu"):
        build_pstf_channel_kernel_table(
            geometric,
            energies=(2.0, 3.0, 5.0, 7.0),
            momenta=(0.5, 0.7, 1.1, 1.3),
            bilinear_terms=(
                PSTFBilinearMatrixElementTerm(
                    eta=1.0,
                    first_pair=(1, 2),
                    second_pair=(3, 4),
                ),
            ),
        )


def _radial_kernel_tensor(shape: tuple[int, ...], scale: float) -> np.ndarray:
    return scale * (np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0)


def _small_pstf_radial_channel_kernel_grid(*, coupling_prefactor: float = 1.0):
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    return build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([2.5, 3.0]),
        p3_momenta=np.array([1.0, 1.5]),
        p4_momentum_grid=np.array([2.0, 3.0, 4.0]),
        q2_weights=np.array([0.7, 0.3]),
        q3_weights=np.array([1.1, 0.2]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
        coupling_prefactor=coupling_prefactor,
    )


def test_pstf_channel_radial_grid_contracts_p4_interpolation_and_six_monomials() -> None:
    F1 = np.array([[0.30, 0.02]])
    F2 = np.array([[0.25, -0.01]])
    F3 = np.array([[0.35, 0.04]])
    F4 = np.array(
        [
            [0.20, 0.02],
            [0.40, 0.06],
        ]
    )
    p1_energy = np.array([2.0])
    p2_energy = np.array([3.0])
    p3_energy = np.array([1.5])
    p4_energy_grid = np.array([3.0, 4.0])
    q2_weights = np.array([0.7])
    q3_weights = np.array([1.1])
    K = {
        "34": _radial_kernel_tensor((1, 1, 1, 2, 2, 2), 0.010),
        "12": _radial_kernel_tensor((1, 1, 1, 2, 2, 2), 0.020),
        "123": _radial_kernel_tensor((1, 1, 1, 2, 2, 2, 2), 0.003),
        "124": _radial_kernel_tensor((1, 1, 1, 2, 2, 2, 2), 0.004),
        "134": _radial_kernel_tensor((1, 1, 1, 2, 2, 2, 2), 0.005),
        "234": _radial_kernel_tensor((1, 1, 1, 2, 2, 2, 2), 0.006),
    }

    result = contract_pstf_channel_radial_grid(
        F1,
        F2,
        F3,
        F4,
        p1_energy,
        p2_energy,
        p3_energy,
        p4_energy_grid,
        q2_weights,
        q3_weights,
        K,
    )

    F4_interp = np.array([0.30, 0.04])
    expected = q2_weights[0] * q3_weights[0] * (
        np.einsum("ade,d,e->a", K["34"][0, 0, 0], F3[0], F4_interp)
        - np.einsum("abc,b,c->a", K["12"][0, 0, 0], F1[0], F2[0])
        + np.einsum("abcd,b,c,d->a", K["123"][0, 0, 0], F1[0], F2[0], F3[0])
        + np.einsum("abce,b,c,e->a", K["124"][0, 0, 0], F1[0], F2[0], F4_interp)
        - np.einsum("abde,b,d,e->a", K["134"][0, 0, 0], F1[0], F3[0], F4_interp)
        - np.einsum("acde,c,d,e->a", K["234"][0, 0, 0], F2[0], F3[0], F4_interp)
    )

    assert result.radial_contract == "pstf_channel_radial_grid_contraction_v1"
    assert result.C_modes == pytest.approx(expected[None, :], rel=0.0, abs=1.0e-15)
    assert result.p4_energies[0, 0, 0] == pytest.approx(3.5)
    assert result.p4_left_indices[0, 0, 0] == 0
    assert result.p4_right_weights[0, 0, 0] == pytest.approx(0.5)
    assert bool(result.valid_radial_mask[0, 0, 0])


def test_pstf_channel_radial_grid_zeros_kinematically_invalid_p4() -> None:
    F1 = np.array([[0.30]])
    F2 = np.array([[0.25]])
    F3 = np.array([[0.35]])
    F4 = np.array([[0.20], [0.40]])
    K = {
        "34": np.ones((1, 1, 1, 1, 1, 1)),
        "12": np.ones((1, 1, 1, 1, 1, 1)),
        "123": np.ones((1, 1, 1, 1, 1, 1, 1)),
        "124": np.ones((1, 1, 1, 1, 1, 1, 1)),
        "134": np.ones((1, 1, 1, 1, 1, 1, 1)),
        "234": np.ones((1, 1, 1, 1, 1, 1, 1)),
    }

    result = contract_pstf_channel_radial_grid(
        F1,
        F2,
        F3,
        F4,
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.6]),
        np.array([1.0, 2.0]),
        np.array([1.0]),
        np.array([1.0]),
        K,
    )

    assert result.C_modes == pytest.approx(np.zeros((1, 1)))
    assert not bool(result.valid_radial_mask[0, 0, 0])


def test_pstf_channel_radial_grid_supports_three_mode_s2_radial_ladder() -> None:
    F1 = np.array([[0.30, 0.02, -0.01]])
    F2 = np.array(
        [
            [0.25, -0.01, 0.03],
            [0.27, 0.02, -0.02],
        ]
    )
    F3 = np.array(
        [
            [0.35, 0.04, -0.03],
            [0.31, -0.02, 0.05],
        ]
    )
    F4 = np.array(
        [
            [0.20, 0.02, -0.01],
            [0.32, 0.04, 0.01],
            [0.44, 0.06, 0.03],
        ]
    )
    p1_energy = np.array([2.0])
    p2_energy = np.array([2.5, 3.0])
    p3_energy = np.array([1.0, 1.5])
    p4_energy_grid = np.array([3.0, 3.5, 4.0])
    q2_weights = np.array([0.6, 0.4])
    q3_weights = np.array([0.7, 0.3])
    K = {
        "34": np.zeros((1, 2, 2, 3, 3, 3)),
        "12": np.zeros((1, 2, 2, 3, 3, 3)),
        "123": np.zeros((1, 2, 2, 3, 3, 3, 3)),
        "124": np.zeros((1, 2, 2, 3, 3, 3, 3)),
        "134": np.zeros((1, 2, 2, 3, 3, 3, 3)),
        "234": np.zeros((1, 2, 2, 3, 3, 3, 3)),
    }
    for i2 in range(2):
        for i3 in range(2):
            for mode in range(3):
                K["34"][0, i2, i3, mode, mode, mode] = 0.01 * (i2 + 1) * (i3 + 1) * (mode + 1)

    result = contract_pstf_channel_radial_grid(
        F1,
        F2,
        F3,
        F4,
        p1_energy,
        p2_energy,
        p3_energy,
        p4_energy_grid,
        q2_weights,
        q3_weights,
        K,
    )

    expected = np.zeros(3)
    for i2 in range(2):
        for i3 in range(2):
            e4 = p1_energy[0] + p2_energy[i2] - p3_energy[i3]
            if e4 == p4_energy_grid[-1]:
                left = p4_energy_grid.size - 2
                weight = 1.0
            else:
                left = int(np.searchsorted(p4_energy_grid, e4, side="right") - 1)
                weight = (e4 - p4_energy_grid[left]) / (p4_energy_grid[left + 1] - p4_energy_grid[left])
            F4_interp = (1.0 - weight) * F4[left] + weight * F4[left + 1]
            expected += q2_weights[i2] * q3_weights[i3] * np.einsum(
                "ade,d,e->a",
                K["34"][0, i2, i3],
                F3[i3],
                F4_interp,
            )

    assert result.C_modes == pytest.approx(expected[None, :], rel=0.0, abs=1.0e-15)
    assert result.valid_radial_mask.all()
    assert result.p4_right_weights[0, 0, 0] == pytest.approx(0.0)
    assert result.p4_right_weights[0, 1, 0] == pytest.approx(1.0)


def test_pstf_radial_channel_kernel_grid_assembles_prefactor_geometry_and_contracts() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    delta = np.ones((2, 2, 2, 2), dtype=float)
    bilinear = (
        PSTFBilinearMatrixElementTerm(
            eta=1.7,
            first_pair=(1, 2),
            second_pair=(3, 4),
        ),
    )
    mass = (
        PSTFMassMatrixElementTerm(
            zeta=0.3,
            pair=(1, 2),
        ),
    )
    prefactor = PSTFRadialInvariantPrefactorConfig(
        symmetry_factor=0.5,
        g2=2.0,
        g3=3.0,
        g4=4.0,
        hbar=1.0,
        c_light=1.0,
    )
    delta_calls: list[tuple[int, int, int, float, float]] = []

    def delta_factory(i1: int, i2: int, i3: int, e4: float, p4: float) -> np.ndarray:
        delta_calls.append((i1, i2, i3, e4, p4))
        return delta

    grid = build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=delta_factory,
        p1_energies=np.array([2.0]),
        p2_energies=np.array([3.0]),
        p3_energies=np.array([1.5]),
        p4_energy_grid=np.array([3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([3.0]),
        p3_momenta=np.array([1.5]),
        p4_momentum_grid=np.array([3.0, 4.0]),
        q2_weights=np.array([0.7]),
        q3_weights=np.array([1.1]),
        bilinear_terms=bilinear,
        mass_terms=mass,
        electron_mass_momentum=0.511,
        coupling_prefactor=1.25,
        radial_prefactor_config=prefactor,
    )

    tau = 2.0 * np.pi
    expected_prefactor = (
        0.5
        * tau**4
        / (2.0 * 2.0)
        * (2.0 * 3.0**2 / (tau**3 * 2.0 * 3.0))
        * (3.0 * 1.5**2 / (tau**3 * 2.0 * 1.5))
        * (4.0 * 3.5 / (tau**3 * 2.0))
    )
    geometric = build_universal_pstf_geometric_kernel_table(
        (basis, basis, basis, basis),
        (weights, weights, weights, weights),
        (directions, directions, directions, directions),
        delta,
        mu_pairs=((1, 2), (3, 4)),
        mumu_pairs=(((1, 2), (3, 4)),),
    )
    channel = build_pstf_channel_kernel_table(
        geometric,
        energies=(2.0, 3.0, 1.5, 3.5),
        momenta=(2.0, 3.0, 1.5, 3.5),
        bilinear_terms=bilinear,
        mass_terms=mass,
        electron_mass_momentum=0.511,
        coupling_prefactor=1.25,
        radial_prefactor=expected_prefactor,
    )

    assert grid.kernel_grid_contract == "pstf_radial_channel_kernel_grid_v1"
    assert grid.p4_energies[0, 0, 0] == pytest.approx(3.5)
    assert grid.p4_momenta[0, 0, 0] == pytest.approx(3.5)
    assert grid.p4_right_weights[0, 0, 0] == pytest.approx(0.5)
    assert grid.radial_prefactors[0, 0, 0] == pytest.approx(expected_prefactor)
    assert delta_calls == [(0, 0, 0, 3.5, 3.5)]
    assert grid.K_by_monomial["34"][0, 0, 0] == pytest.approx(channel.K_by_monomial["34"])

    F1 = np.array([[0.30, 0.02]])
    F2 = np.array([[0.25, -0.01]])
    F3 = np.array([[0.35, 0.04]])
    F4 = np.array([[0.20, 0.02], [0.40, 0.06]])
    wrapped = contract_pstf_radial_channel_kernel_grid(F1, F2, F3, F4, grid)
    direct = contract_pstf_channel_radial_grid(
        F1,
        F2,
        F3,
        F4,
        grid.p1_energies,
        grid.p2_energies,
        grid.p3_energies,
        grid.p4_energy_grid,
        grid.q2_weights,
        grid.q3_weights,
        grid.K_by_monomial,
    )
    assert wrapped.C_modes == pytest.approx(direct.C_modes, rel=0.0, abs=1.0e-15)


def test_pstf_radial_channel_kernel_grid_uses_internal_fast_channel_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])

    def fail_public_channel_builder(*args: object, **kwargs: object) -> object:
        raise AssertionError("radial grid assembly should not allocate public channel tables per radial tuple")

    monkeypatch.setattr(
        pstf_contractions,
        "build_pstf_channel_kernel_table",
        fail_public_channel_builder,
    )

    grid = pstf_contractions.build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([3.0]),
        p3_energies=np.array([1.5]),
        p4_energy_grid=np.array([3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([3.0]),
        p3_momenta=np.array([1.5]),
        p4_momentum_grid=np.array([3.0, 4.0]),
        q2_weights=np.array([0.7]),
        q3_weights=np.array([1.1]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.7,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
        mass_terms=(PSTFMassMatrixElementTerm(zeta=0.3, pair=(1, 2)),),
        mass_quartic_terms=(PSTFMassQuarticMatrixElementTerm(zeta=-0.125),),
        electron_mass_momentum=0.511,
        coupling_prefactor=1.25,
    )

    assert grid.K_by_monomial["34"][0, 0, 0].shape == (2, 2, 2)
    assert np.max(np.abs(grid.K_by_monomial["34"])) > 0.0


def test_pstf_radial_channel_kernel_grid_precomputes_radial_quadrature_weights() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    q2 = np.array([0.7, 0.3])
    q3 = np.array([1.1, 0.2])

    grid = build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([2.5, 3.0]),
        p3_momenta=np.array([1.0, 1.5]),
        p4_momentum_grid=np.array([2.0, 3.0, 4.0]),
        q2_weights=q2,
        q3_weights=q3,
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    expected = grid.valid_radial_mask.astype(float) * q2[None, :, None] * q3[None, None, :]
    assert grid.radial_quadrature_weights == pytest.approx(expected, rel=0.0, abs=0.0)


def test_pstf_radial_channel_kernel_grid_round_trips_npz_cache(tmp_path) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    path = tmp_path / "radial_grid.npz"

    grid = build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([2.5, 3.0]),
        p3_momenta=np.array([1.0, 1.5]),
        p4_momentum_grid=np.array([2.0, 3.0, 4.0]),
        q2_weights=np.array([0.7, 0.3]),
        q3_weights=np.array([1.1, 0.2]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    written = write_pstf_radial_channel_kernel_grid_npz(path, grid)
    loaded = read_pstf_radial_channel_kernel_grid_npz(written)

    assert written == path
    assert loaded.kernel_grid_contract == grid.kernel_grid_contract
    assert loaded.radial_quadrature_weights == pytest.approx(grid.radial_quadrature_weights)
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        assert loaded.K_by_monomial[monomial] == pytest.approx(grid.K_by_monomial[monomial])


def test_pstf_radial_channel_kernel_grid_batch_matches_individual_contractions() -> None:
    pstf_contractions._RADIAL_BATCH_EINSUM_PATH_CACHE.clear()
    grid_a = _small_pstf_radial_channel_kernel_grid(coupling_prefactor=1.0)
    grid_b = _small_pstf_radial_channel_kernel_grid(coupling_prefactor=0.35)
    F1_a = np.array([[0.30, 0.02]])
    F2_a = np.array([[0.25, -0.01], [0.27, 0.03]])
    F3_a = np.array([[0.35, 0.04], [0.31, -0.02]])
    F4_a = np.array([[0.20, 0.02], [0.32, 0.04], [0.44, 0.06]])
    F1_b = np.array([[0.28, -0.03]])
    F2_b = np.array([[0.21, 0.02], [0.24, -0.01]])
    F3_b = np.array([[0.39, -0.02], [0.29, 0.05]])
    F4_b = np.array([[0.18, -0.01], [0.31, 0.03], [0.42, 0.05]])

    batch = build_pstf_radial_channel_kernel_grid_batch(
        (grid_a, grid_b),
        max_kernel_nbytes=10_000_000,
    )
    assert batch is not None

    batched = contract_pstf_radial_channel_kernel_grid_batch(
        np.stack((F1_a, F1_b), axis=0),
        np.stack((F2_a, F2_b), axis=0),
        np.stack((F3_a, F3_b), axis=0),
        np.stack((F4_a, F4_b), axis=0),
        batch,
    )
    expected = (
        contract_pstf_radial_channel_kernel_grid(F1_a, F2_a, F3_a, F4_a, grid_a),
        contract_pstf_radial_channel_kernel_grid(F1_b, F2_b, F3_b, F4_b, grid_b),
    )

    assert batch.batch_contract == "pstf_radial_channel_kernel_grid_batch_v1"
    assert len(pstf_contractions._RADIAL_BATCH_EINSUM_PATH_CACHE) == 6
    assert len(batched) == 2
    for actual, reference in zip(batched, expected):
        assert actual.C_modes == pytest.approx(reference.C_modes, rel=0.0, abs=1.0e-15)
        assert np.array_equal(actual.p4_left_indices, reference.p4_left_indices)
        assert actual.p4_right_weights == pytest.approx(reference.p4_right_weights)
        assert np.array_equal(actual.valid_radial_mask, reference.valid_radial_mask)

    cached_size = len(pstf_contractions._RADIAL_BATCH_EINSUM_PATH_CACHE)
    repeated = contract_pstf_radial_channel_kernel_grid_batch(
        np.stack((F1_a, F1_b), axis=0),
        np.stack((F2_a, F2_b), axis=0),
        np.stack((F3_a, F3_b), axis=0),
        np.stack((F4_a, F4_b), axis=0),
        batch,
    )
    assert len(pstf_contractions._RADIAL_BATCH_EINSUM_PATH_CACHE) == cached_size
    for actual, reference in zip(repeated, expected):
        assert actual.C_modes == pytest.approx(reference.C_modes, rel=0.0, abs=1.0e-15)


def test_pstf_radial_channel_kernel_grid_batch_respects_memory_budget() -> None:
    grid = _small_pstf_radial_channel_kernel_grid()

    assert build_pstf_radial_channel_kernel_grid_batch((grid,), max_kernel_nbytes=1) is None


def test_pstf_radial_channel_kernel_grid_reuses_geometric_cache() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    common_kwargs = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([3.0]),
        p3_energies=np.array([1.5]),
        p4_energy_grid=np.array([3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([3.0]),
        p3_momenta=np.array([1.5]),
        p4_momentum_grid=np.array([3.0, 4.0]),
        q2_weights=np.array([0.7]),
        q3_weights=np.array([1.1]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )
    cache: dict[object, object] = {}

    build_pstf_radial_channel_kernel_grid(**common_kwargs, geometric_table_cache=cache)
    cache_size = len(cache)
    build_pstf_radial_channel_kernel_grid(**common_kwargs, geometric_table_cache=cache)

    assert cache_size == 1
    assert len(cache) == cache_size


def test_pstf_radial_channel_kernel_grid_reuses_static_geometry_without_external_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    original = pstf_contractions.build_universal_pstf_geometric_kernel_table
    calls: list[int] = []

    def counting_builder(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pstf_contractions,
        "build_universal_pstf_geometric_kernel_table",
        counting_builder,
    )

    grid = pstf_contractions.build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([2.5, 3.0]),
        p3_momenta=np.array([1.0, 1.5]),
        p4_momentum_grid=np.array([2.0, 3.0, 4.0]),
        q2_weights=np.array([0.7, 0.3]),
        q3_weights=np.array([1.1, 0.2]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    assert int(np.sum(grid.valid_radial_mask)) == 4
    assert len(calls) == 1


def test_pstf_radial_channel_kernel_grid_static_vectorized_matches_callable_path() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    delta = np.ones((2, 2, 2, 2), dtype=float)
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        p1_energies=np.array([2.0, 2.4]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0, 5.0]),
        p1_momenta=np.array([1.9, 2.2]),
        p2_momenta=np.array([2.4, 2.8]),
        p3_momenta=np.array([0.8, 1.2]),
        p4_momentum_grid=np.array([1.8, 2.7, 3.6, 4.5]),
        q2_weights=np.array([0.7, 0.3]),
        q3_weights=np.array([1.1, 0.2]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
            PSTFBilinearMatrixElementTerm(
                eta=-0.25,
                first_pair=(1, 4),
                second_pair=(2, 3),
            ),
        ),
        mass_terms=(
            PSTFMassMatrixElementTerm(zeta=0.2, pair=(1, 2)),
        ),
        mass_quartic_terms=(
            PSTFMassQuarticMatrixElementTerm(zeta=-0.05),
        ),
        electron_mass_momentum=0.6,
        coupling_prefactor=1.7,
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(
            g2=2.0,
            g3=1.0,
            g4=2.0,
        ),
    )

    vectorized = build_pstf_radial_channel_kernel_grid(
        momentum_delta_weights=delta,
        **common,
    )
    callable_grid = build_pstf_radial_channel_kernel_grid(
        momentum_delta_weights=lambda *_args, **_kwargs: delta,
        **common,
    )

    np.testing.assert_array_equal(
        vectorized.valid_radial_mask,
        callable_grid.valid_radial_mask,
    )
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        np.testing.assert_allclose(
            vectorized.K_by_monomial[monomial],
            callable_grid.K_by_monomial[monomial],
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_bd174_static_radial_kernel_inplace_handles_edge_terms() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    delta = np.ones((2, 2, 2, 2), dtype=float)
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        p1_energies=np.array([2.0, 2.4]),
        p2_energies=np.array([2.5]),
        p3_energies=np.array([1.0]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([1.9, 2.2]),
        p2_momenta=np.array([2.4]),
        p3_momenta=np.array([0.8]),
        p4_momentum_grid=np.array([1.8, 2.7, 3.6]),
        q2_weights=np.array([0.7]),
        q3_weights=np.array([1.1]),
        electron_mass_momentum=0.6,
        coupling_prefactor=1.7,
    )

    vectorized_mass = build_pstf_radial_channel_kernel_grid(
        momentum_delta_weights=delta,
        mass_quartic_terms=(PSTFMassQuarticMatrixElementTerm(zeta=-0.05),),
        **common,
    )
    callable_mass = build_pstf_radial_channel_kernel_grid(
        momentum_delta_weights=lambda *_args, **_kwargs: delta,
        mass_quartic_terms=(PSTFMassQuarticMatrixElementTerm(zeta=-0.05),),
        **common,
    )
    zero_terms = build_pstf_radial_channel_kernel_grid(
        momentum_delta_weights=delta,
        **common,
    )

    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        np.testing.assert_allclose(
            vectorized_mass.K_by_monomial[monomial],
            callable_mass.K_by_monomial[monomial],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(zero_terms.K_by_monomial[monomial], 0.0)


def test_pstf_radial_channel_kernel_grid_skips_static_geometry_when_no_radial_tuples_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    original = pstf_contractions.build_universal_pstf_geometric_kernel_table
    calls: list[int] = []

    def counting_builder(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pstf_contractions,
        "build_universal_pstf_geometric_kernel_table",
        counting_builder,
    )

    grid = pstf_contractions.build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([1.0]),
        p2_energies=np.array([1.0]),
        p3_energies=np.array([3.0]),
        p4_energy_grid=np.array([1.0, 2.0]),
        p1_momenta=np.array([1.0]),
        p2_momenta=np.array([1.0]),
        p3_momenta=np.array([3.0]),
        p4_momentum_grid=np.array([1.0, 2.0]),
        q2_weights=np.array([1.0]),
        q3_weights=np.array([1.0]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    assert not bool(np.any(grid.valid_radial_mask))
    assert len(calls) == 0


def test_pstf_radial_channel_kernel_grid_keeps_callable_geometry_tuple_dependent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    original = pstf_contractions.build_universal_pstf_geometric_kernel_table
    calls: list[int] = []

    def counting_builder(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    def delta_factory(i1: int, i2: int, i3: int, e4: float, p4: float) -> np.ndarray:
        del i1, e4, p4
        return (1.0 + float(i2) + 0.25 * float(i3)) * np.ones((2, 2, 2, 2), dtype=float)

    monkeypatch.setattr(
        pstf_contractions,
        "build_universal_pstf_geometric_kernel_table",
        counting_builder,
    )

    grid = pstf_contractions.build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=delta_factory,
        p1_energies=np.array([2.0]),
        p2_energies=np.array([2.5, 3.0]),
        p3_energies=np.array([1.0, 1.5]),
        p4_energy_grid=np.array([2.0, 3.0, 4.0]),
        p1_momenta=np.array([2.0]),
        p2_momenta=np.array([2.5, 3.0]),
        p3_momenta=np.array([1.0, 1.5]),
        p4_momentum_grid=np.array([2.0, 3.0, 4.0]),
        q2_weights=np.array([0.7, 0.3]),
        q3_weights=np.array([1.1, 0.2]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    assert int(np.sum(grid.valid_radial_mask)) == 4
    assert len(calls) == 4
    assert not np.allclose(grid.K_by_monomial["34"][0, 0, 0], grid.K_by_monomial["34"][0, 1, 0])


def test_pstf_radial_channel_kernel_grid_zeros_invalid_energy_tuple() -> None:
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    basis = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([1.0, 1.0])
    grid = build_pstf_radial_channel_kernel_grid(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.ones((2, 2, 2, 2), dtype=float),
        p1_energies=np.array([1.0]),
        p2_energies=np.array([1.0]),
        p3_energies=np.array([1.6]),
        p4_energy_grid=np.array([1.0, 2.0]),
        p1_momenta=np.array([1.0]),
        p2_momenta=np.array([1.0]),
        p3_momenta=np.array([1.6]),
        p4_momentum_grid=np.array([1.0, 2.0]),
        q2_weights=np.array([1.0]),
        q3_weights=np.array([1.0]),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(
                eta=1.0,
                first_pair=(1, 2),
                second_pair=(3, 4),
            ),
        ),
    )

    assert not bool(grid.valid_radial_mask[0, 0, 0])
    assert grid.radial_prefactors[0, 0, 0] == pytest.approx(0.0)
    assert np.max(np.abs(grid.K_by_monomial["34"])) == pytest.approx(0.0)


def test_local_pstf_statistical_kernel_matches_direct_nodal_projection() -> None:
    basis, weights = _lrs_basis()
    table = build_local_pstf_statistical_kernel_table(basis, weights)

    F1 = np.array([0.30, 0.02, 0.05])
    F2 = np.array([0.25, -0.01, -0.03])
    F3 = np.array([0.34, 0.03, 0.04])
    F4 = np.array([0.21, -0.02, 0.02])

    result = contract_local_pstf_statistical_kernel(F1, F2, F3, F4, table)
    f1 = F1 @ basis
    f2 = F2 @ basis
    f3 = F3 @ basis
    f4 = F4 @ basis
    direct = (
        f3 * f4
        - f1 * f2
        + f1 * f2 * f3
        + f1 * f2 * f4
        - f1 * f3 * f4
        - f2 * f3 * f4
    )

    assert result.contraction_contract == "local_pstf_six_monomial_projection_v1"
    assert tuple(result.monomial_mode_contributions) == COLLISION_STATISTICAL_MONOMIALS
    assert "1234" not in result.monomial_mode_contributions
    assert result.nodal_source == pytest.approx(direct, rel=0.0, abs=1.0e-15)
    assert result.C_modes == pytest.approx(_project_modes(direct, basis, weights), rel=0.0, abs=1.0e-15)


def test_local_pstf_statistical_kernel_vanishes_at_fd_isotropic_balance() -> None:
    basis, weights = _lrs_basis()
    table = build_local_pstf_statistical_kernel_table(basis, weights)
    y1 = 0.6
    y2 = 1.1
    y3 = 0.8
    y4 = y1 + y2 - y3
    F1 = np.array([float(fermi_dirac_from_logit(y1)), 0.0, 0.0])
    F2 = np.array([float(fermi_dirac_from_logit(y2)), 0.0, 0.0])
    F3 = np.array([float(fermi_dirac_from_logit(y3)), 0.0, 0.0])
    F4 = np.array([float(fermi_dirac_from_logit(y4)), 0.0, 0.0])

    result = contract_local_pstf_statistical_kernel(F1, F2, F3, F4, table)

    assert np.max(np.abs(result.C_modes)) < 1.0e-15
    assert np.max(np.abs(result.nodal_source)) < 1.0e-15


def test_local_pstf_statistical_kernel_projects_quadrupole_without_odd_leakage() -> None:
    basis, weights = _lrs_basis(n_mu=20)
    table = build_local_pstf_statistical_kernel_table(basis, weights)

    F1 = np.array([0.30, 0.0, 0.08])
    F2 = np.array([0.24, 0.0, -0.04])
    F3 = np.array([0.33, 0.0, 0.07])
    F4 = np.array([0.22, 0.0, 0.05])

    result = contract_local_pstf_statistical_kernel(F1, F2, F3, F4, table)

    assert abs(float(result.C_modes[2])) > 0.0
    assert abs(float(result.C_modes[1])) < 1.0e-15
    assert result.max_occupancy_overshoot == pytest.approx(0.0)


def test_local_pstf_statistical_kernel_supports_nonlrs_s2_minus_mode() -> None:
    grid = build_non_lrs_s2_grid(N_mu=4, N_phi=6)
    table = build_local_pstf_statistical_kernel_table(grid.basis_matrix, grid.angular_weights)

    F1 = np.array([0.30, 0.02, 0.06])
    F2 = np.array([0.24, -0.01, -0.04])
    F3 = np.array([0.33, 0.03, 0.05])
    F4 = np.array([0.22, 0.01, 0.03])

    result = contract_local_pstf_statistical_kernel(F1, F2, F3, F4, table)
    direct = (
        (F3 @ grid.basis_matrix) * (F4 @ grid.basis_matrix)
        - (F1 @ grid.basis_matrix) * (F2 @ grid.basis_matrix)
        + (F1 @ grid.basis_matrix) * (F2 @ grid.basis_matrix) * (F3 @ grid.basis_matrix)
        + (F1 @ grid.basis_matrix) * (F2 @ grid.basis_matrix) * (F4 @ grid.basis_matrix)
        - (F1 @ grid.basis_matrix) * (F3 @ grid.basis_matrix) * (F4 @ grid.basis_matrix)
        - (F2 @ grid.basis_matrix) * (F3 @ grid.basis_matrix) * (F4 @ grid.basis_matrix)
    )

    assert result.C_modes == pytest.approx(
        _project_modes(direct, grid.basis_matrix, grid.angular_weights),
        rel=0.0,
        abs=1.0e-14,
    )
    assert abs(float(result.C_modes[2])) > 0.0


def test_local_pstf_statistical_kernel_rejects_unphysical_nodal_occupancy() -> None:
    basis, weights = _lrs_basis()
    table = build_local_pstf_statistical_kernel_table(basis, weights)

    with pytest.raises(ValueError, match="nodal occupation"):
        contract_local_pstf_statistical_kernel(
            np.array([0.95, 0.0, 0.30]),
            np.array([0.20, 0.0, 0.0]),
            np.array([0.30, 0.0, 0.0]),
            np.array([0.40, 0.0, 0.0]),
            table,
        )
