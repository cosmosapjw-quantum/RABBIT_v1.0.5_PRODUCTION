from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.deterministic_reference import (
    _pair_matrix_element,
    _scattering_matrix_element,
)
from rabbit.collisions.kernels import G_F_MEV, G_L_NUE, G_L_NUX, G_R_NUE, G_R_NUX
from rabbit.collisions import (
    build_default_supported_weak_process_catalog as exported_build_default_supported_weak_process_catalog,
    build_default_ur_weak_process_catalog as exported_build_default_ur_weak_process_catalog,
    build_finite_mass_pair_annihilation_process_descriptor as exported_build_finite_mass_pair_annihilation_process_descriptor,
    build_finite_mass_nue_scattering_process_descriptor as exported_build_finite_mass_nue_scattering_process_descriptor,
    build_ur_nunu_pairwise_process_descriptor as exported_build_ur_nunu_pairwise_process_descriptor,
    pstf_process_descriptor_key as exported_pstf_process_descriptor_key,
    pstf_process_particle_mode_labels as exported_pstf_process_particle_mode_labels,
)
from rabbit.collisions.pstf_contractions import (
    PSTFRadialInvariantPrefactorConfig,
    build_pstf_radial_channel_kernel_grid,
    contract_pstf_radial_channel_kernel_grid,
)
from rabbit.collisions.pstf_process_catalog import (
    PSTFProcessRadialCollisionResult,
    build_pstf_process_radial_moment_weights,
    compute_pstf_process_radial_moments,
    compute_pstf_process_radial_moments_preweighted,
    build_pstf_process_radial_channel_kernel_grid,
    build_default_supported_weak_process_catalog,
    build_default_ur_weak_process_catalog,
    build_finite_mass_pair_annihilation_process_descriptor,
    build_finite_mass_nue_scattering_process_descriptor,
    build_ur_nue_scattering_process_descriptor,
    build_ur_nunu_diagonal_process_descriptor,
    build_ur_nunu_pairwise_process_descriptor,
    build_ur_pair_annihilation_process_descriptor,
    evaluate_pstf_process_matrix_element,
    evaluate_pstf_process_radial_collision_source,
    project_pstf_radial_collision_source_number_neutral,
    pstf_process_descriptor_key,
    pstf_process_particle_mode_labels,
)
import rabbit.collisions.pstf_process_catalog as catalog


def _zero_mu_by_pair() -> dict[tuple[int, int], float]:
    return {
        (1, 2): 0.0,
        (1, 3): 0.0,
        (1, 4): 0.0,
        (2, 3): 0.0,
        (2, 4): 0.0,
        (3, 4): 0.0,
    }


def _finite_mass_nue_elastic_expected(
    *,
    species: str,
    electron_mass: float,
    energies: tuple[float, float, float, float],
    momenta: tuple[float, float, float, float],
    mu_by_pair: dict[tuple[int, int], float],
    antineutrino: bool = False,
    positron: bool = False,
    include_fermi_prefactor: bool = False,
) -> float:
    if species in ("nue", "nuebar"):
        gL, gR = G_L_NUE, G_R_NUE
    elif species == "nux":
        gL, gR = G_L_NUX, G_R_NUX
    else:
        raise ValueError(species)
    pi12 = energies[0] * energies[1] - momenta[0] * momenta[1] * mu_by_pair[(1, 2)]
    pi14 = energies[0] * energies[3] - momenta[0] * momenta[3] * mu_by_pair[(1, 4)]
    m2 = electron_mass**2
    s = m2 + 2.0 * pi12
    u = m2 - 2.0 * pi14
    crossed = bool(antineutrino) ^ bool(positron)
    if crossed:
        squared = gL**2 * (u - m2) ** 2 + gR**2 * (s - m2) ** 2
    else:
        squared = gL**2 * (s - m2) ** 2 + gR**2 * (u - m2) ** 2
    interference = gL * gR * m2 * (s + u - 2.0 * m2)
    value = 8.0 * (squared + interference)
    if include_fermi_prefactor:
        value *= G_F_MEV**2
    return float(value)


def _finite_mass_pair_annihilation_expected(
    *,
    species: str,
    electron_mass: float,
    energies: tuple[float, float, float, float],
    momenta: tuple[float, float, float, float],
    mu_by_pair: dict[tuple[int, int], float],
    target_antineutrino: bool = False,
    include_fermi_prefactor: bool = False,
) -> float:
    if species in ("nue", "nuebar"):
        gL, gR = G_L_NUE, G_R_NUE
    elif species == "nux":
        gL, gR = G_L_NUX, G_R_NUX
    else:
        raise ValueError(species)
    m2 = electron_mass**2
    pi12 = energies[0] * energies[1] - momenta[0] * momenta[1] * mu_by_pair[(1, 2)]
    if target_antineutrino:
        pi_u = energies[1] * energies[3] - momenta[1] * momenta[3] * mu_by_pair[(2, 4)]
    else:
        pi_u = energies[0] * energies[3] - momenta[0] * momenta[3] * mu_by_pair[(1, 4)]
    s = 2.0 * pi12
    u = m2 - 2.0 * pi_u
    value = 8.0 * (
        gL**2 * (u - m2) ** 2
        + gR**2 * (s - m2) ** 2
        + gL * gR * m2 * (s + u - 2.0 * m2)
    )
    if include_fermi_prefactor:
        value *= G_F_MEV**2
    return float(value)


def test_ur_nue_scattering_process_descriptor_matches_existing_hm_formula() -> None:
    desc = build_ur_nue_scattering_process_descriptor("nue")

    assert desc.process == "nu_e_elastic_ur"
    assert desc.species == "nue"
    assert desc.particle_labels == ("nu_alpha", "e_pm", "nu_alpha", "e_pm")
    assert desc.coupling_prefactor == pytest.approx(1.0)
    assert [term.first_pair for term in desc.bilinear_terms] == [
        (1, 2),
        (3, 4),
        (1, 4),
        (2, 3),
    ]
    assert [term.second_pair for term in desc.bilinear_terms] == [
        (1, 2),
        (3, 4),
        (1, 4),
        (2, 3),
    ]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [
            G_L_NUE**2 + G_R_NUE**2,
            G_L_NUE**2 + G_R_NUE**2,
            G_L_NUE**2 - G_R_NUE**2,
            G_L_NUE**2 - G_R_NUE**2,
        ]
    )

    energies = (1.3, 2.1, 0.7, 2.7)
    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        mu_by_pair=_zero_mu_by_pair(),
    )
    expected = _scattering_matrix_element(G_L_NUE, G_R_NUE, *energies)
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-14)


def test_finite_mass_nue_scattering_descriptor_matches_hm_closed_form() -> None:
    electron_mass = 0.51099895
    desc = build_finite_mass_nue_scattering_process_descriptor(
        "nue",
        electron_mass_momentum=electron_mass,
        charged_lepton="e_minus",
    )

    assert desc.process == "nu_e_minus_elastic_finite_mass_hm"
    assert desc.species == "nue"
    assert desc.particle_labels == ("nu_alpha", "e_minus", "nu_alpha", "e_minus")
    assert desc.matrix_element_contract == "pstf_finite_mass_hm_nue_scattering_descriptor_v1"
    assert desc.electron_mass_momentum == pytest.approx(electron_mass)
    assert [term.first_pair for term in desc.bilinear_terms] == [(1, 2), (1, 4)]
    assert [term.second_pair for term in desc.bilinear_terms] == [(1, 2), (1, 4)]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [32.0 * G_L_NUE**2, 32.0 * G_R_NUE**2]
    )
    assert [term.pair for term in desc.mass_terms] == [(1, 2), (1, 4)]
    assert [term.zeta for term in desc.mass_terms] == pytest.approx(
        [16.0 * G_L_NUE * G_R_NUE, -16.0 * G_L_NUE * G_R_NUE]
    )

    p1 = 1.4
    p2 = 0.8
    p3 = 1.1
    p4 = 0.6
    energies = (
        p1,
        float(np.sqrt(p2 * p2 + electron_mass * electron_mass)),
        p3,
        float(np.sqrt(p4 * p4 + electron_mass * electron_mass)),
    )
    momenta = (p1, p2, p3, p4)
    mu = {(1, 2): -0.35, (1, 4): 0.2}

    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
    )
    expected = _finite_mass_nue_elastic_expected(
        species="nue",
        electron_mass=electron_mass,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-14)


def test_finite_mass_positron_scattering_descriptor_swaps_chiral_squared_terms() -> None:
    electron_mass = 0.51099895
    desc = build_finite_mass_nue_scattering_process_descriptor(
        "nue",
        electron_mass_momentum=electron_mass,
        charged_lepton="e_plus",
        include_fermi_prefactor=True,
    )

    assert desc.process == "nu_e_plus_elastic_finite_mass_hm"
    assert desc.species == "nue"
    assert desc.particle_labels == ("nu_alpha", "e_plus", "nu_alpha", "e_plus")
    assert desc.coupling_prefactor == pytest.approx(G_F_MEV**2)
    assert [term.first_pair for term in desc.bilinear_terms] == [(1, 2), (1, 4)]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [32.0 * G_R_NUE**2, 32.0 * G_L_NUE**2]
    )

    p1 = 1.4
    p2 = 0.8
    p3 = 1.1
    p4 = 0.6
    energies = (
        p1,
        float(np.sqrt(p2 * p2 + electron_mass * electron_mass)),
        p3,
        float(np.sqrt(p4 * p4 + electron_mass * electron_mass)),
    )
    momenta = (p1, p2, p3, p4)
    mu = {(1, 2): -0.35, (1, 4): 0.2}

    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
    )
    expected = _finite_mass_nue_elastic_expected(
        species="nue",
        electron_mass=electron_mass,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
        positron=True,
        include_fermi_prefactor=True,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-35)


def test_finite_mass_nuebar_positron_scattering_descriptor_uncrosses_chiral_terms() -> None:
    desc = build_finite_mass_nue_scattering_process_descriptor(
        "nuebar",
        charged_lepton="e_plus",
    )

    assert desc.process == "nu_e_plus_elastic_finite_mass_hm"
    assert desc.species == "nuebar"
    assert desc.particle_labels == ("nubar_alpha", "e_plus", "nubar_alpha", "e_plus")
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [32.0 * G_L_NUE**2, 32.0 * G_R_NUE**2]
    )


def test_finite_mass_nue_scattering_descriptor_labels_and_radial_source() -> None:
    descriptor = build_finite_mass_nue_scattering_process_descriptor("nux", charged_lepton="e_minus")
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
    )
    result = evaluate_pstf_process_radial_collision_source(
        descriptor,
        F1_modes=np.asarray([[0.31]]),
        F2_modes=np.asarray([[0.23]]),
        F3_modes=np.asarray([[0.37]]),
        F4_modes=np.asarray([[0.19], [0.41]]),
        **common,
    )

    assert pstf_process_particle_mode_labels(descriptor) == ("nux", "e_minus", "nux", "e_minus")
    assert result.descriptor is descriptor
    assert np.isfinite(result.C_modes).all()
    assert abs(float(result.C_modes[0, 0])) > 0.0


def test_ur_pair_annihilation_descriptor_matches_existing_hm_formula_and_fermi_prefactor() -> None:
    desc = build_ur_pair_annihilation_process_descriptor("nux", include_fermi_prefactor=True)

    assert desc.process == "nu_nubar_to_ee_ur"
    assert desc.species == "nux"
    assert desc.particle_labels == ("nu_alpha", "nubar_alpha", "e_plus", "e_minus")
    assert desc.coupling_prefactor == pytest.approx(G_F_MEV**2)
    assert [term.first_pair for term in desc.bilinear_terms] == [
        (1, 3),
        (2, 4),
        (1, 4),
        (2, 3),
    ]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [
            G_L_NUX**2 + G_R_NUX**2,
            G_L_NUX**2 + G_R_NUX**2,
            G_L_NUX**2 - G_R_NUX**2,
            G_L_NUX**2 - G_R_NUX**2,
        ]
    )

    energies = (1.7, 0.9, 1.1, 1.5)
    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        mu_by_pair=_zero_mu_by_pair(),
        include_coupling_prefactor=False,
    )
    expected = _pair_matrix_element(G_L_NUX, G_R_NUX, *energies)
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-15)
    assert evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        mu_by_pair=_zero_mu_by_pair(),
        include_coupling_prefactor=True,
    ) == pytest.approx(G_F_MEV**2 * expected, rel=0.0, abs=1.0e-42)


def test_pair_annihilation_nuebar_descriptor_labels_antineutrino_target() -> None:
    desc = build_ur_pair_annihilation_process_descriptor("nuebar")

    assert desc.species == "nuebar"
    assert desc.particle_labels == ("nubar_alpha", "nu_alpha", "e_plus", "e_minus")
    assert pstf_process_particle_mode_labels(desc) == (
        "nuebar",
        "nue",
        "e_plus",
        "e_minus",
    )

    energies = (1.7, 0.9, 1.1, 1.5)
    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        mu_by_pair=_zero_mu_by_pair(),
    )
    expected = _pair_matrix_element(G_L_NUE, G_R_NUE, *energies)
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-15)


def test_finite_mass_pair_annihilation_descriptor_matches_hm_crossing_closed_form() -> None:
    electron_mass = 0.51099895
    desc = build_finite_mass_pair_annihilation_process_descriptor(
        "nue",
        electron_mass_momentum=electron_mass,
        include_fermi_prefactor=True,
    )

    assert desc.process == "nu_nubar_to_ee_finite_mass_hm"
    assert desc.species == "nue"
    assert desc.particle_labels == ("nu_alpha", "nubar_alpha", "e_plus", "e_minus")
    assert desc.matrix_element_contract == "pstf_finite_mass_hm_pair_annihilation_descriptor_v1"
    assert [term.first_pair for term in desc.bilinear_terms] == [(1, 4), (1, 2)]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx(
        [32.0 * G_L_NUE**2, 32.0 * G_R_NUE**2]
    )
    assert [term.pair for term in desc.mass_terms] == [(1, 2), (1, 4)]
    assert [term.zeta for term in desc.mass_terms] == pytest.approx(
        [16.0 * G_L_NUE * G_R_NUE - 32.0 * G_R_NUE**2, -16.0 * G_L_NUE * G_R_NUE]
    )
    assert [term.zeta for term in desc.mass_quartic_terms] == pytest.approx(
        [8.0 * (G_R_NUE**2 - G_L_NUE * G_R_NUE)]
    )
    assert desc.coupling_prefactor == pytest.approx(G_F_MEV**2)

    p1 = 1.4
    p2 = 0.9
    p3 = 0.7
    p4 = 0.6
    energies = (
        p1,
        p2,
        float(np.sqrt(p3 * p3 + electron_mass * electron_mass)),
        float(np.sqrt(p4 * p4 + electron_mass * electron_mass)),
    )
    momenta = (p1, p2, p3, p4)
    mu = {(1, 2): -0.25, (1, 4): 0.35}

    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
    )
    expected = _finite_mass_pair_annihilation_expected(
        species="nue",
        electron_mass=electron_mass,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
        include_fermi_prefactor=True,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-35)


def test_finite_mass_pair_annihilation_descriptor_uses_incoming_neutrino_for_nuebar_target() -> None:
    electron_mass = 0.51099895
    desc = build_finite_mass_pair_annihilation_process_descriptor("nuebar", electron_mass_momentum=electron_mass)

    assert desc.species == "nuebar"
    assert desc.particle_labels == ("nubar_alpha", "nu_alpha", "e_plus", "e_minus")
    assert [term.first_pair for term in desc.bilinear_terms] == [(2, 4), (1, 2)]
    assert [term.pair for term in desc.mass_terms] == [(1, 2), (2, 4)]

    p1 = 0.9
    p2 = 1.4
    p3 = 0.7
    p4 = 0.6
    energies = (
        p1,
        p2,
        float(np.sqrt(p3 * p3 + electron_mass * electron_mass)),
        float(np.sqrt(p4 * p4 + electron_mass * electron_mass)),
    )
    momenta = (p1, p2, p3, p4)
    mu = {(1, 2): -0.25, (2, 4): 0.35}

    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
    )
    expected = _finite_mass_pair_annihilation_expected(
        species="nuebar",
        electron_mass=electron_mass,
        energies=energies,
        momenta=momenta,
        mu_by_pair=mu,
        target_antineutrino=True,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-14)


def test_ur_diagonal_nunu_descriptor_matches_pairwise_reference_kernel() -> None:
    desc = build_ur_nunu_diagonal_process_descriptor(
        target_species="nue",
        partner_species="nux",
        epsilon_alpha_beta=0.375,
    )

    assert desc.process == "nu_nu_diagonal_ur"
    assert desc.species == "nue"
    assert desc.partner_species == "nux"
    assert desc.particle_labels == ("nu_alpha", "nu_beta", "nu_alpha", "nu_beta")
    assert [term.first_pair for term in desc.bilinear_terms] == [(1, 2), (3, 4)]
    assert [term.eta for term in desc.bilinear_terms] == pytest.approx([0.375, 0.375])

    energies = (1.2, 1.8, 0.9, 2.1)
    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        mu_by_pair=_zero_mu_by_pair(),
    )
    expected = 0.375 * ((energies[0] * energies[1]) ** 2 + (energies[2] * energies[3]) ** 2)
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-15)


def test_ur_nunu_pairwise_descriptor_sets_identical_and_offdiagonal_fierz_factors() -> None:
    identical = build_ur_nunu_pairwise_process_descriptor("nue", "nue")
    offdiagonal = build_ur_nunu_pairwise_process_descriptor("nue", "nux")

    assert identical.partner_species == "nue"
    assert offdiagonal.partner_species == "nux"
    assert [term.eta for term in identical.bilinear_terms] == pytest.approx([2.0, 2.0])
    assert [term.eta for term in offdiagonal.bilinear_terms] == pytest.approx([1.0, 1.0])

    energies = (1.2, 1.8, 0.9, 2.1)
    mu = _zero_mu_by_pair()
    identical_value = evaluate_pstf_process_matrix_element(identical, energies=energies, mu_by_pair=mu)
    offdiagonal_value = evaluate_pstf_process_matrix_element(offdiagonal, energies=energies, mu_by_pair=mu)
    assert identical_value == pytest.approx(2.0 * offdiagonal_value, rel=0.0, abs=1.0e-15)


def test_default_ur_weak_process_catalog_covers_supported_bridge_channels() -> None:
    catalog = build_default_ur_weak_process_catalog()
    keys = [pstf_process_descriptor_key(descriptor) for descriptor in catalog]

    assert len(catalog) == 12
    assert len(set(keys)) == len(keys)
    assert "nu_e_elastic_ur:nue" in keys
    assert "nu_e_elastic_ur:nuebar" in keys
    assert "nu_nubar_to_ee_ur:nux" in keys
    assert "nu_nu_diagonal_ur:nue:nux" in keys
    assert "nu_nu_diagonal_ur:nux:nuebar" in keys
    assert "nu_nu_diagonal_ur:nue:nue" not in keys
    assert "nu_nu_diagonal_ur:nux:nux" not in keys

    offdiagonal = next(
        descriptor
        for descriptor in catalog
        if pstf_process_descriptor_key(descriptor) == "nu_nu_diagonal_ur:nux:nue"
    )
    assert [term.eta for term in offdiagonal.bilinear_terms] == pytest.approx([1.0, 1.0])
    assert sum(key.startswith("nu_nu_diagonal_ur:") for key in keys) == 6


def test_default_supported_weak_process_catalog_uses_finite_mass_nue_elastic_descriptors() -> None:
    electron_mass = 0.25
    catalog = build_default_supported_weak_process_catalog(electron_mass_momentum=electron_mass)
    keys = [pstf_process_descriptor_key(descriptor) for descriptor in catalog]

    assert len(catalog) == 18
    assert len(set(keys)) == len(keys)
    assert "nu_e_minus_elastic_finite_mass_hm:nue" in keys
    assert "nu_e_plus_elastic_finite_mass_hm:nue" in keys
    assert "nu_e_minus_elastic_finite_mass_hm:nuebar" in keys
    assert "nu_e_plus_elastic_finite_mass_hm:nuebar" in keys
    assert "nu_e_minus_elastic_finite_mass_hm:nux" in keys
    assert "nu_e_plus_elastic_finite_mass_hm:nux" in keys
    assert "nu_e_elastic_ur:nue" not in keys
    assert "nu_nubar_to_ee_finite_mass_hm:nue" in keys
    assert "nu_nubar_to_ee_finite_mass_hm:nuebar" in keys
    assert "nu_nubar_to_ee_finite_mass_hm:nux" in keys
    assert "nu_nubar_to_ee_ur:nux" not in keys
    assert "nu_nu_diagonal_ur:nue:nux" in keys
    assert "nu_nu_diagonal_ur:nue:nue" in keys
    assert "nu_nu_diagonal_ur:nuebar:nuebar" in keys
    assert "nu_nu_diagonal_ur:nux:nux" in keys
    elastic = [
        descriptor
        for descriptor in catalog
        if descriptor.process in ("nu_e_minus_elastic_finite_mass_hm", "nu_e_plus_elastic_finite_mass_hm")
    ]
    assert len(elastic) == 6
    assert all(descriptor.electron_mass_momentum == pytest.approx(electron_mass) for descriptor in elastic)
    assert all(descriptor.mass_terms for descriptor in elastic)
    assert sum(pstf_process_particle_mode_labels(descriptor)[1] == "e_minus" for descriptor in elastic) == 3
    assert sum(pstf_process_particle_mode_labels(descriptor)[1] == "e_plus" for descriptor in elastic) == 3
    pair = [descriptor for descriptor in catalog if descriptor.process == "nu_nubar_to_ee_finite_mass_hm"]
    assert len(pair) == 3
    assert all(descriptor.mass_terms and descriptor.mass_quartic_terms for descriptor in pair)
    nunu = [descriptor for descriptor in catalog if descriptor.process == "nu_nu_diagonal_ur"]
    assert len(nunu) == 9
    identical = [
        descriptor
        for descriptor in nunu
        if descriptor.partner_species == descriptor.species
    ]
    assert len(identical) == 3
    assert all([term.eta for term in descriptor.bilinear_terms] == pytest.approx([2.0, 2.0]) for descriptor in identical)

    ur_catalog = build_default_supported_weak_process_catalog(electron_scattering_model="ur")
    assert [pstf_process_descriptor_key(descriptor) for descriptor in ur_catalog] == [
        pstf_process_descriptor_key(descriptor)
        for descriptor in build_default_ur_weak_process_catalog(include_identical_nunu=True)
    ]
    with pytest.raises(ValueError, match="electron_scattering_model"):
        build_default_supported_weak_process_catalog(electron_scattering_model="bad")


def test_default_ur_weak_process_catalog_can_include_identical_reference_descriptors() -> None:
    catalog = build_default_ur_weak_process_catalog(include_identical_nunu=True)
    keys = [pstf_process_descriptor_key(descriptor) for descriptor in catalog]

    assert len(catalog) == 15
    assert len(set(keys)) == len(keys)
    assert "nu_nu_diagonal_ur:nue:nue" in keys
    assert "nu_nu_diagonal_ur:nuebar:nuebar" in keys
    assert "nu_nu_diagonal_ur:nux:nux" in keys
    identical = next(
        descriptor
        for descriptor in catalog
        if pstf_process_descriptor_key(descriptor) == "nu_nu_diagonal_ur:nux:nux"
    )
    assert [term.eta for term in identical.bilinear_terms] == pytest.approx([2.0, 2.0])


def test_default_ur_weak_process_catalog_preserves_keys_with_fermi_prefactor() -> None:
    plain = build_default_ur_weak_process_catalog()
    with_prefactor = build_default_ur_weak_process_catalog(include_fermi_prefactor=True)

    assert [pstf_process_descriptor_key(descriptor) for descriptor in with_prefactor] == [
        pstf_process_descriptor_key(descriptor) for descriptor in plain
    ]
    assert all(
        descriptor.coupling_prefactor == pytest.approx(G_F_MEV**2)
        for descriptor in with_prefactor
    )


def test_process_catalog_public_package_exports_supported_catalog_api() -> None:
    assert exported_build_default_supported_weak_process_catalog is build_default_supported_weak_process_catalog
    assert exported_build_default_ur_weak_process_catalog is build_default_ur_weak_process_catalog
    assert (
        exported_build_finite_mass_pair_annihilation_process_descriptor
        is build_finite_mass_pair_annihilation_process_descriptor
    )
    assert (
        exported_build_finite_mass_nue_scattering_process_descriptor
        is build_finite_mass_nue_scattering_process_descriptor
    )
    assert exported_build_ur_nunu_pairwise_process_descriptor is build_ur_nunu_pairwise_process_descriptor
    assert exported_pstf_process_descriptor_key is pstf_process_descriptor_key
    assert exported_pstf_process_particle_mode_labels is pstf_process_particle_mode_labels


def test_pstf_process_particle_mode_labels_map_supported_ur_catalog() -> None:
    labels_by_key = {
        pstf_process_descriptor_key(descriptor): pstf_process_particle_mode_labels(descriptor)
        for descriptor in build_default_ur_weak_process_catalog()
    }

    assert labels_by_key["nu_e_elastic_ur:nue"] == ("nue", "e_pm", "nue", "e_pm")
    assert labels_by_key["nu_e_elastic_ur:nuebar"] == ("nuebar", "e_pm", "nuebar", "e_pm")
    assert labels_by_key["nu_nubar_to_ee_ur:nue"] == ("nue", "nuebar", "e_plus", "e_minus")
    assert labels_by_key["nu_nubar_to_ee_ur:nuebar"] == ("nuebar", "nue", "e_plus", "e_minus")
    assert labels_by_key["nu_nubar_to_ee_ur:nux"] == ("nux", "nux", "e_plus", "e_minus")
    assert labels_by_key["nu_nu_diagonal_ur:nux:nue"] == ("nux", "nue", "nux", "nue")


def test_default_ur_weak_process_catalog_builds_finite_radial_sources() -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
    )
    F1 = np.asarray([[0.31]])
    F2 = np.asarray([[0.23]])
    F3 = np.asarray([[0.37]])
    F4 = np.asarray([[0.19], [0.41]])

    max_abs_by_key: dict[str, float] = {}
    for descriptor in build_default_ur_weak_process_catalog():
        result = evaluate_pstf_process_radial_collision_source(
            descriptor,
            F1_modes=F1,
            F2_modes=F2,
            F3_modes=F3,
            F4_modes=F4,
            **common,
        )
        max_abs_by_key[pstf_process_descriptor_key(descriptor)] = float(np.max(np.abs(result.C_modes)))

    assert len(max_abs_by_key) == 12
    assert min(max_abs_by_key.values()) > 0.0
    assert np.isfinite(tuple(max_abs_by_key.values())).all()


def test_process_matrix_element_accepts_reversed_mu_pair_keys() -> None:
    desc = build_ur_nunu_diagonal_process_descriptor(epsilon_alpha_beta=1.0)

    energies = (2.0, 3.0, 5.0, 7.0)
    momenta = (0.5, 0.7, 1.1, 1.3)
    got = evaluate_pstf_process_matrix_element(
        desc,
        energies=energies,
        momenta=momenta,
        mu_by_pair={(2, 1): 0.25, (4, 3): -0.5},
    )
    pi12 = energies[0] * energies[1] - momenta[0] * momenta[1] * 0.25
    pi34 = energies[2] * energies[3] - momenta[2] * momenta[3] * (-0.5)
    assert got == pytest.approx(pi12**2 + pi34**2, rel=0.0, abs=1.0e-15)


def test_pstf_process_descriptor_builds_same_radial_grid_as_manual_terms() -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_pair_annihilation_process_descriptor("nue")
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.25]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
    )

    from_descriptor = build_pstf_process_radial_channel_kernel_grid(descriptor, **common)
    manual = build_pstf_radial_channel_kernel_grid(
        **common,
        bilinear_terms=descriptor.bilinear_terms,
        mass_terms=descriptor.mass_terms,
        electron_mass_momentum=descriptor.electron_mass_momentum,
        coupling_prefactor=descriptor.coupling_prefactor,
    )

    assert from_descriptor.kernel_grid_contract == "pstf_radial_channel_kernel_grid_v1"
    assert from_descriptor.K_by_monomial["34"] == pytest.approx(
        manual.K_by_monomial["34"],
        rel=0.0,
        abs=1.0e-18,
    )
    assert from_descriptor.radial_prefactors == pytest.approx(manual.radial_prefactors)


def test_pstf_process_radial_collision_source_evaluator_returns_concrete_modes() -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_nue_scattering_process_descriptor("nue")
    common = dict(
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
    )
    F1 = np.asarray([[0.31]])
    F2 = np.asarray([[0.23]])
    F3 = np.asarray([[0.37]])
    F4 = np.asarray([[0.19], [0.41]])

    result = evaluate_pstf_process_radial_collision_source(
        descriptor,
        F1_modes=F1,
        F2_modes=F2,
        F3_modes=F3,
        F4_modes=F4,
        **common,
    )
    grid = build_pstf_process_radial_channel_kernel_grid(descriptor, **common)
    manual = contract_pstf_radial_channel_kernel_grid(F1, F2, F3, F4, grid)

    assert result.process_contract == "pstf_process_radial_collision_source_v1"
    assert result.descriptor is descriptor
    assert result.grid.kernel_grid_contract == "pstf_radial_channel_kernel_grid_v1"
    assert result.contraction.radial_contract == "pstf_channel_radial_grid_contraction_v1"
    assert result.C_modes == pytest.approx(manual.C_modes, rel=0.0, abs=1.0e-18)
    assert abs(float(result.C_modes[0, 0])) > 0.0
    assert bool(result.grid.valid_radial_mask[0, 0, 0]) is True


def test_pstf_radial_result_exact_checks_do_not_use_allclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_nue_scattering_process_descriptor("nue")
    source = evaluate_pstf_process_radial_collision_source(
        descriptor,
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
        F1_modes=np.asarray([[0.31]]),
        F2_modes=np.asarray([[0.23]]),
        F3_modes=np.asarray([[0.37]]),
        F4_modes=np.asarray([[0.19], [0.41]]),
    )
    moment_weights = build_pstf_process_radial_moment_weights(
        source.grid,
        p1_weights=np.asarray([0.6]),
        number_power=2,
        energy_power=3,
    )

    def fail_allclose(*_args, **_kwargs):
        raise AssertionError("exact radial result checks should use array_equal")

    monkeypatch.setattr(catalog.np, "allclose", fail_allclose)

    rebuilt = PSTFProcessRadialCollisionResult(
        descriptor=source.descriptor,
        grid=source.grid,
        contraction=source.contraction,
        C_modes=source.contraction.C_modes.copy(),
    )
    moments = compute_pstf_process_radial_moments_preweighted(rebuilt, moment_weights)

    assert np.array_equal(rebuilt.C_modes, source.C_modes)
    assert moments.number_moment == pytest.approx(
        float(np.sum(moment_weights.number_weights * rebuilt.C_modes[:, moment_weights.mode_index]))
    )


def test_pstf_radial_projection_reuses_precomputed_moment_pinv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_nue_scattering_process_descriptor("nue")
    source = evaluate_pstf_process_radial_collision_source(
        descriptor,
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
        F1_modes=np.asarray([[0.31]]),
        F2_modes=np.asarray([[0.23]]),
        F3_modes=np.asarray([[0.37]]),
        F4_modes=np.asarray([[0.19], [0.41]]),
    )
    moment_weights = build_pstf_process_radial_moment_weights(
        source.grid,
        p1_weights=np.asarray([0.6]),
        number_power=2,
        energy_power=3,
    )

    def fail_pinv(*_args, **_kwargs):
        raise AssertionError("projection should reuse precomputed moment pseudo-inverse")

    monkeypatch.setattr(catalog.np.linalg, "pinv", fail_pinv)

    projected = project_pstf_radial_collision_source_number_neutral(source, moment_weights)

    assert projected.process_contract == "pstf_nunu_number_neutral_radial_source_v1"


def test_pstf_process_radial_moments_integrate_monopole_source_modes() -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_nue_scattering_process_descriptor("nue")
    result = evaluate_pstf_process_radial_collision_source(
        descriptor,
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
        F1_modes=np.asarray([[0.31]]),
        F2_modes=np.asarray([[0.23]]),
        F3_modes=np.asarray([[0.37]]),
        F4_modes=np.asarray([[0.19], [0.41]]),
    )

    moments = compute_pstf_process_radial_moments(
        result,
        p1_weights=np.asarray([0.6]),
        number_power=2,
        energy_power=3,
    )

    c0 = float(result.C_modes[0, 0])
    assert moments.moment_contract == "pstf_process_radial_moments_v1"
    assert moments.mode_index == 0
    assert moments.number_moment == pytest.approx(0.6 * 2.0**2 * c0, rel=0.0, abs=1.0e-18)
    assert moments.energy_moment == pytest.approx(0.6 * 2.0**3 * c0, rel=0.0, abs=1.0e-18)
    assert moments.max_abs_C_mode == pytest.approx(abs(c0), rel=0.0, abs=1.0e-18)
    assert moments.process == "nu_e_elastic_ur"
    assert moments.species == "nue"


def test_pstf_process_radial_moments_preweighted_matches_public_helper() -> None:
    basis = np.asarray([[1.0]])
    weights = np.asarray([1.0])
    directions = np.asarray([[0.0, 0.0, 1.0]])
    descriptor = build_ur_nue_scattering_process_descriptor("nue")
    result = evaluate_pstf_process_radial_collision_source(
        descriptor,
        basis_by_particle=(basis, basis, basis, basis),
        angular_weights_by_particle=(weights, weights, weights, weights),
        direction_vectors_by_particle=(directions, directions, directions, directions),
        momentum_delta_weights=np.asarray([[[[0.5]]]]),
        p1_energies=np.asarray([2.0]),
        p2_energies=np.asarray([3.0]),
        p3_energies=np.asarray([1.5]),
        p4_energy_grid=np.asarray([3.0, 4.0]),
        p1_momenta=np.asarray([0.2]),
        p2_momenta=np.asarray([0.3]),
        p3_momenta=np.asarray([0.4]),
        p4_momentum_grid=np.asarray([0.35, 0.45]),
        q2_weights=np.asarray([0.7]),
        q3_weights=np.asarray([0.8]),
        radial_prefactor_config=PSTFRadialInvariantPrefactorConfig(g2=2.0, g3=1.0, g4=2.0),
        F1_modes=np.asarray([[0.31]]),
        F2_modes=np.asarray([[0.23]]),
        F3_modes=np.asarray([[0.37]]),
        F4_modes=np.asarray([[0.19], [0.41]]),
    )

    weight_spec = build_pstf_process_radial_moment_weights(
        result.grid,
        p1_weights=np.asarray([0.6]),
        number_power=2,
        energy_power=3,
    )
    public = compute_pstf_process_radial_moments(
        result,
        p1_weights=np.asarray([0.6]),
        number_power=2,
        energy_power=3,
    )
    fast = compute_pstf_process_radial_moments_preweighted(result, weight_spec)

    assert weight_spec.moment_weight_contract == "pstf_process_radial_moment_weights_v1"
    assert fast.number_moment == pytest.approx(public.number_moment, rel=0.0, abs=1.0e-18)
    assert fast.energy_moment == pytest.approx(public.energy_moment, rel=0.0, abs=1.0e-18)
    assert fast.C_mode == pytest.approx(public.C_mode, rel=0.0, abs=1.0e-18)


def test_pstf_process_catalog_rejects_unknown_species_and_bad_epsilon() -> None:
    with pytest.raises(ValueError, match="species"):
        build_ur_nue_scattering_process_descriptor("numu")
    with pytest.raises(ValueError, match="epsilon_alpha_beta"):
        build_ur_nunu_diagonal_process_descriptor(epsilon_alpha_beta=0.0)
    with pytest.raises(ValueError, match="species_labels"):
        build_default_ur_weak_process_catalog(species_labels=("nue", "nue"))
