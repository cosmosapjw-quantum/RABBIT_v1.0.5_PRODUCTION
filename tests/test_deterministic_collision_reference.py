from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.species import BANK_DEGENERACY, Species
from rabbit.collisions.nu_e_scattering import NuEScatteringOperator
from rabbit.collisions.pair_processes import PairProcessOperator
from rabbit.collisions.deterministic_reference import (
    COLLISION_STATISTICAL_COEFFICIENTS,
    COLLISION_STATISTICAL_MONOMIALS,
    build_fixed_collision_quadrature,
    collision_statistical_factor,
    collision_statistical_polynomial_terms,
    detailed_balance_residual,
    evaluate_nue_scattering_reference_batch,
    evaluate_nunu_diagonal_twoto2_reference_batch,
    evaluate_nue_scattering_reference,
    evaluate_pair_annihilation_reference_batch,
    evaluate_nunu_diagonal_reference,
    evaluate_nunu_diagonal_twoto2_reference,
    evaluate_pair_annihilation_reference,
)
from rabbit.transport.augmented_pstf_distribution import fermi_dirac_from_logit


def test_fixed_collision_quadrature_is_replay_deterministic() -> None:
    a = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=5,
        N_nue_y3=6,
        N_pair_y2=7,
        N_pair_leg=8,
    )
    b = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=5,
        N_nue_y3=6,
        N_pair_y2=7,
        N_pair_leg=8,
    )

    assert a == b
    assert np.all(np.asarray(a.q_nodes) > 0.0)
    assert np.all(np.asarray(a.nue_y2_nodes) > 0.0)
    assert np.all(np.abs(np.asarray(a.pair_leg_nodes)) < 1.0)


def test_fixed_collision_quadrature_rejects_non_positive_orders() -> None:
    with pytest.raises(ValueError, match="N_q must be positive"):
        build_fixed_collision_quadrature(
            N_q=0,
            N_nue_y2=4,
            N_nue_y3=4,
            N_pair_y2=4,
            N_pair_leg=4,
        )

    with pytest.raises(ValueError, match="N_q must be an exact integer"):
        build_fixed_collision_quadrature(
            N_q=4.5,
            N_nue_y2=4,
            N_nue_y3=4,
            N_pair_y2=4,
            N_pair_leg=4,
        )


def test_collision_statistical_factor_vanishes_at_fd_detailed_balance() -> None:
    y1 = np.array([0.2, 0.5, 1.0])
    y2 = np.array([0.7, 1.1, 1.3])
    y3 = np.array([0.4, 0.8, 1.2])
    y4 = y1 + y2 - y3
    assert np.all(y4 > 0.0)

    f1 = fermi_dirac_from_logit(y1)
    f2 = fermi_dirac_from_logit(y2)
    f3 = fermi_dirac_from_logit(y3)
    f4 = fermi_dirac_from_logit(y4)

    assert detailed_balance_residual(f1, f2, f3, f4) < 1.0e-15


def test_collision_statistical_factor_is_six_monomial_pauli_polynomial() -> None:
    f1 = np.array([0.10, 0.35, 0.80])
    f2 = np.array([0.20, 0.45, 0.70])
    f3 = np.array([0.30, 0.55, 0.60])
    f4 = np.array([0.40, 0.65, 0.50])

    terms = collision_statistical_polynomial_terms(f1, f2, f3, f4)
    direct = f3 * f4 * (1.0 - f1) * (1.0 - f2) - f1 * f2 * (1.0 - f3) * (1.0 - f4)

    assert COLLISION_STATISTICAL_MONOMIALS == ("34", "12", "123", "124", "134", "234")
    assert COLLISION_STATISTICAL_COEFFICIENTS == {
        "34": 1.0,
        "12": -1.0,
        "123": 1.0,
        "124": 1.0,
        "134": -1.0,
        "234": -1.0,
    }
    assert tuple(terms) == COLLISION_STATISTICAL_MONOMIALS
    assert "1234" not in terms
    assert np.allclose(sum(terms.values()), direct, rtol=0.0, atol=1.0e-16)
    assert np.allclose(collision_statistical_factor(f1, f2, f3, f4), direct, rtol=0.0, atol=1.0e-16)


def test_collision_statistical_factor_detects_nonequilibrium() -> None:
    S = collision_statistical_factor(0.1, 0.2, 0.3, 0.4)

    assert abs(float(S)) > 0.0


def test_collision_statistical_factor_rejects_unphysical_occupancy() -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        collision_statistical_factor(1.2, 0.2, 0.3, 0.4)


def test_deterministic_references_use_validated_internal_statistical_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rabbit.collisions import deterministic_reference as reference

    quadrature = build_fixed_collision_quadrature(
        N_q=2,
        N_nue_y2=2,
        N_nue_y3=2,
        N_pair_y2=2,
        N_pair_leg=2,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = np.clip(fermi_dirac_from_logit(q) * (1.0 + 0.02 * q / float(np.max(q))), 0.0, 1.0)

    def fail_public_wrapper(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("reference kernels must not call the public validation wrapper")

    monkeypatch.setattr(reference, "collision_statistical_factor", fail_public_wrapper)

    scatter = evaluate_nue_scattering_reference(f, quadrature=quadrature, T_MeV=1.0, species="nue")
    pair = evaluate_pair_annihilation_reference(f, f, quadrature=quadrature, T_MeV=1.0, species="nue")
    nunu = evaluate_nunu_diagonal_twoto2_reference(
        f,
        f,
        quadrature=quadrature,
        T_MeV=1.0,
        gamma=1.0,
        species="nue",
    )

    assert scatter.C.shape == f.shape
    assert pair.C.shape == f.shape
    assert nunu.C.shape == f.shape


def test_deterministic_reference_vectorized_kernels_match_legacy_scalar_loops() -> None:
    from rabbit.collisions import deterministic_reference as reference

    quadrature = build_fixed_collision_quadrature(
        N_q=5,
        N_nue_y2=5,
        N_nue_y3=5,
        N_pair_y2=5,
        N_pair_leg=5,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = np.clip(fermi_dirac_from_logit(q) * (1.0 + 0.03 * np.sin(q)), 0.0, 1.0)
    partner = np.clip(fermi_dirac_from_logit(q) * (1.0 - 0.02 * np.cos(q)), 0.0, 1.0)

    def legacy_scattering() -> tuple[np.ndarray, float]:
        T = reference._positive_temperature(1.0)
        eta_e = reference._electron_degeneracy_parameter(0.02, T)
        arrays = reference._quadrature_arrays(quadrature)
        reference._require_matching_nodes(
            arrays["q_nodes"],
            arrays["nue_y3_nodes"],
            partner_name="nue_y3_nodes",
            process="nu-e scattering",
        )
        checked = reference._distribution_1d(f, q_nodes=q, name="f_nu")
        _species, G_L, G_R = reference._couplings_for_species("nue")
        y2_nodes = arrays["nue_y2_nodes"]
        y2_weights = reference._laguerre_plain_weights(y2_nodes, arrays["nue_y2_weights"])
        y3_nodes = arrays["nue_y3_nodes"]
        y3_weights = reference._laguerre_plain_weights(y3_nodes, arrays["nue_y3_weights"])
        prefactor = reference.hm_reduced_collision_prefactor(T)
        C = np.zeros_like(q, dtype=float)
        max_balance = 0.0
        for i, y1 in enumerate(q):
            f1 = float(checked[i])
            total = 0.0
            for y3, w3 in zip(y3_nodes, y3_weights):
                f3 = reference._interp_distribution(float(y3), q_nodes=q, f_values=checked)
                for y2, w2 in zip(y2_nodes, y2_weights):
                    y4 = float(y1 + y2 - y3)
                    if y4 <= 0.0:
                        continue
                    f2 = float(reference._fermi_dirac(y2, eta=eta_e))
                    f4 = float(reference._fermi_dirac(y4, eta=eta_e))
                    S = float(reference._collision_statistical_factor_unchecked(f1, f2, f3, f4))
                    max_balance = max(max_balance, abs(S))
                    M2 = reference._scattering_matrix_element(G_L, G_R, float(y1), float(y2), float(y3), y4)
                    total += float(w3 * w2) * M2 * S
            C[i] = prefactor * total / max(float(y1) ** 2, 1.0e-30)
        return C, max_balance

    def legacy_pair() -> tuple[np.ndarray, float]:
        T = reference._positive_temperature(1.0)
        eta_e = reference._electron_degeneracy_parameter(0.02, T)
        arrays = reference._quadrature_arrays(quadrature)
        reference._require_matching_nodes(
            arrays["q_nodes"],
            arrays["pair_y2_nodes"],
            partner_name="pair_y2_nodes",
            process="pair-process",
        )
        target = reference._distribution_1d(f, q_nodes=q, name="f_target")
        partner_checked = reference._distribution_1d(partner, q_nodes=q, name="f_partner")
        _species, G_L, G_R = reference._couplings_for_species("nue")
        y2_nodes = arrays["pair_y2_nodes"]
        y2_weights = reference._laguerre_plain_weights(y2_nodes, arrays["pair_y2_weights"])
        leg_nodes = arrays["pair_leg_nodes"]
        leg_weights = arrays["pair_leg_weights"]
        prefactor = reference.hm_reduced_collision_prefactor(T)
        C = np.zeros_like(q, dtype=float)
        max_balance = 0.0
        for i, y1 in enumerate(q):
            f1 = float(target[i])
            total = 0.0
            for y2, w2 in zip(y2_nodes, y2_weights):
                y_sum = float(y1 + y2)
                if y_sum <= 0.0:
                    continue
                f2 = reference._interp_distribution(float(y2), q_nodes=q, f_values=partner_checked)
                y3_nodes = 0.5 * y_sum * (leg_nodes + 1.0)
                y3_weights = 0.5 * y_sum * leg_weights
                for y3, w3 in zip(y3_nodes, y3_weights):
                    y4 = float(y_sum - y3)
                    if y4 <= 0.0:
                        continue
                    f3 = float(reference._fermi_dirac(y3, eta=-eta_e))
                    f4 = float(reference._fermi_dirac(y4, eta=eta_e))
                    S = float(reference._collision_statistical_factor_unchecked(f1, f2, f3, f4))
                    max_balance = max(max_balance, abs(S))
                    M2 = reference._pair_matrix_element(G_L, G_R, float(y1), float(y2), float(y3), y4)
                    total += float(w2 * w3) * M2 * S
            C[i] = prefactor * total / max(float(y1) ** 2, 1.0e-30)
        return C, max_balance

    nunu_temperature = 1.7
    nunu_epsilon = 0.4

    def legacy_nunu() -> tuple[np.ndarray, float]:
        gamma = reference._nonnegative_rate(3.0, name="gamma")
        T = reference._positive_temperature(nunu_temperature)
        epsilon = float(nunu_epsilon)
        arrays = reference._quadrature_arrays(quadrature)
        checked_a = reference._distribution_1d(f, q_nodes=q, name="f_alpha")
        checked_b = reference._distribution_1d(partner, q_nodes=q, name="f_beta")
        plain_weights = reference._laguerre_plain_weights(q, arrays["q_weights"])
        prefactor = epsilon * gamma * reference.hm_reduced_collision_prefactor(T)
        C = np.zeros_like(q, dtype=float)
        max_balance = 0.0
        for i, y1 in enumerate(q):
            f1 = float(checked_a[i])
            total = 0.0
            for y3, f3, w3 in zip(q, checked_a, plain_weights):
                for y2, f2, w2 in zip(q, checked_b, plain_weights):
                    y4 = float(y1 + y2 - y3)
                    if y4 <= 0.0:
                        continue
                    f4 = reference._interp_distribution(y4, q_nodes=q, f_values=checked_b)
                    S = float(reference._collision_statistical_factor_unchecked(f1, float(f2), float(f3), f4))
                    max_balance = max(max_balance, abs(S))
                    M2 = float((y1 * y2) ** 2 + (y3 * y4) ** 2)
                    total += float(w3 * w2) * M2 * S
            C[i] = prefactor * total / max(float(y1) ** 2, 1.0e-30)
        return C, max_balance

    scatter = evaluate_nue_scattering_reference(
        f,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.02,
    )
    pair = evaluate_pair_annihilation_reference(
        f,
        partner,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.02,
    )
    nunu = evaluate_nunu_diagonal_twoto2_reference(
        f,
        partner,
        quadrature=quadrature,
        T_MeV=nunu_temperature,
        gamma=3.0,
        epsilon_alpha_beta=nunu_epsilon,
        species="nue",
    )

    legacy_C, legacy_balance = legacy_scattering()
    assert np.allclose(scatter.C, legacy_C, rtol=1.0e-14, atol=1.0e-34)
    assert scatter.detailed_balance_residual == pytest.approx(legacy_balance, rel=1.0e-14)
    legacy_C, legacy_balance = legacy_pair()
    assert np.allclose(pair.C, legacy_C, rtol=1.0e-14, atol=1.0e-34)
    assert pair.detailed_balance_residual == pytest.approx(legacy_balance, rel=1.0e-14)
    legacy_C, legacy_balance = legacy_nunu()
    assert np.allclose(nunu.C, legacy_C, rtol=1.0e-14, atol=1.0e-34)
    assert nunu.detailed_balance_residual == pytest.approx(legacy_balance, rel=1.0e-14)


def test_deterministic_reference_batch_helpers_match_single_references() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=5,
        N_nue_y2=5,
        N_nue_y3=5,
        N_pair_y2=5,
        N_pair_leg=5,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f_batch = np.asarray(
        [
            fermi_dirac_from_logit(q + 0.07 * np.sin(q)),
            fermi_dirac_from_logit(0.91 * q + 0.03 * q**2),
            fermi_dirac_from_logit(1.08 * q - 0.04 * np.cos(q)),
        ],
        dtype=float,
    )
    partner_batch = np.asarray(
        [
            fermi_dirac_from_logit(1.03 * q - 0.02 * q**2),
            fermi_dirac_from_logit(0.86 * q + 0.05 * np.cos(q)),
            fermi_dirac_from_logit(1.12 * q + 0.03 * np.sin(2.0 * q)),
        ],
        dtype=float,
    )

    scatter_batch = evaluate_nue_scattering_reference_batch(
        f_batch,
        quadrature=quadrature,
        T_MeV=1.1,
        species="nue",
        electron_chemical_potential_MeV=0.018,
    )
    pair_batch = evaluate_pair_annihilation_reference_batch(
        f_batch,
        partner_batch,
        quadrature=quadrature,
        T_MeV=1.1,
        species="nue",
        electron_chemical_potential_MeV=0.018,
    )
    nunu_batch = evaluate_nunu_diagonal_twoto2_reference_batch(
        f_batch,
        partner_batch,
        quadrature=quadrature,
        T_MeV=1.04,
        gamma=2.5,
        epsilon_alpha_beta=0.7,
        species="nue",
    )

    assert scatter_batch.C.shape == f_batch.shape
    assert pair_batch.C.shape == f_batch.shape
    assert nunu_batch.C.shape == f_batch.shape
    for idx in range(f_batch.shape[0]):
        scatter = evaluate_nue_scattering_reference(
            f_batch[idx],
            quadrature=quadrature,
            T_MeV=1.1,
            species="nue",
            electron_chemical_potential_MeV=0.018,
        )
        pair = evaluate_pair_annihilation_reference(
            f_batch[idx],
            partner_batch[idx],
            quadrature=quadrature,
            T_MeV=1.1,
            species="nue",
            electron_chemical_potential_MeV=0.018,
        )
        nunu = evaluate_nunu_diagonal_twoto2_reference(
            f_batch[idx],
            partner_batch[idx],
            quadrature=quadrature,
            T_MeV=1.04,
            gamma=2.5,
            epsilon_alpha_beta=0.7,
            species="nue",
        )
        assert np.allclose(scatter_batch.C[idx], scatter.C, rtol=1.0e-14, atol=1.0e-34)
        assert np.allclose(pair_batch.C[idx], pair.C, rtol=1.0e-14, atol=1.0e-34)
        assert np.allclose(nunu_batch.C[idx], nunu.C, rtol=1.0e-14, atol=1.0e-34)
        assert scatter_batch.energy_transfer_by_species["nue"][idx] == pytest.approx(
            scatter.energy_transfer_by_species["nue"], rel=1.0e-14
        )
        assert pair_batch.number_residual[idx] == pytest.approx(pair.number_residual, rel=1.0e-14)
        assert nunu_batch.detailed_balance_residual[idx] == pytest.approx(
            nunu.detailed_balance_residual, rel=1.0e-14
        )


def test_legacy_python_collision_operators_share_six_monomial_factor() -> None:
    f1, f2, f3, f4 = 0.11, 0.23, 0.37, 0.41
    expected = float(collision_statistical_factor(f1, f2, f3, f4))

    assert NuEScatteringOperator()._statistical_factor(f1, f2, f3, f4) == pytest.approx(expected)
    assert PairProcessOperator()._statistical_factor(f1, f2, f3, f4) == pytest.approx(expected)


def test_nue_scattering_reference_vanishes_at_fd_detailed_balance() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    result = evaluate_nue_scattering_reference(f, quadrature=quadrature, T_MeV=1.0, species="nue")

    assert result.C.shape == f.shape
    assert np.max(np.abs(result.C)) < 1.0e-28
    assert result.detailed_balance_residual < 1.0e-14
    assert result.quadrature_contract == "fixed_gauss_sn_v1"


def test_nue_scattering_reference_supports_signed_electron_mu_blocking() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    fd = fermi_dirac_from_logit(q)
    distorted = np.clip(fd + 0.08 * np.exp(-q), 0.0, 1.0)

    balanced = evaluate_nue_scattering_reference(
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.2,
    )
    zero_mu = evaluate_nue_scattering_reference(
        distorted,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.0,
    )
    signed_mu = evaluate_nue_scattering_reference(
        distorted,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.2,
    )
    signed_mu_negative = evaluate_nue_scattering_reference(
        distorted,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nuebar",
        electron_chemical_potential_MeV=-0.2,
    )

    assert balanced.detailed_balance_residual < 1.0e-14
    assert set(signed_mu_negative.energy_transfer_by_species) == {"nuebar"}
    assert np.max(np.abs(signed_mu.C - zero_mu.C)) > 0.0
    assert np.max(np.abs(signed_mu_negative.C - zero_mu.C)) > 0.0
    with pytest.raises(ValueError, match="electron_chemical_potential_MeV"):
        evaluate_nue_scattering_reference(
            distorted,
            quadrature=quadrature,
            T_MeV=1.0,
            species="nue",
            electron_chemical_potential_MeV=float("nan"),
        )


def test_nue_scattering_reference_rejects_unmatched_fd_interpolation_grid() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=5,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    with pytest.raises(ValueError, match="requires q_nodes to match nue_y3_nodes"):
        evaluate_nue_scattering_reference(f, quadrature=quadrature, T_MeV=1.0)


def test_nue_scattering_reference_detects_nonequilibrium() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = 0.9 * fermi_dirac_from_logit(q)

    result = evaluate_nue_scattering_reference(f, quadrature=quadrature, T_MeV=1.0, species="nux")

    assert np.max(np.abs(result.C)) > 0.0
    assert np.isfinite(result.energy_transfer_by_species["nux"])
    assert np.isfinite(result.number_residual)


def test_nue_scattering_reference_has_replay_stable_polynomial_values() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = 0.9 * fermi_dirac_from_logit(q)

    result = evaluate_nue_scattering_reference(f, quadrature=quadrature, T_MeV=1.0, species="nux")

    assert result.C == pytest.approx(
        np.array(
            [
                4.34496120259406211e-25,
                1.69381908519573465e-27,
                -4.25385896971177635e-28,
                -8.86658886297798405e-30,
            ]
        ),
        rel=1.0e-12,
        abs=1.0e-35,
    )
    assert result.energy_transfer_by_species["nux"] == pytest.approx(-1.61319536068136122e-25, rel=1.0e-12, abs=1.0e-35)
    assert result.number_residual == pytest.approx(1.13485076118620539e-26, rel=1.0e-12, abs=1.0e-36)
    assert result.detailed_balance_residual == pytest.approx(1.52490334379220410e-03, rel=1.0e-12, abs=1.0e-15)


def test_pair_annihilation_reference_vanishes_at_fd_detailed_balance() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    result = evaluate_pair_annihilation_reference(
        f,
        f,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nuebar",
    )

    assert result.C.shape == f.shape
    assert np.max(np.abs(result.C)) < 1.0e-28
    assert result.detailed_balance_residual < 1.0e-14


def test_pair_annihilation_reference_supports_signed_electron_mu_blocking() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    fd = fermi_dirac_from_logit(q)
    distorted = np.clip(fd + 0.08 * np.exp(-q), 0.0, 1.0)

    balanced = evaluate_pair_annihilation_reference(
        fd,
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.2,
    )
    zero_mu = evaluate_pair_annihilation_reference(
        distorted,
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.0,
    )
    signed_mu = evaluate_pair_annihilation_reference(
        distorted,
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=0.2,
    )
    signed_mu_negative = evaluate_pair_annihilation_reference(
        distorted,
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nue",
        electron_chemical_potential_MeV=-0.2,
    )
    balanced_antinue_negative = evaluate_pair_annihilation_reference(
        fd,
        fd,
        quadrature=quadrature,
        T_MeV=1.0,
        species="nuebar",
        electron_chemical_potential_MeV=-0.2,
    )

    assert balanced.detailed_balance_residual < 1.0e-14
    assert balanced_antinue_negative.detailed_balance_residual < 1.0e-14
    assert set(balanced_antinue_negative.energy_transfer_by_species) == {"nuebar"}
    assert np.max(np.abs(signed_mu.C - zero_mu.C)) > 0.0
    assert np.max(np.abs(signed_mu_negative.C - zero_mu.C)) > 0.0
    assert np.max(np.abs(signed_mu_negative.C - signed_mu.C)) > 0.0
    with pytest.raises(ValueError, match="electron_chemical_potential_MeV"):
        evaluate_pair_annihilation_reference(
            distorted,
            fd,
            quadrature=quadrature,
            T_MeV=1.0,
            species="nue",
            electron_chemical_potential_MeV=float("nan"),
        )


def test_pair_annihilation_reference_rejects_unmatched_partner_grid() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=5,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    with pytest.raises(ValueError, match="requires q_nodes to match pair_y2_nodes"):
        evaluate_pair_annihilation_reference(f, f, quadrature=quadrature, T_MeV=1.0)


def test_pair_annihilation_reference_detects_nonequilibrium_and_rejects_bad_input() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)
    distorted = 0.85 * f

    result = evaluate_pair_annihilation_reference(distorted, f, quadrature=quadrature, T_MeV=1.0, species="nue")

    assert np.max(np.abs(result.C)) > 0.0
    assert np.isfinite(result.energy_transfer_by_species["nue"])

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        evaluate_pair_annihilation_reference(1.2 * np.ones_like(f), f, quadrature=quadrature, T_MeV=1.0)


def test_pair_annihilation_reference_has_replay_stable_polynomial_values() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)
    distorted = 0.85 * f

    result = evaluate_pair_annihilation_reference(distorted, f, quadrature=quadrature, T_MeV=1.0, species="nue")

    assert result.C == pytest.approx(
        np.array(
            [
                4.25552567200949263e-23,
                9.44288353695837994e-25,
                9.23147498018590255e-26,
                3.65622867603994910e-27,
            ]
        ),
        rel=1.0e-12,
        abs=1.0e-33,
    )
    assert result.energy_transfer_by_species["nue"] == pytest.approx(6.24459604518936051e-23, rel=1.0e-12, abs=1.0e-33)
    assert result.number_residual == pytest.approx(1.85734843050565568e-23, rel=1.0e-12, abs=1.0e-33)
    assert result.detailed_balance_residual == pytest.approx(1.53044840543285487e-02, rel=1.0e-12, abs=1.0e-14)


def test_pair_annihilation_reference_labels_target_species_for_asymmetric_inputs() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)
    target = 0.85 * f

    result = evaluate_pair_annihilation_reference(target, f, quadrature=quadrature, T_MeV=1.0, species="nuebar")
    swapped = evaluate_pair_annihilation_reference(f, target, quadrature=quadrature, T_MeV=1.0, species="nue")

    assert set(result.energy_transfer_by_species) == {"nuebar"}
    assert np.max(np.abs(result.C - swapped.C)) > 0.0


def test_nunu_diagonal_reference_is_quiet_for_common_distribution() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    results = evaluate_nunu_diagonal_reference(
        {"nue": f, "nuebar": f, "nux": f},
        quadrature=quadrature,
        T_nu_e=1.0,
        T_nu_x=1.0,
        gamma=2.0,
    )

    assert set(results) == {"nue", "nuebar", "nux"}
    assert all(np.max(np.abs(result.C)) == pytest.approx(0.0) for result in results.values())
    assert all(result.detailed_balance_residual == pytest.approx(0.0) for result in results.values())


def test_nunu_diagonal_reference_conserves_weighted_species_moments() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    results = evaluate_nunu_diagonal_reference(
        {
            "nue": 0.80 * f,
            "nuebar": 0.95 * f,
            "nux": 1.05 * f,
        },
        quadrature=quadrature,
        T_nu_e=1.0,
        T_nu_x=0.9,
        gamma=3.0,
    )

    assert np.max(np.abs(results["nue"].C)) > 0.0
    assert np.max(np.abs(results["nux"].C)) > 0.0
    energy = results["nue"].energy_transfer_by_species
    assert set(energy) == {"nue", "nuebar", "nux"}
    assert abs(energy["nue"]) > 0.0
    weighted_energy = sum(BANK_DEGENERACY[Species(key)] * value for key, value in energy.items())
    assert abs(weighted_energy) < 1.0e-15
    assert all(result.number_residual < 1.0e-15 for result in results.values())
    assert results["nue"].detailed_balance_residual > 0.0
    assert results["nux"].detailed_balance_residual > 0.0


def test_nunu_diagonal_twoto2_reference_is_quiet_at_fd_to_interpolation_tolerance() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=12,
        N_nue_y2=4,
        N_nue_y3=12,
        N_pair_y2=12,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    result = evaluate_nunu_diagonal_twoto2_reference(
        f,
        f,
        quadrature=quadrature,
        T_MeV=1.0,
        gamma=2.0,
        epsilon_alpha_beta=1.0,
        species="nue",
    )

    assert np.max(np.abs(result.C)) < 1.0e-22
    assert result.energy_transfer_by_species["nue"] == pytest.approx(0.0, abs=1.0e-22)


def test_nunu_diagonal_twoto2_reference_has_replay_stable_polynomial_values() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    result = evaluate_nunu_diagonal_twoto2_reference(
        0.5 * f,
        1.2 * f,
        quadrature=quadrature,
        T_MeV=1.0,
        gamma=2.0,
        epsilon_alpha_beta=1.0,
        species="nue",
    )

    assert result.C == pytest.approx(
        np.array(
            [
                3.69933818669869885e-23,
                8.93805842205367016e-25,
                2.10437355076981463e-25,
                6.28383838877293332e-27,
            ]
        ),
        rel=1.0e-12,
        abs=1.0e-33,
    )
    assert result.energy_transfer_by_species["nue"] == pytest.approx(1.15923420545259778e-22, rel=1.0e-12, abs=1.0e-32)
    assert result.number_residual == pytest.approx(2.81086743037296522e-23, rel=1.0e-12, abs=1.0e-33)
    assert result.detailed_balance_residual == pytest.approx(7.11621560436361683e-03, rel=1.0e-12, abs=1.0e-15)


def test_nunu_diagonal_reference_rejects_invalid_inputs() -> None:
    quadrature = build_fixed_collision_quadrature(
        N_q=4,
        N_nue_y2=4,
        N_nue_y3=4,
        N_pair_y2=4,
        N_pair_leg=4,
    )
    q = np.asarray(quadrature.q_nodes, dtype=float)
    f = fermi_dirac_from_logit(q)

    with pytest.raises(ValueError, match="must contain exactly"):
        evaluate_nunu_diagonal_reference({"nue": f, "nux": f}, quadrature=quadrature, T_nu_e=1.0, T_nu_x=1.0, gamma=1.0)

    with pytest.raises(ValueError, match="shape matching"):
        evaluate_nunu_diagonal_reference(
            {"nue": f[:-1], "nuebar": f, "nux": f},
            quadrature=quadrature,
            T_nu_e=1.0,
            T_nu_x=1.0,
            gamma=1.0,
        )

    with pytest.raises(ValueError, match="gamma must be provided"):
        evaluate_nunu_diagonal_reference(
            {"nue": f, "nuebar": f, "nux": f},
            quadrature=quadrature,
            T_nu_e=1.0,
            T_nu_x=1.0,
        )

    with pytest.raises(ValueError, match="gamma must be non-negative"):
        evaluate_nunu_diagonal_reference(
            {"nue": f, "nuebar": f, "nux": f},
            quadrature=quadrature,
            T_nu_e=1.0,
            T_nu_x=1.0,
            gamma=-1.0,
        )

    with pytest.raises(ValueError, match="T_MeV must be positive"):
        evaluate_nunu_diagonal_reference(
            {"nue": f, "nuebar": f, "nux": f},
            quadrature=quadrature,
            T_nu_e=0.0,
            T_nu_x=1.0,
            gamma=1.0,
        )

    with pytest.raises(ValueError, match="gamma must be provided"):
        evaluate_nunu_diagonal_twoto2_reference(f, f, quadrature=quadrature, T_MeV=1.0, gamma=None)

    with pytest.raises(ValueError, match="epsilon_alpha_beta must be positive"):
        evaluate_nunu_diagonal_twoto2_reference(
            f,
            f,
            quadrature=quadrature,
            T_MeV=1.0,
            gamma=1.0,
            epsilon_alpha_beta=0.0,
        )
