"""Full static ``T_gamma`` collision-action tangent for D-080B.

The frozen private comparator uses ``T_gamma`` both in electron/positron
Fermi-Dirac factors and in the incoming-electron half-line quadrature.  The
latter moves the quadrature measure, relativistic two-body kinematics, weak
matrix elements, outgoing-neutrino interpolation points, and support masks.
This module differentiates the same fixed-support numerical operator.

Only the static collision action is covered.  ``T_cm`` and the neutrino
cloglog spectra are fixed input data, the self-interaction block has zero
``T_gamma`` tangent, and no integrator or full RHS Jacobian is called.

Units follow the comparator's natural-unit MeV convention.  The returned
collision derivative has one fewer power of MeV than the primal action.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_tangent_primitives import (
    matrix,
    modal_product,
    pauli_gain_minus_loss_jvp,
    safe_relative,
)
from scripts.audit._d080_tgamma_primitives import (
    D080TgammaLinearizationError,
    TgammaKinematicTangent,
    evaluate_elastic_tgamma_kinematic_tangent,
    modal_basis_derivative,
)

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"


@dataclass(frozen=True)
class TgammaCollisionJvpResult:
    """Primal collision action and its exact fixed-support ``T_gamma`` column."""

    base: ind.IndependentCollisionAction
    branch_signature: str
    electron: FloatArray
    total: FloatArray
    modal_electron: FloatArray
    modal_total: FloatArray
    measure: FloatArray
    matrix: FloatArray
    pauli: FloatArray
    projection: FloatArray
    elastic: FloatArray
    pair: FloatArray
    electron_families: Mapping[str, FloatArray]
    electron_bath_energy_by_family: Mapping[str, float]
    neutrino_energy_transfer: float
    electron_bath_energy_transfer: float
    energy_tangent_by_component: Mapping[str, tuple[float, float]]
    first_law_tangent_residual: float
    charge_conjugation_residual: float
    mu_tau_residual: float
    minimum_support_margin: float
    minimum_lambda_margin: float
    base_reconstruction_residual: float
    component_sum_residual: float


def _finite(name: str, value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise D080TgammaLinearizationError(f"{name} contains NaN/Inf")
    return result


def _relative(left: ArrayLike, right: ArrayLike) -> float:
    return safe_relative(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )


def _signature_update(
    digest: "hashlib._Hash",
    label: str,
    value: ArrayLike,
) -> None:
    array = np.asarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())


def _electron_matrix_raw(
    target: str,
    category: str,
    batch: ind._KinematicBatch,
    electron_mass: float,
) -> tuple[FloatArray, FloatArray]:
    """Reproduce the comparator's unprojected matrix value and scale."""

    left, right = ind._electron_couplings(target)
    anti = ind._is_antineutrino(target)
    ks = batch.d12 * batch.d34
    kt = batch.d14 * batch.d23
    ku = batch.d13 * batch.d24
    interference_13 = electron_mass**2 * batch.d13
    interference_12 = electron_mass**2 * batch.d12
    if category == "elastic_minus":
        if anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * interference_13,
        )
    elif category == "elastic_plus":
        if not anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * interference_13,
        )
    elif category == "pair":
        if anti:
            left, right = right, left
        terms = (
            left * left * kt,
            right * right * ku,
            left * right * interference_12,
        )
    else:
        raise ValueError(f"unknown electron category {category!r}")
    common = (
        128.0 if category == "pair" else 64.0
    ) * ind.G_F_MEV_MINUS_2**2
    raw = np.where(
        batch.support,
        common * (terms[0] + terms[1] + terms[2]),
        0.0,
    )
    scale = np.where(
        batch.support,
        common * (np.abs(terms[0]) + np.abs(terms[1]) + np.abs(terms[2])),
        0.0,
    )
    return _finite("raw electron matrix", raw), _finite("electron matrix scale", scale)


def _elastic_matrix_value_and_tangent(
    *,
    target: str,
    category: str,
    tangent: TgammaKinematicTangent,
    electron_mass: float,
    config: ind.IndependentCollisionConfig,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return the comparator matrix and its piecewise smooth tangent."""

    batch = tangent.base
    left, right = ind._electron_couplings(target)
    anti = ind._is_antineutrino(target)
    ks = batch.d12 * batch.d34
    kt = batch.d14 * batch.d23
    d_ks = tangent.d_d12 * batch.d34 + batch.d12 * tangent.d_d34
    d_kt = tangent.d_d14 * batch.d23 + batch.d14 * tangent.d_d23
    d_interference_13 = electron_mass**2 * tangent.d_d13
    if category == "elastic_minus":
        if anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * electron_mass**2 * batch.d13,
        )
        d_terms = (
            left * left * d_ks,
            right * right * d_kt,
            -left * right * d_interference_13,
        )
    elif category == "elastic_plus":
        if not anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * electron_mass**2 * batch.d13,
        )
        d_terms = (
            left * left * d_ks,
            right * right * d_kt,
            -left * right * d_interference_13,
        )
    else:
        raise ValueError("elastic tangent requires an elastic category")

    common = 64.0 * ind.G_F_MEV_MINUS_2**2
    raw = np.where(
        batch.support,
        common * (terms[0] + terms[1] + terms[2]),
        0.0,
    )
    d_raw = np.where(
        batch.support,
        common * (d_terms[0] + d_terms[1] + d_terms[2]),
        0.0,
    )
    scale = np.where(
        batch.support,
        common * (np.abs(terms[0]) + np.abs(terms[1]) + np.abs(terms[2])),
        0.0,
    )
    tolerance = (
        config.matrix_roundoff_ulps
        * np.finfo(np.float64).eps
        * np.maximum(scale, np.finfo(np.float64).tiny)
    )
    if np.any(raw < -tolerance):
        raise D080TgammaLinearizationError(
            "materially negative matrix element in tangent path"
        )
    corrected = np.asarray(raw < 0.0, dtype=bool)
    value = np.where(corrected, 0.0, raw)
    derivative = np.where(corrected, 0.0, d_raw)
    comparator_value = ind._electron_matrix(
        target, category, batch, electron_mass, config
    )[0]
    if _relative(value, comparator_value) > 2.0e-14:
        raise D080TgammaLinearizationError(
            "matrix reconstruction diverges from frozen comparator"
        )
    return (
        _finite("electron matrix", comparator_value),
        _finite("electron matrix tangent", derivative),
        corrected,
    )


def _measure_and_tangent(
    *,
    tangent: TgammaKinematicTangent,
    p1: float,
    outer_weight: float,
    domain: NDArray[np.bool_],
) -> tuple[FloatArray, FloatArray]:
    batch = tangent.base
    constant = outer_weight / (256.0 * ind.PI**4 * p1)
    base_full = (
        constant
        * batch.quadrature_weight
        * np.square(batch.p2)
        * batch.phase_space
        / batch.e2
    )
    derivative_full = constant * (
        tangent.d_quadrature_weight
        * np.square(batch.p2)
        * batch.phase_space
        / batch.e2
        + batch.quadrature_weight
        * 2.0
        * batch.p2
        * tangent.d_p2
        * batch.phase_space
        / batch.e2
        + batch.quadrature_weight
        * np.square(batch.p2)
        * tangent.d_phase_space
        / batch.e2
        - batch.quadrature_weight
        * np.square(batch.p2)
        * batch.phase_space
        * tangent.d_e2
        / np.square(batch.e2)
    )
    base = np.asarray(base_full[domain], dtype=np.float64)
    frozen = ind._event_measure(batch, p1, outer_weight, domain)
    if _relative(base, frozen) > 2.0e-14:
        raise D080TgammaLinearizationError(
            "measure reconstruction diverges from frozen comparator"
        )
    return _finite("event measure", frozen), _finite(
        "event measure tangent", derivative_full[domain]
    )


def _spectral_logit_location_tangent(
    *,
    spectra: ind._SpectralLogits,
    species: str,
    y: FloatArray,
    d_y: FloatArray,
) -> FloatArray:
    index = ind.PAIR_INDEX[ind._species_flavour(species)]
    derivative_basis = modal_basis_derivative(spectra.grid, y)
    derivative = (derivative_basis @ spectra.coefficients[index]) * d_y
    return _finite("moving spectral-logit tangent", derivative)


def _moving_modal_product(
    rates: FloatArray,
    y: FloatArray,
    d_y: FloatArray,
    grid: ind.IndependentNoQkeGrid,
) -> FloatArray:
    result = np.zeros((rates.shape[0], grid.order), dtype=np.float64)
    for start in range(0, rates.shape[1], ind._EVENT_BLOCK):
        stop = min(start + ind._EVENT_BLOCK, rates.shape[1])
        basis_tangent = modal_basis_derivative(
            grid, y[start:stop]
        ) * d_y[start:stop, None]
        result += rates[:, start:stop] @ basis_tangent
    return _finite("moving modal projection tangent", result)


def electron_tgamma_branch_signature(
    *,
    grid: ind.IndependentNoQkeGrid,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> str:
    """Hash support/domain and matrix-clipping branches for one static state."""

    tcm = float(temperature_cm_mev)
    tg = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if min(tcm, tg) <= 0.0 or mass < 0.0:
        raise ValueError("invalid thermodynamic input")
    events = ind.independent_electron_events()
    elastic, pairs = events[:12], events[12:]
    electron_p2, electron_weights = ind._electron_half_line_rule(
        config.electron_radial_order, tg
    )
    neutrino_p2 = tcm * grid.nodes
    neutrino_weights = tcm * grid.weights
    digest = hashlib.sha256()
    digest.update(
        f"{grid.order}:{grid.y_max}:{tcm:.17e}:{mass:.17e}".encode()
    )

    for node_index, y1 in enumerate(grid.nodes):
        p1 = tcm * float(y1)
        elastic_batch = ind._two_body_kinematics(
            p1=p1,
            p2_nodes=electron_p2,
            p2_weights=electron_weights,
            mass2=mass,
            mass3=0.0,
            mass4=mass,
            config=config,
        )
        y3 = elastic_batch.p3_magnitude / tcm
        domain = (
            elastic_batch.support
            & (y3 > 0.0)
            & (y3 < grid.y_max)
        )
        _signature_update(digest, f"elastic-support-{node_index}", elastic_batch.support)
        _signature_update(digest, f"elastic-domain-{node_index}", domain)
        for event_index, event in enumerate(elastic):
            raw, _scale = _electron_matrix_raw(
                event.target, event.category, elastic_batch, mass
            )
            _signature_update(
                digest,
                f"elastic-matrix-{node_index}-{event_index}",
                raw < 0.0,
            )

        pair_batch = ind._two_body_kinematics(
            p1=p1,
            p2_nodes=neutrino_p2,
            p2_weights=neutrino_weights,
            mass2=0.0,
            mass3=mass,
            mass4=mass,
            config=config,
        )
        _signature_update(digest, f"pair-support-{node_index}", pair_batch.support)
        for event_index, event in enumerate(pairs):
            raw, _scale = _electron_matrix_raw(
                event.target, "pair", pair_batch, mass
            )
            _signature_update(
                digest,
                f"pair-matrix-{node_index}-{event_index}",
                raw < 0.0,
            )
    return digest.hexdigest()


def evaluate_tgamma_collision_action_jvp(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> TgammaCollisionJvpResult:
    """Differentiate the complete static collision action in ``T_gamma``."""

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    tcm = float(temperature_cm_mev)
    tg = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if min(tcm, tg) <= 0.0 or mass < 0.0:
        raise ValueError("invalid thermodynamic input")

    branch_signature = electron_tgamma_branch_signature(
        grid=grid,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        electron_mass_mev=mass,
    )
    spectra = ind._SpectralLogits(grid, ind._native_pair_logits(c))
    base = ind.evaluate_independent_collision_action(
        grid=grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        config=config,
        electron_mass_mev=mass,
    )
    events = ind.independent_electron_events()
    elastic_events, pair_events = events[:12], events[12:]
    basis = spectra.native_basis
    angular_size = (
        config.incoming_polar_order
        * config.final_polar_order
        * config.final_azimuth_order
    )

    component_names = ("measure", "matrix", "pauli", "projection")
    modal_components = {
        name: np.zeros((6, grid.order), dtype=np.float64)
        for name in component_names
    }
    modal_elastic = np.zeros((6, grid.order), dtype=np.float64)
    modal_pair = np.zeros((6, grid.order), dtype=np.float64)
    reconstructed_base_modal = np.zeros((6, grid.order), dtype=np.float64)
    family_modal = {
        ind._electron_family_key(event): np.zeros((6, grid.order), dtype=np.float64)
        for event in events
    }
    family_qem = {key: 0.0 for key in family_modal}
    energy_components = {
        name: [0.0, 0.0]
        for name in ("measure", "matrix", "pauli", "kinematic-weight")
    }
    base_qnu = 0.0
    base_qem = 0.0
    minimum_support_margin = float("inf")
    minimum_lambda_margin = float("inf")
    neutrino_p2 = tcm * grid.nodes
    neutrino_weights = tcm * grid.weights

    for node_index, y1 in enumerate(grid.nodes):
        p1 = tcm * float(y1)
        outer = tcm**3 * grid.weights[node_index] * y1**2 / ind.TWO_PI_SQUARED

        kinematic = evaluate_elastic_tgamma_kinematic_tangent(
            p1=p1,
            temperature_gamma_mev=tg,
            electron_mass_mev=mass,
            config=config,
        )
        minimum_support_margin = min(
            minimum_support_margin, kinematic.minimum_support_margin
        )
        minimum_lambda_margin = min(
            minimum_lambda_margin, kinematic.minimum_lambda_margin
        )
        batch = kinematic.base
        y3 = batch.p3_magnitude / tcm
        d_y3 = kinematic.d_p3_magnitude / tcm
        domain = batch.support & (y3 > 0.0) & (y3 < grid.y_max)
        mask = domain.ravel()
        y3_valid = y3[domain]
        d_y3_valid = d_y3[domain]
        measure_value, measure_tangent = _measure_and_tangent(
            tangent=kinematic,
            p1=p1,
            outer_weight=float(outer),
            domain=domain,
        )
        u2 = -batch.e2[domain] / tg
        u4 = -batch.e4[domain] / tg
        d_u2 = -kinematic.d_e2[domain] / tg + batch.e2[domain] / tg**2
        d_u4 = -kinematic.d_e4[domain] / tg + batch.e4[domain] / tg**2

        base_rates = np.zeros((len(elastic_events), batch.support.size), dtype=np.float64)
        derivative_rates = {
            name: np.zeros_like(base_rates)
            for name in ("measure", "matrix", "pauli")
        }

        for event_index, event in enumerate(elastic_events):
            u1 = float(spectra.native(event.target)[node_index])
            u3 = spectra.at(event.target, y3_valid)
            ind._strict_interpolated_logit(u3)
            d_u3 = _spectral_logit_location_tangent(
                spectra=spectra,
                species=event.target,
                y=y3_valid,
                d_y=d_y3_valid,
            )
            matrix_value, matrix_tangent, _corrected = (
                _elastic_matrix_value_and_tangent(
                    target=event.target,
                    category=event.category,
                    tangent=kinematic,
                    electron_mass=mass,
                    config=config,
                )
            )
            matrix_domain = matrix_value[domain]
            d_matrix_domain = matrix_tangent[domain]
            pauli_value = ind._stable_pauli_gain_minus_loss(u1, u2, u3, u4)
            d_pauli = pauli_gain_minus_loss_jvp(
                u1,
                u2,
                u3,
                u4,
                0.0,
                d_u2,
                d_u3,
                d_u4,
            )
            rate = measure_value * matrix_domain * pauli_value
            d_measure_rate = measure_tangent * matrix_domain * pauli_value
            d_matrix_rate = measure_value * d_matrix_domain * pauli_value
            d_pauli_rate = measure_value * matrix_domain * d_pauli
            if not all(
                np.all(np.isfinite(value))
                for value in (
                    rate,
                    d_measure_rate,
                    d_matrix_rate,
                    d_pauli_rate,
                )
            ):
                raise D080TgammaLinearizationError(
                    "nonfinite elastic event tangent"
                )
            base_rates[event_index, mask] = rate
            derivative_rates["measure"][event_index, mask] = d_measure_rate
            derivative_rates["matrix"][event_index, mask] = d_matrix_rate
            derivative_rates["pauli"][event_index, mask] = d_pauli_rate

            weight_nu = p1 - batch.e3[domain]
            weight_em = batch.e2[domain] - batch.e4[domain]
            d_weight_nu = -kinematic.d_e3[domain]
            d_weight_em = kinematic.d_e2[domain] - kinematic.d_e4[domain]
            base_qnu += float(np.sum(rate * weight_nu, dtype=np.float64))
            base_qem += float(np.sum(rate * weight_em, dtype=np.float64))
            for name, d_rate in (
                ("measure", d_measure_rate),
                ("matrix", d_matrix_rate),
                ("pauli", d_pauli_rate),
            ):
                energy_components[name][0] += float(
                    np.sum(d_rate * weight_nu, dtype=np.float64)
                )
                energy_components[name][1] += float(
                    np.sum(d_rate * weight_em, dtype=np.float64)
                )
            d_qnu_weight = float(np.sum(rate * d_weight_nu, dtype=np.float64))
            d_qem_weight = float(np.sum(rate * d_weight_em, dtype=np.float64))
            energy_components["kinematic-weight"][0] += d_qnu_weight
            energy_components["kinematic-weight"][1] += d_qem_weight
            family_qem[ind._electron_family_key(event)] += (
                float(np.sum(
                    (d_measure_rate + d_matrix_rate + d_pauli_rate)
                    * weight_em,
                    dtype=np.float64,
                ))
                + d_qem_weight
            )

        base_incoming = (
            np.sum(base_rates, axis=1, dtype=np.float64)[:, None]
            * basis[node_index]
        )
        base_outgoing = modal_product(base_rates[:, mask], y3_valid, grid)
        derivative_modes: dict[str, tuple[FloatArray, FloatArray]] = {}
        for name, d_rates in derivative_rates.items():
            incoming = (
                np.sum(d_rates, axis=1, dtype=np.float64)[:, None]
                * basis[node_index]
            )
            outgoing = modal_product(d_rates[:, mask], y3_valid, grid)
            derivative_modes[name] = (incoming, outgoing)
        moving_outgoing = _moving_modal_product(
            base_rates[:, mask], y3_valid, d_y3_valid, grid
        )

        for event_index, event in enumerate(elastic_events):
            species_index = ind.SPECIES_INDEX[event.target]
            base_contribution = (
                base_incoming[event_index] - base_outgoing[event_index]
            )
            reconstructed_base_modal[species_index] += base_contribution
            event_total = np.zeros(grid.order, dtype=np.float64)
            for name in ("measure", "matrix", "pauli"):
                incoming, outgoing = derivative_modes[name]
                contribution = incoming[event_index] - outgoing[event_index]
                modal_components[name][species_index] += contribution
                event_total += contribution
            projection_contribution = -moving_outgoing[event_index]
            modal_components["projection"][species_index] += projection_contribution
            event_total += projection_contribution
            modal_elastic[species_index] += event_total
            family_modal[ind._electron_family_key(event)][species_index] += event_total

        pair_batch = ind._two_body_kinematics(
            p1=p1,
            p2_nodes=neutrino_p2,
            p2_weights=neutrino_weights,
            mass2=0.0,
            mass3=mass,
            mass4=mass,
            config=config,
        )
        pair_domain = pair_batch.support
        pair_mask = pair_domain.ravel()
        pair_measure = ind._event_measure(
            pair_batch, p1, float(outer), pair_domain
        )
        pair_base_rates = np.zeros(
            (len(pair_events), pair_batch.support.size), dtype=np.float64
        )
        pair_derivative_rates = np.zeros_like(pair_base_rates)
        u3_pair = -pair_batch.e3[pair_domain] / tg
        u4_pair = -pair_batch.e4[pair_domain] / tg
        d_u3_pair = pair_batch.e3[pair_domain] / tg**2
        d_u4_pair = pair_batch.e4[pair_domain] / tg**2

        for event_index, event in enumerate(pair_events):
            partner = ind._cp_partner(event.target)
            u1 = float(spectra.native(event.target)[node_index])
            u2_pair = np.repeat(
                spectra.native(partner), angular_size
            )[pair_mask]
            matrix_value = ind._electron_matrix(
                event.target, "pair", pair_batch, mass, config
            )[0][pair_domain]
            pauli_value = ind._stable_pauli_gain_minus_loss(
                u1, u2_pair, u3_pair, u4_pair
            )
            d_pauli = pauli_gain_minus_loss_jvp(
                u1,
                u2_pair,
                u3_pair,
                u4_pair,
                0.0,
                0.0,
                d_u3_pair,
                d_u4_pair,
            )
            rate = pair_measure * matrix_value * pauli_value
            d_rate = pair_measure * matrix_value * d_pauli
            if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(d_rate)):
                raise D080TgammaLinearizationError("nonfinite pair-event tangent")
            pair_base_rates[event_index, pair_mask] = rate
            pair_derivative_rates[event_index, pair_mask] = d_rate
            weight_nu = p1 + pair_batch.p2[pair_domain]
            weight_em = -pair_batch.e3[pair_domain] - pair_batch.e4[pair_domain]
            base_qnu += float(np.sum(rate * weight_nu, dtype=np.float64))
            base_qem += float(np.sum(rate * weight_em, dtype=np.float64))
            energy_components["pauli"][0] += float(
                np.sum(d_rate * weight_nu, dtype=np.float64)
            )
            energy_components["pauli"][1] += float(
                np.sum(d_rate * weight_em, dtype=np.float64)
            )
            family_qem[ind._electron_family_key(event)] += float(
                np.sum(d_rate * weight_em, dtype=np.float64)
            )

        pair_base_incoming_1 = (
            np.sum(pair_base_rates, axis=1, dtype=np.float64)[:, None]
            * basis[node_index]
        )
        pair_base_incoming_2 = (
            pair_base_rates.reshape(
                len(pair_events), grid.order, angular_size
            ).sum(axis=2)
            @ basis
        )
        pair_incoming_1 = (
            np.sum(pair_derivative_rates, axis=1, dtype=np.float64)[:, None]
            * basis[node_index]
        )
        pair_incoming_2 = (
            pair_derivative_rates.reshape(
                len(pair_events), grid.order, angular_size
            ).sum(axis=2)
            @ basis
        )
        for event_index, event in enumerate(pair_events):
            family = ind._electron_family_key(event)
            partner = ind._cp_partner(event.target)
            for species, base_contribution, contribution in (
                (
                    event.target,
                    pair_base_incoming_1[event_index],
                    pair_incoming_1[event_index],
                ),
                (
                    partner,
                    pair_base_incoming_2[event_index],
                    pair_incoming_2[event_index],
                ),
            ):
                species_index = ind.SPECIES_INDEX[species]
                reconstructed_base_modal[species_index] += base_contribution
                modal_components["pauli"][species_index] += contribution
                modal_pair[species_index] += contribution
                family_modal[family][species_index] += contribution

    if not np.isfinite(minimum_support_margin) or not np.isfinite(
        minimum_lambda_margin
    ):
        raise D080TgammaLinearizationError("missing elastic support margins")

    native_components = {
        name: ind._native_action(grid, values, tcm)
        for name, values in modal_components.items()
    }
    modal_total = sum(
        (modal_components[name] for name in component_names),
        np.zeros((6, grid.order), dtype=np.float64),
    )
    total = ind._native_action(grid, modal_total, tcm)
    elastic = ind._native_action(grid, modal_elastic, tcm)
    pair = ind._native_action(grid, modal_pair, tcm)
    family_actions = {
        key: ind._native_action(grid, values, tcm)
        for key, values in family_modal.items()
    }
    reconstructed_base = ind._native_action(
        grid, reconstructed_base_modal, tcm
    )
    base_qnu_authority = float(
        base.diagnostics["event_neutrino_energy_transfer"]
    )
    base_qem_authority = float(base.electron_bath_energy_transfer)
    base_reconstruction_residual = max(
        _relative(reconstructed_base, base.electron),
        _relative(np.array([base_qnu]), np.array([base_qnu_authority])),
        _relative(np.array([base_qem]), np.array([base_qem_authority])),
    )
    component_sum = sum(
        (native_components[name] for name in component_names),
        np.zeros_like(total),
    )
    component_sum_residual = _relative(total, component_sum)
    neutrino_energy_transfer = float(
        sum(values[0] for values in energy_components.values())
    )
    electron_bath_energy_transfer = float(
        sum(values[1] for values in energy_components.values())
    )
    denominator = max(
        abs(neutrino_energy_transfer) + abs(electron_bath_energy_transfer),
        np.finfo(np.float64).tiny,
    )
    first_law_residual = abs(
        neutrino_energy_transfer + electron_bath_energy_transfer
    ) / denominator
    pair_total = 0.5 * np.stack(
        (
            total[0] + total[1],
            total[2] + total[3],
            total[4] + total[5],
        )
    )
    charge_conjugation_residual = float(
        max(_relative(total[2 * index], total[2 * index + 1]) for index in range(3))
    )
    mu_tau_residual = float(_relative(pair_total[1], pair_total[2]))

    arrays = (
        total,
        modal_total,
        elastic,
        pair,
        *(native_components[name] for name in component_names),
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise D080TgammaLinearizationError("nonfinite collision-column output")
    scalars = (
        neutrino_energy_transfer,
        electron_bath_energy_transfer,
        first_law_residual,
        charge_conjugation_residual,
        mu_tau_residual,
        minimum_support_margin,
        minimum_lambda_margin,
        base_reconstruction_residual,
        component_sum_residual,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise D080TgammaLinearizationError("nonfinite collision-column diagnostic")

    return TgammaCollisionJvpResult(
        base=base,
        branch_signature=branch_signature,
        electron=np.asarray(total, dtype=np.float64),
        total=np.asarray(total, dtype=np.float64),
        modal_electron=np.asarray(modal_total, dtype=np.float64),
        modal_total=np.asarray(modal_total, dtype=np.float64),
        measure=np.asarray(native_components["measure"], dtype=np.float64),
        matrix=np.asarray(native_components["matrix"], dtype=np.float64),
        pauli=np.asarray(native_components["pauli"], dtype=np.float64),
        projection=np.asarray(native_components["projection"], dtype=np.float64),
        elastic=np.asarray(elastic, dtype=np.float64),
        pair=np.asarray(pair, dtype=np.float64),
        electron_families=family_actions,
        electron_bath_energy_by_family={
            key: float(value) for key, value in family_qem.items()
        },
        neutrino_energy_transfer=neutrino_energy_transfer,
        electron_bath_energy_transfer=electron_bath_energy_transfer,
        energy_tangent_by_component={
            key: (float(value[0]), float(value[1]))
            for key, value in energy_components.items()
        },
        first_law_tangent_residual=float(first_law_residual),
        charge_conjugation_residual=charge_conjugation_residual,
        mu_tau_residual=mu_tau_residual,
        minimum_support_margin=float(minimum_support_margin),
        minimum_lambda_margin=float(minimum_lambda_margin),
        base_reconstruction_residual=float(base_reconstruction_residual),
        component_sum_residual=float(component_sum_residual),
    )
