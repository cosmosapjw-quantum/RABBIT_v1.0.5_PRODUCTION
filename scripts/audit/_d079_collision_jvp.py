"""Exact c-only JVP of the frozen independent collision action.

Kinematics, matrix elements, support masks, quadrature, interpolation, and
Galerkin maps are reused byte-for-byte from the private comparator. Only each
Pauli gain-minus-loss factor is differentiated. Temperatures are fixed in this
module; no trajectory or full-state Jacobian is implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit._d079_tangent_primitives import (
    D079LinearizationError,
    TangentSpectralLogits,
    matrix,
    modal_product,
    pauli_gain_minus_loss_jvp,
    safe_relative,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CollisionJvpResult:
    """Base collision action plus its c-directional derivative."""

    base: ind.IndependentCollisionAction
    direction_cloglog: FloatArray
    direction_logit: FloatArray
    electron: FloatArray
    self_interaction: FloatArray
    total: FloatArray
    modal_electron: FloatArray
    modal_self_interaction: FloatArray
    modal_total: FloatArray
    electron_bath_energy_transfer: float
    neutrino_energy_transfer: float
    self_event_energy_residual: float
    self_number_moment: float
    self_energy_moment: float
    first_law_tangent_residual: float
    charge_conjugation_residual: float
    mu_tau_residual: float


def _assemble_self_jvp(
    grid: ind.IndependentNoQkeGrid,
    spectra: ind._SpectralLogits,
    tangent: TangentSpectralLogits,
    temperature: float,
    config: ind.IndependentCollisionConfig,
) -> tuple[FloatArray, float]:
    events = ind.independent_self_events()
    modal = np.zeros((6, grid.order), dtype=np.float64)
    energy_residual = 0.0
    p2_nodes = temperature * grid.nodes
    p2_weights = temperature * grid.weights
    basis = spectra.native_basis
    angular_size = config.incoming_polar_order * config.final_polar_order * 4

    for node_index, y1 in enumerate(grid.nodes):
        p1 = temperature * float(y1)
        outer = temperature**3 * grid.weights[node_index] * y1**2 / ind.TWO_PI_SQUARED
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=p2_nodes, p2_weights=p2_weights,
            mass2=0.0, mass3=0.0, mass4=0.0, config=config,
        )
        y3 = batch.p3_magnitude / temperature
        y4 = batch.p4_magnitude / temperature
        domain = (
            batch.support
            & (y3 > 0.0) & (y3 < grid.y_max)
            & (y4 > 0.0) & (y4 < grid.y_max)
        )
        mask = domain.ravel()
        measure = ind._event_measure(batch, p1, outer, domain)
        y3v, y4v = y3[domain], y4[domain]
        drates = np.zeros((len(events), batch.support.size), dtype=np.float64)
        matrices: dict[tuple[str, float], FloatArray] = {}

        for event_index, event in enumerate(events):
            s1, s2, s3, s4 = event.legs
            u1 = float(spectra.native(s1)[node_index])
            u2 = np.repeat(spectra.native(s2), angular_size)[mask]
            u3, u4 = spectra.at(s3, y3v), spectra.at(s4, y4v)
            ind._strict_interpolated_logit(u3)
            ind._strict_interpolated_logit(u4)
            du1 = float(tangent.native(s1)[node_index])
            du2 = np.repeat(tangent.native(s2), angular_size)[mask]
            du3, du4 = tangent.at(s3, y3v), tangent.at(s4, y4v)
            key = (event.kernel, event.coefficient)
            if key not in matrices:
                matrices[key] = ind._self_matrix(event, batch, config)[0]
            drate = measure * matrices[key][domain] * pauli_gain_minus_loss_jvp(
                u1, u2, u3, u4, du1, du2, du3, du4
            )
            if not np.all(np.isfinite(drate)):
                raise D079LinearizationError("nonfinite self-event tangent rate")
            drates[event_index, mask] = drate
            energy_residual += float(np.sum(
                drate * (p1 + batch.p2[domain] - batch.e3[domain] - batch.e4[domain]),
                dtype=np.float64,
            ))

        sums = np.sum(drates, axis=1, dtype=np.float64)
        leg_modes = (
            sums[:, None] * basis[node_index],
            drates.reshape(len(events), grid.order, angular_size).sum(axis=2) @ basis,
            modal_product(drates[:, mask], y3v, grid),
            modal_product(drates[:, mask], y4v, grid),
        )
        for event_index, event in enumerate(events):
            for sign, species, values in zip(
                (1.0, 1.0, -1.0, -1.0), event.legs, leg_modes
            ):
                modal[ind.SPECIES_INDEX[species]] += sign * values[event_index]

    if not np.all(np.isfinite(modal)) or not np.isfinite(energy_residual):
        raise D079LinearizationError("nonfinite assembled self tangent")
    return modal, float(energy_residual)


def _assemble_electron_jvp(
    grid: ind.IndependentNoQkeGrid,
    spectra: ind._SpectralLogits,
    tangent: TangentSpectralLogits,
    temperature_cm: float,
    temperature_gamma: float,
    electron_mass: float,
    config: ind.IndependentCollisionConfig,
) -> tuple[FloatArray, float, float]:
    events = ind.independent_electron_events()
    elastic, pairs = events[:12], events[12:]
    basis = spectra.native_basis
    modal = np.zeros((6, grid.order), dtype=np.float64)
    dqnu = dqem = 0.0
    electron_p2, electron_weights = ind._electron_half_line_rule(
        config.electron_radial_order, temperature_gamma
    )
    neutrino_p2 = temperature_cm * grid.nodes
    neutrino_weights = temperature_cm * grid.weights
    angular_size = config.incoming_polar_order * config.final_polar_order * 4

    for node_index, y1 in enumerate(grid.nodes):
        p1 = temperature_cm * float(y1)
        outer = temperature_cm**3 * grid.weights[node_index] * y1**2 / ind.TWO_PI_SQUARED

        # nu + e^+- -> nu + e^+-. Bath-leg c-tangents are zero.
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=electron_p2, p2_weights=electron_weights,
            mass2=electron_mass, mass3=0.0, mass4=electron_mass, config=config,
        )
        y3 = batch.p3_magnitude / temperature_cm
        domain = batch.support & (y3 > 0.0) & (y3 < grid.y_max)
        mask = domain.ravel()
        measure = ind._event_measure(batch, p1, outer, domain)
        y3v = y3[domain]
        drates = np.zeros((len(elastic), batch.support.size), dtype=np.float64)
        u2 = -batch.e2[domain] / temperature_gamma
        u4 = -batch.e4[domain] / temperature_gamma
        zero2, zero4 = np.zeros_like(u2), np.zeros_like(u4)

        for event_index, event in enumerate(elastic):
            u1 = float(spectra.native(event.target)[node_index])
            u3 = spectra.at(event.target, y3v)
            ind._strict_interpolated_logit(u3)
            du1 = float(tangent.native(event.target)[node_index])
            du3 = tangent.at(event.target, y3v)
            matrix_element = ind._electron_matrix(
                event.target, event.category, batch, electron_mass, config
            )[0]
            drate = measure * matrix_element[domain] * pauli_gain_minus_loss_jvp(
                u1, u2, u3, u4, du1, zero2, du3, zero4
            )
            if not np.all(np.isfinite(drate)):
                raise D079LinearizationError("nonfinite elastic tangent rate")
            drates[event_index, mask] = drate
            dqnu += float(np.sum(drate * (p1 - batch.e3[domain]), dtype=np.float64))
            dqem += float(np.sum(
                drate * (batch.e2[domain] - batch.e4[domain]), dtype=np.float64
            ))

        incoming = np.sum(drates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
        outgoing = modal_product(drates[:, mask], y3v, grid)
        for event_index, event in enumerate(elastic):
            modal[ind.SPECIES_INDEX[event.target]] += incoming[event_index] - outgoing[event_index]

        # nu + antinu -> e- + e+. Only incoming neutrino legs vary.
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=neutrino_p2, p2_weights=neutrino_weights,
            mass2=0.0, mass3=electron_mass, mass4=electron_mass, config=config,
        )
        domain = batch.support
        mask = domain.ravel()
        measure = ind._event_measure(batch, p1, outer, domain)
        drates = np.zeros((len(pairs), batch.support.size), dtype=np.float64)
        u3 = -batch.e3[domain] / temperature_gamma
        u4 = -batch.e4[domain] / temperature_gamma
        zero3, zero4 = np.zeros_like(u3), np.zeros_like(u4)

        for event_index, event in enumerate(pairs):
            partner = ind._cp_partner(event.target)
            u1 = float(spectra.native(event.target)[node_index])
            u2 = np.repeat(spectra.native(partner), angular_size)[mask]
            du1 = float(tangent.native(event.target)[node_index])
            du2 = np.repeat(tangent.native(partner), angular_size)[mask]
            matrix_element = ind._electron_matrix(
                event.target, "pair", batch, electron_mass, config
            )[0]
            drate = measure * matrix_element[domain] * pauli_gain_minus_loss_jvp(
                u1, u2, u3, u4, du1, du2, zero3, zero4
            )
            if not np.all(np.isfinite(drate)):
                raise D079LinearizationError("nonfinite pair tangent rate")
            drates[event_index, mask] = drate
            dqnu += float(np.sum(
                drate * (p1 + batch.p2[domain]), dtype=np.float64
            ))
            dqem += float(np.sum(
                drate * (-batch.e3[domain] - batch.e4[domain]), dtype=np.float64
            ))

        incoming1 = np.sum(drates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
        incoming2 = drates.reshape(len(pairs), grid.order, angular_size).sum(axis=2) @ basis
        for event_index, event in enumerate(pairs):
            modal[ind.SPECIES_INDEX[event.target]] += incoming1[event_index]
            modal[ind.SPECIES_INDEX[ind._cp_partner(event.target)]] += incoming2[event_index]

    if not np.all(np.isfinite(modal)) or not np.isfinite(dqnu) or not np.isfinite(dqem):
        raise D079LinearizationError("nonfinite assembled electron tangent")
    return modal, float(dqnu), float(dqem)


def evaluate_collision_action_jvp(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    direction_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
) -> CollisionJvpResult:
    """Differentiate the full static collision action in one c-direction."""

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    v = matrix("direction_cloglog", direction_cloglog, c.shape)
    tcm, tg, mass = map(float, (temperature_cm_mev, temperature_gamma_mev, electron_mass_mev))
    if not np.isfinite(tcm) or tcm <= 0.0 or not np.isfinite(tg) or tg <= 0.0:
        raise ValueError("temperatures must be finite and positive")
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("electron mass must be finite and nonnegative")

    spectra = ind._SpectralLogits(grid, ind._native_pair_logits(c))
    tangent = TangentSpectralLogits(grid, c, v)
    base = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=c,
        temperature_cm_mev=tcm, temperature_gamma_mev=tg,
        config=config, electron_mass_mev=mass,
    )
    self_modal, self_energy_residual = _assemble_self_jvp(
        grid, spectra, tangent, tcm, config
    )
    electron_modal, dqnu, dqem = _assemble_electron_jvp(
        grid, spectra, tangent, tcm, tg, mass, config
    )
    self_action = ind._native_action(grid, self_modal, tcm)
    electron_action = ind._native_action(grid, electron_modal, tcm)
    total_modal = self_modal + electron_modal
    total = self_action + electron_action
    if not all(np.all(np.isfinite(x)) for x in (
        self_action, electron_action, total, self_modal, electron_modal, total_modal
    )):
        raise D079LinearizationError("nonfinite collision JVP output")

    moments = ind.independent_action_moments(
        grid=grid, action=self_action, temperature_cm_mev=tcm
    )
    denominator = max(abs(dqnu) + abs(dqem), np.finfo(np.float64).tiny)
    pair_total = 0.5 * np.stack((
        total[0] + total[1], total[2] + total[3], total[4] + total[5]
    ))
    return CollisionJvpResult(
        base=base,
        direction_cloglog=v,
        direction_logit=tangent.native_values.copy(),
        electron=np.asarray(electron_action, dtype=np.float64),
        self_interaction=np.asarray(self_action, dtype=np.float64),
        total=np.asarray(total, dtype=np.float64),
        modal_electron=np.asarray(electron_modal, dtype=np.float64),
        modal_self_interaction=np.asarray(self_modal, dtype=np.float64),
        modal_total=np.asarray(total_modal, dtype=np.float64),
        electron_bath_energy_transfer=float(dqem),
        neutrino_energy_transfer=float(dqnu),
        self_event_energy_residual=float(self_energy_residual),
        self_number_moment=float(moments.signed_number_rate),
        self_energy_moment=float(moments.signed_energy_rate),
        first_law_tangent_residual=float(abs(dqnu + dqem) / denominator),
        charge_conjugation_residual=float(max(
            safe_relative(total[2*i], total[2*i+1]) for i in range(3)
        )),
        mu_tau_residual=float(safe_relative(pair_total[1], pair_total[2])),
    )
