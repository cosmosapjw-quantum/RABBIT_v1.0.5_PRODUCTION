"""BD622 D-047 — OWNER-B target replay: row-6 orientation-chart closure vs the
D-028 native mu-tau covariance residual.

Read-only with respect to ``_independent_noqke.py``: the module is imported
unmodified and the reverse ordered member is evaluated by an external mirror
of the ``_assemble_self`` event loop using the module's own helpers. The
closure construction is the D-046-validated two-ordered-member Reynolds
one-half average. Thresholds are frozen in
``docs/audit/BD622_D047_ownerb_target_replay_contract_2026-07-27.md`` before
execution. This script changes no gate, no frozen byte, and does not reopen
the recorded D-028 FAIL.

Usage: python3 scripts/audit/d047_ownerb_r6_target_replay.py [--out PATH]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from rabbit.decoupling import _independent_noqke as ind

M_PLUS = ("nu_mu", "antinu_mu", "nu_tau", "antinu_tau")
M_MINUS = ("nu_tau", "antinu_tau", "nu_mu", "antinu_mu")
P_SPECIES = (0, 1, 4, 5, 2, 3)  # mu <-> tau on the six-species axis
CAP = 1e-10
TIGHT = 1e-13


def rel_linf(left: np.ndarray, right: np.ndarray) -> float:
    return ind._relative_max_difference(np.asarray(left), np.asarray(right))


def member_modal(grid, spectra, temperature, config, legs):
    """External mirror of the _assemble_self loop for one pair-conversion event."""
    event = ind.IndependentSelfEvent(tuple(legs), "pair_conversion", "K_t", 32.0)
    modal = np.zeros((6, grid.order), dtype=np.float64)
    entropy_production = 0.0
    p2_nodes, p2_weights = temperature * grid.nodes, temperature * grid.weights
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
        mask, base = domain.ravel(), ind._event_measure(batch, p1, outer, domain)
        y3_valid, y4_valid = y3[domain], y4[domain]
        s1, s2, s3, s4 = event.legs
        u1 = float(spectra.native(s1)[node_index])
        u2 = np.repeat(spectra.native(s2), angular_size)[mask]
        u3, u4 = spectra.at(s3, y3_valid), spectra.at(s4, y4_valid)
        ind._strict_interpolated_logit(u3)
        ind._strict_interpolated_logit(u4)
        matrix, _count, _correction = ind._self_matrix(event, batch, config)
        rate = base * matrix[domain] * ind._stable_pauli_gain_minus_loss(u1, u2, u3, u4)
        if not np.all(np.isfinite(rate)):
            raise ind.IndependentNoQkeError("nonfinite replay rate")
        rates = np.zeros((1, batch.support.size), dtype=np.float64)
        rates[0, mask] = rate
        entropy_production += float(np.sum(rate * (u3 + u4 - u1 - u2), dtype=np.float64))
        sums = np.sum(rates, axis=1, dtype=np.float64)
        leg_modes = (
            sums[:, None] * basis[node_index],
            rates.reshape(1, grid.order, angular_size).sum(axis=2) @ basis,
            ind._modal_product(rates[:, mask], y3_valid, grid),
            ind._modal_product(rates[:, mask], y4_valid, grid),
        )
        for sign, species, values in zip((1.0, 1.0, -1.0, -1.0), event.legs, leg_modes):
            modal[ind.SPECIES_INDEX[species]] += sign * values[0]
    return modal, entropy_production


def permute_rows(array):
    return np.asarray(array)[list(P_SPECIES)]


def pair_stack(native):
    return 0.5 * np.stack(
        (native[0] + native[1], native[2] + native[3], native[4] + native[5])
    )


def pair_residual(native):
    pairs = pair_stack(native)
    return rel_linf(pairs[1], pairs[2])


def d028_state(grid):
    scales = (1.01, 0.995, 0.995)
    return ind.pair_logits_to_cloglog(np.stack([-grid.nodes / s for s in scales]))


def asym_state(grid):
    y = grid.nodes
    logits = np.stack(
        (
            -y + 0.08 * y * np.exp(-y / 3.0),
            -y - 0.05 * y**2 / (1.0 + y**2) * np.exp(-y / 5.0),
            -y + 0.03 * np.sin(y / 2.0) * np.exp(-y / 6.0),
        )
    )
    return ind.pair_logits_to_cloglog(logits)


def permute_state(pair_cloglog):
    return np.asarray(pair_cloglog)[[0, 2, 1]]


def main(argv):
    out_path = None
    args = list(argv[1:])
    while args:
        arg = args.pop(0)
        if arg == "--out":
            if not args:
                print("ERROR --out requires a path")
                return 20
            out_path = args.pop(0)
        else:
            print(f"ERROR unknown argument {arg!r}")
            return 20
    temperature = 10.0
    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()
    state = d028_state(grid)

    action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=state,
        temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
        config=config,
    )
    spectra = ind._SpectralLogits(grid, ind._native_pair_logits(state))
    modal_p, entropy_p = member_modal(grid, spectra, temperature, config, M_PLUS)
    modal_m, entropy_m = member_modal(grid, spectra, temperature, config, M_MINUS)

    native_p = ind._native_action(grid, modal_p, temperature)
    live_row6 = np.asarray(action.self_rows[6])
    t_b = rel_linf(native_p, live_row6)
    bitwise_b = bool(np.array_equal(native_p, live_row6))

    swap_gap = np.abs(modal_m - permute_rows(modal_p))
    scale_c = max(float(np.max(np.abs(modal_m))), float(np.max(np.abs(modal_p))),
                  float(np.finfo(float).tiny))
    t_c = float(np.max(swap_gap)) / scale_c
    bitwise_c = bool(np.array_equal(modal_m, permute_rows(modal_p)))

    modal_closed = 0.5 * (modal_p + modal_m)
    native_closed_row6 = ind._native_action(grid, modal_closed, temperature)
    t_d = pair_residual(native_closed_row6)

    modal_total = np.asarray(action.modal_total)
    modal_total_closed = modal_total - modal_p + modal_closed
    native_total_closed = ind._native_action(grid, modal_total_closed, temperature)
    t_e = pair_residual(native_total_closed)

    modal_total_norow6 = modal_total - modal_p
    t_f = pair_residual(ind._native_action(grid, modal_total_norow6, temperature))

    modal_self_closed = np.asarray(action.modal_self_interaction) - modal_p + modal_closed
    native_self_closed = ind._native_action(grid, modal_self_closed, temperature)
    moments = ind.independent_action_moments(
        grid=grid, action=native_self_closed, temperature_cm_mev=temperature
    )
    number_ratio = abs(moments.signed_number_rate) / max(moments.absolute_number_rate,
                                                         np.finfo(float).tiny)
    energy_ratio = abs(moments.signed_energy_rate) / max(moments.absolute_energy_rate,
                                                         np.finfo(float).tiny)

    t_h = pair_residual(native_p)

    # Gated identity on a genuinely mu != tau state (W6 S-split logit profiles,
    # evaluated on this GL48-Y24 cell at 10 MeV as an informative adaptation).
    astate = asym_state(grid)
    aspectra = ind._SpectralLogits(grid, ind._native_pair_logits(astate))
    pspectra = ind._SpectralLogits(grid, ind._native_pair_logits(permute_state(astate)))
    amodal_m, aentropy_m = member_modal(grid, aspectra, temperature, config, M_MINUS)
    amodal_p_on_pf, _ = member_modal(grid, pspectra, temperature, config, M_PLUS)
    agap = np.abs(amodal_m - permute_rows(amodal_p_on_pf))
    ascale = max(float(np.max(np.abs(amodal_m))),
                 float(np.max(np.abs(amodal_p_on_pf))), float(np.finfo(float).tiny))
    t_c2 = float(np.max(agap)) / ascale
    bitwise_c2 = bool(np.array_equal(amodal_m, permute_rows(amodal_p_on_pf)))
    amodal_p, aentropy_p = member_modal(grid, aspectra, temperature, config, M_PLUS)
    aclosed = 0.5 * (amodal_p + amodal_m)
    anative_closed = ind._native_action(grid, aclosed, temperature)
    amoments = ind.independent_action_moments(
        grid=grid, action=anative_closed, temperature_cm_mev=temperature
    )

    resid_repro = float(action.diagnostics["mu_tau_residual"])
    checks = {
        "T-A_reproduction": {"value": resid_repro,
                             "ok": bool(1e-10 < resid_repro < 1e-8)},
        "T-B_parity": {"value": t_b, "bitwise": bitwise_b, "ok": bool(t_b <= TIGHT)},
        "T-C_identity_d028_state": {"value": t_c, "bitwise": bitwise_c,
                                    "ok": bool(t_c <= TIGHT)},
        "T-C2_identity_asym_state": {"value": t_c2, "bitwise": bitwise_c2,
                                     "ok": bool(t_c2 <= TIGHT)},
        "T-D_closed_row6_antisym": {"value": t_d, "ok": bool(t_d <= TIGHT)},
        "T-E_closed_total_residual": {"value": t_e, "ok": bool(t_e <= CAP)},
        "T-F_no_row6_residual": {"value": t_f, "ok": bool(t_f <= CAP)},
        "T-G_closed_self_conservation": {"number_ratio": float(number_ratio),
                                         "energy_ratio": float(energy_ratio),
                                         "ok": bool(number_ratio <= 1e-12
                                                    and energy_ratio <= 1e-12)},
        "T-H_negative_control": {"value": t_h, "ok": bool(t_h > CAP)},
    }
    informative = {
        "d028_entropy_production_member_plus": entropy_p,
        "d028_entropy_production_member_minus": entropy_m,
        "asym_entropy_production_member_plus": aentropy_p,
        "asym_entropy_production_member_minus": aentropy_m,
        "asym_closed_number_ratio": float(abs(amoments.signed_number_rate)
                                          / max(amoments.absolute_number_rate,
                                                np.finfo(float).tiny)),
        "asym_closed_energy_ratio": float(abs(amoments.signed_energy_rate)
                                          / max(amoments.absolute_energy_rate,
                                                np.finfo(float).tiny)),
        "recorded_d028_residual": 4.666064056497196e-10,
        "w5_replay_residual": 4.784280e-10,
        "charge_conjugation_residual": float(
            action.diagnostics["charge_conjugation_residual"]),
        "whole_reaction_domain_rejections": float(
            action.whole_reaction_domain_rejections),
    }
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report = {"contract": "BD622_D047_ownerb_target_replay_contract_2026-07-27",
              "checks": checks, "informative": informative, "verdict": verdict}
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
