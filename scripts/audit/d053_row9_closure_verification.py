"""BD622 D-053 — row-9 closure verification (E1, E3, E4, E5).

Thresholds frozen in
``docs/audit/BD622_D053_row9_closure_contract_2026-07-28.md``.

Usage: PYTHONPATH=src python3 scripts/audit/d053_row9_closure_verification.py [--out PATH]
"""

from __future__ import annotations

import json
import os
import sys

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_pin, "1")

import numpy as np

from rabbit.decoupling import _independent_noqke as ind

ROW9_EMU_MEMBER = ("nu_e", "antinu_e", "nu_mu", "antinu_mu")


def rel_linf(left, right):
    return ind._relative_max_difference(np.asarray(left), np.asarray(right))


def pair_stack(native):
    native = np.asarray(native)
    return 0.5 * np.stack(
        (native[0] + native[1], native[2] + native[3], native[4] + native[5])
    )


def member_modal(grid, spectra, temperature, config, legs):
    """External mirror of the _assemble_self loop for one pair-conversion event."""
    event = ind.IndependentSelfEvent(tuple(legs), "pair_conversion", "K_t", 16.0)
    modal = np.zeros((6, grid.order), dtype=np.float64)
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
        rates = np.zeros((1, batch.support.size), dtype=np.float64)
        rates[0, mask] = rate
        sums = np.sum(rates, axis=1, dtype=np.float64)
        leg_modes = (
            sums[:, None] * basis[node_index],
            rates.reshape(1, grid.order, angular_size).sum(axis=2) @ basis,
            ind._modal_product(rates[:, mask], y3_valid, grid),
            ind._modal_product(rates[:, mask], y4_valid, grid),
        )
        for sign, species, values in zip((1.0, 1.0, -1.0, -1.0), event.legs, leg_modes):
            modal[ind.SPECIES_INDEX[species]] += sign * values[0]
    return modal


def check_e1():
    events = ind.independent_self_events()
    conversions = [e for e in events if e.category == "pair_conversion"]
    expected = set()
    for x, y in (("e", "mu"), ("e", "tau"), ("mu", "tau")):
        expected.add((f"nu_{x}", f"antinu_{x}", f"nu_{y}", f"antinu_{y}"))
        expected.add((f"nu_{y}", f"antinu_{y}", f"nu_{x}", f"antinu_{x}"))
    from collections import defaultdict

    agg = defaultdict(float)
    for event in events:
        for species in set(event.legs):
            agg[(event.category, species)] += event.coefficient * event.legs.count(species)
    target = defaultdict(float)
    for row in ind.independent_self_reactions():
        target[(row.category, row.target)] += row.coefficient
    ok = (
        len(events) == 27
        and ind.independent_pair_row_fingerprint() == (2, 1, 6, 2, 2, 2, 4, 4, 4)
        and len(conversions) == 6
        and {e.legs for e in conversions} == expected
        and all(e.kernel == "K_t" and e.coefficient == 16.0 for e in conversions)
        and dict(agg) == dict(target)
    )
    return {"ok": bool(ok), "events": len(events), "conversions": len(conversions)}


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

    e1 = check_e1()

    temperature = 10.0
    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()
    tiny = float(np.finfo(float).tiny)

    emu_scales = (0.995, 0.995, 1.01)
    emu_state = ind.pair_logits_to_cloglog(
        np.stack([-grid.nodes / s for s in emu_scales])
    )
    emu_action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=emu_state,
        temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
        config=config,
    )
    pair_self = pair_stack(emu_action.self_interaction)
    r_emu_self = rel_linf(pair_self[0], pair_self[1])
    spectra = ind._SpectralLogits(grid, ind._native_pair_logits(emu_state))
    single = member_modal(grid, spectra, temperature, config, ROW9_EMU_MEMBER)
    single_native = ind._native_action(grid, single, temperature)
    single_pairs = pair_stack(single_native)
    r_single = rel_linf(single_pairs[0], single_pairs[1])
    e3 = {"ok": bool(r_emu_self <= 1e-10 and r_single > 1e-10),
          "self_emu_residual": float(r_emu_self),
          "single_member_residual": float(r_single)}

    d028_scales = (1.01, 0.995, 0.995)
    d028_state = ind.pair_logits_to_cloglog(
        np.stack([-grid.nodes / s for s in d028_scales])
    )
    action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=d028_state,
        temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
        config=config,
    )
    resid = float(action.diagnostics["mu_tau_residual"])
    e4 = {"ok": bool(1e-12 <= resid <= 1e-10), "mu_tau_residual": resid}

    moments = ind.independent_action_moments(
        grid=grid, action=action.self_interaction, temperature_cm_mev=temperature
    )
    number_ratio = abs(moments.signed_number_rate) / max(moments.absolute_number_rate, tiny)
    energy_ratio = abs(moments.signed_energy_rate) / max(moments.absolute_energy_rate, tiny)
    first_law = float(action.diagnostics["first_law_residual"])
    cp = float(action.diagnostics["charge_conjugation_residual"])
    entropy = float(action.diagnostics["entropy_production"])
    common = ind.pair_logits_to_cloglog(np.stack([-grid.nodes for _ in range(3)]))
    null_action = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=common,
        temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
        config=config,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid, pair_cloglog=common,
        temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
    )
    null = ind.independent_action_moments(
        grid=grid, action=null_action.total, temperature_cm_mev=temperature
    )
    h_number = null.absolute_number_rate / (thermo.hubble_mev * thermo.number_density_neutrino)
    h_energy = null.absolute_energy_rate / (thermo.hubble_mev * thermo.energy_density_neutrino)
    e5 = {"ok": bool(number_ratio <= 1e-12 and energy_ratio <= 1e-12
                     and first_law <= 1e-8 and cp <= 1e-10 and entropy >= -1e-24
                     and h_number <= 1e-10 and h_energy <= 1e-10),
          "number_ratio": float(number_ratio), "energy_ratio": float(energy_ratio),
          "first_law_residual": first_law, "charge_conjugation_residual": cp,
          "entropy_production": entropy,
          "h_normalized_number": float(h_number), "h_normalized_energy": float(h_energy)}

    checks = {"E1_catalogue": e1, "E3_emu_discriminator": e3,
              "E4_mu_tau_regression": e4, "E5_invariants": e5}
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    report = {"contract": "BD622_D053_row9_closure_contract_2026-07-28",
              "checks": checks, "verdict": verdict}
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
