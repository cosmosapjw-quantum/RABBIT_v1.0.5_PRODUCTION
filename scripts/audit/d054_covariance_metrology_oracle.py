"""BD622 D-054 — covariance-metrology oracle (M1..M5).

Thresholds and the B_native formula are frozen in
``docs/audit/BD622_D054_covariance_metrology_contract_2026-07-28.md``.

Usage: PYTHONPATH=src python3 scripts/audit/d054_covariance_metrology_oracle.py [--out PATH]
"""

from __future__ import annotations

import json
import os
import sys

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import hashlib

import numpy as np

from rabbit.decoupling import _independent_noqke as ind

MODULE_SHA_PREFIX = "760a7c04"
TINY = float(np.finfo(float).tiny)
EPS = float(np.finfo(float).eps)
FAMILY = {"S-A": (1.01, 0.995, 0.995), "S-B": (1.0, 1.0, 1.0),
          "S-C": (1.02, 0.99, 0.99)}


def rel_linf(left, right):
    return ind._relative_max_difference(np.asarray(left), np.asarray(right))


def pair_rows(matrix):
    matrix = np.asarray(matrix)
    return (0.5 * (matrix[2] + matrix[3]), 0.5 * (matrix[4] + matrix[5]))


def abs_modal_product(rates_abs, y, grid):
    basis_abs = np.abs(grid.modal_basis(y))
    out = np.zeros((rates_abs.shape[0], grid.order), dtype=np.float64)
    for start in range(0, rates_abs.shape[1], 4096):
        stop = start + 4096
        out += rates_abs[:, start:stop] @ basis_abs[start:stop]
    return out


def self_shadow(grid, spectra, temperature, config):
    """Signed + absolute-value shadow of the self assembly, with addend count."""
    events = ind.independent_self_events()
    signed = np.zeros((6, grid.order), dtype=np.float64)
    shadow = np.zeros((6, grid.order), dtype=np.float64)
    addends = 0
    p2_nodes, p2_weights = temperature * grid.nodes, temperature * grid.weights
    basis = spectra.native_basis
    basis_abs = np.abs(basis)
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
        valid = int(np.count_nonzero(mask))
        addends += valid * len(events)
        rates = np.zeros((len(events), batch.support.size), dtype=np.float64)
        matrix_cache = {}
        for event_index, event in enumerate(events):
            s1, s2, s3, s4 = event.legs
            u1 = float(spectra.native(s1)[node_index])
            u2 = np.repeat(spectra.native(s2), angular_size)[mask]
            u3, u4 = spectra.at(s3, y3_valid), spectra.at(s4, y4_valid)
            ind._strict_interpolated_logit(u3)
            ind._strict_interpolated_logit(u4)
            key = (event.kernel, event.coefficient)
            matrix_cache.setdefault(key, ind._self_matrix(event, batch, config))
            matrix, _count, _corr = matrix_cache[key]
            rate = base * matrix[domain] * ind._stable_pauli_gain_minus_loss(u1, u2, u3, u4)
            rates[event_index, mask] = rate
        rates_abs = np.abs(rates)
        sums = np.sum(rates, axis=1, dtype=np.float64)
        sums_abs = np.sum(rates_abs, axis=1, dtype=np.float64)
        leg_modes = (
            sums[:, None] * basis[node_index],
            rates.reshape(len(events), grid.order, angular_size).sum(axis=2) @ basis,
            ind._modal_product(rates[:, mask], y3_valid, grid),
            ind._modal_product(rates[:, mask], y4_valid, grid),
        )
        leg_modes_abs = (
            sums_abs[:, None] * basis_abs[node_index],
            np.abs(rates_abs.reshape(len(events), grid.order, angular_size)).sum(axis=2) @ basis_abs,
            abs_modal_product(rates_abs[:, mask], y3_valid, grid),
            abs_modal_product(rates_abs[:, mask], y4_valid, grid),
        )
        for event_index, event in enumerate(events):
            for sign, species, values, values_abs in zip(
                (1.0, 1.0, -1.0, -1.0), event.legs, leg_modes, leg_modes_abs
            ):
                index = ind.SPECIES_INDEX[species]
                signed[index] += sign * values[event_index]
                shadow[index] += values_abs[event_index]
    addends += grid.order + len(events)
    return signed, shadow, addends


def native_map_terms(grid, modal_total, temperature):
    basis = grid.modal_basis(grid.nodes)
    prefactor = temperature**3 / ind.TWO_PI_SQUARED
    scale = prefactor * np.square(grid.nodes)
    a_cond = float(np.max(np.sum(np.abs(basis), axis=1) / scale))
    abs_map = (np.abs(np.asarray(modal_total)) @ np.abs(basis).T) / scale[None, :]
    gamma48 = 48 * EPS / (1.0 - 48 * EPS)
    b_map = float(gamma48 * np.max(abs_map))
    basis_ld = np.asarray(
        np.polynomial.legendre.legvander(
            (2.0 * grid.nodes.astype(np.longdouble) / np.longdouble(grid.y_max)) - 1.0,
            grid.order - 1,
        ),
        dtype=np.longdouble,
    ) * np.sqrt((2.0 * np.arange(grid.order, dtype=np.longdouble) + 1.0) / np.longdouble(grid.y_max))
    delta_basis = float(np.max(np.abs(basis.astype(np.longdouble) - basis_ld)
                               / np.maximum(np.abs(basis_ld), np.longdouble(TINY))))
    native_ld = (np.asarray(modal_total, dtype=np.longdouble) @ basis_ld.T) / (
        np.longdouble(prefactor) * np.square(grid.nodes.astype(np.longdouble))
    )
    native64 = (np.asarray(modal_total) @ basis.T) / (prefactor * np.square(grid.nodes)[None, :])
    return a_cond, b_map, delta_basis, native64, np.asarray(native_ld, dtype=np.float64)


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

    with open(ind.__file__, "rb") as fh:
        module_sha = hashlib.sha256(fh.read()).hexdigest()
    if not module_sha.startswith(MODULE_SHA_PREFIX):
        print(f"MODULE_SHA_MISMATCH {module_sha}")
        return 30
    if not float(np.finfo(np.longdouble).eps) < 1e-18:
        print("LONGDOUBLE_UNAVAILABLE")
        return 30

    temperature = 10.0
    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()
    mass_weight = grid.weights * np.square(grid.nodes)

    states = {}
    all_ok = True
    for name, scales in FAMILY.items():
        state = ind.pair_logits_to_cloglog(
            np.stack([-grid.nodes / s for s in scales])
        )
        action = ind.evaluate_independent_collision_action(
            grid=grid, pair_cloglog=state,
            temperature_cm_mev=temperature, temperature_gamma_mev=temperature,
            config=config,
        )
        modal_total = np.asarray(action.modal_total)
        modal_electron = np.asarray(action.modal_electron)
        mu_modal, tau_modal = pair_rows(modal_total)
        r_weak = rel_linf(mu_modal, tau_modal)
        electron_gap = max(
            float(np.max(np.abs(modal_electron[2] - modal_electron[4]))),
            float(np.max(np.abs(modal_electron[3] - modal_electron[5]))),
        )
        total = np.asarray(action.total)
        mu_nat, tau_nat = pair_rows(total)
        r_mass = rel_linf(mass_weight * mu_nat, mass_weight * tau_nat)
        r_native = float(action.diagnostics["mu_tau_residual"])
        d_native = max(float(np.max(np.abs(mu_nat))), float(np.max(np.abs(tau_nat))), TINY)

        spectra = ind._SpectralLogits(grid, ind._native_pair_logits(state))
        _signed, shadow, addends = self_shadow(grid, spectra, temperature, config)
        mu_sh, tau_sh = pair_rows(shadow)
        s_abs = max(float(np.max(np.abs(mu_sh))), float(np.max(np.abs(tau_sh))))
        gamma_n = addends * EPS / (1.0 - addends * EPS)
        b_sum = gamma_n * s_abs
        n_weak = float(np.max(np.abs(mu_modal - tau_modal)))
        s_signed = max(float(np.max(np.abs(mu_modal))), float(np.max(np.abs(tau_modal))))

        a_cond, b_map, delta_basis, native64, native_ld = native_map_terms(
            grid, modal_total, temperature
        )
        b_basis = delta_basis * s_signed
        mu64, tau64 = pair_rows(native64)
        mu_ld, tau_ld = pair_rows(native_ld)
        w_ld = max(float(np.max(np.abs(mu64 - mu_ld))), float(np.max(np.abs(tau64 - tau_ld)))) / d_native

        b_native = 4.0 * (a_cond * (n_weak + b_sum + b_basis) + b_map + w_ld * d_native) / d_native

        ok = (r_weak <= 1e-10 and r_mass <= 1e-10 and electron_gap == 0.0
              and r_native <= b_native and r_native <= 1e-10)
        all_ok = all_ok and ok
        states[name] = {
            "ok": bool(ok), "R_weak": float(r_weak), "R_mass": float(r_mass),
            "R_native": r_native, "B_native": float(b_native),
            "electron_pair_gap": electron_gap,
            "terms": {"N_weak": n_weak, "B_sum": float(b_sum), "B_basis": float(b_basis),
                      "B_map": b_map, "A_cond": a_cond, "W_ld": float(w_ld),
                      "D_native": d_native, "addend_count": int(addends),
                      "S_abs": s_abs, "S_signed": s_signed,
                      "delta_basis": delta_basis},
        }

    family_max = {
        key: max(states[name][key] for name in states)
        for key in ("R_weak", "R_mass", "R_native", "B_native", "electron_pair_gap")
    }
    verdict = "PASS" if all_ok else "FAIL"
    report = {"contract": "BD622_D054_covariance_metrology_contract_2026-07-28",
              "module_sha256": module_sha,
              "longdouble": str(np.finfo(np.longdouble).eps),
              "family_max": family_max,
              "states": states, "verdict": verdict}
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
