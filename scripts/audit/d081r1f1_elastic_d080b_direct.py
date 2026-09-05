#!/usr/bin/env python3
"""Frozen D-080A/B prefactor arrays and two separately gated Rust comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit import _d080b_tgamma_collision as d080b
from scripts.audit._d080_tgamma_primitives import (
    evaluate_elastic_tgamma_kinematic_tangent,
)
from scripts.audit.d081r1f1_p0b_tgamma_kinematics_oracle import bit_array, bits

CAP = 1.0e-7  # Frozen before execution; derivative arrays supply their own scale.
CASES = ((2.0, 2.05), (0.75, 0.4))
FIELDS = ("p2", "e2", "phase_space", "quadrature_weight",
          "d12", "d13", "d14", "d23", "d24", "d34")
AUTHORITY = {
    "scripts/audit/_d080_tgamma_primitives.py": "c585d5865fd68a90a04a76ab540b8437fba8cfce",
    "scripts/audit/_d080b_tgamma_collision.py": "78489c43f3046db09d8ba2d96070124ed7b0aa91",
    "src/rabbit/decoupling/_independent_noqke.py": "de44feee0aa484abe26976c7dc34c579643005b5",
}


def write(path, payload):
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    print(f"{path.name} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")


def mask(values):
    return np.asarray(values, dtype=bool).ravel(order="C").tolist()


def generate(directory):
    for name, expected in AUTHORITY.items():
        data = Path(name).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert actual == expected, (name, actual, expected)
    config = ind.IndependentCollisionConfig()
    cases = []
    for p1, temperature in CASES:
        tangent = evaluate_elastic_tgamma_kinematic_tangent(
            p1=p1, temperature_gamma_mev=temperature,
            electron_mass_mev=ind.M_ELECTRON_MEV, config=config)
        base = tangent.base
        domain = np.asarray(base.support, dtype=bool)
        indices = np.flatnonzero(domain.ravel(order="C"))
        measure = d080b._measure_and_tangent(
            tangent=tangent, p1=p1, outer_weight=1.0, domain=domain)
        outside = d080b._measure_and_tangent(
            tangent=tangent, p1=p1, outer_weight=1.0, domain=~domain)
        inputs = {name: bit_array(getattr(base, name)) for name in FIELDS}
        inputs.update({"d_" + name: bit_array(getattr(tangent, "d_" + name))
                       for name in FIELDS})
        routes = []
        for event in ind.independent_electron_events():
            if event.category == "pair":
                continue
            value, derivative, corrected = d080b._elastic_matrix_value_and_tangent(
                target=event.target, category=event.category, tangent=tangent,
                electron_mass=ind.M_ELECTRON_MEV, config=config)
            raw, scale = d080b._electron_matrix_raw(
                event.target, event.category, base, ind.M_ELECTRON_MEV)
            routes.append({
                "target": event.target, "category": event.category,
                "M": bit_array(value), "M_T": bit_array(derivative),
                "raw": bit_array(raw), "scale": bit_array(scale),
                "corrected": mask(corrected),
                "kink": mask(domain & ~corrected & (raw == 0.0) & (derivative != 0.0)),
            })
        assert len(routes) == 12
        cases.append({
            "p1": bits(p1), "temperature": bits(temperature),
            "electron_mass": bits(ind.M_ELECTRON_MEV), "outer_weight": bits(1.0),
            "shape": list(domain.shape), "sample_indices": list(range(domain.size)),
            "support": mask(base.support), "tangent_support": mask(tangent.support),
            "domain": mask(domain), "domain_indices": indices.tolist(), "input": inputs,
            "measure": {"W": bit_array(measure[0]), "W_T": bit_array(measure[1])},
            "outside_indices": np.flatnonzero(~domain.ravel(order="C")).tolist(),
            "outside_measure": {"W": bit_array(outside[0]), "W_T": bit_array(outside[1])},
            "routes": routes,
        })
    write(directory / "oracle.json", {
        "schema": "rabbit.elastic_d080b_direct.v1", "authority": AUTHORITY,
        "cap": CAP, "domain_policy": "base.support; no spectral/action projection cut",
        "config": asdict(config), "cases": cases,
    })


def decode(array):
    values = np.asarray([np.nan if value is None else
                         np.asarray(int(value, 16), dtype=np.uint64).view(np.float64).item()
                         for value in array["bits"]], dtype=np.float64)
    assert values.size == int(np.prod(array["shape"])), array["shape"]
    return values


def metric(actual, expected, indices):
    assert len(actual) == len(expected) == len(indices) > 0
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    difference = np.abs(actual - expected)
    scale = max(np.max(np.abs(actual)), np.max(np.abs(expected)), np.finfo(float).tiny)
    local = difference / np.maximum(np.maximum(np.abs(actual), np.abs(expected)),
                                    np.finfo(float).tiny)
    worst = int(np.argmax(difference))
    local_worst = int(np.argmax(local))
    return {
        "samples": len(indices), "max_absolute": float(difference[worst]),
        "global_scale": float(scale), "global_relative": float(difference[worst] / scale),
        "max_absolute_index": int(indices[worst]),
        "worst_local_relative": float(local[local_worst]),
        "worst_local_index": int(indices[local_worst]),
        "worst_actual_bits": bits(actual[worst]), "worst_expected_bits": bits(expected[worst]),
    }


def compare(directory, selected_mode=None):
    oracle = json.loads((directory / "oracle.json").read_text())
    assert oracle["cap"] == CAP
    reports, failures = [], []

    def gate(ok, context):
        if not ok:
            failures.append(context)

    modes = (selected_mode,) if selected_mode else ("same_input", "end_to_end")
    for mode in modes:
        outputs = json.loads((directory / f"{mode}.json").read_text())
        assert len(outputs) == len(oracle["cases"]) == 2
        for case_index, (reference, output) in enumerate(zip(oracle["cases"], outputs)):
            context = {"mode": mode, "case": case_index}
            n = int(np.prod(reference["shape"]))
            assert n == 27648
            equality = {name: output[name] == reference[name] for name in (
                "shape", "sample_indices", "support", "tangent_support", "domain",
                "domain_indices", "p1", "temperature", "electron_mass", "outer_weight")}
            gate(all(equality.values()), {**context, "mask_or_input_mismatch": equality})
            assert output["sample_indices"] == list(range(n)), "noncanonical Rust index order"
            indices = np.asarray(reference["domain_indices"], dtype=int)
            assert indices.tolist() == np.flatnonzero(reference["domain"]).tolist()
            assert len(set(indices)) == len(indices)
            if mode == "same_input":
                gate(reference["input"] == output["input"], {**context, "input_bits": False})
            assert len(output["routes"]) == len(reference["routes"]) == 12
            for ref_route, route in zip(reference["routes"], output["routes"]):
                key = {**context, "target": ref_route["target"], "category": ref_route["category"]}
                assert (route["target"], route["category"]) == (key["target"], key["category"])
                support = np.asarray(output["support"], dtype=bool)
                corrected = np.asarray(route["corrected"], dtype=bool)
                kink = np.asarray(route["kink"], dtype=bool)
                assert len(support) == len(corrected) == len(kink) == n
                masks = {**equality, "correction": route["corrected"] == ref_route["corrected"],
                         "kink": route["kink"] == ref_route["kink"]}
                gate(all(masks.values()), {**key, "mask_mismatch": masks})
                smooth = (support & np.asarray(reference["support"], dtype=bool)
                          & ~corrected & ~np.asarray(ref_route["corrected"], dtype=bool)
                          & ~kink & ~np.asarray(ref_route["kink"], dtype=bool))
                matrix_indices = np.flatnonzero(smooth)
                zero_branch = ~support | corrected
                matrix = decode(route["M"])
                derivative = decode(route["M_T"])
                gate(bool(np.all(matrix[zero_branch] == 0.0)
                          and np.all(derivative[zero_branch] == 0.0)),
                     {**key, "unsupported_or_corrected_zero": False})
                gate(all(route["M_T"]["bits"][i] is None
                         and route["status"][i] == "NondifferentiableDiscreteEvent"
                         for i in np.flatnonzero(kink)),
                     {**key, "typed_kink_refusal": False})
                report = {**key, "masks_exact": masks, "total": n,
                          "supported": int(support.sum()), "domain": len(indices),
                          "unsupported": int((~support).sum()), "corrected": int(corrected.sum()),
                          "kink": int(kink.sum()), "kink_indices": np.flatnonzero(kink).tolist(),
                          "smooth_matrix": len(matrix_indices), "metrics": {}}
                for quantity in ("W", "W_T", "M", "M_T"):
                    if quantity.startswith("W"):
                        # Explicit original-index gather against D-080B's domain-reduced array.
                        actual = decode(output["measure"][quantity])[indices]
                        expected = decode(reference["measure"][quantity])
                        selected = indices
                        outside_indices = reference["outside_indices"]
                        outside = decode(output["measure"][quantity])[outside_indices]
                        frozen_outside = decode(reference["outside_measure"][quantity])
                        gate(bool(np.all(outside == 0.0) and np.all(frozen_outside == 0.0)),
                             {**key, "quantity": quantity, "unsupported_measure_zero": False})
                    else:
                        selected = matrix_indices
                        actual = decode(route[quantity])[selected]
                        expected = decode(ref_route[quantity])[selected]
                    values = metric(actual, expected, selected)
                    report["metrics"][quantity] = values
                    gate(values["global_relative"] <= CAP, {**key, "quantity": quantity, **values})
                reports.append(report)
                print(json.dumps(report, sort_keys=True, allow_nan=False))
    assert len(reports) == 24 * len(modes)
    assert sum(len(r["metrics"]) for r in reports) == 96 * len(modes)
    write(directory / f"metrics-{selected_mode or 'both'}.json", {"cap": CAP, "reports": reports, "failures": failures,
                                       "first_failure": failures[0] if failures else None})
    assert not failures, f"STOP_DIRECT_PARITY: {failures[0]}"
    print(f"PASS_D080B_DIRECT_PREFACTOR_PARITY_{selected_mode or 'both'}_{96 * len(modes)}_METRICS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "compare"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--mode", choices=("same_input", "end_to_end"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.operation == "generate":
        generate(args.directory)
    else:
        compare(args.directory, args.mode)
