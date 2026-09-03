#!/usr/bin/env python3
"""Recompute and semantically validate the D-081R1D3 electron authority.

The full and component JSON files remain immutable byte-level authorities.
This validator independently reruns the pinned Python collision operator and
checks exact structure plus a fixed 4096-epsilon, cancellation-aware semantic
envelope.  It never rewrites either canonical fixture.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.decoupling import _independent_noqke as oracle  # noqa: E402
from check_electron_collision_action_metrology import (  # noqa: E402
    SemanticAudit,
    audit,
    bits_float,
    decode_array,
)

COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
FULL_FIXTURE = Path(__file__).with_name("full_collision_action_case.json")
COMPONENT_FIXTURE = Path(__file__).with_name("electron_collision_action_metrology.json")
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_FULL_FIXTURE_BLOB = "c94d2e72a1f8300b7c20c9c793417a5c4a5fa302"
EXPECTED_COMPONENT_FIXTURE_BLOB = "b927389e5aa0c11d41d2e63c83b04ae633fc464d"
ORDER = 8
Y_MAX = 8.0


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def float_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def encode_array(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {
        "shape": list(array.shape),
        "bits": [f"{value:016x}" for value in array.view(np.uint64).ravel().tolist()],
    }


def encode_float_map(values: dict[str, float]) -> dict[str, str]:
    return {key: float_bits(float(values[key])) for key in sorted(values)}


def action_moments(
    grid: oracle.IndependentNoQkeGrid,
    action: np.ndarray,
    temperature_cm: float,
) -> dict[str, str]:
    return encode_float_map(
        asdict(
            oracle.independent_action_moments(
                grid=grid,
                action=action,
                temperature_cm_mev=temperature_cm,
            )
        )
    )


def build_case(
    case: dict[str, object],
    grid: oracle.IndependentNoQkeGrid,
    full_audit: SemanticAudit,
    full_scales: dict[str, float],
) -> dict[str, object]:
    name = str(case["name"])
    pair_cloglog = decode_array(case["pair_cloglog"])
    temperature_cm = bits_float(str(case["temperature_cm_bits"]))
    temperature_gamma = bits_float(str(case["temperature_gamma_bits"]))
    electron_mass = oracle.M_ELECTRON_MEV
    spectra = oracle._SpectralLogits(  # noqa: SLF001
        grid,
        oracle._native_pair_logits(pair_cloglog),  # noqa: SLF001
    )
    modal, family_modal, bath_by_family, meta = oracle._assemble_electron(  # noqa: SLF001
        grid,
        spectra,
        temperature_cm,
        temperature_gamma,
        electron_mass,
        oracle.IndependentCollisionConfig(),
    )
    native = oracle._native_action(grid, modal, temperature_cm)  # noqa: SLF001
    family_native = {
        key: oracle._native_action(grid, value, temperature_cm)  # noqa: SLF001
        for key, value in family_modal.items()
    }

    full_audit.array(
        encode_array(modal),
        case["arrays"]["electron_modal"],
        full_scales["modal"],
        f"{name}.full.electron_modal",
    )
    full_audit.array(
        encode_array(native),
        case["arrays"]["electron_native"],
        full_scales["native"],
        f"{name}.full.electron_native",
    )
    for key, value in family_native.items():
        full_audit.array(
            encode_array(value),
            case["electron_families"][key],
            full_scales["family"],
            f"{name}.full.family.{key}",
        )
        full_audit.scalar(
            float(bath_by_family[key]),
            bits_float(str(case["electron_bath_energy_by_family"][key])),
            full_scales["bath"],
            f"{name}.full.bath.{key}",
        )

    elastic_keys = [key for key in family_modal if not key.endswith(":pair")]
    pair_keys = [key for key in family_modal if key.endswith(":pair")]
    if len(elastic_keys) != 12 or len(pair_keys) != 3:
        raise AssertionError("electron family decomposition changed")
    elastic_modal = np.sum(
        np.stack([family_modal[key] for key in elastic_keys]),
        axis=0,
        dtype=np.float64,
    )
    pair_modal = np.sum(
        np.stack([family_modal[key] for key in pair_keys]),
        axis=0,
        dtype=np.float64,
    )
    elastic_native = oracle._native_action(grid, elastic_modal, temperature_cm)  # noqa: SLF001
    pair_native = oracle._native_action(grid, pair_modal, temperature_cm)  # noqa: SLF001
    modal_difference = float(np.max(np.abs(elastic_modal + pair_modal - modal)))
    modal_scale = max(
        float(np.max(np.abs(modal))),
        float(np.max(np.abs(elastic_modal))),
        float(np.max(np.abs(pair_modal))),
        np.finfo(np.float64).tiny,
    )
    if modal_difference > 256.0 * np.finfo(np.float64).eps * modal_scale:
        raise AssertionError("elastic+pair modal decomposition failed")

    native_residual = elastic_native + pair_native - native
    native_difference = float(np.max(np.abs(native_residual)))
    native_scale = max(
        float(np.max(np.abs(native))),
        float(np.max(np.abs(elastic_native))),
        float(np.max(np.abs(pair_native))),
        np.finfo(np.float64).tiny,
    )
    native_roundoff_budget = 256.0 * np.finfo(np.float64).eps * native_scale
    if native_difference > native_roundoff_budget:
        raise AssertionError(
            "elastic+pair native decomposition failed: "
            f"difference={native_difference:.17e}, budget={native_roundoff_budget:.17e}"
        )

    qnu = float(meta["neutrino_energy_transfer"])
    qem = float(meta["electromagnetic_energy_transfer"])
    neutrino_h = float(meta["neutrino_entropy_rate"])
    electromagnetic_h = float(meta["electromagnetic_entropy_rate"])
    first_law_denominator = max(abs(qnu) + abs(qem), np.finfo(np.float64).tiny)
    first_law_residual = abs(qnu + qem) / first_law_denominator
    node_neutrino_h = float(
        sum(
            np.dot(
                modal[index],
                spectra.coefficients[
                    oracle.PAIR_INDEX[oracle._species_flavour(species)]  # noqa: SLF001
                ],
            )
            for index, species in enumerate(oracle.SPECIES)
        )
    )
    entropy_duality_residual = abs(node_neutrino_h - neutrino_h) / max(
        abs(node_neutrino_h) + abs(neutrino_h),
        np.finfo(np.float64).tiny,
    )
    entropy_production = -(neutrino_h + electromagnetic_h)
    if first_law_residual > 5.0e-13:
        raise AssertionError(f"{name} first-law residual too large")
    entropy_scale = max(
        abs(neutrino_h) + abs(electromagnetic_h), np.finfo(np.float64).tiny
    )
    if entropy_production < -5.0e-13 * entropy_scale:
        raise AssertionError(f"{name} negative entropy production")
    if entropy_duality_residual > 5.0e-12:
        raise AssertionError(f"{name} entropy duality residual too large")

    electron_rejections = int(meta["whole_reaction_domain_rejections"])
    combined_rejections = int(case["whole_reaction_domain_rejections"])
    if combined_rejections <= electron_rejections:
        raise AssertionError("combined rejection counter lost the self contribution")

    return {
        "name": name,
        "temperature_cm_bits": float_bits(temperature_cm),
        "temperature_gamma_bits": float_bits(temperature_gamma),
        "electron_mass_bits": float_bits(electron_mass),
        "family_order": list(family_modal),
        "whole_reaction_domain_rejections": electron_rejections,
        "elastic_domain_rejections": electron_rejections,
        "pair_domain_rejections": 0,
        "combined_domain_rejections": combined_rejections,
        "self_domain_rejections_by_difference": combined_rejections - electron_rejections,
        "matrix_roundoff_corrections": int(meta["matrix_roundoff_corrections"]),
        "largest_matrix_roundoff_correction_bits": float_bits(
            float(meta["largest_matrix_roundoff_correction"])
        ),
        "diagnostics": {
            "neutrino_energy_transfer_bits": float_bits(qnu),
            "electromagnetic_energy_transfer_bits": float_bits(qem),
            "first_law_residual_bits": float_bits(first_law_residual),
            "neutrino_h_rate_bits": float_bits(neutrino_h),
            "electromagnetic_h_rate_bits": float_bits(electromagnetic_h),
            "entropy_production_bits": float_bits(entropy_production),
            "node_neutrino_h_rate_bits": float_bits(node_neutrino_h),
            "entropy_duality_residual_bits": float_bits(entropy_duality_residual),
            "elastic_bath_energy_transfer_bits": float_bits(
                sum(float(bath_by_family[key]) for key in elastic_keys)
            ),
            "pair_bath_energy_transfer_bits": float_bits(
                sum(float(bath_by_family[key]) for key in pair_keys)
            ),
            "category_native_reconstruction_difference_bits": float_bits(native_difference),
            "category_native_reconstruction_budget_bits": float_bits(native_roundoff_budget),
        },
        "bath_energy_by_family": encode_float_map(
            {key: float(value) for key, value in bath_by_family.items()}
        ),
        "moments": action_moments(grid, native, temperature_cm),
        "category_arrays": {
            "elastic_modal": encode_array(elastic_modal),
            "pair_modal": encode_array(pair_modal),
            "elastic_native": encode_array(elastic_native),
            "pair_native": encode_array(pair_native),
        },
    }


def main() -> None:
    identities = {
        "comparator": git_blob(COMPARATOR),
        "full_fixture": git_blob(FULL_FIXTURE),
        "component_fixture": git_blob(COMPONENT_FIXTURE),
    }
    expected = {
        "comparator": EXPECTED_COMPARATOR_BLOB,
        "full_fixture": EXPECTED_FULL_FIXTURE_BLOB,
        "component_fixture": EXPECTED_COMPONENT_FIXTURE_BLOB,
    }
    if identities != expected:
        raise SystemExit(f"authority mismatch: {identities} != {expected}")

    full = json.loads(FULL_FIXTURE.read_text(encoding="utf-8"))
    canonical = json.loads(COMPONENT_FIXTURE.read_text(encoding="utf-8"))
    full_by_name = {case["name"]: case for case in full["cases"]}
    thermal = full_by_name["thermal_split"]
    full_scales = {
        "modal": float(np.max(np.abs(decode_array(thermal["arrays"]["electron_modal"])))),
        "native": float(np.max(np.abs(decode_array(thermal["arrays"]["electron_native"])))),
        "family": max(
            float(np.max(np.abs(decode_array(value))))
            for value in thermal["electron_families"].values()
        ),
        "bath": max(
            abs(bits_float(str(value)))
            for value in thermal["electron_bath_energy_by_family"].values()
        ),
    }
    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    full_semantic = SemanticAudit()
    cases = [
        build_case(case, grid, full_semantic, full_scales)
        for case in full["cases"]
    ]
    by_name = {case["name"]: case for case in cases}
    thermal_candidate = by_name["thermal_split"]
    qnu = bits_float(thermal_candidate["diagnostics"]["neutrino_energy_transfer_bits"])
    qem = bits_float(thermal_candidate["diagnostics"]["electromagnetic_energy_transfer_bits"])
    if not (qnu > 0.0 and qem < 0.0):
        raise AssertionError("thermal restoring energy-transfer sign failed")

    candidate = {
        "schema": "rabbit.d081r1d3.electron_action_metrology.v1",
        "private_comparator_git_blob": identities["comparator"],
        "full_collision_fixture_git_blob": identities["full_fixture"],
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "electron_mass_bits": float_bits(oracle.M_ELECTRON_MEV),
        "elastic_event_count": 12,
        "pair_event_count": 3,
        "electron_event_count": 15,
        "component_boundary": (
            "electron metadata are emitted directly by _assemble_electron; "
            "combined counters remain provenance cross-checks only"
        ),
        "category_native_reconstruction_budget_ulps": 256,
        "cases": cases,
    }
    component_semantic = audit(candidate, canonical)
    payload = {
        "schema": "rabbit.d081r1d3.electron_authority_validation.v1",
        "authority_blobs": identities,
        "full_fixture_semantic": {
            "worst_label": full_semantic.worst_label,
            "worst_ratio": full_semantic.worst_ratio,
            "worst_difference": full_semantic.worst_difference,
            "worst_allowed": full_semantic.worst_allowed,
        },
        "component_fixture_semantic": {
            "worst_label": component_semantic.worst_label,
            "worst_ratio": component_semantic.worst_ratio,
            "worst_difference": component_semantic.worst_difference,
            "worst_allowed": component_semantic.worst_allowed,
        },
        "thermal_energy_transfer": {"qnu": qnu, "qem": qem},
        "candidate_sha256": hashlib.sha256(
            (json.dumps(candidate, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest(),
        "verdict": "PASS",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
