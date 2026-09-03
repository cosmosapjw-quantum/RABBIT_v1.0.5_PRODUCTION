#!/usr/bin/env python3
"""Freeze component-specific D-081R1D3 electron-action metrology.

The pre-existing full-collision fixture stores the authoritative electron
action arrays and family rows, but its support and matrix counters are
combined self+electron totals. This companion fixture calls the frozen
private ``_assemble_electron`` authority directly so the Rust electron-only
object is compared with electron-only metadata.
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

COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
FULL_FIXTURE = Path(__file__).with_name("full_collision_action_case.json")
OUTPUT = Path(__file__).with_name("electron_collision_action_metrology.json")
EXPECTED_COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_FULL_FIXTURE_BLOB = "c94d2e72a1f8300b7c20c9c793417a5c4a5fa302"
ORDER = 8
Y_MAX = 8.0


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value: float) -> str:
    return f"{np.float64(value).view(np.uint64).item():016x}"


def bits_float(bits: str) -> float:
    digits = bits.removeprefix("0x")
    return float(np.asarray([int(digits, 16)], dtype=np.uint64).view(np.float64)[0])


def decode_array(encoded: dict[str, object]) -> np.ndarray:
    shape = tuple(int(value) for value in encoded["shape"])
    raw = np.asarray(
        [int(str(value).removeprefix("0x"), 16) for value in encoded["bits"]],
        dtype=np.uint64,
    )
    return np.ascontiguousarray(raw.view(np.float64).reshape(shape))


def encode_array(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return {
        "shape": list(array.shape),
        "bits": [f"{value:016x}" for value in array.view(np.uint64).ravel().tolist()],
    }


def encode_float_map(values: dict[str, float]) -> dict[str, str]:
    return {key: float_bits(float(values[key])) for key in sorted(values)}


def assert_bit_identical(actual: np.ndarray, encoded: dict[str, object], label: str) -> None:
    expected = decode_array(encoded)
    if actual.shape != expected.shape or not np.array_equal(
        np.asarray(actual, dtype=np.float64).view(np.uint64),
        expected.view(np.uint64),
    ):
        difference = float(np.max(np.abs(np.asarray(actual) - expected), initial=0.0))
        raise AssertionError(
            f"{label} is not bit-identical to the full fixture: {difference:.17e}"
        )


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

    assert_bit_identical(modal, case["arrays"]["electron_modal"], f"{name} electron modal")
    assert_bit_identical(native, case["arrays"]["electron_native"], f"{name} electron native")
    for key, value in family_native.items():
        assert_bit_identical(
            value,
            case["electron_families"][key],
            f"{name} electron family {key}",
        )
        expected_bath = bits_float(str(case["electron_bath_energy_by_family"][key]))
        if np.float64(bath_by_family[key]).view(np.uint64) != np.float64(
            expected_bath
        ).view(np.uint64):
            raise AssertionError(f"{name} bath ledger mismatch for {key}")

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
    elastic_native = oracle._native_action(  # noqa: SLF001
        grid, elastic_modal, temperature_cm
    )
    pair_native = oracle._native_action(grid, pair_modal, temperature_cm)  # noqa: SLF001
    if not np.allclose(elastic_modal + pair_modal, modal, rtol=0.0, atol=2.0e-38):
        raise AssertionError("elastic+pair modal decomposition failed")

    native_residual = elastic_native + pair_native - native
    native_difference = float(np.max(np.abs(native_residual)))
    native_scale = float(
        max(
            np.max(np.abs(native)),
            np.max(np.abs(elastic_native)),
            np.max(np.abs(pair_native)),
            np.finfo(np.float64).tiny,
        )
    )
    native_roundoff_budget = 256.0 * np.finfo(np.float64).eps * native_scale
    if native_difference > native_roundoff_budget:
        raise AssertionError(
            "elastic+pair native decomposition failed: "
            f"difference={native_difference:.17e}, "
            f"budget={native_roundoff_budget:.17e}"
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
        raise AssertionError(
            f"{name} first-law residual too large: {first_law_residual:.17e}"
        )
    entropy_scale = max(
        abs(neutrino_h) + abs(electromagnetic_h), np.finfo(np.float64).tiny
    )
    if entropy_production < -5.0e-13 * entropy_scale:
        raise AssertionError(
            f"{name} negative entropy production: {entropy_production:.17e}"
        )
    if entropy_duality_residual > 5.0e-12:
        raise AssertionError(
            f"{name} electron-action entropy duality residual too large: "
            f"{entropy_duality_residual:.17e}"
        )

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
        "self_domain_rejections_by_difference": (
            combined_rejections - electron_rejections
        ),
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
            "category_native_reconstruction_difference_bits": float_bits(
                native_difference
            ),
            "category_native_reconstruction_budget_bits": float_bits(
                native_roundoff_budget
            ),
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
    comparator_blob = git_blob(COMPARATOR)
    full_fixture_blob = git_blob(FULL_FIXTURE)
    if comparator_blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(
            f"frozen comparator mismatch: {comparator_blob} != "
            f"{EXPECTED_COMPARATOR_BLOB}"
        )
    if full_fixture_blob != EXPECTED_FULL_FIXTURE_BLOB:
        raise SystemExit(
            f"full fixture mismatch: {full_fixture_blob} != "
            f"{EXPECTED_FULL_FIXTURE_BLOB}"
        )

    full = json.loads(FULL_FIXTURE.read_text(encoding="utf-8"))
    if full["schema"] != "rabbit.d081r1.full_collision_action.v1":
        raise AssertionError("unexpected full-action fixture schema")
    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    cases = [build_case(case, grid) for case in full["cases"]]
    by_name = {case["name"]: case for case in cases}
    thermal = by_name["thermal_split"]
    qnu = bits_float(thermal["diagnostics"]["neutrino_energy_transfer_bits"])
    qem = bits_float(thermal["diagnostics"]["electromagnetic_energy_transfer_bits"])
    if not (qnu > 0.0 and qem < 0.0):
        raise AssertionError("thermal restoring energy-transfer sign failed")

    payload = {
        "schema": "rabbit.d081r1d3.electron_action_metrology.v1",
        "private_comparator_git_blob": comparator_blob,
        "full_collision_fixture_git_blob": full_fixture_blob,
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
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
