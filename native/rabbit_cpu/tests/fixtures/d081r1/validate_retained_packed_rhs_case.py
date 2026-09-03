#!/usr/bin/env python3
"""Validate a D-081R1E0 retained packed-RHS Python oracle fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "rabbit.d081r1e0.retained_packed_rhs_oracle.v1"
CLAIM_CEILING = "FROZEN_RETAINED_ORDER60_PYTHON_PACKED_RHS_ORACLE_ONLY"
D4_FINAL_HEAD = "002086662bf2e553c78f4b247868cb1fd9e43f21"
COMPARATOR_BLOB = "de44feee0aa484abe26976c7dc34c579643005b5"
TRAJECTORY_CORE_BLOB = "465a73f0ce40f7149bebdc2d67103f388e2344d9"
RETAINED_SHA256 = "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
ORDER = 60
STATE_SIZE = 182


def bits_float(value: str) -> float:
    return float(np.asarray([int(value, 16)], dtype=np.uint64).view(np.float64)[0])


def decode_array(value: dict[str, object]) -> np.ndarray:
    shape = tuple(int(item) for item in value["shape"])
    encoded = value["bits"]
    if not isinstance(encoded, list):
        raise AssertionError("array bits must be a list")
    raw = np.asarray([int(str(item), 16) for item in encoded], dtype=np.uint64)
    array = np.ascontiguousarray(raw.view(np.float64).reshape(shape))
    if hashlib.sha256(array.tobytes(order="C")).hexdigest() != value["sha256"]:
        raise AssertionError("array SHA-256 mismatch")
    if not np.all(np.isfinite(array)):
        raise AssertionError("fixture contains a nonfinite array")
    return array


def decode_float_map(value: dict[str, object]) -> dict[str, float]:
    return {str(key): bits_float(str(encoded)) for key, encoded in value.items()}


def exact_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(
        np.ascontiguousarray(left).view(np.uint64),
        np.ascontiguousarray(right).view(np.uint64),
    )


def validate(payload: dict[str, object]) -> None:
    assert payload["schema"] == SCHEMA
    assert payload["claim_ceiling"] == CLAIM_CEILING

    authorities = payload["authorities"]
    assert authorities["d081r1d4_final_head"] == D4_FINAL_HEAD
    assert authorities["private_comparator_git_blob"] == COMPARATOR_BLOB
    assert authorities["trajectory_core_git_blob"] == TRAJECTORY_CORE_BLOB
    assert authorities["retained_state_sha256"] == RETAINED_SHA256
    assert len(str(authorities["generator_git_blob"])) == 40

    environment = payload["environment"]
    assert environment == {
        "python": "3.12.3",
        "numpy": "2.4.4",
        "scipy": "1.17.1",
        "openblas_threads": "1",
        "omp_threads": "1",
        "mkl_threads": "1",
    }

    configuration = payload["configuration"]
    assert configuration["order"] == ORDER
    assert configuration["state_size"] == STATE_SIZE
    assert bits_float(configuration["y_max_bits"]) == 30.0
    assert configuration["incoming_polar_order"] == 4
    assert configuration["final_polar_order"] == 4
    assert configuration["final_azimuth_order"] == 4
    assert configuration["electron_radial_order"] == 24
    expansion = bits_float(configuration["expansion_n_bits"])
    t_start = bits_float(configuration["temperature_start_mev_bits"])
    t_cm = bits_float(configuration["temperature_cm_mev_bits"])
    assert np.float64(t_start * np.exp(-expansion)).view(np.uint64) == np.float64(t_cm).view(
        np.uint64
    )

    arrays = {key: decode_array(value) for key, value in payload["arrays"].items()}
    assert arrays["packed_state"].shape == (STATE_SIZE,)
    assert arrays["pair_cloglog"].shape == (3, ORDER)
    assert arrays["occupation"].shape == (3, ORDER)
    assert arrays["cloglog_chain"].shape == (3, ORDER)
    assert arrays["grid_nodes"].shape == (ORDER,)
    assert arrays["grid_weights"].shape == (ORDER,)
    for key in (
        "self_native",
        "electron_native",
        "total_native",
        "self_modal",
        "electron_modal",
        "total_modal",
    ):
        assert arrays[key].shape == (6, ORDER)
    assert arrays["pair_rate"].shape == (3, ORDER)
    assert arrays["spectral_rhs"].shape == (3, ORDER)
    assert arrays["packed_rhs_trajectory_core"].shape == (STATE_SIZE,)
    assert arrays["packed_rhs_reconstructed"].shape == (STATE_SIZE,)

    state = arrays["packed_state"]
    assert exact_array_equal(state[: 3 * ORDER].reshape(3, ORDER), arrays["pair_cloglog"])
    assert state[3 * ORDER].view(np.uint64) == np.float64(
        bits_float(configuration["temperature_gamma_mev_bits"])
    ).view(np.uint64)
    assert state[-1].view(np.uint64) == np.float64(
        bits_float(configuration["elapsed_mev_inverse_bits"])
    ).view(np.uint64)

    assert np.all((arrays["occupation"] > 0.0) & (arrays["occupation"] < 1.0))
    assert np.all(arrays["cloglog_chain"] > 0.0)
    assert np.all(arrays["grid_weights"] > 0.0)
    assert np.all(np.diff(arrays["grid_nodes"]) > 0.0)

    assert exact_array_equal(
        arrays["total_native"], arrays["self_native"] + arrays["electron_native"]
    )
    assert exact_array_equal(
        arrays["total_modal"], arrays["self_modal"] + arrays["electron_modal"]
    )
    expected_pair_rate = 0.5 * np.stack(
        (
            arrays["total_native"][0] + arrays["total_native"][1],
            arrays["total_native"][2] + arrays["total_native"][3],
            arrays["total_native"][4] + arrays["total_native"][5],
        )
    )
    assert exact_array_equal(arrays["pair_rate"], expected_pair_rate)
    assert exact_array_equal(
        arrays["packed_rhs_trajectory_core"], arrays["packed_rhs_reconstructed"]
    )
    assert exact_array_equal(
        arrays["packed_rhs_reconstructed"][: 3 * ORDER].reshape(3, ORDER),
        arrays["spectral_rhs"],
    )

    scalars = {key: bits_float(value) for key, value in payload["scalars"].items()}
    assert arrays["packed_rhs_reconstructed"][3 * ORDER].view(np.uint64) == np.float64(
        scalars["temperature_rhs_bits"]
    ).view(np.uint64)
    assert arrays["packed_rhs_reconstructed"][-1].view(np.uint64) == np.float64(
        scalars["elapsed_rhs_bits"]
    ).view(np.uint64)
    assert scalars["occupation_min_bits"] > 0.0
    assert scalars["occupation_max_bits"] < 1.0
    assert scalars["chain_min_bits"] > 0.0
    assert scalars["first_law_residual_bits"] <= 5.0e-13

    thermo = decode_float_map(payload["thermodynamics"])
    eos = decode_float_map(payload["electromagnetic_eos"])
    assert all(np.isfinite(value) and value > 0.0 for value in thermo.values())
    assert all(np.isfinite(value) and value > 0.0 for value in eos.values())
    assert np.float64(1.0 / thermo["hubble_mev"]).view(np.uint64) == np.float64(
        scalars["elapsed_rhs_bits"]
    ).view(np.uint64)
    expected_temperature_rhs = (
        -3.0 * (eos["rho"] + eos["pressure"])
        + scalars["electromagnetic_energy_transfer_bits"] / thermo["hubble_mev"]
    ) / eos["drho_dtemperature"]
    assert np.float64(expected_temperature_rhs).view(np.uint64) == np.float64(
        scalars["temperature_rhs_bits"]
    ).view(np.uint64)

    first_law = abs(
        scalars["neutrino_energy_transfer_bits"]
        + scalars["electromagnetic_energy_transfer_bits"]
    ) / max(
        abs(scalars["neutrino_energy_transfer_bits"])
        + abs(scalars["electromagnetic_energy_transfer_bits"]),
        np.finfo(float).tiny,
    )
    assert np.float64(first_law).view(np.uint64) == np.float64(
        scalars["first_law_residual_bits"]
    ).view(np.uint64)

    for moment in payload["moments"].values():
        assert all(np.isfinite(value) for value in decode_float_map(moment).values())
    assert all(
        np.isfinite(value) for value in decode_float_map(payload["action_diagnostics"]).values()
    )

    metrology = payload["metrology"]
    assert metrology["rhs_paths_bitwise_identical"] is True
    assert metrology["native_component_sum_bitwise_identical"] is True
    assert metrology["modal_component_sum_bitwise_identical"] is True
    assert int(metrology["whole_reaction_domain_rejections"]) >= 0
    assert int(metrology["matrix_roundoff_corrections"]) >= 0
    assert len(metrology["self_row_hashes"]) == 9
    assert len(metrology["electron_family_hashes"]) == 15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()
    fixture = Path(args.fixture)
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    validate(payload)
    print("D-081R1E0 retained packed-RHS oracle validation: PASS")
    print(f"fixture_sha256={hashlib.sha256(fixture.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
