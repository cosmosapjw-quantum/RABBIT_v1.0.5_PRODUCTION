#!/usr/bin/env python3
"""Freeze component-specific D-081R1D2 self-action metrology.

The pre-existing full-collision fixture intentionally stores combined
self+electron counters.  This companion fixture calls the frozen private
``_assemble_self`` authority directly so a self-only Rust object is never
compared with combined-component metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rabbit.decoupling import _independent_noqke as oracle  # noqa: E402

COMPARATOR = ROOT / "src/rabbit/decoupling/_independent_noqke.py"
FULL_FIXTURE = Path(__file__).with_name("full_collision_action_case.json")
OUTPUT = Path(__file__).with_name("self_collision_action_metrology.json")
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


def main() -> None:
    comparator_blob = git_blob(COMPARATOR)
    full_fixture_blob = git_blob(FULL_FIXTURE)
    if comparator_blob != EXPECTED_COMPARATOR_BLOB:
        raise SystemExit(
            f"frozen comparator mismatch: {comparator_blob} != {EXPECTED_COMPARATOR_BLOB}"
        )
    if full_fixture_blob != EXPECTED_FULL_FIXTURE_BLOB:
        raise SystemExit(
            f"full fixture mismatch: {full_fixture_blob} != {EXPECTED_FULL_FIXTURE_BLOB}"
        )

    grid = oracle.build_independent_grid(order=ORDER, y_max=Y_MAX)
    equilibrium = np.stack([-grid.nodes, -grid.nodes, -grid.nodes])
    profile = 0.02 * (1.0 - 2.0 * grid.nodes / Y_MAX)
    mu_tau = np.stack([-grid.nodes, -grid.nodes + profile, -grid.nodes - profile])
    cases = (
        ("equilibrium", equilibrium, 2.0, 2.0),
        ("thermal_split", equilibrium, 2.0, 2.05),
        ("mu_tau_split", mu_tau, 2.0, 2.0),
    )

    payload: dict[str, object] = {
        "schema": "rabbit.d081r1d2.self_action_metrology.v1",
        "private_comparator_git_blob": comparator_blob,
        "full_collision_fixture_git_blob": full_fixture_blob,
        "order": ORDER,
        "y_max_bits": float_bits(Y_MAX),
        "self_event_count": len(oracle.independent_self_events()),
        "component_boundary": (
            "self fields are emitted directly by _assemble_self; combined fields "
            "come from evaluate_independent_collision_action only as a provenance "
            "cross-check and are not self-action acceptance targets"
        ),
        "cases": [],
    }

    output_cases: list[dict[str, object]] = []
    for name, logits, temperature_cm, temperature_gamma in cases:
        pair_cloglog = oracle.pair_logits_to_cloglog(logits)
        spectra = oracle._SpectralLogits(
            grid, oracle._native_pair_logits(pair_cloglog)
        )
        _, _, self_meta = oracle._assemble_self(
            grid,
            spectra,
            temperature_cm,
            oracle.IndependentCollisionConfig(),
        )
        combined = oracle.evaluate_independent_collision_action(
            grid=grid,
            pair_cloglog=pair_cloglog,
            temperature_cm_mev=temperature_cm,
            temperature_gamma_mev=temperature_gamma,
        )

        self_rejections = int(self_meta["whole_reaction_domain_rejections"])
        self_corrections = int(self_meta["matrix_roundoff_corrections"])
        combined_rejections = int(combined.whole_reaction_domain_rejections)
        combined_corrections = int(combined.matrix_roundoff_corrections)
        if self_rejections >= combined_rejections:
            raise AssertionError("combined rejection count must contain an electron component")
        if self_corrections > combined_corrections:
            raise AssertionError("self correction count exceeds combined correction count")

        output_cases.append(
            {
                "name": name,
                "self": {
                    "whole_reaction_domain_rejections": self_rejections,
                    "matrix_roundoff_corrections": self_corrections,
                    "largest_matrix_roundoff_correction_bits": float_bits(
                        float(self_meta["largest_matrix_roundoff_correction"])
                    ),
                },
                "combined": {
                    "whole_reaction_domain_rejections": combined_rejections,
                    "matrix_roundoff_corrections": combined_corrections,
                    "largest_matrix_roundoff_correction_bits": float_bits(
                        float(combined.largest_matrix_roundoff_correction)
                    ),
                },
                "electron_by_difference": {
                    "whole_reaction_domain_rejections": (
                        combined_rejections - self_rejections
                    ),
                    "matrix_roundoff_corrections": (
                        combined_corrections - self_corrections
                    ),
                },
            }
        )

    payload["cases"] = output_cases
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
