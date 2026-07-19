from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_artifact(path: Path, *, policy: str, T: float, Yp: float) -> None:
    payload = {
        "rows": [
            {
                "stage_collision_payload_policy": policy,
                "T_final_MeV": T,
                "Yp": Yp,
                "DH": 2.5e-5,
                "N_eff_3T": 3.03,
                "completion_class": "bounded_partial",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_payload_reuse_parity_signoff_blocks_unmatched_terminal_temperature(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.json"
    reuse = tmp_path / "reuse.json"
    _write_artifact(current, policy="current_state", T=0.0700, Yp=0.247)
    _write_artifact(reuse, policy="thermo_state_tolerance_reuse", T=0.0705, Yp=0.247)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_payload_reuse_parity_signoff.py",
            str(current),
            str(reuse),
            "--json",
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "BLOCKED_UNMATCHED_TERMINAL_T"
    assert payload["default_on_allowed"] is False


def test_payload_reuse_parity_signoff_passes_matched_stop_within_tolerance(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current.json"
    reuse = tmp_path / "reuse.json"
    _write_artifact(current, policy="current_state", T=0.0700, Yp=0.247000)
    _write_artifact(reuse, policy="thermo_state_tolerance_reuse", T=0.0700, Yp=0.247002)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_payload_reuse_parity_signoff.py",
            str(current),
            str(reuse),
            "--json",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["verdict"] == "PASS_WITH_DEFAULT_ON_BLOCKED_BY_PR_B"
    assert payload["default_on_allowed"] is False
    assert payload["observables"]["Yp"]["relative_delta"] < 1.0e-4
