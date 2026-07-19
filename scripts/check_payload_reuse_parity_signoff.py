from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


_TERMINAL_T_KEYS = (
    "T_final_MeV",
    "terminal_T_gamma_MeV",
    "T_gamma_terminal_MeV",
)
_OBSERVABLE_KEYS = {
    "Yp": ("Yp", "Y_p", "raw_Yp", "raw_Y_p"),
    "DH": ("DH", "D_over_H", "D/H"),
    "N_eff_3T": ("N_eff_3T",),
}


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _first_row(payload: Mapping[str, Any], *, path: Path) -> Mapping[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty rows list.")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise ValueError(f"{path} rows[0] must be a JSON object.")
    return row


def _finite_row_float(row: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _relative_delta(current: float, reuse: float) -> float | None:
    scale = max(abs(current), abs(reuse))
    if scale <= 0.0:
        return 0.0 if current == reuse else None
    return (reuse - current) / scale


def build_signoff_report(
    current_artifact: Path,
    reuse_artifact: Path,
    *,
    max_terminal_T_delta: float,
    Yp_relative_tolerance: float,
    DH_relative_tolerance: float,
    N_eff_3T_absolute_tolerance: float,
) -> dict[str, Any]:
    current_payload = _load_json(current_artifact)
    reuse_payload = _load_json(reuse_artifact)
    current = _first_row(current_payload, path=current_artifact)
    reuse = _first_row(reuse_payload, path=reuse_artifact)
    current_T = _finite_row_float(current, _TERMINAL_T_KEYS)
    reuse_T = _finite_row_float(reuse, _TERMINAL_T_KEYS)
    terminal_delta = None
    terminal_matched = False
    if current_T is not None and reuse_T is not None:
        terminal_delta = abs(float(reuse_T) - float(current_T))
        terminal_matched = terminal_delta <= float(max_terminal_T_delta)

    tolerances = {
        "Yp": {"mode": "relative", "limit": float(Yp_relative_tolerance)},
        "DH": {"mode": "relative", "limit": float(DH_relative_tolerance)},
        "N_eff_3T": {"mode": "absolute", "limit": float(N_eff_3T_absolute_tolerance)},
    }
    observables: dict[str, dict[str, Any]] = {}
    observable_failures: list[str] = []
    for name, keys in _OBSERVABLE_KEYS.items():
        current_value = _finite_row_float(current, keys)
        reuse_value = _finite_row_float(reuse, keys)
        entry: dict[str, Any] = {
            "current": current_value,
            "reuse": reuse_value,
            "delta_reuse_minus_current": None,
            "relative_delta": None,
            "absolute_delta": None,
            "tolerance": tolerances[name],
            "status": "missing",
        }
        if current_value is not None and reuse_value is not None:
            delta = float(reuse_value) - float(current_value)
            rel = _relative_delta(float(current_value), float(reuse_value))
            abs_delta = abs(delta)
            entry.update(
                {
                    "delta_reuse_minus_current": delta,
                    "relative_delta": rel,
                    "absolute_delta": abs_delta,
                }
            )
            if name == "N_eff_3T":
                passed = abs_delta <= float(N_eff_3T_absolute_tolerance)
            else:
                passed = rel is not None and abs(float(rel)) <= float(
                    tolerances[name]["limit"]
                )
            entry["status"] = "pass" if passed else "fail"
        if entry["status"] != "pass":
            observable_failures.append(name)
        observables[name] = entry

    blockers = ["PR_B_LRS_NON_LRS_PARITY_AND_COLD_N_EFF_3T_FLOOR_NOT_PROVEN"]
    if not terminal_matched:
        verdict = "BLOCKED_UNMATCHED_TERMINAL_T"
        exit_code = 2
    elif observable_failures:
        verdict = "FAIL_OBSERVABLE_PARITY"
        exit_code = 1
    else:
        verdict = "PASS_WITH_DEFAULT_ON_BLOCKED_BY_PR_B"
        exit_code = 0
    return {
        "contract": "payload_reuse_parity_signoff_v1",
        "current_artifact": str(current_artifact),
        "reuse_artifact": str(reuse_artifact),
        "current_policy": current.get("stage_collision_payload_policy"),
        "reuse_policy": reuse.get("stage_collision_payload_policy"),
        "terminal_temperature": {
            "current_T_final_MeV": current_T,
            "reuse_T_final_MeV": reuse_T,
            "absolute_delta": terminal_delta,
            "max_allowed_delta": float(max_terminal_T_delta),
            "matched": bool(terminal_matched),
        },
        "observables": observables,
        "observable_failures": observable_failures,
        "default_on_allowed": False,
        "default_on_blockers": blockers,
        "raw_observables_preserved": True,
        "verdict": verdict,
        "exit_code": int(exit_code),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check current-state vs thermo-state payload-reuse artifact parity "
            "without authorizing default-on before PR-B parity/floor evidence."
        )
    )
    parser.add_argument("current_artifact", type=Path)
    parser.add_argument("reuse_artifact", type=Path)
    parser.add_argument("--max-terminal-T-delta", type=float, default=1.0e-8)
    parser.add_argument("--Yp-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--DH-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--N-eff-3T-absolute-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_signoff_report(
        args.current_artifact,
        args.reuse_artifact,
        max_terminal_T_delta=float(args.max_terminal_T_delta),
        Yp_relative_tolerance=float(args.Yp_relative_tolerance),
        DH_relative_tolerance=float(args.DH_relative_tolerance),
        N_eff_3T_absolute_tolerance=float(args.N_eff_3T_absolute_tolerance),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
