#!/usr/bin/env python3
"""Fail-fast checker for AP65 component wall attribution summaries."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _looks_like_summary(data: Mapping[str, Any]) -> bool:
    return isinstance(data.get("component_wall_attribution"), Mapping) and isinstance(
        data.get("timer_availability"),
        Mapping,
    )


def _load_summarizer():
    script = Path(__file__).resolve().with_name("summarize_perf_artifacts.py")
    spec = importlib.util.spec_from_file_location("summarize_perf_artifacts", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load summarizer at {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SUMMARIZER = _load_summarizer()
REQUIRED_VALUE_OR_REASON_TIMER_FIELDS = tuple(
    sorted(_SUMMARIZER.VALUE_OR_REASON_TIMER_FIELDS)
)
REQUIRED_RUNTIME_TIMER_FIELDS = tuple(
    field
    for field in _SUMMARIZER.ATTRIBUTION_COMPLETENESS_TIMER_FIELDS
    if field not in _SUMMARIZER.VALUE_OR_REASON_TIMER_FIELDS
)
MAX_RESIDUAL_UNATTRIBUTED_WALL_FRACTION = 0.20


def _residual_unattributed_fraction(attribution: Mapping[str, Any]) -> float | None:
    fraction = _finite_number(attribution.get("residual_unattributed_wall_fraction"))
    if fraction is not None:
        return fraction
    residual = _finite_number(attribution.get("residual_unattributed_wall_seconds"))
    total = _finite_number(attribution.get("total_wall_seconds"))
    if residual is None or total is None or total <= 0.0:
        return None
    return residual / total


def _summary_from_path(path: Path) -> Mapping[str, Any]:
    if path.is_file():
        data = _load_json(path)
        if data is not None and _looks_like_summary(data):
            return data
        return _SUMMARIZER.summarize_artifacts(path)

    return _SUMMARIZER.summarize_artifacts(path)


def _check_component_wall(summary: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    attribution = summary.get("component_wall_attribution")
    if not isinstance(attribution, Mapping):
        return ["component_wall_attribution missing"]

    row_count = int(attribution.get("attribution_row_count", 0) or 0)
    total = _finite_number(attribution.get("total_wall_seconds"))
    attributed = _finite_number(attribution.get("attributed_wall_seconds_total"))
    residual = _finite_number(attribution.get("residual_unattributed_wall_seconds"))
    if row_count <= 0:
        if total is not None and total > 1.0e-9:
            failures.append(
                "component attribution has nonzero total wall but no rows: "
                f"total={total}"
            )
        else:
            failures.append("component attribution has no rows")
    if total is None:
        failures.append("total_wall_seconds is missing or nonfinite")
    if attributed is None:
        failures.append("attributed_wall_seconds_total is missing or nonfinite")
    if residual is None:
        failures.append("residual_unattributed_wall_seconds is missing or nonfinite")
    elif residual < -1.0e-9:
        failures.append(f"residual_unattributed_wall_seconds is negative: {residual}")
    if total is not None and attributed is not None and attributed - total > 1.0e-9:
        failures.append(
            "component attribution exceeds total wall: "
            f"attributed={attributed}, total={total}"
        )
    fraction = _residual_unattributed_fraction(attribution)
    if (
        row_count > 0
        and fraction is not None
        and fraction > MAX_RESIDUAL_UNATTRIBUTED_WALL_FRACTION
    ):
        failures.append(
            "residual_unattributed_wall_fraction is high: "
            f"{fraction:.6g} (> {MAX_RESIDUAL_UNATTRIBUTED_WALL_FRACTION:g})"
        )
    return failures


def _component_wall_warnings(summary: Mapping[str, Any]) -> list[str]:
    attribution = summary.get("component_wall_attribution")
    if not isinstance(attribution, Mapping):
        return []
    return []


def _check_timer_availability(summary: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    attribution = summary.get("component_wall_attribution")
    row_count = 0
    if isinstance(attribution, Mapping):
        row_count = int(attribution.get("attribution_row_count", 0) or 0)
    availability = summary.get("timer_availability")
    if not isinstance(availability, Mapping):
        return ["timer_availability missing"]

    for field in REQUIRED_RUNTIME_TIMER_FIELDS:
        entry = availability.get(field)
        if not isinstance(entry, Mapping):
            failures.append(f"{field} availability missing")
            continue
        value_count = int(entry.get("value_count", 0) or 0)
        missing_count = int(entry.get("missing_count", 0) or 0)
        nonfinite_count = int(entry.get("nonfinite_count", 0) or 0)
        if nonfinite_count:
            failures.append(f"{field} has {nonfinite_count} nonfinite value(s)")
        if missing_count:
            failures.append(f"{field} missing on {missing_count} attribution row(s)")
        if row_count > 0 and value_count < row_count:
            failures.append(
                f"{field} is not populated on every attribution row "
                f"({value_count}/{row_count})"
            )

    for field in REQUIRED_VALUE_OR_REASON_TIMER_FIELDS:
        entry = availability.get(field)
        if not isinstance(entry, Mapping):
            failures.append(f"{field} availability missing")
            continue
        value_count = int(entry.get("value_count", 0) or 0)
        reason_count = int(entry.get("unavailable_reason_count", 0) or 0)
        missing_count = int(entry.get("missing_count", 0) or 0)
        nonfinite_count = int(entry.get("nonfinite_count", 0) or 0)
        if nonfinite_count:
            failures.append(f"{field} has {nonfinite_count} nonfinite value(s)")
        if row_count > 0 and value_count + reason_count < row_count:
            failures.append(
                f"{field} has neither value nor unavailable reason on every "
                f"attribution row ({value_count}+{reason_count}/{row_count})"
            )
        if missing_count and value_count + reason_count < row_count:
            failures.append(f"{field} missing on {missing_count} attribution row(s)")
    return failures


def check_summary(summary: Mapping[str, Any]) -> list[str]:
    failures = []
    failures.extend(_check_component_wall(summary))
    failures.extend(_check_timer_availability(summary))
    return failures


def warning_summary(summary: Mapping[str, Any]) -> list[str]:
    return _component_wall_warnings(summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", help="AP65 artifact directory or summary JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = _summary_from_path(Path(args.artifact_root))
    failures = check_summary(summary)
    warnings = warning_summary(summary)
    if failures:
        print("FAIL component wall attribution")
        for failure in failures:
            print(f"- {failure}")
        return 1
    if warnings:
        print("WARN component wall attribution")
        for warning in warnings:
            print(f"- {warning}")
    print("PASS component wall attribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
