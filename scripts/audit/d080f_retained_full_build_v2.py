#!/usr/bin/env python3
"""D-080F v2 runner with cancellation-aware dense-action metrology.

The physical matrix construction remains the frozen D-080F implementation.
This wrapper preserves the v1 forward-relative diagnostic, evaluates a
prospectively sealed contribution-scaled action metric on two original and two
holdout directions, and uses the unchanged 5e-10 threshold for route selection.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from scripts.audit import _d080f_frozen_full_build as core
from scripts.audit import d080f_retained_full_build as legacy
from scripts.audit._d080e_prepared_jvp import evaluate_prepared_c_only_rhs_jvp
from scripts.audit._d080f_action_metric import matrix_action_block_residual

_ORIGINAL_ASSEMBLE = legacy.assemble_sealed_static_jacobian
_METROLOGY: dict[str, object] = {}


def _holdout_directions(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Two deterministic directions absent from the v1/diagnostic runs."""

    x = np.linspace(-1.0, 1.0, int(order), dtype=np.float64)
    raw = (
        np.stack(
            (
                np.cos(2.17 * x + 0.31),
                0.43 * np.sin(1.61 * x - 0.27),
                np.exp(-0.63 * (x - 0.18) ** 2) - 0.52,
            )
        ),
        np.stack(
            (
                np.sin(2.43 * x - 0.17),
                np.cos(1.91 * x + 0.41) - 0.23 * x,
                0.29 - 0.71 * x + 0.36 * x * x,
            )
        ),
    )
    output: list[np.ndarray] = []
    for value in raw:
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("invalid D-080F holdout direction")
        output.append(np.asarray(value / norm, dtype=np.float64))
    return output[0], output[1]


def _audited_assemble(*args, **kwargs):
    result = _ORIGINAL_ASSEMBLE(*args, **kwargs)
    if args:
        sealed = args[0]
    else:
        sealed = kwargs["sealed"]
    prepared = sealed.prepared
    order = prepared.raw_grid.order

    directions = tuple(core._deterministic_probe_directions(order)) + tuple(
        _holdout_directions(order)
    )
    labels = (
        "original_direction_0",
        "original_direction_1",
        "holdout_direction_0",
        "holdout_direction_1",
    )
    reports: list[dict[str, object]] = []
    maximum_action = 0.0
    maximum_forward = 0.0
    for label, direction in zip(labels, directions):
        direct = evaluate_prepared_c_only_rhs_jvp(prepared, direction)
        full_direction = np.concatenate((direction.ravel(), [0.0, 0.0]))
        candidate = result.jacobian @ full_direction
        forward = legacy.rhs_block_relative(candidate, direct.jvp, order)
        action = matrix_action_block_residual(
            jacobian=result.jacobian,
            direction=full_direction,
            reference_action=direct.jvp,
            order=order,
        )
        maximum_action = max(maximum_action, action.maximum)
        maximum_forward = max(maximum_forward, forward)
        reports.append(
            {
                "label": label,
                "legacy_forward_relative_residual": float(forward),
                "action_metric": asdict(action),
            }
        )

    _METROLOGY.clear()
    _METROLOGY.update(
        {
            "legacy_internal_forward_maximum": float(
                result.maximum_prepared_action_residual
            ),
            "four_direction_forward_maximum": float(maximum_forward),
            "four_direction_action_maximum": float(maximum_action),
            "reports": reports,
        }
    )
    return replace(
        result,
        maximum_prepared_action_residual=float(maximum_action),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_directory() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--out-dir") + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("D-080F v2 requires --out-dir") from error


def _postprocess(output: Path) -> None:
    receipt_path = output / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "rabbit.d080f.retained_full_build.v2"
    receipt["equivalence_metrology"] = {
        "amendment": "BD632_D080F_EQUIVALENCE_METROLOGY_AMENDMENT_2026-09-02.md",
        "preserved_failed_run": 33592265710,
        "localization_run": 33593370075,
        "threshold_unchanged": 5.0e-10,
        "route_uses": "maximum contribution-scaled action residual across two original and two preregistered holdout directions",
        "legacy_forward_metric_is_diagnostic_only": True,
        **_METROLOGY,
    }
    correctness = receipt["correctness"]
    correctness["maximum_prepared_action_residual"] = float(
        _METROLOGY["four_direction_action_maximum"]
    )
    correctness["maximum_equivalence_residual"] = max(
        correctness["maximum_prepared_action_residual"],
        correctness["maximum_serial_basis_column_residual"],
    )
    correctness["legacy_forward_relative_maximum"] = float(
        _METROLOGY["four_direction_forward_maximum"]
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reports = _METROLOGY["reports"]
    labels = [str(item["label"]) for item in reports]
    forward = [
        max(float(item["legacy_forward_relative_residual"]), np.finfo(float).tiny)
        for item in reports
    ]
    action = [
        max(float(item["action_metric"]["maximum"]), np.finfo(float).tiny)
        for item in reports
    ]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    figure = plt.figure(figsize=(8.2, 4.8), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(x - width / 2.0, forward, width, label="legacy forward-relative")
    axis.bar(x + width / 2.0, action, width, label="contribution-scaled action")
    axis.axhline(5.0e-10, linestyle="--", linewidth=1.0, label="frozen gate")
    axis.set_yscale("log")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("dimensionless residual")
    axis.set_title("D-080F dense-action equivalence metrology")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    figure.savefig(output / "action_metrology.png", dpi=180)
    plt.close(figure)

    summary_path = output / "probe_summary.md"
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Cancellation-aware metrology amendment\n\n"
            f"- legacy four-direction forward maximum: `{_METROLOGY['four_direction_forward_maximum']:.16e}`\n"
            f"- four-direction action maximum: `{_METROLOGY['four_direction_action_maximum']:.16e}`\n"
            "- numerical threshold: unchanged at `5e-10`\n"
            "- holdout directions: two, frozen before this rerun\n"
            "- selected basis-column metric: unchanged\n"
        )

    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    legacy.assemble_sealed_static_jacobian = _audited_assemble
    legacy.main()
    if not _METROLOGY:
        raise RuntimeError("D-080F v2 metrology was not populated")
    _postprocess(_output_directory())


if __name__ == "__main__":
    main()
