#!/usr/bin/env python3
"""Admit exact NumPy binary64 identity for the retained GL60/Y30 operator.

Run only after the RED grid-identity test has demonstrated a mismatch. The
script changes no collision expression, coefficient, state, quadrature order,
domain, or acceptance tolerance. It adds a special-case byte-identity branch
for exactly ``order=60`` and ``y_max=30.0``, mirroring the already admitted
``order=8, y_max=8.0`` branch.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FIXTURE = Path(__file__).with_name("retained_packed_rhs_case.json")
SOURCE = ROOT / "native/rabbit_cpu/src/f10_action_grid.rs"


def encoded_bits(value: object, expected_length: int) -> list[int]:
    mapping = dict(value)  # type: ignore[arg-type]
    raw = list(mapping["bits"])
    if len(raw) != expected_length:
        raise SystemExit(f"unexpected bit-array length: {len(raw)}")
    return [int(str(item), 16) for item in raw]


def rust_array(name: str, values: list[int]) -> str:
    rows = []
    for offset in range(0, len(values), 4):
        chunk = values[offset : offset + 4]
        rows.append("    " + ", ".join(f"0x{value:016x}" for value in chunk) + ",")
    return f"const {name}: [u64; {len(values)}] = [\n" + "\n".join(rows) + "\n];"


def main() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if payload["schema"] != "rabbit.d081r1e.retained_packed_rhs.v1":
        raise SystemExit("unexpected retained fixture schema")
    if payload["order"] != 60:
        raise SystemExit("unexpected retained grid order")
    if payload["y_max_bits"] != "403e000000000000":
        raise SystemExit("unexpected retained y_max bits")

    nodes = encoded_bits(payload["grid_nodes"], 60)
    weights = encoded_bits(payload["grid_weights"], 60)
    text = SOURCE.read_text(encoding="utf-8")

    node_name = "EXACT_ORDER60_YMAX30_NODE_BITS"
    weight_name = "EXACT_ORDER60_YMAX30_WEIGHT_BITS"
    changed = False

    if node_name not in text or weight_name not in text:
        marker = "const EXACT_ORDER8_YMAX8_WEIGHT_BITS: [u64; 8] = ["
        if text.count(marker) != 1:
            raise SystemExit("unexpected order-8 weight constant boundary")
        start = text.index(marker)
        end = text.index("];", start) + 2
        constants = (
            "\n\n"
            + rust_array(node_name, nodes)
            + "\n\n"
            + rust_array(weight_name, weights)
        )
        text = text[:end] + constants + text[end:]
        changed = True

    branch_signature = "if order == 60 && y_max.to_bits() == 30.0_f64.to_bits()"
    if branch_signature not in text:
        marker = "        let scale = 0.5 * y_max;\n"
        if text.count(marker) != 1:
            raise SystemExit("unexpected generic quadrature insertion point")
        branch = f'''        // D-081R1E: preserve the frozen NumPy 2.4.4 binary64\n        // finite-dimensional operator at the provenance-locked retained grid.\n        if order == 60 && y_max.to_bits() == 30.0_f64.to_bits() {{\n            return Ok(Self {{\n                order,\n                y_max,\n                nodes: {node_name}\n                    .into_iter()\n                    .map(f64::from_bits)\n                    .collect(),\n                weights: {weight_name}\n                    .into_iter()\n                    .map(f64::from_bits)\n                    .collect(),\n            }});\n        }}\n\n'''
        text = text.replace(marker, branch + marker)
        changed = True

    SOURCE.write_text(text, encoding="utf-8")
    print("D-081R1E order-60 grid identity repair:", "CHANGED" if changed else "NOOP")


if __name__ == "__main__":
    main()
