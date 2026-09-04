#!/usr/bin/env python3
"""Register the bounded D-081R1F0 Rust modules idempotently.

The derivative implementation is kept in new source files. This script makes
only the three module-registration edits required for compilation; the sealed
workflow runs formatting and the full verification suite before publishing the
resulting source mutation.
"""

from __future__ import annotations

from pathlib import Path


SELF = Path("native/rabbit_cpu/src/f10_self_action.rs")
ELECTRON = Path("native/rabbit_cpu/src/f10_electron_action.rs")
LIB = Path("native/rabbit_cpu/src/lib.rs")


def insert_once(path: Path, anchor: str, insertion: str) -> None:
    text = path.read_text(encoding="utf-8")
    if insertion.strip() in text:
        return
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(f"{path}: expected one anchor, found {count}")
    path.write_text(text.replace(anchor, anchor + insertion, 1), encoding="utf-8")


def main() -> None:
    insert_once(
        SELF,
        "#![cfg_attr(not(test), allow(dead_code))]\n",
        "\npub(crate) mod c_jvp;\n",
    )
    insert_once(
        ELECTRON,
        "#![cfg_attr(not(test), allow(dead_code))]\n",
        "\npub(crate) mod c_jvp;\n",
    )
    insert_once(
        LIB,
        "mod f10_action_spectral;\n",
        "mod f10_action_tangent;\n",
    )
    insert_once(
        LIB,
        "mod f10_combined_action;\n",
        "mod f10_combined_action_jvp;\n",
    )
    insert_once(
        LIB,
        "mod f10_packed_rhs;\n",
        "mod f10_packed_rhs_jvp;\n",
    )
    print("D-081R1F0 module registration: READY")


if __name__ == "__main__":
    main()
