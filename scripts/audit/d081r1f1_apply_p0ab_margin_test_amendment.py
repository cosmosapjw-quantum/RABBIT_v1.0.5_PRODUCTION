#!/usr/bin/env python3
"""Replay the pinned P0AB metrology amendment and remove its retired helper.

The full reference-conditioned test transformation is pinned at commit
42c2375516eec56b8b04142df18d3fad13012837. This bounded wrapper executes that
exact historical script and then removes `contribution_scaled`, which ceased to
be used after the invariant gates were rewritten. No production source,
physics equation, numerical threshold, branch policy, or oracle is changed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


PINNED_COMMIT = "42c2375516eec56b8b04142df18d3fad13012837"
SCRIPT_PATH = "scripts/audit/d081r1f1_apply_p0ab_margin_test_amendment.py"
TEST = Path("native/rabbit_cpu/src/f10_tgamma_adversarial_repair_tests.rs")


def run_pinned_amendment() -> None:
    source = subprocess.check_output(
        ["git", "show", f"{PINNED_COMMIT}:{SCRIPT_PATH}"],
        text=True,
    )
    namespace = {
        "__name__": "__main__",
        "__file__": f"{PINNED_COMMIT}:{SCRIPT_PATH}",
    }
    exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)


def remove_retired_helper() -> bool:
    text = TEST.read_text(encoding="utf-8")
    start_marker = "fn contribution_scaled("
    count = text.count(start_marker)
    if count == 0:
        return False
    if count != 1:
        raise SystemExit(f"retired helper: expected at most one match, found {count}")
    start = text.index(start_marker)
    end = text.index("\n}\n", start) + len("\n}\n")
    while end < len(text) and text[end] == "\n":
        end += 1
    TEST.write_text(text[:start] + text[end:], encoding="utf-8")
    return True


def main() -> None:
    run_pinned_amendment()
    removed = remove_retired_helper()
    print(
        "D-081R1F1 P0AB reference-conditioned invariant amendment: "
        + ("CHANGED_AND_RETIRED_HELPER_REMOVED" if removed else "APPLIED_NO_HELPER")
    )


if __name__ == "__main__":
    main()
