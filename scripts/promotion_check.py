#!/usr/bin/env python3
"""Foundation F13 — Forbidden -> Permitted promotion check.

Per the v2 upgrade plan: a Forbidden Claim flips to Permitted *only* when
every test in its ClaimGate's required_test_node_ids passes.  This script
performs the check and emits a PR-ready diff.  No auto-mutation.

Usage:
    python scripts/promotion_check.py --status        # ledger summary
    python scripts/promotion_check.py --gate ID       # status of a single gate
    python scripts/promotion_check.py --diff          # emit diff for green gates
    python scripts/promotion_check.py --strict        # exit nonzero if any
                                                       # documentation-only
                                                       # gate is missing tests

The diff output is printed to stdout and not applied to disk; a human
reviews the diff and lands it via a normal PR.

This script intentionally does not modify SUPPORTED_CAPABILITIES.md or
feature_capabilities.py.  The deliberate human-in-the-loop step is part of
the "no claim drift" invariant.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from xml.etree import ElementTree as ET

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SC_PATH = REPO_ROOT / "SUPPORTED_CAPABILITIES.md"

# Hardening (external audit BD601, 2026-06-30):
#  * a gate must never go green on a SKIPPED required node -- pytest exits 0 on skip, so the old
#    `returncode == 0 -> pass` check counted skips as passes. We now parse JUnit XML and require
#    every selected testcase to be a genuine PASS (no skipped/failure/error).
#  * the gate runner must not inherit a marker filter (PYTEST_ADDOPTS / ini addopts) that could
#    deselect a required node out of collection -- that would silently turn RED-fail into RED-missing
#    or, worse, hide a node entirely. We clear PYTEST_ADDOPTS and override ini addopts to empty.
#  * every pytest subprocess gets a timeout so a hung gate fails closed instead of stalling the run.
_PYTEST_TIMEOUT_COLLECT_S = 900
_PYTEST_TIMEOUT_RUN_S = 1800


def _clean_env() -> dict:
    """Subprocess env with PYTEST_ADDOPTS cleared so a marker filter cannot deselect required nodes."""
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = ""
    return env


def _junit_all_passed(xml_path: pathlib.Path) -> bool:
    """True iff the JUnit report has at least one testcase and EVERY testcase is a genuine pass
    (no <skipped>, <failure>, or <error> child). A skipped required node is therefore NOT a pass."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError, OSError):
        return False
    cases = list(root.iter("testcase"))
    if not cases:
        return False
    for case in cases:
        for child in case:
            if child.tag in ("skipped", "failure", "error"):
                return False
    return True

# Local import of the gate registry.
sys.path.insert(0, str(REPO_ROOT / "src"))
from rabbit.config.claim_gates import ALL_GATES, ClaimGate, gates_by_id  # noqa: E402


@dataclass
class GateStatus:
    gate: ClaimGate
    collectible: list[str]
    missing: list[str]
    passed: list[str]
    failed: list[str]

    @property
    def is_green(self) -> bool:
        return (
            self.gate.required_test_node_ids != ()
            and not self.missing
            and not self.failed
        )

    @property
    def is_documentation_only(self) -> bool:
        return self.gate.required_test_node_ids == ()


def _collect_node_ids() -> set[str]:
    """Return every pytest node id currently collectible (with addopts neutralized so a marker
    filter cannot hide required nodes)."""
    try:
        out = subprocess.run(
            ["venv/bin/python", "-m", "pytest", "--collect-only", "-q", "--no-header",
             "-o", "addopts="],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_clean_env(),
            timeout=_PYTEST_TIMEOUT_COLLECT_S,
        )
    except subprocess.TimeoutExpired:
        print("collect-only timed out; treating all nodes as missing.", file=sys.stderr)
        return set()
    nodes: set[str] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("tests/") and "::" in line:
            nodes.add(line)
        elif line.startswith("tests/") and line.endswith(".py"):
            nodes.add(line)
    return nodes


def _check_gate(gate: ClaimGate, all_nodes: set[str]) -> GateStatus:
    collectible: list[str] = []
    missing: list[str] = []
    passed: list[str] = []
    failed: list[str] = []

    for node in gate.required_test_node_ids:
        # Allow file-level references: "tests/foo.py" matches any
        # collectible node under that file.
        if "::" not in node:
            file_match = any(n.startswith(node + "::") or n == node for n in all_nodes)
            if file_match:
                collectible.append(node)
            else:
                missing.append(node)
        else:
            if node in all_nodes:
                collectible.append(node)
            else:
                missing.append(node)

    # Run the tests that are collectible to determine pass/fail. A gate goes green ONLY if every
    # selected testcase is a genuine pass -- a SKIPPED required node must NOT count as a pass
    # (pytest exits 0 on skip, so the returncode alone is insufficient; we parse JUnit XML).
    if collectible:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
            junit_path = pathlib.Path(tf.name)
        try:
            try:
                subprocess.run(
                    ["venv/bin/python", "-m", "pytest", *collectible, "--tb=no", "-q",
                     "-o", "addopts=", f"--junit-xml={junit_path}"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    env=_clean_env(),
                    timeout=_PYTEST_TIMEOUT_RUN_S,
                )
                all_passed = _junit_all_passed(junit_path)
            except subprocess.TimeoutExpired:
                all_passed = False  # a hung gate fails closed
        finally:
            junit_path.unlink(missing_ok=True)
        if all_passed:
            passed = list(collectible)
        else:
            failed = list(collectible)

    return GateStatus(
        gate=gate, collectible=collectible, missing=missing, passed=passed, failed=failed
    )


def _emit_diff(green: list[GateStatus]) -> str:
    """Produce a unified-diff-style preview for the SUPPORTED_CAPABILITIES.md
    edits implied by the green gates.  This is informational only; no file
    is written."""
    if not green:
        return "# No gates green; nothing to promote.\n"

    lines: list[str] = ["# === PROMOTION DIFF (informational only) ===\n"]
    lines.append(f"# {len(green)} gate(s) ready to flip Forbidden -> Permitted.\n")
    lines.append(f"# File:  {SC_PATH.relative_to(REPO_ROOT)}\n\n")
    lines.append("# In the <!-- BEGIN:CLAIMS --> ... <!-- END:CLAIMS --> block:\n")
    for status in green:
        g = status.gate
        lines.append(f"# --- gate: {g.claim_id} ---\n")
        lines.append(f"-  - {g.forbidden_text}\n")
        lines.append(f"+  - {g.permitted_text}    # promoted by F13 gate\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true",
                        help="Print a ledger summary of every gate.")
    parser.add_argument("--gate", default=None,
                        help="Run a single gate by claim_id.")
    parser.add_argument("--diff", action="store_true",
                        help="Emit PR-ready diff for green gates.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero on any non-green non-documentation gate.")
    args = parser.parse_args()

    print(f"Discovering pytest node IDs...", file=sys.stderr)
    all_nodes = _collect_node_ids()
    print(f"Collected {len(all_nodes)} node(s).", file=sys.stderr)

    targets = ALL_GATES
    if args.gate:
        try:
            targets = (gates_by_id()[args.gate],)
        except KeyError:
            print(f"Unknown gate: {args.gate!r}", file=sys.stderr)
            return 2

    statuses: list[GateStatus] = []
    for g in targets:
        if not g.required_test_node_ids:
            statuses.append(GateStatus(gate=g, collectible=[], missing=[], passed=[], failed=[]))
            continue
        statuses.append(_check_gate(g, all_nodes))

    if args.status or not (args.diff or args.strict):
        print()
        print(f"{'Gate':40s}  {'State':14s}  Tests")
        print("-" * 90)
        for s in statuses:
            if s.is_documentation_only:
                state = "doc-only"
            elif s.is_green:
                state = "GREEN"
            elif s.failed:
                state = "RED-fail"
            elif s.missing:
                state = "RED-missing"
            else:
                state = "RED"
            n_total = len(s.gate.required_test_node_ids)
            n_pass = len(s.passed)
            print(f"{s.gate.claim_id:40s}  {state:14s}  {n_pass}/{n_total}")
        print()

    if args.diff:
        green = [s for s in statuses if s.is_green]
        print(_emit_diff(green))

    if args.strict:
        non_green = [s for s in statuses if not s.is_green and not s.is_documentation_only]
        if non_green:
            print(f"strict mode: {len(non_green)} gate(s) not green.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
