#!/usr/bin/env python3
"""Canonical sealed-hash verifier for harness assignments (D-058).

Usage: python3 .agent-harness/scripts/verify_assignment.py <assignment.json>

Prints the three digests an agent must copy verbatim into `spawn_contract`,
recomputing each the canonical way:
  - assignment_sha256: raw sha256 of the assignment file bytes
  - review_role_sha256: path-aware aggregate over `review_role_files`
    (relpath\\0bytes\\0 per file, in list order) via _harness.hash_files
  - result_template_sha256: raw sha256 of the `result_template` file bytes
Exits 0 iff every sealed hash matches (`VERIFY: PASS`), 1 otherwise.

Advisory only: `VERIFY: PASS` is not authorization. The SubagentStop hook
independently recomputes and enforces every digest against the registered
in-run assignment; a PASS here on a mislabeled or out-of-run file does not
make its result admissible.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from _harness import hash_files, root


def fail(reason: str) -> int:
    print(f"error: {reason}")
    print("VERIFY: FAIL")
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: verify_assignment.py <assignment.json>")
    repo = root()
    arg = Path(sys.argv[1])
    path = arg if arg.is_absolute() else (repo / arg)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return fail(f"cannot read assignment: {exc}")
    try:
        assignment = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return fail(f"assignment is not valid JSON: {exc}")
    if not isinstance(assignment, dict):
        return fail("assignment is not a JSON object")

    print(f"assignment_sha256=sha256:{hashlib.sha256(raw).hexdigest()}")
    try:
        schema_version = int(assignment.get("schema_version", 1))
    except (TypeError, ValueError):
        return fail(f"schema_version is not numeric: {assignment.get('schema_version')!r}")
    if schema_version < 2:
        print("schema_version=1: no sealed resource hashes; assignment_sha256 only")
        print("VERIFY: PASS")
        return 0

    ok = True
    role_files = [str(x) for x in assignment.get("review_role_files") or []]
    if not role_files:
        print("review_role_files=MISSING FAIL")
        ok = False
    else:
        try:
            aggregate, _ = hash_files(repo, role_files)
        except OSError as exc:
            return fail(f"cannot read review_role_files: {exc}")
        role_ok = str(assignment.get("review_role_sha256")) == f"sha256:{aggregate}"
        print(f"review_role_sha256=sha256:{aggregate} {'PASS' if role_ok else 'FAIL'}")
        ok = ok and role_ok

    template_rel = str(assignment.get("result_template") or "")
    if not template_rel:
        print("result_template=MISSING FAIL")
        ok = False
    else:
        try:
            template_raw = (repo / template_rel).read_bytes()
        except OSError as exc:
            return fail(f"cannot read result_template: {exc}")
        template_hash = hashlib.sha256(template_raw).hexdigest()
        template_ok = (
            str(assignment.get("result_template_sha256")) == f"sha256:{template_hash}"
        )
        print(
            f"result_template_sha256=sha256:{template_hash} "
            f"{'PASS' if template_ok else 'FAIL'}"
        )
        ok = ok and template_ok

    print(f"VERIFY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
