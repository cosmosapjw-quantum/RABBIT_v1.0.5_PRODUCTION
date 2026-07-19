#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

from _common import load_json, read_stdin_json, repo_root

MARKER_RE = re.compile(r"(?:^|\n)HARNESS_RESULT:\s*(\{[^\n]+\})\s*\Z")
ASSIGNMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
VALID_STATUS = {"pass", "fail", "inconclusive", "error"}


def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))


def main() -> None:
    event = read_stdin_json()
    root = repo_root()
    harness = root / ".agent-harness"
    active_path = harness / "ACTIVE_RUN"

    # Outside an explicitly active harness run, do not impose the result envelope.
    if not active_path.exists() or not active_path.read_text(encoding="utf-8").strip():
        return

    if bool(event.get("stop_hook_active")):
        return

    message = str(event.get("last_assistant_message") or "")
    match = MARKER_RE.search(message)
    if not match:
        block(
            "Before stopping, write the assignment result artifact and finish with exactly one line: "
            "HARNESS_RESULT: {\"assignment_id\":\"...\",\"context_version\":\"...\","
            "\"status\":\"pass|fail|inconclusive|error\",\"result_path\":\"...\"}"
        )
        return

    try:
        envelope = json.loads(match.group(1))
    except json.JSONDecodeError:
        block("HARNESS_RESULT is not valid single-line JSON. Correct it before stopping.")
        return

    required = {"assignment_id", "context_version", "status", "result_path"}
    missing = sorted(required - set(envelope))
    if missing:
        block(f"HARNESS_RESULT is missing required fields: {', '.join(missing)}")
        return

    assignment_id = str(envelope["assignment_id"])
    if not ASSIGNMENT_ID_RE.fullmatch(assignment_id):
        block("HARNESS_RESULT assignment_id has an invalid form.")
        return

    active_run = active_path.read_text(encoding="utf-8").strip()
    run_dir = harness / "runs" / active_run
    assignment_path = run_dir / "assignments" / f"{assignment_id}.json"
    assignment = load_json(assignment_path, None)
    if not isinstance(assignment, dict):
        block(f"Registered assignment does not exist: {assignment_path.relative_to(root)}")
        return
    if str(assignment.get("run_id")) != active_run:
        block("Registered assignment run_id does not match ACTIVE_RUN.")
        return

    index = load_json(harness / "context" / "CONTEXT_INDEX.json", {}) or {}
    current_version = str(index.get("context_version", "UNBUILT"))
    if str(envelope["context_version"]) != current_version:
        block(
            f"Stale result context version {envelope['context_version']!r}; current version is "
            f"{current_version!r}. Reconcile or explicitly report an error result against the current context."
        )
        return
    if str(assignment.get("context_version")) != current_version:
        block(
            f"Registered assignment {assignment_id!r} is stale for context "
            f"{current_version!r}."
        )
        return

    if str(envelope["status"]) not in VALID_STATUS:
        block(f"Invalid result status {envelope['status']!r}; use one of {sorted(VALID_STATUS)}")
        return

    result_path = Path(str(envelope["result_path"]))
    if result_path.is_absolute():
        block("result_path must be repository-relative.")
        return
    resolved = (root / result_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        block("result_path escapes the repository root.")
        return
    declared_result_path = str(assignment.get("result_path") or "")
    if str(result_path) != declared_result_path:
        block(
            f"HARNESS_RESULT result_path {str(result_path)!r} does not match the registered "
            f"path {declared_result_path!r}."
        )
        return
    expected_result = (run_dir / "results" / f"{assignment_id}.json").resolve()
    if resolved != expected_result:
        block("Registered result_path is not the assignment's unique active-run result file.")
        return
    if not resolved.is_file():
        block(f"Declared result artifact does not exist: {result_path}")
        return

    result = load_json(resolved, None)
    if not isinstance(result, dict):
        block("Declared result artifact is not a JSON object.")
        return
    expected_fields = {
        "run_id": active_run,
        "assignment_id": assignment_id,
        "context_version": current_version,
        "status": str(envelope["status"]),
        "result_path": declared_result_path,
        "agent_type": str(assignment.get("agent_type")),
    }
    mismatches = [
        f"{key}: expected {expected!r}, got {result.get(key)!r}"
        for key, expected in expected_fields.items()
        if str(result.get(key)) != expected
    ]
    if mismatches:
        block("Result artifact does not match its assignment/envelope: " + "; ".join(mismatches))
        return


if __name__ == "__main__":
    main()
