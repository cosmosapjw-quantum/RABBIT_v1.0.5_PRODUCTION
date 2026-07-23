#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from _common import load_json, read_stdin_json, repo_root

HARNESS_SCRIPTS = Path(__file__).resolve().parents[2] / ".agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_SCRIPTS))

from _harness import (  # noqa: E402
    assignment_runtime_agent_type,
    validate_assignment_contract,
    validate_assignment_resource_hashes,
    validate_result_contract,
)

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
    event_agent_type = str(event.get("agent_type") or "")
    expected_runtime_type = assignment_runtime_agent_type(assignment)
    if event_agent_type != expected_runtime_type:
        block(
            "SubagentStop runtime agent_type does not match the registered "
            "runtime_agent_type."
        )
        return
    event_agent_id = str(event.get("agent_id") or "")
    if not event_agent_id:
        block("SubagentStop event omitted agent_id; result identity cannot be verified.")
        return

    index = load_json(harness / "context" / "CONTEXT_INDEX.json", {}) or {}
    current_version = str(index.get("context_version", "UNBUILT"))
    assignment_errors = validate_assignment_contract(
        assignment,
        expected_run_id=active_run,
        expected_context_version=current_version,
        role_files=index.get("role_files", {}),
    )
    if assignment_errors:
        block("Registered assignment contract is invalid: " + "; ".join(assignment_errors))
        return
    resource_errors = validate_assignment_resource_hashes(
        root,
        assignment,
        role_files=index.get("role_files", {}),
    )
    if resource_errors:
        block("Registered assignment resources are stale: " + "; ".join(resource_errors))
        return
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
    contract_errors = validate_result_contract(
        result,
        assignment,
        expected_run_id=active_run,
        expected_context_version=current_version,
        expected_agent_id=event_agent_id,
        expected_assignment_sha256=(
            "sha256:" + hashlib.sha256(assignment_path.read_bytes()).hexdigest()
        ),
    )
    if str(result.get("status")) != str(envelope["status"]):
        contract_errors.append(
            "status mismatch between result artifact and HARNESS_RESULT envelope"
        )
    if contract_errors:
        retry_note = (
            " This is a retry after an earlier blocked stop; validation remains fail-closed."
            if bool(event.get("stop_hook_active"))
            else ""
        )
        block(
            "Result artifact violates RESULT_ENVELOPE: "
            + "; ".join(contract_errors)
            + retry_note
        )
        return


if __name__ == "__main__":
    main()
