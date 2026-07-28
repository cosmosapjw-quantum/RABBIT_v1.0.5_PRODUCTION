#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from _common import lease_path, load_json, read_stdin_json, repo_root

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

    # Run-identity lease (D-058): resolve the run this agent was bound to at
    # SubagentStart instead of trusting the mutable global ACTIVE_RUN pointer.
    event_agent_id = str(event.get("agent_id") or "")
    lease_file = lease_path(harness, event_agent_id) if event_agent_id else None
    lease = load_json(lease_file, None) if lease_file is not None else None
    if lease is None and lease_file is not None and lease_file.exists():
        block(
            "Run lease for this agent exists but cannot be parsed; refusing to "
            "resolve the run via the mutable ACTIVE_RUN pointer."
        )
        return

    active_run = ""
    if active_path.exists():
        active_run = active_path.read_text(encoding="utf-8").strip()

    # Outside an explicitly active harness run *and* with no Start-time lease,
    # do not impose the result envelope.
    if not active_run and lease is None:
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

    if lease is not None:
        if (
            not isinstance(lease, dict)
            or str(lease.get("agent_id") or "") != event_agent_id
            or not ASSIGNMENT_ID_RE.fullmatch(str(lease.get("run_id") or ""))
        ):
            block(
                "Run lease for this agent is malformed; refusing to resolve the "
                "run via the mutable ACTIVE_RUN pointer."
            )
            return
        bound_run = str(lease["run_id"])
    else:
        bound_run = active_run

    run_dir = harness / "runs" / bound_run
    assignment_path = run_dir / "assignments" / f"{assignment_id}.json"
    try:
        assignment_raw = assignment_path.read_bytes()
        assignment = json.loads(assignment_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        assignment = None
    if not isinstance(assignment, dict):
        block(f"Registered assignment does not exist: {assignment_path.relative_to(root)}")
        return
    if str(assignment.get("run_id")) != bound_run:
        block("Registered assignment run_id does not match the bound run identity.")
        return
    # Single read: the digest and the parsed contract come from the same bytes.
    recomputed_sha = "sha256:" + hashlib.sha256(assignment_raw).hexdigest()
    if lease is not None:
        sealed = (lease.get("assignment_digests") or {}).get(assignment_id)
        if sealed is None:
            block(
                f"Assignment {assignment_id!r} was not registered when this agent "
                "started (missing from the Start-time lease)."
            )
            return
        if sealed != recomputed_sha:
            block(
                "Assignment file changed after SubagentStart: the Start-time sealed "
                "digest does not match the current bytes."
            )
            return
    event_agent_type = str(event.get("agent_type") or "")
    expected_runtime_type = assignment_runtime_agent_type(assignment)
    if event_agent_type != expected_runtime_type:
        block(
            "SubagentStop runtime agent_type does not match the registered "
            "runtime_agent_type."
        )
        return
    if not event_agent_id:
        block("SubagentStop event omitted agent_id; result identity cannot be verified.")
        return

    index = load_json(harness / "context" / "CONTEXT_INDEX.json", {}) or {}
    current_version = str(index.get("context_version", "UNBUILT"))
    assignment_errors = validate_assignment_contract(
        assignment,
        expected_run_id=bound_run,
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
        expected_run_id=bound_run,
        expected_context_version=current_version,
        expected_agent_id=event_agent_id,
        expected_assignment_sha256=recomputed_sha,
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

    # Acceptance: consume the lease so a reused agent_id starts clean. Blocked
    # stops above leave the lease in place for the fail-closed retry.
    if lease_file is not None:
        lease_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
