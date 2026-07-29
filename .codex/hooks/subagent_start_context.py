#!/usr/bin/env python3
"""SubagentStart preflight: inject canonical context and seal a run-identity lease.

Receipt visibility (D-065 obligation 2, round-4 finding): the obligation reads
"make receipt/lease creation failure a hard Start failure." The lease half was
fully met (see the run-identity lease block below); the receipt half was not
implemented at all -- this hook never referenced `.agent-harness/admissions/`.
A run with zero receipts minted anywhere still produced `Hook preflight: PASS`
and `systemMessage: null`. Receipt failure was only ever caught parent-side
(`admit_agent.py` exits 1 before the agent is even spawned) or Stop-side
(`subagent_stop_validate.py`, potentially hours later). That was fail-closed
end-to-end, but it did not discharge the Start half of the obligation, and
Start *can* check, because receipts carry `expected_agent_id`.

THE RULE -- decided deliberately, not left to convention: absence of a receipt
for this `agent_id` is reported but is NOT fatal, because not every spawned
agent is a registered assignment; the parent runs plenty of unregistered
helper agents that were never meant to hold one, and Start hard-failing on all
of them would make the common case indistinguishable from the dangerous one.
But a receipt that names this exact `agent_id` and is unusable -- malformed,
unreadable, already `consumed`, bound to a different run, or ambiguous because
more than one receipt claims the same `expected_agent_id` -- IS a hard Start
failure, because in each of those cases something concrete and identifiable
went wrong for this specific agent_id, and staying silent about it is exactly
the failure mode the round-4 review flagged. Silence must never be the
permissive default: whichever of the two states holds, this hook says so, in
the injected context, load-bearing in `Hook preflight: PASS|FAIL`.

Concretely, at Start this hook scans `.agent-harness/admissions/<ACTIVE_RUN>/`
(read-only -- it never claims, locks, or mutates a receipt; consumption stays
at Stop, under the `O_EXCL` claim in subagent_stop_validate.py) for receipts
whose `expected_agent_id` equals this event's `agent_id`:

  * No active run, no agent_id, or an agent_id that fails the same safe-key
    regex the run lease uses -> the check is skipped and reported as "not
    checked", mirroring the existing lease behaviour for the same inputs
    exactly. A deliberate no-op, not a silent one: the reason is always
    printed.
  * Directory missing (no receipts minted yet in this run) or present with
    zero matches for this agent_id -> reported as "none found", not fatal.
  * Exactly one matching receipt, state "open", its own `run_id` field equal
    to the active run -> reported as admitted, not fatal.
  * Exactly one matching receipt but state != "open", or its `run_id` field
    disagrees with the active run -> hard FAIL. The receipt exists and names
    this agent but cannot admit it.
  * More than one receipt claims the same `expected_agent_id`, or any receipt
    file in the run's admission directory (not only ones that turn out to
    match this agent_id) cannot be listed, read, or parsed as a JSON object
    -> hard FAIL. A parse failure on some *other* assignment's receipt is
    folded in too, deliberately: this hook cannot prove that an unparseable
    file was not meant for this agent_id, and an unprovable negative must not
    default to PASS. Ambiguity is a FAIL, not a skip.

This mirrors, but does not duplicate the authority of, subagent_stop_validate.py.
Start reports what it can see before the agent has produced a result (there is
no HARNESS_RESULT envelope yet to check `admission_proof` against); Stop still
performs the authoritative token/attribution check and the single-use consume.
Start must never contradict Stop's later verdict, so it only reads.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import (
    emit_additional_context,
    lease_path,
    load_json,
    read_stdin_json,
    repo_root,
    write_json_atomic,
)

HARNESS_SCRIPTS = Path(__file__).resolve().parents[2] / ".agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_SCRIPTS))

from _harness import ADMISSION_KEY_RE  # noqa: E402


def read_bounded(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"[missing file: {path}]"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated at {max_chars} characters; read the file directly for the rest]"


def context_integrity(root: Path, index: dict[str, Any], pack_path: Path) -> list[str]:
    errors: list[str] = []
    digest = hashlib.sha256()
    file_hashes = index.get("file_hashes", {})
    for rel in index.get("shared_files", []):
        path = root / str(rel)
        try:
            data = path.read_bytes()
        except OSError:
            errors.append(f"missing shared context file: {rel}")
            continue
        actual_file_hash = hashlib.sha256(data).hexdigest()
        if str(file_hashes.get(rel)) != actual_file_hash:
            errors.append(f"stale shared context hash: {rel}")
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    version = str(index.get("context_version", "UNBUILT"))
    if digest.hexdigest() != version:
        errors.append("context_version does not match the shared files")
    try:
        first_lines = "\n".join(pack_path.read_text(encoding="utf-8").splitlines()[:6])
    except OSError:
        errors.append("canonical context pack is unreadable")
    else:
        if f"Context version: `{version}`" not in first_lines:
            errors.append("canonical context pack carries a different version")
    return errors


def active_run_id(harness: Path) -> str:
    active_path = harness / "ACTIVE_RUN"
    if not active_path.is_file():
        return ""
    return active_path.read_text(encoding="utf-8").strip()


def list_admission_matches(
    admissions_dir: Path, agent_id: str
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """Read-only scan for receipts in ``admissions_dir`` bound to ``agent_id``.

    Returns ``(matches, errors)``. ``errors`` accumulates every read/parse
    failure encountered while scanning the directory, not only ones for files
    that turn out to match ``agent_id`` -- see the module docstring for why an
    unparseable receipt for a *different* assignment still has to fail closed
    here.
    """
    matches: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    if admissions_dir.is_symlink():
        errors.append(f"admission directory path is a symlink: {admissions_dir.name}")
        return matches, errors
    try:
        entries = sorted(admissions_dir.iterdir())
    except FileNotFoundError:
        return matches, errors
    except OSError as exc:
        errors.append(
            f"admission receipts directory is unreadable ({exc.__class__.__name__})"
        )
        return matches, errors
    for entry in entries:
        if entry.name.startswith("."):
            continue  # atomic-write temp files: .tmp.<pid>.<assignment_id>.json
        if entry.suffix != ".json":
            continue  # e.g. Stop's O_EXCL claim files: <assignment_id>.json.claim
        if entry.is_symlink():
            errors.append(
                f"admission receipt path is a symlink; refusing to follow it: {entry.name}"
            )
            continue
        if not entry.is_file():
            continue
        try:
            raw = entry.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(
                f"admission receipt unreadable ({exc.__class__.__name__}): {entry.name}"
            )
            continue
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"admission receipt is not valid JSON: {entry.name}")
            continue
        if not isinstance(receipt, dict):
            errors.append(f"admission receipt is not a JSON object: {entry.name}")
            continue
        if str(receipt.get("expected_agent_id") or "") == agent_id:
            matches.append((entry, receipt))
    return matches, errors


def describe_admission_receipt(
    harness: Path, active_run: str, agent_id: str
) -> tuple[str, list[str]]:
    """Start-time, read-only classification of the receipt bound to ``agent_id``.

    Returns ``(note, errors)``. ``note`` is always non-empty and always goes
    into the injected context (visibility is unconditional); ``errors`` feeds
    the same ``start_errors`` list the run-identity lease uses, so an unusable
    or ambiguous receipt is load-bearing in ``Hook preflight: PASS|FAIL`` the
    same way a lease-write failure already is. See the module docstring for
    the admitted / unusable / absent rule.
    """
    if not ADMISSION_KEY_RE.fullmatch(agent_id or ""):
        return "not checked (agent_id is not a safe admission key)", []

    admissions_dir = harness / "admissions" / active_run
    matches, scan_errors = list_admission_matches(admissions_dir, agent_id)
    if scan_errors:
        detail = "; ".join(scan_errors)
        return (
            f"AMBIGUOUS ({detail})",
            [f"admission receipt state for agent_id={agent_id!r} is ambiguous: {detail}"],
        )
    if not matches:
        return (
            f"none found for agent_id={agent_id!r} in run {active_run!r} "
            "(not every spawned agent holds a registered assignment; if this "
            "one is meant to submit a HARNESS_RESULT, SubagentStop still "
            "requires a matching receipt)",
            [],
        )
    if len(matches) > 1:
        assignment_ids = ", ".join(
            sorted(str(r.get("assignment_id") or p.stem) for p, r in matches)
        )
        return (
            f"AMBIGUOUS ({len(matches)} receipts claim expected_agent_id="
            f"{agent_id!r}: {assignment_ids})",
            [
                f"{len(matches)} admission receipts claim expected_agent_id="
                f"{agent_id!r}: {assignment_ids}"
            ],
        )
    entry, receipt = matches[0]
    assignment_id = str(receipt.get("assignment_id") or entry.stem)
    state = str(receipt.get("state") or "")
    receipt_run = str(receipt.get("run_id") or "")
    if receipt_run != active_run:
        return (
            f"UNUSABLE (receipt for assignment {assignment_id!r} is bound to run "
            f"{receipt_run!r}, not active run {active_run!r})",
            [
                f"admission receipt for agent_id={agent_id!r} is bound to a "
                f"different run ({receipt_run!r} != {active_run!r})"
            ],
        )
    if state != "open":
        return (
            f"UNUSABLE (receipt for assignment {assignment_id!r} is "
            f"{state!r}, not 'open')",
            [
                f"admission receipt for agent_id={agent_id!r} (assignment "
                f"{assignment_id!r}) is {state!r}, not 'open'; this agent was "
                "not admitted"
            ],
        )
    return f"open, bound to assignment {assignment_id!r} in run {active_run!r}", []


def inject_subagent_context(event: dict[str, Any], root: Path) -> None:
    harness = root / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    pack_path = harness / "generated" / "CONTEXT_PACK.md"
    index = load_json(index_path, {}) or {}

    if not index or not pack_path.exists():
        emit_additional_context(
            "SubagentStart",
            "CONTEXT CONTRACT VIOLATION: the canonical context pack is absent or unbuilt. "
            "Do not perform substantive work. Return an error asking the parent to run "
            "`python3 .agent-harness/scripts/build_context_pack.py`.",
            warning="Subagent started without a built context pack",
        )
        return

    integrity_errors = context_integrity(root, index, pack_path)
    if integrity_errors:
        emit_additional_context(
            "SubagentStart",
            "CONTEXT CONTRACT VIOLATION: " + "; ".join(integrity_errors)
            + ". Do not perform substantive work. Ask the parent to rebuild and validate the pack.",
            warning="Subagent started with stale canonical context",
        )
        return

    max_chars = int(index.get("max_injected_chars", 24000))
    version = str(index.get("context_version", "UNBUILT"))
    active_run = active_run_id(harness) or "none"
    agent_id = str(event.get("agent_id") or "")
    runtime_agent_type = str(event.get("agent_type") or "unknown")
    start_errors: list[str] = []
    if not agent_id:
        start_errors.append("SubagentStart event omitted agent_id")
    if runtime_agent_type not in index.get("role_files", {}):
        start_errors.append(
            f"no registered runtime context for agent_type={runtime_agent_type!r}"
        )

    # Run-identity lease (D-058): seal the Start-time run id and the digests of
    # every assignment registered in it, so SubagentStop validates against this
    # run even if another main session moves ACTIVE_RUN before this agent stops.
    lease_note = "not recorded (no active run or missing agent_id)"
    if active_run != "none" and agent_id:
        target = lease_path(harness, agent_id)
        if target is None:
            lease_note = "not recorded (agent_id is not a safe lease key)"
            start_errors.append(
                "agent_id is not a safe lease key, so no run lease could be sealed"
            )
        else:
            digests: dict[str, str] = {}
            for path in sorted(
                (harness / "runs" / active_run / "assignments").glob("*.json")
            ):
                try:
                    digests[path.stem] = (
                        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                except OSError:
                    continue
            try:
                write_json_atomic(
                    target,
                    {
                        "schema_version": 1,
                        "agent_id": agent_id,
                        "agent_type": runtime_agent_type,
                        "run_id": active_run,
                        "context_version": version,
                        "created_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "assignment_digests": digests,
                    },
                )
                lease_note = f"recorded for run {active_run}"
            except OSError as exc:
                # D-067: a lease that cannot be written is a hard Start failure.
                # There is no ACTIVE_RUN fallback any more -- SubagentStop blocks
                # a leaseless agent inside an active run, so this degradation is
                # fail-closed rather than silently reverting to the mutable pointer.
                lease_note = (
                    f"NOT RECORDED ({exc.__class__.__name__}); SubagentStop will "
                    "block this agent"
                )
                start_errors.append(
                    f"run lease could not be written ({exc.__class__.__name__}); "
                    "this agent cannot produce an admissible result"
                )

    # Admission-receipt visibility (D-065 obligation 2, receipt half; round-4
    # review finding). Independent of, and does not gate, the lease block
    # above: a receipt problem must not be masked by a healthy lease or vice
    # versa. Read-only -- see the module docstring for the full rule.
    if active_run == "none" or not agent_id:
        admission_note = "not checked (no active run or missing agent_id)"
    else:
        admission_note, admission_errors = describe_admission_receipt(
            harness, active_run, agent_id
        )
        start_errors.extend(admission_errors)

    pieces = [read_bounded(pack_path, max_chars)]
    used = len(pieces[0])

    role_files = index.get("role_files", {}).get(runtime_agent_type, [])
    for rel in role_files:
        path = root / rel
        remaining = max_chars - used
        if remaining <= 512:
            break
        role_text = read_bounded(path, remaining)
        pieces.append(f"\n\n## Role context: {rel}\n{role_text}")
        used += len(role_text)

    contract = f"""[MANDATORY SUBAGENT BOOTSTRAP]
Runtime agent type: {runtime_agent_type}
Agent ID: {agent_id or "MISSING"}
Active run: {active_run}
Run lease: {lease_note}
Admission receipt: {admission_note}
Canonical context version: {version}

VS Code collaboration does not expose `spawn_agent` to project PreToolUse hooks,
so this hook cannot see your prompt and cannot know which assignment you were
launched for. The parent binds that with a single-use admission token minted
before you were spawned; echoing it is how you prove which assignment is yours.
Before any broad search or analysis:
1. Verify that the first five prompt lines are exactly RUN_ID, ASSIGNMENT_ID,
   CONTEXT_VERSION, INDEPENDENCE_MODE, and ADMISSION_TOKEN, and that
   CONTEXT_VERSION equals `{version}`.
2. Read `.agent-harness/runs/<RUN_ID>/assignments/<ASSIGNMENT_ID>.json` and record the
   `assignment_sha256` printed by the verifier in step 4.
3. Verify that assignment `runtime_agent_type` equals `{runtime_agent_type}` and that
   its compatibility `agent_type` has the same value.
4. Run the single command
   `python3 .agent-harness/scripts/verify_assignment.py .agent-harness/runs/<RUN_ID>/assignments/<ASSIGNMENT_ID>.json`
   and require `VERIFY: PASS`. Copy its printed `assignment_sha256`,
   `review_role_sha256`, and `result_template_sha256` verbatim into
   `spawn_contract` (v1 assignments print `assignment_sha256` only). Do not
   hand-compute these: `review_role_sha256` is a path-aware aggregate digest
   (relpath\\0bytes\\0 per file, in `review_role_files` order), not a raw file hash.
5. Read only files listed by the assignment, plus targeted evidence needed to verify a cited claim.
6. Do not read sibling result files unless INDEPENDENCE_MODE is `adjudication` or the assignment explicitly allows them.
7. Write only to the unique result path in the assignment.
8. Copy Agent ID `{agent_id or "MISSING"}`, runtime type `{runtime_agent_type}`, and
   the assignment `review_role` into the result artifact.
9. Populate top-level `spawn_contract` with the four verified non-secret prompt
   values (RUN_ID, ASSIGNMENT_ID, CONTEXT_VERSION, INDEPENDENCE_MODE — never the
   ADMISSION_TOKEN, which is a single-use secret and must not be copied into any
   artifact),
   `prompt_header_verified=true`, `subagent_start_injected=true`,
   `subagent_start_preflight="PASS"`, the assignment SHA-256, runtime type,
   review-role verification/hash, and result-template verification/hash.
10. End with the required one-line HARNESS_RESULT JSON envelope, including
    `admission_proof` set to the exact ADMISSION_TOKEN value from your prompt.
    Do not read, guess, or reuse any other agent's token: the receipt is
    single-use and bound to one assignment, and a mismatch blocks your stop.
If any required field or file is missing, stop substantive work and return status `error`.

Hook preflight: {"PASS — canonical context and runtime identity verified; assignment role verification required" if not start_errors else "FAIL — " + "; ".join(start_errors)}

""" + "\n".join(pieces)
    emit_additional_context(
        "SubagentStart",
        contract,
        warning="SubagentStart preflight failed; do not perform substantive work" if start_errors else None,
    )


def main() -> None:
    event = read_stdin_json()
    root = repo_root()
    inject_subagent_context(event, root)


if __name__ == "__main__":
    main()
