#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import lease_path, load_json, read_stdin_json, repo_root, write_json_atomic

HARNESS_SCRIPTS = Path(__file__).resolve().parents[2] / ".agent-harness" / "scripts"
sys.path.insert(0, str(HARNESS_SCRIPTS))

from _harness import (  # noqa: E402
    ADMISSION_KEY_RE,
    admission_path,
    assignment_runtime_agent_type,
    token_digest,
    validate_assignment_contract,
    validate_assignment_resource_hashes,
    validate_result_contract,
)

MARKER_RE = re.compile(r"(?:^|\n)HARNESS_RESULT:\s*(\{[^\n]+\})\s*\Z")
ASSIGNMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
VALID_STATUS = {"pass", "fail", "inconclusive", "error"}
ENVELOPE_HINT = (
    "HARNESS_RESULT: {\"assignment_id\":\"...\",\"context_version\":\"...\","
    "\"status\":\"pass|fail|inconclusive|error\",\"result_path\":\"...\","
    "\"admission_proof\":\"<ADMISSION_TOKEN from your prompt>\"}"
)


def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))


# Per-(run, agent_id) mutual-exclusion lock. Dot-prefixed *and* carrying a
# non-".json" suffix, so both receipt scanners skip it twice over --
# admit_agent.other_assignment_bound_to_agent and
# subagent_start_context.list_admission_matches each `continue` on a leading dot
# and again on a suffix that is not ".json" -- and validate_harness's
# `admissions/*/*.json` glob cannot descend into it. It lives under
# `.agent-harness/admissions/`, which is gitignored in full (.gitignore:64) and
# is already the home of the per-assignment `.claim` files, so it adds no new
# untracked path to the working tree.
AGENT_LOCK_DIRNAME = ".agent-locks"


def agent_lock_path(harness: Path, run_id: str, agent_id: str) -> Path | None:
    """Path of the per-(run, agent) consume lock; None for unsafe path keys."""
    if not ADMISSION_KEY_RE.fullmatch(run_id or ""):
        return None
    if not ADMISSION_KEY_RE.fullmatch(agent_id or ""):
        return None
    return harness / "admissions" / run_id / AGENT_LOCK_DIRNAME / f"{agent_id}.lock"


def acquire_agent_lock(lock_file: Path, agent_id: str) -> tuple[int | None, str]:
    """Take the exclusive per-agent consume lock, or say why not.

    D-065 obligation 1 (round-6 finding): the round-5 fix put
    ``conflicting_agent_assignment`` -- a plain ledger *read* -- in front of a
    consume whose only atomic primitive was the per-*assignment* ``O_EXCL``
    claim. Two assignments take two different claim files, so one ``agent_id``
    stopping twice concurrently was excluded by nothing at all: a textbook
    check-then-act, measured to admit both stops in most trials. This lock is
    the missing mutual exclusion, and it is keyed on the thing the obligation is
    about -- the ``agent_id`` -- so the read and the act it guards cannot be
    interleaved by another stop of the same agent.

    ``flock`` rather than a second ``O_EXCL`` file, deliberately:

    * The kernel releases it when the descriptor closes or the process dies.
      "Released on every failure path" therefore holds structurally, including
      the paths that are a crash rather than a ``return`` -- there is no
      unlink step that a future edit could forget, and no stale lock can wedge
      an ``agent_id`` for the rest of a run.
    * ``LOCK_NB`` makes it a *try*-lock: contention fails immediately instead of
      waiting. Nothing in the consume path ever blocks on a lock, so there is no
      hold-and-wait and deadlock is impossible by construction. The ordering is
      nonetheless fixed and one-way -- agent lock first, then the per-assignment
      ``O_EXCL`` claim -- so two stops can never take the two locks in opposite
      orders even if a future change made either of them blocking.

    The lock is *not* the record of the binding; the append-only ledger is. It
    is released at the end of every stop, including a successful one, which is
    what keeps the idempotent re-stop path alive: a resumed agent re-stopping on
    its own assignment simply takes the free lock again and is then admitted by
    the unchanged-bytes check inside.

    Returns ``(fd, error)``:
      * ``(fd, "")``    -- held; the caller must close ``fd`` to release it.
      * ``(None, "")``  -- another stop for this ``agent_id`` holds it right now.
      * ``(None, msg)`` -- the lock could not be evaluated. Fails closed: an
        unusable lock directory is an error, never a skip, because skipping it
        reopens exactly the window this closes.
    """
    directory = lock_file.parent
    if directory.is_symlink():
        return None, "the agent lock directory is a symlink"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, (
            f"the agent lock directory could not be created ({exc.__class__.__name__})"
        )
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        # O_NOFOLLOW turns a planted symlink at the lock path into ELOOP here
        # rather than into a lock taken on some other file.
        return None, f"the agent lock could not be opened ({exc.__class__.__name__})"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, ""
    except OSError as exc:
        os.close(fd)
        return None, f"the agent lock could not be taken ({exc.__class__.__name__})"
    try:
        # Diagnostic only. Nothing ever reads this back to make a decision --
        # the lock is the flock on the descriptor, not the bytes in the file --
        # but a failed write means the directory is not usable for the ledger
        # append either, so it fails closed rather than being ignored.
        os.ftruncate(fd, 0)
        os.write(fd, f"{agent_id} pid={os.getpid()}\n".encode("utf-8"))
    except OSError as exc:
        os.close(fd)
        return None, f"the agent lock could not be written ({exc.__class__.__name__})"
    return fd, ""


def conflicting_agent_assignment(
    ledger_path: Path, agent_id: str, assignment_id: str
) -> tuple[str, str]:
    """Has ``agent_id`` already consumed a DIFFERENT assignment in this run?

    D-065 obligation 1 (round-5 finding): the obligation is one spawned
    agent_id bound to one run, assignment, and digest. admit_agent.py now
    refuses to mint a second receipt for the same agent_id, but that guard
    only covers receipts minted after the fix; this is the Stop-time backstop
    for anything else -- a receipt minted before the fix, or written directly.
    A consume row for the SAME assignment_id is not a conflict: that is the
    idempotent re-stop path a resumed agent relies on, handled separately by
    the admission-receipt state check below.

    This read is only sound because its caller holds the per-(run, agent_id)
    lock across both it and the consume that follows (D-065 obligation 1,
    round-6 finding). On its own it is a check-then-act: two concurrent stops by
    one agent_id both read "no conflict" and both then consumed, because the
    only atomic step downstream is keyed on the *assignment*, and two
    assignments take two different claim files. Do not call this outside
    ``acquire_agent_lock``.

    Returns ``(conflicting_assignment_id, error)``. A missing ledger is not an
    error -- nothing has been consumed yet in this run. An unreadable or
    malformed ledger *is* an error and must fail closed: this is the
    enforcement point for the binding, so an unprovable negative must not
    default to "no conflict" (mirrors validate_harness.read_ledger and
    admit_agent.last_consumed_result_sha256).
    """
    if not ledger_path.exists():
        return "", ""
    try:
        raw = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        return "", f"could not be read ({exc.__class__.__name__})"
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return "", f"line {number} is malformed"
        if not isinstance(row, dict):
            return "", f"line {number} is not an object"
        if (
            row.get("event") == "consumed"
            and str(row.get("agent_id") or "") == agent_id
            and str(row.get("assignment_id") or "") != assignment_id
        ):
            return str(row.get("assignment_id") or ""), ""
    return "", ""


def main() -> None:
    event = read_stdin_json()
    root = repo_root()
    harness = root / ".agent-harness"
    active_path = harness / "ACTIVE_RUN"

    # Run-identity lease (D-058): resolve the run this agent was bound to at
    # SubagentStart instead of trusting the mutable global ACTIVE_RUN pointer.
    event_agent_id = str(event.get("agent_id") or "")
    if not event_agent_id:
        block(
            "SubagentStop event omitted agent_id; no result can be attributed to "
            "this agent."
        )
        return
    lease_file = lease_path(harness, event_agent_id)
    if lease_file is None:
        block(
            f"agent_id {event_agent_id!r} is not a safe lease key, so SubagentStart "
            "could not seal a run lease for it and no result is admissible."
        )
        return
    lease = load_json(lease_file, None)
    if lease is None and lease_file.exists():
        block(
            "Run lease for this agent exists but cannot be parsed; refusing to "
            "resolve the run via the mutable ACTIVE_RUN pointer."
        )
        return

    active_run = ""
    if active_path.exists():
        try:
            active_run = active_path.read_text(encoding="utf-8").strip()
        except OSError:
            active_run = ""

    # Outside an explicitly active harness run *and* with no Start-time lease,
    # do not impose the result envelope. "Harness installed" is judged from the
    # context index rather than the mutable pointer, so deleting or blanking
    # ACTIVE_RUN no longer switches validation off (D-067 review F-D067-06).
    if not active_run and lease is None:
        # "Installed" is the presence of the harness directory itself, not of any
        # single file inside it: keying off one file just relocates the bypass
        # to deleting that file (D-067 round-2 review F-6).
        #
        # It is also checked next to this hook file, not only under the resolved
        # root. `repo_root()` degrades to cwd when git is unavailable, so running
        # from outside the repo would otherwise leave `harness` pointing at a
        # non-existent directory and switch every check off -- the same bypass
        # relocated once more (round-3 review F-R3-13). The cwd-derived path
        # still wins for resolution, which is what keeps fixtures isolated.
        installed_here = Path(__file__).resolve().parents[2] / ".agent-harness"
        if harness.is_dir() or installed_here.is_dir():
            block(
                "The agent harness is installed but ACTIVE_RUN is missing or empty "
                "and this agent has no Start-time lease, so no result can be bound "
                "to a run. Initialize a run with init_run.py before spawning agents."
            )
            return
        return

    message = str(event.get("last_assistant_message") or "")
    match = MARKER_RE.search(message)
    if not match:
        block(
            "Before stopping, write the assignment result artifact and finish with "
            "exactly one line: " + ENVELOPE_HINT
        )
        return

    try:
        envelope = json.loads(match.group(1))
    except json.JSONDecodeError:
        block("HARNESS_RESULT is not valid single-line JSON. Correct it before stopping.")
        return

    required = {
        "assignment_id",
        "context_version",
        "status",
        "result_path",
        "admission_proof",
    }
    missing = sorted(required - set(envelope))
    if missing:
        block(
            f"HARNESS_RESULT is missing required fields: {', '.join(missing)}. "
            "Expected form: " + ENVELOPE_HINT
        )
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
                "Run lease for this agent is malformed; its run identity cannot "
                "be established."
            )
            return
        bound_run = str(lease["run_id"])
    else:
        # D-067: no ACTIVE_RUN fallback. Inside an active run, a stopping agent
        # with no Start-time lease is unbindable, and resolving it through the
        # mutable pointer is exactly the fail-open path D-065 rejected.
        block(
            "No Start-time run lease exists for this agent, so its run identity "
            "cannot be established. Refusing to fall back to the mutable "
            "ACTIVE_RUN pointer. Re-run the agent after confirming that "
            "SubagentStart can write .agent-harness/leases/."
        )
        return

    run_dir = harness / "runs" / bound_run

    # Everything from here on is the consume path, and all of it runs while this
    # process holds the per-(run, agent_id) lock -- acquired BEFORE the ledger
    # conflict check it guards, released on every exit from the guarded region
    # (D-065 obligation 1, round-6 finding). See acquire_agent_lock for why this
    # is an flock try-lock and why the ordering against the per-assignment
    # O_EXCL claim cannot deadlock.
    lock_file = agent_lock_path(harness, bound_run, event_agent_id)
    if lock_file is None:
        block(
            "Bound run id or agent id is not a safe agent-lock path key, so this "
            "agent's consume cannot be serialised against a concurrent stop."
        )
        return
    lock_fd, lock_error = acquire_agent_lock(lock_file, event_agent_id)
    if lock_error:
        block(
            f"The agent consume lock for {event_agent_id!r} in run {bound_run!r} "
            f"could not be acquired: {lock_error}. Refusing to consume an "
            "admission receipt without the mutual exclusion that binds one "
            "agent_id to one assignment (D-065 obligation 1)."
        )
        return
    if lock_fd is None:
        block(
            f"Another SubagentStop for agent_id {event_agent_id!r} is inside the "
            f"consume path for run {bound_run!r} right now. One agent_id may "
            "consume at most one assignment per run (D-065 obligation 1), so "
            "concurrent stops by one agent are serialised rather than "
            "interleaved. Retry this stop once the other has finished."
        )
        return
    try:
        consume_under_agent_lock(
            event=event,
            root=root,
            harness=harness,
            lease=lease,
            lease_file=lease_file,
            event_agent_id=event_agent_id,
            bound_run=bound_run,
            run_dir=run_dir,
            assignment_id=assignment_id,
            envelope=envelope,
        )
    finally:
        # Closing the descriptor is what releases the flock, so every exit from
        # the guarded region -- return, block, or exception -- releases it. The
        # lock *file* is deliberately left in place: unlinking a flocked path
        # lets a racing process create a different inode under the same name and
        # believe it holds the same lock.
        os.close(lock_fd)


def consume_under_agent_lock(
    *,
    event: dict[str, Any],
    root: Path,
    harness: Path,
    # Never None: main() blocks and returns when the Start-time lease is absent,
    # so the run identity is already proven by the time this is called. The
    # `lease is not None` guards inside the body are left as they were rather
    # than pruned, to keep this a pure extraction of the round-5 revision.
    lease: dict[str, Any],
    lease_file: Path,
    event_agent_id: str,
    bound_run: str,
    run_dir: Path,
    assignment_id: str,
    envelope: dict[str, Any],
) -> None:
    """The guarded region: verify the binding, then consume the receipt.

    Split out of ``main`` for one reason only -- so the per-agent lock can wrap
    the whole check-then-act sequence in a single ``try/finally`` instead of
    every one of the ~20 ``block(); return`` paths inside it having to remember
    to release. The body is otherwise unchanged from the round-5 revision.
    """
    # Exact agent-to-assignment binding, the ledger half (D-065 obligation 1,
    # round-5 finding). admit_agent.py now refuses to mint a second receipt for
    # the same agent_id naming a different assignment, but that is a mint-time
    # guard; this is the Stop-time enforcement point, and it is the one that
    # actually consumes a receipt. A conflict here means this agent_id already
    # has a 'consumed' row for a DIFFERENT assignment in this run -- the same
    # assignment_id is the idempotent re-stop path and is deliberately excluded
    # by conflicting_agent_assignment.
    conflict_assignment_id, ledger_error = conflicting_agent_assignment(
        run_dir / "ADMISSIONS.jsonl", event_agent_id, assignment_id
    )
    if ledger_error:
        block(
            f"Admission ledger for run {bound_run!r} {ledger_error}; refusing to "
            "accept a result whose agent-to-assignment binding cannot be "
            "verified."
        )
        return
    if conflict_assignment_id:
        block(
            f"agent_id {event_agent_id!r} already consumed an admission receipt "
            f"for assignment {conflict_assignment_id!r} in run {bound_run!r}. "
            "One agent_id may consume at most one assignment per run (D-065 "
            f"obligation 1); this stop cannot also be attributed to assignment "
            f"{assignment_id!r}."
        )
        return

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

    # Exact agent-to-assignment binding (D-067). The lease proves which *run* the
    # agent belongs to; it cannot prove which assignment, because SubagentStart
    # never sees the spawn prompt. The parent-minted admission receipt supplies
    # that binding: only the agent holding this assignment's token can consume it.
    admission_file = admission_path(harness, bound_run, assignment_id)
    if admission_file is None:
        block("Bound run id or assignment id is not a safe admission path key.")
        return
    if admission_file.is_symlink():
        block("Admission receipt path is a symlink; refusing to follow it.")
        return
    admission = load_json(admission_file, None)
    if not isinstance(admission, dict):
        block(
            f"No usable parent-authenticated admission receipt for assignment "
            f"{assignment_id!r} in run {bound_run!r} (absent or unparseable). The "
            "parent must mint one with `python3 "
            f".agent-harness/scripts/admit_agent.py --assignment-id {assignment_id}` "
            "before the agent is spawned."
        )
        return
    if admission.get("schema_version") != 1:
        block(
            "Admission receipt has an unsupported schema_version "
            f"{admission.get('schema_version')!r}."
        )
        return
    if (
        str(admission.get("run_id")) != bound_run
        or str(admission.get("assignment_id")) != assignment_id
    ):
        block("Admission receipt does not match the bound run and assignment.")
        return
    if str(admission.get("assignment_sha256")) != recomputed_sha:
        block(
            "Admission receipt was minted against different assignment bytes than "
            "the ones being validated."
        )
        return
    proof = str(envelope.get("admission_proof") or "")
    if not proof:
        block(
            "HARNESS_RESULT admission_proof is empty; paste the ADMISSION_TOKEN from "
            "your prompt."
        )
        return
    if not hmac.compare_digest(token_digest(proof), str(admission.get("token_digest"))):
        block(
            f"admission_proof does not match the receipt for {assignment_id!r}. An "
            "agent may only submit the assignment it was admitted for."
        )
        return
    expected_agent_id = str(admission.get("expected_agent_id") or "")
    if expected_agent_id and expected_agent_id != event_agent_id:
        block(
            f"Admission receipt for {assignment_id!r} was minted for agent "
            f"{expected_agent_id!r}, not {event_agent_id!r}."
        )
        return

    # Single-use, with one deliberate exception: a re-stop by the *same* agent
    # whose result bytes are unchanged is idempotent. Without this a resumed
    # subagent could never terminate, because acceptance consumes the receipt
    # (D-067 review F-D067-03).
    admission_state = str(admission.get("state"))
    already_consumed_by_this_agent = (
        admission_state == "consumed"
        and str(admission.get("consumed_by_agent_id") or "") == event_agent_id
    )
    if admission_state != "open" and not already_consumed_by_this_agent:
        block(
            f"Admission receipt for {assignment_id!r} is {admission_state!r}, not "
            "'open'. Receipts are single-use; a replayed or already-consumed "
            "receipt cannot admit another agent's result."
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
    # A symlink at the declared path resolves to the canonical location while the
    # real bytes live elsewhere, so the attribution digest would describe a file
    # the run does not own (D-067 review F-D067-06).
    if (root / result_path).is_symlink():
        block("Declared result artifact is a symlink; write the file itself.")
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

    # Single read, as for the assignment: the validated object and the attributed
    # digest must describe the same bytes (D-067 round-2 review F-14).
    try:
        result_raw = resolved.read_bytes()
        result = json.loads(result_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        result = None
    if not isinstance(result, dict):
        block("Declared result artifact is not a readable JSON object.")
        return
    result_sha = "sha256:" + hashlib.sha256(result_raw).hexdigest()
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

    # Acceptance. Claim the receipt exclusively first: O_EXCL is the only step
    # here that is atomic against a concurrent stop, so it, not the earlier
    # state read, is what makes the receipt single-use (D-067 review F-D067-02).
    claim_file = admission_file.with_name(admission_file.name + ".claim")
    try:
        fd = os.open(claim_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        holder = ""
        try:
            holder = claim_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        if holder != event_agent_id:
            block(
                f"Admission receipt for {assignment_id!r} is already claimed by "
                f"{holder or 'another agent'}; refusing a concurrent second consume."
            )
            return
        # Same agent re-stopping. The only admissible reason a claim exists for
        # this agent is that a previous stop already consumed the receipt and
        # recorded the attribution. Anything else -- a planted claim, a consume
        # that died before writing the record -- must block, or attribution
        # could be skipped entirely (D-067 round-2 review F-11).
        if admission_state != "consumed":
            block(
                f"A claim exists for {assignment_id!r} but the receipt is "
                f"{admission_state!r} with no attribution record. Refusing to "
                "accept a result that was never attributed; the parent must "
                "reissue the receipt with admit_agent.py --reopen."
            )
            return
        if str(admission.get("result_sha256")) != result_sha:
            block(
                "Result artifact changed after this agent's result was admitted; "
                "refusing to re-admit different bytes under the same receipt."
            )
            return
        return
    except OSError as exc:
        block(
            f"Admission receipt could not be claimed ({exc.__class__.__name__}); "
            "refusing to accept a result that cannot be attributed to this agent."
        )
        return
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(event_agent_id + "\n")

    consumed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        # Ledger first, receipt second -- the same ordering admit_agent.py uses.
        # With the receipt written first, a failed ledger append left a consumed
        # receipt with no attribution row, and the retry then took the idempotent
        # path and was accepted (D-067 round-3 review F-R3-11). Written this way,
        # a failed append leaves the receipt open and the retry re-consumes it.
        #
        # Durable attribution: the receipt directory is gitignored working state,
        # so the record that survives into committed evidence is this append-only
        # ledger inside the run directory (D-067 review F-D067-07).
        with (run_dir / "ADMISSIONS.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "event": "consumed",
                        "run_id": bound_run,
                        "assignment_id": assignment_id,
                        "agent_id": event_agent_id,
                        "agent_type": event_agent_type,
                        "result_sha256": result_sha,
                        "assignment_sha256": recomputed_sha,
                        "token_digest": str(admission.get("token_digest")),
                        "at": consumed_at,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        write_json_atomic(
            admission_file,
            {
                **admission,
                "state": "consumed",
                "consumed_at": consumed_at,
                "consumed_by_agent_id": event_agent_id,
                "consumed_by_agent_type": event_agent_type,
                "result_sha256": result_sha,
            },
        )
    except OSError as exc:
        # Release the claim so the retry is not wedged behind a half-finished
        # consume that never produced an attribution row.
        claim_file.unlink(missing_ok=True)
        block(
            f"Admission receipt could not be consumed ({exc.__class__.__name__}); "
            "refusing to accept a result that cannot be attributed to this agent."
        )
        return

    # Mark the lease consumed rather than deleting it: SubagentStart overwrites
    # it for a reused agent_id anyway, and keeping it lets an idempotent re-stop
    # of the same agent still resolve its bound run instead of livelocking.
    try:
        write_json_atomic(
            lease_file,
            {
                **lease,
                "state": "consumed",
                "consumed_assignment_id": assignment_id,
                "consumed_at": consumed_at,
            },
        )
    except OSError:
        lease_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
