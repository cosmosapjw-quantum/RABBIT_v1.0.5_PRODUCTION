#!/usr/bin/env python3
"""Mint a single-use admission receipt binding one spawned agent to one assignment.

Why this exists (D-067, discharging D-065 finding F-D065-01): the SubagentStart
hook event carries only ``agent_id`` and ``agent_type``. It never carries the
spawn prompt, so no hook can infer which assignment an agent was launched for.
The D-058 lease therefore sealed *every* assignment registered in the run, and
SubagentStop admitted whichever one the stopping agent declared for itself. A
controlled same-run substitution passed.

The fix moves the binding to the only party that knows it: the parent. Before
spawning, the parent runs this script, which stores ``sha256(token)`` in the
receipt and prints ``token`` exactly once for pasting into the spawn prompt.
SubagentStop admits the agent only if the ``admission_proof`` in its
HARNESS_RESULT envelope hashes to the stored digest, then consumes the receipt,
recording the writing ``agent_id`` and the result SHA-256 as the
write-attribution record.

TOKEN HANDLING -- this script never writes the raw token to disk, but the token
does not stay secret by itself. It is pasted into a spawn prompt, echoed in the
agent's final message, and therefore appears in any transcript or captured
SubagentStop event JSON. A captured *blocked* stop is the dangerous case,
because the receipt is still open. Two consequences, both load-bearing:

* Always pass ``--expect-agent-id`` when the parent chooses the agent id (it
  does under the recorded VS Code LOCAL-ADAPT). A leaked token is then useless
  to any other agent.
* Scrub ``admission_proof`` before retaining a stop-event JSON as run evidence;
  ``scrub_admission_proof.py`` does this, replacing the value with its digest.

An already-consumed receipt cannot be re-admitted regardless.

``--agent-id-unknown`` -- WHY THE MINT-TIME GUARD CANNOT APPLY IN THIS MODE, and
what is claimed instead (D-065 obligation 1, round-6 finding). The mint-time
guard ``other_assignment_bound_to_agent`` keys on ``expected_agent_id``. In this
mode there is no such key and no substitute for one: the runtime has not yet
assigned an agent id, so the agent this receipt will admit does not exist at
mint time. ``run_id`` and ``assignment_id`` do not identify an agent, and no
other field in the receipt does either. A mint-time agent guard is therefore not
merely absent here, it is not constructible.

The obligation is met at the only point where the agent id first exists -- Stop.
``subagent_stop_validate.py`` holds a per-``(run_id, agent_id)`` lock across the
whole consume path and refuses a second consume for an agent_id that already has
a consume row for a different assignment. Neither step reads
``expected_agent_id``, so both apply in full in this mode. What was a silent
exemption before -- the guard was gated on ``if expect_agent_id:`` and so simply
did not run -- is now a stated scope limit rather than a gap:

* DELIVERED in this mode: one ``agent_id`` consumes at most one assignment per
  run, atomically, whatever tokens it holds.
* NOT DELIVERED in this mode: *which* assignment a given agent ends up bound to.
  With the receipt bound to the token alone, the Stop-time
  ``expected_agent_id`` check is a no-op, so two agents holding two open
  token-only receipts may swap which one each consumes. That is D-067's
  anti-substitution property, not D-065 obligation 1, and it is the reason
  ``--expect-agent-id`` is the required default.

The mode is therefore warned about on stderr at mint time and recorded durably:
both the receipt and the append-only ledger row carry
``agent_binding: "token-only"``, so an auditor can tell after the fact which
receipts never carried an agent binding instead of having to infer it from an
empty string.

``--reopen`` OF A RECEIPT THAT WAS NEVER CONSUMED -- D-070 round-6 finding
F-R6-02, closed here. ``created_at`` used to be stamped ``utc_now()`` on every
mint, ``--reopen`` included, with no special case for reopening a receipt whose
prior state was never ``"consumed"``. One documented, non-forged
``--reopen --reason "..."`` call -- exactly the recovery this script's own
usage example recommends for a dead agent -- could therefore reset the
in-flight clock on the same never-verified bytes indefinitely, and the receipt
on disk actively misled a human reading it. ``validate_harness.py`` no longer
trusts ``created_at`` alone for its PENDING_ADMISSION_MAX_AGE_HOURS carve-out
(``open_admission_chain_start`` reads the append-only ledger instead), and
this script now asks the same question of the same ledger before stamping
anything: is a chain already open for the assignment being (re)minted? If the
ledger's last event for it is a consume, or there is none at all, no -- this
mint legitimately starts a fresh chain and ``created_at`` is ``now``. If the
last event is an unconsumed ``minted``/``reopened`` row, the chain is still
open; its true start is carried forward into both ``created_at`` and the new
explicit ``chain_started_at`` field -- on the receipt and on the ledger row --
instead of being reset. At the CLI this reopen-of-the-never-consumed shape is
otherwise indistinguishable from a declared supersession after a genuine
consume; the difference is named loudly on stderr every time it happens
(``warn()`` below), never silently, and ``superseded_result_sha256`` is
accompanied by an explicit ``supersedes_admitted_result`` boolean so an empty
declaration cannot be mistaken for a real one.

THREAT MODEL -- read this before claiming more than it delivers. The harness and
its subagents run as the same OS user, so an agent that deliberately writes
outside its declared result path can forge any repo-local artifact, including a
receipt, a lease, and the ledger. What this mechanism removes is *confusion*:
prompt-level substitution, same-run assignment mix-ups, races between
overlapping mains, and unattributable writes. Deliberate forgery is not
prevented.

Usage:
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX \\
        --expect-agent-id canary-01
    python3 .agent-harness/scripts/admit_agent.py --assignment-id A-XXXX \\
        --reopen --reason "first agent died before stopping"
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path

from _harness import (
    ADMISSION_KEY_RE,
    ASSIGNMENT_ID_RE,
    active_run_id,
    admission_path,
    assignment_runtime_agent_type,
    dump_json_atomic,
    load_json,
    loads_strict,
    root,
    token_digest,
    utc_now,
)

TOKEN_BYTES = 32


def fail(message: str) -> None:
    raise SystemExit(f"admit_agent: {message}")


def warn(message: str) -> None:
    """Non-fatal notice on stderr, so it cannot be mistaken for mint output.

    stdout is parsed for ``ADMISSION_TOKEN=`` and friends; anything advisory has
    to stay off it.
    """
    print(f"admit_agent: WARNING: {message}", file=sys.stderr)


def _read_ledger_rows(ledger_path: Path, purpose: str) -> list[dict]:
    """Every row of the append-only admission ledger, fail-closed.

    Shared by every reader of ``ADMISSIONS.jsonl`` in this script, so a
    missing, unreadable, or malformed ledger is one code path, not several
    slightly different ways of going missing. A ledger that genuinely does not
    exist yet (this assignment's first-ever mint) returns ``[]`` -- that is
    the ordinary state of a fresh run, not a failure. Anything else that
    prevents a full, parsed read aborts the mint (``purpose`` names why),
    because ``purpose``.
    """
    if not ledger_path.exists():
        return []
    try:
        raw = ledger_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail(f"admission ledger could not be read ({exc.__class__.__name__}); {purpose}")
        return []
    rows: list[dict] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = loads_strict(line)
        except json.JSONDecodeError:
            fail(f"admission ledger line {number} is malformed; {purpose}")
            return []
        if not isinstance(row, dict):
            fail(f"admission ledger line {number} is not an object; {purpose}")
            return []
        rows.append(row)
    return rows


def last_consumed_result_sha256(ledger_path: Path, assignment_id: str) -> str:
    """The digest of the bytes the ledger last admitted for this assignment.

    A reopen must *declare* which admitted digest it supersedes (D-067 round-4
    review, defect 2). The declaration is taken from the append-only ledger
    rather than from the receipt, because the receipt directory is gitignored
    working state and may have been replaced or corrupted -- the ledger is the
    record that survives into committed evidence. Fails closed: an existing
    ledger that cannot be read or parsed aborts the mint rather than emitting a
    reopen row that declares nothing.

    This intentionally answers a different question than
    ``open_chain_started_at`` below: it is the last digest EVER admitted for
    this assignment, across every past chain, not only the currently-open one.
    A reopen of a chain that was itself never consumed can still owe a
    declaration for an earlier chain's admitted digest, if one exists and has
    not been superseded since -- ``check_declared_supersession`` in
    ``validate_harness.py`` walks the same rows the same way.
    """
    digest = ""
    for row in _read_ledger_rows(
        ledger_path, "a reopen must declare the digest it supersedes"
    ):
        if (
            row.get("event") == "consumed"
            and str(row.get("assignment_id")) == assignment_id
        ):
            digest = str(row.get("result_sha256") or "")
    return digest


def open_chain_started_at(ledger_path: Path, assignment_id: str) -> str:
    """When this assignment's currently-open admission chain was first minted,
    or ``""`` if no chain is open right now.

    D-070 round-6 finding F-R6-02. Mirrors
    ``validate_harness.open_admission_chain_start`` exactly, on purpose: the
    validator measures its PENDING_ADMISSION_MAX_AGE_HOURS carve-out from
    whichever timestamp this function would return, so this script has to ask
    the same question of the same rows before it stamps ``created_at`` --
    otherwise the receipt can (once again) claim an age the ledger disagrees
    with. Implemented independently rather than imported, matching
    ``last_consumed_result_sha256`` above: this script's own fail-closed
    ledger read stays self-contained.

    A chain is the run of ``minted``/``reopened`` rows for one assignment that
    no ``consumed`` row has closed yet. Walking the rows in order: a
    ``consumed`` row closes the chain (resets to no-chain-open); a
    ``minted``/``reopened`` row STARTS one only when none is currently open,
    and its ``at`` is what gets returned. A further ``minted``/``reopened``
    row while a chain is already open -- reopening a receipt that was never
    consumed -- does not move the start forward.
    """
    started = ""
    for row in _read_ledger_rows(
        ledger_path,
        "a reopen must not be able to hide how long its admission chain has "
        "actually been open",
    ):
        if str(row.get("assignment_id")) != assignment_id:
            continue
        event = row.get("event")
        if event == "consumed":
            started = ""
        elif event in ("minted", "reopened") and not started:
            started = str(row.get("at") or "")
    return started


def chain_age_note(started: str, now: str) -> str:
    """`` (N.Nh)`` for a warning message, or ``""`` if either side won't parse.

    Advisory only. The timestamps here were themselves produced by
    ``utc_now()`` or read verbatim off the ledger by ``open_chain_started_at``
    above, which already fails closed on a ledger it cannot read or parse; a
    malformed individual timestamp inside an otherwise well-formed ledger must
    not block a mint on account of a cosmetic detail in a warning message.
    """
    try:
        delta = datetime.fromisoformat(now) - datetime.fromisoformat(started)
    except ValueError:
        return ""
    return f" ({delta.total_seconds() / 3600:.1f}h)"


def scan_run_receipts(admissions_dir: Path) -> list[tuple[str, dict]]:
    """Every receipt in this run's admission directory, as ``(assignment_id, receipt)``.

    Fails closed on any entry that cannot be listed, read, or parsed as a JSON
    object -- not only ones that turn out to matter to the caller -- because an
    unparseable file cannot be proven irrelevant. This mirrors
    ``list_admission_matches`` in ``subagent_start_context.py``, which folds the
    same class of failure into a hard FAIL rather than a skip.

    The directory is scanned in full before any caller predicate is applied, so
    a corrupt receipt is fatal even when an earlier, well-formed one would have
    answered the caller's question. That is what the guards below already
    promise; short-circuiting on the first match would let a corrupt sibling
    escape notice purely because of sort order.

    Skipped, by two independent rules so that neither alone is load-bearing:
    dot-prefixed names (atomic-write temps ``.tmp.<pid>.<aid>.json``, and Stop's
    ``.agent-locks/`` directory) and any suffix that is not ``.json`` (Stop's
    per-assignment ``O_EXCL`` files, ``<assignment_id>.json.claim``).
    """
    if admissions_dir.is_symlink():
        fail(f"admission directory path is a symlink: {admissions_dir}")
        return []
    try:
        entries = sorted(admissions_dir.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        fail(
            "admission receipts directory could not be listed "
            f"({exc.__class__.__name__}); the agent-to-assignment binding "
            "cannot be verified"
        )
        return []
    receipts: list[tuple[str, dict]] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.suffix != ".json":
            continue
        if entry.is_symlink():
            fail(
                "admission receipt path is a symlink; refusing to follow it: "
                f"{entry.name}"
            )
            return []
        if not entry.is_file():
            continue
        try:
            raw = entry.read_text(encoding="utf-8")
        except OSError as exc:
            fail(
                f"admission receipt unreadable ({exc.__class__.__name__}): "
                f"{entry.name}; the agent-to-assignment binding cannot be verified"
            )
            return []
        try:
            receipt = loads_strict(raw)
        except json.JSONDecodeError:
            fail(
                f"admission receipt is not valid JSON: {entry.name}; the "
                "agent-to-assignment binding cannot be verified"
            )
            return []
        if not isinstance(receipt, dict):
            fail(
                f"admission receipt is not a JSON object: {entry.name}; the "
                "agent-to-assignment binding cannot be verified"
            )
            return []
        receipts.append((str(receipt.get("assignment_id") or entry.stem), receipt))
    return receipts


def other_assignment_bound_to_agent(
    admissions_dir: Path, agent_id: str, assignment_id: str
) -> str:
    """The assignment_id of another receipt in this run already bound to
    ``agent_id`` by an open or consumed receipt, or ``""`` if none exists.

    D-065 obligation 1 (round-5 finding): the obligation is one spawned
    ``agent_id`` bound to one run, assignment, and digest. What was enforced
    before this was one *token* to one assignment -- nothing stopped minting a
    second receipt naming the same ``expect_agent_id`` for a different
    assignment, so a single agent_id could consume two assignments in one run.
    This is the mint-time half of the fix; ``subagent_stop_validate.py`` is the
    other half, and the only half that exists under ``--agent-id-unknown``
    (module docstring).

    A receipt for the SAME ``assignment_id`` is deliberately excluded: that is
    the ordinary ``--reopen`` path (re-minting for the same assignment and
    agent), which is legitimate and handled by the existing `prior` logic in
    `main`, not by this guard.
    """
    for other_assignment_id, receipt in scan_run_receipts(admissions_dir):
        if other_assignment_id == assignment_id:
            continue
        if str(receipt.get("expected_agent_id") or "") != agent_id:
            continue
        if str(receipt.get("state") or "") in ("open", "consumed"):
            return other_assignment_id
    return ""


def open_token_only_assignments(admissions_dir: Path, assignment_id: str) -> list[str]:
    """Other assignments in this run whose receipt is open and agent-unbound.

    These are the receipts that make the residual ambiguity of
    ``--agent-id-unknown`` concrete: each can be consumed by whichever agent
    holds its token, so the harness does not determine which agent ends up on
    which assignment. Naming them is what keeps the mode warned rather than
    silent.
    """
    return sorted(
        other_assignment_id
        for other_assignment_id, receipt in scan_run_receipts(admissions_dir)
        if other_assignment_id != assignment_id
        and str(receipt.get("state") or "") == "open"
        and not str(receipt.get("expected_agent_id") or "")
    )


def release_claim_under_stop_lock(
    harness: Path, run_id: str, prior: dict | None, claim: Path
) -> None:
    """Delete the O_EXCL claim, but never out from under a running Stop.

    `subagent_stop_validate.py` calls this claim "the only step here that is
    atomic against a concurrent stop", and it is what makes a receipt
    single-use. A reopen has to release it, or the newly admitted agent can
    never consume. It used to do that with a bare `claim.unlink(missing_ok=True)`
    taking no lock at all: BD623 R6 destroyed a live claim while Stop held its
    per-(run, agent) flock, and a probe confirmed the flock WOULD have blocked
    the delete had it ever been requested.

    So the reopen takes the same lock Stop takes, for the agent the receipt
    being replaced was bound to. The path is constructed here rather than
    imported because the definition lives in the Stop hook, which every live
    canary attests by digest -- importing would mean editing it, which retires
    C11, C12 and C13. `subagent_stop_validate.agent_lock_path` remains the
    authority for this layout; the two must move together.

    The agent to lock on is read from the claim file, which Stop writes its own
    `agent_id` into -- so this works for a token-only receipt too, where
    `expected_agent_id` is empty by construction. Note that a claim SURVIVING is
    the normal post-consume state, not a sign of a stop in flight: Stop deletes
    it only when a consume fails. "In flight" is the flock being held, and
    nothing else.
    """
    holder = ""
    try:
        holder = claim.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if not holder:
        holder = str((prior or {}).get("expected_agent_id") or "")
    if not holder or not ADMISSION_KEY_RE.fullmatch(holder):
        claim.unlink(missing_ok=True)
        return
    lock_file = harness / "admissions" / run_id / ".agent-locks" / f"{holder}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail(
                f"refusing to reopen while agent {holder!r} is stopping: its consume "
                "lock is held, so deleting the claim now would remove the only step "
                "that is atomic against that stop. Wait for it to finish."
            )
        claim.unlink(missing_ok=True)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to ACTIVE_RUN; must equal it (admission is for the live run).",
    )
    parser.add_argument(
        "--expect-agent-id",
        default="",
        help="Bind the receipt to this agent_id as well as to the token. "
        "SubagentStop then rejects any other agent even if it holds the token. "
        "Required unless --agent-id-unknown is passed.",
    )
    parser.add_argument(
        "--agent-id-unknown",
        action="store_true",
        help="Mint without an agent_id binding, for runtimes that assign the id "
        "themselves and do not expose it before the spawn. The receipt is bound "
        "to the token only: the one-assignment-per-agent_id ceiling still holds "
        "(enforced at Stop, which is where the agent id first exists), but this "
        "receipt cannot say WHICH agent may consume it. Warns on stderr and "
        "records agent_binding=token-only in the receipt and the ledger. Pass "
        "--reason to record why the agent id was unavailable.",
    )
    parser.add_argument(
        "--reopen",
        action="store_true",
        help="Replace an existing receipt for this assignment. Requires --reason. "
        "Appends a declared `reopened` ledger row naming the result digest it "
        "supersedes, which is what licenses the second consume row that follows: "
        "validate_harness errors on an undeclared second consume. Until the "
        "reopened assignment is consumed again, it also reports any result whose "
        "bytes differ from the last consume row. Reopening a receipt that was "
        "never consumed does not reset its in-flight clock: created_at and "
        "chain_started_at carry the ledger's true chain start forward "
        "unchanged, and a loud stderr warning names how long that chain has "
        "already been open.",
    )
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    repo = root()
    harness = repo / ".agent-harness"

    assignment_id = str(args.assignment_id)
    if not ASSIGNMENT_ID_RE.fullmatch(assignment_id):
        fail(f"assignment_id {assignment_id!r} has an invalid form")
    expect_agent_id = str(args.expect_agent_id or "")
    if expect_agent_id and not ADMISSION_KEY_RE.fullmatch(expect_agent_id):
        fail(f"expect-agent-id {expect_agent_id!r} has an invalid form")
    if not expect_agent_id and not args.agent_id_unknown:
        # The D-065 obligation is an agent-to-assignment binding. Leaving the
        # agent half to convention is how it quietly degrades to token-only
        # (D-067 round-3 review F-R3-15).
        fail(
            "--expect-agent-id is required so the receipt binds an agent, not "
            "just a token. Pass --agent-id-unknown only when the runtime assigns "
            "the agent id itself and does not expose it before the spawn."
        )

    active = active_run_id(repo)
    run_id = str(args.run_id or active)
    if run_id != active:
        fail(
            f"run_id {run_id!r} is not the active run {active!r}; admission is only "
            "minted for the live run"
        )

    assignment_file = harness / "runs" / run_id / "assignments" / f"{assignment_id}.json"
    try:
        assignment_raw = assignment_file.read_bytes()
    except OSError:
        fail(f"registered assignment does not exist: {assignment_file.relative_to(repo)}")
        return
    assignment = load_json(assignment_file)
    if not isinstance(assignment, dict):
        fail("registered assignment is not a JSON object")
        return
    if str(assignment.get("run_id")) != run_id:
        fail("registered assignment run_id does not match the active run")

    index = load_json(harness / "context" / "CONTEXT_INDEX.json")
    context_version = str(index.get("context_version", "UNBUILT"))
    if str(assignment.get("context_version")) != context_version:
        fail(
            f"registered assignment is stale for context {context_version!r}; rebuild "
            "the pack and reissue the assignment before admitting an agent"
        )

    target = admission_path(harness, run_id, assignment_id)
    if target is None:
        fail("run_id or assignment_id is not a safe admission path key")
        return

    # One agent_id binds to one assignment per run (D-065 obligation 1), not
    # merely one token to one assignment. Minting a second receipt naming an
    # expect_agent_id already bound elsewhere -- open or consumed -- would
    # recreate exactly the gap this guard closes: --reopen for the SAME
    # assignment is excluded from this check (see
    # other_assignment_bound_to_agent) because that is the legitimate
    # supersession path handled below.
    admissions_dir = harness / "admissions" / run_id
    agent_binding = "agent-id" if expect_agent_id else "token-only"
    if expect_agent_id:
        conflict_assignment_id = other_assignment_bound_to_agent(
            admissions_dir, expect_agent_id, assignment_id
        )
        if conflict_assignment_id:
            fail(
                f"agent_id {expect_agent_id!r} already holds an open or "
                f"consumed admission receipt for assignment "
                f"{conflict_assignment_id!r} in run {run_id!r}. One admission "
                "token binds one agent_id to exactly one assignment (D-065 "
                f"obligation 1); mint {assignment_id!r} for a different "
                f"--expect-agent-id, or reopen {conflict_assignment_id!r} "
                "instead if that is the assignment this agent should hold."
            )
            return
    else:
        # --agent-id-unknown. The guard above is not merely skipped here, it is
        # not constructible: there is no agent id yet to key it on (module
        # docstring). What must not happen is what happened before -- the whole
        # gate being `if expect_agent_id:` with no `else`, so the mode was
        # silently exempt and nothing on disk said so. The scope is stated on
        # stderr and recorded in `agent_binding` on both the receipt and the
        # append-only ledger row.
        warn(
            f"minting {assignment_id!r} with --agent-id-unknown: the receipt is "
            "bound to its token only.\n"
            "  Still guaranteed: one agent_id consumes at most one assignment "
            "per run. subagent_stop_validate.py enforces that atomically under a "
            "per-(run, agent_id) lock and does not key on expected_agent_id, so "
            "it holds in this mode too (D-065 obligation 1).\n"
            "  NOT guaranteed: which agent may consume THIS receipt. The "
            "Stop-time expected_agent_id check is a no-op for an unbound "
            "receipt, so token-holders are interchangeable.\n"
            "  Recorded as agent_binding=token-only in the receipt and in the "
            "run's ADMISSIONS.jsonl. Pass --expect-agent-id instead whenever the "
            "parent chooses the agent id; pass --reason to record why it cannot."
        )
        also_open = open_token_only_assignments(admissions_dir, assignment_id)
        if also_open:
            warn(
                f"run {run_id!r} already has {len(also_open)} other open, "
                f"agent-unbound receipt(s): {', '.join(also_open)}. Any spawned "
                "agent holding any of these tokens can consume the matching "
                "assignment, so the agent-to-assignment pairing across this set "
                "is not determined by the harness -- only the one-assignment-per-"
                "agent ceiling is."
            )
        if not args.reason.strip():
            warn(
                "no --reason was given for --agent-id-unknown, so the run "
                "carries no record of why the agent id was unavailable."
            )

    prior = None
    if target.exists():
        try:
            prior = load_json(target)
        except (OSError, json.JSONDecodeError):
            prior = {"state": "unreadable", "created_at": "", "consumed_by_agent_id": ""}
        if not isinstance(prior, dict):
            prior = {"state": "malformed", "created_at": "", "consumed_by_agent_id": ""}
    if prior is not None:
        if not args.reopen:
            fail(
                f"an admission receipt already exists for {assignment_id!r} in state "
                f"{str(prior.get('state'))!r}. Receipts are single-use; pass --reopen "
                "--reason '...' to replace it deliberately."
            )
        if not args.reason.strip():
            fail("--reopen requires a non-empty --reason")

    token = secrets.token_urlsafe(TOKEN_BYTES)
    # A reopen must also release the O_EXCL claim, or the new agent cannot
    # consume the receipt it was just admitted for.
    claim = target.with_name(target.name + ".claim")
    minted_at = utc_now()
    ledger_path = harness / "runs" / run_id / "ADMISSIONS.jsonl"

    # F-R6-02: ask the ledger, before stamping anything, whether a chain is
    # already open for this assignment -- the same question
    # validate_harness.open_admission_chain_start asks of the same rows. If
    # one is open, its true start is carried forward into created_at and the
    # new chain_started_at field instead of being reset; if none is open (no
    # rows yet, or the last event was a consume), this mint legitimately opens
    # a fresh chain and both timestamps are `now`. This is deliberately
    # unconditional -- not gated on `prior` or `args.reopen` -- so a receipt
    # that was deleted out from under an open chain cannot buy a falsely fresh
    # created_at either.
    existing_chain_start = open_chain_started_at(ledger_path, assignment_id)
    if existing_chain_start:
        created_at = existing_chain_start
        chain_started_at = existing_chain_start
    else:
        created_at = minted_at
        chain_started_at = minted_at

    # The abuse shape (D-070 round-6, item 3): reopening a receipt whose chain
    # was never closed by a consume is, at the CLI, indistinguishable from a
    # declared supersession after a genuine one -- `--reopen --reason '...'`
    # either way. A second required flag was considered and rejected: it would
    # gate this module's own documented one-step recovery for a dead agent
    # (see the --reopen usage example above) behind an extra step for what is,
    # operationally, the overwhelmingly common legitimate case, and
    # test_admit_agent_reopen_same_assignment_is_exempt_from_the_binding_guard
    # already exercises exactly this call shape expecting it to succeed
    # unaided. A loud, unconditional warning naming the true chain age is the
    # check that cannot be silenced instead: it costs nothing on the
    # legitimate path and cannot be suppressed on the abusive one.
    reopening_never_consumed = prior is not None and bool(existing_chain_start)
    if reopening_never_consumed:
        warn(
            f"--reopen for {assignment_id!r} is reopening a receipt that was "
            f"never consumed. Its admission chain has been open since "
            f"{existing_chain_start}{chain_age_note(existing_chain_start, minted_at)}, "
            "and this reopen does not restart that clock: created_at and "
            "chain_started_at are carried forward unchanged, and "
            "validate_harness.py measures its in-flight ceiling from that "
            "same ledger-recorded start. If the prior agent genuinely died "
            "before stopping this is the correct call; if it is still "
            "running, this reopen races it for the receipt instead of "
            "waiting for it to consume the one it already holds."
        )

    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "assignment_id": assignment_id,
        "assignment_sha256": "sha256:" + hashlib.sha256(assignment_raw).hexdigest(),
        "runtime_agent_type": assignment_runtime_agent_type(assignment),
        "context_version": context_version,
        "token_digest": token_digest(token),
        "expected_agent_id": expect_agent_id,
        # Explicit, so "this receipt names no agent" is a recorded fact rather
        # than something an auditor has to infer from an empty string that could
        # equally mean "field written by an older tool" (D-065 round-6).
        "agent_binding": agent_binding,
        "state": "open",
        "created_at": created_at,
        # D-070 round-6, item 2: an explicit, receipt-visible statement of the
        # same fact the append-only ledger row below also carries, so a human
        # reading only the (gitignored, working-state) receipt is not misled
        # even without cross-referencing the ledger.
        "chain_started_at": chain_started_at,
    }
    if prior is not None:
        receipt["reopened_from"] = {
            "state": str(prior.get("state")),
            "created_at": str(prior.get("created_at")),
            "consumed_at": str(prior.get("consumed_at") or ""),
            "consumed_by_agent_id": str(prior.get("consumed_by_agent_id") or ""),
            "consumed_by_agent_type": str(prior.get("consumed_by_agent_type") or ""),
            "result_sha256": str(prior.get("result_sha256") or ""),
            "reason": args.reason.strip(),
        }

    # A reopen is a *declared* supersession, not merely a second mint. Before
    # this the reopen row was indistinguishable from a first mint (`"event":
    # "minted"`), so a second consume row could appear with nothing on the
    # append-only record saying that superseding the first was intended -- which
    # is how an edited artifact was laundered back to green (D-067 round-4
    # review, defect 2). The distinct event, the superseded digest, and the
    # mandatory reason are what validate_harness requires before it will accept
    # more than one consume row for an assignment.
    row: dict[str, object] = {
        "event": "reopened" if prior is not None else "minted",
        "run_id": run_id,
        "assignment_id": assignment_id,
        "assignment_sha256": receipt["assignment_sha256"],
        "token_digest": receipt["token_digest"],
        "expected_agent_id": expect_agent_id,
        # The receipt directory is gitignored working state; this append-only
        # ledger is the record that survives into committed evidence. Carrying
        # the binding mode here is what lets an auditor -- or validate_harness --
        # see after the fact which receipts were minted with no agent binding.
        "agent_binding": agent_binding,
        "reopened": bool(prior is not None),
        "at": minted_at,
        # D-070 round-6, item 2. validate_harness.py reconstructs this from
        # the append-only rows regardless -- that stays the authority, this
        # field does not replace it -- but stamping it here too makes the
        # intent auditable on the row itself instead of only inferable by
        # replaying the whole ledger.
        "chain_started_at": chain_started_at,
    }
    if agent_binding == "token-only" and args.reason.strip():
        row["agent_binding_reason"] = args.reason.strip()
    if prior is not None:
        superseded = last_consumed_result_sha256(ledger_path, assignment_id)
        # D-070 round-6, item 4: an empty declaration (nothing was ever
        # admitted for this assignment to supersede) used to be
        # indistinguishable on the row from a real one, because both are just
        # `superseded_result_sha256: ""` vs a real digest -- the emptiness
        # alone doesn't say whether that was found or merely defaulted to.
        # `supersedes_admitted_result` states which, explicitly, rather than
        # leaving it to be inferred by cross-referencing `superseded_state`.
        supersedes_admitted_result = bool(superseded)
        if not supersedes_admitted_result:
            # No consume row yet: the receipt being replaced was never admitted
            # (a dead agent, a failed stop). Fall back to whatever the receipt
            # itself recorded so the row is still explicit about what it found.
            superseded = str(prior.get("result_sha256") or "")
        row["superseded_result_sha256"] = superseded
        row["supersedes_admitted_result"] = supersedes_admitted_result
        row["superseded_state"] = str(prior.get("state"))
        row["superseded_consumed_by_agent_id"] = str(
            prior.get("consumed_by_agent_id") or ""
        )
        row["reason"] = args.reason.strip()

    # Ledger first, receipt second. If the ledger append fails the receipt was
    # never created, so no open receipt survives whose token was never printed
    # (D-067 round-2 review F-13).
    try:
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        fail(f"admission ledger could not be appended ({exc.__class__.__name__})")
    try:
        dump_json_atomic(target, receipt)
        release_claim_under_stop_lock(harness, run_id, prior, claim)
    except OSError as exc:
        fail(f"admission receipt could not be written ({exc.__class__.__name__})")

    print(f"RUN_ID={run_id}")
    print(f"ASSIGNMENT_ID={assignment_id}")
    print(f"CONTEXT_VERSION={context_version}")
    print(f"INDEPENDENCE_MODE={assignment.get('independence_mode')}")
    print(f"ADMISSION_TOKEN={token}")
    print(f"receipt: {target.relative_to(repo)}")


if __name__ == "__main__":
    main()
