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
permissive default: whichever state holds, this hook says so, in the injected
context, load-bearing in `Hook preflight: PASS|FAIL`.

ROUND-6 CORRECTION (obligation 2, still NOT DISCHARGED): the two states above
were argued to be exhaustive on the theory that Start "receives only agent_id
and agent_type ... never any statement of the parent's intent," so a failed
mint and a legitimately unregistered helper agent were "indistinguishable from
inside the hook" (docs/audit/BD622_D067_admission_binding_2026-07-29.md S2).
That is false. `admit_agent.py` writes the run's `ADMISSIONS.jsonl` ledger
*before* it writes the receipt file (see the "Ledger first, receipt second"
comment there); a `minted` (or `reopened`) row carries `expected_agent_id`,
and Start can read it just as it reads receipts. Reproduced: `chmod 0o500` on
`.agent-harness/admissions/<run>/` leaves the ledger append succeeding (it
lives under `.agent-harness/runs/<run>/`, a different directory) while the
receipt write fails, so a `minted` row survives with no receipt on disk. Before
this correction, `list_admission_matches` found zero files and this hook
reported "none found" / PASS -- exactly the obligation-2 case, passing.

ROUND-7 CORRECTION (obligation 2, still NOT DISCHARGED): the round-6 rule above
was written as universal and implemented as conditional. The ledger was consulted
only inside `if not matches:` -- i.e. only when this agent_id had NO receipt file
anywhere -- so two reproductions walked past it on the round-6 bytes:

  * ONE RECEIPT MASKS THE ORPHAN. Mint A for agent G fails its receipt write
    (`chmod 0o500` on the admissions dir), leaving an orphan `minted` row naming
    G. Mint B for the SAME agent G then *succeeds*, because `admit_agent.py`'s
    mint-time guard `other_assignment_bound_to_agent` scans receipt FILES and
    A's never landed. G now holds an open receipt for B while the ledger records
    an incomplete admission for A. Round-6 Start reported
    `Admission receipt: open, bound to A-...-2` and `Hook preflight: PASS` --
    the obligation-2 case, passing, with the stated fatal condition ("the
    ledger's latest minted/reopened row for some assignment names this agent_id,
    and no consumed row follows") HOLDING and never being evaluated.
  * TOKEN-ONLY MINTS WERE UNREACHABLE. `--agent-id-unknown` writes
    `expected_agent_id: ""` by construction, so a check keyed on that field can
    never match one. Reproduced, PASS, and documented nowhere.

The check is therefore now evaluated on EVERY path, and what "orphan" means had
to be re-derived once receipt files are in scope: "latest mint is unconsumed" on
its own is also the shape of every healthy, not-yet-consumed admission, so an
unqualified universal check would fail the admitted case the PASS arm exists for.
The predicate is per-assignment, and it asks whether the mint produced a receipt
that admits THIS agent for THAT assignment:

  * No active run, no agent_id, or an agent_id that fails the same safe-key
    regex the run lease uses -> the check is skipped and reported as "not
    checked", mirroring the existing lease behaviour for the same inputs
    exactly. A deliberate no-op, not a silent one: the reason is always
    printed.
  * Exactly one matching receipt, state "open", its own `run_id` field equal
    to the active run -> reported as admitted, and it CLEARS the ledger's open
    mint for that same assignment_id, and only that one.
  * Exactly one matching receipt but state != "open", or its `run_id` field
    disagrees with the active run -> hard FAIL. The receipt exists and names
    this agent but cannot admit it. It clears nothing.
  * More than one receipt claims the same `expected_agent_id`, or any receipt
    file in the run's admission directory (not only ones that turn out to
    match this agent_id) cannot be listed, read, or parsed as a JSON object
    -> hard FAIL. A parse failure on some *other* assignment's receipt is
    folded in too, deliberately: this hook cannot prove that an unparseable
    file was not meant for this agent_id, and an unprovable negative must not
    default to PASS. Ambiguity is a FAIL, not a skip.
  * The run's admission LEDGER is then consulted REGARDLESS of what the receipt
    scan found (round-7 correction; this used to sit under `if not matches:`):
      - the ledger cannot be read, or any line in it cannot be parsed as a JSON
        object -> hard FAIL, same "unprovable negative" logic as an unparseable
        receipt file. A malformed ledger line is never silently dropped.
      - a `minted`/`reopened` row that is the *latest* for its assignment, is
        not followed by a `consumed` row for that assignment, and names this
        `agent_id` -> hard FAIL, UNLESS the admitted receipt above is for that
        same assignment. This is the obligation-2 case in its universal form:
        holding a valid receipt for assignment B is not evidence that the mint
        for assignment A completed, and an agent bound to two assignments in
        one run is precisely what D-065 obligation 1 forbids. It is a FAIL for
        a *different* assignment than the one held, and a PASS for the same
        one -- not a blanket FAIL on "any unconsumed mint naming me", which
        would fail every healthy admitted agent.
      - the ledger does not exist, or holds no such row for this `agent_id`
        -> genuinely "none found", not fatal. This is the case the original
        two-state rule was right about: most spawned agents hold no registered
        assignment at all, and failing them would make the common case
        indistinguishable from the dangerous one.
    A `reopened` row is a later, more specific statement about the same
    assignment than an earlier `minted` row, so it supersedes that row for
    this purpose regardless of which agent either one names -- mirrors
    `admit_agent.py`'s own "ledger first, receipt second" replace semantics.
    A `consumed` row proves a receipt *did* exist on disk at some point (Stop
    only writes it after successfully claiming and validating an admission
    file), so its presence must never be read back as a failure -- that would
    resurrect a closed, successful admission as a phantom Start failure.

`--agent-id-unknown` (`expected_agent_id: ""`) -- STATED, NOT SILENT. No sound
agent key exists for these rows, and this is provable rather than merely
unattempted: the mode exists because the runtime has not yet assigned an agent
id, so at mint time the agent this receipt will admit does not exist, and neither
`run_id` nor `assignment_id` nor any other recorded field identifies one.
`admit_agent.py`'s own docstring reaches the same conclusion about its mint-time
guard ("not merely absent here, it is not constructible"). Start is worse off
still: the receipt is bound to the token, the token lives in the spawn prompt,
and no hook sees the prompt -- that is this module's founding premise.

So the row is checked at RUN scope instead of agent scope, and named as such: an
unconsumed token-only mint whose assignment has NO receipt file on disk at all is
a hard FAIL for whichever agent starts next, with the note and the error both
saying explicitly that the hook cannot tell whether this agent is the one it was
for. Three reasons this is the right call rather than a report-and-pass:
  1. It cannot fire in a healthy run. The row only survives without a receipt
     when the receipt write actually failed, and `admit_agent.py` exits non-zero
     without printing a token when it does -- so the parent already knows. The
     legitimate-unregistered-helper case (which stays non-fatal above) involves
     rows naming OTHER agents in a run whose mints all succeeded; it is not this.
  2. It self-clears through the ordinary recovery path. A successful re-mint
     appends a later row for the same assignment, which supersedes this one, and
     lands a receipt file, which clears it either way.
  3. Fail-closed on ambiguity is the standing rule here, and the alternative is
     the exact failure this obligation is about: the agent the mint was for does
     hours of work and is refused at Stop for a receipt that never existed.
The residual is stated rather than hidden: the FAIL is deliberately over-broad
WITHIN the run, because narrowing it would require the agent binding the mode
declines to record. `--expect-agent-id` -- which `admit_agent.py` requires unless
`--agent-id-unknown` is passed explicitly -- avoids the over-breadth entirely,
and Stop's per-`(run_id, agent_id)` consume lock remains the compensating control
for what this mode gives up.

This mirrors, but does not duplicate the authority of, subagent_stop_validate.py.
Start reports what it can see before the agent has produced a result (there is
no HARNESS_RESULT envelope yet to check `admission_proof` against); Stop still
performs the authoritative token/attribution check and the single-use consume.
Start must never contradict Stop's later verdict, so it only reads -- the
ledger consultation above is a read of an append-only file, same as the
receipt scan; it never claims, locks, or mutates anything.
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
) -> tuple[list[tuple[Path, dict[str, Any]]], set[str], list[str]]:
    """Read-only scan for receipts in ``admissions_dir`` bound to ``agent_id``.

    Returns ``(matches, assignment_ids_on_disk, errors)``. ``errors`` accumulates
    every read/parse failure encountered while scanning the directory, not only
    ones for files that turn out to match ``agent_id`` -- see the module
    docstring for why an unparseable receipt for a *different* assignment still
    has to fail closed here.

    ``assignment_ids_on_disk`` is every assignment a readable receipt file exists
    for, in any state and naming any agent (round-7). It is what clears a
    token-only (``--agent-id-unknown``) ledger row, which names no agent and so
    can only be asked the run-scope question "did this mint produce a receipt at
    all"; it is deliberately NOT what clears an agent-keyed row, because a
    receipt naming somebody else never admitted this agent.
    """
    matches: list[tuple[Path, dict[str, Any]]] = []
    on_disk: set[str] = set()
    errors: list[str] = []
    if admissions_dir.is_symlink():
        errors.append(f"admission directory path is a symlink: {admissions_dir.name}")
        return matches, on_disk, errors
    try:
        entries = sorted(admissions_dir.iterdir())
    except FileNotFoundError:
        return matches, on_disk, errors
    except OSError as exc:
        errors.append(
            f"admission receipts directory is unreadable ({exc.__class__.__name__})"
        )
        return matches, on_disk, errors
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
        on_disk.add(str(receipt.get("assignment_id") or entry.stem))
        if str(receipt.get("expected_agent_id") or "") == agent_id:
            matches.append((entry, receipt))
    return matches, on_disk, errors


def read_admission_ledger(ledger_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse the run's append-only admission ledger into ordered rows.

    Read-only, mirrors ``list_admission_matches``: fails closed on any garbage
    rather than skipping it, because a malformed line cannot be proven
    irrelevant to the ``agent_id`` being checked (round-6 finding -- see the
    module docstring). A missing ledger is not an error: it means nothing has
    ever been minted in this run, which callers must check for separately via
    ``ledger_path.is_file()`` before treating a read failure as fatal.
    """
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        raw = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"admission ledger is unreadable ({exc.__class__.__name__})")
        return rows, errors
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"admission ledger line {number} is malformed")
            continue
        if not isinstance(row, dict):
            errors.append(f"admission ledger line {number} is not an object")
            continue
        rows.append(row)
    return rows, errors


def open_mints(ledger_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Assignments whose latest recorded mint was never consumed -> its agent binding.

    Walks the ledger in file order (append-only, so this is chronological) and
    keeps, per ``assignment_id``, only the most recent ``minted``/``reopened``
    row -- a ``reopened`` row is a later and more specific statement about the
    same assignment, so it supersedes an earlier ``minted`` row regardless of
    which agent either one names. A ``consumed`` row for that assignment seen
    anywhere after such a row proves a receipt file did exist at some point
    (``subagent_stop_validate.py`` only appends ``consumed`` after claiming and
    validating an on-disk admission file), so it drops the assignment entirely:
    a legitimately consumed admission must never be resurrected as a failure.

    The value is that row's ``expected_agent_id``, which is ``""`` exactly when
    the mint was made with ``--agent-id-unknown``. Callers must not treat ``""``
    as an agent id -- ``ADMISSION_KEY_RE`` requires at least one character, so a
    real ``agent_id`` can never compare equal to it, and the token-only arm in
    ``orphaned_mints`` handles those rows separately and says so.
    """
    latest_agent: dict[str, str] = {}
    consumed: set[str] = set()
    for row in ledger_rows:
        assignment_id = str(row.get("assignment_id") or "")
        if not assignment_id:
            continue
        event = row.get("event")
        if event in ("minted", "reopened"):
            latest_agent[assignment_id] = str(row.get("expected_agent_id") or "")
            consumed.discard(assignment_id)
        elif event == "consumed" and assignment_id in latest_agent:
            consumed.add(assignment_id)
    return {
        assignment_id: expected
        for assignment_id, expected in latest_agent.items()
        if assignment_id not in consumed
    }


def orphaned_mints(
    ledger_rows: list[dict[str, Any]],
    agent_id: str,
    admitted_assignment_id: str,
    receipts_on_disk: set[str],
) -> tuple[list[str], list[str]]:
    """Mints the parent recorded but did not complete, split by what they name.

    Returns ``(bound_to_this_agent, token_only)``, both sorted assignment_ids
    (round-7 finding, D-065 obligation 2). Evaluated on every Start path, not
    only when the receipt scan came up empty -- that conditional evaluation is
    what let one valid receipt mask an orphaned mint for a different assignment.

    ``bound_to_this_agent``: the latest unconsumed mint for the assignment names
    this ``agent_id``, and ``admitted_assignment_id`` -- the assignment this
    agent holds an open, right-run receipt for, or ``""`` -- is not it. Excluding
    exactly that one assignment is what keeps the healthy admitted case a PASS:
    every open receipt has an unconsumed mint row behind it, so an unqualified
    "unconsumed mint names me" test would fail every properly admitted agent.
    Anything else that is on disk clears nothing, because a receipt naming
    another agent, or in a non-``open`` state, never admitted this one.

    ``token_only``: the latest unconsumed mint carries no ``expected_agent_id``
    (``--agent-id-unknown``) and NO receipt file exists for its assignment at
    all. There is no sound agent key for these rows -- see the module docstring
    for why one is not constructible rather than merely unimplemented -- so they
    are checked at run scope and reported as unattributable. A receipt file for
    that assignment in any state clears it: the question a token-only row can
    answer is only "did this mint produce a receipt", never "for whom".
    """
    latest = open_mints(ledger_rows)
    bound_to_this_agent = sorted(
        assignment_id
        for assignment_id, expected in latest.items()
        if expected == agent_id and assignment_id != admitted_assignment_id
    )
    token_only = sorted(
        assignment_id
        for assignment_id, expected in latest.items()
        if not expected and assignment_id not in receipts_on_disk
    )
    return bound_to_this_agent, token_only


def classify_receipt_matches(
    matches: list[tuple[Path, dict[str, Any]]], active_run: str, agent_id: str
) -> tuple[str, list[str], str]:
    """The receipt-file half of the verdict: ``(note, errors, admitted_assignment_id)``.

    Unchanged in substance from the round-6 rule -- open/bound is reported and
    not fatal, and consumed, wrong-run, or ambiguous is a hard FAIL. The third
    return value is the assignment this agent is genuinely admitted for, or
    ``""``; it is the only thing that clears an agent-keyed orphaned mint, so a
    FAIL verdict here deliberately returns ``""`` and clears nothing.
    """
    if not matches:
        return (
            f"none found for agent_id={agent_id!r} in run {active_run!r} "
            "(not every spawned agent holds a registered assignment; if this "
            "one is meant to submit a HARNESS_RESULT, SubagentStop still "
            "requires a matching receipt)",
            [],
            "",
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
            "",
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
            "",
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
            "",
        )
    return (
        f"open, bound to assignment {assignment_id!r} in run {active_run!r}",
        [],
        assignment_id,
    )


def describe_admission_receipt(
    harness: Path, active_run: str, agent_id: str
) -> tuple[str, list[str]]:
    """Start-time, read-only classification of the receipt bound to ``agent_id``.

    Returns ``(note, errors)``. ``note`` is always non-empty and always goes
    into the injected context (visibility is unconditional); ``errors`` feeds
    the same ``start_errors`` list the run-identity lease uses, so an unusable,
    ambiguous, or orphaned-mint receipt state is load-bearing in
    ``Hook preflight: PASS|FAIL`` the same way a lease-write failure already
    is. See the module docstring for the admitted / unusable / absent /
    orphaned-mint / unattributable-token-only rule.

    Both halves run on every path (round-7): the receipt scan classifies what is
    on disk, and the ledger is then consulted regardless of what it found, so a
    valid receipt for one assignment can no longer suppress the report of an
    incomplete mint for another. When the receipt verdict is already fatal it
    keeps its own note verbatim -- those AMBIGUOUS/UNUSABLE strings are the
    receipt-state rules' exact output -- and the orphan is carried in ``errors``,
    which is what ``Hook preflight: FAIL -- ...`` prints.
    """
    if not ADMISSION_KEY_RE.fullmatch(agent_id or ""):
        return "not checked (agent_id is not a safe admission key)", []

    admissions_dir = harness / "admissions" / active_run
    matches, receipts_on_disk, scan_errors = list_admission_matches(
        admissions_dir, agent_id
    )
    if scan_errors:
        detail = "; ".join(scan_errors)
        return (
            f"AMBIGUOUS ({detail})",
            [f"admission receipt state for agent_id={agent_id!r} is ambiguous: {detail}"],
        )

    # Round-7 finding: this read used to sit under `if not matches:`, so one
    # valid receipt for assignment B suppressed the whole check and an orphaned
    # mint for assignment A went unreported. The ledger is now consulted on
    # every path -- see the module docstring.
    ledger_rows: list[dict[str, Any]] = []
    ledger_path = harness / "runs" / active_run / "ADMISSIONS.jsonl"
    if ledger_path.is_file():
        ledger_rows, ledger_errors = read_admission_ledger(ledger_path)
        if ledger_errors:
            detail = "; ".join(ledger_errors)
            return (
                f"AMBIGUOUS (admission ledger for run {active_run!r} {detail})",
                [
                    f"admission ledger for run {active_run!r} is unreadable or "
                    f"malformed ({detail}); whether a mint for agent_id="
                    f"{agent_id!r} was ever intended cannot be ruled out"
                ],
            )

    note, errors, admitted_assignment_id = classify_receipt_matches(
        matches, active_run, agent_id
    )
    receipt_verdict_is_fatal = bool(errors)
    bound_orphans, token_only_orphans = orphaned_mints(
        ledger_rows, agent_id, admitted_assignment_id, receipts_on_disk
    )
    if not bound_orphans and not token_only_orphans:
        return note, errors

    problems: list[str] = []
    if bound_orphans:
        assignment_list = ", ".join(bound_orphans)
        held = (
            f"; the only receipt this agent does hold is for a different "
            f"assignment ({admitted_assignment_id!r})"
            if admitted_assignment_id
            else ""
        )
        problems.append(
            f"admission ledger records a minted/reopened row for "
            f"agent_id={agent_id!r} naming assignment(s) {assignment_list} in "
            f"run {active_run!r}, but no usable receipt exists on disk{held}"
        )
        errors.append(
            f"admission ledger shows the parent intended to admit "
            f"agent_id={agent_id!r} for assignment(s) {assignment_list} in run "
            f"{active_run!r}, but the mint did not produce a usable receipt "
            f"(missing, or overwritten by a failed reopen){held}"
        )
    if token_only_orphans:
        assignment_list = ", ".join(token_only_orphans)
        problems.append(
            f"run {active_run!r} has an unconsumed token-only "
            f"(--agent-id-unknown) minted/reopened ledger row for assignment(s) "
            f"{assignment_list} and no receipt file for it on disk; that row "
            "records no expected_agent_id, so this hook cannot tell whether "
            f"agent_id={agent_id!r} is the agent it was for"
        )
        errors.append(
            f"admission ledger for run {active_run!r} records an incomplete "
            f"token-only (--agent-id-unknown) mint for assignment(s) "
            f"{assignment_list}: the ledger row landed, no receipt file did, "
            "and the row names no expected_agent_id, so it cannot be ruled out "
            f"that agent_id={agent_id!r} is the agent that mint was for. This "
            "check is run-scoped by necessity, not by choice -- pass "
            "--expect-agent-id so a failed mint can be attributed to one agent"
        )

    # An already-fatal receipt verdict keeps its own note verbatim: the
    # AMBIGUOUS/UNUSABLE strings are the receipt-state rules' exact output and
    # must not be overwritten by this one. The orphan is still load-bearing --
    # its error is appended either way, and the errors are what `Hook preflight:
    # FAIL -- ...` prints.
    if not receipt_verdict_is_fatal:
        note = "MINT FAILED (" + ". ".join(problems) + ")"
    return note, errors


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
