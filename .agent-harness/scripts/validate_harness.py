#!/usr/bin/env python3
"""Structural validation of the agent harness.

Also emits the legacy results manifest that pins the pre-admission boundary:

    python3 .agent-harness/scripts/validate_harness.py --emit-legacy-manifest

See ``LEGACY_MANIFEST_REL`` below for what that manifest is for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

from _harness import (
    active_run_id,
    hash_files,
    load_json,
    loads_strict,
    render_context_pack,
    root,
    utc_now,
    validate_assignment_contract,
    validate_assignment_resource_hashes,
    validate_result_contract,
)

# D-067 round-4 review, defect 1. Attribution used to be checked only inside the
# active run, and only in the ledger->result direction, so a fabricated result
# dropped into any of the other run directories was invisible. The fix is to
# enumerate every result artifact on disk and demand that each one be either
# attributed by a consumed ledger row or pinned by this manifest.
#
# The manifest exists because most run directories predate the admission
# mechanism and have results but no ADMISSIONS.jsonl. Skipping those runs would
# reintroduce the hole (an attacker would simply drop the file in a ledger-less
# run), so instead the boundary is pinned: every result that existed in a
# ledger-less run when the manifest was generated is recorded with its digest.
# A new file fails wherever it is dropped, an edit to a legacy result fails, and
# genuine history passes.
LEGACY_MANIFEST_REL = ".agent-harness/context/LEGACY_RESULTS_MANIFEST.json"
LEGACY_MANIFEST_SCHEMA = 1
LEGACY_MANIFEST_CRITERION = (
    "every *.json in .agent-harness/runs/<run>/results/ for each run directory "
    "that had no ADMISSIONS.jsonl at generation time"
)
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

# D-070 round-5 review, finding 1. The manifest's git binding stopped at the
# working tree. `git status --porcelain` reports a *modified tracked* file, so
# planting a result and blessing it with a matching manifest entry failed only
# until the next `git commit`; after one commit the porcelain was clean again
# and the validator reported `ok (2 pinned)`. One commit defeated the pin.
#
# The manifest describes a CLOSED set -- the result artifacts that existed
# before the admission mechanism landed -- so it should never legitimately grow.
# Its identity is therefore pinned on two further surfaces that are checked by
# two different checkers:
#
#   * the entry COUNT is declared in SSOT_FACTS.json, which
#     check_ssot_consistency.py now MEASURES by counting the manifest's
#     `entries` array, not by reading its `entry_count` field back out of the
#     same file -- that older form only proved the file equalled itself. The
#     `entry_count` field is still bound separately, as a `current` assertion,
#     alongside the prose that quotes the number in FROZEN_DECISIONS.md and
#     VALIDATION_LEDGER.md;
#   * the canonical DIGEST of the entry set is pinned here, in the checker
#     itself, so swapping one entry for another at an unchanged count fails too.
#
# WHAT THIS DOES AND DOES NOT BUY. It does not defeat the accepted same-OS-user
# residual: an agent that deliberately writes outside its declared result path
# can still edit the manifest, this constant, SSOT_FACTS.json and the prose
# together. What it removes is the single-step laundering route above. Growing
# the pinned set now takes coordinated edits to four places across three
# independently checked surfaces (manifest bytes, this checker, the declared
# facts file, and the two prose surfaces the facts file binds), each covered by
# a different test suite. That is the whole claim: "one commit" becomes
# "coordinated edits across cross-checked surfaces", not "cannot be forged".
SSOT_FACTS_REL = ".agent-harness/context/SSOT_FACTS.json"
LEGACY_MANIFEST_PIN_FACT_ID = "legacy_results_manifest_entry_count"
# Canonical digest of the pinned entry set as of manifest head a1cdd8a. Recompute
# with `--emit-legacy-manifest`, which prints it, and expect the diff to be
# reviewed alongside the SSOT_FACTS.json value and the prose that quotes it.
LEGACY_MANIFEST_PINNED_SHA256 = (
    "sha256:14c89504abffed6b1c719bce34f3469670e634f7bf3908215b7552d060be18e8"
)

# D-070 round-5 review, finding F-R5-04. A result whose receipt is still `open`
# is tolerated so a live run is not wedged (round-2 finding F-12), but that
# tolerance used to be unbounded: mint a receipt through admit_agent.py, write a
# result, never stop, and those bytes were excused for ever and in whichever run
# directory the receipt sat in. The carve-out is now bounded to an admission
# that can still plausibly be IN FLIGHT -- see pending_carve_out_refusal.
PENDING_ADMISSION_MAX_AGE_HOURS = 24

# How a result artifact came to be trusted. Exactly two verdicts mean "a durable
# record vouches for these exact bytes":
ATTRIBUTED = "attributed"  # a consumed ADMISSIONS.jsonl row whose digest matches
LEGACY_PINNED = "legacy-pinned"  # pinned by LEGACY_RESULTS_MANIFEST.json
# ...and everything else is a reason not to trust them.
LEDGER_DIGEST_MISMATCH = "ledger-digest-mismatch"
LEGACY_DIGEST_MISMATCH = "legacy-digest-mismatch"
PENDING = "pending"  # open receipt, still inside the in-flight carve-out
CARVE_OUT_REFUSED = "carve-out-refused"
UNATTRIBUTED = "unattributed"
UNREADABLE = "unreadable"
# D-070 round-6 review, finding 2. merge_results.py globbed results/*.json and
# merged whatever it found, so unattributed or still-pending bytes reached
# MERGED_RESULTS.json while this validator reported green -- a second, unguarded
# path to treating a result as real. It now imports this set and
# collect_result_attribution below rather than restating the rule, so the merge
# gate and the validator cannot drift apart.
MERGEABLE_ATTRIBUTION = frozenset({ATTRIBUTED, LEGACY_PINNED})


def sha256_of(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_dirs(harness: Path) -> list[Path]:
    return sorted(p for p in harness.glob("runs/*") if p.is_dir())


def result_files(harness: Path) -> Iterator[tuple[str, str, Path]]:
    """Every result artifact on disk as ``(run_id, assignment_id, path)``.

    ``*.json`` at the top level of ``results/`` is exactly what merge_results.py
    consumes, so it is exactly what has to be attributable.
    """
    for run_dir in run_dirs(harness):
        for path in sorted((run_dir / "results").glob("*.json")):
            yield run_dir.name, path.stem, path


def read_ledger(ledger_path: Path) -> tuple[list[dict], list[str]]:
    """Parse an admission ledger into ordered rows. Fails closed on any garbage."""
    errors: list[str] = []
    rows: list[dict] = []
    try:
        raw = ledger_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(
            f"Admission ledger is unreadable ({exc.__class__.__name__}): "
            f"{ledger_path.parent.name}/{ledger_path.name}"
        )
        return rows, errors
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = loads_strict(line)
        except json.JSONDecodeError:
            errors.append(
                f"Malformed admission ledger line: {ledger_path.name}:{number}"
            )
            continue
        if not isinstance(row, dict):
            errors.append(
                f"Admission ledger line is not an object: {ledger_path.name}:{number}"
            )
            continue
        rows.append(row)
    return rows, errors


def check_declared_supersession(run_name: str, rows: list[dict]) -> list[str]:
    """A second consume row is admissible only if a reopen row declared it.

    D-067 round-4 review, defect 2. The ledger used to be reduced with
    ``ledger_consumed[aid] = row``, so the last consume row silently won and the
    earlier one -- the permanent statement about the bytes originally admitted --
    was discarded. Combined with ``--reopen`` writing an indistinguishable
    ``"minted"`` row, and with the O_EXCL ``.claim`` file being gitignored and so
    deletable without a git trace, that allowed an edited artifact to be
    re-admitted and the validator to go green again.

    Walking the append-only rows in order restores the pin: superseding an
    admitted digest must be *declared* by an ``admit_agent.py --reopen`` row
    naming that digest and carrying a reason, not merely performed.
    """
    errors: list[str] = []
    last_consume: dict[str, dict] = {}
    reopens_since_consume: dict[str, list[dict]] = {}
    for row in rows:
        aid = str(row.get("assignment_id"))
        event = row.get("event")
        if event == "reopened":
            reopens_since_consume.setdefault(aid, []).append(row)
        elif event == "consumed":
            prior = last_consume.get(aid)
            if prior is not None:
                prior_sha = str(prior.get("result_sha256"))
                declared = [
                    entry
                    for entry in reopens_since_consume.get(aid, [])
                    if str(entry.get("superseded_result_sha256")) == prior_sha
                    and str(entry.get("reason") or "").strip()
                ]
                if not declared:
                    errors.append(
                        "Admission ledger records a second consume with no declared "
                        f"reopen between them: {run_name}/{aid} admitted "
                        f"{prior_sha} and then {str(row.get('result_sha256'))}. "
                        "Supersession must be declared with admit_agent.py "
                        "--reopen --reason '...'."
                    )
            last_consume[aid] = row
            reopens_since_consume[aid] = []
    return errors


def check_one_agent_one_assignment(run_name: str, rows: list[dict]) -> list[str]:
    """One spawned ``agent_id`` consumes at most one assignment per run.

    D-065 obligation 1, round-6. admit_agent.py refuses to mint a second receipt
    for an agent that already holds one, and subagent_stop_validate.py refuses to
    consume it -- but neither left a DETECTOR behind. The Stop-time guard read
    the ledger without a lock over the append that follows, and the O_EXCL claim
    file is keyed per assignment, so two concurrent stops by one agent on two
    assignments took two different claims and raced: both were admitted in 126 of
    200 measured trials on pristine bytes, and the resulting tree validated
    ``ok: true`` in 200 of 200. A live guard that is bypassed leaves no trace
    unless something reads the record afterwards. The ledger already says who
    consumed what, so the violation is visible after the fact -- whether the race
    was won, lost, or the guard was never there.

    Walking the append-only rows in order, an agent is bound to the FIRST
    assignment it consumes. A later consume of a DIFFERENT assignment by the same
    agent is an error unless the binding was released in between by an
    ``admit_agent.py --reopen`` row for the assignment that agent holds, naming
    it in ``superseded_consumed_by_agent_id`` and carrying a reason. That is the
    same declared-supersession gesture check_declared_supersession requires, read
    for a different question: there, may these bytes be replaced; here, is this
    agent free to be bound somewhere else. An undeclared reopen, one naming
    another agent, or one for another assignment releases nothing.

    Re-consuming the SAME assignment is deliberately not this check's business: a
    resumed agent re-stopping is ordinary, and whether a second consume is
    licensed at all is check_declared_supersession's question, asked of the same
    rows. A consume row with no ``agent_id`` is an error here, because an
    unattributable consume cannot be shown NOT to be a second one.
    """
    errors: list[str] = []
    bound: dict[str, str] = {}
    for row in rows:
        event = row.get("event")
        assignment_id = str(row.get("assignment_id"))
        if event == "reopened":
            released_agent = str(row.get("superseded_consumed_by_agent_id") or "")
            reason = str(row.get("reason") or "").strip()
            if (
                released_agent
                and reason
                and bound.get(released_agent) == assignment_id
            ):
                del bound[released_agent]
        elif event == "consumed":
            agent_id = str(row.get("agent_id") or "")
            if not agent_id:
                errors.append(
                    "Admission ledger records a consume with no agent_id: "
                    f"{run_name}/{assignment_id}. An unattributable consume "
                    "cannot be shown not to be a second one for an agent that "
                    "already holds an assignment."
                )
                continue
            held = bound.get(agent_id)
            if held is not None and held != assignment_id:
                errors.append(
                    f"Admission ledger records agent {agent_id!r} consuming two "
                    f"assignments in run {run_name}: {held} and then "
                    f"{assignment_id}, with no declared release between them. One "
                    "agent_id binds to one assignment per run (D-065 obligation "
                    "1); releasing it takes an admit_agent.py --reopen --reason "
                    f"'...' row for {held} naming that agent."
                )
                continue
            bound[agent_id] = assignment_id
    return errors


def load_legacy_manifest(
    repo: Path,
) -> tuple[dict[tuple[str, str], str], str, list[str]]:
    """Read the pinned legacy boundary. Absent means nothing is grandfathered."""
    errors: list[str] = []
    path = repo / LEGACY_MANIFEST_REL
    if not path.is_file():
        # Strictest reading, and the fail-closed direction: with no manifest
        # every result must carry a ledger row. Deleting the manifest therefore
        # tightens the rule rather than relaxing it, and the git check below
        # turns deleting a committed one into an error of its own.
        return {}, "absent (no legacy results are grandfathered)", errors
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(
            f"Legacy results manifest is unreadable ({exc.__class__.__name__}); "
            "it is required for result attribution."
        )
        return {}, "unreadable", errors
    try:
        payload = loads_strict(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"Legacy results manifest is not valid JSON: {exc}")
        return {}, "unparseable", errors
    if not isinstance(payload, dict):
        errors.append("Legacy results manifest is not a JSON object.")
        return {}, "malformed", errors
    if payload.get("schema_version") != LEGACY_MANIFEST_SCHEMA:
        errors.append(
            "Legacy results manifest has an unsupported schema_version "
            f"{payload.get('schema_version')!r}; expected {LEGACY_MANIFEST_SCHEMA}."
        )
        return {}, "malformed", errors
    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("Legacy results manifest has no `entries` list.")
        return {}, "malformed", errors
    index: dict[tuple[str, str], str] = {}
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"Legacy results manifest entry {number} is not an object.")
            continue
        run_id = entry.get("run_id")
        assignment_id = entry.get("assignment_id")
        digest = entry.get("sha256")
        if not (
            isinstance(run_id, str)
            and run_id
            and isinstance(assignment_id, str)
            and assignment_id
            and isinstance(digest, str)
            and SHA256_RE.fullmatch(digest)
        ):
            errors.append(f"Legacy results manifest entry {number} is malformed.")
            continue
        key = (run_id, assignment_id)
        if key in index:
            errors.append(
                f"Duplicate legacy results manifest entry: {run_id}/{assignment_id}"
            )
            continue
        index[key] = digest
    # `entry_count` is the field the declared-facts cross-check reads, so it has
    # to agree with the entries it counts, or the two checkers would be looking
    # at different numbers. Checked after the entries themselves so a malformed
    # entry is still named.
    declared_count = payload.get("entry_count")
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        errors.append("Legacy results manifest has no integer `entry_count`.")
        return {}, "malformed", errors
    if declared_count != len(entries):
        errors.append(
            f"Legacy results manifest declares entry_count {declared_count} but "
            f"carries {len(entries)} entries."
        )
        return {}, "malformed", errors
    return index, f"ok ({len(index)} pinned)", errors


def canonical_legacy_digest(index: dict[tuple[str, str], str]) -> str:
    """Digest of the pinned SET, independent of the manifest's formatting.

    Taken from the parsed index rather than the file bytes so that reindenting
    or reordering the JSON is not an integrity failure, while adding, removing
    or swapping any pinned artifact is.
    """
    payload = "\n".join(
        f"{run_id}\0{assignment_id}\0{digest}"
        for (run_id, assignment_id), digest in sorted(index.items())
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_tracked(repo: Path, rel: str) -> tuple[bool | None, str]:
    """``(is_tracked, failure)``; ``failure`` is an exception class name or ""."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, exc.__class__.__name__
    return completed.returncode == 0, ""


def declared_legacy_entry_count(repo: Path) -> tuple[int | None, list[str]]:
    """The manifest's entry count as declared on the cross-checked SSOT surface.

    Returns ``None`` when SSOT_FACTS.json is absent, which is a synthetic tree
    with no second surface to cross-check against -- the same applicability rule
    the gate registry uses below. Deleting a *tracked* facts file to reach that
    branch is closed here, exactly as the registry-deletion bypass is.
    """
    errors: list[str] = []
    path = repo / SSOT_FACTS_REL
    if not path.is_file():
        tracked, failure = git_tracked(repo, SSOT_FACTS_REL)
        if failure:
            errors.append(
                "Could not determine whether the declared-facts file is tracked "
                f"({failure}); git is required for validation."
            )
        elif tracked:
            errors.append(
                "Declared-facts file is tracked in git but missing from the "
                "working tree; the legacy results manifest pin cannot be "
                "disabled by deleting its input."
            )
        return None, errors
    try:
        payload = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(
            f"Declared-facts file is unreadable or invalid JSON "
            f"({exc.__class__.__name__}); the legacy results manifest pin cannot "
            "be checked."
        )
        return None, errors
    facts = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(facts, list):
        errors.append("Declared-facts file has no `facts` list.")
        return None, errors
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        if fact.get("fact_id") != LEGACY_MANIFEST_PIN_FACT_ID:
            continue
        value = fact.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(
                f"Declared fact {LEGACY_MANIFEST_PIN_FACT_ID!r} has no integer "
                "value; the legacy results manifest pin cannot be checked."
            )
            return None, errors
        return value, errors
    errors.append(
        f"Declared-facts file declares no {LEGACY_MANIFEST_PIN_FACT_ID!r}; the "
        "legacy results manifest would then be pinned by nothing outside itself."
    )
    return None, errors


def check_legacy_manifest_pin(
    repo: Path,
    legacy_index: dict[tuple[str, str], str],
    legacy_status: str,
) -> tuple[str, list[str]]:
    """Cross-check the manifest's identity against the declared count and digest."""
    errors: list[str] = []
    declared, pin_errors = declared_legacy_entry_count(repo)
    errors.extend(pin_errors)
    if declared is None:
        return legacy_status, errors
    if not legacy_status.startswith("ok"):
        errors.append(
            f"Declared-facts file pins {declared} legacy result artifacts, but the "
            f"legacy results manifest is {legacy_status}. The manifest cannot be "
            "removed or broken to drop the pin."
        )
        return legacy_status, errors
    if declared != len(legacy_index):
        errors.append(
            f"Legacy results manifest holds {len(legacy_index)} entries but the "
            f"declared-facts file pins {declared}. The manifest describes a closed "
            "pre-admission set; growing it requires the declared count, the prose "
            "that quotes it, and the digest pinned in validate_harness.py to move "
            "together."
        )
    actual = canonical_legacy_digest(legacy_index)
    if actual != LEGACY_MANIFEST_PINNED_SHA256:
        errors.append(
            "Legacy results manifest entries do not match the digest pinned in "
            f"validate_harness.py: {actual} != {LEGACY_MANIFEST_PINNED_SHA256}. "
            "An entry was added, removed or altered."
        )
    if not errors:
        legacy_status += f"; pinned by declared fact and digest ({declared})"
    return legacy_status, errors


def open_admission_chain_start(rows: list[dict], assignment_id: str) -> str | None:
    """When the assignment's currently-OPEN admission chain was first minted.

    D-070 round-6 review, finding F-R6-02. The in-flight ceiling used to be
    measured from the receipt's own ``created_at``, and ``admit_agent.py`` writes
    ``"created_at": utc_now()`` unconditionally on every mint, ``--reopen``
    included. One documented, non-forged reopen therefore reset the clock on the
    exact same never-verified bytes, and could do so indefinitely: the ceiling
    was not a bound. The append-only ledger is the record that survives into
    committed evidence, so the age is taken from it instead.

    A *chain* is the run of mint rows for one assignment that no consume row has
    closed yet:

    * a ``minted``/``reopened`` row arriving when no chain is open STARTS one,
      and its ``at`` is what the ceiling is measured from;
    * a further ``minted``/``reopened`` row arriving while a chain is still open
      is a re-mint of an admission that was NEVER consumed -- the abuse -- so the
      first timestamp is kept and the clock does not restart;
    * a ``consumed`` row CLOSES the chain. A reopen after it supersedes bytes
      that really were admitted, which is a declared supersession
      (check_declared_supersession makes it name the digest it replaces) and a
      legitimate fresh admission, so the next mint starts a new chain whose clock
      legitimately starts over.

    Returns None when no chain is open: no mint row at all, or the last event for
    this assignment was a consume. ``admit_agent.py`` appends the ledger row
    *before* writing the receipt, so an open receipt in that state is left-over
    or planted rather than in flight, and the caller refuses the carve-out.
    """
    started: str | None = None
    for row in rows:
        if str(row.get("assignment_id")) != assignment_id:
            continue
        event = row.get("event")
        if event == "consumed":
            started = None
        elif event in ("minted", "reopened") and started is None:
            started = str(row.get("at") or "")
    return started


def pending_carve_out_refusal(
    harness: Path,
    run_name: str,
    assignment_id: str,
    receipt: dict,
    active: str | None,
    now: datetime,
    ledger_rows: list[dict] | None,
) -> str | None:
    """Why this open receipt does NOT excuse a result, or None when it does.

    D-070 round-5 review, finding F-R5-04. The carve-out exists so a genuinely
    live run is not wedged between the moment an agent writes its artifact and
    the moment SubagentStop consumes its receipt (round-2 finding F-12). That is
    a window inside ONE run -- the live one -- and it closes in minutes, or in
    however long a blocked stop takes to be retried. Everything below states
    that window and nothing wider, so a live run still cannot be wedged:

    * the run must be the ACTIVE one. admit_agent.py mints only for the active
      run, so an open receipt in any other run directory is left-over state, not
      an agent in flight. Before this, the global forward sweep let one such
      receipt excuse a result in that run directory indefinitely.
    * the receipt must name the assignment whose result it is excusing, so a
      receipt cannot be moved to cover a different artifact.
    * the assignment must still be registered in the active run. Nothing can be
      in flight for an assignment the run does not have.
    * the admission must be younger than PENDING_ADMISSION_MAX_AGE_HOURS on BOTH
      surfaces that date it: the receipt's own ``created_at``, and the first
      mint of its currently-open chain on the append-only ledger. This is what
      turns "for ever" into a bound. The ceiling sits far above any real
      write-to-stop window -- the artifact is written immediately before the
      stop, and only a blocked stop being retried holds it open -- so an agent
      genuinely in flight is not wedged by it. An agent sitting on an unadmitted
      artifact for over a day is not in flight, and saying so is the point.

    Reading BOTH timestamps is D-070 round-6 finding F-R6-02, and the two are
    load-bearing in opposite directions. The receipt bound catches a receipt
    back-dated or forward-dated against its ledger. The ledger bound catches the
    reported defect: admit_agent.py stamps ``created_at`` afresh on every mint,
    ``--reopen`` included, so one documented, non-forged reopen used to reset
    the ceiling on the same never-verified bytes, indefinitely. Because a
    refusal is returned when EITHER surface is out of bounds, the rule is
    strictly narrower than the one it replaces -- nothing that was refused
    before is excused now -- and the only thing that legitimately restarts the
    clock is a consume, i.e. bytes actually being admitted. See
    open_admission_chain_start for why a reopen after a genuine consume still
    starts a fresh chain, and a reopen of a never-consumed receipt does not.

    This does not make an unattributed result safe and does not claim to: while
    the carve-out holds, the bytes are tolerated, not verified. What changes is
    that the tolerance is scoped to one run, tied to a registered assignment,
    capped in time against a record the holder cannot refresh, and reported with
    its digest and a count.
    """
    if active is None or run_name != active:
        return (
            f"its open admission receipt sits in run {run_name!r}, which is not "
            f"the active run ({active!r}); admit_agent.py mints only for the "
            "active run, so an open receipt anywhere else is left-over state"
        )
    named = (str(receipt.get("run_id")), str(receipt.get("assignment_id")))
    if named != (run_name, assignment_id):
        return (
            f"its open admission receipt names {named[0]}/{named[1]}, not the "
            "artifact it is being read to excuse"
        )
    registered = harness / "runs" / run_name / "assignments" / f"{assignment_id}.json"
    if not registered.is_file():
        return (
            "no assignment is registered for it in the active run, so no agent "
            "can be in flight for it"
        )
    raw = str(receipt.get("created_at") or "")
    try:
        created = datetime.fromisoformat(raw)
    except ValueError:
        created = None
    if created is None or created.tzinfo is None:
        return (
            f"its open admission receipt carries no parseable UTC created_at "
            f"({raw!r}), so how long it has been open cannot be established"
        )
    age = now - created
    if age < timedelta(0):
        return (
            f"its open admission receipt is dated {raw}, in the future; the "
            "carve-out cannot be extended by dating a receipt forward"
        )
    if age > timedelta(hours=PENDING_ADMISSION_MAX_AGE_HOURS):
        return (
            f"its open admission receipt has been open for {age.total_seconds() / 3600:.1f}h, "
            f"past the {PENDING_ADMISSION_MAX_AGE_HOURS}h in-flight ceiling. Let "
            "the agent stop and consume the admission, or delete the artifact"
        )
    # The ledger half of the bound (F-R6-02). Everything above reads the receipt,
    # which the holder can re-mint; this reads the append-only record, which it
    # cannot.
    if ledger_rows is None:
        return (
            "its run has no readable admission ledger, so the first mint of its "
            "admission chain -- which is what the in-flight ceiling is measured "
            "from -- cannot be established"
        )
    chain_raw = open_admission_chain_start(ledger_rows, assignment_id)
    if chain_raw is None:
        return (
            "the append-only admission ledger records no OPEN mint for it: either "
            "there is no mint row at all, or its last ledger event is a consume. "
            "admit_agent.py appends the ledger row before it writes the receipt, "
            "so an open receipt in that state is left-over or planted rather than "
            "in flight"
        )
    try:
        chain_start = datetime.fromisoformat(chain_raw)
    except ValueError:
        chain_start = None
    if chain_start is None or chain_start.tzinfo is None:
        return (
            "the admission ledger row that opened its admission chain carries no "
            f"parseable UTC timestamp ({chain_raw!r}), so how long that chain has "
            "been open cannot be established"
        )
    chain_age = now - chain_start
    if chain_age < timedelta(0):
        return (
            f"the admission ledger row that opened its admission chain is dated "
            f"{chain_raw}, in the future; the carve-out cannot be extended by "
            "dating a mint forward"
        )
    if chain_age > timedelta(hours=PENDING_ADMISSION_MAX_AGE_HOURS):
        return (
            "its admission chain has been open since "
            f"{chain_raw} ({chain_age.total_seconds() / 3600:.1f}h), past the "
            f"{PENDING_ADMISSION_MAX_AGE_HOURS}h in-flight ceiling. The ceiling is "
            "measured from the FIRST mint of the chain on the append-only ledger, "
            "not from the receipt's created_at, which admit_agent.py rewrites on "
            "every mint including --reopen: re-minting a receipt that was never "
            "consumed does not restart the clock. Let the agent stop and consume "
            "the admission, or delete the artifact"
        )
    return None


def git_porcelain_errors(repo: Path, pathspec: str, label: str) -> list[str]:
    """Report tracked-file drift under ``pathspec``. Absent git is an error."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", pathspec],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # Fail closed: an unrunnable integrity check is not a passed one.
        return [
            f"{label} integrity check could not run "
            f"({exc.__class__.__name__}); git is required for validation."
        ]
    # Only tracked-file changes matter here: `??` entries are new paths that have
    # not been force-added yet, which is the normal state of a live run and of a
    # freshly generated manifest (round-2 review F-8).
    return [
        f"{label} modified after commit: {line.strip()}"
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("??")
    ]


def emit_legacy_manifest(repo: Path, force: bool) -> None:
    """Regenerate LEGACY_RESULTS_MANIFEST.json from the tree as it stands now."""
    harness = repo / ".agent-harness"
    target = repo / LEGACY_MANIFEST_REL
    if target.exists() and not force:
        raise SystemExit(
            f"validate_harness: {LEGACY_MANIFEST_REL} already exists. The pinned "
            "boundary is meant to be written once; pass --force only when you "
            "intend to re-pin it, and expect the diff to be reviewed."
        )
    entries: list[dict[str, str]] = []
    for run_dir in run_dirs(harness):
        if (run_dir / "ADMISSIONS.jsonl").is_file():
            continue
        for path in sorted((run_dir / "results").glob("*.json")):
            entries.append(
                {
                    "run_id": run_dir.name,
                    "assignment_id": path.stem,
                    "sha256": sha256_of(path),
                }
            )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "validate_harness: could not record the HEAD commit for the legacy "
            f"manifest ({exc.__class__.__name__}); git is required."
        ) from exc
    payload = {
        "schema_version": LEGACY_MANIFEST_SCHEMA,
        "purpose": (
            "Pin the pre-admission boundary. validate_harness.py requires every "
            "result artifact in every run directory to be either attributed by a "
            "consumed ADMISSIONS.jsonl row whose digest matches the bytes on "
            "disk, or listed here with a matching digest. Anything else is an "
            "error, so a fabricated result fails wherever it is dropped and an "
            "edit to a legacy result fails too."
        ),
        "generated_by": (
            "python3 .agent-harness/scripts/validate_harness.py "
            "--emit-legacy-manifest"
        ),
        "generated_at": utc_now(),
        "generated_at_head": head,
        "selection_criterion": LEGACY_MANIFEST_CRITERION,
        "entry_count": len(entries),
        "entries": sorted(
            entries, key=lambda entry: (entry["run_id"], entry["assignment_id"])
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    canonical = canonical_legacy_digest(
        {(entry["run_id"], entry["assignment_id"]): entry["sha256"] for entry in entries}
    )
    print(
        json.dumps(
            {
                "ok": True,
                "wrote": LEGACY_MANIFEST_REL,
                "entry_count": len(entries),
                "generated_at_head": head,
                "sha256": sha256_of(target),
                "canonical_entries_sha256": canonical,
                "next": (
                    "Re-pinning the boundary is a cross-surface edit: set "
                    f"LEGACY_MANIFEST_PINNED_SHA256 in validate_harness.py to "
                    f"{canonical}, set the "
                    f"{LEGACY_MANIFEST_PIN_FACT_ID!r} value in {SSOT_FACTS_REL} "
                    f"to {len(entries)}, and update the prose that quotes it."
                ),
            },
            indent=2,
        )
    )


class AttributionReport(NamedTuple):
    """What every result artifact on disk is attributed by, and why not."""

    errors: list[str]
    open_admissions: list[str]
    pending_results: list[dict[str, str]]
    legacy_status: str
    # (run_id, assignment_id) -> one of the verdict constants above. Only
    # MERGEABLE_ATTRIBUTION members mean a durable record vouches for the bytes.
    status: dict[tuple[str, str], str]
    # The digest each verdict was reached about, so a caller that re-reads the
    # file can prove it is looking at the same bytes this pass judged.
    digest: dict[tuple[str, str], str]


def collect_result_attribution(
    repo: Path,
    harness: Path,
    *,
    active: str | None,
    now: datetime,
) -> AttributionReport:
    """Decide, for every result artifact on disk, what vouches for its bytes.

    This is the single implementation of the attribution rule. ``main`` folds its
    errors into the validator's verdict; ``merge_results.py`` imports it and
    refuses to merge anything whose verdict is not in MERGEABLE_ATTRIBUTION
    (D-070 round-6 review, finding 2 -- the merge used to glob ``results/*.json``
    and reduce whatever it found, so a still-pending or wholly unattributed
    artifact reached MERGED_RESULTS.json while this validator reported green).
    Both callers therefore enforce the same rule by construction, which a second
    implementation could not guarantee.

    The rule: every result artifact is attributed by a consumed ledger row whose
    digest matches, or pinned by the legacy manifest with a matching digest, or
    belongs to an admitted agent that has not stopped yet. Nothing else passes.
    """
    errors: list[str] = []
    open_admissions: list[str] = []
    open_receipts: dict[tuple[str, str], dict] = {}
    consumed: dict[tuple[str, str], dict] = {}
    status: dict[tuple[str, str], str] = {}
    digest: dict[tuple[str, str], str] = {}
    for path in sorted((harness / "admissions").glob("*/*.json")):
        if path.name.startswith(".tmp."):
            continue
        try:
            receipt = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            errors.append(f"Admission receipt is unreadable: {path.relative_to(repo)}")
            continue
        if not isinstance(receipt, dict):
            errors.append(f"Admission receipt is not an object: {path.relative_to(repo)}")
            continue
        state = str(receipt.get("state"))
        if state == "open":
            open_admissions.append(f"{path.parent.name}/{path.stem}")
            open_receipts[(path.parent.name, path.stem)] = receipt
        elif state == "consumed":
            consumed[(path.parent.name, path.stem)] = receipt

    # Every result in the active run must be attributable: a consumed receipt
    # whose recorded digest still matches the bytes on disk. This is what
    # detects a fabricated or post-hoc edited result artifact, which the
    # git-porcelain check in main cannot see for an untracked new file
    # (D-067 review F-D067-08).
    pending_results: list[dict[str, str]] = []
    if active:
        for result_path in sorted((harness / "runs" / active / "results").glob("*.json")):
            receipt = consumed.get((active, result_path.stem))
            if receipt is None:
                if (active, result_path.stem) in open_receipts:
                    # The agent may still be in flight. Whether that actually
                    # excuses these bytes is decided in exactly one place, the
                    # forward sweep below, so the carve-out cannot be granted
                    # here and bounded there (D-070 round-5 review F-R5-04).
                    continue
                errors.append(
                    "Result artifact has no admission receipt at all: "
                    f"{result_path.relative_to(repo)}"
                )
                continue
            try:
                result_sha = sha256_of(result_path)
            except OSError as exc:
                # An artifact whose bytes cannot be read cannot be reconciled
                # against the digest that admitted it. Reported, not raised: a
                # crash is fail-closed but unreadable, and every caller that
                # parses this script's JSON sees an empty stdout (the
                # D-070 F-R5-08 class).
                errors.append(
                    f"Result artifact could not be read ({exc.__class__.__name__}): "
                    f"{result_path.relative_to(repo)}"
                )
                continue
            if str(receipt.get("result_sha256")) != result_sha:
                errors.append(
                    "Result artifact changed after it was admitted: "
                    f"{result_path.relative_to(repo)}"
                )

    # The receipts above are gitignored working state; the record that survives
    # into committed evidence is the per-run append-only ledger. Cross-check them
    # against each other, so a consumed receipt with no ledger line -- or a
    # ledger line contradicting the receipt -- is visible (round-2 review F-7).
    attributed: dict[tuple[str, str], dict] = {}
    # Rows are kept per run for the in-flight ceiling below, which is measured
    # from the ledger and not from the receipt (F-R6-02). A run whose ledger is
    # missing, unreadable, or carries a malformed line gets no entry at all, so
    # the carve-out is refused rather than granted on an unparsed record.
    ledger_rows_by_run: dict[str, list[dict]] = {}
    for run_dir in run_dirs(harness):
        ledger_path = run_dir / "ADMISSIONS.jsonl"
        run_receipts = {
            aid: receipt for (run, aid), receipt in consumed.items() if run == run_dir.name
        }
        if not ledger_path.is_file():
            errors.extend(
                f"Consumed receipt has no admission ledger: {run_dir.name}/{aid}"
                for aid in sorted(run_receipts)
            )
            continue
        rows, ledger_errors = read_ledger(ledger_path)
        errors.extend(ledger_errors)
        if not ledger_errors:
            ledger_rows_by_run[run_dir.name] = rows
        # Every consume row is retained, not just the last one: collapsing them
        # is what let a second consume quietly overwrite the digest the first had
        # permanently pinned (D-067 round-4 review, defect 2).
        errors.extend(check_declared_supersession(run_dir.name, rows))
        # The other question the same rows answer: not "may these bytes be
        # replaced" but "did one agent consume two assignments" (D-065
        # obligation 1, round-6). Both are read from the append-only record, so
        # a live guard that was raced or bypassed is still caught here.
        errors.extend(check_one_agent_one_assignment(run_dir.name, rows))
        ledger_consumed: dict[str, dict] = {}
        for ledger_row in rows:
            if ledger_row.get("event") == "consumed":
                ledger_consumed[str(ledger_row.get("assignment_id"))] = ledger_row
        attributed.update(
            ((run_dir.name, aid), row) for aid, row in ledger_consumed.items()
        )
        # Reverse reconciliation, ledger -> result. The receipt-driven check above
        # is defeated by `admit_agent.py --reopen`, which replaces a consumed
        # receipt with an open one and so demotes an edited artifact to merely
        # "pending" (D-067 round-3 review F-R3-10). The ledger is append-only, so
        # a consume row is a permanent statement about specific bytes: once an
        # assignment has been admitted, its result may never differ from the last
        # row recorded for it, whatever the receipt now says.
        for aid, row in sorted(ledger_consumed.items()):
            result_file = run_dir / "results" / f"{aid}.json"
            if not result_file.is_file():
                errors.append(
                    "Admitted result artifact is missing: "
                    f"{run_dir.name}/{aid}"
                )
                continue
            try:
                ledger_sha = sha256_of(result_file)
            except OSError as exc:
                status[(run_dir.name, aid)] = UNREADABLE
                errors.append(
                    f"Admitted result artifact could not be read "
                    f"({exc.__class__.__name__}): {run_dir.name}/{aid}"
                )
                continue
            digest[(run_dir.name, aid)] = ledger_sha
            if str(row.get("result_sha256")) != ledger_sha:
                status[(run_dir.name, aid)] = LEDGER_DIGEST_MISMATCH
                errors.append(
                    "Result artifact differs from the bytes recorded in the "
                    f"admission ledger: {run_dir.name}/{aid}"
                )
            else:
                status[(run_dir.name, aid)] = ATTRIBUTED

        for aid, receipt in sorted(run_receipts.items()):
            row = ledger_consumed.get(aid)
            if row is None:
                errors.append(
                    f"Consumed receipt has no ledger entry: {run_dir.name}/{aid}"
                )
                continue
            for field in ("result_sha256", "token_digest"):
                if str(row.get(field)) != str(receipt.get(field)):
                    errors.append(
                        f"Admission ledger disagrees with the receipt on {field}: "
                        f"{run_dir.name}/{aid}"
                    )
            if str(row.get("agent_id")) != str(receipt.get("consumed_by_agent_id")):
                errors.append(
                    "Admission ledger disagrees with the receipt on the writing "
                    f"agent: {run_dir.name}/{aid}"
                )

    # The legacy manifest is the other half of the attribution rule, so it needs
    # the same append-only treatment as retained run evidence: once committed,
    # editing it to bless an extra file, or deleting it, is itself an error.
    errors.extend(
        git_porcelain_errors(repo, LEGACY_MANIFEST_REL, "Legacy results manifest")
    )
    legacy_index, legacy_status, legacy_errors = load_legacy_manifest(repo)
    errors.extend(legacy_errors)
    if legacy_status.startswith("ok"):
        # The git check above only binds a *tracked* manifest: an untracked one
        # shows as `??` and is tolerated, exactly as a new run directory is. Say
        # so out loud rather than letting an uncommitted pin read as a sealed
        # one -- while the manifest is uncommitted, an added entry leaves no git
        # trace.
        tracked, failure = git_tracked(repo, LEGACY_MANIFEST_REL)
        if failure:
            errors.append(
                "Could not determine whether the legacy results manifest is "
                f"tracked ({failure}); git is required for "
                "validation."
            )
        elif not tracked:
            legacy_status += "; UNCOMMITTED, so the pin is not yet git-bound"
    # ...and being tracked is not enough either, because `git commit` clears the
    # porcelain check. The manifest's own identity is cross-checked against the
    # declared count and the pinned digest; see LEGACY_MANIFEST_PINNED_SHA256.
    legacy_status, pin_errors = check_legacy_manifest_pin(
        repo, legacy_index, legacy_status
    )
    errors.extend(pin_errors)

    # Forward reconciliation, result -> attribution, across EVERY run directory.
    # Everything above walks from a record to the file it names, so a file that
    # no record names was invisible: a wholly fabricated result dropped into a
    # non-active run's results/ left the validator reporting ok (D-067 round-4
    # review, defect 1). `.agent-harness/runs/` is gitignored in full, so the
    # porcelain check in main cannot see it either -- such a file produces no git
    # status entry at all, not even `??`.
    for run_name, assignment_id, result_path in result_files(harness):
        key = (run_name, assignment_id)
        if key in attributed:
            # Digest already reconciled by the ledger -> result loop above.
            continue
        try:
            result_digest = sha256_of(result_path)
        except OSError as exc:
            status[key] = UNREADABLE
            errors.append(
                f"Result artifact could not be read ({exc.__class__.__name__}): "
                f"{run_name}/{assignment_id}"
            )
            continue
        digest[key] = result_digest
        pinned = legacy_index.get(key)
        if pinned is not None:
            if pinned != result_digest:
                status[key] = LEGACY_DIGEST_MISMATCH
                errors.append(
                    "Legacy result artifact changed since the legacy manifest "
                    f"pinned it: {run_name}/{assignment_id}"
                )
            else:
                status[key] = LEGACY_PINNED
            continue
        open_receipt = open_receipts.get(key)
        if open_receipt is not None:
            # An admitted agent that has written its artifact but not stopped.
            # Reported rather than errored so a live run is not wedged
            # (D-067 round-2 review F-12) -- but only while the admission can
            # still plausibly be in flight (D-070 round-5 review F-R5-04) and
            # only while the chain that admission belongs to is itself young
            # (D-070 round-6 review F-R6-02).
            refusal = pending_carve_out_refusal(
                harness,
                run_name,
                assignment_id,
                open_receipt,
                active,
                now,
                ledger_rows_by_run.get(run_name),
            )
            if refusal is None:
                status[key] = PENDING
                pending_results.append(
                    {
                        "run_id": run_name,
                        "assignment_id": assignment_id,
                        "sha256": result_digest,
                        "receipt_created_at": str(open_receipt.get("created_at") or ""),
                    }
                )
                continue
            status[key] = CARVE_OUT_REFUSED
            errors.append(
                "Result artifact is not attributed: the open-admission carve-out "
                f"is refused for {run_name}/{assignment_id} ({result_digest}) "
                f"because {refusal}."
            )
            continue
        status[key] = UNATTRIBUTED
        errors.append(
            "Result artifact is not attributed: no consumed admission ledger row "
            "and no legacy manifest entry for "
            f"{run_name}/{assignment_id} ({result_digest})."
        )
    return AttributionReport(
        errors=errors,
        open_admissions=open_admissions,
        pending_results=pending_results,
        legacy_status=legacy_status,
        status=status,
        digest=digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-legacy-manifest",
        action="store_true",
        help="Write .agent-harness/context/LEGACY_RESULTS_MANIFEST.json from the "
        "current tree instead of validating, pinning every result artifact that "
        "lives in a run directory with no admission ledger.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --emit-legacy-manifest, overwrite an existing manifest.",
    )
    args = parser.parse_args()
    repo = root()
    if args.emit_legacy_manifest:
        emit_legacy_manifest(repo, force=args.force)
        return
    harness = repo / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    index = load_json(index_path)
    files = list(index.get("shared_files", []))
    actual, entries = hash_files(repo, files)
    errors: list[str] = []

    if actual != index.get("context_version"):
        errors.append("Context files changed after the pack was built.")
    pack_path = harness / "generated" / "CONTEXT_PACK.md"
    if not pack_path.is_file():
        errors.append("Generated CONTEXT_PACK.md is missing.")
    else:
        # Was a substring search over the pack's first six lines, which left the
        # whole body unchecked: BD623 R1 passed a pack cut to 283 of 157,567
        # characters, and one whose body was a forged frozen-decision row. The
        # pack is a verbatim concatenation of files this function has already
        # hashed, so re-rendering answers "is this what those inputs render to"
        # directly, which is the question the six-line check only appeared to ask.
        expected_pack = render_context_pack(
            repo, actual, str(index.get("built_at", "")), entries
        )
        if pack_path.read_text(encoding="utf-8") != expected_pack:
            errors.append(
                "Generated CONTEXT_PACK.md is not what its inputs render to; "
                "rebuild it with build_context_pack.py."
            )

    active = None
    try:
        active = active_run_id(repo)
    except SystemExit:
        pass
    # D-070 round-5 review, finding F-R5-08 (pre-existing, present at a1cdd8a).
    # An ACTIVE_RUN naming a directory that does not exist used to raise
    # FileNotFoundError out of load_json: fail-closed, but as a traceback on
    # stderr with an empty stdout, so every caller that parses this script's JSON
    # saw a crash rather than a verdict. Same for an unreadable or budget-less
    # RUN_PLAN.json. They are errors now, reported in the normal payload.
    if active and not (harness / "runs" / active).is_dir():
        errors.append(
            "ACTIVE_RUN names a run directory that does not exist: "
            f".agent-harness/runs/{active}. Point ACTIVE_RUN at a run created by "
            "init_run.py, or clear it."
        )
    elif active:
        run_dir = harness / "runs" / active
        plan: object = None
        try:
            plan = load_json(run_dir / "RUN_PLAN.json")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"Active run plan is missing or unreadable ({exc.__class__.__name__}): "
                f".agent-harness/runs/{active}/RUN_PLAN.json"
            )
        assignments = sorted((run_dir / "assignments").glob("*.json"))
        if plan is not None and not isinstance(plan, dict):
            errors.append(
                f"Active run plan is not a JSON object: "
                f".agent-harness/runs/{active}/RUN_PLAN.json"
            )
        elif isinstance(plan, dict):
            if plan.get("context_version") != index.get("context_version"):
                errors.append(
                    "Active run was initialized against a stale context version."
                )
            budget = plan.get("budget")
            max_total = budget.get("max_total") if isinstance(budget, dict) else None
            if not isinstance(max_total, int) or isinstance(max_total, bool):
                errors.append(
                    "Active run plan has no integer budget.max_total, so the "
                    "assignment ceiling cannot be checked."
                )
            elif len(assignments) > max_total:
                errors.append("Assignment count exceeds run max_total.")
        ids: set[str] = set()
        declared_results: dict[str, str] = {}
        for path in assignments:
            value = load_json(path)
            assignment_errors = validate_assignment_contract(
                value,
                expected_run_id=active,
                expected_context_version=str(index.get("context_version")),
                role_files=index.get("role_files", {}),
            )
            errors.extend(
                f"Assignment contract violation ({path.name}): {error}"
                for error in assignment_errors
            )
            resource_errors = validate_assignment_resource_hashes(
                repo,
                value,
                role_files=index.get("role_files", {}),
            )
            errors.extend(
                f"Assignment resource violation ({path.name}): {error}"
                for error in resource_errors
            )
            aid = value.get("assignment_id")
            if aid in ids:
                errors.append(f"Duplicate assignment_id: {aid}")
            ids.add(aid)
            if value.get("context_version") != index.get("context_version"):
                errors.append(f"Stale assignment context: {path.name}")
            if value.get("run_id") != active:
                errors.append(f"Assignment run_id mismatch: {path.name}")
            expected_result = f".agent-harness/runs/{active}/results/{aid}.json"
            declared_result = str(value.get("result_path") or "")
            if declared_result != expected_result:
                errors.append(f"Assignment result_path mismatch: {path.name}")
            if declared_result in declared_results:
                errors.append(
                    f"Duplicate assignment result_path: {path.name} and "
                    f"{declared_results[declared_result]}"
                )
            declared_results[declared_result] = path.name

        result_paths = sorted((run_dir / "results").glob("*.json"))
        for result_path in result_paths:
            assignment_path = run_dir / "assignments" / result_path.name
            if not assignment_path.is_file():
                errors.append(f"Result has no registered assignment: {result_path.name}")
                continue
            try:
                assignment = load_json(assignment_path)
                result = load_json(result_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                # Reported, not raised. An unreadable result artifact used to
                # come out of here as a traceback with an empty stdout, so every
                # caller that parses this script's JSON saw a crash rather than a
                # verdict -- the D-070 F-R5-08 class, closed there for ACTIVE_RUN
                # and left open here. The bytes are still refused: attribution
                # below reports the same file as unreadable.
                errors.append(
                    "Result or its registered assignment is unreadable or invalid "
                    f"JSON ({exc.__class__.__name__}): {result_path.name}"
                )
                continue
            result_errors = validate_result_contract(
                result,
                assignment,
                expected_run_id=active,
                expected_context_version=str(index.get("context_version")),
                expected_assignment_sha256=(
                    "sha256:" + hashlib.sha256(assignment_path.read_bytes()).hexdigest()
                ),
            )
            errors.extend(
                f"Result contract violation ({result_path.name}): {error}"
                for error in result_errors
            )

    # D-067: retained run evidence is append-only. A tracked file under runs/ that
    # differs from its committed bytes is the D-066 write-attribution incident
    # class, so surface it here rather than discovering it in a later audit.
    errors.extend(
        git_porcelain_errors(repo, ".agent-harness/runs", "Tracked run evidence")
    )
    # Attribution -- which result artifacts a durable record vouches for -- is
    # computed by one shared pass, so merge_results.py gates on exactly the rule
    # this validator enforces instead of a second copy of it that could drift
    # (D-070 round-6 review, finding 2).
    attribution = collect_result_attribution(
        repo, harness, active=active, now=datetime.now(timezone.utc)
    )
    errors.extend(attribution.errors)
    open_admissions = attribution.open_admissions
    pending_results = attribution.pending_results
    legacy_status = attribution.legacy_status

    # D-070: SSOT semantic consistency. The structural hashes above prove the
    # context files are unchanged since the pack was built; they say nothing
    # about whether those files agree with each other. F-D065-05 was exactly
    # that gap -- the gate registry recorded PASS while the prose surfaces still
    # said FAIL, and no machine check existed. Run it as a subprocess rather
    # than importing it: it is a large fail-closed checker with its own
    # regression suite, and keeping the two concerns in separate processes means
    # a change to one cannot quietly alter the other's verdict.
    # Applicability is decided by the authoritative registry, not by a flag. A
    # synthetic harness built by a fixture has no gate registry and nothing to be
    # inconsistent with; the real project always does. The one way that could
    # become a bypass -- deleting the registry to silence the check -- is closed
    # just below, because a tracked deletion is itself an error.
    registry = harness / "context" / "GATE_REGISTRY.json"
    ssot_applicable = registry.is_file()
    ssot_status = "ok" if ssot_applicable else "not applicable (no gate registry)"
    if not ssot_applicable:
        tracked, failure = git_tracked(repo, str(registry.relative_to(repo)))
        if failure:
            errors.append(
                "Could not determine whether the gate registry is tracked "
                f"({failure}); git is required for validation."
            )
        elif tracked:
            errors.append(
                "Gate registry is tracked in git but missing from the working "
                "tree; the SSOT consistency check cannot be disabled by "
                "deleting its input."
            )

    ssot = Path(__file__).resolve().parent / "check_ssot_consistency.py"
    if not ssot_applicable:
        pass
    elif not ssot.is_file():
        errors.append(f"SSOT consistency checker is missing: {ssot.name}")
    else:
        try:
            completed = subprocess.run(
                [sys.executable, str(ssot)],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            # Fail closed: an unrunnable consistency check is not a passed one.
            errors.append(
                "SSOT consistency check could not run "
                f"({exc.__class__.__name__}); it is required for validation."
            )
        else:
            if completed.returncode != 0:
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    errors.append(
                        "SSOT consistency check failed and its output was not "
                        f"readable JSON (exit {completed.returncode})."
                    )
                else:
                    reported = payload.get("errors")
                    if not isinstance(reported, list) or not reported:
                        errors.append(
                            "SSOT consistency check exited "
                            f"{completed.returncode} but reported no errors; "
                            "treating the disagreement itself as a failure."
                        )
                    else:
                        errors.extend(f"SSOT: {entry}" for entry in reported)

    # Canary freshness (D-070 Part B10, round-9 finding F-R9-004). Four canaries
    # died of an attestation that outlived the code it attested, and each time
    # the remedy was a written rule that the next canary then broke. Nothing
    # checked it: `grep -rln start_hook_sha256` returned zero .py files. It is
    # measured here so a stale canary stops the commit instead of waiting to be
    # noticed by whoever next reads the record.
    canary = Path(__file__).resolve().parent / "check_canary_freshness.py"
    canary_status = "not run"
    if not canary.is_file():
        errors.append(f"Canary freshness checker is missing: {canary.name}")
    else:
        try:
            completed = subprocess.run(
                [sys.executable, str(canary)],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            # Fail closed, exactly as the SSOT check above does.
            errors.append(
                "Canary freshness check could not run "
                f"({exc.__class__.__name__}); it is required for validation."
            )
        else:
            if completed.returncode == 0:
                canary_status = "ok"
            else:
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    errors.append(
                        "Canary freshness check failed and its output was not "
                        f"readable JSON (exit {completed.returncode})."
                    )
                else:
                    reported = payload.get("errors")
                    if not isinstance(reported, list) or not reported:
                        errors.append(
                            "Canary freshness check exited "
                            f"{completed.returncode} but reported no errors; "
                            "treating the disagreement itself as a failure."
                        )
                    else:
                        errors.extend(f"CANARY: {entry}" for entry in reported)

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": True,
                "context_version": actual,
                "active_run": active,
                "open_admissions": open_admissions,
                # Reported with their digests and counted, because a pending
                # result is tolerated, not verified (D-070 round-5 F-R5-04).
                "pending_result_count": len(pending_results),
                "pending_results": pending_results,
                "legacy_results_manifest": legacy_status,
                "ssot_consistency": ssot_status,
                "canary_freshness": canary_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
