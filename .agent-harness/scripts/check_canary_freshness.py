#!/usr/bin/env python3
"""Fail when a live canary attestation no longer describes the code it attested.

WHY THIS EXISTS (D-070 Part B10, closing round-9 finding F-R9-004).

A canary is a live end-to-end exercise of the admission machinery: it dispatches
a real Stop against real hooks and records what happened. Its whole evidential
value is that it ran against *these* bytes. The moment the hook it exercised is
edited, the attestation stops being evidence and becomes a claim about a file
that no longer exists in that form -- while still reading, to anyone scanning the
record, exactly as it did when it was true.

Four canaries have died of precisely that:

* C1 -- rejected at round 4, attestation taken before the sealing commit;
* C4 -- rejected for the same reason, 286 seconds early;
* C5 -- found stale at round 7: it pinned ``stop_hook_sha256 dc19da4d`` while
  HEAD had moved to ``0533da88``;
* C6 -- found stale at round 9: it pinned ``start_hook_sha256 af8f7016`` while
  HEAD held ``8b283512``, and the single intervening commit was the one that
  rewrote ``describe_admission_receipt`` -- the very function that produced
  C6's own attested log line.

Each time the chosen remedy was a *rule*: "the canary is the last act before the
sealing commit." C6's own artifact writes that rule down, and C6 then broke it.
A rule that has failed four times is not a control. Round 9 measured why:
``grep -rln start_hook_sha256`` returned zero ``.py`` files, so nothing checked
it. The staleness was always discovered by a human reading the record, which is
the one mechanism this project has repeatedly proved unreliable.

WHAT THIS COSTS, STATED PLAINLY. Editing an attested hook now fails the harness
until either a fresh canary is run or the stale one is explicitly superseded.
That is deliberate and it is the entire point: a canary that was not re-run after
the code changed is not evidence, and the cheapest way to keep the record honest
is to make the record's staleness stop the commit rather than wait to be noticed.

WHAT COUNTS AS LIVE. Every canary declares which canaries it replaces. A canary
that some other canary supersedes is history and is never measured against
current bytes -- the same discipline the frozen/historical fact roles use. Only
canaries that nothing supersedes must still describe the working tree.

FAIL CLOSED. An attestation whose ``canary`` id is missing, whose attested file
does not exist, or whose digest field names a file this script cannot map, is an
error rather than a skip. An unrunnable freshness check is not a passed one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from _harness import loads_strict, root

ARTIFACT_CLASS = "CANARY_ATTESTATION"
RUNS_ROOT = ".agent-harness/runs"

# Digest field -> the file it attests. Adding a hook to a canary means adding it
# here, which is a reviewed change to this file rather than a free-text key in a
# data file that nothing resolves.
ATTESTED_FILES = {
    "start_hook_sha256": ".codex/hooks/subagent_start_context.py",
    "stop_hook_sha256": ".codex/hooks/subagent_stop_validate.py",
}
DIGEST_SUFFIX = "_sha256"
# Fields that end in _sha256 but do not attest a tracked source file.
NON_FILE_DIGESTS = {"result_sha256", "assignment_sha256", "token_digest"}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_attestations(
    repo: Path,
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Every retained canary attestation as (relative path, document), plus errors.

    Searched RECURSIVELY. The previous single-level ``*/artifacts/*.json`` glob
    made an attestation one directory deeper or shallower completely invisible --
    no error, not even a count -- which is a fail-open hiding place, not a
    filter (round 10).
    """
    found: list[tuple[str, dict[str, Any]]] = []
    errors: list[str] = []
    runs = repo / RUNS_ROOT
    if not runs.is_dir():
        return found, errors
    for path in sorted(runs.rglob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            document = loads_strict(raw)
        except json.JSONDecodeError:
            # Strict parsing rejected it. Skipping would be FAIL-OPEN: an
            # attestation with a duplicate key would become invisible, and an
            # invisible attestation cannot be reported stale, so ambiguous bytes
            # would be a way to retire a canary from its own freshness check.
            #
            # ROUND 13 (registered reviewer, CRITICAL). The first version of this
            # guard re-read the file permissively and reported it only when the
            # LOOSE parse still said ``artifact_class == CANARY_ATTESTATION``.
            # That check was worthless, because a loose parse is last-value-wins
            # and ``artifact_class`` is a key like any other: duplicating it with
            # a decoy as the final value made the document classify as "not ours"
            # and vanish with no trace at all -- not counted, not live, not
            # superseded, no error -- while carrying a hostile hook digest. C9 was
            # made to disappear that way in a controlled reproduction, exit 0.
            #
            # The lesson is exact: the CLASSIFICATION is part of what is ambiguous,
            # so it cannot be the thing that decides whether to report. Any file
            # here that a permissive parser accepts and a strict one refuses is an
            # error, full stop, whatever it claims to be. Measured cost: 0 of 712
            # JSON files under this root.
            try:
                json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue  # genuinely not parseable JSON; not ours, never was
            errors.append(
                f"{path.relative_to(repo)}: this file has a repeated object key, so what "
                "it says depends on which parser reads it -- including what it says its "
                "own artifact_class is. Ambiguous bytes under the runs root are refused "
                "rather than classified, because a document that silently failed to parse "
                "would be a canary that can never be reported stale."
            )
            continue
        if isinstance(document, dict) and document.get("artifact_class") == ARTIFACT_CLASS:
            found.append((str(path.relative_to(repo)), document))
    return found, errors


def canary_id(rel: str, document: dict[str, Any]) -> str | None:
    """The canary's id: its own field, else inferred from the run directory name.

    C5 predates the ``canary`` field and carries no id of its own. Its bytes are
    retained evidence and are NOT edited to add one -- rewriting a retained
    attestation to satisfy a checker is the exact move this harness exists to
    make impossible. The id is instead read from the path, which is part of the
    same retained record: ``.../runs/run-...-canary-c5/artifacts/...`` -> ``C5``.
    """
    declared = document.get("canary")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    for part in PurePosixPath(rel).parts:
        if part.startswith("run-") and "-canary-" in part:
            tail = part.rsplit("-canary-", 1)[1]
            if tail and tail[0].lower() == "c" and tail[1:].isdigit():
                return tail.upper()
    return None


RECEIPT_FIELDS = (
    "canary_lease_run_id",
    "consumed_assignment_id",
    "consumed_by_agent_id",
    "result_sha256",
)


def receipt_backing(repo: Path, rel: str, canary: str, document: dict[str, Any]) -> list[str]:
    """Errors unless this attestation is backed by a real consumed receipt.

    ROUND 10's DECISIVE FINDING. A canary attestation was just a file, and
    ``supersedes_canary`` was self-asserted by the very artifact whose freshness
    was in question. One typed JSON file -- no dispatch, no lease, no receipt --
    with its own digests set to the current bytes retired the real C7 and
    returned ``ok: true`` while the attested hook carried a hostile edit. The
    detector written to stop four canaries dying silently was itself defeated by
    a text editor.

    A canary now has to point at the admission ledger row its dispatch actually
    produced: the run, the assignment, the agent id, and the SHA-256 of the
    result the Stop hook pinned. All four are checked, and the result file is
    re-hashed rather than trusted. Forgery is not made impossible -- harness and
    subagents share one OS user, which is the stated residual and is not closed
    here -- but the cost moves from writing one file to forging an append-only
    ledger row AND a result whose digest matches it AND keeping both consistent
    with every other reader of that ledger.
    """
    errors: list[str] = []
    missing = [field for field in RECEIPT_FIELDS if not str(document.get(field) or "").strip()]
    if missing:
        return [
            f"{rel}: canary {canary} declares no {', '.join(missing)}, so nothing ties it "
            "to a dispatch that actually happened. An attestation that names no receipt "
            "is a file somebody wrote, not evidence that anything ran."
        ]
    run_id = str(document["canary_lease_run_id"])
    ledger = repo / RUNS_ROOT / run_id / "ADMISSIONS.jsonl"
    if not ledger.is_file():
        return [
            f"{rel}: canary {canary} names run {run_id!r}, which has no ADMISSIONS.jsonl."
        ]
    rows: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = loads_strict(line)
        except json.JSONDecodeError:
            return [
                f"{rel}: canary {canary} names run {run_id!r} whose ADMISSIONS.jsonl has an "
                "unreadable line. A ledger that cannot be parsed cannot back anything."
            ]
        if isinstance(row, dict):
            rows.append(row)
    match = next(
        (
            row
            for row in rows
            if row.get("event") == "consumed"
            and row.get("assignment_id") == document["consumed_assignment_id"]
            and row.get("agent_id") == document["consumed_by_agent_id"]
            and row.get("result_sha256") == document["result_sha256"]
        ),
        None,
    )
    if match is None:
        return [
            f"{rel}: canary {canary} claims assignment "
            f"{document['consumed_assignment_id']!r} was consumed by agent "
            f"{document['consumed_by_agent_id']!r} with result "
            f"{str(document['result_sha256'])[:15]}..., but no such consumed row exists in "
            f"{run_id}/ADMISSIONS.jsonl."
        ]
    result_path = repo / RUNS_ROOT / run_id / "results" / f"{document['consumed_assignment_id']}.json"
    if not result_path.is_file():
        errors.append(
            f"{rel}: canary {canary} is backed by a ledger row whose result "
            f"{result_path.relative_to(repo)} is missing, so the digest it pins cannot be "
            "re-derived."
        )
    else:
        actual = "sha256:" + sha256_of(result_path)
        if actual != document["result_sha256"]:
            errors.append(
                f"{rel}: canary {canary} pins result {str(document['result_sha256'])[:15]}... "
                f"but {result_path.relative_to(repo)} hashes to {actual[:15]}.... The ledger "
                "row and the artifact disagree."
            )
    return errors


PENDING_COMMIT = "PENDING_COMMIT"


def declared_provenance(
    repo: Path, rel: str, canary: str, document: dict[str, Any]
) -> list[str]:
    """Errors unless the commit a canary NAMES actually contains the bytes it attests.

    ROUND 13 (registered adjudicator, F-R13-05). C8 declared
    ``head_commit 4022385``, where the stop hook hashes to ``0533da88`` -- the
    very value C8 said had gone stale. The bytes it attested first exist at
    ``9df6cd4``, C8's own sealing commit. Nothing caught it, because this
    checker only ever compared attested digests against the WORKING TREE. A
    provenance field that nothing verifies is decoration, and decoration in an
    evidence record is worse than an absent field: it reads as a citation.

    The underlying problem is structural and worth stating, because it makes the
    naive fix impossible. A canary is dispatched as the last act BEFORE the
    sealing commit, so at the moment it writes its attestation the commit that
    will contain its bytes does not exist yet. ``git rev-parse HEAD`` therefore
    returns the PREVIOUS commit -- which is exactly the wrong answer, and is
    exactly what C8 recorded.

    So the field follows the convention this repository already uses for a
    declared fact whose commit is not yet known: write ``PENDING_COMMIT`` and
    pin it in a follow-up. That is legal only while the attestation itself is
    uncommitted; once it is tracked and clean, a real hash is required and the
    bytes at that hash must match. An attestation cannot sit in the record
    permanently claiming its provenance is still pending.
    """
    errors: list[str] = []
    declared = str(document.get("head_commit") or "").strip()
    if not declared:
        return errors
    attested = {
        field: str(document.get(field) or "")
        for field in document
        if field.endswith(DIGEST_SUFFIX) and field not in NON_FILE_DIGESTS
        and ATTESTED_FILES.get(field)
    }
    if not attested:
        return errors

    if declared == PENDING_COMMIT:
        # Legal in the commit that INTRODUCES the attestation, and only there.
        # A stricter "must be pinned once committed" is unsatisfiable: the pin
        # names the sealing commit, which does not exist until the seal happens,
        # so the very commit that seals the canary would fail its own check. The
        # obligation lands on the NEXT commit, which is the same shape the facts
        # file uses -- write PENDING_COMMIT, pin it in a follow-up.
        if not is_tracked(repo, rel) or _git_dirty(repo, rel):
            return errors
        sealed_at = _git_last_commit(repo, rel)
        head = _git_head(repo)
        if sealed_at and head and sealed_at != head:
            errors.append(
                f"{rel}: canary {canary} still declares head_commit {PENDING_COMMIT!r}, but it was "
                f"sealed at {sealed_at[:12]} and HEAD has since moved to {head[:12]}. Pin it to the "
                "commit that contains these bytes, the same way a declared fact's 'as_of_commit' is "
                "pinned by a follow-up. A permanently pending provenance claim is an unverifiable one."
            )
        return errors

    for field, digest in sorted(attested.items()):
        target = ATTESTED_FILES[field]
        blob = _git_show(repo, declared, target)
        if blob is None:
            errors.append(
                f"{rel}: canary {canary} declares head_commit {declared[:12]}, where {target} "
                "cannot be read. A commit that does not contain the attested file cannot be "
                "its provenance."
            )
            continue
        actual = hashlib.sha256(blob).hexdigest()
        if actual != digest:
            errors.append(
                f"{rel}: canary {canary} declares head_commit {declared[:12]} and attests "
                f"{target} at {digest[:8]}, but at that commit the file hashes to {actual[:8]}. "
                "The canary did not run against the bytes of the commit it names. Because a "
                "canary runs BEFORE its own sealing commit, `git rev-parse HEAD` at write time "
                f"is the PREVIOUS commit -- declare {PENDING_COMMIT!r} and pin it afterwards."
            )
    return errors


def _git_last_commit(repo: Path, rel: str) -> str | None:
    done = _git(repo, "log", "-1", "--format=%H", "--", rel)
    if done is None or done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _git_head(repo: Path) -> str | None:
    done = _git(repo, "rev-parse", "HEAD")
    if done is None or done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _git_dirty(repo: Path, rel: str) -> bool:
    done = _git(repo, "status", "--porcelain", "--", rel)
    return done is None or bool(done.stdout.strip())


def _git_show(repo: Path, commit: str, rel: str) -> bytes | None:
    try:
        done = subprocess.run(
            ["git", "show", f"{commit}:{rel}"], cwd=repo, capture_output=True, check=False
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def _supersession_cycles(declares: dict[str, set[str]]) -> list[list[str]]:
    """Every directed cycle in the supersession graph, shortest first.

    Round 11: the previous rule compared each pair and caught `A <-> B` only.
    `C1 -> C2 -> C3 -> C1` passed with `ok: true` and reported ALL THREE as
    superseded, so three canaries retired each other into invisibility and the
    checker announced the survivors as if nothing were missing. Pairwise
    comparison cannot see that; a graph walk sees it at every length, and there
    is no length at which a cycle is legitimate.
    """
    colour: dict[str, int] = {}
    cycles: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
        colour[node] = 1  # grey: on the current path
        for nxt in sorted(declares.get(node, ())):
            if colour.get(nxt) == 1:
                bare = path[path.index(nxt):]
                # Rotate to start at the smallest id so the MESSAGE is stable
                # whichever node the walk happened to enter from. There is no
                # de-duplication here and none is needed: a node is visited at
                # most once, so each back edge is examined at most once, so one
                # cycle cannot be reported twice. A `seen` set was written here
                # first and the mutation battery proved it dead -- it survived
                # its own removal, which is this project's definition of a guard
                # that is not a guard.
                spin = bare.index(min(bare))
                cycles.append(bare[spin:] + bare[:spin] + [min(bare)])
            elif colour.get(nxt) is None:
                visit(nxt, path + [nxt])
        colour[node] = 2  # black: finished

    for start in sorted(declares):
        if colour.get(start) is None:
            visit(start, [start])
    return sorted(cycles, key=lambda c: (len(c), c))


def superseded_ids(
    attestations: list[tuple[str, dict[str, Any]]], authorised_paths: set[str]
) -> tuple[set[str], list[str]]:
    """Canary ids retired by another canary, plus errors for illegal retirements.

    Only a canary that is itself receipt-backed may retire anything -- otherwise
    a forged file retires the real one, which is exactly what round 10 did. A
    canary may not retire ITSELF, and two canaries may not retire each other:
    both are ways to make a stale canary unreachable by the freshness check
    while leaving it in the record looking authoritative.
    """
    retired: set[str] = set()
    errors: list[str] = []
    declares: dict[str, set[str]] = {}
    for rel, document in attestations:
        mine = canary_id(rel, document)
        raw = document.get("supersedes_canary")
        if isinstance(raw, str):
            raw = [raw]
        if mine is not None and isinstance(raw, list):
            declares.setdefault(mine, set()).update(str(item) for item in raw)
    for cycle in _supersession_cycles(declares):
        errors.append(
            f"canaries {' -> '.join(cycle)} form a supersession cycle. A cycle leaves "
            "every canary in it unreachable by the freshness check while all of them "
            "still read as authoritative; supersession must run one way, newest last."
        )
    for rel, document in attestations:
        mine = canary_id(rel, document)
        declared = document.get("supersedes_canary")
        if isinstance(declared, str):
            declared = [declared]
        if not isinstance(declared, list) or not declared:
            continue
        names = {str(item) for item in declared}
        if mine is not None and mine in names:
            errors.append(
                f"{rel}: canary {mine} names ITSELF in 'supersedes_canary', which would "
                "retire it from its own freshness check. A canary cannot supersede itself."
            )
            names.discard(mine)
        # Authority is held by this DOCUMENT, not by its id. Round 13 (registered
        # reviewer, F-R13-003): `authorised` used to be a set of ids, so any file
        # that merely REUSED an already-backed id inherited the right to retire
        # others -- a sacrificial document borrowing C8's id retired C9 without
        # carrying a single receipt field of its own. It was caught, but only
        # because it also collided with the duplicate-id check, which is a
        # different rule that happens to overlap. Coupling a security property to
        # an unrelated rule's coverage is how a guard silently stops guarding when
        # the other rule moves, so the authority is now per (id, attesting file).
        if mine is None or rel not in authorised_paths:
            errors.append(
                f"{rel}: canary {mine or '(no id)'} tries to retire "
                f"{', '.join(sorted(names))} but THIS attestation is not itself backed by a "
                "consumed admission receipt. Authority belongs to the document that really "
                "ran, not to any file that reuses its id."
            )
            continue
        retired.update(names)
    return retired, errors


POLICY_FILE = ".agent-harness/context/CANARY_POLICY.json"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )
    except OSError:
        return None


def is_git_repo(repo: Path) -> bool:
    done = _git(repo, "rev-parse", "--git-dir")
    return done is not None and done.returncode == 0


def is_tracked(repo: Path, rel: str) -> bool:
    done = _git(repo, "ls-files", "--error-unmatch", "--", rel)
    return done is not None and done.returncode == 0


def retained_evidence(repo: Path, rel: str, document: dict[str, Any]) -> list[str]:
    """The files that must survive a fresh clone for this canary to be evidence.

    ``.gitignore`` excludes ``/.agent-harness/runs/`` wholesale and canary
    directories are force-added one at a time. Round 11 measured the
    consequence: a canary written but never ``git add -f``-ed satisfied a
    declared floor of 1, so a working tree reported ok while every other
    checkout of the same commit held no canary evidence at all. The floor was
    counting files on one disk, not evidence in the repository.

    The attestation alone is not enough. Its receipt row and the result that row
    pins are what make it checkable, so all three must be retained or the next
    checkout inherits an attestation it cannot verify.
    """
    files = [rel]
    run_id = str(document.get("canary_lease_run_id") or "").strip()
    assignment = str(document.get("consumed_assignment_id") or "").strip()
    if run_id:
        files.append(f"{RUNS_ROOT}/{run_id}/ADMISSIONS.jsonl")
        if assignment:
            files.append(f"{RUNS_ROOT}/{run_id}/results/{assignment}.json")
    return files


def required_live(repo: Path, errors: list[str]) -> int:
    """How many live canaries this tree must retain, from its declaration.

    Absent declaration means none required, which is what lets a synthetic
    fixture repository hold no canary evidence without inventing some. Deleting
    the declaration to silence the floor is refused: if git tracks the file and
    the working tree has lost it, that is an error, not a permission.
    """
    path = repo / POLICY_FILE
    if not path.is_file():
        if _git(repo, "rev-parse", "--git-dir") is None:
            errors.append(
                f"{POLICY_FILE} is absent and git could not be consulted; the canary "
                "policy cannot be established."
            )
            return 0
        if is_tracked(repo, POLICY_FILE):
            errors.append(
                f"{POLICY_FILE} is tracked in git but missing from the working tree; the "
                "canary floor cannot be lowered by deleting its declaration."
            )
        return 0
    # The declaration EXISTS. Round 11: the tracked-deletion guard above ran only
    # on the absent branch, so removing the tracked declaration from the index
    # and leaving an untracked `min_live: 0` in its place lowered the floor to
    # zero with no error at all -- the file was present, so nothing asked where
    # it came from. A declaration that a fresh clone will not receive is not a
    # declaration; it is a local opinion.
    if not is_git_repo(repo):
        errors.append(
            f"{POLICY_FILE} declares retained canary evidence, but this tree is not a git "
            "repository, so no retention claim about it can be checked. The declaration "
            "belongs in a tracked file or not at all."
        )
    elif not is_tracked(repo, POLICY_FILE):
        errors.append(
            f"{POLICY_FILE} exists but git does not track it, so the floor it declares "
            "would vanish on a fresh checkout. An untracked declaration cannot lower or "
            "raise the floor for anyone but this working tree."
        )
    try:
        document = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{POLICY_FILE} is unreadable or invalid JSON ({exc.__class__.__name__}).")
        return 0
    value = document.get("min_live") if isinstance(document, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{POLICY_FILE} has no non-negative integer 'min_live'.")
        return 0
    return value


def check(repo: Path) -> list[str]:
    attestations, errors = find_attestations(repo)

    # Receipt backing is decided FIRST, because it is what gives a canary the
    # authority to retire another. A file nobody dispatched retires nothing.
    authorised_paths: set[str] = set()
    backing_errors: dict[str, list[str]] = {}
    for rel, document in attestations:
        name = canary_id(rel, document)
        if name is None:
            continue
        problems = receipt_backing(repo, rel, name, document)
        backing_errors[rel] = problems
        if not problems:
            authorised_paths.add(rel)

    retired, supersession_errors = superseded_ids(attestations, authorised_paths)
    errors.extend(supersession_errors)

    # Ids that count toward the floor, and the file each was counted from. A SET,
    # because round 11 measured two attestations declaring the same canary id
    # satisfying a declared floor of two: the checker reported `live: [C1, C1]`
    # and called it clean. One canary duplicated is one canary.
    countable: dict[str, str] = {}
    duplicated: dict[str, list[str]] = {}
    for rel, document in attestations:
        canary = canary_id(rel, document)
        if canary is None:
            errors.append(
                f"{rel}: a CANARY_ATTESTATION with no 'canary' id, in a run directory "
                "whose name does not end in a canary id either, cannot be superseded "
                "by anything, so it can never be retired and can never be checked."
            )
            continue
        if canary in retired:
            continue
        if canary in countable:
            duplicated.setdefault(canary, [countable[canary]]).append(rel)
        else:
            countable[canary] = rel
        # A live canary must be backed. A retired one need not be re-checked --
        # it is history, and its replacement carries the obligation.
        errors.extend(backing_errors.get(rel, []))

        digest_fields = [
            key
            for key in document
            if key.endswith(DIGEST_SUFFIX) and key not in NON_FILE_DIGESTS
        ]
        if not digest_fields:
            errors.append(
                f"{rel}: canary {canary} attests no hook digest, so nothing ties it to "
                "the code it exercised. A canary that cannot go stale cannot be "
                "evidence that anything still holds."
            )
            continue

        errors.extend(declared_provenance(repo, rel, canary, document))

        for field in sorted(digest_fields):
            target = ATTESTED_FILES.get(field)
            if target is None:
                errors.append(
                    f"{rel}: canary {canary} attests {field!r}, which this checker "
                    "cannot map to a file. Add it to ATTESTED_FILES; an unmappable "
                    "digest is an error, never a skip."
                )
                continue
            declared = document.get(field)
            if not isinstance(declared, str) or not declared.strip():
                errors.append(f"{rel}: canary {canary} has an empty {field!r}.")
                continue
            path = repo / target
            if not path.is_file():
                errors.append(
                    f"{rel}: canary {canary} attests {target}, which does not exist."
                )
                continue
            actual = sha256_of(path)
            if actual != declared:
                errors.append(
                    f"{rel}: canary {canary} is STALE. It attests {target} at "
                    f"{declared[:8]} but the working tree holds {actual[:8]}. The canary "
                    "did not run against these bytes, so it is not evidence about them. "
                    "Re-run it as the last act before the sealing commit, or record a "
                    "replacement canary whose 'supersedes_canary' names "
                    f"{canary!r}."
                )

    # Zero live canaries must not read as "all clean". `.gitignore` excludes
    # `/.agent-harness/runs/` wholesale and only specific canary directories are
    # force-added, so a checkout that misses them found NO attestations and
    # reported ok -- the detector's own default was fail-open (round 10).
    for canary, wheres in sorted(duplicated.items()):
        errors.append(
            f"canary {canary} is attested by {len(wheres)} live files "
            f"({', '.join(sorted(wheres))}). Two files claiming one canary id make it "
            "ambiguous which bytes are the evidence, and they counted twice toward the "
            "live floor. Give each canary its own id, or retire one by supersession."
        )

    minimum = required_live(repo, errors)
    # Only RETAINED evidence counts. A canary that a fresh clone will not receive
    # cannot discharge a floor that exists precisely because run directories are
    # gitignored -- counting it reproduces the fail-open default the floor was
    # written to close, one level up.
    if minimum and is_git_repo(repo):
        for canary in sorted(countable):
            rel = countable[canary]
            document = next(doc for where, doc in attestations if where == rel)
            missing = [
                path for path in retained_evidence(repo, rel, document)
                if not is_tracked(repo, path)
            ]
            if missing:
                del countable[canary]
                errors.append(
                    f"{rel}: canary {canary} does not count toward the live floor "
                    f"because git does not track {', '.join(missing)}. "
                    f"{RUNS_ROOT} is gitignored wholesale, so this evidence exists on "
                    "this disk and nowhere else. Force-add it, or it is not retained."
                )

    if len(countable) < minimum:
        errors.append(
            f"{RUNS_ROOT}: {len(countable)} live retained canary attestation(s) found, at "
            f"least {minimum} required. Zero is not 'clean' -- run directories are "
            "gitignored and force-added individually, so a missing attestation looks "
            "exactly like a passing check. Retain one, or dispatch one."
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args()
    repo = root()
    errors = check(repo)
    attestations, _discovery_errors = find_attestations(repo)
    backed = {
        rel
        for rel, document in attestations
        for name in [canary_id(rel, document)]
        if name is not None and not receipt_backing(repo, rel, name, document)
    }
    retired, _ = superseded_ids(attestations, backed)
    live = [
        canary_id(rel, document)
        for rel, document in attestations
        if canary_id(rel, document) not in retired
    ]
    payload = {
        "ok": not errors,
        "attestations": len(attestations),
        "live": sorted(str(item) for item in live if item),
        "superseded": sorted(retired),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
