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

from _harness import root

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


def find_attestations(repo: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every retained canary attestation, as (relative path, document).

    Searched RECURSIVELY. The previous single-level ``*/artifacts/*.json`` glob
    made an attestation one directory deeper or shallower completely invisible --
    no error, not even a count -- which is a fail-open hiding place, not a
    filter (round 10).
    """
    found: list[tuple[str, dict[str, Any]]] = []
    runs = repo / RUNS_ROOT
    if not runs.is_dir():
        return found
    for path in sorted(runs.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # Not every artifact is JSON or ours; only a readable object that
            # declares itself a canary attestation is in scope.
            continue
        if isinstance(document, dict) and document.get("artifact_class") == ARTIFACT_CLASS:
            found.append((str(path.relative_to(repo)), document))
    return found


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
            row = json.loads(line)
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


def superseded_ids(
    attestations: list[tuple[str, dict[str, Any]]], authorised: set[str]
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
    for mine, names in sorted(declares.items()):
        for other in sorted(names):
            if mine in declares.get(other, set()):
                errors.append(
                    f"canaries {mine} and {other} retire each other. A cycle leaves both "
                    "unreachable by the freshness check while both still read as "
                    "authoritative; supersession must run one way, newest last."
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
        if mine is None or mine not in authorised:
            errors.append(
                f"{rel}: canary {mine or '(no id)'} tries to retire "
                f"{', '.join(sorted(names))} but is not itself backed by a consumed "
                "admission receipt. Only a canary that really ran may retire another."
            )
            continue
        retired.update(names)
    return retired, errors


POLICY_FILE = ".agent-harness/context/CANARY_POLICY.json"


def required_live(repo: Path, errors: list[str]) -> int:
    """How many live canaries this tree must retain, from its declaration.

    Absent declaration means none required, which is what lets a synthetic
    fixture repository hold no canary evidence without inventing some. Deleting
    the declaration to silence the floor is refused: if git tracks the file and
    the working tree has lost it, that is an error, not a permission.
    """
    path = repo / POLICY_FILE
    if not path.is_file():
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", POLICY_FILE],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            ).returncode == 0
        except OSError as exc:
            errors.append(
                f"{POLICY_FILE} is absent and git could not be consulted "
                f"({exc.__class__.__name__}); the canary policy cannot be established."
            )
            return 0
        if tracked:
            errors.append(
                f"{POLICY_FILE} is tracked in git but missing from the working tree; the "
                "canary floor cannot be lowered by deleting its declaration."
            )
        return 0
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{POLICY_FILE} is unreadable or invalid JSON ({exc.__class__.__name__}).")
        return 0
    value = document.get("min_live") if isinstance(document, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{POLICY_FILE} has no non-negative integer 'min_live'.")
        return 0
    return value


def check(repo: Path) -> list[str]:
    errors: list[str] = []
    attestations = find_attestations(repo)

    # Receipt backing is decided FIRST, because it is what gives a canary the
    # authority to retire another. A file nobody dispatched retires nothing.
    authorised: set[str] = set()
    backing_errors: dict[str, list[str]] = {}
    for rel, document in attestations:
        name = canary_id(rel, document)
        if name is None:
            continue
        problems = receipt_backing(repo, rel, name, document)
        backing_errors[rel] = problems
        if not problems:
            authorised.add(name)

    retired, supersession_errors = superseded_ids(attestations, authorised)
    errors.extend(supersession_errors)

    live_count = 0
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
        live_count += 1
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
    minimum = required_live(repo, errors)
    if live_count < minimum:
        errors.append(
            f"{RUNS_ROOT}: {live_count} live canary attestation(s) found, at least "
            f"{minimum} required. Zero is not 'clean' -- run directories are "
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
    attestations = find_attestations(repo)
    backed = {
        name
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
