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
    """Every retained canary attestation, as (relative path, document)."""
    found: list[tuple[str, dict[str, Any]]] = []
    runs = repo / RUNS_ROOT
    if not runs.is_dir():
        return found
    for path in sorted(runs.glob("*/artifacts/*.json")):
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


def superseded_ids(attestations: list[tuple[str, dict[str, Any]]]) -> set[str]:
    """Canary ids that some other canary declares it replaces."""
    retired: set[str] = set()
    for _rel, document in attestations:
        declared = document.get("supersedes_canary")
        if isinstance(declared, str):
            declared = [declared]
        if isinstance(declared, list):
            retired.update(str(item) for item in declared)
    return retired


def check(repo: Path) -> list[str]:
    errors: list[str] = []
    attestations = find_attestations(repo)
    retired = superseded_ids(attestations)

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
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args()
    repo = root()
    errors = check(repo)
    attestations = find_attestations(repo)
    retired = superseded_ids(attestations)
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
