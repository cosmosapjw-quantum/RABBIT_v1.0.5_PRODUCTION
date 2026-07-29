#!/usr/bin/env python3
"""Reduce the active run's ADMITTED result envelopes into MERGED_RESULTS.json.

D-070 round-6 review, finding 2. This script used to glob ``results/*.json`` and
merge whatever it found, with no reference to admissions, receipts, the ledger or
the legacy manifest -- so unattributed or still-pending bytes were folded into a
run's merged output while validate_harness.py reported green. The merge was a
second, unguarded path to treating a result as real, and MERGED_RESULTS.json is
what the adjudicator reads, so laundering happened there rather than in the
validator.

The rule is now the validator's rule, not a copy of it: attribution is computed
by ``validate_harness.collect_result_attribution`` and a result is merged only
when its verdict is in ``validate_harness.MERGEABLE_ATTRIBUTION`` -- a consumed
ADMISSIONS.jsonl row whose digest matches the bytes on disk, or a matching
LEGACY_RESULTS_MANIFEST.json pin. A second implementation of the rule would
drift; importing the first cannot.

WHY A STILL-PENDING RESULT IS REFUSED OUTRIGHT rather than merged behind a flag.
The open-admission carve-out exists for exactly one reason: a live run must not
be wedged between the moment an agent writes its artifact and the moment
SubagentStop consumes its receipt. Validation runs continuously during a run, so
it has to tolerate that window; merging does not. It is an explicit operator step
taken once the required results are in, its output is a durable artifact that
feeds adjudication, and the wait is the length of one Stop dispatch. So nothing
is wedged by refusing, while a `--include-pending` flag would re-open the hole
with a gesture cheaper than the reopen it is supposed to require. Wait for the
stop, or drop the artifact.

Fail-closed throughout: an unreadable result, an unreadable ledger or manifest,
or any attribution error refuses the whole merge. A partial merge is not written,
and an existing MERGED_RESULTS.json is left untouched rather than replaced by a
weaker one.

Usage:
    python3 .agent-harness/scripts/merge_results.py
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import NoReturn

import validate_harness
from _harness import active_run_id, dump_json, load_json, root, utc_now


def canonical_key(finding: dict) -> tuple:
    refs = tuple(sorted(str(x) for x in finding.get("evidence_refs", [])))
    return (
        str(finding.get("claim_id", "")),
        str(finding.get("evidence_fingerprint", "")),
        str(finding.get("verdict", "")),
        refs,
    )


def refuse(run_id: str, reasons: list[str]) -> NoReturn:
    """Print the refusal as JSON on stdout and exit non-zero, writing nothing."""
    print(
        json.dumps(
            {
                "ok": False,
                "run_id": run_id,
                "wrote": None,
                "errors": reasons,
                "next": (
                    "Every merged result must be attributed by a consumed "
                    "ADMISSIONS.jsonl row whose digest matches the bytes on disk, "
                    "or pinned in LEGACY_RESULTS_MANIFEST.json. Let the agent stop "
                    "so its admission is consumed, delete the artifact, or run "
                    "validate_harness.py for the full attribution verdict."
                ),
            },
            indent=2,
        )
    )
    raise SystemExit(1)


def main() -> None:
    repo = root()
    run_id = active_run_id(repo)
    harness = repo / ".agent-harness"
    run_dir = harness / "runs" / run_id

    # One shared pass, so the merge gate and the validator cannot disagree about
    # what "attributed" means.
    attribution = validate_harness.collect_result_attribution(
        repo, harness, active=run_id, now=datetime.now(timezone.utc)
    )

    paths = sorted((run_dir / "results").glob("*.json"))
    reasons: list[str] = []
    # Any attribution error at all means the record that would vouch for these
    # bytes is itself untrustworthy -- an unreadable receipt, a garbage ledger
    # line, a broken or unpinned manifest, a digest that no longer matches. None
    # of those can be reasoned around here, so none of them are.
    reasons.extend(f"attribution: {error}" for error in attribution.errors)
    for path in paths:
        key = (run_id, path.stem)
        verdict = attribution.status.get(key, validate_harness.UNATTRIBUTED)
        if verdict not in validate_harness.MERGEABLE_ATTRIBUTION:
            reasons.append(
                f"refusing to merge {path.relative_to(repo)}: attribution verdict "
                f"is {verdict!r}, not one of "
                f"{sorted(validate_harness.MERGEABLE_ATTRIBUTION)}"
            )
    if reasons:
        refuse(run_id, reasons)

    results: list[dict] = []
    provenance: list[dict[str, str]] = []
    for path in paths:
        key = (run_id, path.stem)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            reasons.append(
                f"result artifact could not be read ({exc.__class__.__name__}): "
                f"{path.relative_to(repo)}"
            )
            continue
        # The attribution pass hashed these bytes a moment ago. Re-checking the
        # digest closes the window between that judgement and this read, so the
        # merge can never fold in bytes that were swapped in after they were
        # vouched for.
        current = "sha256:" + hashlib.sha256(raw).hexdigest()
        expected = attribution.digest.get(key, "")
        if current != expected:
            reasons.append(
                f"result artifact changed while the merge was running: "
                f"{path.relative_to(repo)} was attributed as {expected} and now "
                f"hashes to {current}"
            )
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Previously collected into the merged output's `errors` list and
            # skipped; an unreadable admitted artifact is a failure, not a note.
            reasons.append(
                f"result artifact is unreadable or not valid JSON "
                f"({exc.__class__.__name__}): {path.relative_to(repo)}"
            )
            continue
        if not isinstance(value, dict):
            reasons.append(
                f"result artifact is not a JSON object: {path.relative_to(repo)}"
            )
            continue
        results.append(value)
        provenance.append(
            {
                "assignment_id": path.stem,
                "sha256": current,
                "attribution": attribution.status[key],
            }
        )
    if reasons:
        refuse(run_id, reasons)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for result in results:
        for finding in result.get("findings", []):
            groups[canonical_key(finding)].append(
                {
                    "assignment_id": result.get("assignment_id"),
                    "runtime_agent_type": result.get(
                        "runtime_agent_type", result.get("agent_type")
                    ),
                    "review_role": result.get(
                        "review_role", result.get("agent_type")
                    ),
                    "finding": finding,
                }
            )

    merged = []
    for key, items in groups.items():
        representative = dict(items[0]["finding"])
        representative["supporting_assignments"] = [item["assignment_id"] for item in items]
        representative["supporting_runtime_agent_types"] = [
            item["runtime_agent_type"] for item in items
        ]
        representative["supporting_review_roles"] = [
            item["review_role"] for item in items
        ]
        representative["supporting_agent_types"] = list(
            representative["supporting_review_roles"]
        )
        representative["duplicate_count"] = len(items)
        merged.append(representative)

    output = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": utc_now(),
        "result_count": len(results),
        "unique_finding_count": len(merged),
        "findings": merged,
        # Retained for consumers of the previous schema, and now always empty by
        # construction: anything that would have landed here refuses the merge.
        "errors": [],
        # What licensed each merged artifact, recorded next to the findings it
        # contributed rather than only in the validator's transient stdout.
        "attribution": {
            "policy": (
                "every merged result is attributed by a consumed ADMISSIONS.jsonl "
                "row whose digest matches the bytes on disk, or pinned by "
                "LEGACY_RESULTS_MANIFEST.json; verified by "
                "validate_harness.collect_result_attribution"
            ),
            "legacy_results_manifest": attribution.legacy_status,
            "results": provenance,
        },
    }
    out = run_dir / "MERGED_RESULTS.json"
    dump_json(out, output)
    print(out.relative_to(repo))


if __name__ == "__main__":
    main()
