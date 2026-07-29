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
from pathlib import Path

from _harness import (
    active_run_id,
    hash_files,
    load_json,
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
            row = json.loads(line)
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
        payload = json.loads(raw)
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
    return index, f"ok ({len(index)} pinned)", errors


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
    print(
        json.dumps(
            {
                "ok": True,
                "wrote": LEGACY_MANIFEST_REL,
                "entry_count": len(entries),
                "generated_at_head": head,
                "sha256": sha256_of(target),
            },
            indent=2,
        )
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
        first_lines = "\n".join(pack_path.read_text(encoding="utf-8").splitlines()[:6])
        if f"Context version: `{actual}`" not in first_lines:
            errors.append("Generated CONTEXT_PACK.md carries a stale context version.")

    active = None
    try:
        active = active_run_id(repo)
    except SystemExit:
        pass
    if active:
        run_dir = harness / "runs" / active
        plan = load_json(run_dir / "RUN_PLAN.json")
        if plan.get("context_version") != index.get("context_version"):
            errors.append("Active run was initialized against a stale context version.")
        assignments = sorted((run_dir / "assignments").glob("*.json"))
        if len(assignments) > int(plan["budget"]["max_total"]):
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
            assignment = load_json(assignment_path)
            result = load_json(result_path)
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
    # The legacy manifest is the other half of the attribution rule, so it needs
    # the same append-only treatment: once committed, editing it to bless an
    # extra file, or deleting it, is itself an error.
    errors.extend(
        git_porcelain_errors(repo, LEGACY_MANIFEST_REL, "Legacy results manifest")
    )

    open_admissions: list[str] = []
    open_receipts: set[tuple[str, str]] = set()
    consumed: dict[tuple[str, str], dict] = {}
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
            open_receipts.add((path.parent.name, path.stem))
        elif state == "consumed":
            consumed[(path.parent.name, path.stem)] = receipt

    # Every result in the active run must be attributable: a consumed receipt
    # whose recorded digest still matches the bytes on disk. This is what
    # detects a fabricated or post-hoc edited result artifact, which the
    # git-porcelain check above cannot see for an untracked new file
    # (D-067 review F-D067-08).
    pending_results: list[str] = []
    if active:
        for result_path in sorted((harness / "runs" / active / "results").glob("*.json")):
            receipt = consumed.get((active, result_path.stem))
            if receipt is None:
                if (active, result_path.stem) in open_receipts:
                    # The agent wrote its artifact but has not stopped yet. Report
                    # it rather than erroring, so a live run is not wedged
                    # (D-067 round-2 review F-12).
                    pending_results.append(result_path.stem)
                    continue
                errors.append(
                    "Result artifact has no admission receipt at all: "
                    f"{result_path.relative_to(repo)}"
                )
                continue
            result_sha = "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest()
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
        # Every consume row is retained, not just the last one: collapsing them
        # is what let a second consume quietly overwrite the digest the first had
        # permanently pinned (D-067 round-4 review, defect 2).
        errors.extend(check_declared_supersession(run_dir.name, rows))
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
            ledger_sha = "sha256:" + hashlib.sha256(result_file.read_bytes()).hexdigest()
            if str(row.get("result_sha256")) != ledger_sha:
                errors.append(
                    "Result artifact differs from the bytes recorded in the "
                    f"admission ledger: {run_dir.name}/{aid}"
                )

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

    # Forward reconciliation, result -> attribution, across EVERY run directory.
    # Everything above walks from a record to the file it names, so a file that
    # no record names was invisible: a wholly fabricated result dropped into a
    # non-active run's results/ left the validator reporting ok (D-067 round-4
    # review, defect 1). `.agent-harness/runs/` is gitignored in full, so the
    # porcelain check above cannot see it either -- such a file produces no git
    # status entry at all, not even `??`.
    #
    # The rule: every result artifact is attributed by a consumed ledger row
    # whose digest matches, or pinned by the legacy manifest with a matching
    # digest, or belongs to an admitted agent that has not stopped yet. Nothing
    # else passes.
    legacy_index, legacy_status, legacy_errors = load_legacy_manifest(repo)
    errors.extend(legacy_errors)
    if legacy_status.startswith("ok"):
        # The git check above only binds a *tracked* manifest: an untracked one
        # shows as `??` and is tolerated, exactly as a new run directory is. Say
        # so out loud rather than letting an uncommitted pin read as a sealed
        # one -- while the manifest is uncommitted, an added entry leaves no git
        # trace.
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", LEGACY_MANIFEST_REL],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            errors.append(
                "Could not determine whether the legacy results manifest is "
                f"tracked ({exc.__class__.__name__}); git is required for "
                "validation."
            )
        else:
            if tracked.returncode != 0:
                legacy_status += "; UNCOMMITTED, so the pin is not yet git-bound"
    for run_name, assignment_id, result_path in result_files(harness):
        key = (run_name, assignment_id)
        if key in attributed:
            # Digest already reconciled by the ledger -> result loop above.
            continue
        try:
            result_digest = sha256_of(result_path)
        except OSError as exc:
            errors.append(
                f"Result artifact could not be read ({exc.__class__.__name__}): "
                f"{run_name}/{assignment_id}"
            )
            continue
        pinned = legacy_index.get(key)
        if pinned is not None:
            if pinned != result_digest:
                errors.append(
                    "Legacy result artifact changed since the legacy manifest "
                    f"pinned it: {run_name}/{assignment_id}"
                )
            continue
        if key in open_receipts:
            # An admitted agent that has written its artifact but not stopped.
            # Reported rather than errored so a live run is not wedged
            # (D-067 round-2 review F-12).
            label = (
                assignment_id
                if run_name == active
                else f"{run_name}/{assignment_id}"
            )
            if label not in pending_results:
                pending_results.append(label)
            continue
        errors.append(
            "Result artifact is not attributed: no consumed admission ledger row "
            "and no legacy manifest entry for "
            f"{run_name}/{assignment_id} ({result_digest})."
        )

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
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", str(registry.relative_to(repo))],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            errors.append(
                "Could not determine whether the gate registry is tracked "
                f"({exc.__class__.__name__}); git is required for validation."
            )
        else:
            if tracked.returncode == 0:
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
                "pending_results": pending_results,
                "legacy_results_manifest": legacy_status,
                "ssot_consistency": ssot_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
