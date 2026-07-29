#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess

from _harness import (
    active_run_id,
    hash_files,
    load_json,
    root,
    validate_assignment_contract,
    validate_assignment_resource_hashes,
    validate_result_contract,
)


def main() -> None:
    repo = root()
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
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", ".agent-harness/runs"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # Fail closed: an unrunnable integrity check is not a passed one.
        errors.append(
            "Tracked-run-evidence integrity check could not run "
            f"({exc.__class__.__name__}); git is required for validation."
        )
    else:
        # Only tracked-file changes matter here: `??` entries are new run
        # directories that have not been force-added yet, which is the normal
        # state of a live run (round-2 review F-8).
        dirty = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and not line.startswith("??")
        ]
        errors.extend(
            f"Tracked run evidence modified after commit: {entry}" for entry in dirty
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
    for run_dir in sorted((harness / "runs").glob("*")):
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
        ledger_consumed: dict[str, dict] = {}
        for number, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Malformed admission ledger line: {ledger_path.name}:{number}")
                continue
            if isinstance(row, dict) and row.get("event") == "consumed":
                ledger_consumed[str(row.get("assignment_id"))] = row
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
