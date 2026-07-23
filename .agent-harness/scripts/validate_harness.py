#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

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

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "context_version": actual, "active_run": active}, indent=2))


if __name__ == "__main__":
    main()
