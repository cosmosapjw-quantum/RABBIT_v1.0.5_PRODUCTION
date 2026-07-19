#!/usr/bin/env python3
from __future__ import annotations

import json

from _harness import active_run_id, hash_files, load_json, root

VALID_STATUS = {"pass", "fail", "inconclusive", "error"}


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
            if not isinstance(result, dict):
                errors.append(f"Result is not a JSON object: {result_path.name}")
                continue
            expected = {
                "run_id": active,
                "assignment_id": assignment.get("assignment_id"),
                "context_version": index.get("context_version"),
                "agent_type": assignment.get("agent_type"),
                "result_path": assignment.get("result_path"),
            }
            for key, expected_value in expected.items():
                if str(result.get(key)) != str(expected_value):
                    errors.append(f"Result {key} mismatch: {result_path.name}")
            if str(result.get("status")) not in VALID_STATUS:
                errors.append(f"Result status is invalid: {result_path.name}")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "context_version": actual, "active_run": active}, indent=2))


if __name__ == "__main__":
    main()
