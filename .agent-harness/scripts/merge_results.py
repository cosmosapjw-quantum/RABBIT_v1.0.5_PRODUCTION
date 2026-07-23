#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict

from _harness import active_run_id, dump_json, load_json, root, utc_now


def canonical_key(finding: dict) -> tuple:
    refs = tuple(sorted(str(x) for x in finding.get("evidence_refs", [])))
    return (
        str(finding.get("claim_id", "")),
        str(finding.get("evidence_fingerprint", "")),
        str(finding.get("verdict", "")),
        refs,
    )


def main() -> None:
    repo = root()
    run_id = active_run_id(repo)
    run_dir = repo / ".agent-harness" / "runs" / run_id
    results = []
    errors = []
    for path in sorted((run_dir / "results").glob("*.json")):
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path.relative_to(repo)), "error": str(exc)})
            continue
        results.append(value)

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
        "errors": errors,
    }
    out = run_dir / "MERGED_RESULTS.json"
    dump_json(out, output)
    print(out.relative_to(repo))


if __name__ == "__main__":
    main()
