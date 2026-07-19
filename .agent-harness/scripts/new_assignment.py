#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _harness import active_run_id, dump_json, load_json, root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--agent-type", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--parent-assignment-id")
    parser.add_argument(
        "--independence-mode",
        choices=["shared-core", "blind-results", "adjudication"],
        default="shared-core",
    )
    parser.add_argument(
        "--discovery-mode",
        choices=["targeted", "independent"],
        default="targeted",
    )
    parser.add_argument("--claim-id", action="append", default=[])
    parser.add_argument("--may-spawn", action="store_true")
    args = parser.parse_args()

    repo = root()
    harness = repo / ".agent-harness"
    run_id = active_run_id(repo)
    run_dir = harness / "runs" / run_id
    plan = load_json(run_dir / "RUN_PLAN.json")
    index = load_json(harness / "context" / "CONTEXT_INDEX.json")
    assignments_dir = run_dir / "assignments"
    existing = sorted(assignments_dir.glob("*.json"))
    max_total = int(plan["budget"]["max_total"])
    if len(existing) >= max_total:
        raise SystemExit(f"Assignment budget exhausted: {len(existing)}/{max_total}")

    depth = 1
    if args.parent_assignment_id:
        parent_path = assignments_dir / f"{args.parent_assignment_id}.json"
        if not parent_path.is_file():
            raise SystemExit(f"Parent assignment not found: {parent_path}")
        parent = load_json(parent_path)
        if not parent.get("may_spawn", False):
            raise SystemExit("Parent assignment does not grant may_spawn=true.")
        depth = int(parent.get("depth", 1)) + 1
    max_depth = int(plan["budget"]["max_depth"])
    if depth > max_depth:
        raise SystemExit(f"Depth budget exceeded: requested {depth}, max {max_depth}")

    out = assignments_dir / f"{args.assignment_id}.json"
    if out.exists():
        raise SystemExit(f"Assignment already exists: {out}")

    value = load_json(harness / "templates" / "ASSIGNMENT.json")
    value.update(
        {
            "run_id": run_id,
            "assignment_id": args.assignment_id,
            "parent_assignment_id": args.parent_assignment_id,
            "depth": depth,
            "agent_type": args.agent_type,
            "context_version": index["context_version"],
            "independence_mode": args.independence_mode,
            "discovery_mode": args.discovery_mode,
            "may_spawn": args.may_spawn,
            "claim_ids": args.claim_id,
            "task": args.task,
            "result_path": f".agent-harness/runs/{run_id}/results/{args.assignment_id}.json",
            "status": "registered",
        }
    )
    dump_json(out, value)
    print(out.relative_to(repo))
    print(
        f"RUN_ID={run_id} ASSIGNMENT_ID={args.assignment_id} "
        f"CONTEXT_VERSION={index['context_version']} INDEPENDENCE_MODE={args.independence_mode}"
    )


if __name__ == "__main__":
    main()
