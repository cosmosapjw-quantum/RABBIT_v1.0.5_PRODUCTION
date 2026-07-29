#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib

from _harness import (
    ASSIGNMENT_ID_RE,
    RESULT_TEMPLATE_PATH,
    active_run_id,
    dump_json,
    hash_files,
    load_json,
    root,
    validate_assignment_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument(
        "--agent-type",
        required=True,
        help="Live Codex runtime agent_type reported by SubagentStart/Stop.",
    )
    parser.add_argument(
        "--review-role",
        help="Logical role context; defaults to --agent-type for compatible runtimes.",
    )
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
    parser.add_argument("--delegable-claim-id", action="append", default=[])
    parser.add_argument("--required-input", action="append", default=[])
    parser.add_argument("--allowed-tool", action="append", default=[])
    parser.add_argument("--allowed-sibling-result", action="append", default=[])
    parser.add_argument("--may-spawn", action="store_true")
    args = parser.parse_args()

    if not ASSIGNMENT_ID_RE.fullmatch(args.assignment_id):
        raise SystemExit("assignment-id has an invalid form")
    if not args.claim_id:
        raise SystemExit("At least one --claim-id is required.")
    if len(set(args.claim_id)) != len(args.claim_id):
        raise SystemExit("Duplicate --claim-id values are forbidden.")
    if args.may_spawn and not args.delegable_claim_id:
        raise SystemExit("--may-spawn requires at least one --delegable-claim-id.")
    if not args.may_spawn and args.delegable_claim_id:
        raise SystemExit("--delegable-claim-id requires --may-spawn.")
    if args.independence_mode == "blind-results" and args.allowed_sibling_result:
        raise SystemExit("blind-results assignments may not read sibling results.")

    repo = root()
    harness = repo / ".agent-harness"
    run_id = active_run_id(repo)
    run_dir = harness / "runs" / run_id
    plan = load_json(run_dir / "RUN_PLAN.json")
    index = load_json(harness / "context" / "CONTEXT_INDEX.json")
    review_role = args.review_role or args.agent_type
    role_files = index.get("role_files", {})
    if review_role not in role_files:
        raise SystemExit(f"No registered role context for review-role={review_role!r}")
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
        delegated = set(parent.get("delegable_claim_ids") or [])
        if not set(args.claim_id).issubset(delegated):
            raise SystemExit("Child claim_ids are not delegated by the parent assignment.")
        depth = int(parent.get("depth", 1)) + 1
    max_depth = int(plan["budget"]["max_depth"])
    if depth > max_depth:
        raise SystemExit(f"Depth budget exceeded: requested {depth}, max {max_depth}")

    out = assignments_dir / f"{args.assignment_id}.json"
    if out.exists():
        raise SystemExit(f"Assignment already exists: {out}")

    result_path = f".agent-harness/runs/{run_id}/results/{args.assignment_id}.json"
    assignment_path = (
        f".agent-harness/runs/{run_id}/assignments/{args.assignment_id}.json"
    )
    value = load_json(harness / "templates" / "ASSIGNMENT.json")
    review_role_files = [str(item) for item in role_files[review_role]]
    review_role_digest, _ = hash_files(repo, review_role_files)
    result_template_digest = hashlib.sha256(
        (repo / RESULT_TEMPLATE_PATH).read_bytes()
    ).hexdigest()
    value.update(
        {
            "run_id": run_id,
            "assignment_id": args.assignment_id,
            "parent_assignment_id": args.parent_assignment_id,
            "depth": depth,
            "agent_type": args.agent_type,
            "runtime_agent_type": args.agent_type,
            "review_role": review_role,
            "review_role_files": review_role_files,
            "review_role_sha256": "sha256:" + review_role_digest,
            "result_template": RESULT_TEMPLATE_PATH,
            "result_template_sha256": "sha256:" + result_template_digest,
            "context_version": index["context_version"],
            "independence_mode": args.independence_mode,
            "discovery_mode": args.discovery_mode,
            "may_spawn": args.may_spawn,
            "delegable_claim_ids": args.delegable_claim_id,
            "claim_ids": args.claim_id,
            "task": args.task,
            "required_inputs": list(
                dict.fromkeys(
                    [
                        ".agent-harness/generated/CONTEXT_PACK.md",
                        assignment_path,
                        RESULT_TEMPLATE_PATH,
                        *review_role_files,
                        *args.required_input,
                    ]
                )
            ),
            "allowed_sibling_results": args.allowed_sibling_result,
            "allowed_tools": list(
                dict.fromkeys(["exec_command", "apply_patch", *args.allowed_tool])
            ),
            "required_outputs": [result_path],
            "result_path": result_path,
            "status": "registered",
        }
    )
    contract_errors = validate_assignment_contract(
        value,
        expected_run_id=run_id,
        expected_context_version=str(index["context_version"]),
        role_files=index.get("role_files", {}),
    )
    if contract_errors:
        raise SystemExit("Invalid assignment contract: " + "; ".join(contract_errors))
    dump_json(out, value)
    print(f"RUN_ID={run_id}")
    print(f"ASSIGNMENT_ID={args.assignment_id}")
    print(f"CONTEXT_VERSION={index['context_version']}")
    print(f"INDEPENDENCE_MODE={args.independence_mode}")
    print(out.relative_to(repo))
    print(
        "next: python3 .agent-harness/scripts/admit_agent.py --assignment-id "
        f"{args.assignment_id}  # mints ADMISSION_TOKEN, the 5th spawn-header line"
    )


if __name__ == "__main__":
    main()
