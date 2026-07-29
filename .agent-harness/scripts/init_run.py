#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from _harness import dump_json, load_json, root, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--spec-ref", default="SPEC.md")
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--head-ref", default="HEAD")
    args = parser.parse_args()

    repo = root()
    harness = repo / ".agent-harness"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = harness / "runs" / run_id

    index = load_json(harness / "context" / "CONTEXT_INDEX.json")
    version = str(index.get("context_version", "UNBUILT"))
    if version == "UNBUILT":
        raise SystemExit("Build the context pack before initializing a run.")

    template = load_json(harness / "templates" / "RUN_PLAN.json")
    template.update(
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "spec_ref": args.spec_ref,
            "base_ref": args.base_ref,
            "head_ref": args.head_ref,
            "context_version": version,
        }
    )
    try:
        run_dir.mkdir(parents=True)  # atomic claim: closes the create/create race
    except FileExistsError:
        raise SystemExit(f"Run already exists: {run_id}")
    for name in ["assignments", "results", "raw_logs", "artifacts"]:
        (run_dir / name).mkdir(exist_ok=True)
    dump_json(run_dir / "RUN_PLAN.json", template)

    active = harness / "ACTIVE_RUN"
    previous = active.read_text(encoding="utf-8").strip() if active.is_file() else ""
    leases_dir = harness / "leases"
    if previous and previous != run_id and leases_dir.is_dir():
        held = []
        for lease in sorted(leases_dir.glob("*.json")):
            if lease.name.startswith(".tmp."):
                continue
            try:
                data = json.loads(lease.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(data, dict)
                and data.get("run_id") == previous
                and data.get("state") != "consumed"
            ):
                held.append(lease.stem)
        if held:
            print(
                f"warning: {len(held)} outstanding subagent lease(s) still reference "
                f"run {previous}; their Stop validation stays bound to that run: "
                + ", ".join(held),
                file=sys.stderr,
            )
    tmp = active.with_name(f"ACTIVE_RUN.tmp.{os.getpid()}")
    tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(tmp, active)
    print(run_id)


if __name__ == "__main__":
    main()
