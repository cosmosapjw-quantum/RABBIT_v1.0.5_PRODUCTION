#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
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
    if run_dir.exists():
        raise SystemExit(f"Run already exists: {run_id}")

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
    for name in ["assignments", "results", "raw_logs", "artifacts"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "RUN_PLAN.json", template)
    (harness / "ACTIVE_RUN").write_text(run_id + "\n", encoding="utf-8")
    print(run_id)


if __name__ == "__main__":
    main()
