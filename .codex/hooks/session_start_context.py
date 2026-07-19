#!/usr/bin/env python3
from __future__ import annotations

from _common import emit_additional_context, load_json, repo_root


def main() -> None:
    root = repo_root()
    harness = root / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    active_path = harness / "ACTIVE_RUN"
    index = load_json(index_path, {}) or {}

    if not index_path.exists():
        emit_additional_context(
            "SessionStart",
            "Shared-context harness is not initialized. Do not spawn subagents until "
            ".agent-harness/context/CONTEXT_INDEX.json exists.",
            warning="Shared-context harness missing",
        )
        return

    version = index.get("context_version", "UNBUILT")
    active_run = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else "none"
    text = f"""[SHARED-CONTEXT HARNESS]
Repository root: {root}
Context version: {version}
Active run: {active_run}
Before spawning agents, run `python3 .agent-harness/scripts/build_context_pack.py` and register each assignment with `new_assignment.py`.
The main agent must include RUN_ID, ASSIGNMENT_ID, CONTEXT_VERSION, and INDEPENDENCE_MODE in every spawn prompt.
AGENTS.md policy and .agent-harness files are authoritative; hidden parent-thread context is not a substitute for the context contract.
"""
    emit_additional_context("SessionStart", text)


if __name__ == "__main__":
    main()
