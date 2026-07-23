#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from _common import emit_additional_context, load_json, read_stdin_json, repo_root

def read_bounded(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"[missing file: {path}]"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated at {max_chars} characters; read the file directly for the rest]"


def context_integrity(root: Path, index: dict[str, Any], pack_path: Path) -> list[str]:
    errors: list[str] = []
    digest = hashlib.sha256()
    file_hashes = index.get("file_hashes", {})
    for rel in index.get("shared_files", []):
        path = root / str(rel)
        try:
            data = path.read_bytes()
        except OSError:
            errors.append(f"missing shared context file: {rel}")
            continue
        actual_file_hash = hashlib.sha256(data).hexdigest()
        if str(file_hashes.get(rel)) != actual_file_hash:
            errors.append(f"stale shared context hash: {rel}")
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    version = str(index.get("context_version", "UNBUILT"))
    if digest.hexdigest() != version:
        errors.append("context_version does not match the shared files")
    try:
        first_lines = "\n".join(pack_path.read_text(encoding="utf-8").splitlines()[:6])
    except OSError:
        errors.append("canonical context pack is unreadable")
    else:
        if f"Context version: `{version}`" not in first_lines:
            errors.append("canonical context pack carries a different version")
    return errors


def active_run_id(harness: Path) -> str:
    active_path = harness / "ACTIVE_RUN"
    if not active_path.is_file():
        return ""
    return active_path.read_text(encoding="utf-8").strip()


def inject_subagent_context(event: dict[str, Any], root: Path) -> None:
    harness = root / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    pack_path = harness / "generated" / "CONTEXT_PACK.md"
    index = load_json(index_path, {}) or {}

    if not index or not pack_path.exists():
        emit_additional_context(
            "SubagentStart",
            "CONTEXT CONTRACT VIOLATION: the canonical context pack is absent or unbuilt. "
            "Do not perform substantive work. Return an error asking the parent to run "
            "`python3 .agent-harness/scripts/build_context_pack.py`.",
            warning="Subagent started without a built context pack",
        )
        return

    integrity_errors = context_integrity(root, index, pack_path)
    if integrity_errors:
        emit_additional_context(
            "SubagentStart",
            "CONTEXT CONTRACT VIOLATION: " + "; ".join(integrity_errors)
            + ". Do not perform substantive work. Ask the parent to rebuild and validate the pack.",
            warning="Subagent started with stale canonical context",
        )
        return

    max_chars = int(index.get("max_injected_chars", 24000))
    version = str(index.get("context_version", "UNBUILT"))
    active_run = active_run_id(harness) or "none"
    agent_id = str(event.get("agent_id") or "")
    runtime_agent_type = str(event.get("agent_type") or "unknown")
    start_errors: list[str] = []
    if not agent_id:
        start_errors.append("SubagentStart event omitted agent_id")
    if runtime_agent_type not in index.get("role_files", {}):
        start_errors.append(
            f"no registered runtime context for agent_type={runtime_agent_type!r}"
        )

    pieces = [read_bounded(pack_path, max_chars)]
    used = len(pieces[0])

    role_files = index.get("role_files", {}).get(runtime_agent_type, [])
    for rel in role_files:
        path = root / rel
        remaining = max_chars - used
        if remaining <= 512:
            break
        role_text = read_bounded(path, remaining)
        pieces.append(f"\n\n## Role context: {rel}\n{role_text}")
        used += len(role_text)

    contract = f"""[MANDATORY SUBAGENT BOOTSTRAP]
Runtime agent type: {runtime_agent_type}
Agent ID: {agent_id or "MISSING"}
Active run: {active_run}
Canonical context version: {version}

VS Code collaboration does not expose `spawn_agent` to project PreToolUse hooks.
The main agent therefore owns registered launch admission and pre/post write hashing.
Before any broad search or analysis:
1. Verify that the first four prompt lines are exactly RUN_ID, ASSIGNMENT_ID,
   CONTEXT_VERSION, and INDEPENDENCE_MODE and that CONTEXT_VERSION equals `{version}`.
2. Read `.agent-harness/runs/<RUN_ID>/assignments/<ASSIGNMENT_ID>.json` and compute its SHA-256.
3. Verify that assignment `runtime_agent_type` equals `{runtime_agent_type}` and that
   its compatibility `agent_type` has the same value.
4. Read the assignment's canonical result template and every `review_role_files`
   entry. Verify `review_role_sha256` and `result_template_sha256` before analysis.
5. Read only files listed by the assignment, plus targeted evidence needed to verify a cited claim.
6. Do not read sibling result files unless INDEPENDENCE_MODE is `adjudication` or the assignment explicitly allows them.
7. Write only to the unique result path in the assignment.
8. Copy Agent ID `{agent_id or "MISSING"}`, runtime type `{runtime_agent_type}`, and
   the assignment `review_role` into the result artifact.
9. Populate top-level `spawn_contract` with the four verified prompt values,
   `prompt_header_verified=true`, `subagent_start_injected=true`,
   `subagent_start_preflight="PASS"`, the assignment SHA-256, runtime type,
   review-role verification/hash, and result-template verification/hash.
10. End with the required one-line HARNESS_RESULT JSON envelope.
If any required field or file is missing, stop substantive work and return status `error`.

Hook preflight: {"PASS — canonical context and runtime identity verified; assignment role verification required" if not start_errors else "FAIL — " + "; ".join(start_errors)}

""" + "\n".join(pieces)
    emit_additional_context(
        "SubagentStart",
        contract,
        warning="SubagentStart preflight failed; do not perform substantive work" if start_errors else None,
    )


def main() -> None:
    event = read_stdin_json()
    root = repo_root()
    inject_subagent_context(event, root)


if __name__ == "__main__":
    main()
