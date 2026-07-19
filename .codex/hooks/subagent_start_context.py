#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from _common import emit_additional_context, load_json, read_stdin_json, repo_root

HEADER_FIELDS = ("RUN_ID", "ASSIGNMENT_ID", "CONTEXT_VERSION", "INDEPENDENCE_MODE")
ASSIGNMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def read_bounded(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"[missing file: {path}]"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated at {max_chars} characters; read the file directly for the rest]"


def scalar_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in scalar_texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in scalar_texts(item)]
    return []


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


def main() -> None:
    event = read_stdin_json()
    agent_type = str(event.get("agent_type") or "unknown")
    root = repo_root()
    harness = root / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    pack_path = harness / "generated" / "CONTEXT_PACK.md"
    active_path = harness / "ACTIVE_RUN"
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
    active_run = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else "none"
    event_text = "\n".join(scalar_texts(event))
    header = {
        field: match.group(1)
        for field in HEADER_FIELDS
        if (match := re.search(rf"(?m)^{field}=([^\s]+)\s*$", event_text))
    }
    header_errors: list[str] = []
    assignment: dict[str, Any] | None = None
    if header:
        missing = [field for field in HEADER_FIELDS if field not in header]
        if missing:
            header_errors.append("spawn header is missing " + ", ".join(missing))
        elif header["RUN_ID"] != active_run:
            header_errors.append("spawn RUN_ID does not match ACTIVE_RUN")
        elif not ASSIGNMENT_ID_RE.fullmatch(header["ASSIGNMENT_ID"]):
            header_errors.append("spawn ASSIGNMENT_ID has an invalid form")
        else:
            assignment_path = (
                harness
                / "runs"
                / active_run
                / "assignments"
                / f"{header['ASSIGNMENT_ID']}.json"
            )
            value = load_json(assignment_path, None)
            if not isinstance(value, dict):
                header_errors.append("spawn assignment is not registered")
            else:
                assignment = value
                expected = {
                    "run_id": header["RUN_ID"],
                    "assignment_id": header["ASSIGNMENT_ID"],
                    "context_version": header["CONTEXT_VERSION"],
                    "independence_mode": header["INDEPENDENCE_MODE"],
                }
                for key, expected_value in expected.items():
                    if str(assignment.get(key)) != expected_value:
                        header_errors.append(f"assignment {key} does not match the spawn header")
                if header["CONTEXT_VERSION"] != version:
                    header_errors.append("spawn CONTEXT_VERSION is stale")
    else:
        header_errors.append(
            "hook event did not expose the spawn header; the subagent must verify all four "
            "fields manually before substantive work"
        )

    if assignment is not None:
        agent_type = str(assignment.get("agent_type") or agent_type)
    if agent_type not in index.get("role_files", {}):
        header_errors.append(f"no registered role context for agent_type={agent_type!r}")

    pieces = [read_bounded(pack_path, max_chars)]
    used = len(pieces[0])

    role_files = index.get("role_files", {}).get(agent_type, [])
    for rel in role_files:
        path = root / rel
        remaining = max_chars - used
        if remaining <= 512:
            break
        role_text = read_bounded(path, remaining)
        pieces.append(f"\n\n## Role context: {rel}\n{role_text}")
        used += len(role_text)

    contract = f"""[MANDATORY SUBAGENT BOOTSTRAP]
Agent type: {agent_type}
Active run: {active_run}
Canonical context version: {version}

Your spawn prompt MUST contain RUN_ID, ASSIGNMENT_ID, CONTEXT_VERSION, and INDEPENDENCE_MODE.
Before any broad search or analysis:
1. Verify the prompt CONTEXT_VERSION equals `{version}`.
2. Read `.agent-harness/runs/<RUN_ID>/assignments/<ASSIGNMENT_ID>.json`.
3. Read only files listed by the assignment, plus targeted evidence needed to verify a cited claim.
4. Do not read sibling result files unless INDEPENDENCE_MODE is `adjudication` or the assignment explicitly allows them.
5. Write only to the unique result path in the assignment.
6. End with the required one-line HARNESS_RESULT JSON envelope.
If any required field or file is missing, stop substantive work and return status `error`.

Hook preflight: {"PASS" if not header_errors else "ADVISORY/FAIL — " + "; ".join(header_errors)}

""" + "\n".join(pieces)
    emit_additional_context(
        "SubagentStart",
        contract,
        warning="Subagent header/role preflight requires manual reconciliation" if header_errors else None,
    )


if __name__ == "__main__":
    main()
