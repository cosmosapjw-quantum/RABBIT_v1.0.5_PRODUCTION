from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

LEASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject a repeated object key instead of silently keeping the last value.

    The same rule as ``_harness.loads_strict``, which that module's docstring
    already claims holds for "every evidence-bearing document in the admission
    path: envelope, receipt, assignment, result, and each ADMISSIONS.jsonl row".
    It did not: the receipt and the run lease reach Stop through ``load_json``
    below, which was a plain ``json.loads``. BD623 R5 fed one receipt carrying
    two ``expected_agent_id`` keys to both readers -- ``admit_agent`` refused it
    as ambiguous while Stop accepted it and resolved the SECOND value, with a
    reader keeping the first seeing a different agent entirely. The same held
    for a lease with two ``run_id`` keys, and the lease's ``run_id`` selects the
    run directory, the ledger, the receipt and the agent lock.

    Ambiguity is refused rather than resolved, because the whole mechanism is an
    argument about which agent wrote which result, and a document that answers
    that differently depending on the parser is not evidence.
    """
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise json.JSONDecodeError(f"duplicate object key {key!r}", "", 0)
        seen.add(key)
    return dict(pairs)


def read_stdin_json() -> dict[str, Any]:
    import sys

    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return Path(out).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return Path.cwd().resolve()


def load_json(path: Path, default: Any = None) -> Any:
    """Parse a hook-side JSON document, refusing ambiguous identity bytes.

    Returning ``default`` for a duplicate-key document is fail-CLOSED at both
    call sites that matter, which is why the rule can land here rather than in
    the two hook files: Stop blocks on a lease that "exists but cannot be
    parsed", and blocks on a receipt that is "absent or unparseable". An
    ambiguous document now takes those paths instead of resolving last-wins.
    """
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
        )
    except (OSError, json.JSONDecodeError):
        return default


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lease_path(harness: Path, agent_id: str) -> Path | None:
    """Per-agent run lease recorded at SubagentStart; None when agent_id is unsafe."""
    if not LEASE_ID_RE.fullmatch(agent_id or ""):
        return None
    return harness / "leases" / f"{agent_id}.json"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{os.getpid()}.{path.name}")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def emit_additional_context(event: str, text: str, *, warning: str | None = None) -> None:
    payload: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": text,
        }
    }
    if warning:
        payload["systemMessage"] = warning
    print(json.dumps(payload, ensure_ascii=False))
