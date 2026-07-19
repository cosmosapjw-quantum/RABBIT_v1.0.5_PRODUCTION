from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_stdin_json() -> dict[str, Any]:
    import sys

    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
