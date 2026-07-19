from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def root() -> Path:
    try:
        text = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if text:
            return Path(text).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    here = Path(__file__).resolve()
    return here.parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_files(repo: Path, files: list[str]) -> tuple[str, list[tuple[str, str]]]:
    digest = hashlib.sha256()
    entries: list[tuple[str, str]] = []
    for rel in files:
        path = repo / rel
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        entries.append((rel, file_hash))
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest(), entries


def active_run_id(repo: Path) -> str:
    path = repo / ".agent-harness" / "ACTIVE_RUN"
    if not path.exists():
        raise SystemExit("No active run. Use init_run.py first.")
    run_id = path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise SystemExit("ACTIVE_RUN is empty.")
    return run_id
