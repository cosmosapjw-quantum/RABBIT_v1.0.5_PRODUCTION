#!/usr/bin/env python3
# Minimal validation dispatcher.
# Customize this per project. This template tries common validation commands
# and records what was actually run.

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "docs" / "harness" / "validation_runs"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ["python", "-m", "pytest", "-q"],
    ["ruff", "check", "."],
    ["mypy", "."],
    ["cargo", "test"],
    ["cargo", "check"],
]

def exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def should_run(command: list[str]) -> bool:
    if command[0] == "python":
        return (ROOT / "pyproject.toml").exists() or (ROOT / "pytest.ini").exists() or (ROOT / "tests").exists()
    if command[0] == "cargo":
        return (ROOT / "Cargo.toml").exists()
    return exists(command[0])

def run(command: list[str]) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=600,
        )
        return {
            "command": command,
            "started": started,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "started": started,
            "returncode": "TIMEOUT",
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
        }

def main() -> int:
    results = []
    for command in CANDIDATES:
        if should_run(command):
            results.append(run(command))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT / f"validation_{stamp}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote validation results: {path}")
    if not results:
        print("No validation commands detected. Customize scripts/run_validation.py.")
        return 2

    failed = [r for r in results if r["returncode"] != 0]
    if failed:
        print(f"{len(failed)} validation command(s) failed.")
        return 1

    print("All detected validation commands passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
