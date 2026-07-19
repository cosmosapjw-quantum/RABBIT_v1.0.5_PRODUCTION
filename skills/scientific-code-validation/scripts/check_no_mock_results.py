#!/usr/bin/env python3
# Detect suspicious mock/demo/fake validation markers.
# Customize patterns per project.

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache",
}

PATTERNS = [
    r"\bfake[_ -]?validation\b",
    r"\bmock[_ -]?result\b",
    r"\btoy[_ -]?result\b",
    r"\bcalibration[_ -]?factor\b",
    r"\bTODO\b.*\bvalidated\b",
    r"\bpass\b.*\bvalidated\b",
    r"\bdummy\b.*\bbenchmark\b",
]

ALLOWLIST_FILES = {
    "check_no_mock_results.py",
    "SKILL.md",
}

def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in ALLOWLIST_FILES:
            continue
        if path.suffix.lower() not in {
            ".py", ".rs", ".jl", ".f90", ".f", ".md", ".tex",
            ".yaml", ".yml", ".toml", ".json", ".sh",
        }:
            continue
        yield path

def main() -> int:
    hits = []
    regs = [re.compile(p, re.IGNORECASE) for p in PATTERNS]
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for reg in regs:
                if reg.search(line):
                    hits.append((path, i, line.strip()))
    if hits:
        print("Suspicious validation/mock/calibration markers found:")
        for path, i, line in hits:
            print(f"{path}:{i}: {line}")
        return 1
    print("No suspicious mock/fake validation markers found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
