#!/usr/bin/env python3
# Lightweight claim-status marker checker.
# This does not understand science. It only checks whether strong claim words
# appear without nearby status markers. Customize for your project.

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()

STRONG_PATTERNS = [
    r"\bwe prove\b",
    r"\bwe demonstrate\b",
    r"\bvalidated\b",
    r"\bproduction[- ]ready\b",
    r"\bexact\b",
    r"\bguarantee\b",
    r"\bfully implemented\b",
    r"\bsubmission[- ]ready\b",
]

STATUS_MARKERS = [
    "IMPLEMENTED", "VALIDATED", "DERIVED", "SPECIFIED",
    "PROPOSED", "SPECULATIVE", "DEPRECATED", "FORBIDDEN",
]

EXCLUDE = {".git", ".venv", "node_modules", "__pycache__"}

def iter_docs():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE for part in path.parts):
            continue
        if path.suffix.lower() in {".md", ".tex", ".rst"}:
            yield path

def main() -> int:
    issues = []
    strong = [re.compile(p, re.IGNORECASE) for p in STRONG_PATTERNS]
    for path in iter_docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if any(p.search(line) for p in strong):
                window = "\n".join(lines[max(0, i-3): min(len(lines), i+4)])
                if not any(marker in window for marker in STATUS_MARKERS):
                    issues.append((path, i + 1, line.strip()))
    if issues:
        print("Strong claims without nearby claim-status marker:")
        for path, line_no, line in issues:
            print(f"{path}:{line_no}: {line}")
        return 1
    print("No unmarked strong claims detected.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
