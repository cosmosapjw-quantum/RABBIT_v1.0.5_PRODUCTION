#!/usr/bin/env python3

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / f"handoff_package_{STAMP}.zip"

INCLUDE_DIRS = [
    "docs/harness",
    "skills",
    "scripts/harness",
]

INCLUDE_FILES = [
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "Cargo.toml",
]

def add_path(zf: zipfile.ZipFile, path: Path):
    if path.is_file():
        zf.write(path, path.relative_to(ROOT))
    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and ".git" not in child.parts:
                zf.write(child, child.relative_to(ROOT))

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in INCLUDE_DIRS:
        p = ROOT / d
        if p.exists():
            add_path(zf, p)
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            add_path(zf, p)

print(f"Wrote {OUT}")
