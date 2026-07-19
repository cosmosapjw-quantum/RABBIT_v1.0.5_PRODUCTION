#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: check_latex_log.py build.log")
    raise SystemExit(2)

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="ignore")

checks = {
    "undefined_refs": r"Reference `[^']+' on page .* undefined|There were undefined references",
    "undefined_cites": r"Citation `[^']+' on page .* undefined|There were undefined citations",
    "missing_files": r"LaTeX Error: File `[^']+' not found",
    "overfull_hbox": r"Overfull \\hbox",
    "duplicate_labels": r"Label `[^']+' multiply defined",
}

print(f"Latex log scan: {log_path}")
any_hit = False
for name, pattern in checks.items():
    hits = re.findall(pattern, text)
    if hits:
        any_hit = True
        print(f"- {name}: {len(hits)} hit(s)")
if not any_hit:
    print("- no common LaTeX warning patterns detected")
