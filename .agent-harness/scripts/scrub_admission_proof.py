#!/usr/bin/env python3
"""Replace raw ``admission_proof`` tokens in retained evidence with their digest.

An admission token is a single-use secret that travels in the spawn prompt and
comes back in the agent's final message. Any captured SubagentStop event JSON
therefore contains it in plaintext, and a capture of a *blocked* stop is the
dangerous one, because that receipt is still open. Run directories are retained
and force-added as evidence, so the token would be committed.

This rewrites every ``admission_proof`` value it finds to ``sha256:<hex>`` of the
original, which is exactly what SubagentStop compares against. The evidence stays
verifiable -- a reader can still check the digest against the receipt's
``token_digest`` -- while the reusable secret is gone.

Usage:
    python3 .agent-harness/scripts/scrub_admission_proof.py <path> [<path> ...]
    python3 .agent-harness/scripts/scrub_admission_proof.py --run <run-id>

Idempotent: an already-scrubbed value is left alone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from _harness import dump_json_atomic, root, token_digest

ALREADY_SCRUBBED = re.compile(r"sha256:[0-9a-f]{64}\Z")


def scrub(value: Any) -> tuple[Any, int]:
    """Return (scrubbed, count) for any nested JSON structure."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            if key == "admission_proof" and isinstance(item, str) and item:
                if ALREADY_SCRUBBED.fullmatch(item):
                    out[key] = item
                else:
                    out[key] = token_digest(item)
                    count += 1
            else:
                out[key], sub = scrub(item)
                count += sub
        return out, count
    if isinstance(value, list):
        items = [scrub(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if isinstance(value, str):
        # HARNESS_RESULT markers are embedded as text inside last_assistant_message.
        changed = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            token = match.group(2)
            if ALREADY_SCRUBBED.fullmatch(token):
                return match.group(0)
            changed += 1
            return f"{match.group(1)}{token_digest(token)}"

        # re.subn counts every match, including the ones `replace` leaves alone,
        # so count the real substitutions instead or the pass is never idempotent.
        # Tolerate escaped quoting: a HARNESS_RESULT marker embedded inside
        # another JSON string appears as \"admission_proof\": \"token\", and a
        # regex anchored on bare quotes misses it (F-R3-12).
        scrubbed = re.sub(
            r'(admission_proof\\?"\s*:\s*\\?")([^"\\]+)', replace, value
        )
        return scrubbed, changed
    return value, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--run", default="", help="Scrub every JSON under this run.")
    args = parser.parse_args()

    repo = root()
    targets: list[Path] = list(args.paths)
    if args.run:
        run_dir = repo / ".agent-harness" / "runs" / args.run
        if not run_dir.is_dir():
            print(f"scrub: no such run: {args.run}", file=sys.stderr)
            return 2
        # Every file, not just *.json: tokens also reach .log and .md evidence
        # (D-067 round-3 review F-R3-12).
        targets.extend(sorted(p for p in run_dir.rglob("*") if p.is_file()))
    if not targets:
        parser.error("give at least one path or --run")

    total = 0
    unreadable: list[Path] = []
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unreadable.append(path)
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # Not JSON: scrub the raw text, which also catches the escaped
            # payloads a structural walk misses.
            scrubbed_text, count = scrub(text)
            if count:
                path.write_text(scrubbed_text, encoding="utf-8")
                print(f"scrubbed {count} token(s) as text: {path}")
                total += count
            continue
        scrubbed, count = scrub(value)
        if count:
            dump_json_atomic(path, scrubbed)
            print(f"scrubbed {count} token(s): {path}")
            total += count
    if unreadable:
        # Silence here would read as "clean" (F-R3-12).
        print(f"NOT SCANNED (unreadable): {len(unreadable)} file(s)", file=sys.stderr)
        for path in unreadable:
            print(f"  {path}", file=sys.stderr)
    print(f"total tokens scrubbed: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
