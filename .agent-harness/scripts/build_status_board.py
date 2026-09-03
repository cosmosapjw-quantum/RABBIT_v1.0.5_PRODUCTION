#!/usr/bin/env python3
"""Render the gate and claim board from the registries. Nothing else may state a status.

WHY THIS EXISTS (D-073).

Live status was independently authored and then parsed. That parser lost the same
argument seven times -- negation cues, affirmative connectors, historicity, decoy
tokens, letter case, invisible characters, homoglyphs -- and a third-party design
re-audit then showed an eighth class it still cannot catch: a false SEMANTIC
claim carrying no literal status token at all. Every one of those exists because
a human wrote a status and a machine tried to read it back.

Generation removes the surface instead of enumerating it again. The registries
are the authority; this renders them; the checker requires the rendered bytes to
appear verbatim wherever the board is shown. A status can then be wrong only by
being wrong in the registry, which is one reviewable place instead of five.

TWO RULES ON THE OUTPUT, BOTH LOAD-BEARING.

1. It never emits a ``## `` heading. ``overlay_regions`` decides which sections
   are live by reading headings, so a generator that could write one could move
   its own liveness -- and a generator bug would become an authority bug with no
   independent check.

2. **It never emits a timestamp or an ISO date.** ``top_table_row`` reads dates
   from a leading table's first cell and ``overlay_regions`` reads them from
   headings. A date here would make the rendered bytes a function of the clock:
   every no-op rebuild would dirty the tree, and the byte-equality check would go
   red whenever one copy was rebuilt and another was not. The board is a pure
   function of the registries or it is not checkable.

WHY NO STORED VERSION. ``build_context_pack.py`` stores a ``context_version`` and
the validator compares it. That is the right shape for a verbatim concatenation,
where the question is "have the inputs changed since the build". This is a
PROJECTION, so the stronger and cheaper question is "is this what the current
inputs render to" -- which re-rendering answers directly, with nothing stored
that could itself go stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS_SCRIPTS = Path(__file__).resolve().parent
if str(HARNESS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HARNESS_SCRIPTS))

from check_ssot_consistency import (  # noqa: E402
    BOARD_ARTIFACT,
    GATE_REGISTRY,
    load_claims,
    load_gates,
    render_board,
)
from _harness import root  # noqa: E402


def main() -> None:
    repo = root()
    errors: list[str] = []
    gates = load_gates(repo, errors)
    claims = load_claims(repo, errors)
    if errors:
        for message in errors:
            print(f"build_status_board: {message}", file=sys.stderr)
        raise SystemExit(1)
    if not gates:
        print("build_status_board: no gates; refusing to render an empty board.", file=sys.stderr)
        raise SystemExit(1)

    registry = json.loads((repo / GATE_REGISTRY).read_text(encoding="utf-8"))
    basis = {
        g["gate_id"]: str(g.get("status_basis_legacy") or "")
        for g in registry.get("gates", [])
    }
    body = render_board(gates, claims, basis)
    target = repo / BOARD_ARTIFACT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Generated status board\n\n"
        "This file is rendered by `.agent-harness/scripts/build_status_board.py` from\n"
        "`.agent-harness/context/GATE_REGISTRY.json` and\n"
        "`.agent-harness/context/CLAIM_REGISTRY.jsonl`. It is the only authority for\n"
        "current gate and claim status. Do not edit it, and do not restate a status\n"
        "anywhere else: prose about a status is not checked and has been wrong before.\n\n"
        + body,
        encoding="utf-8",
    )
    print(f"Built {BOARD_ARTIFACT}")
    print(f"rows={len(gates) + len(claims)} (gates={len(gates)} claims={len(claims)})")


if __name__ == "__main__":
    main()
