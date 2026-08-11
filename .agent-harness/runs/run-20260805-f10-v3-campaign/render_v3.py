"""Render the V3 report body FROM ANALYSIS_V3.json. SCRATCH ONLY. No hand-transcribed figures."""

from __future__ import annotations

import json, sys
from pathlib import Path


def f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 1e5 or abs(x) < 1e-3:
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    return str(x)


def main() -> None:
    root = Path(sys.argv[1])
    a = json.loads((root / "ANALYSIS_V3.json").read_text())
    L = []
    w = L.append

    w("## Ledger outcome")
    w("")
    c = a["counts"]
    w(f"**{c['CLOSED']} CLOSED · {c['PARTIAL']} PARTIAL · {c['UNMEASURED']} UNMEASURED** "
      f"of {sum(c.values())} tracked items. Decision: `{a['decision']}`.")
    w("")
    w("| Item | Leg | Status | Conditions true / total | Blocking reason if not CLOSED |")
    w("|---|---|---|---|---|")
    for iid, it in a["items"].items():
        conds = it["conditions"]
        t = sum(1 for x in conds if x["value"] is True)
        blocker = next((x.get("unevaluable") or "condition false"
                        for x in conds if x["value"] is not True), "")
        w(f"| `{iid}` | {it['leg']} | **{it['status']}** | {t}/{len(conds)} | {blocker[:90]} |")
    w("")

    w("## Per-item evidence (rendered from the adjudicator's JSON)")
    w("")
    for iid, it in a["items"].items():
        w(f"### `{iid}` — {it['status']}")
        w("")
        for x in it["conditions"]:
            v = "TRUE" if x["value"] is True else ("FALSE" if x["value"] is False else "UNEVALUABLE")
            w(f"- [{v}] {x['sealed_text']}")
            if x.get("unevaluable"):
                w(f"  - reason: {x['unevaluable']}")
            elif x.get("evidence") is not None:
                ev = json.dumps(x["evidence"], default=str)
                if len(ev) > 900:
                    ev = ev[:900] + "…(truncated; full value in ANALYSIS_V3.json)"
                w(f"  - evidence: `{ev}`")
        w("")

    w("## Exclusions, restated")
    w("")
    for xid, why in a["excluded"].items():
        w(f"- **{xid}** — {why}")
    w("")
    w(f"**Claim ceiling:** {a['claim_ceiling']}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
