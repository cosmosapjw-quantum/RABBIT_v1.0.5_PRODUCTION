"""V3 adjudicator. SCRATCH ONLY.

Criteria come from the sealed protocol by sha256 -- never re-typed. Two-way cover between the
sealed items and the predicate table; arity must match per item; an unevaluable condition is
recorded with its reason and caps the item at PARTIAL. Statuses per the sealed rule:
CLOSED = all conditions true; PARTIAL = mixed; UNMEASURED = none true.
"""

from __future__ import annotations

import json, statistics, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v3 import load_sealed_criteria, REPO, EPS   # noqa: E402

V2RUN = REPO / ".agent-harness/runs/run-20260805-f10-v2-diagnostic/domain"
R4REF = Path(__file__).resolve().parent / "r4_reference.json"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class Unevaluable(RuntimeError):
    pass


# ---------- helpers ----------------------------------------------------------------------------

def leg_dir(root: Path, leg: str) -> Path:
    d = root / leg / "domain"
    if not d.exists():
        raise Unevaluable(f"leg output missing: {d}")
    return d


def factor_rows(d: Path) -> list[dict]:
    rows = load(d / "jac_factor.jsonl")
    if not rows:
        raise Unevaluable(f"no jac_factor rows in {d}")
    return rows


# ---------- v3a --------------------------------------------------------------------------------

def u1_c1(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    obs = sorted(d.glob("obs_jac_*.npz"))
    if len(obs) < 2:
        raise Unevaluable(f"only {len(obs)} pinned observation Jacobians exist")
    return True, {"pinned_obs_jacobians": [p.name for p in obs]}


def u1_c2(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    rc = sorted(d.glob("ratcheted_cols_*.npz"))
    if len(rc) < 2:
        raise Unevaluable(f"only {len(rc)} ratcheted-column recomputes exist")
    return True, {"ratcheted_recomputes": [p.name for p in rc]}


def u1_c3(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    pairs = []
    for rcp in sorted(d.glob("ratcheted_cols_*.npz")):
        c = rcp.stem.split("_")[-1]
        op = d / f"obs_jac_{c}.npz"
        if not op.exists():
            continue
        z_o, z_r = np.load(op), np.load(rcp)
        J, cols, C = z_o["J"], z_r["cols"], z_r["C"]
        dev = []
        for k, j in enumerate(cols):
            pin = J[:, j]
            scale = np.abs(pin).max()
            if scale == 0:
                continue
            dev.append(float(np.abs(C[:, k] - pin).max() / scale))
        allcols = np.abs(J).max(axis=0)
        pairs.append({"checkpoint": c, "n_ratcheted_cols": int(len(cols)),
                      "ratcheted_col_relative_deviation": {
                          "median": float(np.median(dev)), "max": float(np.max(dev)),
                          "n_over_1pct": int(sum(x > 0.01 for x in dev)),
                          "n_over_10pct": int(sum(x > 0.10 for x in dev))},
                      "note": "non-ratcheted columns identical by construction (same pinned J)"})
    if not pairs:
        raise Unevaluable("no obs/ratcheted pair shares a checkpoint")
    ctx["u1_pairs"] = pairs
    return True, {"pairs": pairs}


def u2_c1(ctx):
    return u1_c3(ctx)[0], {"via": "U1 artifacts"}


def u2_c2(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    fr = factor_rows(d)
    ratcheted = set(fr[-1]["ratcheted_components"])
    overlaps = []
    for rcp in sorted(d.glob("ratcheted_cols_*.npz")):
        c = rcp.stem.split("_")[-1]
        op = d / f"obs_jac_{c}.npz"
        if not op.exists():
            continue
        z_o, z_r = np.load(op), np.load(rcp)
        J, cols, C = z_o["J"], z_r["cols"], z_r["C"]
        devs = {}
        for k, j in enumerate(cols):
            pin = J[:, int(j)]
            s = np.abs(pin).max()
            devs[int(j)] = float(np.abs(C[:, k] - pin).max() / s) if s else 0.0
        top = {j for j, v in sorted(devs.items(), key=lambda kv: -kv[1])[:max(3, len(devs) // 3)]}
        inter = top & ratcheted
        overlaps.append({"checkpoint": c, "top_deviation_set": sorted(top)[:20],
                         "overlap_with_ratcheted": len(inter), "top_size": len(top)})
    if not overlaps:
        raise Unevaluable("no pair to compute the overlap statistic on")
    return True, {"ratcheted_set_size": len(ratcheted), "overlaps": overlaps}


def u4_c1(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    out = []
    for op in sorted(d.glob("obs_jac_*.npz")):
        z = np.load(op)
        w = z["eig"]
        out.append({"file": op.name, "N": float(z["t"]),
                    "abscissa": float(w.real.max()), "n_pos": int((w.real > 0).sum()),
                    "n_complex_pairs": int((np.abs(w.imag) > 1e-12 * np.abs(w)).sum() // 2)})
    if not out:
        raise Unevaluable("no pinned-step creep-state spectrum exists")
    return True, {"spectra": out}


def u15b_c1(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    fr = factor_rows(d)
    init = [r for r in fr if r.get("tag") == "init"]
    return bool(init), {"init_rows": len(init),
                        "init_factor_max_over_sqrt_eps": (init[0]["max"] / init[0]["sqrt_eps"]) if init else None}


def u15b_c2(ctx):
    d = leg_dir(ctx["root"], "v3a_r2")
    tr, nw = load(d / "trials.jsonl"), load(d / "newton_calls.jsonl")
    fr = factor_rows(d)
    cov = {"trials_first": min((t["raw"] for t in tr), default=None),
           "newton_first": min((x["raw"] for x in nw), default=None),
           "factor_first": min(r["at_raw"] for r in fr)}
    ok = all(v is not None for v in cov.values()) and \
        max(t["raw"] for t in tr if t["raw"] <= 351) >= 180 if tr else False
    return bool(ok), {"coverage": cov,
                      "trials_within_351": sum(1 for t in tr if t["raw"] <= 351),
                      "newton_within_351": sum(1 for x in nw if x["raw"] <= 351)}


# ---------- v3b --------------------------------------------------------------------------------

def u3_c1(ctx):
    d = leg_dir(ctx["root"], "v3b_r2")
    s = json.loads((d / "summary.json").read_text())
    fr = factor_rows(d)
    done = s.get("solver_status") == "finished" or (s.get("stop_reason") or "").startswith("call_cap") \
        or (s.get("stop_reason") or "") == "wall_cap"
    return bool(done), {"stop_reason": s.get("stop_reason"), "solver_status": s.get("solver_status"),
                        "N_reached": s.get("N_reached"), "factor_rows": len(fr)}


def u3_c2(ctx):
    d = leg_dir(ctx["root"], "v3b_r2")
    fr = factor_rows(d)
    traj = [{"tag": r.get("tag"), "at_raw": r["at_raw"],
             "max_over_sqrt_eps": r["max"] / r["sqrt_eps"],
             "median_over_sqrt_eps": r["median"] / r["sqrt_eps"],
             "n_above_10x": r["n_above_10x"], "n_above_100x": r["n_above_100x"]} for r in fr]
    ctx["u3_traj"] = traj
    return True, {"refreshes": len(traj), "trajectory": traj}


def u3_c3(ctx):
    d = leg_dir(ctx["root"], "v3b_r2")
    fr = factor_rows(d)
    exceeded = any(r["max"] > 100 * r["sqrt_eps"] for r in fr)
    peak = max(r["max"] / r["sqrt_eps"] for r in fr)
    return True, {"base_factor_exceeded_100x": bool(exceeded), "peak_over_sqrt_eps": peak}


def u8b_c1(ctx):
    d = leg_dir(ctx["root"], "v3b_r2")
    rhs = load(d / "rhs_calls.jsonl")
    if not rhs:
        raise Unevaluable("no rhs rows for base leg")
    ctx["v3b_rhs"] = rhs
    return True, {"rows": len(rhs)}


def u8b_c2(ctx):
    rhs = ctx.get("v3b_rhs") or load(leg_dir(ctx["root"], "v3b_r2") / "rhs_calls.jsonl")
    changes = [{"i": b["i"], "N": b["N"], "from": a["dom_rej"], "to": b["dom_rej"]}
               for a, b in zip(rhs, rhs[1:]) if a["dom_rej"] != b["dom_rej"]]
    return True, {"distinct_values": len({r["dom_rej"] for r in rhs}),
                  "n_changes": len(changes), "change_points": changes[:40]}


# ---------- v3c --------------------------------------------------------------------------------

SEG_NS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0)


def _segments(ctx):
    segs = {}
    for n in SEG_NS:
        d = ctx["root"] / f"v3c_N{n}" / "domain"
        if d.exists():
            segs[n] = d
    return segs


def u5_c1(ctx):
    segs = _segments(ctx)
    if len(segs) < 6:
        raise Unevaluable(f"only {len(segs)} of 6 segments have output")
    bad = {}
    for n, d in segs.items():
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        if not s or not (d / "jac_factor.jsonl").exists() or not (d / "rhs_calls.jsonl").exists():
            bad[n] = "missing summary/factor/rhs"
    if bad:
        raise Unevaluable(f"segments incomplete: {bad}")
    return True, {"segments": {str(n): str(d) for n, d in segs.items()}}


def u5_c2(ctx):
    crit = ctx["criteria"]["items"]["U5_late_epoch_map"]["thresholds"]
    segs = _segments(ctx)
    if not segs:
        raise Unevaluable("no segments")
    out = {}
    for n, d in sorted(segs.items()):
        s = json.loads((d / "summary.json").read_text())
        rhs = load(d / "rhs_calls.jsonl")
        fr = load(d / "jac_factor.jsonl")
        init_end = s.get("init_raw_calls") or 0
        init_end = max(init_end, 183)
        win = [r for r in rhs if init_end < r["i"] <= init_end + 1000]
        if len(win) < 50:
            out[str(n)] = {"class": "UNRESOLVED", "reason": f"only {len(win)} calls in rate window"}
            continue
        rate = (win[-1]["N"] - win[0]["N"]) / (win[-1]["i"] - win[0]["i"])
        ratchet = any(r["max"] > crit["ratchet_if_factor_exceeds_x_sqrt_eps"] * r["sqrt_eps"] for r in fr)
        if rate < crit["creep_if_N_per_call_below"]:
            cls = "CREEP"
        elif rate > crit["normal_if_N_per_call_above"]:
            cls = "NORMAL"
        else:
            cls = "AMBIGUOUS"
        out[str(n)] = {"class": cls, "N_per_call": rate, "ratchet": bool(ratchet),
                       "stop_reason": s.get("stop_reason"), "N_reached": s.get("N_reached"),
                       "raw_calls": s.get("raw_calls_integration")}
    ctx["u5_map"] = out
    return True, {"classification": out, "thresholds": crit}


def u5_c3(ctx):
    segs = _segments(ctx)
    missing = [str(n) for n, d in segs.items()
               if not (d / "lift_record.json").exists()
               or "CONSTRUCTED" not in (d / "lift_record.json").read_text()]
    return not missing, {"segments_missing_caveat": missing}


# ---------- tier 0, computed in-adjudicator ----------------------------------------------------

def u6_c1(ctx):
    ref = json.loads(R4REF.read_text())["domain"]
    rhs = load(V2RUN / "rhs_calls.jsonl")
    by_i = {r["i"]: r for r in rhs}
    rows, mism, skipped = [], [], []
    for k, v in sorted(ref.items(), key=lambda kv: int(kv[0])):
        i = int(k)
        if i not in by_i:
            skipped.append(i); continue
        got = by_i[i]
        pair = {"i": i, "r4_N": v["N"], "new_N": f"{got['N']:.4f}",
                "r4_T": v["T_cm"], "new_T": f"{got['T_cm']:.5f}"}
        rows.append(pair)
        if pair["r4_N"] != pair["new_N"] or pair["r4_T"] != pair["new_T"]:
            mism.append(pair)
    creep = [i for i in range(951, 1202, 50)]
    creep_compared = [r["i"] for r in rows if r["i"] in creep]
    ctx["u6"] = {"compared": len(rows), "mismatches": mism, "skipped": skipped,
                 "creep_window_compared": creep_compared}
    ok = set(creep_compared) == set(creep) and not mism
    return ok, ctx["u6"]


def u6_c2(ctx):
    u = ctx.get("u6") or u6_c1(ctx)[1]
    return True, {"skipped_counted": u["skipped"], "n_skipped": len(u["skipped"])}


def u7_c1(ctx):
    ref = json.loads(R4REF.read_text())["domain"]
    tr = load(V2RUN / "trials.jsonl")
    acc = load(V2RUN / "accepted.jsonl")
    align = []
    for k in ("701", "751", "801", "851", "901", "951", "1001"):
        if k not in ref:
            continue
        i = int(k); n_pr = ref[k]["N"]
        cand_t = [t for t in tr if f"{t['t']:.4f}" == n_pr]
        cand_a = [a for a in acc if f"{a['t']:.4f}" == n_pr]
        align.append({"eval": i, "printed_N": n_pr,
                      "matches_trial": [{"t": t["t"], "outcome": t["outcome"], "raw": t["raw"]}
                                        for t in cand_t[:3]],
                      "matches_accepted": [{"t": a["t"], "raw": a["raw"]} for a in cand_a[:3]]})
    ctx["u7_align"] = align
    return bool(align), {"alignment": align}


def u7_c2(ctx):
    align = ctx.get("u7_align") or u7_c1(ctx)[1]["alignment"]
    a1813 = next((a for a in align if a["printed_N"] == "0.1813"), None)
    a1629 = next((a for a in align if a["printed_N"] == "0.1629"), None)
    if not a1813 or not a1629:
        raise Unevaluable("0.1813 or 0.1629 not present in the aligned window")
    v_1813 = {"print": "0.1813",
              "assignment": "trial" if a1813["matches_trial"] and not a1813["matches_accepted"]
              else "accepted" if a1813["matches_accepted"] else "unmatched",
              "evidence": a1813}
    v_1629 = {"print": "0.1629",
              "assignment": "accepted" if a1629["matches_accepted"]
              else "trial" if a1629["matches_trial"] else "unmatched",
              "evidence": a1629}
    ok = v_1813["assignment"] != "unmatched" and v_1629["assignment"] != "unmatched"
    return ok, {"0.1813": v_1813, "0.1629": v_1629}


def u8a_c1(ctx):
    rhs = load(V2RUN / "rhs_calls.jsonl")
    if not rhs:
        raise Unevaluable("V2 rhs rows missing")
    changes = [{"i": b["i"], "from": a["dom_rej"], "to": b["dom_rej"]}
               for a, b in zip(rhs, rhs[1:]) if a["dom_rej"] != b["dom_rej"]]
    return True, {"rows": len(rhs), "distinct": len({r["dom_rej"] for r in rhs}),
                  "n_changes": len(changes), "change_points": changes[:20]}


def _artifact(ctx, name, fields):
    p = ctx["root"] / "tier0" / name
    if not p.exists():
        raise Unevaluable(f"artifact missing: {p.name}")
    d = json.loads(p.read_text())
    missing = [f for f in fields if f not in d]
    if missing:
        return False, {"artifact": p.name, "missing_fields": missing}
    return True, {"artifact": p.name, "fields_present": fields,
                  "payload_excerpt": {k: d[k] for k in fields[:4]}}


def u9_c1(ctx):
    return _artifact(ctx, "U9_scipy_version_diff.json",
                     ["versions_compared", "num_jac_diff_lines", "bdf_diff_lines"])


def u9_c2(ctx):
    return _artifact(ctx, "U9_scipy_version_diff.json",
                     ["behavioural_difference", "verdict", "line_references"])


def u10_c1(ctx):
    return _artifact(ctx, "U10_factor_lifetime.json", ["storage_site", "line_references"])


def u10_c2(ctx):
    return _artifact(ctx, "U10_factor_lifetime.json", ["reset_on_new_instance"])


def u10_c3(ctx):
    return _artifact(ctx, "U10_factor_lifetime.json", ["lsoda_shares_mechanism", "lsoda_explanation"])


def u12_c1(ctx):
    return _artifact(ctx, "U12_h_nonrecovery.json",
                     ["control_flow_lines", "v2_counts", "max_reachable_order_given_failures",
                      "closure_statement"])


def u13_c1(ctx):
    acc = load(V2RUN / "accepted.jsonl")
    nw = load(V2RUN / "newton_calls.jsonl")
    steady = [a for a in acc if 351 <= a["raw"] <= 900]
    creep = [a for a in acc if a["raw"] > 951]
    if not steady or not creep:
        raise Unevaluable("V2 windows empty")
    nw_steady = [x for x in nw if 351 <= x["raw"] <= 900]
    nw_creep = [x for x in nw if x["raw"] > 951]
    return True, {
        "design_input_not_readjudication": True,
        "steady_window_calls": "351-900",
        "h_median_steady": statistics.median(a["h_abs"] for a in steady),
        "h_median_creep": statistics.median(a["h_abs"] for a in creep),
        "newton_nonconv_rate_steady": sum(1 for x in nw_steady if not x["converged"]) / max(len(steady), 1),
        "newton_nonconv_rate_creep": sum(1 for x in nw_creep if not x["converged"]) / max(len(creep), 1),
    }


def u15a_c1(ctx):
    tr = load(V2RUN / "trials.jsonl")
    nw = load(V2RUN / "newton_calls.jsonl")
    early_t = [t for t in tr if t["raw"] <= 351]
    early_n = [x for x in nw if x["raw"] <= 351]
    if not early_t:
        raise Unevaluable("no V2 trials within calls 1-351")
    return True, {"trials": len(early_t),
                  "outcomes": {o: sum(1 for t in early_t if t["outcome"] == o)
                               for o in {t["outcome"] for t in early_t}},
                  "newton_calls": len(early_n),
                  "newton_nonconv": sum(1 for x in early_n if not x["converged"]),
                  "h_first": early_t[0]["h_abs"] if early_t else None,
                  "h_at_351": max((t["h_abs"] for t in early_t), default=None),
                  "factor_availability": "V2 logged jac_factor only at refreshes (first at call 521); "
                                         "startup factor evolution unavailable there, RECORDED AS SUCH; "
                                         "V3a's init-tagged log closes it going forward"}


PREDICATES = {
    "U1_ratchet_corruption": [u1_c1, u1_c2, u1_c3],
    "U2_p1_conjunct2": [u2_c1, u2_c2],
    "U4_creep_spectrum": [u4_c1],
    "U15b_startup_full": [u15b_c1, u15b_c2],
    "U3_base_ratchet": [u3_c1, u3_c2, u3_c3],
    "U8b_base_domrej": [u8b_c1, u8b_c2],
    "U5_late_epoch_map": [u5_c1, u5_c2, u5_c3],
    "U6_v2_repro_creep": [u6_c1, u6_c2],
    "U7_drop_closure": [u7_c1, u7_c2],
    "U8a_v2_domrej": [u8a_c1],
    "U9_scipy_version_diff": [u9_c1, u9_c2],
    "U10_factor_lifetime": [u10_c1, u10_c2, u10_c3],
    "U12_h_nonrecovery": [u12_c1],
    "U13_threshold_recal": [u13_c1],
    "U15a_startup_v2": [u15a_c1],
}


def adjudicate(root: Path) -> dict:
    criteria = load_sealed_criteria()
    sealed = criteria["items"]
    missing = sorted(set(sealed) - set(PREDICATES))
    extra = sorted(set(PREDICATES) - set(sealed))
    arity = {m: (len(sealed[m]["closed_when"]), len(PREDICATES[m]))
             for m in sealed if m in PREDICATES
             if len(sealed[m]["closed_when"]) != len(PREDICATES[m])}
    if missing or extra or arity:
        return {"decision": "FAIL_PROTOCOL",
                "cover": {"sealed_without_predicate": missing,
                          "predicate_without_seal": extra, "arity_mismatch": arity}}
    ctx = {"root": root, "criteria": criteria}
    items = {}
    for iid, spec in sealed.items():
        rows = []
        for k, (text, fn) in enumerate(zip(spec["closed_when"], PREDICATES[iid])):
            try:
                ok, ev = fn(ctx)
                rows.append({"index": k, "sealed_text": text, "value": bool(ok), "evidence": ev})
            except Unevaluable as exc:
                rows.append({"index": k, "sealed_text": text, "value": None,
                             "unevaluable": str(exc)})
            except Exception as exc:                        # a predicate crash is preserved, not hidden
                rows.append({"index": k, "sealed_text": text, "value": None,
                             "unevaluable": f"predicate error: {type(exc).__name__}: {exc}"})
        vals = [r["value"] for r in rows]
        if all(v is True for v in vals):
            status = "CLOSED"
        elif any(v is True for v in vals):
            status = "PARTIAL"
        else:
            status = "UNMEASURED"
        items[iid] = {"leg": spec["leg"], "status": status, "conditions": rows}
    counts = {s: sum(1 for i in items.values() if i["status"] == s)
              for s in ("CLOSED", "PARTIAL", "UNMEASURED")}
    return {"decision": "ADJUDICATED", "criteria_sha_verified": True,
            "items": items, "counts": counts,
            "excluded": criteria["excluded"], "claim_ceiling": criteria["claim_ceiling"]}


def main() -> None:
    root = Path(sys.argv[1])
    out = adjudicate(root)
    (root / "ANALYSIS_V3.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"decision": out["decision"],
                      "counts": out.get("counts"),
                      "statuses": {k: v["status"] for k, v in out.get("items", {}).items()}},
                     indent=2))


if __name__ == "__main__":
    main()
