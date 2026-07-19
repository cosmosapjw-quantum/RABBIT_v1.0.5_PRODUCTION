#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def maybe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_rows(path):
    data = json.loads(Path(path).read_text())
    for key in ("rows", "states", "records", "data"):
        if key in data and isinstance(data[key], list):
            return data[key]
    if isinstance(data, list):
        return data
    raise KeyError(f"Could not find rows in {path}")


def species_rows(rows, sp):
    out = []
    for i, r in enumerate(rows):
        s = r.get("species", {}).get(sp, {})
        if not s:
            continue
        raw_signed = maybe_float(s.get("raw_signed", s.get("raw_qdot_signed", s.get("raw_qdot"))))
        orth_signed = maybe_float(s.get("orth_Tmu_signed", s.get("orth_qdot_Tmu_signed", s.get("orth_qdot_Tmu"))))
        proj = maybe_float(s.get("proj_Tmu", s.get("proj_Tmu_signed")))
        raw = abs(raw_signed) if np.isfinite(raw_signed) else abs(maybe_float(s.get("raw_qdot")))
        orth = abs(orth_signed) if np.isfinite(orth_signed) else abs(maybe_float(s.get("orth_qdot_Tmu")))
        ratio = orth / raw if raw > 0 else np.nan

        out.append({
            "idx": i,
            "label": r.get("label", f"accepted_{i}"),
            "N": maybe_float(r.get("N")),
            "sigma_plus": maybe_float(r.get("sigma_plus")),
            "Xn": maybe_float(r.get("Xn")),
            "T_gamma": maybe_float(r.get("T_gamma")),
            "T_nu_e": maybe_float(r.get("T_nu_e")),
            "T_nu_x": maybe_float(r.get("T_nu_x")),
            "raw_signed": raw_signed,
            "orth_signed": orth_signed,
            "proj_Tmu": proj,
            "ratio": ratio,
            "C_norm": maybe_float(s.get("C_norm")),
            "deltaI_norm": maybe_float(s.get("deltaI_norm")),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--half-window", type=int, default=18)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    root = Path(args.json_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for case_key, case in summary.items():
        rows = load_rows(root / f"{case_key}.json")
        for sp in ("nue", "nux"):
            rr = species_rows(rows, sp)
            by_idx = {x["idx"]: x for x in rr}

            for anchor in ("onset", "peak_raw", "peak_orth", "peak_Cnorm", "peak_deltaI"):
                ent = case.get(sp, {}).get(anchor, None)
                if not ent:
                    continue
                idx = ent.get("idx", None)
                if idx is None:
                    continue
                idx = int(idx)

                lo = max(0, idx - args.half_window)
                hi = min(len(rr), idx + args.half_window + 1)
                ww = rr[lo:hi]
                xs = [z["idx"] for z in ww]

                fig, axes = plt.subplots(5, 1, figsize=(8, 12), constrained_layout=True)

                axes[0].plot(xs, [z["raw_signed"] for z in ww], label="raw")
                axes[0].plot(xs, [z["orth_signed"] for z in ww], label="orth")
                axes[0].plot(xs, [z["proj_Tmu"] for z in ww], label="proj")
                axes[0].axvline(idx, linestyle="--")
                axes[0].legend()
                axes[0].set_ylabel("qdot terms")

                axes[1].plot(xs, [z["ratio"] for z in ww])
                axes[1].axvline(idx, linestyle="--")
                axes[1].set_ylabel("|orth|/|raw|")

                axes[2].plot(xs, [z["C_norm"] for z in ww], label="C_norm")
                axes[2].plot(xs, [z["deltaI_norm"] for z in ww], label="deltaI_norm")
                axes[2].axvline(idx, linestyle="--")
                axes[2].legend()
                axes[2].set_yscale("log")
                axes[2].set_ylabel("norms")

                axes[3].plot(xs, [z["sigma_plus"] for z in ww], label="sigma_plus")
                axes[3].plot(xs, [z["Xn"] for z in ww], label="Xn")
                axes[3].axvline(idx, linestyle="--")
                axes[3].legend()
                axes[3].set_ylabel("geometry / Xn")

                axes[4].plot(xs, [z["T_gamma"] for z in ww], label="Tg")
                axes[4].plot(xs, [z["T_nu_e"] for z in ww], label="Tnu_e")
                axes[4].plot(xs, [z["T_nu_x"] for z in ww], label="Tnu_x")
                axes[4].axvline(idx, linestyle="--")
                axes[4].legend()
                axes[4].set_ylabel("temperatures")
                axes[4].set_xlabel("accepted-state idx")

                fig.suptitle(f"{case_key} | {sp} | {anchor} | idx={idx}")
                outpng = outdir / f"{case_key}_{sp}_{anchor}.png"
                fig.savefig(outpng, dpi=160)
                plt.close(fig)
                print(f"[saved] {outpng}")

if __name__ == "__main__":
    main()
