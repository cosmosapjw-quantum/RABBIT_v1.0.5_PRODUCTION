#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def extract_rows(rows, sp):
    out = []
    for i, r in enumerate(rows):
        s = r["species"][sp]
        out.append({
            "idx": i,
            "N": float(r["N"]),
            "sigma_plus": float(r["sigma_plus"]),
            "Xn": float(r["Xn"]),
            "T_gamma": float(r["T_gamma"]),
            "T_nu_e": float(r["T_nu_e"]),
            "T_nu_x": float(r["T_nu_x"]),
            "raw": float(s["raw_qdot"]),
            "proj": float(s["proj_qdot_Tmu"]),
            "orth": float(s["orth_qdot_Tmu"]),
            "C_norm": float(s["C_norm"]),
            "deltaI_norm": float(s["deltaI_norm"]),
            "flip_raw": int(s["signflip_index_raw"]),
            "flip_orth": int(s["signflip_index_orthTmu"]),
        })
    return out

def window(rows, center, half):
    lo = max(0, center - half)
    hi = min(len(rows), center + half + 1)
    return rows[lo:hi]

def make_plot(rows, center_idx, half, title, out_png):
    w = window(rows, center_idx, half)
    N = [r["N"] for r in w]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(N, [r["raw"] for r in w], label="raw_qdot")
    ax.plot(N, [r["proj"] for r in w], label="proj_qdot_Tmu")
    ax.plot(N, [r["orth"] for r in w], label="orth_qdot_Tmu")
    ax.set_xlabel("N")
    ax.set_ylabel("qdot")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(N, [r["sigma_plus"] for r in w], label="sigma_plus")
    ax.plot(N, [r["Xn"] for r in w], label="Xn")
    ax.plot(N, [r["T_nu_e"] - r["T_nu_x"] for r in w], label="T_nu_e - T_nu_x")
    ax.set_xlabel("N")
    ax.set_title(title + " : geometry / thermo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png.with_name(out_png.stem + "_geom.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(N, [r["C_norm"] for r in w], label="C_norm")
    ax.plot(N, [r["deltaI_norm"] for r in w], label="deltaI_norm")
    ax.set_xlabel("N")
    ax.set_title(title + " : bridge amplitudes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png.with_name(out_png.stem + "_bridge.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(N, [r["flip_raw"] for r in w], label="signflip_raw")
    ax.plot(N, [r["flip_orth"] for r in w], label="signflip_orthTmu")
    ax.set_xlabel("N")
    ax.set_title(title + " : signflip indices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png.with_name(out_png.stem + "_flip.png"))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--json-root", required=True,
                    help="Directory containing micromacro_*.json")
    ap.add_argument("--half-window", type=int, default=15)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for case_key, case in summary.items():
        jpath = Path(args.json_root) / f"{case_key}.json"
        data = json.loads(jpath.read_text())
        rows = data["rows"]

        for sp in ["nue", "nux"]:
            rr = extract_rows(rows, sp)

            for tag in ["onset", "peak_raw", "peak_orth", "peak_Cnorm", "peak_deltaI"]:
                info = case[sp][tag]
                if info is None:
                    continue
                idx = int(info["idx"])
                title = f"{case_key} / {sp} / {tag}"
                out_png = outdir / f"{case_key}_{sp}_{tag}.png"
                make_plot(rr, idx, args.half_window, title, out_png)

if __name__ == "__main__":
    main()
