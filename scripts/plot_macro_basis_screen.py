#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metric(summary, key, outpath):
    names = [r["basis"] for r in summary]
    vals = [float(r[key]) for r in summary]
    x = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    summary = data["summary"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for key in ["mean_cos", "worst_cos", "mean_resid", "sign_rate"]:
        plot_metric(summary, key, outdir / f"{key}.png")

    print(f"[saved] {outdir}")


if __name__ == "__main__":
    main()
