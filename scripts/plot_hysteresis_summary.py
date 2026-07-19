#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_xy(rows, key):
    x = np.array([r["relax"] for r in rows], dtype=float)
    y = np.array([np.nan if r.get(key) is None else r.get(key) for r in rows], dtype=float)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    p = Path(args.input).resolve()
    data = json.loads(p.read_text())

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for rows, label in [(data["grid_forward"], "forward"), (data["grid_reverse"], "reverse")]:
        x, y = load_xy(rows, "sigma_plus")
        ax[0, 0].plot(x, y, marker="o", label=label)
        x, y = load_xy(rows, "pi_plus")
        ax[0, 1].plot(x, y, marker="o", label=label)
        x, y = load_xy(rows, "Xn")
        ax[1, 0].plot(x, y, marker="o", label=label)

    ax[0, 0].set_title("sigma_plus vs relax")
    ax[0, 1].set_title("pi_plus vs relax")
    ax[1, 0].set_title("Xn vs relax")
    ax[1, 1].axis("off")
    for a in ax.ravel()[:3]:
        a.legend()
        a.grid(True, alpha=0.3)

    out = p.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
