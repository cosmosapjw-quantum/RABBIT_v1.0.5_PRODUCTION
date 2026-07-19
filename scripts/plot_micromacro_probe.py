#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--species", default="nue")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.probe).read_text())
    rows = data["rows"]
    sp = args.species

    N = np.array([r["N"] for r in rows], dtype=float)
    sigma = np.array([r["sigma_plus"] for r in rows], dtype=float)

    raw = np.array([r["species"][sp]["raw_qdot"] for r in rows], dtype=float)
    projT = np.array([r["species"][sp]["proj_qdot_T"] for r in rows], dtype=float)
    projTmu = np.array([r["species"][sp]["proj_qdot_Tmu"] for r in rows], dtype=float)
    orthT = np.array([r["species"][sp]["orth_qdot_T"] for r in rows], dtype=float)
    orthTmu = np.array([r["species"][sp]["orth_qdot_Tmu"] for r in rows], dtype=float)
    tail_raw = np.array([r["species"][sp]["tail_frac_raw_last3"] for r in rows], dtype=float)
    tail_orth = np.array([r["species"][sp]["tail_frac_orthTmu_last3"] for r in rows], dtype=float)
    cnorm = np.array([r["species"][sp]["C_norm"] for r in rows], dtype=float)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        path = outdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[saved] {path}")

    fig = plt.figure(figsize=(7,4.5))
    plt.plot(N, raw, label="raw qdot")
    plt.plot(N, projT, label="projT qdot")
    plt.plot(N, projTmu, label="projTmu qdot")
    plt.plot(N, orthTmu, label="orth(T,mu) qdot")
    plt.yscale("symlog", linthresh=1e-12)
    plt.xlabel("N")
    plt.ylabel("qdot-like moment")
    plt.legend()
    save(fig, f"{sp}_qdot_decomposition.png")

    fig = plt.figure(figsize=(7,4.5))
    plt.plot(N, tail_raw, label="raw tail frac last3")
    plt.plot(N, tail_orth, label="orth(T,mu) tail frac last3")
    plt.xlabel("N")
    plt.ylabel("tail fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    save(fig, f"{sp}_tail_fraction.png")

    fig = plt.figure(figsize=(7,4.5))
    plt.plot(N, sigma, label="sigma_plus")
    plt.plot(N, cnorm, label="||C||")
    plt.yscale("symlog", linthresh=1e-12)
    plt.xlabel("N")
    plt.ylabel("value")
    plt.legend()
    save(fig, f"{sp}_sigma_vs_Cnorm.png")


if __name__ == "__main__":
    main()
