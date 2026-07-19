#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# utilities
# ----------------------------
def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def to_float(x: Any):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def load_json(path: Path):
    return json.loads(path.read_text())


def normalize_sigma_key(k):
    try:
        return float(k)
    except Exception:
        return k


# ----------------------------
# parsers
# ----------------------------
def parse_relax_branch_scan_rows(data, source_name="rows"):
    """
    Expected shape:
    [
      {
        "Sigma_H": 0.5,
        "relax": 0.1,
        "shared": {...},
        "species": {...}
      },
      ...
    ]
    """
    rows = []
    if not isinstance(data, list):
        return rows

    for r in data:
        sigma = to_float(r.get("Sigma_H"))
        relax = to_float(r.get("relax"))
        if sigma is None or relax is None:
            continue

        for arm in ["shared", "species"]:
            block = r.get(arm, {})
            if not isinstance(block, dict):
                continue
            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "arm": arm,
                "source": source_name,
                "sigma_plus": to_float(block.get("sigma_plus")),
                "pi_plus": to_float(block.get("pi_plus")),
                "Xn": to_float(block.get("Xn")),
                "lambda_np": to_float(block.get("lambda_np")),
            })
    return rows


def parse_hysteresis_json(data, source_name="hysteresis"):
    """
    Expected shape:
    {
      "target": "shared"|"species",
      "sigma": 0.5,
      "grid_forward": [...],
      "grid_reverse": [...]
    }
    """
    rows = []
    if not isinstance(data, dict):
        return rows

    target = data.get("target", "unknown")
    sigma = to_float(data.get("sigma"))
    if sigma is None:
        return rows

    for key in ["grid_forward", "grid_reverse"]:
        seq = data.get(key, [])
        if not isinstance(seq, list):
            continue
        sweep = "forward" if key == "grid_forward" else "reverse"
        for r in seq:
            relax = to_float(r.get("relax"))
            if relax is None:
                continue
            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "arm": f"{target}_{sweep}",
                "source": source_name,
                "sigma_plus": to_float(r.get("sigma_plus")),
                "pi_plus": to_float(r.get("pi_plus")),
                "Xn": to_float(r.get("Xn")),
                "lambda_np": to_float(r.get("lambda_np")),
            })
    return rows


def parse_compare_summary_json(data, source_name="compare_summary"):
    """
    Expected shape:
    {
      "0.3": [
        {
          "relax": 0.1,
          "shared_Xn": ...,
          "species_Xn": ...,
          ...
        },
        ...
      ],
      ...
    }
    """
    rows = []
    if not isinstance(data, dict):
        return rows

    # Heuristic: keys are sigma strings and values are lists
    if not all(isinstance(v, list) for v in data.values()):
        return rows

    for sigma_key, seq in data.items():
        sigma = to_float(normalize_sigma_key(sigma_key))
        if sigma is None:
            continue
        for r in seq:
            relax = to_float(r.get("relax"))
            if relax is None:
                continue

            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "arm": "shared",
                "source": source_name,
                "sigma_plus": to_float(r.get("shared_sigma_plus")),
                "pi_plus": to_float(r.get("shared_pi_plus")),
                "Xn": to_float(r.get("shared_Xn")),
                "lambda_np": to_float(r.get("shared_lambda_np")),
            })
            rows.append({
                "Sigma_H": sigma,
                "relax": relax,
                "arm": "species",
                "source": source_name,
                "sigma_plus": to_float(r.get("species_sigma_plus")),
                "pi_plus": to_float(r.get("species_pi_plus_total")),
                "Xn": to_float(r.get("species_Xn")),
                "lambda_np": to_float(r.get("species_lambda_np")),
            })
    return rows


def auto_parse(path: Path):
    data = load_json(path)
    parsers = [
        parse_relax_branch_scan_rows,
        parse_hysteresis_json,
        parse_compare_summary_json,
    ]
    all_rows = []
    for p in parsers:
        rows = p(data, source_name=path.name)
        if rows:
            all_rows.extend(rows)
    return all_rows


# ----------------------------
# plotting
# ----------------------------
ARM_STYLE = {
    "shared": dict(marker="o", linestyle="-", linewidth=1.6),
    "species": dict(marker="s", linestyle="-", linewidth=1.6),
    "shared_forward": dict(marker="o", linestyle="-", linewidth=1.6),
    "shared_reverse": dict(marker="o", linestyle="--", linewidth=1.4),
    "species_forward": dict(marker="s", linestyle="-", linewidth=1.6),
    "species_reverse": dict(marker="s", linestyle="--", linewidth=1.4),
}


def plot_panel(ax, rows, quantity, title):
    sigmas = sorted(set(r["Sigma_H"] for r in rows))
    cmap = plt.get_cmap("tab10")

    for i, sigma in enumerate(sigmas):
        sigma_rows = [r for r in rows if r["Sigma_H"] == sigma]
        arms = sorted(set(r["arm"] for r in sigma_rows))

        for arm in arms:
            rs = [r for r in sigma_rows if r["arm"] == arm and r.get(quantity) is not None]
            if not rs:
                continue
            rs = sorted(rs, key=lambda z: z["relax"])
            x = [r["relax"] for r in rs]
            y = [r[quantity] for r in rs]
            style = ARM_STYLE.get(arm, dict(marker=".", linestyle="-", linewidth=1.2))
            label = f"Σ={sigma:g} | {arm}"
            ax.plot(x, y, color=cmap(i % 10), markersize=5, alpha=0.95, label=label, **style)

    ax.set_title(title)
    ax.set_xlabel("relax")
    ax.grid(True, alpha=0.3)


def maybe_add_zero_line(ax, quantity):
    if quantity in {"sigma_plus", "pi_plus"}:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Existing json files. Can mix rows/hysteresis/compare-summary formats.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output png path",
    )
    ap.add_argument(
        "--dump-merged",
        default=None,
        help="Optional path to save merged normalized rows as json",
    )
    args = ap.parse_args()

    merged = []
    for p in args.inputs:
        path = Path(p).resolve()
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        rows = auto_parse(path)
        if not rows:
            print(f"[warn] unrecognized format: {path}")
            continue
        merged.extend(rows)

    if not merged:
        raise SystemExit("[FAIL] no recognizable rows found in provided jsons")

    # deduplicate exact duplicates
    uniq = {}
    for r in merged:
        key = (
            r["Sigma_H"], r["relax"], r["arm"], r["source"],
            r.get("sigma_plus"), r.get("pi_plus"), r.get("Xn"), r.get("lambda_np")
        )
        uniq[key] = r
    rows = list(uniq.values())

    if args.dump_merged:
        out_json = Path(args.dump_merged).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2))
        print(f"[saved merged rows] {out_json}")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    panels = [
        ("sigma_plus", "bifurcation: sigma_plus"),
        ("pi_plus", "bifurcation: pi_plus"),
        ("Xn", "bifurcation: Xn"),
        ("lambda_np", "bifurcation: lambda_np"),
    ]

    for ax, (q, title) in zip(axes.ravel(), panels):
        plot_panel(ax, rows, q, title)
        maybe_add_zero_line(ax, q)
        ax.set_ylabel(q)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # unique legend
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_handles.append(h)
            new_labels.append(l)

    fig.legend(new_handles, new_labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"[saved figure] {out}")
    print(f"[rows parsed] {len(rows)}")


if __name__ == "__main__":
    main()
