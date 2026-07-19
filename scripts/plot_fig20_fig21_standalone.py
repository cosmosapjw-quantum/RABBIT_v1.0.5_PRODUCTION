from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "production_report_cache"
ANCHOR_NEFF = 3.044
SIGMA_LEGACY_GUARD_MAX = 0.75


def choose_figure_dir() -> Path:
    candidates = [
        ROOT / "docs" / "RABBIT_report" / "figures",
        ROOT / "figures",
    ]
    for d in candidates:
        if d.exists():
            return d
    d = ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


FIG_DIR = choose_figure_dir()


def set_output_dir(path: str | Path) -> Path:
    global FIG_DIR
    FIG_DIR = Path(path).expanduser().resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def set_cache_dir(path: str | Path) -> Path:
    global CACHE
    CACHE = Path(path).expanduser().resolve()
    return CACHE


def save(fig, name: str):
    out = FIG_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing cache file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plain_ticks(ax):
    fmtx = mticker.ScalarFormatter(useMathText=True)
    fmtx.set_scientific(False)
    fmtx.set_useOffset(False)
    fmty = mticker.ScalarFormatter(useMathText=True)
    fmty.set_scientific(False)
    fmty.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmtx)
    ax.yaxis.set_major_formatter(fmty)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)


def scale_for_plot(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0.0:
        return 1.0, ""
    exp = int(math.floor(math.log10(vmax)))
    if -2 <= exp <= 2:
        return 1.0, ""
    return 10.0 ** exp, rf" [$\times 10^{{{exp}}}$]"


def profile_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(1, len(x)):
        y0 = y[i - 1] - threshold
        y1 = y[i] - threshold
        if y0 == 0.0:
            return float(x[i - 1])
        if y0 * y1 <= 0.0 and y1 != y0:
            t = -y0 / (y1 - y0)
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


def discover_fig20():
    typei_dir = CACHE / "typeI"
    if not typei_dir.exists():
        raise RuntimeError(f"missing cache dir: {typei_dir}")

    pat = re.compile(r"fig20_(.+?)_sigma_([0-9.]+)\.json$")
    data = {"linearized": [], "characteristic": []}

    for p in sorted(typei_dir.glob("fig20_*.json")):
        m = pat.match(p.name)
        if not m:
            continue
        raw_model = m.group(1)
        sigma = float(m.group(2))
        payload = load_json(p)
        if "N_eff_measured" not in payload:
            continue

        # model classification
        name = raw_model.lower()
        if "teff" in name:
            continue
        if "linear" in name or "pstf" in name:
            model = "linearized"
        elif "characteristic" in name:
            model = "characteristic"
        else:
            continue

        data[model].append((sigma, float(payload["N_eff_measured"])))

    for k in data:
        data[k] = sorted(data[k], key=lambda x: x[0])

    if not data["linearized"] and not data["characteristic"]:
        raise RuntimeError("no usable fig20 cache files found")

    return data


def fig20_from_cache():
    data = discover_fig20()

    model_meta = [
        ("linearized", "linearized PSTF", "C0"),
        ("characteristic", "characteristic", "C3"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    drift_dict = {}
    all_delta = []
    all_drift = []

    for key, label, color in model_meta:
        pairs = data.get(key, [])
        if not pairs:
            continue

        sigmas = np.array([x[0] for x in pairs], dtype=float)
        neff = np.array([x[1] for x in pairs], dtype=float)

        delta = neff - ANCHOR_NEFF
        drift = delta - delta[0]

        drift_dict[key] = (sigmas, drift)
        all_delta.append(delta)
        all_drift.append(drift)

    # panel (a): anchor-relative shift
    if all_delta:
        delta_all = np.concatenate(all_delta)
    else:
        delta_all = np.array([0.0])

    scale_a, suffix_a = scale_for_plot(delta_all)

    for key, label, color in model_meta:
        if key not in drift_dict:
            continue
        pairs = data[key]
        sigmas = np.array([x[0] for x in pairs], dtype=float)
        neff = np.array([x[1] for x in pairs], dtype=float)
        delta = neff - ANCHOR_NEFF
        axes[0].plot(sigmas, delta / scale_a, marker="o", lw=2.2, ms=5.5, color=color, label=label)

    axes[0].axhline(0.0, color="0.5", ls=":", lw=1.0)
    axes[0].set_xlabel(r"$\Sigma_H$")
    axes[0].set_ylabel(r"$\Delta N_{\rm eff}\equiv N_{\rm eff}^{\rm measured}-3.044$" + suffix_a)
    axes[0].set_title(r"(a) Anchor-relative shift in $N_{\rm eff}$")
    axes[0].legend(fontsize=8)
    plain_ticks(axes[0])

    # panel (b): drift from Sigma=0 baseline
    if all_drift:
        drift_all = np.concatenate(all_drift)
    else:
        drift_all = np.array([0.0])

    scale_b, suffix_b = scale_for_plot(drift_all)

    for key, label, color in model_meta:
        if key not in drift_dict:
            continue
        sigmas, drift = drift_dict[key]
        axes[1].plot(sigmas, drift / scale_b, marker="o", lw=2.2, ms=5.5, color=color, label=label)

    axes[1].axhline(0.0, color="0.5", ls=":", lw=1.0)
    axes[1].set_xlabel(r"$\Sigma_H$")
    axes[1].set_ylabel(
        r"$\delta\Delta N_{\rm eff}\equiv \Delta N_{\rm eff}(\Sigma_H)-\Delta N_{\rm eff}(0)$" + suffix_b
    )
    axes[1].set_title(r"(b) Drift relative to the $\Sigma_H=0$ baseline")
    axes[1].legend(fontsize=8)
    plain_ticks(axes[1])

    save(fig, "fig20.png")


def discover_chi2_grid():
    chi2_dir = CACHE / "chi2"
    if not chi2_dir.exists():
        raise RuntimeError(f"missing cache dir: {chi2_dir}")

    pat = re.compile(r"eta_([0-9.]+)_sigma_([0-9.]+)\.json$")
    eta_vals = set()
    sigma_vals = set()
    table = {}

    for p in sorted(chi2_dir.glob("eta_*_sigma_*.json")):
        m = pat.match(p.name)
        if not m:
            continue
        eta = float(m.group(1))
        sigma = float(m.group(2))
        payload = load_json(p)
        if "chi2" not in payload:
            continue
        chi2 = float(payload["chi2"])
        eta_vals.add(eta)
        sigma_vals.add(sigma)
        table[(eta, sigma)] = chi2

    if not table:
        raise RuntimeError("no chi2 cache files found")

    etas = np.array(sorted(eta_vals), dtype=float)
    sigmas = np.array(sorted(sigma_vals), dtype=float)

    chi2 = np.full((len(etas), len(sigmas)), np.nan, dtype=float)
    for i, eta in enumerate(etas):
        for j, sigma in enumerate(sigmas):
            if (eta, sigma) not in table:
                raise RuntimeError(f"incomplete chi2 grid: missing eta={eta}, sigma={sigma}")
            chi2[i, j] = table[(eta, sigma)]

    return etas, sigmas, chi2


def fig21_from_cache():
    etas, sigmas, chi2 = discover_chi2_grid()

    chi2_min = np.nanmin(chi2)
    dchi2 = chi2 - chi2_min

    ibest, jbest = np.unravel_index(np.nanargmin(chi2), chi2.shape)
    eta_best = etas[ibest]
    sigma_best = sigmas[jbest]

    # FLRW reference point: best eta on sigma=0 slice
    j0 = int(np.argmin(np.abs(sigmas - 0.0)))
    i0 = int(np.nanargmin(chi2[:, j0]))
    eta_flrw = etas[i0]
    sigma_flrw = sigmas[j0]

    # profile over eta for each sigma
    dchi2_prof_sigma = np.nanmin(dchi2, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    E, S = np.meshgrid(etas, sigmas, indexing="ij")

    vmax = max(12.0, float(np.nanpercentile(dchi2, 95)))
    levels_fill = np.linspace(0.0, vmax, 25)

    cf = axes[0].contourf(E, S, dchi2, levels=levels_fill, cmap="viridis", extend="max")

    contour_levels = [2.30, 6.18, 11.83]
    valid_levels = [lv for lv in contour_levels if lv < np.nanmax(dchi2)]
    if valid_levels:
        c = axes[0].contour(E, S, dchi2, levels=valid_levels, colors="white", linewidths=1.5)
        axes[0].clabel(c, inline=True, fontsize=8, fmt=lambda x: rf"$\Delta\chi^2={x:.2f}$")

    axes[0].plot(
        eta_best, sigma_best,
        marker="*", color="red", ms=14, mec="white", mew=0.8,
        label="global best fit"
    )
    axes[0].plot(
        eta_flrw, sigma_flrw,
        marker="+", color="red", ms=14, mew=2.2,
        label=r"best FLRW slice ($\Sigma_H=0$)"
    )
    axes[0].axhline(
        SIGMA_LEGACY_GUARD_MAX,
        color="0.35",
        ls="--",
        lw=1.0,
        label="legacy runtime guard (not validated)",
    )
    axes[0].text(
        0.98, 0.96,
        r"white contours: $\Delta\chi^2=2.30,6.18,11.83$",
        transform=axes[0].transAxes, fontsize=7.5, color="white",
        ha="right", va="top",
    )

    axes[0].set_xlabel(r"$\eta_{10}$")
    axes[0].set_ylabel(r"$\Sigma_H$")
    axes[0].set_title(r"(a) $\Delta\chi^2(\eta_{10},\Sigma_H)$")
    axes[0].legend(fontsize=8, loc="upper left")
    plain_ticks(axes[0])

    cbar = fig.colorbar(cf, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta\chi^2$")

    axes[1].plot(sigmas, dchi2_prof_sigma, "C0-o", lw=2.3, ms=5.8)
    for y, ls, lab in [
        (1.00, ":", "68% (1 dof)"),
        (3.84, "--", "95% (1 dof)"),
        (6.63, "-.", "99% (1 dof)"),
    ]:
        axes[1].axhline(y, color="0.4", ls=ls, lw=1.1, label=lab)
    axes[1].axvline(
        SIGMA_LEGACY_GUARD_MAX,
        color="0.35",
        ls="--",
        lw=1.0,
        label="legacy runtime guard (not validated)",
    )

    sigma95 = profile_crossing(sigmas, dchi2_prof_sigma, 3.84)
    if sigma95 is not None:
        axes[1].axvline(
            sigma95, color="C3", ls=":", lw=1.35,
            label=rf"historical diagnostic crossing $\Sigma_H\simeq {sigma95:.2f}$",
        )
        axes[1].annotate(
            rf"$\Sigma_H\simeq {sigma95:.2f}$",
            xy=(sigma95, 3.84), xytext=(-62, 18),
            textcoords="offset points", fontsize=8, color="C3",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"),
        )

    axes[1].plot(
        sigma_best,
        dchi2_prof_sigma[np.argmin(np.abs(sigmas - sigma_best))],
        marker="*", color="red", ms=12, mec="white", mew=0.8
    )

    axes[1].set_xlabel(r"$\Sigma_H$")
    axes[1].set_ylabel(r"$\Delta\chi^2_{\rm prof}(\Sigma_H)$")
    axes[1].set_title(r"(b) Profile likelihood in $\Sigma_H$")
    axes[1].legend(fontsize=8, loc="upper left")
    plain_ticks(axes[1])

    save(fig, "fig21.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot report Fig. 20 and Fig. 21 from cached JSON.")
    parser.add_argument("--outdir", default=str(FIG_DIR), help="Figure output directory.")
    parser.add_argument("--cache-dir", default=str(CACHE), help="Cache directory containing typeI/ and chi2/.")
    args = parser.parse_args(argv)
    set_output_dir(args.outdir)
    set_cache_dir(args.cache_dir)
    fig20_from_cache()
    fig21_from_cache()
    print(f"saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
