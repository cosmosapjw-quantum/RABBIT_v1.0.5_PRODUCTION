from pathlib import Path
import re
from textwrap import dedent

p = Path("scripts/plot_story_figures.py")
s = p.read_text()

# ------------------------------------------------------------------
# helper insertion: only add once
# ------------------------------------------------------------------
helper = dedent(r'''
def _rate_relative_response(delta, baseline, floor=1e-12):
    import numpy as np
    delta = np.asarray(delta, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    mask = np.abs(baseline) > floor
    out = np.full_like(delta, np.nan, dtype=float)
    out[mask] = np.abs(delta[mask]) / np.abs(baseline[mask])
    return out, mask
''').strip() + "\n\n"

if "_rate_relative_response(" not in s:
    # insert after _save definition if present, otherwise after imports
    m = re.search(r"def _save\(.*?\n(?:    .*\n)+", s, flags=re.DOTALL)
    if m:
        insert_pos = m.end()
        s = s[:insert_pos] + "\n" + helper + s[insert_pos:]
    else:
        # fallback: after imports
        m2 = re.search(r"(import .*?\n)(?:from .*?\n|import .*?\n)+", s, flags=re.DOTALL)
        if m2:
            insert_pos = m2.end()
            s = s[:insert_pos] + "\n" + helper + s[insert_pos:]
        else:
            raise SystemExit("could not find insertion point for helper")

# ------------------------------------------------------------------
# replace fig_story_distribution_weak_bridge
# ------------------------------------------------------------------
new_bridge = dedent(r'''
def fig_story_distribution_weak_bridge(sigma: float = 0.30):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # expected schema from reconstruct_bridge_cache
    rel = Path("story_bridge") / f"bridge_sigma_{sigma:.2f}.json"
    data = _loadj(rel)

    q = np.asarray(data["q_nodes"], dtype=float)
    f_ref = np.asarray(data["f0_reference"], dtype=float)
    f_bridge = np.asarray(data["f0_bridge"], dtype=float)
    rel_aug = np.asarray(data["f0_relative_shift"], dtype=float)

    N = np.asarray(data["N_values"], dtype=float)
    lam_np = np.asarray(data["lambda_np_values"], dtype=float)
    lam_pn = np.asarray(data["lambda_pn_values"], dtype=float)
    dlam_np = np.asarray(data["delta_np_values"], dtype=float)
    dlam_pn = np.asarray(data["delta_pn_values"], dtype=float)

    qcum = np.asarray(data["cumulative_q"], dtype=float)
    cnp = np.asarray(data["cumulative_np_shift"], dtype=float)
    cpn = np.asarray(data["cumulative_pn_shift"], dtype=float)

    rep_idx = int(data["representative_index"])
    rep_Tg = float(data["representative_T_gamma"])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)

    # --------------------------------------------------------------
    # (a) weak-rate relevant monopole distribution
    # truncate to support where reference is still meaningful
    # --------------------------------------------------------------
    support_mask = f_ref > 1e-8
    if not np.any(support_mask):
        support_mask = np.ones_like(f_ref, dtype=bool)

    q_sup = q[support_mask]
    f_ref_sup = f_ref[support_mask]
    f_bridge_sup = f_bridge[support_mask]

    axes[0, 0].semilogy(q_sup, f_ref_sup, color="0.45", lw=2.0, label="FLRW FD reference")
    axes[0, 0].semilogy(q_sup, f_bridge_sup, color="C3", lw=2.3, label=rf"characteristic, $\Sigma_H={sigma:.2f}$")
    axes[0, 0].set_xlabel(r"$q$")
    axes[0, 0].set_ylabel(r"$\tilde f_0(q)$")
    axes[0, 0].set_title(r"(a) Weak-rate-relevant monopole distribution")
    axes[0, 0].legend(fontsize=8)

    # --------------------------------------------------------------
    # (b) relative augmentation only on support domain
    # --------------------------------------------------------------
    rel_mask = f_ref > 1e-6
    q_rel = q[rel_mask]
    rel_aug_plot = rel_aug[rel_mask]

    axes[0, 1].plot(q_rel, rel_aug_plot, color="C0", lw=2.2)
    axes[0, 1].axhline(0.0, color="0.5", ls=":", lw=1.0)
    axes[0, 1].text(
        0.03, 0.92,
        "domain truncated where the FD reference becomes negligible",
        transform=axes[0, 1].transAxes,
        fontsize=8, color="0.35"
    )
    axes[0, 1].set_xlabel(r"$q$")
    axes[0, 1].set_ylabel(r"$[\tilde f_0-f_{\rm FD}]/f_{\rm FD}$")
    axes[0, 1].set_title(r"(b) Relative augmentation on the rate-support domain")

    # --------------------------------------------------------------
    # (c) fractional rate response along the trajectory
    # replace absolute rates by |delta lambda / lambda_FLRW|
    # --------------------------------------------------------------
    rel_np_rate, mask_np = _rate_relative_response(dlam_np, lam_np, floor=1e-20)
    rel_pn_rate, mask_pn = _rate_relative_response(dlam_pn, lam_pn, floor=1e-20)

    if np.any(mask_np):
        axes[1, 0].semilogy(N[mask_np], rel_np_rate[mask_np], color="C3", lw=2.2, label=r"$|\delta\lambda_{n\to p}|/\lambda_{n\to p}^{\rm FLRW}$")
    if np.any(mask_pn):
        axes[1, 0].semilogy(N[mask_pn], rel_pn_rate[mask_pn], color="C0", lw=2.2, label=r"$|\delta\lambda_{p\to n}|/\lambda_{p\to n}^{\rm FLRW}$")

    axes[1, 0].axvline(N[rep_idx], color="0.3", ls="--", lw=1.2)
    axes[1, 0].text(
        0.04, 0.91,
        rf"representative epoch: $T_\gamma \approx {rep_Tg:.2f}$ MeV",
        transform=axes[1, 0].transAxes,
        fontsize=8, color="0.35"
    )
    axes[1, 0].axvspan(3.0, 7.0, color="orange", alpha=0.08)
    axes[1, 0].set_xlabel(r"$N$")
    axes[1, 0].set_ylabel(r"fractional rate response")
    axes[1, 0].set_title(r"(c) Weak-rate response along the trajectory")
    axes[1, 0].legend(fontsize=8, loc="lower left")

    # --------------------------------------------------------------
    # (d) cumulative rate shift
    # --------------------------------------------------------------
    axes[1, 1].plot(qcum, cnp, color="C3", lw=2.2, label=r"$\delta\lambda_{n\to p}(q<q_{\max})$")
    axes[1, 1].plot(qcum, cpn, color="C0", lw=2.2, label=r"$\delta\lambda_{p\to n}(q<q_{\max})$")
    axes[1, 1].axhline(0.0, color="0.5", ls=":", lw=1.0)
    axes[1, 1].set_xlabel(r"$q_{\max}$")
    axes[1, 1].set_ylabel(r"cumulative fractional shift")
    axes[1, 1].set_title(rf"(d) Cumulative rate shift at $T_\gamma \approx {rep_Tg:.2f}$ MeV")
    axes[1, 1].legend(fontsize=8, loc="upper left")

    _save(fig, "fig_story_distribution_weak_bridge.png")
''').strip()

pat_bridge = re.compile(r"def fig_story_distribution_weak_bridge\(.*?(?=\n\ndef |\nif __name__ == '__main__':)", re.DOTALL)
if not pat_bridge.search(s):
    raise SystemExit("could not find fig_story_distribution_weak_bridge()")
s = pat_bridge.sub(lambda m: new_bridge, s)

# ------------------------------------------------------------------
# replace fig_story_teff_eta_energy_map
# ------------------------------------------------------------------
new_teff = dedent(r'''
def fig_story_teff_eta_energy_map():
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), constrained_layout=True)

    mu = np.linspace(-1.0, 1.0, 400)
    theta = 1.0 + 0.22 * mu + 0.18 * (mu**2 - 1.0 / 3.0)

    # (a) angular compression layer
    axes[0].plot(mu, theta, color="C0", lw=2.2)
    axes[0].axhline(1.0, color="0.5", ls=":", lw=1.0)
    axes[0].set_xlabel(r"$\mu$")
    axes[0].set_ylabel(r"$\Theta(\mu)$")
    axes[0].set_title(r"(a) Angular compression layer $T_{A_\ell}$")

    # (b) illustrative energy-shape spectrum
    n = np.arange(0, 7)
    coeff = np.array([0.0, 0.0, 0.14, 0.07, 0.025, 0.010, 0.004])
    axes[1].bar(n, np.abs(coeff), color="C3")
    axes[1].set_xlabel(r"energy mode $n$")
    axes[1].set_ylabel(r"$|c_n|$")
    axes[1].set_title(r"(b) Energy-shape sector $c_{nA_\ell}$")
    axes[1].text(
        0.04, 0.92,
        "illustrative spectrum for the augmented representation",
        transform=axes[1].transAxes,
        fontsize=8, color="0.35"
    )

    # (c) schematic map only
    axes[2].axis("off")
    txt = (
        "schematic only\n\n"
        "M_ang(T_A_l): angular compression layer\n"
        "+ eta_A_l: direction-dependent degeneracy\n"
        "+ c_nA_l: energy-shape modes\n\n"
        "small c_{n>=2}: Teff-only interpretation adequate\n"
        "nonzero eta_A_l: number-energy decoupling active\n"
        "large c_{n>=2}: spectral distortion / upgrade needed"
    )
    axes[2].text(
        0.03, 0.95, txt,
        va="top", ha="left", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.4", alpha=0.95)
    )
    axes[2].set_title("(c) Augmented representation map (schematic)")

    _save(fig, "fig_story_teff_eta_energy_map.png")
''').strip()

pat_teff = re.compile(r"def fig_story_teff_eta_energy_map\(.*?(?=\n\ndef |\nif __name__ == '__main__':)", re.DOTALL)
if not pat_teff.search(s):
    raise SystemExit("could not find fig_story_teff_eta_energy_map()")
s = pat_teff.sub(lambda m: new_teff, s)

# ------------------------------------------------------------------
# optional: make causal chain title shorter/cleaner if old version present
# ------------------------------------------------------------------
s = s.replace(
    "Scipy Bianchi I production story: geometry -> transport -> weak rates -> observables -> constraint",
    "Scipy Bianchi I production story"
)

p.write_text(s)
print("patched:", p)
