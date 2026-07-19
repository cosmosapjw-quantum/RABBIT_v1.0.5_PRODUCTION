#!/usr/bin/env python3
"""Nonlinear Boltzmann ablation study for LRS Bianchi Type I.

Disentangles three independent approximations:
  A1. ℓ-truncation (ell_max = 2, 4, 6, 8)
  A2. Ψ-linearization (linear vs nonlinear)
  A3. q-dependent source (q-dep vs q-indep)

Runs a matrix of configurations at each Σ value and compares:
  - Y_p, D/H
  - max|Ψ₀|, max|Ψ₂|, max|Ψ₄|, ...
  - ΔY_p relative to baseline

Output:
  - NONLINEAR_ABLATION_RESULTS.json
  - NONLINEAR_ABLATION_REPORT.md
"""
import sys, os, time, json, warnings
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_laguerre

# RABBIT imports
from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.geometry.constraints import friedmann_residual_typeI
from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1
from rabbit.thermo.eos_photon_electron import rho_plasma, _RHO_GAMMA_PREFACTOR
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.network.abundances_standard import (
    abundance_rhs_phase1, abundance_rhs_phase2, phase1_to_phase2,
    N_SPECIES,
)
from rabbit.transport.nonlinear_boltzmann import (
    nonlinear_transport_rhs, psi0_max_amplitude, psi2_max_amplitude,
    psi_ell_amplitudes, _legendre_matrices,
)

# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

_TAU_N = 879.6
_ETA = 6.104e-10
_N_EFF = 3.044
_F_NU = 0.40520
_MEV_TO_S = 1.0 / 6.58212e-22
_G_N = 6.70883e-45  # G_N in MeV^-2
_RHO_GAMMA = _RHO_GAMMA_PREFACTOR

B_QUAD = 8.0 / 15.0
A_FEEDBACK = 6.0


def _hubble(T_gamma, T_nu, N_eff, Sigma_sq):
    rho_g = _RHO_GAMMA * T_gamma**4
    rho_nu = N_eff * (7.0 / 8.0) * _RHO_GAMMA * T_nu**4
    rho_total = rho_g + rho_nu
    H_sq = 8.0 * np.pi * _G_N / 3.0 * rho_total / max(1.0 - Sigma_sq, 1e-10)
    return np.sqrt(max(H_sq, 0.0))


# ═══════════════════════════════════════════════════════════════════════
# §2. Momentum grid
# ═══════════════════════════════════════════════════════════════════════

def make_q_grid(N_q):
    nodes, weights = roots_laguerre(N_q)
    return nodes.astype(np.float64), weights.astype(np.float64)


def fermi_dirac(q):
    return 1.0 / (np.exp(q) + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §3. Anisotropic stress from modes
# ═══════════════════════════════════════════════════════════════════════

def compute_Pi_from_modes(psi_modes, q_nodes, q_weights, f_nu, f0_eq):
    """Π₊ = A × f_ν × π̃, where π̃ = species-averaged reduced quadrupole."""
    n_species = psi_modes.shape[0]
    n_ell = psi_modes.shape[1]
    if n_ell < 2:
        return 0.0

    # Energy weight: w_i = f₀(q_i) × q_i⁴ × w_lag_i × exp(q_i)
    # (exp(q) cancels the Laguerre weight)
    energy_w = f0_eq * q_nodes**4 * q_weights * np.exp(q_nodes)
    norm = np.sum(energy_w)
    if norm < 1e-30:
        return 0.0

    pi_tilde = 0.0
    for s in range(n_species):
        psi2_q = psi_modes[s, 1, :]  # ℓ=2 mode
        pi_tilde += np.sum(psi2_q * energy_w) / norm
    pi_tilde /= n_species

    return A_FEEDBACK * f_nu * pi_tilde


# ═══════════════════════════════════════════════════════════════════════
# §4. Coupled RHS with configurable transport
# ═══════════════════════════════════════════════════════════════════════

def coupled_rhs(N, y, *,
                q_nodes, q_weights, f0_eq, n_species_transport,
                ell_max, N_mu,
                tau_n, eta, N_eff, f_nu,
                n_reactions, phase,
                linearized, q_independent_source):
    """Full coupled RHS with configurable transport mode."""

    ell_list = list(range(0, ell_max + 1, 2))
    n_ell = len(ell_list)
    N_q = len(q_nodes)
    n_transport = n_species_transport * n_ell * N_q

    # Layout
    I_SP = 0
    I_SM = 1
    I_HIER = 2
    i_hier_end = I_HIER + n_transport
    i_tg = i_hier_end
    n_species_net = 2 if phase == 1 else N_SPECIES
    i_net = i_tg + 1
    n_total = i_net + n_species_net

    Sigma_plus = y[I_SP]
    Sigma_minus = y[I_SM]
    hier_flat = y[I_HIER:i_hier_end]
    T_gamma = y[i_tg]
    X = y[i_net:i_net + n_species_net]

    if T_gamma < 1e-6:
        return np.zeros(n_total)

    Sigma_sq = Sigma_plus**2 + Sigma_minus**2

    # Reshape hierarchy
    psi_modes = hier_flat.reshape(n_species_transport, n_ell, N_q)

    # ── Transport → π → Geometry ──
    Pi_plus = compute_Pi_from_modes(psi_modes, q_nodes, q_weights, f_nu, f0_eq)
    Omega = max(0.0, 1.0 - Sigma_sq)

    dSp, dSm = compute_typeI_geometry_rhs(Sigma_plus, Sigma_minus,
                                           Pi_plus, 0.0, Omega)

    # ── Transport RHS (nonlinear or linear) ──
    dpsi = nonlinear_transport_rhs(
        Sigma_plus, psi_modes, q_nodes, f0_eq,
        ell_max=ell_max, N_mu=N_mu,
        linearized=linearized,
        q_independent_source=q_independent_source,
    )

    # ── Thermo ──
    dTg = dT_gamma_dN_tier1(T_gamma)
    T_nu = T_nu_from_T_gamma_tier1(T_gamma)
    H = _hubble(T_gamma, T_nu, N_eff, Sigma_sq) * _MEV_TO_S

    # ── Weak rates from monopole ──
    # Extract f_total = f₀(1 + Ψ₀) for species 0,1
    psi0_nue = psi_modes[0, 0, :]
    psi0_nuebar = psi_modes[1, 0, :]
    f_nue = f0_eq * (1.0 + psi0_nue)
    f_nuebar = f0_eq * (1.0 + psi0_nuebar)

    # Positivity guard (physical: f must be in [0,1] for fermions)
    f_nue = np.clip(f_nue, 0.0, 1.0)
    f_nuebar = np.clip(f_nuebar, 0.0, 1.0)

    weak = compute_live_weak_rates(
        f_nue, f_nuebar, q_nodes,
        T_gamma, T_nu, tau_n,
        compute_iso_reference=False, correction_level=0)
    lnp = weak.lambda_np
    lpn = weak.lambda_pn

    # ── Network ──
    if phase == 1:
        dX = np.zeros(n_species_net)
        dX[0] = abundance_rhs_phase1(X[0], lnp, lpn) / max(H, 1e-100)
        dX[1] = -dX[0]
    else:
        dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn,
                                   n_reactions=n_reactions) / max(H, 1e-100)

    # ── Pack ──
    dy = np.zeros(n_total)
    dy[I_SP] = dSp
    dy[I_SM] = dSm
    dy[I_HIER:i_hier_end] = dpsi.ravel()
    dy[i_tg] = dTg
    dy[i_net:i_net + n_species_net] = dX

    return dy


# ═══════════════════════════════════════════════════════════════════════
# §5. Driver
# ═══════════════════════════════════════════════════════════════════════

def run_bbn(Sigma_H, ell_max=2, N_q=20, N_mu=0,
            linearized=False, q_independent_source=False,
            T_start=10.0, T_handoff=0.08, T_end=0.005,
            n_reactions=12, rtol=1e-8, atol=1e-10):
    """Run full coupled BBN with configurable transport.

    Returns dict with observables and diagnostics.
    """
    q_nodes, q_weights = make_q_grid(N_q)
    f0_eq = fermi_dirac(q_nodes)
    n_species_transport = 6

    ell_list = list(range(0, ell_max + 1, 2))
    n_ell = len(ell_list)
    n_transport = n_species_transport * n_ell * N_q

    if N_mu == 0:
        N_mu = max(2 * ell_max + 4, 16)

    # ── Initial conditions ──
    # Phase 1: 2 network species (Xn, Xp)
    n_net_p1 = 2
    n_total_p1 = 2 + n_transport + 1 + n_net_p1
    i_tg = 2 + n_transport
    i_net = i_tg + 1

    # Equilibrium Xn
    T_nu_init = T_nu_from_T_gamma_tier1(T_start)
    from rabbit.weak.live_rates import compute_live_weak_rates as _clwr
    f_init = f0_eq.copy()
    weak0 = _clwr(f_init, f_init, q_nodes, T_start, T_nu_init, _TAU_N,
                   compute_iso_reference=False, correction_level=0)
    Xn_eq = float(weak0.lambda_pn / max(weak0.lambda_np + weak0.lambda_pn, 1e-100))

    y0 = np.zeros(n_total_p1)
    y0[0] = Sigma_H  # Σ₊
    y0[i_tg] = T_start
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    common_kw = dict(
        q_nodes=q_nodes, q_weights=q_weights, f0_eq=f0_eq,
        n_species_transport=n_species_transport,
        ell_max=ell_max, N_mu=N_mu,
        tau_n=_TAU_N, eta=_ETA, N_eff=_N_EFF, f_nu=_F_NU,
        n_reactions=n_reactions,
        linearized=linearized,
        q_independent_source=q_independent_source,
    )

    # ── Phase 1 ──
    def rhs_p1(N, y):
        return coupled_rhs(N, y, phase=1, **common_kw)

    def event_p1(N, y):
        return y[i_tg] - T_handoff
    event_p1.terminal = True
    event_p1.direction = -1

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol1 = solve_ivp(rhs_p1, [0, 50], y0, method='Radau',
                         events=event_p1, rtol=rtol, atol=atol,
                         max_step=0.5)

    if sol1.status < 0:
        return {"success": False, "error": f"Phase 1 failed: {sol1.message}"}

    y_p1 = sol1.y[:, -1]
    N_handoff = sol1.t[-1]

    # Extract diagnostics at handoff
    psi_handoff = y_p1[2:2+n_transport].reshape(n_species_transport, n_ell, N_q)
    diag_handoff = psi_ell_amplitudes(psi_handoff, tuple(ell_list))
    Xn_handoff = y_p1[i_net]

    # ── Phase 2 ──
    n_net_p2 = N_SPECIES
    n_total_p2 = 2 + n_transport + 1 + n_net_p2
    i_tg_p2 = 2 + n_transport
    i_net_p2 = i_tg_p2 + 1

    X_phase2 = phase1_to_phase2(Xn_handoff)
    y_handoff = np.zeros(n_total_p2)
    y_handoff[0] = y_p1[0]  # Σ₊
    y_handoff[1] = y_p1[1]  # Σ₋
    y_handoff[2:2+n_transport] = y_p1[2:2+n_transport]
    y_handoff[i_tg_p2] = y_p1[i_tg]
    y_handoff[i_net_p2:i_net_p2 + n_net_p2] = X_phase2

    def rhs_p2(N, y):
        return coupled_rhs(N, y, phase=2, **common_kw)

    def event_p2(N, y):
        return y[i_tg_p2] - T_end
    event_p2.terminal = True
    event_p2.direction = -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol2 = solve_ivp(rhs_p2, [N_handoff, N_handoff + 30], y_handoff,
                         method='Radau', events=event_p2,
                         rtol=rtol, atol=atol, max_step=0.5)

    elapsed = time.perf_counter() - t0

    if sol2.status < 0:
        return {"success": False, "error": f"Phase 2 failed: {sol2.message}"}

    y_final = sol2.y[:, -1]
    X_final = y_final[i_net_p2:i_net_p2 + n_net_p2]

    # Extract final diagnostics
    psi_final = y_final[2:2+n_transport].reshape(n_species_transport, n_ell, N_q)
    diag_final = psi_ell_amplitudes(psi_final, tuple(ell_list))

    Yp = float(X_final[5])
    DH = float(X_final[2] / (2.0 * max(X_final[1], 1e-30)))

    return {
        "success": True,
        "Yp": Yp,
        "DH": DH,
        "Xn_freeze": float(Xn_handoff),
        "Sigma_final": float(y_final[0]),
        "T_final": float(y_final[i_tg_p2]),
        "elapsed_s": round(elapsed, 2),
        "n_steps_p1": len(sol1.t),
        "n_steps_p2": len(sol2.t),
        "psi_amplitudes_handoff": diag_handoff,
        "psi_amplitudes_final": diag_final,
        "config": {
            "Sigma_H": Sigma_H, "ell_max": ell_max, "N_q": N_q, "N_mu": N_mu,
            "linearized": linearized, "q_independent_source": q_independent_source,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# §6. Ablation study
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR = Path(__file__).parent.parent / "audit_outputs"
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  Nonlinear Boltzmann Ablation Study")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    results = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "description": "Disentangle ℓ-truncation, Ψ-linearization, q-source approximations",
        "runs": [],
    }

    SIGMAS = [0.0, 0.1, 0.3, 0.5]
    ELL_MAX_LIST = [2, 4, 6, 8]
    MODES = [
        # (label, linearized, q_independent_source)
        ("current_linear_qindep", True, True),       # Current code
        ("linear_qdep", True, False),                 # Fix q-source only
        ("nonlinear_qdep", False, False),             # Full nonlinear
    ]

    N_Q = 20

    for sigma in SIGMAS:
        print(f"\n{'='*60}")
        print(f"  Σ = {sigma}")
        print(f"{'='*60}")

        for ell_max in ELL_MAX_LIST:
            for label, lin, qi in MODES:
                tag = f"Σ={sigma}_L{ell_max}_{label}"
                print(f"\n  [{tag}]", end=" ", flush=True)

                r = run_bbn(
                    Sigma_H=sigma, ell_max=ell_max, N_q=N_Q,
                    linearized=lin, q_independent_source=qi,
                )

                if r["success"]:
                    print(f"Yp={r['Yp']:.10f}  DH={r['DH']:.6e}  "
                          f"max|Ψ₀|={r['psi_amplitudes_final'].get('max_Psi_0', 0):.2e}  "
                          f"max|Ψ₂|={r['psi_amplitudes_final'].get('max_Psi_2', 0):.2e}  "
                          f"({r['elapsed_s']}s)")
                else:
                    print(f"FAILED: {r.get('error', '?')}")

                r["tag"] = tag
                results["runs"].append(r)

    # ── Analysis ──
    print(f"\n\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    # Group by Σ, extract baselines
    for sigma in SIGMAS:
        print(f"\n  Σ = {sigma}:")
        baseline = None
        for r in results["runs"]:
            if not r["success"]:
                continue
            c = r["config"]
            if c["Sigma_H"] == sigma and c["ell_max"] == 2 and c["linearized"] and c["q_independent_source"]:
                baseline = r
                break

        if baseline is None:
            print("    (no baseline)")
            continue

        Yp_base = baseline["Yp"]
        print(f"    Baseline (L2/linear/qindep): Yp={Yp_base:.10f}")

        for r in results["runs"]:
            if not r["success"]:
                continue
            c = r["config"]
            if c["Sigma_H"] != sigma:
                continue
            dYp = r["Yp"] - Yp_base
            psi0 = r["psi_amplitudes_final"].get("max_Psi_0", 0)
            psi2 = r["psi_amplitudes_final"].get("max_Psi_2", 0)
            print(f"    L{c['ell_max']:1d} {'lin' if c['linearized'] else 'NL':3s} "
                  f"{'qi' if c['q_independent_source'] else 'qd':2s}: "
                  f"Yp={r['Yp']:.10f}  ΔYp={dYp:+.2e}  "
                  f"|Ψ₀|={psi0:.2e}  |Ψ₂|={psi2:.2e}")

    # ── Write outputs ──
    json_path = OUT_DIR / "NONLINEAR_ABLATION_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {json_path}")

    # ── Report ──
    md_path = OUT_DIR / "NONLINEAR_ABLATION_REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Nonlinear Boltzmann Ablation Study\n\n")
        f.write(f"Generated: {results['generated']}\n\n")
        f.write("## Configuration Matrix\n\n")
        f.write("| Σ | ℓ_max | Mode | Yp | ΔYp (vs baseline) | max|Ψ₀| | max|Ψ₂| | Time |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for sigma in SIGMAS:
            baseline_yp = None
            for r in results["runs"]:
                c = r["config"]
                if c["Sigma_H"] == sigma and c["ell_max"] == 2 and c["linearized"] and c["q_independent_source"] and r["success"]:
                    baseline_yp = r["Yp"]
            for r in results["runs"]:
                if not r["success"]:
                    continue
                c = r["config"]
                if c["Sigma_H"] != sigma:
                    continue
                mode = f"{'lin' if c['linearized'] else 'NL'}/{'qi' if c['q_independent_source'] else 'qd'}"
                dYp = r["Yp"] - baseline_yp if baseline_yp else 0
                psi0 = r["psi_amplitudes_final"].get("max_Psi_0", 0)
                psi2 = r["psi_amplitudes_final"].get("max_Psi_2", 0)
                f.write(f"| {c['Sigma_H']} | {c['ell_max']} | {mode} | "
                        f"{r['Yp']:.10f} | {dYp:+.2e} | {psi0:.2e} | {psi2:.2e} | {r['elapsed_s']}s |\n")
        f.write("\n## Key\n\n")
        f.write("- **lin/qi**: Current code (linearized Ψ, q-independent source)\n")
        f.write("- **lin/qd**: Linearized Ψ, exact q-dependent source\n")
        f.write("- **NL/qd**: Full nonlinear (no Ψ-linearization, exact q-dependent source)\n")
        f.write("- **ΔYp**: Difference from ℓ_max=2/lin/qi baseline at same Σ\n")
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
