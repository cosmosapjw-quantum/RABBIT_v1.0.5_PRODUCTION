#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _norm(x) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.sum(arr * arr)))


def _fd_eq(q):
    q = np.asarray(q, dtype=np.float64)
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["shared", "species"], required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--relax", type=float, required=True)
    p.add_argument("--cl", type=int, default=0)
    p.add_argument("--tier", type=int, default=2)
    p.add_argument("--nq", type=int, default=20)
    p.add_argument("--nmu", type=int, default=12)
    p.add_argument("--reactions", type=int, default=12)
    p.add_argument("--sample-dN", type=float, default=0.01)
    p.add_argument("--max-collision-calls", type=int, default=2000)
    p.add_argument("--out", type=str, required=True)
    return p


def main():
    args = make_parser().parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["RABBIT_COLLISION_BRIDGE_RELAX"] = str(args.relax)

    tap = {
        "meta": {
            "target": args.target,
            "sigma": args.sigma,
            "relax_env": args.relax,
            "cl": args.cl,
            "tier": args.tier,
            "nq": args.nq,
            "nmu": args.nmu,
            "reactions": args.reactions,
            "sample_dN": args.sample_dN,
        },
        "counters": {
            "coupled_rhs_calls": 0,
            "phase1_trace_points": 0,
            "shared_bridge_calls": 0,
            "species_bridge_calls": 0,
            "transport_calls": 0,
        },
        "phase1_trace": [],
        "shared_bridge_trace": [],
        "species_bridge_trace": [],
        "transport_trace": [],
        "script_output_path": None,
        "script_output": None,
    }

    # imports after env set
    import rabbit.drivers.full_coupled_typeI as fc
    import rabbit.transport.teff_collision_bridge as tcb
    import rabbit.transport.species_tagged_bridge as stb
    from rabbit.thermo.incomplete_decoupling import compute_energy_exchange_rate

    orig_coupled_rhs = fc.coupled_rhs
    orig_transport = getattr(fc, "characteristic_transport_rhs", None)
    orig_shared_bridge = tcb.apply_gather_scatter_collision
    orig_species_bridge = stb.apply_species_tagged_bridge

    last_trace_N = {"value": None}

    def record_phase1_state(N, y, kwargs):
        tier = int(kwargs.get("tier", 1))
        N_mu = int(kwargs.get("N_mu", 12))
        transport_mode = kwargs.get("transport_mode", None)
        use_char = str(getattr(transport_mode, "value", transport_mode)).lower().startswith("character")

        if not use_char:
            return

        try:
            layout = fc._layout_characteristic(N_mu, tier=tier)
            (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total,
             i_I, i_J, i_S) = layout
        except Exception:
            return

        y = np.asarray(y, dtype=np.float64)
        row = {
            "N": float(N),
            "sigma_plus": float(y[fc._I_SP]),
            "sigma_minus": float(y[fc._I_SM]),
            "T_gamma": float(y[i_tg]),
            "Xn": float(y[i_net]),
            "Xp_or_Xp2": float(y[i_net + 1]) if i_net + 1 < len(y) else None,
        }

        if tier >= 2:
            row["T_nu_e"] = float(y[i_tne])
            row["T_nu_x"] = float(y[i_tnx])
        else:
            row["T_nu_e"] = None
            row["T_nu_x"] = None

        try:
            I_vals = y[i_I:i_I + N_mu]
            J_vals = y[i_J:i_J + N_mu]
            S_val = float(y[i_S])
            row["I0"] = float(I_vals[0]) if len(I_vals) else None
            row["J0"] = float(J_vals[0]) if len(J_vals) else None
            row["S"] = S_val

            cc = kwargs.get("_char_cache", None)
            if cc is not None:
                mu = fc.mu_current(cc["X0"], cc["signs"], S_val)
                pi_plus = fc.char_extract_stress(I_vals, J_vals, mu, cc["w0"], kwargs.get("f_nu", 0.4052))
                row["pi_plus"] = float(pi_plus)
                row["sigma_sq"] = float(row["sigma_plus"] ** 2 + row["sigma_minus"] ** 2)
                row["omega_margin"] = float(1.0 - row["sigma_sq"])
            else:
                row["pi_plus"] = None
        except Exception:
            row["pi_plus"] = None

        tap["phase1_trace"].append(row)
        tap["counters"]["phase1_trace_points"] += 1

    def wrapped_coupled_rhs(N, y, *a, **kw):
        tap["counters"]["coupled_rhs_calls"] += 1
        phase = int(kw.get("phase", 0))
        if phase == 1:
            lastN = last_trace_N["value"]
            if lastN is None or abs(float(N) - lastN) >= args.sample_dN:
                record_phase1_state(N, y, kw)
                last_trace_N["value"] = float(N)
        return orig_coupled_rhs(N, y, *a, **kw)

    def wrapped_transport(*a, **kw):
        out = orig_transport(*a, **kw)
        tap["counters"]["transport_calls"] += 1
        if len(tap["transport_trace"]) < args.max_collision_calls:
            try:
                sigma_plus = float(a[0])
                dI, dJ, dS = out
                tap["transport_trace"].append({
                    "call": tap["counters"]["transport_calls"],
                    "sigma_plus": sigma_plus,
                    "dI_free_norm": _norm(dI),
                    "dJ_free_norm": _norm(dJ),
                    "dS_free_abs": float(abs(dS)),
                })
            except Exception:
                pass
        return out

    def wrapped_shared_bridge(I, J, w0, q_nodes, q_weights, T_gamma, T_nu, H, *a, **kw):
        out_obj = orig_shared_bridge(I, J, w0, q_nodes, q_weights, T_gamma, T_nu, H, *a, **kw)
        tap["counters"]["shared_bridge_calls"] += 1
        if len(tap["shared_bridge_trace"]) < args.max_collision_calls:
            q_nodes_np = np.asarray(q_nodes, dtype=np.float64)
            q_weights_np = np.asarray(q_weights, dtype=np.float64)
            C = np.asarray(getattr(out_obj, "C_monopole", np.zeros_like(q_nodes_np)), dtype=np.float64)
            dI = np.asarray(getattr(out_obj, "delta_I", np.zeros_like(w0)), dtype=np.float64)
            fmono = np.asarray(getattr(out_obj, "f_monopole", np.zeros_like(q_nodes_np)), dtype=np.float64)
            feq = _fd_eq(q_nodes_np)
            qdot = float(compute_energy_exchange_rate(C, q_nodes_np, q_weights_np, float(T_nu)))
            tap["shared_bridge_trace"].append({
                "call": tap["counters"]["shared_bridge_calls"],
                "T_gamma": float(T_gamma),
                "T_nu": float(T_nu),
                "H": float(H),
                "C_norm": _norm(C),
                "deltaI_norm": _norm(dI),
                "fdev_norm": _norm(fmono - feq),
                "qdot_shape": qdot,
                "tangency_D2": float(getattr(out_obj, "tangency_D2", np.nan)),
                "delta_rho_nu": float(getattr(out_obj, "delta_rho_nu", np.nan)),
                "theta_ray_norm": _norm(getattr(out_obj, "theta_per_ray", 0.0)),
                "relax_env": args.relax,
            })
        return out_obj

    def wrapped_species_bridge(*, species, I, J, w0, q_nodes, q_weights, T_gamma, T_nu_e, T_nu_x, H):
        out_obj = orig_species_bridge(
            species=species, I=I, J=J, w0=w0, q_nodes=q_nodes, q_weights=q_weights,
            T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H=H
        )
        tap["counters"]["species_bridge_calls"] += 1
        if len(tap["species_bridge_trace"]) < args.max_collision_calls:
            q_nodes_np = np.asarray(q_nodes, dtype=np.float64)
            q_weights_np = np.asarray(q_weights, dtype=np.float64)
            C = np.asarray(getattr(out_obj, "C_monopole", np.zeros_like(q_nodes_np)), dtype=np.float64)
            dI = np.asarray(getattr(out_obj, "delta_I", np.zeros_like(w0)), dtype=np.float64)
            fmono = np.asarray(getattr(out_obj, "f_monopole", np.zeros_like(q_nodes_np)), dtype=np.float64)
            feq = _fd_eq(q_nodes_np)
            qdot = float(compute_energy_exchange_rate(C, q_nodes_np, q_weights_np, float(T_nu_e)))
            tap["species_bridge_trace"].append({
                "call": tap["counters"]["species_bridge_calls"],
                "species": str(species),
                "T_gamma": float(T_gamma),
                "T_nu_e": float(T_nu_e),
                "T_nu_x": float(T_nu_x),
                "H": float(H),
                "C_norm": _norm(C),
                "deltaI_norm": _norm(dI),
                "fdev_norm": _norm(fmono - feq),
                "qdot_target": qdot,
                "amp": float(getattr(out_obj, "species_tagged_amp", np.nan)),
                "alpha": float(getattr(out_obj, "species_tagged_alpha", np.nan)),
                "qdot_shape": float(getattr(out_obj, "species_tagged_qdot_shape", np.nan)),
                "tangency_D2": float(getattr(out_obj, "tangency_D2", np.nan)),
                "delta_rho_nu": float(getattr(out_obj, "delta_rho_nu", np.nan)),
                "theta_ray_norm": _norm(getattr(out_obj, "theta_per_ray", 0.0)),
                "relax_env": args.relax,
            })
        return out_obj

    # apply patches
    fc.coupled_rhs = wrapped_coupled_rhs
    if orig_transport is not None:
        fc.characteristic_transport_rhs = wrapped_transport
    tcb.apply_gather_scatter_collision = wrapped_shared_bridge
    stb.apply_species_tagged_bridge = wrapped_species_bridge

    raw_out = out.with_suffix(".raw.json")
    tap["script_output_path"] = str(raw_out)

    if args.target == "shared":
        target_script = ROOT / "scripts" / "trace_scipy_typeI_phase1_external.py"
        sys.argv = [
            str(target_script),
            "--sigma", str(args.sigma),
            "--cl", str(args.cl),
            "--nq", str(args.nq),
            "--reactions", str(args.reactions),
            "--out", str(raw_out),
        ]
    else:
        target_script = ROOT / "scripts" / "trace_species_rays_phase1.py"
        sys.argv = [
            str(target_script),
            "--sigma", str(args.sigma),
            "--relax", str(args.relax),
            "--cl", str(args.cl),
            "--nq", str(args.nq),
            "--nmu", str(args.nmu),
            "--out", str(raw_out),
        ]

    try:
        runpy.run_path(str(target_script), run_name="__main__")
    except SystemExit:
        pass

    if raw_out.exists():
        tap["script_output"] = json.loads(raw_out.read_text())

    # save json
    out.write_text(json.dumps(tap, indent=2, default=_json_default))

    # plots
    phase_png = out.with_name(out.stem + "_phase.png")
    coll_png = out.with_name(out.stem + "_collision.png")

    tr = tap["phase1_trace"]
    if tr:
        N = np.array([r["N"] for r in tr], dtype=float)
        sig = np.array([r.get("sigma_plus", np.nan) for r in tr], dtype=float)
        pi = np.array([np.nan if r.get("pi_plus") is None else r.get("pi_plus") for r in tr], dtype=float)
        Xn = np.array([r.get("Xn", np.nan) for r in tr], dtype=float)
        lnp = np.array([np.nan for _ in tr], dtype=float)

        if tap["script_output"] is not None:
            pass

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0, 0].plot(N, sig)
        ax[0, 0].set_title("sigma_plus vs N")
        ax[0, 1].plot(N, pi)
        ax[0, 1].set_title("pi_plus vs N")
        ax[1, 0].plot(N, Xn)
        ax[1, 0].set_title("Xn vs N")
        sc = ax[1, 1].scatter(sig, pi, c=N, s=12)
        ax[1, 1].set_title("(sigma_plus, pi_plus) phase")
        ax[1, 1].set_xlabel("sigma_plus")
        ax[1, 1].set_ylabel("pi_plus")
        fig.colorbar(sc, ax=ax[1, 1], label="N")
        fig.tight_layout()
        fig.savefig(phase_png, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    if tap["shared_bridge_trace"]:
        xs = np.arange(len(tap["shared_bridge_trace"]))
        ax[0].plot(xs, [r["C_norm"] for r in tap["shared_bridge_trace"]], label="shared C_norm")
        ax[0].plot(xs, [r["deltaI_norm"] for r in tap["shared_bridge_trace"]], label="shared deltaI_norm")
        ax[1].plot(xs, [r["qdot_shape"] for r in tap["shared_bridge_trace"]], label="shared qdot_shape")
    if tap["species_bridge_trace"]:
        species_names = sorted(set(r["species"] for r in tap["species_bridge_trace"]))
        for sp in species_names:
            rows = [r for r in tap["species_bridge_trace"] if r["species"] == sp]
            xs = np.arange(len(rows))
            ax[0].plot(xs, [r["C_norm"] for r in rows], label=f"{sp} C_norm")
            ax[0].plot(xs, [r["deltaI_norm"] for r in rows], label=f"{sp} deltaI_norm", linestyle="--")
            ax[1].plot(xs, [r["qdot_target"] for r in rows], label=f"{sp} qdot_target")
    if tap["transport_trace"]:
        xs = np.arange(len(tap["transport_trace"]))
        ax[0].plot(xs, [r["dI_free_norm"] for r in tap["transport_trace"]], label="free dI_norm", alpha=0.8)
    ax[0].set_title("collision / free norms")
    ax[1].set_title("qdot diagnostics")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(coll_png, dpi=160)
    plt.close(fig)

    print(json.dumps({
        "saved_json": str(out),
        "saved_phase_png": str(phase_png),
        "saved_collision_png": str(coll_png),
        "counters": tap["counters"],
    }, indent=2))


if __name__ == "__main__":
    main()
