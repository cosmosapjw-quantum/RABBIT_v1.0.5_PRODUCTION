"""V3 campaign instrument. SCRATCH ONLY. Executes the protocol sealed at commit 0af4c7c.

Fixes relative to V2, each tied to a recorded defect:
  * observation Jacobians run under an OBSERVATION budget exempt from the integration call cap
    (V2's checkpoints died on CapReached because this exemption, present in V1, was dropped);
  * accepted states are written to DISK the moment a checkpoint index is crossed (V2 held them in
    memory and lost all of them with the process);
  * jac_factor is logged at the init refresh too (V2's first record was at call 521);
  * a STARTUP OBSERVATION-MANIFEST SELF-TEST produces every named artifact class end-to-end on an
    order-8 miniature, including the budget exemption, before the real integration begins -- a leg
    that cannot produce its observations refuses to start (FAIL_PROTOCOL at minute zero);
  * criteria are digest-loaded from the sealed protocol (V2 pattern, unchanged).
"""

from __future__ import annotations

import argparse, hashlib, json, re, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path("/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "audit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _trajectory_core as core                                              # noqa: E402
import instrumented_rhs                                                      # noqa: E402
from instrumented_bdf import InstrumentedBDF, verify_verbatim as verify_bdf  # noqa: E402
from rabbit.decoupling import _independent_noqke as ind                      # noqa: E402

PROTOCOL = REPO / "docs/audit/BD622_V3_protocol_2026-08-05.md"
CRITERIA_SHA = "2a42fd263bcf278aea1f9da9f0651cc569d8a60b5f9cb0e14134542510e49e03"
PINS = {
    "frozen_module_sha256": "760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a",
    "scipy": "1.17.0", "numpy": "2.4.2",
    "scipy_bdf_source_sha256": "a5b75a2ae8aca2e66cc35ed268af51d8a9685878209d5cf77a44c8ccef3b76e6",
}
INPUT_COMMIT = "0af4c7c70513da3444356d0c1d71d364900c114f"
EPS = np.finfo(float).eps

LEGS = {
    "v3a": {"order": 60, "y_max": 30.0, "call_cap": 4000, "wall_cap": 7.0 * 3600,
            "state_checkpoints": (1200, 2000, 3000, 3900), "obs_jacobians": 4,
            "ratcheted_recompute": True},
    "v3b": {"order": 48, "y_max": 24.0, "call_cap": 6000, "wall_cap": 6.0 * 3600,
            "state_checkpoints": (1500, 3000, 4500, 5800), "obs_jacobians": 2,
            "ratcheted_recompute": False},
    "v3c": {"order": 60, "y_max": 30.0, "call_cap": 1500, "wall_cap": 2.5 * 3600,
            "state_checkpoints": (), "obs_jacobians": 0, "ratcheted_recompute": False},
}


def load_sealed_criteria() -> dict:
    text = PROTOCOL.read_text(encoding="utf-8")
    m = re.search(r"## 4\. Sealed adjudication criteria.*?\n```json\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit("FAIL_PROTOCOL: sealed criteria block not found")
    block = m.group(1)
    got = hashlib.sha256(block.encode()).hexdigest()
    if got != CRITERIA_SHA:
        raise SystemExit(f"FAIL_PROTOCOL: criteria digest {got} != sealed {CRITERIA_SHA}")
    return json.loads(block)


class CapReached(RuntimeError):
    pass


class Sink:
    def __init__(self, outdir: Path, call_cap: int, wall_cap: float, state_checkpoints):
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        op = lambda n: (outdir / n).open("w", buffering=1)
        self.f_rhs, self.f_trial, self.f_acc = op("rhs_calls.jsonl"), op("trials.jsonl"), op("accepted.jsonl")
        self.f_newt, self.f_step = op("newton_calls.jsonl"), op("step_events.jsonl")
        self.f_jac, self.f_factor = op("jacobian_events.jsonl"), op("jac_factor.jsonl")
        self.call_cap, self.wall_cap = call_cap, wall_cap
        self.state_checkpoints = list(state_checkpoints)
        self.saved_states: dict[int, str] = {}
        self.n_raw = 0
        self.started = time.monotonic()
        self.stop_reason = None
        self.budget_mode = "integration"   # observations are exempt from the call cap
        self.solver = None
        self._step_index = 0
        self.last_accepted: tuple[float, np.ndarray] | None = None

    def raw_calls(self):
        return self.n_raw

    def _enforce(self):
        if self.budget_mode != "integration":
            # observations obey only the wall cap
            if time.monotonic() - self.started >= self.wall_cap:
                self.stop_reason = "wall_cap"; raise CapReached(self.stop_reason)
            return
        if self.n_raw >= self.call_cap:
            self.stop_reason = f"call_cap:{self.call_cap}"; raise CapReached(self.stop_reason)
        if time.monotonic() - self.started >= self.wall_cap:
            self.stop_reason = "wall_cap"; raise CapReached(self.stop_reason)

    def raw_call(self, n, t_cm, t_gamma, action, thermo, stats):
        self.n_raw += 1
        total = np.abs(np.asarray(action.total))
        chan = np.abs(np.asarray(action.electron)) + np.abs(np.asarray(action.self_interaction))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, chan / np.maximum(total, np.finfo(float).tiny), np.nan)
        self.f_rhs.write(json.dumps({
            "i": self.n_raw, "wall": round(time.monotonic() - self.started, 3),
            "N": float(n), "T_cm": float(t_cm), "T_gamma": float(t_gamma),
            "dom_rej": int(action.whole_reaction_domain_rejections),
            "roundoff_max": float(action.largest_matrix_roundoff_correction),
            "cancel_max": float(np.nanmax(ratio)),
            "first_law_residual": float(action.diagnostics["first_law_residual"]),
            "budget": self.budget_mode,
        }) + "\n")
        self._enforce()

    def log_factor(self, tag: str, t: float):
        fac = getattr(self.solver, "jac_factor", None)
        if fac is None:
            return
        fa = np.asarray(fac, dtype=float)
        self.f_factor.write(json.dumps({
            "tag": tag, "at_raw": self.n_raw, "t": float(t), "step": self._step_index,
            "sqrt_eps": float(np.sqrt(EPS)),
            "min": float(fa.min()), "max": float(fa.max()), "median": float(np.median(fa)),
            "n_above_10x": int((fa > 10 * np.sqrt(EPS)).sum()),
            "n_above_100x": int((fa > 100 * np.sqrt(EPS)).sum()),
            "n_above_1e4x": int((fa > 1e4 * np.sqrt(EPS)).sum()),
            "ratcheted_components": [int(i) for i in np.where(fa > 10 * np.sqrt(EPS))[0]][:120],
            "factor": [float(x) for x in fa],
        }) + "\n")

    def step_enter(self, idx, t, h_abs, order, min_step):
        self._step_index = idx
        self.f_step.write(json.dumps({"step": idx, "event": "enter", "t": float(t),
                                      "h_abs": float(h_abs), "order": int(order),
                                      "min_step": float(min_step), "raw": self.n_raw}) + "\n")

    def step_exit(self, idx, h_abs, order, kind, error_norms=None, factor=None):
        self.f_step.write(json.dumps({"step": idx, "event": "exit", "kind": kind,
                                      "h_abs": float(h_abs), "order": int(order),
                                      "error_norms": error_norms, "factor": factor,
                                      "raw": self.n_raw}) + "\n")

    def newton(self, idx, t_new, h_abs, order, converged, n_iter, current_jac):
        self.f_newt.write(json.dumps({"step": idx, "t_new": float(t_new), "h_abs": float(h_abs),
                                      "order": int(order), "converged": bool(converged),
                                      "n_iter": int(n_iter), "jac_was_current": bool(current_jac),
                                      "raw": self.n_raw}) + "\n")

    def jacobian_event(self, t, raw_calls):
        self.f_jac.write(json.dumps({"t": float(t), "raw_calls": int(raw_calls),
                                     "at_raw": self.n_raw, "step": self._step_index,
                                     "budget": self.budget_mode}) + "\n")
        self.log_factor("refresh", t)

    def lu_event(self):
        pass

    def error_test(self, idx, t_new, h_abs, order, error, scale, error_norm, n_iter):
        scaled = np.abs(error / scale)
        top = np.argsort(scaled)[::-1][:5]
        self.f_step.write(json.dumps({"step": idx, "event": "error_test", "t_new": float(t_new),
                                      "h_abs": float(h_abs), "order": int(order),
                                      "n_iter": int(n_iter), "error_norm": float(error_norm),
                                      "argmax": int(np.argmax(scaled)),
                                      "top5": [int(i) for i in top], "raw": self.n_raw}) + "\n")

    def trial(self, idx, t, h_abs, order, outcome):
        self.f_trial.write(json.dumps({"step": idx, "t": float(t), "h_abs": float(h_abs),
                                       "order": int(order), "outcome": outcome, "raw": self.n_raw,
                                       "wall": round(time.monotonic() - self.started, 3)}) + "\n")

    def accepted(self, idx, t_new, h_abs, order, y_new, error_norm, n_iter):
        self.last_accepted = (float(t_new), y_new.copy())
        # write the state to disk the moment a checkpoint index is crossed -- V2's states died in RAM
        for c in self.state_checkpoints:
            if c not in self.saved_states and self.n_raw >= c:
                p = self.outdir / f"state_{c}.npz"
                np.savez_compressed(p, t=t_new, y=y_new, raw=self.n_raw, h=h_abs, order=order)
                self.saved_states[c] = p.name
        self.f_acc.write(json.dumps({"step": idx, "t": float(t_new), "h_abs": float(h_abs),
                                     "order": int(order), "error_norm": float(error_norm),
                                     "n_iter": int(n_iter), "raw": self.n_raw,
                                     "wall": round(time.monotonic() - self.started, 3)}) + "\n")

    def close(self):
        for f in (self.f_rhs, self.f_trial, self.f_acc, self.f_newt, self.f_step,
                  self.f_jac, self.f_factor):
            f.close()


# ---- observations (budget-exempt) ------------------------------------------------------------

def obs_pinned_jacobian(rhs, sink, t, y, out_path: Path, log):
    """Pinned sqrt(EPS)-step Jacobian, observation budget."""
    prev = sink.budget_mode; sink.budget_mode = "observation"
    try:
        f0 = rhs(t, y)
        ys = np.sqrt(EPS) * np.maximum(np.abs(y), 1e-9)
        h = (y + ys) - y
        J = np.empty((y.size, y.size))
        for j in range(y.size):
            yp = y.copy(); yp[j] += h[j]
            J[:, j] = (rhs(t, yp) - f0) / h[j]
        w = np.linalg.eigvals(J[:-2, :-2])
        np.savez_compressed(out_path, J=J, h=h, y=y, t=t, eig=w)
        log(f"obs_jac -> {out_path.name}: N={t:.6f} abscissa={w.real.max():.4e} "
            f"n_pos={(w.real > 0).sum()}")
        return J, f0
    finally:
        sink.budget_mode = prev


def obs_ratcheted_columns(rhs, sink, t, y, f0, factor_vec, cols, out_path: Path, log):
    """The SAME columns, differenced at the logged ratcheted factors, observation budget."""
    prev = sink.budget_mode; sink.budget_mode = "observation"
    try:
        cols = [int(c) for c in cols]
        C = np.empty((y.size, len(cols))); hs = []
        for k, j in enumerate(cols):
            h = (y[j] + factor_vec[j] * max(abs(y[j]), 1e-9)) - y[j]
            yp = y.copy(); yp[j] += h
            C[:, k] = (rhs(t, yp) - f0) / h
            hs.append(float(h))
        np.savez_compressed(out_path, cols=np.array(cols), C=C, h=np.array(hs), t=t,
                            factor=np.array([factor_vec[j] for j in cols]))
        log(f"ratcheted_cols -> {out_path.name}: {len(cols)} columns")
    finally:
        sink.budget_mode = prev


# ---- lifted initial conditions for v3c -------------------------------------------------------

R4_REPORT = REPO / ".agent-harness/runs/run-20260729-f10-d069-trajectory-r4/raw_logs/r4_trajectory_report.json"


def lifted_state(n_star: float, setup, out_path: Path) -> np.ndarray:
    """60/30 state constructed from the retained base checkpoint at N = n_star.

    y<=24: linear interpolation of each species' cloglog over y from the base 48/24 grid
    (np.interp; clamped at the ends, which matters only for the first node 0.0118 < base 0.0147).
    y>24: the equilibrium spectrum via the frozen module's own transform
    (pair_logits_to_cloglog of the Fermi-Dirac logits -y).
    Elapsed-time variable set to 0.0 -- structurally decoupled (its Jacobian column is exactly
    zero, measured in V1C), so it cannot affect the dynamics.
    All construction parameters recorded in lift_record.json. CONSTRUCTED INITIAL CONDITION:
    physics comparisons against the base trajectory are out of bounds; pathology presence only.
    """
    rep = json.loads(R4_REPORT.read_text())
    cps = rep["base"]["checkpoint_states"]
    cp = min(cps, key=lambda c: abs(c["N"] - n_star))
    if abs(cp["N"] - n_star) > 1e-9:
        raise SystemExit(f"FAIL_PROTOCOL: no base checkpoint at N={n_star} (nearest {cp['N']})")
    base_setup = core.build_setup(order=48, y_max=24.0, label="lift-src")
    base_nodes = np.asarray(base_setup.grid.nodes)
    tgt_nodes = np.asarray(setup.grid.nodes)
    c_base = np.asarray(cp["cloglog"])                       # (3, 48)
    eq_logits = np.stack([-tgt_nodes for _ in range(3)])
    c_eq = ind.pair_logits_to_cloglog(eq_logits)             # (3, 60) equilibrium
    c_tgt = np.empty((3, setup.order))
    kept = tgt_nodes <= 24.0
    for s in range(3):
        c_tgt[s, kept] = np.interp(tgt_nodes[kept], base_nodes, c_base[s])
        c_tgt[s, ~kept] = c_eq[s, ~kept]
    y0 = np.concatenate((c_tgt.ravel(), [float(cp["T_gamma_mev"]), 0.0]))
    out_path.write_text(json.dumps({
        "caveat": "CONSTRUCTED INITIAL CONDITION -- pathology presence only; no physics comparison",
        "n_star": n_star, "source_checkpoint_N": cp["N"],
        "source": str(R4_REPORT), "interp": "np.interp linear in y, clamped ends",
        "tail": "equilibrium via pair_logits_to_cloglog(-y)",
        "n_kept_nodes": int(kept.sum()), "n_tail_nodes": int((~kept).sum()),
        "first_target_node": float(tgt_nodes[0]), "first_base_node": float(base_nodes[0]),
        "T_gamma": float(cp["T_gamma_mev"]),
        "elapsed_time_set_to": 0.0,
        "elapsed_time_justification": "Jacobian column exactly zero (V1C measurement)",
    }, indent=2))
    return y0


# ---- startup observation-manifest self-test --------------------------------------------------

def manifest_selftest(outroot: Path, leg: str, log) -> dict:
    """Produce every named artifact class end-to-end on an order-8 miniature, incl. the budget
    exemption. A leg that cannot produce its observations refuses to start."""
    st = outroot / "selftest"; st.mkdir(parents=True, exist_ok=True)
    setup = core.build_setup(order=8, y_max=24.0, label="selftest")
    sink = Sink(st, call_cap=60, wall_cap=600, state_checkpoints=(40,))
    instrumented_rhs.SINK = sink
    stats = core.Stats()
    rhs = instrumented_rhs.make_rhs(setup, stats, core.Deadline(600))
    _c0, y0 = core.initial_state(setup)
    ok, err = True, []
    try:
        solver = InstrumentedBDF(rhs, 0.0, y0, 1.0, rtol=1e-6, atol=1e-9, vectorized=False)
        sink.solver = solver
        solver.attach_probe(sink)
        sink.log_factor("init", 0.0)                       # factor at init -- U15b path
        try:
            while solver.status == "running":
                solver.step()
        except CapReached:
            pass
        # budget exemption: observation must run past the (already reached) call cap
        t_s, y_s = sink.last_accepted if sink.last_accepted else (0.0, y0)
        # exercise the in-run refresh path explicitly: a healthy miniature may never fail Newton,
        # so jacobian_event would otherwise go untested until the real run
        prev = sink.budget_mode; sink.budget_mode = "observation"
        try:
            solver.jac(t_s, y_s)
        finally:
            sink.budget_mode = prev
        J, f0 = obs_pinned_jacobian(rhs, sink, t_s, y_s, st / "obs_jac_selftest.npz", log)
        fac = np.asarray(solver.jac_factor, dtype=float)
        obs_ratcheted_columns(rhs, sink, t_s, y_s, f0, fac, [0, 1], st / "ratcheted_selftest.npz", log)
    except BaseException as exc:
        ok, err = False, [f"{type(exc).__name__}: {exc}"]
    sink.close()
    required = ["rhs_calls.jsonl", "trials.jsonl", "accepted.jsonl", "newton_calls.jsonl",
                "step_events.jsonl", "jacobian_events.jsonl", "jac_factor.jsonl",
                "state_40.npz", "obs_jac_selftest.npz", "ratcheted_selftest.npz"]
    missing = [r for r in required if not (st / r).exists() or (st / r).stat().st_size == 0]
    # jac_factor must contain an 'init' tagged row
    has_init = any(json.loads(l).get("tag") == "init"
                   for l in (st / "jac_factor.jsonl").read_text().splitlines() if l.strip()) \
        if (st / "jac_factor.jsonl").exists() else False
    result = {"ok": ok and not missing and has_init, "errors": err,
              "missing_artifacts": missing, "init_factor_logged": has_init}
    (outroot / "selftest_result.json").write_text(json.dumps(result, indent=2))
    return result


# ---- legs -------------------------------------------------------------------------------------

def run_leg(leg: str, outroot: Path, lift_n: float | None, log) -> dict:
    cfg = LEGS[leg]
    setup = core.build_setup(order=cfg["order"], y_max=cfg["y_max"], label=leg)
    d = outroot / "domain"; d.mkdir(parents=True, exist_ok=True)
    sink = Sink(d, cfg["call_cap"], cfg["wall_cap"], cfg["state_checkpoints"])
    instrumented_rhs.SINK = sink
    stats = core.Stats()
    rhs = instrumented_rhs.make_rhs(setup, stats, core.Deadline(cfg["wall_cap"] * 4), log=log)
    if leg == "v3c":
        y0 = lifted_state(lift_n, setup, d / "lift_record.json")
        t0 = float(lift_n)
    else:
        _c0, y0 = core.initial_state(setup); t0 = 0.0
    n_max = float(np.log(setup.t_start / setup.t_gamma_end) + 2.0)
    log(f"{leg}: order {cfg['order']} y_max {cfg['y_max']} start N={t0} "
        f"caps calls={cfg['call_cap']} wall={cfg['wall_cap']}s")
    result = {"leg": leg, "config": cfg["order"], "y_max": cfg["y_max"], "start_N": t0,
              "caps": {k: cfg[k] for k in ("call_cap", "wall_cap")}, "lift_N": lift_n}
    solver = None
    try:
        solver = InstrumentedBDF(rhs, t0, y0, n_max, rtol=setup.rtol, atol=setup.atol,
                                 vectorized=False)
        result["init_raw_calls"] = sink.n_raw
        sink.solver = solver
        solver.attach_probe(sink)
        sink.log_factor("init", t0)
        while solver.status == "running":
            solver.step()
        result["solver_status"] = solver.status
    except CapReached as exc:
        log(f"{leg}: STOPPED BY CAP -- {exc}")
    except BaseException as exc:
        sink.stop_reason = f"exception:{type(exc).__name__}:{exc}"
        result["exception"] = sink.stop_reason
        log(f"{leg}: EXCEPTION (preserved) -- {type(exc).__name__}: {exc}")

    result["stop_reason"] = sink.stop_reason
    result["raw_calls_integration"] = sink.n_raw
    result["saved_states"] = sink.saved_states
    if sink.last_accepted:
        result["N_reached"] = sink.last_accepted[0]
    (d / "summary.json").write_text(json.dumps(result, indent=2))
    log(f"{leg}: integration done -- {sink.n_raw} calls, N={result.get('N_reached')}")

    # observations, budget-exempt, from the states saved to disk
    fac = np.asarray(getattr(solver, "jac_factor", np.full(y0.size, np.sqrt(EPS))), dtype=float) \
        if solver is not None else np.full(y0.size, np.sqrt(EPS))
    done = 0
    for c, name in sorted(sink.saved_states.items()):
        if done >= cfg["obs_jacobians"]:
            break
        try:
            z = np.load(d / name)
            J, f0 = obs_pinned_jacobian(rhs, sink, float(z["t"]), z["y"],
                                        d / f"obs_jac_{c}.npz", log)
            if cfg["ratcheted_recompute"]:
                ratcheted = np.where(fac > 10 * np.sqrt(EPS))[0]
                if ratcheted.size:
                    obs_ratcheted_columns(rhs, sink, float(z["t"]), z["y"], f0, fac,
                                          ratcheted[:60], d / f"ratcheted_cols_{c}.npz", log)
            done += 1
        except CapReached as exc:
            log(f"obs at {c}: stopped by wall cap -- {exc}"); break
        except BaseException as exc:
            log(f"obs at {c}: FAILED (preserved) -- {type(exc).__name__}: {exc}")
    sink.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=list(LEGS), required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--lift-n", type=float, default=None)
    args = ap.parse_args()
    out = Path(args.outroot); out.mkdir(parents=True, exist_ok=True)
    logf = (out / "driver.log").open("w", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n")

    criteria = load_sealed_criteria()
    checks = {"bdf": verify_bdf(), "rhs": instrumented_rhs.verify_verbatim()}

    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    pinned = git("rev-parse", INPUT_COMMIT)
    actual = {
        "frozen_module_sha256": hashlib.sha256((REPO / "src/rabbit/decoupling/_independent_noqke.py").read_bytes()).hexdigest(),
        "scipy": __import__("scipy").__version__, "numpy": np.__version__,
        "scipy_bdf_source_sha256": hashlib.sha256(
            Path(__import__("scipy.integrate._ivp.bdf", fromlist=["x"]).__file__).read_bytes()).hexdigest(),
    }
    mismatch = {k: (PINS[k], v) for k, v in actual.items() if PINS[k] != v}
    surfaces = ("src/", "scripts/", ".agent-harness/", ".codex/")
    is_anc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", pinned, "HEAD"]).returncode == 0
    moved = [p for p in git("diff", "--name-only", f"{pinned}..HEAD").splitlines() if p.startswith(surfaces)]
    dirty = [l for l in git("status", "--porcelain").splitlines() if l[3:].startswith(surfaces)]
    if mismatch or not is_anc or moved or dirty or not checks["bdf"]["verbatim_ok"] or not checks["rhs"]["verbatim_ok"]:
        (out / "FAIL_PROTOCOL.json").write_text(json.dumps(
            {"mismatch": mismatch, "ancestor": is_anc, "moved": moved, "dirty": dirty,
             "checks": checks}, indent=2))
        log("FAIL_PROTOCOL at pin verification"); raise SystemExit(2)

    st = manifest_selftest(out, args.leg, log)
    if not st["ok"]:
        (out / "FAIL_PROTOCOL.json").write_text(json.dumps(
            {"observation_manifest_selftest": st}, indent=2))
        log(f"FAIL_PROTOCOL: observation-manifest self-test failed: {st}"); raise SystemExit(2)
    log(f"selftest OK -- every named observation produced on the miniature")

    (out / "pins_verified.json").write_text(json.dumps(
        {"pins": actual, "pinned_commit": pinned, "head": git("rev-parse", "HEAD"),
         "criteria_sha256": CRITERIA_SHA, "criteria_items": len(criteria["items"]),
         "selftest": st, "checks": checks}, indent=2))

    res = run_leg(args.leg, out, args.lift_n, log)
    log(f"{args.leg} COMPLETE: {json.dumps({k: res.get(k) for k in ('stop_reason', 'raw_calls_integration', 'N_reached')})}")
    logf.close()


if __name__ == "__main__":
    main()
