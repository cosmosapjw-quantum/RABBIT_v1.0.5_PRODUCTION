# BD616 — clean-core backend bake-off (E5) + Rust/Rodas5P decision

Date: 2026-07-09

Status: **IMPLEMENTED** (probe script + smoke test + decision run). Two
decisions recorded, both evidence-backed:
- Rust collision-kernel lane (PR5a): **PROCEED — but see the honest reframing**;
  the strategic call is deferred to the user (below).
- Rodas5P production promotion (PR5b): **BLOCKED by the wall-time gate**; BDF
  stays the production default.

## What was measured

`scripts/probe_clean_core_backend_bakeoff.py` (new; the E5 script BD611 §176-209
specified but that did not yet exist). Same physics case, three levels, compile
separated from warm steady-state, `JAX_PLATFORMS=cpu`, warmups discarded,
min/median/max over 5 reps. Command:

```
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
  venv/bin/python scripts/probe_clean_core_backend_bakeoff.py \
  --n-q 16 24 --repeat 5 --warmups 2 --rodas5p \
  --out audit_outputs/bd616_backend_bakeoff.json
```

Full JSON: `audit_outputs/bd616_backend_bakeoff.json`.

### Kernel level (median wall, n_q=24)
| kernel | numpy (Python triple-loop) | JAX (compiled) | ratio | JAX compile |
|---|---:|---:|---:|---:|
| nu_e_scatter | 5.01e-1 s | 5.37e-5 s | 9341x | 0.106 s |
| pair_process | 4.64e-1 s | 9.05e-5 s | 5125x | 0.335 s |

### Endpoint level (full solve wall, N_q, single process, peak RSS ~0.1 GB)
| method | n_q=16 | n_q=24 | Yp (n_q=24) | D/H (n_q=24) |
|---|---:|---:|---:|---:|
| BDF (baseline) | 1.19 s | 1.24 s | 0.2425313 | 2.4899e-05 |
| RODAS5P (adapter) | 5.93 s | 5.71 s | 0.2425313 | 2.4899e-05 |

## Decision 1 — Rust kernel lane (PR5a)

The literal E5 rule (same-case compiled-over-numpy speedup ≥ 3x) is met by a
huge margin (5000–9000x) → mechanical verdict `proceed`. **But the ratio is
JAX-over-numpy, not Rust-over-JAX**, and that reframes the decision:

- The 5000–9000x figure only says the collision kernel *must be compiled* — a
  Python triple-loop is untenable. **JAX already captures this win in
  production.** A Rust kernel would run at roughly JAX's runtime (≈5–9e-5 s),
  i.e. Rust-vs-JAX ≈ 1x on raw speed.
- Amdahl: collisions are ~33% of the endpoint wall (BD591), so even an infinite
  kernel speedup caps the endpoint gain at **~1.49x**. A kernel win is not an
  endpoint win.
- Therefore Rust's real value is **not raw speed**. It is: (a) dropping the JAX
  dependency (the meta-goal of this whole line of work), (b) eliminating JAX's
  per-shape compile latency (measured 0.1–0.34 s, paid on every new grid shape),
  and (c) collapsing the numpy/JAX twin-maintenance surface that BD612 F-1
  showed forces dual-lane edits.

Because the raw-speed justification is marginal and the true justification is the
strategic JAX-removal choice, **building the Rust crate is a scope/strategy call
for the user, not a mechanical "proceed"** — surfaced rather than executed
autonomously. If pursued, PR5a builds a separate maturin/PyO3 `rabbit_kernels`
crate (main setuptools package untouched, fail-loud dispatch, third parity lane),
per the approved plan.

## Decision 2 — Rodas5P promotion (PR5b): BLOCKED

Endpoint *parity* is excellent — RODAS5P matches BDF to Yp 7 digits
(|ΔY_p|=4.6e-9 at the smoke config, identical to display precision at n_q=16/24).
But the **wall-time gate fails**: RODAS5P is **4.6–5.0x slower** than BDF
(5.9 s vs 1.2 s at n_q=16). Root cause is exactly as the plan predicted: the
solver's dense FD Jacobian costs N+1 full-RHS evaluations per step and is
recomputed every step (`jac_reuse_max_steps=1`, the exact-Rosenbrock default),
and the coupled RHS is expensive (collisions + network).

Per the promotion gate (BD615 / plan PR5b), a failed wall gate means **do not
flip the default** — `PRODUCTION_CONFIG.method` stays BDF. The RODAS5P lane
remains available, opt-in, and parity-locked for future work.

**Identified unblock lever (future, gated):** `jac_reuse_max_steps > 1` reuses a
Jacobian across accepted steps and directly attacks the dominant cost (the FD
Jacobian RHS budget). It is an approximation (Rosenbrock assumes an exact per-step
Jacobian), so it must clear its own endpoint-parity + wall gate before any preset
enables it, and it needs `SolverConfig → adapter` plumbing that PR3 deliberately
left out. Not attempted here.

## Validation

```
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_bakeoff_probe_smoke.py
  -> 1 passed in 2.36s
decision run: completed, peak RSS ~0.1 GB, no memory growth across n_q=16,24 x (kernel+rhs+endpoint x BDF,RODAS5P).
```

## Cost line

- added_lines: ~320 (probe script + smoke test + this note + artifact)
- deleted_lines: 0
- files_touched: 1 script[new] + 1 test[new] + 1 note + 1 audit_outputs artifact
- runtime_behavior_changed: no (measurement only)
- physics_behavior_changed: no
- known_blocker_reduced: yes (the backend-migration decision is now measured, not
  felt: Rust=strategic-not-speed, Rodas5P-promotion=blocked-on-wall)
- blocker_movement_ratio: 0.5
- validation_strengthened: yes (E5 probe + smoke lock)
- cost_effectiveness_verdict: ACCEPT
