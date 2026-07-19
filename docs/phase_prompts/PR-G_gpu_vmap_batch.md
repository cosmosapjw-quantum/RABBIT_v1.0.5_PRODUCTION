# PR-G — GPU vmap Batched Solve (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

37-DOF (or 25 post-PR-A) characteristic state is kernel-launch
bound on GPU.  CPU-preferred is the default runtime policy for
single solves.  GPU becomes useful only for large-batch inference
(N ≥ 64 breakeven, sweet spot ≥ 256).  **Rodas5P must stay**;
we add a vmap wrapper *on top* of the pure-JAX solve core.

Invariants: Rodas5P, CPU-preferred default remains, float64, all
existing single-solve entry points unchanged.

Dependencies: PR-A and PR-J strongly recommended (smaller state,
analytic Jacobian reduce per-element compute so batch amortisation
matters more).  Not strict.

---

## 1. Phase objective

Add `run_char_batch(sigmas, common_config)` (and equivalent
`run_char_batch_tier2`) that solves N independent
(`Σ_H`, possibly varying `η`, `τ_n`) configurations in a single
vmap'd Rodas5P pass.  Extend `_solve_core` in
`solver_jax_rodas5p.py` with an event-masked variant that permits
per-element termination under a `lax.while_loop` that continues
until **every** element has finished or `max_steps` is reached.

Target performance on GPU (measured, not estimated):
- N=1:   slower than CPU (expected)
- N=64:  on par
- N=256: 3–5× faster than CPU per solve
- N=1024: 10–20× faster

---

## 2. Literature anchors

### 2.1 Internal
- `src/rabbit/jax/solver_jax_rodas5p.py` —
  `_solve_core`, `_cached_solver_runner`.
- `src/rabbit/jax/driver_typeI_char.py` —
  `_get_char_rhs`, `_run_char_impl`.
- `docs/JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.2` — target design.

### 2.2 External
- JAX `lax.while_loop` semantics under `vmap`:
  <https://docs.jax.dev/en/latest/notebooks/control-flow.html>
  (confirm that `vmap(lax.while_loop(cond, body, init))` continues
  until the reducing `cond` across batch elements is false).
- XLA memory preallocation flags:
  <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>
  (for the VRAM-control env-var block in §3.4).

### 2.3 Paper-equation cross-check
None new.  This PR is pure infrastructure; it does not add or
change any physics formula.

---

## 3. Skeleton code

### 3.1 Event-masked solve core

```python
# solver_jax_rodas5p.py — new function
def _solve_core_event_masked(
    rhs_fn: Callable,
    jac_fn: Callable,
    y0_batch: jnp.ndarray,          # (N_batch, D)
    N_start: jnp.ndarray,           # scalar, shared
    N_end: jnp.ndarray,             # scalar, shared
    event_fn: Callable,             # (N, y) -> scalar; positive → not yet triggered
    rtol: float, atol: float,
    max_steps: int, h_init: float,
    h_min: float, h_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """vmap-friendly Rodas5P with per-element finish mask.

    Returns (y_final, N_final, success_mask, n_steps).  All vectors
    have leading batch dim except (n_steps) which is a scalar summary.

    Semantics:
      - Finished elements freeze (y_new = y_old, N_new = N_old).
      - The while_loop exits when every element has triggered OR
        n_steps exceeds max_steps.
      - Event refinement (bisection) is applied per element only
        if that element is transitioning from unfinished to finished
        on this step.
    """
    D = y0_batch.shape[1]; N_batch = y0_batch.shape[0]

    def cond_fn(carry):
        N_arr, y_batch, h, ep, n_steps, finished_mask, ev_prev = carry
        return jnp.logical_and(
            jnp.any(jnp.logical_not(finished_mask)),
            n_steps < max_steps,
        )

    def body_fn(carry):
        N_arr, y_batch, h, ep, n_steps, finished_mask, ev_prev = carry

        # per-element Rodas5P step
        def step_one(y, h_elem, J_elem):
            return _rodas5p_step(rhs_fn, N_arr, y, h_elem, J_elem, rtol, atol)
        J_batch = jax.vmap(jac_fn, in_axes=(None, 0))(N_arr, y_batch)
        y_new_batch, err_batch, _ = jax.vmap(step_one)(y_batch, h, J_batch)

        # Per-element event check
        ev_new = jax.vmap(event_fn, in_axes=(None, 0))(N_arr + h, y_new_batch)
        triggered = (ev_prev > 0.0) & (ev_new <= 0.0)

        # Freeze finished elements (no update)
        keep_mask = finished_mask[:, None]
        y_out_batch = jnp.where(keep_mask, y_batch, y_new_batch)

        # Per-element: if it just triggered, refine via bisection
        # (Simplified: bisection body omitted; full version mirrors
        # _refine_terminal_event but vectorised.)
        ...

        new_finished = finished_mask | triggered
        return (N_arr + h, y_out_batch, h_new, err_batch, n_steps + 1,
                new_finished, ev_new)

    init_state = (
        N_start, y0_batch, jnp.full((N_batch,), h_init),
        jnp.ones((N_batch,)), 0,
        jnp.zeros((N_batch,), dtype=jnp.bool_),
        jax.vmap(event_fn, in_axes=(None, 0))(N_start, y0_batch),
    )
    final = lax.while_loop(cond_fn, body_fn, init_state)
    _, y_final, _, _, n_steps, finished_mask, _ = final
    N_final = jnp.where(finished_mask, N_end, jnp.nan)
    return y_final, N_final, finished_mask, n_steps
```

### 3.2 Batch driver wrapper

```python
# driver_typeI_char.py — new top-level helper
def run_char_batch_tier1(
    sigmas: jnp.ndarray,
    *,
    correction_level: int = 0,
    N_mu: int = 12, N_q: int = 20,
    n_reactions: int = 12,
    tau_n: float = 878.4, eta: float = 6.104e-10,
    N_eff: float = 3.044, f_nu: float = 0.40520,
    T_start: float = 10.0, T_handoff: float = 0.08, T_end: float = 0.005,
    rtol: float = 1e-8, atol: float = 1e-10,
    max_steps: int = 2000, event_refine_steps: int = 24,
    device_policy: str = "cpu_preferred",
) -> dict:
    """Batched LRS tier-1 char solve across N Sigma_H_plus values.

    All non-sigma parameters are shared.  Returns a dict with
    observables batched over the leading dim:
        Y_p   : (N,)
        D/H   : (N,)
        Li7H  : (N,)
        Li6H  : (N,)
        N_eff : (N,)
        Xn_freeze : (N,)
        success : (N,) bool
    """
    # Build RHS/Jacobian once (shared across batch)
    rhs_p1, layout_p1 = _get_char_rhs(phase=1, correction_level=..., ...)
    rhs_p2, layout_p2 = _get_char_rhs(phase=2, correction_level=..., ...)

    # Construct batched y0 across sigmas
    y0_batch = _build_y0_batch_tier1(sigmas, layout_p1, ...)

    # Phase 1
    y1_batch, N1_batch, ok1, _ = _solve_core_event_masked(
        rhs_p1, jac_fn_p1, y0_batch, 0.0, 50.0, event_p1, ...
    )
    # Handoff
    y2_init = jax.vmap(lambda y1: _handoff_tier1(y1, layout_p2))(y1_batch)

    # Phase 2
    y2_batch, N2_batch, ok2, _ = _solve_core_event_masked(
        rhs_p2, jac_fn_p2, y2_init, N1_batch, 30.0 + N1_batch, event_p2, ...
    )
    return jax.vmap(_observables_tier1)(y2_batch) | {"success": ok1 & ok2}
```

Apply device policy with `jax.default_device(...)` at the entry,
same as the scalar driver.

### 3.3 Tier-2 batch

Analogous `run_char_batch_tier2` using `thermo_tier=2` in `_get_char_rhs`.

### 3.4 Env-var documentation

Add to the module docstring of `driver_typeI_char.py`:

```
GPU usage:
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    # or
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
    # then, in Python:
    from rabbit.jax.driver_typeI_char import run_char_batch_tier1
    res = run_char_batch_tier1(jnp.array([0.0, 0.05, 0.1, ...]),
                                device_policy="gpu_then_cpu_retry")
```

### 3.5 Unit tests

```python
# tests/test_pr_g_vmap_batch.py
import pytest
pytest.importorskip("jax")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from rabbit.jax.driver_typeI_char import (
    JAXTypeICharConfig, run_full_coupled_typeI_char_jax,
    run_char_batch_tier1,
)

def test_batch_matches_sequential():
    """N=8 vmap'd solve must be bitwise identical to N sequential solves."""
    sigmas = jnp.array([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    seq = [run_full_coupled_typeI_char_jax(JAXTypeICharConfig(
        Sigma_H_plus=float(s), correction_level=0, N_q=20, N_mu=12,
        n_reactions=12, thermo_tier=1,
    )) for s in sigmas]
    batch = run_char_batch_tier1(sigmas, correction_level=0)
    for k, (s_scalar, s_batch) in enumerate(zip([r.Yp for r in seq], batch["Yp"])):
        assert abs(s_scalar - float(s_batch)) < 1e-12

def test_batch_tier2_matches_sequential():
    ...

def test_device_policy_honoured():
    """cpu_preferred must not touch the GPU even if one is present."""
    ...

def test_breakeven_benchmark_cpu():
    """At N=64 CPU, batch should not exceed N × scalar cost."""
    ...
```

---

## 4. WBS

1. **Event-masked solve core** in `solver_jax_rodas5p.py`.
2. **Event refinement under vmap** — bisection on the cubic-Hermite
   dense output for transitioning elements only.
3. **`run_char_batch_tier1`** + **`run_char_batch_tier2`**.
4. **Batch bitwise-parity test** (sequential vs batched).
5. **Breakeven benchmark on CPU** (required).
6. **Breakeven benchmark on GPU** (if GPU is available in CI;
   otherwise record "deferred" in audit).
7. **Docs update**: env-var block; GPU optimisation plan §2.2
   status transition.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read `_solve_core` (pure-JAX path) in
  `solver_jax_rodas5p.py` and confirm it is already vmap-compatible
  (returns only jnp arrays; no Python type conversions inside).
- Confirm `_cached_solver_runner` caches by `id(rhs_fn)` and that
  using the new masked core bypasses the existing cache cleanly
  (keep the existing cache for scalar entries; batch uses its own).
- Grep for `jax.default_device` in `driver_typeI.py` /
  `driver_typeI_char.py` to replicate the device-policy wrapper.

### Stage 2 — External
- `WebSearch "jax vmap lax.while_loop termination semantics"` —
  confirm exit-when-all-done behaviour is correct; document pitfall
  if any element remains "active" due to an event never triggering.
- `WebSearch "XLA_PYTHON_CLIENT_PREALLOCATE false OOM mitigation"`
  — confirm env vars are stable across JAX versions.

### Stage 3 — Self CoT
- **Worst-case** step count: the slowest batch element dictates
  total steps.  If element 0 converges in 200 steps and element N
  in 600 steps, the batch runs 600 steps even though most elements
  are frozen.  Total cost ≈ 600 × N vs 200+…+600 ≈ sum if sequential.
  Quantify the "slowest-element tax".
- **Correctness** of the frozen branch: after an element triggers,
  its state must not change.  Test by running a batch where one
  element finishes at step 200 and the other at step 600, then
  verify the first element's final state at the end of step 600
  matches its state at step 200.
- **Event refinement under vmap**: bisection costs `refine_steps`
  RHS evaluations per element.  Under vmap, finished elements
  must skip the bisection (no-op) — verify via a `jnp.where` inside
  the refinement loop.

Record in `docs/audit/PR-G_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] Batch vs sequential bitwise parity (≤ 1e-12) at N=8.
- [ ] Tier-2 batch parity.
- [ ] CPU breakeven benchmark matrix filled.
- [ ] GPU breakeven (or deferred note).
- [ ] `cpu_preferred` single-solve performance unchanged.
- [ ] No regression in existing tier-1/tier-2 parity tests.

---

## 7. Adversarial audit prompt

> Audit PR-G (vmap batch solve).  Verify:
> (1) bitwise parity between scalar solve and batch solve at N=8
> (tier-1 and tier-2);
> (2) finished-element state frozen correctly (no drift after event
> triggers);
> (3) `cpu_preferred` single-solve unchanged in timing;
> (4) Rodas5P step structure intact (no substitution of a non-stiff
> tableau);
> (5) VRAM under 2 GB at N=256 with mem_fraction=0.10.  Cap 400 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** swap Rodas5P for an explicit (non-stiff) batched solver.
2. **Do not** remove the scalar `_solve_core` / `jax_rodas5p_solve`
   entries.  Backward compatibility is required.
3. **Do not** attempt event refinement before the masked-while_loop
   core is bitwise-parity-verified.
4. **Do not** use the slowest-element tax as a reason to drop
   events.  Drop events only for runs where `T_handoff` is fixed
   a priori (optional performance path, not default).

---

## 9. Hallucination prevention
- Do not invent JAX primitives (`jax.batched_while_loop` does not
  exist — vmap of `lax.while_loop` is the supported idiom).
- Do not claim GPU speedup without measuring.  If the CI runner has
  no GPU, write "deferred to GPU benchmark rig" in the audit.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md §3.1`
Add a bullet: "Batch mode via `run_char_batch_tier{1,2}` (PR-G) —
CPU remains preferred at N ≤ 64, GPU begins to pay off at N ≥ 256
with env-var preallocation override."

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-G entry with breakeven measurement table.

### 10.3 `docs/JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.2`
Transition from "planned" to "delivered in PR-G"; embed the
breakeven numbers.

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_g_vmap_batch.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v

test -f docs/audit/PR-G.md
test -f docs/audit/PR-G_stage1.md
test -f docs/audit/PR-G_stage2.md
test -f docs/audit/PR-G_stage3.md
test -f docs/audit/PR-G_breakeven.md
git add docs/audit/PR-G*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-G: vmap-batched Rodas5P solve for GPU throughput

Adds _solve_core_event_masked (pure-JAX, vmap-safe, per-element
finish mask) to src/rabbit/jax/solver_jax_rodas5p.py, and
run_char_batch_tier{1,2} dispatchers on top of
driver_typeI_char.py.  Bitwise parity with scalar solves at N=8
enforced as a regression.  CPU single-solve path and invariants
unchanged.  GPU utility begins at N>=64 per breakeven measurement
recorded in docs/audit/PR-G_breakeven.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Bitwise parity at N=8 fails (even one element disagrees > 1e-12).
- Scalar-path warm-run timing regresses.
- Event refinement under vmap produces NaN for any finished
  element.
- GPU peak VRAM > 2 GB at `MEM_FRACTION=0.10, N=256`.

Abort → `docs/audit/PR-G_abort.md`.
