# PR-J — Analytic Jacobian Blocks (Phase Prompt)

> Feed this document verbatim to a fresh Claude session.  Shared
> framework: [README.md](README.md).

---

## 0. Load-bearing project context (verbatim)

RABBIT JAX characteristic driver (LRS, tier 1 & 2):
`src/rabbit/jax/driver_typeI_char.py`.  Integrator: **Rodas5P**
(do not replace).  Rodas5P requires a Jacobian at every accepted
step.  `jax.jacfwd(rhs, argnums=1)` performs `dim(state)` forward
RHS evaluations per Jacobian — at 25 DOF (post-PR-A) this is the
dominant single-solve cost.

Invariants: Rodas5P stays, CPU-preferred default, float64 state,
stable-identity RHS cache, transported monopole in weak rates,
publication parity target |ΔY_p| < 5 × 10⁻⁵.

Depends on: **PR-A completed**.  Pre-condition check: state dim
must be 25 (tier-1 phase-2) / 27 (tier-2 phase-2).  If not, stop.

---

## 1. Phase objective

Replace `jax.jacfwd(rhs, argnums=1)` inside
`src/rabbit/jax/solver_jax_rodas5p.py::_cached_jacfwd` (as used by
the characteristic driver) with a hand-assembled analytic Jacobian
that exploits the known block structure of the characteristic RHS.

Target: per-step Jacobian cost drops from ~25 forward RHS evals to
~1 effective RHS eval + closed-form algebra.  Warm single-solve
time target: ≤ 65 %% of the post-PR-A baseline.

---

## 2. Literature anchors

### 2.1 Internal
- RHS body: `src/rabbit/jax/driver_typeI_char.py::_rhs_core`.
- Ray map: `src/rabbit/jax/characteristic_rays_jax.py` —
  `mu_current_jax`, `jacobian_jax` (from PR-A), `P2_jax`,
  `characteristic_rhs_jax`, `extract_stress_jax`,
  `extract_monopole_jax`.
- Rodas5P: `src/rabbit/jax/solver_jax_rodas5p.py`, especially
  `_cached_jacfwd`, `_rodas5p_step`, `_rodas5p_step_schur`.
- Paper eq (54–56) — `dI/dN, dJ/dN, dS/dN`.
- Paper eq (57) — `Π_+ = f_ν Σ_j w_j J_j P_2(μ_j) e^{-8 I_j}`.
- Paper eq (58) — `f̃_0(q) = ½ Σ_j w_j J_j f_FD(q e^{2 I_j})`.
- Paper eq (147) — `dΣ_+/dN = -(2-q) Σ_+ + Π_+`.

### 2.2 External
- Rosenbrock–Wanner linear-algebra costs: Hairer & Wanner, *Solving
  ODEs II: Stiff and DAE Problems*, Chapter IV.8.  Confirm that the
  Rosenbrock step only needs `J` once per accepted step (not per
  stage).
- JAX linearisation primitives: `jax.vjp`, `jax.jvp`, `jax.linearize`.

### 2.3 Paper-equation cross-check
- [ ] Eq 47: `d ln E / dN = -1 - Σ P_2(μ)` — used for I, drop the
      -1 (already in Hubble expansion).
- [ ] Eq 51 (from PR-A): analytic `J(S, μ₀)` closes the block.
- [ ] Eq 57: stress formula.  Its derivative w.r.t. `I_j` gives
      the `∂(dΣ_+)/∂I_j` block.

---

## 3. Skeleton code

### 3.1 Block derivations (pseudocode, verify each symbolically)

Let `D = dim(y) = 25` at tier-1 phase-2.  The Jacobian `J[D, D]` is
block-structured as:

```
rows\cols    Σ_+   Σ_-   I_j     S        T_γ   X_i
Σ_+        a_pp   0    ∂Π/∂I   ∂Π/∂S    0     0
Σ_-         ×    a_mm   0       0         0     0
I_j       b_j     0     0      c_j         0    0
S           1     0     0       0          0    0
T_γ         0     0     0       0        d_T    0
X_i       e_i     0    f_ij    g_i       h_i   J_XX
```

Closed-form blocks:

```python
# ── geometry row ──
# a_pp = -(1 - Σ²) + 2 Σ_+·Σ_+ = 2Σ_+² - 1 + Σ² = 3Σ_+² + Σ_-² - 1
# a_mm = -(1 - Σ²) + 2 Σ_-·Σ_- = Σ_+² + 3Σ_-² - 1
a_pp = 3.0 * Sigma_plus**2 + Sigma_minus**2 - 1.0
a_mm = Sigma_plus**2 + 3.0 * Sigma_minus**2 - 1.0

# ∂Π_+/∂I_j = -8 f_ν w_j J_j P_2(μ_j) e^{-8 I_j}
# (derivative of Π_+ = f_ν Σ_j w_j J_j P_2 e^{-8I_j})
dPi_dI = -8.0 * f_nu * w0 * J_vals * P2 * jnp.exp(-8.0 * I_vals)

# ∂Π_+/∂S combines ∂μ/∂S, ∂J/∂S, ∂P_2/∂μ:
#   μ(S) = sign(μ₀) √(X0 e^{6S}/(1 + X0 e^{6S}))
#   ∂μ/∂S = 3 μ (1 - μ²)                              (from dμ/dN = 3Σ μ(1-μ²), dS/dN=Σ)
#   J(S, μ₀) = e^{-6S} (1-μ₀²)² / (1-μ²)²
#   ∂J/∂S = J · (-6 + 4 μ (∂μ/∂S) / (1-μ²))
#         = J · (-6 + 12 μ²)                           (substituting)
#         = 6 J (2 μ² - 1) = 6 J · (P_2(μ) − 1/2) · 2
#   ∂P_2/∂μ = 3 μ
#
# So ∂Π_+/∂S = f_ν Σ_j w_j e^{-8I_j} · [ (∂J/∂S) P_2 + J · 3μ · (∂μ/∂S) ]
dJ_dS = 6.0 * J_vals * (2.0 * mu * mu - 1.0)
dmu_dS = 3.0 * mu * (1.0 - mu * mu)
exp_m8I = jnp.exp(-8.0 * I_vals)
dPi_dS = f_nu * jnp.sum(w0 * exp_m8I * (dJ_dS * P2 + J_vals * 3.0 * mu * dmu_dS))

# ── transport row ──
# b_j = ∂(dI_j)/∂Σ_+ = P_2(μ_j)
# c_j = ∂(dI_j)/∂S = Σ_+ · ∂P_2/∂μ · ∂μ/∂S = Σ_+ · 3μ_j · 3μ_j (1-μ_j²)
#     = 9 Σ_+ μ_j² (1 - μ_j²)
b_j = P2
c_j = 9.0 * Sigma_plus * mu * mu * (1.0 - mu * mu)

# ── S row ──
# dS/dN = Σ_+  ⇒  ∂(dS)/∂Σ_+ = 1, all others 0

# ── T_γ row ──  (tier-dependent)
# tier-1: dT_γ/dN depends only on T_γ via tier1_dT_gamma_dN_jax
# tier-2: coupled 3T, nontrivial (T_γ, T_νₑ, T_νₓ) block
# Implement via targeted jax.jvp on a 1-dim (or 3-dim) sub-state,
# NOT full jacfwd.  The thermo block is 1×1 (tier-1) or 3×3 (tier-2).
```

The network (X) row is the only non-trivial block.  `dX/dN` depends
on `X` (the stoichiometric Jacobian), `T_γ`, weak rates `λ_np, λ_pn`,
and therefore on `I_j, S` via the monopole `f̃_0(q)`.

### 3.2 Fallback for the X-row

Full analytic derivation of `∂(dX)/∂I_j` requires chain-ruling
through `f̃_0(q)` and into `compute_live_rates_from_monopoles_cl*_jax`.
This is several hundred LOC and risky.  Recommended approach:
**targeted `jax.vjp` for just the X rows**, which costs
`n_species = 9` reverse-mode passes — still 3× cheaper than the
previous `jacfwd` (25 forward passes).

```python
# Assemble the X-row with jax.vjp
#   _, f_vjp = jax.vjp(rhs_fn, N, y)
#   e_net_rows = jnp.eye(state_dim)[layout["i_net"]:layout["i_net"]+n_species]
#   rows = jax.vmap(lambda e: f_vjp((0.0, e))[1])(e_net_rows)
# rows[k] is ∂(dy_{i_net+k})/∂y — use its slices to fill the X row.
```

### 3.3 Analytic Jacobian assembler

```python
# New helper in solver_jax_rodas5p.py
def build_analytic_jac_fn(
    rhs_fn: Callable,
    layout: dict,
    N_mu: int,
    n_species: int,
    thermo_tier: int,
    aux_weak_jac_fn: Callable,   # from driver, handles the X-row via vjp
) -> Callable:
    """Return a jitted (N, y) -> J function that assembles the
    Jacobian block-by-block using closed-form expressions for the
    geometry/transport/S blocks and targeted vjp for the network
    row.

    Keeps the same (D, D) output shape as jax.jacfwd so that
    _rodas5p_step and _rodas5p_step_schur are drop-in compatible.
    """
    i_Sp = layout["i_Sp"]; i_Sm = layout["i_Sm"]
    i_I = layout["i_I"]; i_S = layout["i_S"]
    i_tg = layout["i_tg"]; i_net = layout["i_net"]

    @jax.jit
    def jac_fn(N, y):
        # 1. Extract state
        Sigma_plus = y[i_Sp]; Sigma_minus = y[i_Sm]
        I_vals = jax.lax.dynamic_slice(y, (i_I,), (N_mu,))
        S_val = y[i_S]
        # 2. Evaluate analytic pieces (μ, J, P_2, e^{-8I})
        # 3. Assemble D×D matrix J_out by placing blocks via
        #    jax.lax.dynamic_update_slice.
        # 4. For the X-row, call aux_weak_jac_fn(N, y) and paste.
        return J_out
    return jac_fn
```

### 3.4 Driver wiring

In `_get_char_rhs`, after compiling the RHS, also compile the
analytic Jacobian and attach it:

```python
from rabbit.jax.solver_jax_rodas5p import jax_rodas5p_solve
# Build analytic Jacobian
jac_fn = build_analytic_jac_fn(compiled, layout, N_mu, n_species,
                                 thermo_tier, aux_weak_jac_fn)
# Pass as `jac_fn` parameter to jax_rodas5p_solve
```

`jax_rodas5p_solve` already accepts a `jac_fn` via
`_cached_solver_runner(... jac_fn=...)`.  No solver-level change
is needed.

### 3.5 Unit test — elementwise jacfwd parity

```python
# tests/test_pr_j_analytic_jacobian.py
"""PR-J: analytic Jacobian agrees elementwise with jacfwd."""
import pytest
pytest.importorskip("jax")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def _build_state(N_mu, n_species, thermo_tier, sigma_plus, T_gamma):
    # minimal y that respects the layout
    ...

@pytest.mark.parametrize("tier", [1, 2])
@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("phase", [1, 2])
def test_analytic_jac_matches_jacfwd(tier, sigma, phase):
    from rabbit.jax.driver_typeI_char import _get_char_rhs, _char_layout
    N_mu, N_q = 12, 20
    n_species = 2 if phase == 1 else 9
    rhs, layout = _get_char_rhs(
        phase=phase, correction_level=0, thermo_tier=tier,
        N_mu=N_mu, N_q=N_q, n_species=n_species,
        n_reactions=12, tau_n=878.4, eta=6.104e-10,
        N_eff=3.044, f_nu=0.40520,
    )
    y = _build_state(N_mu, n_species, tier, sigma, T_gamma=1.0)
    J_fwd = np.asarray(jax.jacfwd(rhs, argnums=1)(jnp.float64(0.0), y))
    # analytic Jacobian, built as described in §3.3
    from rabbit.jax.solver_jax_rodas5p import build_analytic_jac_fn
    # (build with the same layout and aux_weak_jac_fn as the driver)
    ...
    J_ana = np.asarray(jac(jnp.float64(0.0), y))
    max_abs = float(np.max(np.abs(J_fwd - J_ana)))
    max_rel = float(np.max(np.abs((J_fwd - J_ana) / np.maximum(np.abs(J_fwd), 1e-30))))
    assert max_abs < 1e-10 and max_rel < 1e-7, (
        f"tier={tier}, Σ={sigma}, phase={phase}: "
        f"max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
    )
```

---

## 4. WBS

1. **Block derivations** (offline / in the Stage-3 CoT).  Write each
   block to `docs/audit/PR-J_derivations.md`.
2. **Implement `jacobian_jax` dependencies**: `∂μ/∂S`, `∂J/∂S`
   helpers in `characteristic_rays_jax.py`.
3. **Implement `build_analytic_jac_fn`** in `solver_jax_rodas5p.py`.
4. **Hybrid X-row via vjp**.
5. **Driver wiring** in `_get_char_rhs` — attach the analytic
   Jacobian; retain `jacfwd` as a debug fallback behind a config
   flag `jacobian_impl: "analytic" | "jacfwd"` (default
   `"analytic"`).
6. **Elementwise unit test** (§3.5).
7. **Parity sweep**: rerun tier-1 and tier-2 parity grids.
8. **Performance measurement**: warm 1-solve before and after.
9. **Documentation updates**.

---

## 5. Three-stage verification protocol

### Stage 1 — Internal

1. Read `RABBIT_report_H1_revision_typo_pass2.pdf` pages 16–18 for
   eq 47/48 (direction equations) and eq 57/58 (observable
   extractors).
2. Read `solver_jax_rodas5p.py` lines 51–141 (block-sparse
   assembler) and confirm that `_rodas5p_step` and
   `_rodas5p_step_schur` treat `J` as an opaque `(D, D)` array —
   drop-in replacement is therefore safe.
3. Grep for call sites of `_cached_jacfwd`:
   ```
   Grep "_cached_jacfwd" src/rabbit/jax/
   ```
   Confirm that the solver-runner cache path (`_cached_solver_runner`)
   already accepts an external `jac_fn` and will route to the
   analytic implementation.

Record verdicts in `docs/audit/PR-J_stage1.md`.

### Stage 2 — External

1. `WebSearch "Rosenbrock Wanner stiff solver analytic jacobian
   block structure Kaps Rentrop"` — confirm that swapping autodiff
   Jacobian for an analytic one preserves the Rosenbrock order
   (requires only `J ≈ ∂f/∂y` within the method's tolerance, no
   exactness requirement).
2. `WebSearch "jax.jvp vs jax.jacfwd cost forward-mode"` — confirm
   that `jax.jvp` with a single tangent is O(1 RHS eval) whereas
   `jacfwd` with D tangents is O(D evals).
3. Cross-check the derivation of `dP_2/dμ = 3μ` (trivial) and
   `dμ/dN = 3Σ μ(1-μ²)` (paper eq 48).

Record verdicts in `docs/audit/PR-J_stage2.md`.

### Stage 3 — Self CoT

1. **Symbolic verification** of every block in §3.1.  Do each
   derivative by hand and record the intermediate steps.  Pay
   particular attention to:
   - Sign of `dJ/dS` (`+6 J (2μ² − 1)` vs `−6 J (1 − 2μ²)`: same).
   - `dμ/dS = 3μ(1−μ²)` (from `dμ/dN = 3Σμ(1−μ²)`, `dS/dN = Σ`).
2. **Dimensional analysis**: every block entry must be dimensionless
   (state is dimensionless, RHS is `state/efold`, efold is
   dimensionless — so `J[i,j]` is dimensionless).
3. **NaN audit**:
   - `(1 - μ²) → 0` guard inherited from `jacobian_jax`.
   - `1e-30` floors on `H_inv_s` etc. already in RHS.
4. **Performance self-check**: at 25 DOF, `jacfwd` = 25 RHS evals.
   Analytic + 9-vjp fallback ≈ 1 RHS eval + 9 reverse-mode = ~10
   RHS-equivalent evals.  Expected speed-up ~2.5×.  Sanity: if
   measured speed-up exceeds 10×, something else has changed — stop
   and investigate.

Record verdicts in `docs/audit/PR-J_stage3.md`.

---

## 6. Self-audit checklist

From [ROADMAP_SELF_AUDIT.md §2](../ROADMAP_SELF_AUDIT.md#2-per-pr-audit-checklist):

- [ ] Elementwise parity `|J_analytic - J_jacfwd| < 1e-10` across
      tier × Σ × phase grid.
- [ ] Tier-1 parity tests green.
- [ ] Tier-2 parity tests green.
- [ ] Warm 1-solve time ≤ 65 %% of post-PR-A baseline.
- [ ] `jacobian_impl="jacfwd"` fallback still works (regression).
- [ ] Paper eq (47–48) provenance recorded.

---

## 7. Adversarial audit handoff

Prompt:
> Audit PR-J (analytic Jacobian for the JAX char driver).
> Specifically verify:
> (1) every analytic block in `build_analytic_jac_fn` matches
> `jax.jacfwd` output elementwise at 1e-10;
> (2) the X-row fallback via `vjp` has `n_species = 9` cost and not
> more;
> (3) no sign error in `dJ/dS` (+6J(2μ²−1), not −);
> (4) Rodas5P step structure is unchanged (same (D,D) interface);
> (5) tier-1 and tier-2 publication parity is preserved within
> 5e-7.  Report issues ranked by severity.  Cap 400 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** hand-derive the X-row.  Use `jax.vjp` fallback.
2. **Do not** skip the elementwise jacfwd comparison "because
   parity passes".  A subtle sign error can cancel on the
   observable and fail under finite-difference gradients later.
3. **Do** keep a `jacobian_impl="jacfwd"` flag for fast debugging.

---

## 9. Hallucination prevention

- Do not copy derivative expressions from memory; re-derive in
  Stage 3 CoT.
- Do not write a function `jnp.derivative_of_...`; no such JAX API
  exists.  Use `jax.jvp` / `jax.vjp` / `jax.grad` explicitly.
- Do not claim "unchanged parity" without running
  `pytest tests/test_jax_typeI_characteristic_parity.py -v`.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §3.3: update the block-sparse discussion — at 25 DOF post-PR-A
  and with analytic Jacobian, revisit the dense vs block-sparse
  comparison (dense may now be 1 RHS + 9 vjp whereas block-sparse
  Schur adds its own overhead; record measurement).

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-J entry.

### 10.3 `docs/JAX_CHAR_GPU_OPTIMIZATION_PLAN.md`
§2.4 transitions from "planned" to "delivered in PR-J".

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_j_analytic_jacobian.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v
pytest tests/ -m "not slow and not gpu" --tb=no -q

test -f docs/audit/PR-J.md
test -f docs/audit/PR-J_stage1.md
test -f docs/audit/PR-J_stage2.md
test -f docs/audit/PR-J_stage3.md
test -f docs/audit/PR-J_derivations.md
git add docs/audit/PR-J*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-J: analytic Jacobian blocks for the JAX characteristic driver

Replaces jax.jacfwd with hand-assembled closed-form Jacobian blocks
for the geometry, transport, S, and thermo rows, and a targeted
jax.vjp fallback for the network (X) row. Elementwise agreement
with jacfwd verified to 1e-10. Tier-1 and tier-2 publication parity
preserved. Warm single-solve time reduced by targeting cost to
approximately 1 RHS + 9 vjp per accepted step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Elementwise jacfwd parity worse than 1e-10 anywhere.
- Any sign error uncovered in Stage 3.
- Tier-1 or tier-2 publication parity regression beyond 5e-7.
- Performance regression instead of improvement.
- Adversarial audit flags a high-severity issue.

Abort note → `docs/audit/PR-J_abort.md`.
