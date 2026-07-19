# PR-A — Analytic J_j Elimination (Phase Prompt)

> **How to use.** Feed this document verbatim to a fresh Claude
> session.  It is self-contained — it does not require reading the
> other phase prompts.  Shared context is in
> [README.md](README.md); the specific sections are copied below.

---

## 0. Load-bearing project context (verbatim)

RABBIT computes BBN observables on an anisotropic Bianchi Type I
background.  Physics reference: `RABBIT_report_H1_revision_typo_pass2.pdf`.

- **Integrator.**  Rodas5P (Steinebach 2023) — **do not replace**.
- **Characteristic driver** (publication grade, LRS, tier 1 & 2):
  `src/rabbit/jax/driver_typeI_char.py`.
- **State layout today** (37 DOF, tier 1, N_μ=12, n_species=9):
  `[Σ_+, Σ_-, I_0..I_11, J_0..J_11, S, T_γ, X_0..X_8]`.
- **Publication parity:** |ΔY_p| ≤ 4 × 10⁻⁸ across Σ_H ∈ [0, 0.5].
- **CPU-preferred default.**  GPU disabled except for batch ≥ 64.
- **float64 state vectors** (dY_p/dη ~ 10⁸; mixed precision banned).

Forbidden operations during this PR:
- Replacing Rodas5P.
- Making up paper equation numbers without reading the PDF.
- Mocking the `J_j` analytic formula with a "close-enough" placeholder.
- Claiming parity without running the tier-1 and tier-2 parity suites.

---

## 1. Phase objective

Replace the numerically-evolved angular Jacobian `J_j` in the
characteristic driver with its closed-form expression in `S` and
`μ_{j,0}`:

```
J_j(N) = exp(-6 S(N)) · (1 - μ_{j,0}²)² / (1 - μ_j(N)²)²
```

This is paper eq (51).  It is the **exact solution** of the ODE
`dJ_j/dN = 3 Σ (1 - 3 μ_j²) J_j` given the analytic direction map
`μ_j(S) = sign(μ_{j,0}) · √(X_{j,0} e^{6S} / (1 + X_{j,0} e^{6S}))`
where `X_{j,0} = μ_{j,0}² / (1 - μ_{j,0}²)`.

State-vector consequence: the `J_j` block (12 DOF at N_μ = 12) is
removed.  State drops from **37 → 25 DOF** at tier-1 phase-2,
**39 → 27** at tier-2 phase-2.

---

## 2. Literature anchors

### 2.1 Internal (must verify at runtime)
- Paper eq (51) — `RABBIT_report_H1_revision_typo_pass2.pdf`, page 17
  (appears right after the analytic direction integration).
- Paper §6.5 — "ODEs for the Ray State": justifies why `J_j` was
  originally carried as an ODE variable.
- SciPy implementation: `src/rabbit/transport/characteristic_rays.py`
  — functions `mu_current`, `characteristic_transport_rhs`,
  `extract_stress`, `extract_monopole`.
- JAX implementation:
  `src/rabbit/jax/characteristic_rays_jax.py` — functions
  `mu_current_jax`, `characteristic_rhs_jax`, `extract_stress_jax`,
  `extract_monopole_jax`.
- JAX driver:
  `src/rabbit/jax/driver_typeI_char.py` — functions `_char_layout`,
  `_rhs_core`, `_run_char_impl`, `_char_active_indices`.

### 2.2 External (must verify via web search)
Not required for this PR — the formula is internal to the paper
and trivially derivable.  Record one web search that confirms
"characteristic-ray BBN Jacobian J(S)" literature consistency; no
contradictory result should be found.

### 2.3 Paper-equation cross-check checklist
- [ ] Quote the literal text of paper eq (51) from the PDF.
- [ ] Confirm the exponent is `-6 S`, not `+6 S` or `-3 S`.
- [ ] Confirm the numerator is `(1 - μ_{j,0}²)²` (squared).
- [ ] Confirm the denominator is `(1 - μ_j(N)²)²` (squared).

---

## 3. Skeleton code

### 3.1 New helper in `characteristic_rays_jax.py`

```python
@jax.jit
def jacobian_jax(X0: jnp.ndarray, S: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Closed-form per-ray angular Jacobian.

    Paper eq (51):  J_j = e^{-6 S} · (1 - μ_{j,0}²)² / (1 - μ_j²)²

    Using X0 = μ_{j,0}² / (1 - μ_{j,0}²)  so that
    1 - μ_{j,0}² = 1 / (1 + X0).

    Parameters
    ----------
    X0  : (N_mu,)  precomputed μ_{j,0}² / (1 - μ_{j,0}²)
    S   : ()       accumulated shear integral ∫₀ᴺ Σ_+(N') dN'
    mu  : (N_mu,)  current direction cosines (from mu_current_jax)
    """
    one_minus_mu0_sq = 1.0 / (1.0 + X0)                  # (N_mu,)
    one_minus_mu_sq  = 1.0 - mu * mu                     # (N_mu,)
    # Floor denominator to avoid division-by-zero near μ → ±1.
    # Gauss–Legendre never places a node at |μ|=1, so the floor is
    # only a defensive guard; it never fires under production configs.
    safe_den = jnp.maximum(one_minus_mu_sq, 1e-30)
    return jnp.exp(-6.0 * S) * (one_minus_mu0_sq ** 2) / (safe_den ** 2)
```

### 3.2 RHS refactor in `driver_typeI_char.py::_rhs_core`

Replace the current J-slot read with a computed helper:

```python
# old:
# J_vals = jax.lax.dynamic_slice(y, (i_J,), (N_mu,))

# new:
mu      = mu_current_jax(X0, signs, S_val)
J_vals  = jacobian_jax(X0, S_val, mu)     # analytic, no state slot
```

Remove `dJ` from the RHS return (characteristic_rhs_jax now returns
only `(dI, dS)`) — see §3.4.

### 3.3 Layout change in `driver_typeI_char.py::_char_layout`

```python
def _char_layout(N_mu: int, n_species: int, thermo_tier: int = 1) -> dict:
    i_I = _IDX_I_START
    # PR-A: drop i_J; S follows directly after I.
    i_S = i_I + int(N_mu)
    i_tg = i_S + 1
    if int(thermo_tier) >= 2:
        i_tne = i_tg + 1
        i_tnx = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tne = -1
        i_tnx = -1
        i_net = i_tg + 1
    n_total = i_net + int(n_species)
    return {
        "i_Sp": _IDX_SP, "i_Sm": _IDX_SM,
        "i_I": i_I,
        "i_J": -1,           # retired; do not read
        "i_S": i_S, "i_tg": i_tg,
        "i_tne": i_tne, "i_tnx": i_tnx,
        "i_net": i_net, "n_total": n_total,
        "thermo_tier": int(thermo_tier),
    }
```

Keep `"i_J": -1` sentinel rather than removing the key, so downstream
code that probes `layout["i_J"]` fails loudly (`IndexError`) instead
of silently.

### 3.4 Update `characteristic_rhs_jax`

```python
def characteristic_rhs_jax(Sigma_plus, I, mu):
    """Return (dI, dS).  J is no longer an ODE state variable."""
    P2 = P2_jax(mu)
    dI = Sigma_plus * P2
    dS = Sigma_plus
    return dI, dS
```

All callers of `characteristic_rhs_jax` must be updated to unpack
two values, not three.

### 3.5 Update observable extractors' callers

Observable extractors (`extract_stress_jax`, `extract_monopole_jax`)
already take `J` as an argument.  They need **no** change; callers
now supply the analytic `J` via `jacobian_jax(X0, S, mu)`.

### 3.6 Update `_run_char_impl`
- Remove the line that initialises `y0[i_J:i_J+N_mu] = 1.0`.
- Remove the line that copies `y_handoff[i_J:i_J+N_mu] = y_p1[i_J:i_J+N_mu]`.
- Update metadata keys to reflect the reduced state dim.

### 3.7 Update `_char_active_indices`

With `J_j` gone, the only passive (strictly-zero self-coupling)
block is `I_j`.  Block-sparse mode is now:
`active = {Σ_+, Σ_-, S, T_γ, (T_νₑ, T_νₓ), X_i}`,
`passive = {I_j}`.

```python
def _char_active_indices(layout: dict, n_species: int) -> list:
    active = [_IDX_SP, _IDX_SM, layout["i_S"], layout["i_tg"]]
    if layout["thermo_tier"] >= 2:
        active.extend([layout["i_tne"], layout["i_tnx"]])
    active.extend(range(layout["i_net"], layout["i_net"] + int(n_species)))
    active.sort()
    return active
```

### 3.8 New unit test — `tests/test_pr_a_analytic_jacobian.py`

```python
"""PR-A regression: analytic J_j agrees with numerically-evolved J_j."""
import pytest
pytest.importorskip("jax")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from rabbit.jax.characteristic_rays_jax import (
    mu_current_jax, jacobian_jax,
)

@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.3, 0.5])
def test_analytic_jacobian_matches_numerical_solution(sigma):
    """For constant Σ_+, the ODE
        dJ/dN = 3 Σ_+ (1 - 3 μ²) J
    with μ(S) = sign(μ₀) √(X0 e^{6S}/(1+X0 e^{6S})) and S(N) = Σ_+·N
    has the closed-form solution of paper eq (51).  Verify this
    directly by integrating the ODE numerically and comparing.
    """
    from scipy.integrate import solve_ivp
    import numpy as np
    N_mu = 12
    from numpy.polynomial.legendre import leggauss
    mu0, _ = leggauss(N_mu)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)
    signs = np.sign(mu0)

    def numerical_J(N_final):
        def rhs(N, J_and_mu):
            J = J_and_mu[:N_mu]
            mu = J_and_mu[N_mu:]
            dJ = 3.0 * sigma * (1.0 - 3.0 * mu**2) * J
            dmu = 3.0 * sigma * mu * (1.0 - mu**2)
            return np.concatenate([dJ, dmu])
        y0 = np.concatenate([np.ones(N_mu), mu0])
        sol = solve_ivp(rhs, (0.0, N_final), y0, rtol=1e-12, atol=1e-14,
                         method="Radau")
        return sol.y[:N_mu, -1]

    for N_final in (0.5, 2.0, 5.0, 8.0):
        J_num = numerical_J(N_final)
        S_val = sigma * N_final     # constant Σ
        mu_now = np.asarray(mu_current_jax(
            jnp.asarray(X0), jnp.asarray(signs), jnp.asarray(S_val)))
        J_ana = np.asarray(jacobian_jax(
            jnp.asarray(X0), jnp.asarray(S_val), jnp.asarray(mu_now)))
        max_err = float(np.max(np.abs(J_num - J_ana)))
        assert max_err < 1e-9, (
            f"Σ={sigma}, N={N_final}: analytic J mismatch {max_err:.2e}"
        )
```

---

## 4. WBS

1. **Analytic-J_j scaffold test** (pre-code).  Implement
   `tests/test_pr_a_analytic_jacobian.py` in §3.8.  Run against the
   *current* codebase (should pass already if the paper eq is
   correctly transcribed; this guards the formula **before** the
   refactor).
2. **Add `jacobian_jax` helper** in
   `src/rabbit/jax/characteristic_rays_jax.py`.
3. **Layout migration** in
   `src/rabbit/jax/driver_typeI_char.py::_char_layout`.
4. **RHS migration**: call `jacobian_jax` inside `_rhs_core`; drop
   `J_vals` slicing; update `characteristic_rhs_jax` return signature;
   update callers in the driver.
5. **Initial-condition and handoff migration** in `_run_char_impl`.
6. **Block-sparse migration** in `_char_active_indices`.
7. **Parity runs** — tier-1 and tier-2 parity tests **must remain
   green at the same thresholds** as before.
8. **Documentation updates** per `ROADMAP_SELF_AUDIT.md §3`.

Each step individually must leave the test suite green; do not batch
steps past a failing state.

---

## 5. Three-stage verification protocol

Execute these three stages in order.  If any stage produces a
negative result, stop and report.  Do not merge a partial fix.

### Stage 1 — Internal repo literature verification

Execute exactly these checks and record verdicts:

1. Read `RABBIT_report_H1_revision_typo_pass2.pdf` pages 16–18 with
   the `Read` tool (`pages="16-18"`).  Quote paper eq (51) verbatim.
   Verify it matches the formula in §1.
2. Grep for every J-related identifier that will be affected:
   ```
   Grep "dJ_j|J_vals|i_J|characteristic_rhs_jax" src/rabbit/jax/
   ```
   Enumerate the call sites.  For every call site, decide whether it
   needs to be updated.
3. Read the SciPy reference `src/rabbit/transport/characteristic_rays.py`
   and confirm the SciPy driver also uses the analytic `J(S)` (it
   does *not* currently — SciPy evolves `J_j` as an ODE state).
   Record this as "SciPy divergence: JAX becomes lower-DOF but
   observationally identical; parity is via observables, not state
   slots".

Record outputs in a scratch file `docs/audit/PR-A_stage1.md`.

### Stage 2 — External web verification

Execute exactly these searches and record verdicts:

1. `WebSearch "Bianchi Type I BBN characteristic ray angular Jacobian"` —
   look for any external treatment of the same quantity.  Not
   expected to find anything (this is paper-specific), but record
   "nothing contradictory" explicitly.
2. `WebSearch "dJ/dN = 3 Sigma (1-3mu^2) J Bianchi"` — look for
   published references to this ODE.  Expected: Wainwright–Ellis
   geodesic literature (~1997).
3. Confirm that the analytic solution's exponent (`-6 S`) is a
   consequence of the `dlnμ²/(1-μ²)/dN = 6 Σ_+` direction equation
   (paper eq 48).  This is basic calculus but must be independently
   derived.

Record outputs in `docs/audit/PR-A_stage2.md`.

### Stage 3 — Self chain-of-thought verification

1. **Derivation from first principles.**  Starting from the two ODEs
   `dJ/dN = 3 Σ (1 - 3 μ²) J`, `dμ/dN = 3 Σ μ (1-μ²)`, show
   analytically that `J = e^{-6S} (1-μ₀²)² / (1-μ²)²`.  Sketch:
   `d(lnJ)/dN = 3Σ(1-3μ²)`,
   `d(ln(1-μ²))/dN = -2 μ · (dμ/dN) / (1-μ²) = -6 Σ μ²`,
   so `d(lnJ)/dN = 3Σ − 3·(−d(ln(1-μ²))/dN) = 3Σ + 3 d(ln(1-μ²))/dN`.
   Wait — let me redo:
   `dJ/dN = 3Σ J − 9 Σ μ² J = 3Σ J (1 − 3μ²)`; want this to equal
   `d[e^{-6S} (1-μ²)^{-2}] / dN · (1-μ_0²)²`.  Compute derivatives
   and confirm.  **Write the derivation out in full, not a handwave.**
2. **Dimensional analysis.**  `J` is dimensionless (angular
   Jacobian), `S` is dimensionless (shear × e-fold).  `e^{-6S}` is
   dimensionless.  The `(1-μ₀²)² / (1-μ²)²` factor is dimensionless.
   Product is dimensionless ✓.
3. **Limit check.**  At `S = 0`, `μ = μ₀` so `J = e⁰ · 1 = 1`.
   Matches initial condition.  At `S → +∞`, `μ² → 1` so `J → ∞`
   (angular compression toward the anisotropy axis).  Physical ✓.
4. **NaN / pole audit.**  `1 - μ² = 0` when `|μ| = 1`.  Gauss–
   Legendre nodes never hit `|μ| = 1`; quadrature weights go to zero
   before the pole.  Add a defensive floor `1e-30` anyway.
5. **Sign audit.**  Compare `J` sign to the SciPy-evolved `J_j` at
   Σ = 0.1, N = 3.  Both should be strictly positive at every ray;
   if either goes negative, the formula or the code is wrong.

Record outputs in `docs/audit/PR-A_stage3.md`.

---

## 6. Self-audit checklist (from ROADMAP_SELF_AUDIT.md)

Fill in `docs/audit/PR-A.md` with the standard template
([ROADMAP_SELF_AUDIT.md §2](../ROADMAP_SELF_AUDIT.md#2-per-pr-audit-checklist))
items:

- [ ] Physics correctness: paper-equation provenance verified
      (§5 Stage 1).
- [ ] No mocks / no hallucinations (§5 Stage 3 + adversarial audit §7).
- [ ] Tier-1 parity unchanged (tests/test_jax_typeI_characteristic_parity.py).
- [ ] Tier-2 parity unchanged (tests/test_jax_typeI_characteristic_tier2.py).
- [ ] New PR-A scaffold test passes.
- [ ] Warm single-solve timing ≤ baseline (no regression).
- [ ] State dim as measured: 25 at tier-1 phase-2 (was 37).
- [ ] STATE_OF_RECORD.md §1.2 / §2.3 / §5 / §6 updated.
- [ ] PR_CATALOG.md entry appended.
- [ ] `docs/audit/PR-A.md` committed.

---

## 7. Adversarial audit handoff

After Stage 3 is complete, spawn an adversarial audit agent:

```
Agent(
    description="PR-A adversarial audit",
    subagent_type="general-purpose",
    prompt=<see below>,
)
```

Prompt for the audit agent:

> You are a third-party physics auditor reviewing PR-A (analytic
> `J_j` elimination) for the RABBIT JAX characteristic-ray BBN
> driver.  Audit the diff harshly.  Specifically verify:
>
> 1. The analytic formula `J_j = e^{-6S} (1-μ_{j,0}²)² / (1-μ_j²)²`
>    is paper eq (51) (read pages 16–18 of
>    `RABBIT_report_H1_revision_typo_pass2.pdf`).
> 2. The state-dim reduction is correctly propagated: initial-condition
>    packing, phase-1→phase-2 handoff, observable extraction, and
>    `_char_active_indices` all handle the shorter state.
> 3. No call site of `characteristic_rhs_jax` unpacks three values
>    (all must unpack `(dI, dS)` after the refactor).
> 4. The tier-1 and tier-2 parity test grids remain green at the
>    same thresholds as before.
> 5. No mock or TODO placeholder sneaks in.
>
> Report issues ranked by severity.  Cap at 400 words.

Commit the agent's verdict to `docs/audit/PR-A.md`.

---

## 8. Anti-local-minimum reminders

Before committing:

1. Re-read §0's list of forbidden operations.  None should have
   been violated.
2. Compare your solution against the **alternative** of keeping `J_j`
   as an ODE state variable.  You picked the analytic removal —
   **why**?  Answer: eq (51) is the *exact* solution, numerical
   evolution is strictly noisier, and the state-dim reduction is
   free.  If you cannot give that answer in one sentence, you did
   not understand what PR-A is for.
3. Quote one line of paper §6.5 or §6.3 that directly motivates this
   replacement.  If you cannot, your implementation has no
   provenance.

---

## 9. Hallucination prevention

- Do **not** invent a function named `jacobian_jax` without
  implementing it.  The skeleton in §3.1 is the actual
  implementation.
- Do **not** cite a paper equation without first reading the PDF
  page and quoting verbatim.
- Do **not** claim the parity suites pass without running them
  (`pytest ... -q`).  The commit script in §11 enforces this.

---

## 10. Documentation updates

Exact edits to make before committing:

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §1.2: state-DOF row for characteristic-ray changes from
  "2·N_μ + 1 + scalars ≈ 37" to "N_μ + 1 + scalars ≈ 25 (PR-A)".
- §4.1: confirm parity numbers unchanged (add post-PR-A column if
  desired).
- §5.1: no new files (`jacobian_jax` is added to existing module).
- §6: test count updates (one new test file +N tests).

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append a new entry using the template from §0 of that file.  Fields
to fill:
- Status: merged
- Scope: JAX characteristic driver refactor
- Key files: `characteristic_rays_jax.py` (+`jacobian_jax`),
  `driver_typeI_char.py` (layout + RHS + handoff),
  `tests/test_pr_a_analytic_jacobian.py` (new).
- Physics added: paper eq (51) applied as analytic ODE solution.
- Parity before/after: unchanged at 5e-8 level.
- Performance: warm 1-solve unchanged or slightly faster (less state).

### 10.3 Topic-guide updates
- `docs/JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.3`: update "Phase 1 —
  analytic J_j elimination" from "planned" to "delivered in PR-A".

---

## 11. Deterministic commit script

Execute this block at the end of the phase.  Do not commit if any
step fails — report and stop.

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

# 1. Parity gates
pytest tests/test_pr_a_analytic_jacobian.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v

# 2. Full fast sweep (new reds must be documented)
pytest tests/ -m "not slow and not gpu" --tb=no -q

# 3. Ensure audit + docs exist and are staged
test -f docs/audit/PR-A.md
test -f docs/audit/PR-A_stage1.md
test -f docs/audit/PR-A_stage2.md
test -f docs/audit/PR-A_stage3.md
git add docs/audit/PR-A*.md

# 4. Ensure doc updates are staged
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

# 5. Commit
git commit -m "$(cat <<'EOF'
PR-A: analytic J_j elimination in the JAX characteristic driver

Replaces the numerically-evolved angular Jacobian J_j with its
closed-form expression (paper eq 51) inside
src/rabbit/jax/characteristic_rays_jax.py. State layout in
src/rabbit/jax/driver_typeI_char.py drops from 37 to 25 DOF at
tier-1 phase-2. Tier-1 and tier-2 publication parity unchanged;
new regression test locks the analytic formula against
numerically-integrated ODE reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git status
```

---

## 12. Abort conditions

Stop the phase (do not commit) if any of the following occur:

- Stage 1 cannot quote paper eq (51) verbatim.
- Stage 2 web search contradicts the formula.
- Stage 3 derivation arrives at a different exponent / sign.
- Tier-1 or tier-2 parity test grid regresses at any Σ × CL cell
  beyond 5 × 10⁻⁷.
- Adversarial audit (§7) raises a **high-severity** issue.
- Warm 1-solve timing regresses by more than 10 %%.

In any abort, write a scratch note to
`docs/audit/PR-A_abort.md` describing what failed and what would
need to change, then stop.
