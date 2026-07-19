# PR-T3A — q-Advection Kernel + Full-Phase-Space State (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

Tier-3 full-collision incomplete decoupling requires the per-ray
distribution `f_j(q_k)` as dynamical state (paper Appendix E,
eq 183–188).  The gravitational redshift term
`(d ln q / dN)_j · q ∂f_j/∂q` becomes a hyperbolic PDE in `q` on
each ray.  The current characteristic driver sidesteps this
because the ray state is a single scalar `I_j` and the distribution
is recovered analytically as `f_FD(q e^{2 I_j})`.  Tier-3 cannot.

This PR builds the q-advection kernel and the new per-ray full-
phase-space state driver, **without** any collision wiring.
Collisions arrive in PR-T3B/C.

Invariants: Rodas5P, CPU-preferred (single solve at tier-3 is
compute-heavy; GPU deferred to PR-G + PR-T3D combined testing),
float64, publication tolerance 5e-5 **for the collisionless
reduction** (full tier-3 targets a different reference — external
cross-codes).

Dependencies: PR-A (state reduction pattern), tier-3 Jacobian
preflight in
[`docs/audit/TIER3_FULL_COLLISION_PREFLIGHT_20260421.md`](../audit/TIER3_FULL_COLLISION_PREFLIGHT_20260421.md)
(full dense `jacfwd` is not the target surface at `~975` DOF).

---

## 1. Phase objective

Build a stable **continuous** q-advection scheme and wire it into a new driver
`src/rabbit/jax/driver_typeI_full_boltzmann.py`.  The driver must
reproduce the collisionless characteristic result to 10⁻⁸ when all
collision operators are disabled — this is the **sanity gate**
that proves the new state representation is self-consistent.

Preflight decision already made:

- the first landed driver uses a continuous semidiscrete q-advection
  RHS inside Rodas5P;
- exact semi-Lagrangian PCHIP remap is retained as the collisionless
  oracle/regression surface, not as the in-stage Rodas5P update.

---

## 2. Literature anchors

### 2.1 Internal
- Paper Appendix E — full phase-space ray formulation (pages 64–66).
- Paper eq 92 — Boltzmann equation along a characteristic ray
  `∂f_j/∂N = -(d ln y/dN)_j · y ∂f_j/∂y + C[f]/H`.
- Paper eq 184 — the same equation in dimensionless `q = p a / T_0`.
- Existing characteristic primitives (to produce the collisionless
  reference for the sanity gate):
  `src/rabbit/jax/characteristic_rays_jax.py`,
  `src/rabbit/jax/driver_typeI_char.py`.

### 2.2 External
- Semi-Lagrangian scheme for advection:
  Staniforth & Côté 1991, *Mon. Wea. Rev.* 119:2206 (generic
  framework).
- Cubic Hermite / PCHIP interpolation:
  Fritsch & Carlson 1980, *SIAM J. Numer. Anal.* 17:238.  PCHIP
  is monotone; standard not-a-knot cubic spline is not, so
  positivity of `f` cannot be guaranteed by the latter alone.
- Applications of semi-Lagrangian in cosmological Boltzmann:
  LASAGNA (Escudero 2019, `arXiv:1812.05605`),
  FortEPiaNO (Froustey & Pitrou & Volpe 2020, `arXiv:2008.01074`).

### 2.3 Paper-equation cross-check
- [ ] Eq 47: `d ln E/dN = -1 - Σ_+ P_2(μ)` (LRS).
- [ ] In dimensionless `q = p a / T_0`:
      `d ln q / dN = d ln(p a) / dN - d ln T_0 / dN = -Σ_+ P_2(μ)`
      (the `-1` term from expansion cancels against `a`).
- [ ] Paper eq 184: advection coefficient `(d ln q / dN)_j = -Σ_+ P_2(μ_j)`.

---

## 3. Skeleton code

### 3.1 Oracle semi-Lagrangian remap on the Laguerre q grid

```python
# src/rabbit/jax/q_advection_jax.py
"""Exact-remap oracle for per-ray full phase-space transport.

Advects f(q) → f(q · exp(-2 dI)) along the characteristic.
dI = ∫_N^{N+dt} Σ_+ P_2(μ(N')) dN' is the per-ray energy-shift
increment.  For a Rodas5P step of size h, dI ≈ Σ_+ P_2(μ_mid) · h
using the midpoint value.

Convention:
- q_nodes are Laguerre nodes (used consistently with the weak-rate
  kernels in weak_live_jax.py); they are NOT log-spaced.  To
  preserve positivity we interpolate in q-space directly.
- The interpolation kernel is PCHIP (monotone cubic) to guarantee
  f ∈ [0, 1] is preserved under advection.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _pchip_slopes(q_nodes: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Fritsch–Carlson PCHIP slopes at internal nodes.

    q_nodes : (N_q,)    strictly increasing
    f       : (N_q,)
    Returns (N_q,) slope estimates preserving monotonicity.
    """
    h = q_nodes[1:] - q_nodes[:-1]                 # (N_q-1,)
    d = (f[1:] - f[:-1]) / h                        # (N_q-1,)
    # interior slopes via harmonic weighting
    hs = h[:-1] + h[1:]                              # (N_q-2,)
    w1 = 2.0 * h[1:] + h[:-1]                       # (N_q-2,)
    w2 = h[1:] + 2.0 * h[:-1]                       # (N_q-2,)
    inv = jnp.where(
        (d[:-1] * d[1:]) > 0.0,
        (w1 + w2) / (w1 / d[:-1] + w2 / d[1:] + 1e-300),
        0.0,
    )                                                # (N_q-2,)
    slopes = jnp.concatenate([d[:1], inv, d[-1:]])   # (N_q,)
    return slopes


def _hermite_interp(q_nodes, f, slopes, q_query):
    """Evaluate cubic Hermite at q_query using (q_nodes, f, slopes).

    q_query : (..., M)
    Returns f_query : (..., M).
    """
    # Find the bracketing interval for each query point
    idx = jnp.clip(jnp.searchsorted(q_nodes, q_query, side="right") - 1,
                   0, q_nodes.shape[0] - 2)
    q0 = q_nodes[idx]; q1 = q_nodes[idx + 1]
    f0 = f[idx];       f1 = f[idx + 1]
    s0 = slopes[idx];  s1 = slopes[idx + 1]
    h = q1 - q0
    t = (q_query - q0) / jnp.maximum(h, 1e-300)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * f0 + h10 * h * s0 + h01 * f1 + h11 * h * s1


@jax.jit
def semi_lagrangian_q_advect(
    f: jnp.ndarray,        # (N_q,) distribution at current N
    q_nodes: jnp.ndarray,  # (N_q,) strictly increasing
    dI: jnp.ndarray,       # scalar energy-shift increment
) -> jnp.ndarray:
    """Advect f(q) → f(q · exp(-2 dI)) via PCHIP semi-Lagrangian.

    Conservation: for Sigma = 0, dI = 0, so the operator reduces to
    identity.
    Positivity: PCHIP preserves monotonicity.
    High-q tail: f_FD(q) → 0; queries beyond q_nodes[-1] clamp to 0.
    Low-q limit: f_FD(0) = 1/2; queries below q_nodes[0] clamp to
    f_nodes[0] (acceptable because Laguerre first node is ≈ 0.07).
    """
    q_query = q_nodes * jnp.exp(-2.0 * dI)
    slopes = _pchip_slopes(q_nodes, f)
    f_advected = _hermite_interp(q_nodes, f, slopes, q_query)
    # Clamp to physical range
    f_advected = jnp.clip(f_advected, 0.0, 1.0)
    # Zero at the high-q tail when the query extrapolates beyond domain
    f_advected = jnp.where(q_query > q_nodes[-1], 0.0, f_advected)
    # Clamp to f[0] when q_query is below q_nodes[0]
    f_advected = jnp.where(q_query < q_nodes[0], f[0], f_advected)
    return f_advected
```

### 3.2 New driver — full phase-space (collisionless first)

```python
# src/rabbit/jax/driver_typeI_full_boltzmann.py
"""Tier-3 full-phase-space ray driver.  Collisionless first (PR-T3A);
ν-e + pair in PR-T3B, ν-ν in PR-T3C.

State layout (LRS, tier-2 thermo by default — tier-3 needs 3T):
    [Σ_+, Σ_-,
     f_{jk} for ν_e, ν_ebar, ν_x, ν_xbar at each (ray j, mom k),
     S (LRS) or (S_+, S_-) (non-LRS),
     T_γ, T_νₑ, T_νₓ,
     X_i (n_species)]

At N_μ=12, N_q=20, 4 species: 4·12·20 = 960 distribution slots.
Plus 2 (Σ) + 1 (S) + 3 (T) + 9 (X) = 975 DOF total.

Species-identical approximation:  In the collisionless limit all
neutrino species evolve identically; we carry 4 distinct species
only because PR-T3B's ν-e collision breaks the degeneracy.  For
PR-T3A sanity gate we verify that with collisions disabled, all 4
species remain identical within 1e-12.
"""
```

### 3.3 Collisionless RHS

```python
def _rhs_core_full_boltzmann_collisionless(
    N, y, *,
    tau_n, eta, N_eff, f_nu,
    mu0, w0, X0, signs,
    q_nodes, q_weights,
    rate_table,
    layout, n_species,
):
    """Full-phase-space RHS with no collision operator.  Must
    reduce bitwise to the characteristic driver via the identity
        f_j(q, N) = f_FD(q e^{2 I_j(N)})
    where I_j(N) is the energy-shift integral from paper eq (54).
    """
    # 1. Unpack state
    Sigma_plus = y[layout["i_Sp"]]
    Sigma_minus = y[layout["i_Sm"]]
    S_val = y[layout["i_S"]]

    # 2. Direction + analytic J (as in PR-A)
    mu = mu_current_jax(X0, signs, S_val)
    P2 = 0.5 * (3.0 * mu * mu - 1.0)
    J_vals = jacobian_jax(X0, S_val, mu)
    dS = Sigma_plus

    # 3. Per-ray per-momentum advection coefficient
    # d ln q / dN = -Σ_+ P_2(μ_j)   (paper eq 184)
    dlnq_dN = -Sigma_plus * P2    # (N_mu,)

    # 4. Per-species distribution evolution (4 identical species
    # under collisionless transport; same formula for all)
    #    ∂f_j(q)/∂N = -(d ln q / dN)_j · q · ∂f_j/∂q
    #
    # Preflight conclusion:
    # - keep advection as a continuous RHS derivative inside Rodas5P;
    # - use a monotone one-sided stencil for the landed driver;
    # - use exact PCHIP remap only as the collisionless oracle.
    ...
```

### 3.4 Collisionless sanity gate

```python
# tests/test_pr_t3a_collisionless_reduction.py
@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.3])
def test_full_boltzmann_collisionless_matches_char(sigma):
    """With all collision operators disabled, tier-3 full-phase-space
    driver must reproduce the characteristic driver (LRS, tier-1)
    to 1e-8.  If this fails, q-advection is wrong."""
    from rabbit.jax.driver_typeI_char import (
        JAXTypeICharConfig, run_full_coupled_typeI_char_jax,
    )
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig, run_full_boltzmann_jax,
    )
    ref = run_full_coupled_typeI_char_jax(JAXTypeICharConfig(
        Sigma_H_plus=sigma, correction_level=0, N_q=20, N_mu=12,
        n_reactions=12, thermo_tier=1,
    ))
    out = run_full_boltzmann_jax(JAXFullBoltzmannConfig(
        Sigma_H_plus=sigma, correction_level=0, N_q=20, N_mu=12,
        n_reactions=12, thermo_tier=1,
        enable_nu_e=False, enable_pair=False, enable_nu_nu=False,
    ))
    assert abs(ref.Yp - out.Yp) < 1e-8, (
        f"Σ={sigma}: collisionless full-Boltzmann Y_p mismatch "
        f"{abs(ref.Yp - out.Yp):.2e}"
    )
```

---

## 4. WBS

1. **Continuous q-advection helper** (`q_advection_jax.py` or equivalent).
2. **PCHIP oracle tests** (monotonicity, conservation, 10⁻¹² free-
   streaming round-trip over 500 steps).
3. **Full-Boltzmann driver shell** (no collisions).
4. **Collisionless RHS** — implement the continuous semidiscrete
   advection path selected by preflight.
5. **Collisionless sanity gate** (test in §3.4).
6. **Tier-3 Jacobian hook** — target the factorized
   `J_q + A_apply * J_collision_core * G_gather` surface from the
   preflight audit rather than a naive dense full-state Jacobian.
7. **Documentation updates**.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read paper Appendix E (pages 64–66) in full.
- Read `characteristic_rays_jax.py::extract_monopole_jax` to
  understand how the current driver builds f̃_0(q) from per-ray
  I_j — this is the "analytic advection" the new driver replaces
  with a numerical scheme.
- Grep for existing uses of `jnp.gradient` or finite-difference
  stencils in the repo (PSTF hierarchy may already use them).

### Stage 2 — External
- `WebSearch "Escudero LASAGNA neutrino Boltzmann q advection"`
  — confirm how external full-transport codes handle the transport step.
- `WebSearch "FortEPiaNO Jacobian neutrino decoupling"`
  — extract the direct-Jacobian lesson for the collision core.
- Confirm that the `q ∂f/∂q` form on a Laguerre grid is
  numerically well-conditioned; Laguerre nodes cluster near 0, so
  the ratio of adjacent node spacings can be large — document the
  condition number.

### Stage 3 — Self CoT
- **Preflight resolution.**  Do not reopen split-vs-inline unless
  new solver hooks land first.  Current path is continuous
  semidiscrete advection inside Rodas5P, with PCHIP remap retained
  as the collisionless oracle.
- **Dimensional analysis** of `d ln q / dN`: dimensionless ✓.
  `q · ∂f/∂q` has units of `f` (dimensionless).  RHS `dy/dN`
  units match ✓.
- **Collisionless reduction.**  On a fine q-grid with PCHIP +
  upwind, the distribution `f(q, N)` must equal
  `f_FD(q · exp(2 I_j(N)))` to numerical precision.  Sanity gate
  (§3.4) verifies this at Σ ∈ {0, 0.1, 0.3}.
- **Positivity.**  Exact PCHIP remap is the oracle.  The landed
  continuous stencil must be monotone enough to avoid meaningful
  negativity in the collisionless sanity gate; document any
  residual clipping explicitly if required.
- **NaN audit**: `∂f/∂q` at `q_nodes[-1]` requires a one-sided
  stencil; define it cleanly.

Record in `docs/audit/PR-T3A_stage{1,2,3}.md`, including any
evidence that would justify revisiting the preflight decision.

---

## 6. Self-audit checklist
- [ ] PCHIP oracle round-trip test (500 steps, free streaming) within
      1e-12.
- [ ] Collisionless sanity gate passes at Σ ∈ {0, 0.1, 0.3}.
- [ ] Positivity: `f_j(q_k) ∈ [0, 1]` throughout test grid.
- [ ] Rodas5P step size does not collapse to h_min at any tested Σ
      (CFL analysis OK).
- [ ] State dim at tier-2 phase-2: 975 (check via layout).
- [ ] Tier-3 factorized Jacobian surface represented cleanly in the
      implementation notes.

---

## 7. Adversarial audit prompt

> Audit PR-T3A (full-phase-space driver, collisionless sanity).
> Verify: (1) continuous q-advection scheme preserves positivity well
> enough for the collisionless sanity gate; (2) collisionless sanity
> gate bitwise within 1e-8; (3) PCHIP oracle is used as regression
> surface rather than in-stage update; (4) tier-3 Jacobian notes track
> the factorized moment-core path; (5) Rodas5P contract remains
> unchanged.  Cap 400 words.

---

## 8. Anti-local-minimum reminders
1. **Do not** adopt operator-splitting without quantifying the
   splitting error and showing it is below tier-3 tolerance at
   all tested `h`.  Prefer inline FD with CFL-limited step.
2. **Do not** replace Laguerre nodes with log-spaced nodes for
   convenience.  The weak-rate kernels in `weak_live_jax.py`
   assume Laguerre.
3. **Do not** start turning on collisions in this PR; that is
   PR-T3B/C.  Scope discipline.

---

## 9. Hallucination prevention
- `jax.numpy.gradient` exists but uses 2nd-order centred
  differences — not upwind.  For upwind, hand-code the stencil.
- `jnp.searchsorted` on Laguerre nodes works; do not use
  `numpy.interp` (incompatible with JIT).

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §1.2: add "Full phase-space ray" row with state DOF ~975 and
  status "partial (collisionless gate in PR-T3A)".
- §5.1: new modules.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-T3A entry.

### 10.3 `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
Flip Phase 1 + Phase 2 status from "planned" to "delivered in
PR-T3A".  Record the inline-FD-vs-split decision.

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_t3a_collisionless_reduction.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v

test -f docs/audit/PR-T3A.md
test -f docs/audit/PR-T3A_stage1.md
test -f docs/audit/PR-T3A_stage2.md
test -f docs/audit/PR-T3A_stage3.md
git add docs/audit/PR-T3A*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-T3A: full-phase-space ray driver (collisionless sanity gate)

Adds src/rabbit/jax/q_advection_jax.py (PCHIP + upwind
finite-difference for q-space advection) and
src/rabbit/jax/driver_typeI_full_boltzmann.py (per-ray
per-momentum state representation, paper Appendix E).  No
collision operators wired yet; PR-T3B adds ν-e elastic + pair and
PR-T3C adds diagonal ν-ν scattering.  Collisionless sanity gate
reproduces the existing characteristic driver to 1e-8 across
Σ ∈ {0, 0.1, 0.3}, proving the new state representation and
advection scheme are self-consistent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Collisionless sanity gate fails at any Σ (> 1e-8).
- Positivity violation at any (q, N) pair.
- Rodas5P step size collapses to `h_min` during any test.
- Stage-3 CoT reveals that the chosen advection scheme introduces
  diffusion above 10⁻⁴ per e-fold at Laguerre resolution.

Abort → `docs/audit/PR-T3A_abort.md`.
