# PR-T3C — Diagonal ν–ν Scattering Operator (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

Tier-3 scope (paper §11.2, Table 2) is "tier-2 plus diagonal ν–ν
scattering" — elastic + pair are handled in PR-T3B; PR-T3C adds
the ν–ν operator that is **absent from the codebase today**.
"Diagonal" means Fierz-diagonal flavour content only: no flavour
mixing, no off-diagonal density-matrix terms.  Those belong to
tier-4 (QKE) and are **explicitly out of scope**.

Invariants: Rodas5P, CPU-preferred, float64, stable-identity RHS
cache, publication tolerance for cross-code parity 5 × 10⁻⁴.

Dependencies: PR-T3B (strict — tier-2 + ν-e + pair must be in
place as the host for the new ν–ν term).  PR-J recommended.

---

## 1. Phase objective

Derive and implement the diagonal ν–ν collision operator
`C_νν[f_α](q)` for the processes
`ν_α + ν_β → ν_α + ν_β` (all α,β with the Fierz-diagonal structure
of the Standard Model).  Wire it as an additive term in the GCS
`COLLIDE` step of the full-Boltzmann driver.  Close the remaining
~0.01 gap in `N_eff` at FLRW relative to PR-T3B.

Acceptance: FLRW `|N_eff - 3.044| < 0.005` with ν-e + pair + ν-ν
all enabled.

---

## 2. Literature anchors

### 2.1 Internal
- Paper §11.2 (tier hierarchy, Table 2).
- Paper §7.1 (scaling with temperature, eq 62–63).
- Paper Appendix E.3 (collision coupling between rays).
- `rabbit/collisions/` — existing ν-e and pair operators as style
  reference (Operator protocol, kernel structure).
- SciPy ν–ν placeholder?  Grep `src/rabbit/collisions/` for any
  existing `nu_nu_*` file.  If a stub exists, use its signatures;
  otherwise create the file from scratch.

### 2.2 External — REQUIRED reading
- **Dolgov, Hansen, Semikoz 1997** `NPB 503:426` (paper ref [20]):
  closed-form kernels for ν–ν scattering in the massless limit.
- **Mangano et al. 2005** `NPB 729:221` (paper ref [17]):
  incorporation of ν–ν into the full-BBN calculation.
- **de Salas & Pastor 2016** `JCAP 07:051` (paper ref [18]):
  modern precision benchmark with ν–ν.
- **Froustey, Pitrou, Volpe 2020** `JCAP 12:015` (paper ref [19]):
  FortEPiaNO, full QKE but the diagonal limit provides a
  cross-check for our tier-3.

### 2.3 Paper-equation cross-check
- [ ] Eq 62–63: scaling `Γ ~ G_F² T⁵`, `H ~ √G_N T²`, so
      `Γ_νν / H ~ T³` — decouples late, same order as ν-e.
- [ ] Tier-3 definition (Table 2, page 24): "Tier 2 plus diagonal
      ν–ν scattering.  Partial implementation noted; full
      production-grade is out of scope for the paper."

---

## 3. Skeleton code

### 3.1 Derivation sketch (must be filled out in Stage 3)

The Standard Model matrix element for
`ν_α(p_1) + ν_β(p_2) → ν_α(p_3) + ν_β(p_4)` (massless limit,
Fierz-diagonal) is
```
|M|²_{αβ diagonal} ∝ G_F² · (p_1 · p_2)(p_3 · p_4) · structure_{αβ}
```
with species-mixing factor `structure_{αβ}` that distinguishes:
- `α = β`: Majorana identical-particle factor → coefficient 2.
- `α ≠ β`: distinguishable → coefficient 1.

Following Dolgov–Hansen–Semikoz (1997) eq (20), the reduced 1D
integral over the partner momentum `y_2` is:
```
C_{νν}[f_α](y_1) = (G_F² T_ν⁵ / π⁵) · ∫ dy_2 dy_3 Θ(y_4) K_{αβ}(y_1, y_2, y_3)
                 · [f_{α,3} f_{β,4} (1 - f_{α,1})(1 - f_{β,2})
                    - f_{α,1} f_{β,2} (1 - f_{α,3})(1 - f_{β,4})]
```
with `K_{αβ}` the angular-integrated coefficient table in their
appendix.  Record the exact coefficients in
`docs/audit/PR-T3C_stage3.md`.

### 3.2 New module

```python
# src/rabbit/collisions/nu_nu_scattering.py — SciPy reference
"""Diagonal ν–ν scattering operator for tier-3 incomplete decoupling.

Implements Dolgov-Hansen-Semikoz (1997) appendix A kernels, in the
ultrarelativistic massless limit, flavour-diagonal.
"""

class NuNuScatteringOperator(CollisionOperator):
    def evaluate(self, f_nue_mono, f_nuebar_mono, f_nux_mono, T_nu_e, T_nu_x, q_nodes, q_weights):
        """Returns dict with four arrays (one per species):
        {"nue": C_nu_e_mono(q), "nuebar": ..., "nux": ..., "nuxbar": ...}.
        """
        ...


# src/rabbit/jax/collisions_jax.py — JAX port
@jax.jit
def C_nu_nu_scattering_jax(
    f_nue_mono, f_nuebar_mono, f_nux_mono,
    T_nu_e, T_nu_x,
    q_nodes, q_weights,
) -> dict[str, jnp.ndarray]:
    """JAX port of NuNuScatteringOperator.  Elementwise match to
    SciPy reference at 1e-12.
    """
    # 1D Laguerre integration over partner momentum y_2.
    # For each q_1 node, loop (vectorised) over y_2 = q_nodes.
    # Form the Dolgov–Hansen–Semikoz kernel and the statistical factor.
    # Return dict keyed by species.
    ...
```

### 3.3 GCS integration

```python
# driver_typeI_full_boltzmann.py — extend COLLIDE step
if enable_nu_nu:
    C_nunu = C_nu_nu_scattering_jax(
        f_tilde_0_nue, f_tilde_0_nuebar, f_tilde_0_nux,
        T_nu_e, T_nu_x, q_nodes, q_weights,
    )
    C0_nue    += C_nunu["nue"]
    C0_nuebar += C_nunu["nuebar"]
    C0_nux    += C_nunu["nux"]
    # (ν_xbar same as ν_x by CP symmetry at tier-3 no-QKE)
```

Weak-rate impact: none directly — `C_νν` is internal to the
neutrino sector.  Energy conservation within that sector gives
`∫ y³ C_νν dy = 0` (tested in §3.4).

### 3.4 Tests

```python
# tests/test_pr_t3c_nu_nu.py
def test_nu_nu_jax_matches_scipy():
    """1e-12 elementwise parity."""
    ...

def test_nu_nu_detailed_balance():
    """C_νν[f_eq] = 0 at T_ν_e = T_ν_x, same T, to 1e-14."""
    ...

def test_nu_nu_energy_conservation():
    """∫ y³ C_νν_α(y) dy summed over α vanishes to 1e-12.
    (Elastic scattering within the neutrino sector conserves
    total neutrino energy density.)"""
    ...

def test_tier3_flrw_neff_locks():
    """With ν-e + pair + ν-ν all enabled at FLRW CL0,
    |N_eff - 3.044| < 0.005.  Cross-code reference:
    LASAGNA / FortEPiaNO."""
    ...

def test_tier3_anisotropic_neff_stability():
    """N_eff moves < 5e-4 across Σ_H ∈ {0, 0.1, 0.3}."""
    ...
```

---

## 4. WBS

1. **Reread Dolgov-Hansen-Semikoz and Mangano 2005 kernels** —
   record the exact `K_{αβ}` coefficients in
   `docs/audit/PR-T3C_stage3.md`.
2. **SciPy reference implementation** — `nu_nu_scattering.py` on
   the SciPy side.  Unit-tested via detailed balance and energy
   conservation.
3. **JAX port** — `C_nu_nu_scattering_jax` in `collisions_jax.py`.
4. **SciPy↔JAX 1e-12 parity test**.
5. **GCS wiring in the full-Boltzmann RHS**.
6. **FLRW N_eff lock at 3.044 ± 0.005**.
7. **Anisotropic stability sweep**.
8. **Documentation updates**.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read paper §7.1, Table 2, and the retired-Teff Section 11 which
  repeatedly references "diagonal ν–ν" as the tier-3 addition.
- Grep for any existing `nu_nu_*` stub in the repo (there may be
  a placeholder file committed during earlier planning).
- Read `rabbit/collisions/kernels.py::CollisionOperator` protocol
  to ensure the new operator conforms.

### Stage 2 — External
- Dolgov-Hansen-Semikoz 1997 — locate the appendix A kernel formula
  explicitly (cross-verify at least two independent citations).
- Mangano 2005 — confirm the tier-3 impact on `N_eff` is
  approximately +0.005 relative to tier-2 (ν-e + pair only).
- Froustey 2020 — confirm the diagonal-limit impact matches
  Mangano's.
- Record each reference URL, excerpted formula lines, and the
  coefficient table in `docs/audit/PR-T3C_stage2.md`.

### Stage 3 — Self CoT
- **Derivation.** Starting from the four-fermion contact Lagrangian,
  derive `|M|²` for `ν_α ν_β → ν_α ν_β` in the ultrarelativistic
  limit, keeping Fierz-diagonal terms only.  Separate `α = β`
  (identical particles, coefficient 2 from Fierz) from `α ≠ β`.
- **Dimensional analysis.**  `C_νν` has units of `s⁻¹`.  After
  dividing by `H` inside the RHS, units become `efold⁻¹` ✓.
- **Detailed balance.**  At `f = f_FD(T_ν)` identical for all
  species, the integrand
  `f_3 f_4 (1-f_1)(1-f_2) - f_1 f_2 (1-f_3)(1-f_4)` vanishes by
  identity (same FD ⇒ statistical factor = 0).  Implement as a
  unit test.
- **Energy conservation.**  Elastic scattering conserves total
  neutrino 4-momentum, so
  `∑_α ∫ y³ C_νν_α(y) dy = 0` exactly.  Test numerically at 1e-12.
- **CP symmetry.**  At tier-3 no-QKE, `f_ν_α = f_ν̄_α` for each
  flavour.  Maintain CP-identity: evolve only four distinct
  distributions `(f_νₑ, f_ν̄ₑ, f_νₓ, f_ν̄ₓ)` but set `f_ν̄_α = f_ν_α`
  initially and check invariance.

Record in `docs/audit/PR-T3C_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] SciPy ν–ν operator exists + passes detailed balance + energy
      conservation.
- [ ] JAX port matches SciPy at 1e-12.
- [ ] FLRW `|N_eff - 3.044| < 0.005` with full tier-3.
- [ ] Anisotropic `|ΔN_eff| < 5 × 10⁻⁴` across Σ ∈ {0, 0.1, 0.3}.
- [ ] No regression in PR-T3B's FLRW test (must still pass).

---

## 7. Adversarial audit prompt

> Audit PR-T3C (diagonal ν–ν).  Verify: (1) SciPy operator matches
> published literature (Dolgov-Hansen-Semikoz coefficients quoted
> with citation); (2) JAX port elementwise parity at 1e-12; (3)
> detailed balance to 1e-14; (4) elastic energy conservation to
> 1e-12; (5) FLRW N_eff within 3.044 ± 0.005; (6) tier-3 anisotropic
> stability.  Cap 500 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** shortcut the Dolgov-Hansen-Semikoz kernel by using
   a "generic" G_F² T⁵ scaling.  The per-flavour coefficients
   matter for N_eff.
2. **Do not** add off-diagonal Fierz terms "while we're at it".
   Tier-4 / QKE is explicitly out of scope.
3. **Do not** relax the FLRW N_eff tolerance if the test initially
   fails; investigate the coefficient table first.

---

## 9. Hallucination prevention
- Do not cite a closed-form kernel without reading the actual
  Dolgov-Hansen-Semikoz appendix.  The paper is `arXiv:hep-ph/9703315`;
  fetch it via `WebFetch` if needed.
- Do not invent species-counting coefficients.  Fierz structure is
  lore-heavy and easy to miscopy.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §1.4 Tier hierarchy row 3: "Full Boltzmann: ν-e elastic + pair +
  **diagonal ν-ν**".
- §4.3: update cross-code tier-3 row — RABBIT should now land
  inside the Mangano / Froustey band.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-T3C entry with measured FLRW N_eff.

### 10.3 `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
Flip Phase 3 status.  Record Dolgov-Hansen-Semikoz citation as
the operator's provenance.

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_t3c_nu_nu.py -v
pytest tests/test_pr_t3b_flrw_neff.py -v    # must still pass
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v

test -f docs/audit/PR-T3C.md
test -f docs/audit/PR-T3C_stage1.md
test -f docs/audit/PR-T3C_stage2.md
test -f docs/audit/PR-T3C_stage3.md
git add docs/audit/PR-T3C*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-T3C: diagonal ν-ν scattering operator for tier-3 incomplete
decoupling

Implements the Dolgov-Hansen-Semikoz (1997) diagonal ν-ν
scattering kernel in src/rabbit/collisions/nu_nu_scattering.py
(SciPy reference) and src/rabbit/jax/collisions_jax.py (JAX port,
elementwise match to 1e-12).  Wires it into the GCS COLLIDE step
of the full-phase-space driver from PR-T3A/B.  Closes the ~0.01
gap to Mangano 2005: FLRW |N_eff - 3.044| < 0.005.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Detailed balance worse than 1e-12.
- Energy conservation worse than 1e-12.
- FLRW |N_eff - 3.044| > 0.008.
- JAX↔SciPy operator parity worse than 1e-12.
- Anisotropic N_eff shift > 10⁻³ across Σ ∈ {0, 0.1, 0.3}.

Abort → `docs/audit/PR-T3C_abort.md`.
