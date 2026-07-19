# PR-N2 — Non-LRS Driver Integration (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

LRS characteristic driver (`driver_typeI_char.py`) is production-
grade at tier-1 and tier-2.  PR-N1 added non-LRS primitives in
`src/rabbit/jax/characteristic_rays_nonlrs_jax.py`.  This PR wires
them into a new driver-level transport mode.

Invariants: Rodas5P, CPU-preferred, float64, transported monopole
in weak rates, publication tolerance 5e-5.

Dependencies: PR-N1 (strict).  Strongly benefits from PR-A (state
reduction) and PR-J (analytic Jacobian) because the non-LRS state
is ~200 DOF (12×16 rays at tensor product) and Rodas5P Jacobian
cost scales with state dim.

---

## 1. Phase objective

Add `transport_mode="characteristic_nonlrs"` to `JAXTypeICharConfig`
and wire it through the driver.  Register a new backend capability
`JAX_TYPEI_CHARACTERISTIC_NONLRS_TIER1` and backend key
`jax_characteristic_nonlrs`.

Publication target: at small shear `Σ_+ = Σ_- = 0.05`, the new
driver matches the linearised-PSTF generic driver (n_ell=3) in
sign and magnitude of Δ Y_p; at LRS (Σ_- = 0, N_φ = 1), the new
driver matches the existing LRS characteristic driver to 1e-10.

---

## 2. Literature anchors

### 2.1 Internal
- PR-N1 module:
  `src/rabbit/jax/characteristic_rays_nonlrs_jax.py`.
- Existing LRS driver:
  `src/rabbit/jax/driver_typeI_char.py` — layout, RHS, handoff.
- Linearised PSTF generic implementation (reference for parity):
  `src/rabbit/transport/typeI_hierarchy.py` (SciPy) and
  `src/rabbit/jax/rhs_typeI.py` (JAX, n_ell=3).
- Paper §2.1–§2.2 (geometry), §3.2 (PSTF generic), §6.6 (observables).

### 2.2 External
- Froustey 2020 (FLRW cross-code).  Not directly relevant to Bianchi
  extension but confirms the weak-rate baseline.
- Wainwright–Ellis chapter on Type I (no new direction-map
  references beyond PR-N1).

### 2.3 Paper-equation cross-check
- [ ] Eq 147: `dΣ_±/dN = -(2-q) Σ_± + Π_±`.  Confirm both
      components are tracked and coupled.
- [ ] Eq 7: Hubble `H = H_FLRW / √(1 - Σ²)` with `Σ² = Σ_+² + Σ_-²`.
- [ ] Eq 6: Friedmann `Ω = 1 - Σ² > 0` (positivity floor in JAX).

---

## 3. Skeleton code

### 3.1 Layout

```python
# driver_typeI_char.py — new helper
def _char_layout_nonlrs(
    N_theta: int, N_phi: int, n_species: int, thermo_tier: int = 1
) -> dict:
    """Non-LRS state vector:
        [Σ_+, Σ_-, I_{θ_j, φ_k}, S_+, S_-, T_γ, (T_νₑ, T_νₓ), X_i]
    J_jk is computed analytically from (S_+, S_-, μ_{jk,0}, φ_{jk,0})
    and does not occupy state slots (PR-A pattern extended).
    """
    N_rays = int(N_theta) * int(N_phi)
    i_I = _IDX_I_START
    i_S_plus = i_I + N_rays
    i_S_minus = i_S_plus + 1
    i_tg = i_S_minus + 1
    if int(thermo_tier) >= 2:
        i_tne = i_tg + 1
        i_tnx = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tne = -1; i_tnx = -1
        i_net = i_tg + 1
    n_total = i_net + int(n_species)
    return {
        "i_Sp": _IDX_SP, "i_Sm": _IDX_SM,
        "i_I": i_I,
        "i_S_plus": i_S_plus, "i_S_minus": i_S_minus,
        "i_tg": i_tg, "i_tne": i_tne, "i_tnx": i_tnx,
        "i_net": i_net, "n_total": n_total,
        "thermo_tier": int(thermo_tier),
        "N_theta": int(N_theta), "N_phi": int(N_phi),
        "N_rays": N_rays,
    }
```

### 3.2 RHS kernel (non-LRS)

```python
def _rhs_core_nonlrs(
    N, y, *,
    phase, correction_level, thermo_tier,
    tau_n, eta, N_eff, f_nu,
    mu0, phi0, w_s2, X0, signs,      # from setup_ray_grid_S2
    q_nodes, q_weights,
    rate_table,
    layout, n_species,
):
    """Non-LRS characteristic RHS.  Mirrors _rhs_core but:
    - reads S_+ and S_- from two state slots,
    - reconstructs (μ, φ) via mu_phi_current_jax,
    - computes J analytically from (S_+, μ₀, μ),
    - extracts Π_+ and Π_- via extract_stress_{plus, minus}_S2,
    - feeds both Π's into the geometry RHS.
    """
    from rabbit.jax.characteristic_rays_nonlrs_jax import (
        mu_phi_current_jax,
        extract_stress_plus_S2,
        extract_stress_minus_S2,
        extract_monopole_S2,
    )
    i_I = layout["i_I"]; N_rays = layout["N_rays"]
    i_S_plus = layout["i_S_plus"]; i_S_minus = layout["i_S_minus"]

    Sigma_plus = y[_IDX_SP]; Sigma_minus = y[_IDX_SM]
    I_vals = jax.lax.dynamic_slice(y, (i_I,), (N_rays,))
    S_plus = y[i_S_plus]; S_minus = y[i_S_minus]

    Sigma_sq = Sigma_plus * Sigma_plus + Sigma_minus * Sigma_minus

    mu, phi = mu_phi_current_jax(mu0, phi0, S_plus, S_minus)

    # Analytic J for non-LRS — extension of PR-A paper-eq-51.
    # For non-LRS, J(S_+, S_-, mu_0, phi_0, mu, phi) factorises into
    # the product of the two-axis stretchings:
    #   J_{jk} = (a_x * a_y * a_z) / (|∂ê/∂ê_0|)
    # where a_α = exp(-S-component_α).  Equivalently (after unit
    # normalisation), J_{jk} = exp(-2 S_+) (1 - μ_0²)/(1 - μ²) on
    # the LRS locus; the non-LRS generalisation is derived in
    # docs/audit/PR-N1_stage3.md.  Placeholder below; Stage-3 CoT
    # of this PR MUST re-derive from first principles.
    J_vals = jacobian_nonlrs_jax(
        mu0, phi0, S_plus, S_minus, mu, phi,
    )   # shape (N_rays,)

    Pi_plus = extract_stress_plus_S2(I_vals, J_vals, mu, w_s2, f_nu)
    Pi_minus = extract_stress_minus_S2(I_vals, J_vals, mu, phi, w_s2, f_nu)

    damping = -(1.0 - Sigma_sq)
    dSp = damping * Sigma_plus + Pi_plus
    dSm = damping * Sigma_minus + Pi_minus

    # Per-ray I evolution (paper eq 54 generalised):
    #   dI_{jk}/dN = Σ_+ P_2(μ_{jk}) + Σ_- · [sin²θ cos(2φ)] / √3
    P2 = 0.5 * (3.0 * mu * mu - 1.0)
    Y22_real = (1.0 - mu * mu) * jnp.cos(2.0 * phi)
    dI = Sigma_plus * P2 + Sigma_minus * (Y22_real / jnp.sqrt(3.0))
    dS_plus = Sigma_plus
    dS_minus = Sigma_minus

    # Thermo + weak + network — identical to LRS path using
    # f_mono = extract_monopole_S2.
    if int(thermo_tier) >= 2:
        ...
    else:
        ...

    # Pack dy
    dy = jnp.zeros_like(y)
    dy = dy.at[_IDX_SP].set(dSp).at[_IDX_SM].set(dSm)
    dy = jax.lax.dynamic_update_slice(dy, dI, (i_I,))
    dy = dy.at[i_S_plus].set(dS_plus)
    dy = dy.at[i_S_minus].set(dS_minus)
    dy = dy.at[layout["i_tg"]].set(dT_gamma)
    if thermo_tier >= 2:
        dy = dy.at[layout["i_tne"]].set(dT_nu_e).at[layout["i_tnx"]].set(dT_nu_x)
    dy = jax.lax.dynamic_update_slice(dy, dX, (layout["i_net"],))
    return dy
```

### 3.3 Analytic J_{jk} for non-LRS

This is the non-LRS extension of paper eq (51).  It **must be
derived in Stage-3 CoT** of this PR.  Starting sketch:
```
J_{jk} = det(∂ê_{jk}(N) / ∂ê_{jk,0})  evaluated on the unit sphere
        = product of axis-stretchings / (‖ê_current‖ to unit sphere)
```
Given the forward map `ê → (a_x ex_0, a_y ey_0, a_z ez_0) / norm`,
the determinant of the Jacobian on S² reduces to a closed form in
`(S_+, S_-, μ_0, φ_0, μ, φ)`.  Verify numerically by differentiating
`mu_phi_current_jax` with `jax.jacfwd`.

If the analytic form is not robustly derivable, **fall back** to
per-step JAX-autodiff of the direction map and multiply by the
small correction — this keeps the driver correct but adds cost.

### 3.4 Config + dispatch

```python
# JAXTypeICharConfig
transport_mode: str = "characteristic"    # or "characteristic_nonlrs"
N_theta: int = 12
N_phi: int = 16   # default at "tensor product 192 rays"

def __post_init__(self):
    ...
    if self.transport_mode == "characteristic_nonlrs":
        # Allow Σ_- ≠ 0
        ...
    elif self.transport_mode == "characteristic":
        # Existing LRS-only guard
        if abs(self.Sigma_H_minus) > 0:
            raise NotImplementedError(...)
```

Register new capability `JAX_TYPEI_CHARACTERISTIC_NONLRS_TIER1`
and backend key `jax_characteristic_nonlrs` in
`backend_capabilities.py`.  Wire dispatch in
`canonical_forward_solver`.

### 3.5 Parity tests

```python
# tests/test_pr_n2_nonlrs_driver.py
- LRS reduction (N_φ=1, Σ_-=0) ↔ LRS char driver, 1e-10.
- Pi_- vanishes at Σ_-=0, N_φ=16 (< 1e-10).
- Swap symmetry (Σ_+, Σ_-) ↔ (Σ_+, -Σ_-) invariant Y_p.
- Small-shear check: (Σ_+, Σ_-) = (0.05, 0.05) vs
  linearised-PSTF generic (n_ell=3) driver; direction agreement
  and magnitude within 30 %% (linearised captures ~70–80 %% of char).
- Large-shear check: (Σ_+, Σ_-) = (0.3, 0.1) completes with
  Friedmann constraint Ω > 0 throughout.
```

---

## 4. WBS

1. **Non-LRS J_{jk} derivation** — record in `docs/audit/PR-N2_stage3.md`.
2. **Layout helper** `_char_layout_nonlrs`.
3. **RHS kernel** `_rhs_core_nonlrs`.
4. **Cache key + factory** extension (tier-1 / tier-2 specialised).
5. **Config `__post_init__` update** to accept `Σ_- ≠ 0` in
   non-LRS mode.
6. **Backend capability registration.**
7. **Dispatch wiring** in `forward_likelihood.py`.
8. **Parity tests** — all five cases above.
9. **Documentation updates.**

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read paper §2.1–§2.2, §3.2, §6 (all of §6 for direction equations).
- Grep for existing generic-Type-I PSTF references:
  ```
  Grep "Sigma_minus|Pi_minus|n_ell=3|generic" src/rabbit/jax/
  ```
- Confirm the linearised-PSTF generic path produces sensible Π_-
  for comparison.

### Stage 2 — External
- `WebSearch "Bianchi Type I anisotropy two shear components geodesic
  Boltzmann quadrupole"` — look for the Π_- formula in published
  cosmology literature (Rothman–Matzner 1982, Wainwright–Ellis).
- `WebSearch "PSTF multipole Bianchi real spherical harmonic Y_2^2"` —
  confirm the √3/2 normalisation.

### Stage 3 — Self CoT
- **Re-derive J_{jk}** analytically from the exponential direction
  map.  Hint: on S², the Jacobian determinant of the map ê_0 → ê
  via axis stretchings is `(a_x a_y a_z) / ‖(a_x ex_0, a_y ey_0,
  a_z ez_0)‖³`.  With `a_x a_y a_z = 1` (shear is trace-free,
  ∑ λ_i = 0, hence ∏ a_α = 1), the determinant reduces to
  `1 / ‖stretched ê_0‖³`.  Verify by numerical differentiation of
  `mu_phi_current_jax`.
- **Dimensional analysis**: J_{jk}, Π_±, f̃_0 all dimensionless.
- **LRS limit check** at Σ_-=0: confirm mathematically that the
  new J_{jk} reduces to paper eq (51).
- **Swap symmetry**: under Σ_- → -Σ_-, the rotation around the
  z-axis sends φ → -φ; verify that the Π_- integrand
  `sin²θ cos(2φ)` is invariant under φ → -φ, so the solid-angle
  integral is unchanged.

Record in `docs/audit/PR-N2_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] LRS reduction test passes (1e-10).
- [ ] Π_- vanishes at Σ_-=0.
- [ ] Swap symmetry holds.
- [ ] Small-shear agreement with linearised-PSTF.
- [ ] Large-shear (Σ_+=0.3, Σ_-=0.1) numerical stability.
- [ ] Non-LRS J_{jk} analytic derivation committed.
- [ ] STATE_OF_RECORD.md §2.3 extended with non-LRS row.

---

## 7. Adversarial audit prompt

> Audit PR-N2 (non-LRS char driver).  Verify:
> (1) J_{jk} analytic formula derivation,
> (2) LRS reduction at N_φ=1 bitwise,
> (3) Π_- = 0 at Σ_- = 0 to 1e-10,
> (4) Friedmann positivity never violated in the tested Σ range,
> (5) swap-symmetry test and parity test grids all green.
> Cap 400 words.

---

## 8. Anti-local-minimum reminders
1. **Do NOT** discard the analytic J_{jk} in favour of `jax.jacfwd`
   on the direction map.  The cost would be `N_rays` jacfwd calls
   per step — prohibitive at 192 rays.
2. **Do NOT** use the tensor-product grid as default if Lebedev
   would be faster — defer that optimisation to a post-PR-N2 PR.
3. **Do NOT** mix up `S_+` (cumulative Σ_+) with `Σ_+` (current
   shear).  Variable names are load-bearing.

---

## 9. Hallucination prevention
- Every new function (`mu_phi_current_jax`, `jacobian_nonlrs_jax`,
  `extract_stress_minus_S2`) must be grep-able in `PR-N1` or `PR-N2`
  source.  No phantom APIs.
- The analytic J_{jk} must have an explicit derivation in
  `docs/audit/PR-N2_stage3.md`.  If the derivation has a gap, the
  fallback numerical-Jacobian approach must be labelled as such in
  `production_authority="numerical_jac_of_direction_map"` metadata.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §1.2: add "Characteristic non-LRS" row (state DOF
  = N_rays + 4 + N_species).
- §2.3: new subsection "JAX characteristic non-LRS driver".
- §4: extend parity tables with non-LRS rows.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-N2 entry.

### 10.3 `docs/IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md`
Flip status of Phases 2–5 from "planned" to "delivered in PR-N2".

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_n2_nonlrs_driver.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v
pytest tests/ -m "not slow and not gpu" --tb=no -q

test -f docs/audit/PR-N2.md
test -f docs/audit/PR-N2_stage1.md
test -f docs/audit/PR-N2_stage2.md
test -f docs/audit/PR-N2_stage3.md
git add docs/audit/PR-N2*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-N2: non-LRS Bianchi Type I characteristic driver

Wires the non-LRS primitives from PR-N1 into a new driver mode
transport_mode="characteristic_nonlrs" on top of
src/rabbit/jax/driver_typeI_char.py. Evolves two shear amplitudes
(Sigma_+, Sigma_-), accumulates two shear integrals (S_+, S_-) for
the S^2 direction map, feeds both Pi_+ and Pi_- into the geometry
sector, and uses the S^2-weighted monopole for live weak rates.
LRS reduction at N_phi=1 preserved bitwise; non-LRS parity with
the linearised-PSTF generic path locked at small shear.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- LRS reduction at N_φ=1 fails at 1e-10.
- Π_- does not vanish at Σ_-=0.
- Friedmann constraint Ω > 0 violated during integration at any
  tested (Σ_+, Σ_-).
- Non-LRS analytic J_{jk} derivation cannot be closed without a
  gap.

Abort → `docs/audit/PR-N2_abort.md`.
