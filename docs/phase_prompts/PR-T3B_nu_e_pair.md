# PR-T3B — ν–e Elastic + Pair Collision Wiring (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

PR-T3A delivered the full-phase-space ray driver with q-advection
and a collisionless sanity gate.  This PR wires the two existing
collision operators — Hannestad–Madsen ν–e elastic and the pair
process — into that driver so it physically evolves the neutrino
distribution and produces `N_eff ≈ 3.044` at FLRW.

Invariants: Rodas5P, CPU-preferred (single solve expected to be
several seconds per solve; batch mode optional via PR-G), float64,
|ΔY_p| < 5 × 10⁻⁴ vs at least one cross-code (LASAGNA /
FortEPiaNO / PRIMAT-AC2024).

Dependencies: PR-T3A (strict); PR-J (strongly recommended —
tier-3 jacfwd cost is prohibitive without analytic blocks).

---

## 1. Phase objective

Extend the full-Boltzmann RHS with the Hannestad–Madsen ν–e kernel
and the pair-process operator applied per-ray, per-momentum via
the gather–collide–scatter (GCS) cycle (paper Appendix E.3,
eq 185–188).  Also wire the collision energy transfer into
`coupled_3T_rhs_jax` so that the 3T thermo self-consistently
receives the plasma ↔ neutrino heat flux.

Target: FLRW `|N_eff - 3.044| < 0.01` at CL0.

---

## 2. Literature anchors

### 2.1 Internal
- Paper §7 (collision integral structure, pages 19–21).
- Paper §8 (Hannestad–Madsen kernel, eq 64–70).
- Paper §9 (pair process, eq 71–74).
- Paper §11.3 (entropy conservation with collision source term,
  eq 80–84).
- Paper Appendix E.3 — GCS cycle, eq 185–188.
- Existing SciPy operators (to reuse, not reimplement):
  - `src/rabbit/collisions/nu_e_scattering.py` — `NuEScatteringOperator`.
  - `src/rabbit/collisions/pair_processes.py` — `PairProcessOperator`.
  - `src/rabbit/collisions/kernels.py` — `CollisionOperator` protocol.
- JAX 3T with external energy sources:
  `src/rabbit/jax/nudec_coupled_jax.py::coupled_3T_rhs_jax`.
  Current signature hard-codes `dQ = 0` from collisions; must be
  extended to accept external sources (backward-compatible via
  optional kwargs).

### 2.2 External
- **Mangano et al. 2005** `Nucl. Phys. B 729, 221`:
  reference calculation for the FLRW `N_eff = 3.044` baseline with
  elastic + pair collisions.
- **Froustey, Pitrou, Volpe 2020** `JCAP 12:015`: FortEPiaNO
  methodology; confirms the GCS-style approach for tier-3 accuracy.
- **Escudero 2019** `JCAP 02:007`: LASAGNA; uses similar structure
  but with QKE on top (tier-4) — ignore the oscillation parts.
- **Paper ref [15]**: Hannestad & Madsen 1995, `PRD 52:1764` —
  original kernel derivation.
- **Paper ref [16]**: Nötzold & Raffelt 1988, `NPB 307:924` —
  coupling coefficients.

### 2.3 Paper-equation cross-check
- [ ] Eq 64: angular-integrated `|M|²` for ν–e elastic.
- [ ] Eq 70: reduced 2D collision integral
      `C_scat[f](y₁) = G_F² T⁴ / (4π³) · ∫∫ dy₂ dy₃ Θ(y₄) |M|²·S(f)`.
- [ ] Eq 72–73: pair process integrand and statistical factor.
- [ ] Eq 84: `dQ/dN = -T_ν⁴/(2π² H) Σ_α ∫ y³ C_α[f_α](y) dy`.

---

## 3. Skeleton code

### 3.1 Extend `coupled_3T_rhs_jax` with optional external sources

```python
# src/rabbit/jax/nudec_coupled_jax.py — backward-compatible extension
@jax.jit
def coupled_3T_rhs_with_sources_jax(
    T_gamma, T_nu_e, T_nu_x,
    dQ_nue_external=0.0, dQ_nux_external=0.0,
    H_MeV=None,
):
    """3T RHS with injectable external collision energy sources.

    dQ_*_external : MeV⁵ per unit N (per e-fold), *positive* when
                    heat flows INTO the corresponding neutrino
                    species.  When zero, reduces to the existing
                    Mangano momentum-averaged model.
    """
    # ... (identical to coupled_3T_rhs_jax body, but add the
    # external sources to dQ_nue / dQ_nux BEFORE computing the
    # dT/dN corrections).
```

Keep the original `coupled_3T_rhs_jax` as a thin wrapper for
backward compatibility:
```python
@jax.jit
def coupled_3T_rhs_jax(T_gamma, T_nu_e, T_nu_x, H_MeV=None):
    return coupled_3T_rhs_with_sources_jax(
        T_gamma, T_nu_e, T_nu_x, 0.0, 0.0, H_MeV=H_MeV,
    )
```

### 3.2 JAX wrappers for existing operators

The existing SciPy operators are `numpy.ndarray`-based.  For tier-3
use we either:
- Wrap them as `jax.pure_callback` (allows NumPy inside JIT but
  incurs host-roundtrip per RHS call — potentially prohibitive for
  Rodas5P's ~4500 RHS evals per solve).
- Port them to JAX (`jax.numpy` equivalents).

**Recommended: port to JAX.**  The kernels are 1D Laguerre
integrations, straightforwardly JITtable.  Existing SciPy tests
([`tests/test_j02_jax_rhs.py`](../../tests/test_j02_jax_rhs.py)
and related) provide ground-truth cases.

```python
# src/rabbit/jax/collisions_jax.py — new module
"""JAX ports of the Hannestad–Madsen ν–e kernel and pair process.

Reproduces rabbit.collisions.nu_e_scattering.NuEScatteringOperator
and rabbit.collisions.pair_processes.PairProcessOperator at
≤ 1e-12 elementwise on the shared Laguerre grid.  The ports are
jitted and vmap-compatible per q-node.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Paper eq 68–69: Standard Model coupling coefficients
# (values locked to the SciPy source):
#   a_e = 1 + 4 sin²θ_W + 8 sin⁴θ_W ≈ 2.353
#   a_x = 1 − 4 sin²θ_W + 8 sin⁴θ_W ≈ 0.503
A_E = 2.3530
A_X = 0.5030


@jax.jit
def C_nu_e_scattering_jax(
    f_mono: jnp.ndarray,      # (N_q,)  monopole f̃_0(q)
    T_gamma: jnp.ndarray,     # MeV
    T_nu: jnp.ndarray,        # MeV (flavour-specific; tier-2 uses T_νₑ or T_νₓ)
    q_nodes: jnp.ndarray,     # (N_q,)  Laguerre on y = p / T_ν
    q_weights: jnp.ndarray,   # (N_q,)  Laguerre weights
    coupling: float = A_E,    # A_E for ν_e, A_X for ν_x
) -> jnp.ndarray:
    """C_0[f̃_0](q) for ν + e⁻ → ν + e⁻ via Hannestad–Madsen.

    Returns (N_q,) array; same shape / order as q_nodes.
    """
    # Implement paper eq 70 with the flavour coupling.  Cross-check
    # against rabbit.collisions.nu_e_scattering.NuEScatteringOperator
    # at 1e-12 for {f=FD(T_nu), T_γ=T_ν, q_nodes=Laguerre-20} input.
    ...


@jax.jit
def C_pair_process_jax(
    f_nue_mono: jnp.ndarray,
    f_nuebar_mono: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu: jnp.ndarray,
    q_nodes: jnp.ndarray, q_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """C_pair[f_nue, f_nuebar](q) for ν + ν̄ ↔ e⁺ + e⁻.

    Returns (C_for_nue, C_for_nuebar); sign convention: positive
    means heating of that species from the pair rest mass."""
    ...
```

### 3.3 GCS loop inside the full-Boltzmann RHS

```python
# driver_typeI_full_boltzmann.py — extended RHS
def _rhs_core_full_boltzmann_tier3(
    N, y, *,
    ...,
    enable_nu_e: bool = True,
    enable_pair: bool = True,
    enable_nu_nu: bool = False,   # False in PR-T3B, True in PR-T3C
    ...
):
    # ... (advection + direction map, as PR-T3A)

    # ── Step 1: GATHER (paper eq 185–186) ──
    # For each species, angle-average over rays to obtain f̃_0(q)
    # and f̃_2(q).
    weight = w0 * jacobian_jax(X0, S_val, mu)        # (N_mu,)
    f_nue_rays = extract_species_rays(y, "nue", layout)  # (N_mu, N_q)
    f_tilde_0_nue = 0.5 * (weight @ f_nue_rays)         # (N_q,)
    P2 = 0.5 * (3.0 * mu * mu - 1.0)
    f_tilde_2_nue = (5.0 / 2.0) * (weight * P2) @ f_nue_rays
    # ... similarly for nuebar, nux, nuxbar.

    # ── Step 2: COLLIDE (paper eq 187) ──
    # Apply the collision operators on f̃_0 (and optionally f̃_2).
    C0_nue = jnp.zeros_like(q_nodes)
    C0_nuebar = jnp.zeros_like(q_nodes)
    C0_nux = jnp.zeros_like(q_nodes)
    if enable_nu_e:
        C0_nue   += C_nu_e_scattering_jax(f_tilde_0_nue,   T_gamma, T_nu_e, q_nodes, q_weights, A_E)
        C0_nuebar += C_nu_e_scattering_jax(f_tilde_0_nuebar, T_gamma, T_nu_e, q_nodes, q_weights, A_E)
        C0_nux   += C_nu_e_scattering_jax(f_tilde_0_nux,   T_gamma, T_nu_x, q_nodes, q_weights, A_X)
    if enable_pair:
        C_pair_nue, C_pair_nuebar = C_pair_process_jax(
            f_tilde_0_nue, f_tilde_0_nuebar, T_gamma, T_nu_e, q_nodes, q_weights,
        )
        C0_nue    += C_pair_nue
        C0_nuebar += C_pair_nuebar
        # pair for ν_x analogous with T_nu_x
    # ν-ν is added in PR-T3C.

    # ── Step 3: APPLY (paper eq 188) ──
    # Add C_0(q) + C_2(q) P_2(μ_j) to each ray's f_j(q).  C_2 is the
    # quadrupole projection; kept zero here (PR-T3C may activate).
    df_nue_rays = (C0_nue / H)[None, :]     # broadcast over rays
    # ... similarly for nuebar, nux, nuxbar.

    # Pack back into dy
    dy = _pack_species_rays(dy, df_nue_rays, "nue", layout)
    ...

    # ── Step 4: 3T thermo with external Q sources ──
    # Paper eq 84: dQ_α/dN = -T_α⁴/(2π² H) · ∑ y³ C_α[f_α](y) dy
    dQ_nue_dN = -(T_nu_e**4 / (2.0 * jnp.pi**2 * H)) * jnp.sum(q_weights * q_nodes**3 * C0_nue)
    # pair contributes to dQ_{ν_α}; paper eq 84 sums all C_α.
    # Sign convention: dQ_α > 0 ⇔ α receives energy from the plasma.
    dQ_nux_dN = ...
    dT_gamma, dT_nu_e, dT_nu_x = coupled_3T_rhs_with_sources_jax(
        T_gamma, T_nu_e, T_nu_x,
        dQ_nue_external=dQ_nue_dN,
        dQ_nux_external=dQ_nux_dN,
        H_MeV=H_MeV,
    )

    # ... (network RHS, same as PR-T3A).
```

### 3.4 Ground-truth tests against SciPy operators

```python
# tests/test_pr_t3b_jax_operator_parity.py
@pytest.mark.parametrize("species_coupling", [("nue", "A_E"), ("nux", "A_X")])
def test_nu_e_jax_matches_scipy(species_coupling):
    """1e-12 elementwise parity: JAX C_nu_e_scattering vs
    SciPy NuEScatteringOperator on the same Laguerre grid, same
    input distribution."""
    ...
```

### 3.5 FLRW N_eff lock

```python
# tests/test_pr_t3b_flrw_neff.py
def test_flrw_neff_3044_at_cl0():
    """FLRW tier-3 with elastic + pair must produce N_eff = 3.044
    ± 0.01 at CL0.  Reference: Mangano 2005."""
    ...
```

---

## 4. WBS

1. **JAX ports of ν–e and pair operators** (`collisions_jax.py`).
2. **Operator-parity tests** (1e-12 vs SciPy).
3. **`coupled_3T_rhs_with_sources_jax`** (backward-compatible
   extension of `coupled_3T_rhs_jax`).
4. **GCS loop wiring** in the full-Boltzmann RHS.
5. **FLRW N_eff lock test** (target 3.044 ± 0.01).
6. **Energy-momentum conservation test**: stress–energy residual
   per step < 10⁻⁸ of ρ_rad.
7. **Detailed-balance regression**: at thermal equilibrium,
   `C[f_eq] = 0` to 10⁻¹² per step.
8. **LASAGNA / FortEPiaNO cross-code pass** (if external data
   available in the repo fixtures; otherwise a published Y_p / N_eff
   number from the paper bibliography).
9. **Documentation updates**.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read paper §7–§9 (pages 19–23) and §11.3 (pages 23–25).
- Read `rabbit.collisions.nu_e_scattering.NuEScatteringOperator`
  to extract the exact kernel form used in the SciPy reference.
- Read `rabbit.collisions.pair_processes.PairProcessOperator`.
- Read `nudec_coupled_jax.coupled_3T_rhs_jax` to design the
  extension signature without breaking the existing caller
  (characteristic tier-2 driver and the linearised PSTF tier-2
  kernels).

### Stage 2 — External
- `WebSearch "Hannestad Madsen 1995 neutrino-electron scattering
  angular integration"` — confirm the reduced 2D form.
- `WebSearch "Mangano 2005 N_eff 3.045 incomplete decoupling
  precision"` — confirm the 3.044 SM benchmark (note: paper has
  since been refined — the current SM value is 3.044, with minor
  updates to 3.043 in Froustey 2020).
- `WebSearch "FortEPiaNO LASAGNA benchmark N_eff Y_p standard BBN"`
  — collect published Y_p / N_eff numbers at matched `η`.
- `WebSearch "pair process e+e- annihilation neutrino heating BBN
  statistical factor"` — confirm sign conventions.

### Stage 3 — Self CoT
- **Sign audit.**  `dQ_α` positive when α *receives* energy.  At
  `T_γ > T_ν`, plasma heats neutrinos, so `dQ_νₑ > 0` and the
  total rhs contribution to `T_νₑ` is positive (δT_ν > 0).
  Verify the sign through the Hannestad–Madsen kernel form:
  integrand `∝ (f_plasma - f_ν) · phase-space`, positive when
  plasma is hotter → correct.
- **Dimensional audit.**  `dQ_α` has units of `[ρ]·[Hubble] =
  MeV⁴·MeV = MeV⁵`.  Dividing by `H_MeV` gives `MeV⁴` = energy
  density per e-fold ✓.
- **FLRW baseline.**  At Σ = 0, all rays have I_j = 0, `J_j = 1`,
  angular distribution is isotropic.  The GCS cycle reduces to
  C_0 acting on an isotropic distribution → back to isotropic.
  The evolution must land at `N_eff = 3.044`.  If it does not,
  something is wrong with the sign or the coupling coefficient.
- **Detailed balance.**  Substitute `f_ν = f_FD(q)` at `T_ν = T_γ`.
  Every collision operator must return `C = 0`.  Implement this
  as a unit test.
- **Adversarial check: quadrupole.**  If `enable_nu_nu=False` and
  `enable_pair=True`, the GCS cycle only injects C_0 (monopole).
  Per-ray C_0 / H is added uniformly across μ_j — no angular
  distortion.  That's the tier-3-without-ν-ν regime.  Verify
  `f̃_2(q)` stays at its collisionless value (not artificially
  driven).

Record in `docs/audit/PR-T3B_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] JAX ν–e operator matches SciPy at 1e-12.
- [ ] JAX pair operator matches SciPy at 1e-12.
- [ ] Detailed balance `C[f_eq] = 0` to 1e-12 at multiple T.
- [ ] FLRW `|N_eff - 3.044| < 0.01` at CL0.
- [ ] Stress–energy conservation ≤ 10⁻⁸ of ρ_rad per step.
- [ ] Anisotropic regression: tier-3 at Σ = 0.1 completes and
      `N_eff` shift < 10⁻³.
- [ ] `coupled_3T_rhs_jax` caller compatibility preserved (existing
      tier-2 driver unaffected).

---

## 7. Adversarial audit prompt

> Audit PR-T3B (ν–e + pair collision wiring for the full-phase-space
> driver).  Verify:
> (1) JAX operator ports match SciPy at 1e-12 elementwise;
> (2) detailed balance `C[f_eq] = 0` at multiple T;
> (3) FLRW `N_eff = 3.044 ± 0.01`;
> (4) sign convention of `dQ_α`: plasma hotter → `dQ_ν > 0`;
> (5) Rodas5P unchanged, step rejection rate monitored near
> freeze-out;
> (6) `coupled_3T_rhs_jax` backward-compatible.
> Cap 500 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** use `jax.pure_callback` to wrap the SciPy operators
   as a shortcut.  Rodas5P will call the RHS thousands of times
   per solve; host-roundtrips will dominate.
2. **Do not** assume detailed balance is automatic because it's
   "supposed to be".  Test it explicitly with a FD equilibrium
   input at multiple T.
3. **Do not** inflate `N_eff` to 3.044 by manually setting an
   overall multiplicative factor.  If `N_eff` is off, the
   coupling coefficients or the rate normalisation is wrong; fix
   the root cause.

---

## 9. Hallucination prevention

- Do not invent a `compute_nu_e_kernel_jax` function if the name
  does not exist.  Grep the repo first.
- Do not cite `Mangano 2005` without reading the abstract for the
  exact `N_eff` number claimed.
- Do not skip the SciPy-parity test.  Elementwise 1e-12 agreement
  is the gate; "close enough" is not.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §1.2: "Full phase-space ray" row flips to
  "tier-3 with ν-e + pair in PR-T3B".
- §2.3: add subsection describing the full-Boltzmann driver with
  collision coupling.
- §4: parity table gains a tier-3 FLRW row.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-T3B entry.  Record measured `N_eff` and comparison
against external references.

### 10.3 `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
Flip Phase 2 + Phase 4 status to "delivered in PR-T3B".

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_t3b_jax_operator_parity.py -v
pytest tests/test_pr_t3b_flrw_neff.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v

test -f docs/audit/PR-T3B.md
test -f docs/audit/PR-T3B_stage1.md
test -f docs/audit/PR-T3B_stage2.md
test -f docs/audit/PR-T3B_stage3.md
git add docs/audit/PR-T3B*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-T3B: wire ν-e elastic + pair collisions into the full-phase-space
driver

Adds JAX ports of the Hannestad-Madsen ν-e scattering kernel and
the pair-process operator in src/rabbit/jax/collisions_jax.py
(element-wise match to the SciPy reference at 1e-12).  Extends
nudec_coupled_jax.coupled_3T_rhs_jax with backward-compatible
external dQ sources; the full-Boltzmann RHS now feeds collision
heat flux per eq 84 of the paper.  FLRW CL0 locks
|N_eff - 3.044| < 0.01 against the Mangano 2005 benchmark.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- SciPy ↔ JAX operator parity worse than 1e-12.
- Detailed balance `C[f_eq] = 0` worse than 1e-12.
- FLRW `|N_eff - 3.044| > 0.015`.
- Stress–energy conservation residual > 10⁻⁸ ρ_rad per step.
- `coupled_3T_rhs_jax` regression breaks existing tier-2 tests.

Abort → `docs/audit/PR-T3B_abort.md`.
