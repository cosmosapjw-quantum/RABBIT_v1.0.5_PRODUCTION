# Implementation Guide — Fully Nonperturbative Incomplete Decoupling (Full Collision Term, no QKE)

**Scope:** Replace RABBIT's current tier-1 / tier-2 collision
approximations with a direct evaluation of the full Boltzmann
collision term `C[f](q, μ)` on each characteristic ray at each
momentum node.  No Teff compression, no QKE closure, no flavour
oscillations.  This implements **Paper II** (paper §12.10, Appendix E)
and corresponds to **Tier 3** of paper Table 2 (tier-2 plus diagonal
ν–ν scattering, no oscillations).

**Physics foundation:** RABBIT report §7–§9 (collision integrals,
Hannestad–Madsen kernel, pair processes), §11 (incomplete decoupling
and N_eff), §12.4.2 (the exact per-momentum scatter alternative),
Appendix E (full phase-space ray Boltzmann equation).

---

## 1. What "fully nonperturbative, no QKE" means precisely

The Boltzmann equation for a massless neutrino on a Bianchi Type I
background is (paper eq 59, 107):

```
∂f/∂N + (d ln q / dN) · q ∂f/∂q + (dμ/dN) ∂f/∂μ = C[f] / H
```

with collision integral (paper eq 60):

```
C[f_α](p₁) = (1/2E₁) Σ_{β,γ,δ} ∫ d³p₂ d³p₃ d³p₄ / (2π)⁹
             × δ⁴(p₁+p₂-p₃-p₄) |M|² S(f₁,f₂,f₃,f₄)
S = f₃ f₄ (1-f₁)(1-f₂) - f₁ f₂ (1-f₃)(1-f₄)
```

**"Fully nonperturbative"** means `C` is evaluated on the *actual*
distribution, not linearised around equilibrium; no temperature
expansion, no Teff scatter projection, no κ_ℓ Γ/H damping fit.

**"Full collision term"** covers the three SM processes (paper §7):
1. Elastic `ν_α + e^± → ν_α + e^±` (all flavours)
2. Pair `ν_α + ν̄_α ↔ e⁺ + e⁻`
3. Diagonal ν–ν `ν_α + ν_β → ν_α + ν_β` (no flavour exchange)

**"No QKE"** excludes the quantum kinetic density-matrix formulation
required for flavour oscillations.  Each species α carries its own
scalar distribution `f_α(q, μ, N)` (no off-diagonal coherences).
This is exactly Paper II's scope; Tier 4 (oscillations) is deferred.

Operational restriction for Tier 3 (paper §11.2, Table 2):

- Elastic ν–e  ✓ (Hannestad–Madsen kernel, already exists:
  `NuEScatteringOperator` in [`rabbit/collisions/nu_e_scattering.py`](../src/rabbit/collisions/nu_e_scattering.py))
- Pair ν_α ν̄_α ↔ e⁺e⁻ ✓ (`PairProcessOperator` in
  [`rabbit/collisions/pair_processes.py`](../src/rabbit/collisions/pair_processes.py))
- **Diagonal ν–ν** partial ✓ on the bounded AP-form preflight surface:
  a shared leading-order `T^5` rate prefactor is used by the NumPy 3T
  thermo source and JAX phase-space helper; `ap_unified_nu_nu_preflight`
  carries an energy-conserving 3T equilibration source; and
  `ap_unified_nu_nu_spectral_preflight` carries a spectral-bank source
  projected to the same energy-conserving moment plus same-sector
  number/energy-neutral shape damping.  The full Dolgov-Hansen-Semikoz
  coefficient-table runtime kernel is still unpromoted.
- Off-diagonal ν–ν, oscillation terms ✗ (Tier 4, out of scope)

---

## 2. State-vector upgrade (paper Appendix E)

The state representation must jump from one scalar per ray (I_j) to
the full momentum-resolved distribution per ray (paper eq 183):

```
Per-ray state: f_j(q_1), f_j(q_2), ..., f_j(q_{N_q}),   j = 1..N_μ
```

Total transport DOF at (N_μ=12, N_q=20):
`6 species × 12 rays × 20 momenta = 1 440 DOF` for all six species
separately.

If the Tier-3 model treats three flavour groups
`(ν_e + ν̄_e, ν_μ + ν̄_μ, ν_τ + ν̄_τ)` with per-group asymmetry
handled via the pair channel, that collapses to `6 × 12 × 20` only
if each species is tracked independently (required for CP-asymmetric
weak rates).  A practical split:

- `f_{ν_e}(q, μ)`, `f_{ν̄_e}(q, μ)`
- `f_{ν_x}(q, μ)` shared by (ν_μ, ν_τ), with
  `f_{ν̄_x}(q, μ)` shared by (ν̄_μ, ν̄_τ)

giving 4 effective distributions × 12 × 20 = **960 transport DOF**.
Plus the existing 2 + 1 + 3 + N_species geometry / thermo / network =
**≈ 975 total DOF**.  Paper §D.1 confirms this is "still tractable
for Rodas5P" (Jacobian ≈ 970²  ≈ 9×10⁵ entries).

### 2.1 Advection term

The gravitational redshift piece of the Boltzmann equation (paper
eq 184) becomes a q-advection PDE rather than a scalar ODE:

```
∂f_j/∂N = -(d ln q / dN)_j · q ∂f_j/∂q + C[f]/H
(d ln q / dN)_j = -Σ_+ P_2(μ_j) - Σ_- · [sin²θ cos 2φ] / √3   (LRS: -Σ_+ P_2)
```

This is a **hyperbolic PDE in q**.  The characteristic method used
in Paper I (state = `I_j` scalar) implicitly integrates this term
analytically via `f_j(q, N) = f_0(q e^{2 I_j})`.  The Paper II
formulation gives up that shortcut and integrates the advection
numerically.

**Required:** a stable q-advection treatment that fits the current
Rodas5P contract.

Preflight conclusion (see
[`docs/audit/TIER3_FULL_COLLISION_PREFLIGHT_20260421.md`](audit/TIER3_FULL_COLLISION_PREFLIGHT_20260421.md)):

- **First landed path:** continuous semidiscrete q-advection inside the
  RHS, using a monotone one-sided stencil as the baseline.
- **Reference/oracle path:** exact characteristic remap
  `q → q e^{-2 ΔI}` with monotone PCHIP reconstruction, used for the
  collisionless sanity gate and future split-integrator experiments.

Reason:

- the PCHIP remap is the best transport oracle and preserves
  `f ∈ [0,1]`;
- but a semi-Lagrangian remap is not a natural in-stage operation for
  the current Rosenbrock error/event controller;
- the first tier-3 production attempt should therefore keep a
  continuous RHS and treat PCHIP remap as the regression target.

Without a controlled q-advection path the numerical diffusion or
spurious negativity will wash out the spectral distortion we are trying
to resolve.

### 2.2 Collision evaluation

The collision integral requires the distribution at *all directions*
— a ray at direction μ_j cannot compute `C[f](q, μ_j)` from its own
data alone.  Paper §12.3 resolves this via the **gather–collide–
scatter (GCS) cycle** already prototyped in
[`rabbit/transport/teff_collision_bridge.py`](../src/rabbit/transport/teff_collision_bridge.py),
now generalised to work at each q-node rather than via Teff
compression (paper Appendix E.3, eq 185–188):

**Gather** (angular moments, exact):

```
f̃_0(q)        = ½ Σ_j w_j J_j f_j(q)
f̃_2(q)        = (5/2) Σ_j w_j J_j P_2(μ_j) f_j(q)
```

**Collide** (multipole truncation at ℓ=2, exact at 𝒪(Σ)):

```
C[f](q, μ)    = C_0[f̃_0](q) + C_2[f̃_0, f̃_2](q) P_2(μ) + 𝒪(Σ²)
C_0[f̃_0](q)  = NuEScatteringOperator(f̃_0) + PairProcessOperator(f̃_0, f̃_0̄) + NuNuOperator(f̃_0, f̃_0̄)
C_2[...] (q) = analogous with the quadrupole source
```

At tier-3 scope and for Σ ≤ 0.5, the ℓ=2 truncation of C is accurate
to 𝒪(Σ²) ~ 0.25 — *that bound is not small*.  We therefore include
C_2 (not just C_0 as in the Teff bridge) and assess convergence
numerically at Σ = 0.3, 0.5.

**Apply** (exact):

```
df_j(q_k)/dN += [C_0[f̃_0](q_k) + C_2[f̃_0, f̃_2](q_k) P_2(μ_j)] / H
```

No Teff projection, no gather-scatter loss.  Spectral distortion is
captured per (q_k, μ_j) pair.

### 2.3 ν–ν scattering operator status

The ν–ν contribution is no longer absent, but it is still bounded.  The
landed path contains a leading-order diagonal no-QKE `ν-ν` rate
normalization, a 3T equilibration source, and a spectral-bank AP source with
number/energy-neutral same-sector shape damping.  It is not yet the full
Mangano/Dolgov-Hansen-Semikoz coefficient-table collision kernel.  The
remaining full-kernel derivation follows Mangano et al. 2005, paper refs
[17,18]:

```
C_{νν}[f_α](p₁) = ½ Σ_β ∫ d³p₂/(2π)³ · (phase factor)
                × |M_{αβ}|² S(f₁,f₂,f₃,f₄)
|M_{αβ}|² ∝ G_F² (p₁·p₂)(p₃·p₄)      (Fierz-diagonal limit, Tier 3)
```

In the ultra-relativistic massless limit and after the angular
integration (analogous to Hannestad–Madsen for ν–e), the rate per
flavour scales as

```
Γ_{νν} ∝ G_F² T_ν⁵,   so  Γ_{νν}/Γ_{νe} ≈ (T_ν / T_γ)⁴ · (g_F(νν) / g_F(νe))
```

At T ~ 1 MeV this is ~25 % of the ν–e rate — significant but
not dominant.  The remaining implementation is a direct analogue of
`NuEScatteringOperator`:

1. 1D Laguerre integral over incoming-neutrino momentum `y_2`.
2. Scalar matrix element (no electron-mass phase space, so simpler).
3. Statistical factor `S(f_1, f_2, f_3, f_4)` using the
   gathered monopole `f̃_0(q)` for the other neutrino.

Remaining estimated size: ~250 LOC mirroring
`rabbit/collisions/nu_e_scattering.py`.

---

## 3. Implementation plan

### Phase 0 — architectural decisions (1 day)

1. **Species grouping.**  Lock on `(ν_e, ν̄_e, ν_x, ν̄_x)` with
   ν_x = ν_μ = ν_τ by symmetry.  This keeps CP-asymmetric weak
   rates (ν_e / ν̄_e) exact while collapsing the μ/τ sector.
2. **q-advection scheme.**  Commit to a continuous semidiscrete
   monotone stencil inside Rodas5P for the first landed path.
   Keep exact PCHIP remap as the collisionless oracle and as the
   future split/IMEX upgrade target.
3. **Multipole truncation of C.**  Keep C_0 + C_2.  Benchmark the
   error from dropping C_4 at Σ = 0.5.

### Phase 1 — q-advection kernel (delivered partially in PR-T3A)

`rabbit/jax/q_advection_jax.py`:

```python
def apply_continuous_q_advection_jax(f_species_rays, coeff_per_ray, q_forward_op, q_backward_op):
    """Continuous RHS transport  ∂_N f = a_j q ∂_q f  with upwind inflow clamps."""


def semi_lagrangian_q_advect(f, q_nodes, dI):
    """Exact-remap PCHIP oracle retained for regression, not in-stage Rodas5P."""
```

Delivered scope:
- continuous upwind semidiscrete transport on the Laguerre grid
- physical inflow boundary conditions (`f(q<q_min)=f(q_min)`,
  `f(q>q_max)=0`)
- monotone PCHIP oracle for collisionless regression

Not delivered yet:
- high-accuracy long-horizon oracle round-trip gate
- split/IMEX remap inside the solver loop

### Phase 2 — per-ray distribution driver (delivered partially in PR-T3A)

`rabbit/jax/driver_typeI_full_boltzmann.py`:

- Current landed state layout: `[Σ_+, Σ_-, f_j(q_k) for all species,
  rays, momenta, S, T_γ, X_i]`
- Phase 1 / Phase 2 handoff: network expands; transport state
  preserved element-wise.
- Current landed kernel `_rhs_core_full_boltzmann_collisionless`:
  - Advection: applied to each species-ray q block.
  - Gather: shared transported monopole from the explicit ray state.
  - Collide: none yet; collision scope is explicitly
    `collisionless_only_v1`.
  - Apply: geometry + tier-1 thermo + phase-1/phase-2 network.

Current PR-T3A limits:
- CPU-preferred only
- LRS only
- tier-1 thermo only
- private experimental surface only; not registered as a backend key
- factorized Jacobian payload now uses an analytic dense q-advection
  base block plus a nonzero projected low-rank update through the
  collisionless `stress + monopole(q)` moment surface

At `N_μ=12, N_q=20` this landed shell is `966` DOF in phase 1 and
`973` DOF in phase 2.  The full `975`-DOF target still requires 3T
thermo and live collision operators.

### Phase 3 — ν–ν scattering operator (~300 LOC + 200 tests)

`rabbit/collisions/nu_nu_scattering.py`:

- Port Mangano 2005 kernel.
- Gauss–Laguerre over the partner-momentum direction.
- Detailed-balance test: `C_{νν}[f_eq] = 0` to 1e-14.
- Energy-conservation test: `∫ y³ C_{νν} dy = 0` to 1e-12
  (elastic within the neutrino sector).

### Phase 4 — collision source to 3T thermo (~100 LOC)

Wire:

```
dQ_{νₑ→plasma}/dN = -∫ (T_νₑ⁴ / 2π²) · y³ · C_{νₑ}[f](y) dy
dQ_{νₓ→plasma}/dN = -∫ (T_νₓ⁴ / 2π²) · y³ · C_{νₓ}[f](y) dy (per species)
dQ_{γ→plasma}/dN  = -Σ_α dQ_{ν_α→plasma}/dN
```

Connect to `coupled_3T_rhs_jax` (see [3T thermo
guide](IMPLEMENTATION_GUIDE_3T_THERMO.md#phase-4-collision-source-term)).

### Phase 5 — live weak rates on evolved spectrum (~40 LOC)

Already live in the repo — `compute_live_rates_from_monopoles_cl*_jax`
accepts any monopole `f_nue_monopole, f_nuebar_monopole`.  Just pass
the gathered f̃_0 from the evolved state.  The existing parity
infrastructure (CL0–CL3) needs no modification.

### Phase 6 — validation (~600 LOC tests)

Five critical gates (paper §11.6, §12.9, §20.2):

1. **Collisionless reduction.**  Disable all three collision
   operators → results must match the Paper I characteristic driver
   bitwise (both are exact solutions of the same collisionless
   Boltzmann equation when evolved on identical grids).
2. **FLRW N_eff recovery.**  With collisions on, Σ = 0 →
   `|N_eff - 3.044| < 0.01` matching SM incomplete-decoupling
   baselines (Mangano et al. 2005, Froustey et al. 2020).
3. **Cross-code parity.**  Compare final (Y_p, D/H) against
   LASAGNA / FortEPiaNO at FLRW (paper [18, 19]).  Publication
   threshold: `|ΔY_p| < 5e-4`, `|ΔD/H|/D/H < 1e-2`.
4. **Energy–momentum conservation.**  Each timestep:
   `|Σ_α dρ_α/dN + H · Σ_α (ρ_α + p_α)| < 1e-8 · H · ρ_rad`.
5. **Spectral distortion sanity.**  Evolved `f_νₑ(q)` vs
   equilibrium: excess at q > 3, deficit at q < 3
   (paper Figure 8 schematic).

### Phase 7 — dispatch (~100 LOC)

- New backend key `jax_full_collision_tier3`.
- New capability `JAX_TYPEI_FULL_BOLTZMANN_TIER3`.
- Metadata: `transport_mode="full_boltzmann_ray"`,
  `collision_closure_mode="full_nonperturbative_tier3_no_qke"`,
  `production_authority="paper_II_candidate"`.

---

## 4. Test matrix

| Σ_+ | Collisions | Target |
|---|---|---|
| 0.0 | off (all three) | bitwise match to Paper I characteristic driver |
| 0.0 | ν–e only (tier 2) | match SciPy tier-2 ν–e result |
| 0.0 | ν–e + pair | FLRW N_eff = 3.044 ± 0.01 |
| 0.0 | full tier-3 | cross-code parity vs LASAGNA / FortEPiaNO |
| 0.1 | full tier-3 | anisotropic N_eff stability |
| 0.3 | full tier-3 | C_4 truncation assessment |
| 0.5 | full tier-3 | nonlinear-shear limit |

---

## 5. Risk register

- **q-advection accuracy/stability tradeoff.**  A continuous one-sided
  stencil is the correct first fit for the current Rodas5P contract,
  but it is diffusive.  Centered stencils are sharper and can break
  positivity.  **Mitigation:** use a monotone one-sided baseline for
  the first landed driver, lock the collisionless reduction against the
  exact PCHIP remap oracle, and revisit split/IMEX semi-Lagrangian only
  after a dedicated solver hook exists.
- **Stiffness from ν–e at T_γ ≳ 3 MeV.**  The coupled equations are
  very stiff when Γ_νe / H ≫ 1.  Rodas5P handles this — the
  Paper II motivation for keeping the Rosenbrock solver.
  Monitor rejection rate near the handoff.
- **Memory footprint.**  975-DOF state × 8 Rodas5P stages × 4 Gustafsson
  buffers ~ 30 × 10³ doubles ≈ 240 KB per solve.  Jacobian 975² × 8 B
  ≈ 7.6 MB — fine on any modern CPU/GPU.
- **`jacfwd` cost.**  975 tangent directions per step.  **This is the
  dominant compute cost.**  Custom analytic Jacobian blocks (see
  [GPU optimization plan §2.4](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md#24-analytic-jacobian-in-place-of-jacfwd))
  become **mandatory**, not optional.  Without analytic Jacobian the
  single-solve cost is prohibitive (≈ 1 hour on CPU).
- **Detailed balance drift.**  Long integrations can accumulate
  numerical error that violates `C[f_eq] = 0`.  Monitor
  `|C[f̃_0]|_∞` when all three distributions equal equilibrium at
  the same temperature — should stay < 10⁻¹² per step.

---

## 6. LOC budget and timeline

| Phase | Description | LOC | Days |
|---|---|---|---|
| 0 | Architectural lock-in | — | 1 |
| 1 | q-advection kernel | 400 + 200 | 2.5 |
| 2 | Per-ray distribution driver | 500 | 3 |
| 3 | ν–ν scattering operator | 300 + 200 | 2.5 |
| 4 | Collision source → thermo | 100 | 0.5 |
| 5 | Live weak re-wiring | 40 | 0.25 |
| 6 | Validation tests | 600 | 3 |
| 7 | Dispatch + metadata | 100 | 0.5 |
| **Phase 8 (required)** | Analytic Jacobian blocks | 700 | 4 |
| **Total** | | **≈ 3 140** | **≈ 17** |

The analytic Jacobian (Phase 8) is listed as *required*, not optional.
Without it, the tier-3 driver is too slow to be useful.

---

## 7. Cross-references

- Tier-3 *requires* the 3T thermo layer
  ([`IMPLEMENTATION_GUIDE_3T_THERMO.md`](IMPLEMENTATION_GUIDE_3T_THERMO.md))
  to be already in place.
- Tier-3 does **not** require non-LRS — it can first be validated at
  LRS (Σ_- = 0), then extended to non-LRS using the S² quadrature
  from the non-LRS guide.
- GPU execution for tier-3 grids ≥ 64 solves benefits materially —
  see [`JAX_CHAR_GPU_OPTIMIZATION_PLAN.md`](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md).
  The 975-DOF state is large enough that kernel-launch overhead is
  amortised on a single solve at that scale.

---

## 8. Validation cross-check targets (external codes)

| Reference | Y_p (FLRW, standard η) | N_eff (FLRW) |
|---|---|---|
| Paper I current (Teff bridge) | 0.24235 (CL0) | 3.011 (tier-1 helper) |
| LASAGNA (Escudero 2019) | 0.2470 (CL2) | 3.044 |
| FortEPiaNO (Froustey 2020) | 0.2470 ± 2e-4 | 3.043 ± 0.001 |
| PArthENoPE (Consiglio 2018) | 0.2466 (CL3) | 3.044 |
| PRIMAT AC2024 (Pitrou 2024) | 0.24703 ± 6e-5 | 3.044 |

Tier-3 RABBIT target: land inside `0.2470 ± 5e-4` at FLRW with
`N_eff = 3.044 ± 0.005`.

---

## 9. What this does NOT deliver

- **Flavour oscillations (Tier 4).**  Requires the QKE formalism
  with a 2×2 complex density matrix per neutrino.  Explicitly out
  of scope.
- **Off-diagonal ν–ν matrix elements.**  Fierz-off-diagonal terms
  contribute O(%) to N_eff; they require flavour indices and are
  bundled with the QKE extension.
- **CMB-level precision.**  This guide targets `N_eff ~ 3.044 ± 0.005`
  and `Y_p ~ 5×10⁻⁴` — adequate for BBN-only analyses but short of
  the `~1×10⁻³` N_eff precision reachable with full Tier-4 + CMB
  joint fits.
- **Collision anisotropy beyond ℓ=2.**  The C_4 projection is
  neglected; validity probed numerically at Σ = 0.3, 0.5.
- **Radiative corrections to weak rates.**  CL0–CL3 already exist
  and plug in via `live_weak_rates_with_budget`; this guide does
  not touch them.

---

## 10. Staged delivery recommendation

If the 17-day budget is not achievable in one sprint, a staged
release in order of decreasing publication impact:

1. **Stage A (3 days):**  Phases 0, 1, partial 2 — q-advection only,
   no ν–ν, no ν–e (just check advection recovers Paper I).
2. **Stage B (4 days):**  Add ν–e elastic + pair via existing
   operators, validate `N_eff = 3.044` at FLRW.  **This is already
   scientifically publishable** — it's Paper II's primary target.
   Bounded preflight status today:
   - `src/rabbit/jax/full_boltzmann_collision_preflight.py` reuses the
     current species-resolved isotropic backbone on the explicit-shell
     monopole surface.
   - the production-bounded source-only closure has zero state Jacobian
     and therefore cannot yet populate the `U C V` moment-core by itself.
   - an audit-only spectral-relaxation closure yields a finite
     block-diagonal moment-core Jacobian, and its lifted explicit-shell
     `U C V` factorization matches dense finite differences in bounded
     tests without changing any public runtime path.
   - a follow-up private runtime hook now lets the full-Boltzmann JAX
     shell run with `collision_mode="spectral_relaxation_preflight"` on
     CPU, but this remains an audit-only closure, not the physical Stage-B
     ν-e/pair operator.
   - the same private runtime hook now also supports `thermo_tier=2`
     in bounded tests, feeding the lifted collision moments into the 3T
     thermo primitives.  The resulting factorized Jacobian is exact on
     non-thermo rows and bounded-approximate on the three thermo rows,
     so this is still a preflight surface rather than the landed
     physical Stage-B runtime.
   - a second bounded private mode now exists behind
     `collision_mode="projected_physical_preflight"`.  It replaces the
     spectral surrogate with a projected physical state-dependent bank
     closure and restores exact dense-AD parity on the bounded tier-2
     regression surface, but it is still not the final direct
     Hannestad–Madsen + pair implementation.
   - the host-side preflight now also exposes `closure_mode="direct_kernel"`,
     which evaluates the existing `NuEScatteringOperator` and
     `PairProcessOperator` on the isotropic species banks via an explicit
     `T_bank/T_gamma` remap.  This path now also restores the required
     `1/H` conversion to `d/dN` and exposes a bounded augmented Jacobian
     on `[f_bank, T_gamma, T_nu_e, T_nu_x, H]`.  It closes the
     operator-backed bank audit.  A bounded private runtime candidate is
     now also available behind `collision_mode="direct_kernel_preflight"`
     by combining host-callback primals with an explicit mixed low-rank
     Jacobian payload, but that runtime path is only locked at the
     `rhs/jacobian` evaluability level so far.
   - **Canonical milestone (PR-T3B canonical #1, publicly dispatchable
     after PR-T3D canonical #2):** `collision_mode='ap_unified_preflight'`
     combines a full Fermi-Dirac `psi_target` with a damping projection
     (energy-neutral) and soft total-rate enforcement (clip [0.1, 5],
     1% activation floor).  The enforcement scalar is sign-safe: it is
     positive only when the raw source energy moment has the same
     heating/cooling sign as the canonical 3T Mangano target, and the
     phase-space driver imports that prefactor from the 3T table instead
     of carrying a hidden duplicate constant.  This AP moment contract is
     production-locked for both heating and cooling branches — passes the
     canonical PR-T3D §5 anisotropy
     gate (`spread ~5.4e-5`) **and** grid scaling (`spread ~8.7e-5`)
     while keeping Rodas5P + JAX/GPU compatibility (no IMEX, no
     operator splitting).  FLRW `N_eff = 3.0345` keeps a documented
     `+0.0095` Mangano gap which is the AP-form model approximation
     limit per `PR-T3B-PF #15` scope reframing.  Reachable from public
     code via `canonical_forward_solver(backend='jax_ap_unified_tier3',
     ...)`; the gap is surfaced in `BBNPrediction.metadata` as
     `flrw_mangano_gap_documented = 0.0095` so callers see it without
     consulting the audit doc.  Closing the gap to the canonical
     `5e-3` target is the bounded post-canonical Option E (in-RHS
     analytic relaxation pre-conditioner via Jacobian augmentation,
     `~1-2 weeks`, see
     `docs/research/PR-T3B_option_E_canonical_post_enhancement.md`).
     The public dispatch also accepts the explicit no-QKE diagonal
     `ν-ν` options `jax_tier3_nu_nu="3t"`,
     `jax_tier3_nu_nu="spectral"`, and
     `jax_tier3_nu_nu="accuracy"`, routing them to the same full
     Boltzmann collision modes that were previously private-only.  The
     default remains `"off"` to preserve the PR-T3D baseline.
3. **Stage C (5 days):**  Add ν–ν diagonal operator, benchmark
   against LASAGNA.
4. **Stage D (4 days):**  Analytic Jacobian, GPU vmap, non-LRS
   coupling.

Stages A–C cover the bounded AP-form Tier 3 / Paper II / no-QKE candidate
surface currently documented in the registry.  They do not close full DH-S
coefficient-table runtime, QKE/flavour-coherence, or public production
promotion by themselves.  Stage D is performance/generalisation after the
bounded physics gates pass.
