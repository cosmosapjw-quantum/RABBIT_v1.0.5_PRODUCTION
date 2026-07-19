# BD612 PR-1 — Collision-Rate Reduction Derivation & Checkpoint (F-1)

Date: 2026-07-08
Status: **IMPLEMENTED / REGRESSION-LOCKED — source prefactor repaired.**
Finding: BD612 F-1 (collision-rate dimension / T-scaling), report §8.
Regression lock: `tests/test_deterministic_reference_rate_scaling.py` now asserts the
T⁵ target and the `4π³` reduced-denominator source convention.

This document records the pre-patch checkpoint and the implemented resolution.
Historical "current code" statements below refer to the pre-repair T⁴ state unless
explicitly marked as the final decision.

---

## 0. Implemented resolution

Decision:

- **T-power:** fixed at source: `G_F²T⁴ -> G_F²T⁵` for the reduced collision
  field `C`, so `C` is again a MeV `df/dt` rate before consumers divide by `H`.
- **Constant:** keep the code's reduced-matrix convention at
  `1/(4π³)`.  The `1/(2π³)` form corresponds to a differently normalized
  reduced matrix element/spin-root convention, not the polynomial currently used
  by `deterministic_reference`, `nu_e_scattering`, `pair_processes`, and
  `jax/collisions_jax`.
- **Fix locus:** source repair, not clean-core compensation.  The clean core
  continues to consume raw `C` as `df/dt`; `full_boltzmann_collision_preflight`
  remains a calibrated candidate path where `source_scale` can absorb raw-source
  magnitude changes.
- **Code:** `src/rabbit/collisions/kernels.py` now owns
  `HM_REDUCED_COLLISION_DENOMINATOR = 4π³` and
  `hm_reduced_collision_prefactor(T) = G_F²T⁵/(4π³)`.
  The same T⁵/`4π³` source convention is applied to the named 11 sites and the
  sibling JAX ν-ν spectral skeleton, whose docstring also exposes `C` as a MeV
  rate.
- **Test lock:** `tests/test_deterministic_reference_rate_scaling.py` now passes
  on T⁵ scaling; it no longer carries the strict-xfail T⁴ falsification marker.

Remaining boundary: this resolves the internal HM/DHS reduced-kernel convention
and the load-bearing T-power defect.  It is not an external validation of the
clean-core `N_eff` endpoint; Gate-B/normalization anchors still need re-derived
post-repair interpretation rather than tuning to a target.

## 1. The observed defect

`deterministic_reference.py:394` (and 5 sibling sites) build the collision field as

```
prefactor = G_F_MEV**2 * T**4 / (4.0*np.pi**3)      # G_F_MEV is MeV^-2
C = prefactor * total / max(q**2, 1e-30)             # total = Σ w·M²·S  (dimensionless)
```

`total` is dimensionless: `M²` is degree-4 in the dimensionless variables `y = E/T`
(`(y₁y₂)²+…`), the Gauss-Laguerre `_laguerre_plain_weights` are dimensionless, and
`S` (the Pauli six-monomial factor) is dimensionless. `q` is dimensionless. Therefore

```
[C] = [G_F² T⁴] = (MeV⁻²)² · (MeV)⁴ = MeV⁰   (DIMENSIONLESS)
```

But `C` is consumed as a **df/dt rate in MeV**:
- `dynamic_collision_core.flrw_dynamic_collision_rhs:205`: `dA/dN = −(C/H_safe)/(f(1−f))`, `H` from `hubble_3T` in MeV → `C/H` must be dimensionless per e-fold → **`[C]` must be MeV**.
- `dynamic_collision_driver._make_rhs:201`: `df = dC_Y / H_safe` → same requirement.

So `C` is one power of T short: it must scale as `G_F² T⁵` (df/dt ~ MeV), not `G_F² T⁴`.

## 2. What is dimensionally forced (high confidence)

The massless 2↔2 collision integral for particle 1 is

```
C[f₁] = 1/(2E₁) ∫ dΠ₂ dΠ₃ dΠ₄ (2π)⁴ δ⁴(p₁+p₂−p₃−p₄) |M|² F[f]
```

with `dΠ = d³p/((2π)³ 2E)`. Power counting for massless quanta (all momenta ∼ T):
- three phase-space measures: `[MeV²]³ = MeV⁶` (and `(2π)⁻⁹`),
- energy-momentum delta `δ⁴`: `MeV⁻⁴` (and `(2π)⁴`),
- `1/(2E₁)`: `MeV⁻¹`,
- `|M|²` for a 4-fermion contact term: `G_F² · E⁴ = G_F² T⁴ · (dimensionless)` — dimensionless overall.

Net: `[C] = MeV^(6−4−1) · G_F²T⁴(dimensionless) = MeV¹`, and every MeV is a `T`, so

```
C ∝ G_F² · T⁵ · (dimensionless reduced integral)          ← REQUIRED
```

**The T-power is forced: T⁴ → T⁵.** This is independent of every convention choice
below. The falsification test `test_energy_transfer_scales_as_T5` encodes exactly this
(bare moment `∫q³C dq` must scale T⁵; the code gives T⁴).

Independent in-repo confirmation — the correctly-scaled sibling rate is used
consistently elsewhere:
- `thermo/rate_prefactors.py:9,16`: `Γ = (7π/12) G_F² T⁵ · a` (MeV).
- `nu_e_scattering.py:308,321`, `pair_processes.py:165`, `nu_nu_scattering.py:14`: all `∝ G_F² T⁵`.

## 3. Pre-repair open questions — superseded by §0 for the implemented convention

### 3a. The (2π) normalization constant: `1/(4π³)` vs textbook `1/(2π³)`

The code's `C = [G_F²T⁴/(4π³)]·(1/y₁²)·∫∫dy₂dy₃[M²·S]` is the standard 2-D reduced
form (three of the nine integrals done analytically via the energy δ and the isotropic
angular reduction, `y₄ = y₁+y₂−y₃`). The `(2π)` power that survives the reduction
(`(2π)⁻⁵` from measures×δ, times the analytic azimuthal/polar 2π's) fixes whether the
constant is `1/(4π³)`, `1/(2π³)`, or another rational×π multiple, and whether it
absorbs the `2⁵`/`2⁴` factors conventionally pulled out of `|M|²`. **This requires
writing out the reduction explicitly (or matching a canonical reference — Dolgov–
Hansen–Semikoz 1997; Hannestad–Madsen 1995; Grohs et al. BURST). It is an expert
call, not dimensional analysis.**

Anchor available: `nu_e_scattering.total_rate = (7π/12)G_F²T⁵·(G_L²+G_R²)` is an
independent, correctly-scaled thermal-average. After the T-power fix, the corrected
`C`'s appropriate moment can be checked against `(7π/12)` at one temperature to pin the
constant.

### 3b. Fix LOCUS — prefactor (11 sites) vs clean-core consumption

The `G_F²T⁴/(4π³)` convention is **shared across 11 sites**, not localized:

| File | Lines |
|---|---|
| `collisions/deterministic_reference.py` | 394, 451, 515, 578, 711, 766 |
| `collisions/nu_e_scattering.py` | 139, 247 |
| `collisions/pair_processes.py` | 112 |
| `jax/collisions_jax.py` | 210, 290 |

Two physically-equivalent resolutions with different code loci and blast radius:

- **(A) Correct the prefactor** `G_F²T⁴ → G_F²T⁵` at the C-field sites. Makes `C` a
  genuine MeV df/dt everywhere. Blast radius = every consumer of these operators
  (clean core AND the tier-3 full-Boltzmann preflight `jax/full_boltzmann_collision_preflight.py`,
  parity tests, replay-pinned values). If any consumer already compensates with an
  implicit `T`, (A) would double-count for that consumer.
- **(B) Correct the clean-core consumption** — multiply `C` by `T_gamma` (one factor
  of the plasma temperature in MeV) at the point of use in
  `dynamic_collision_core`/`dynamic_collision_driver`. Localizes the change to the
  F-1 live path; leaves the shared operators (and their other consumers / replay pins)
  untouched. Correct only if `C` is deliberately a dimensionless "reduced collision
  function" in this codebase's convention and the clean core is the sole mis-consumer.

**Deciding (A) vs (B) requires confirming how the OTHER consumer
(`full_boltzmann_collision_preflight`) uses `C`** — whether it treats `C` as df/dt
(→ (A), shared bug) or applies a compensating `T` (→ (B), clean-core-only bug). This is
the load-bearing question for the patch and is flagged for the physicist.

**Consumption finding (this session):** `full_boltzmann_collision_preflight.py` does
NOT consume the raw operator `C` as df/dt. It builds `C_mono` through a calibrated
scaling — `_evaluate_projected_physical_species_core` returns `(C_mono, gamma_over_H,
source_scale)` (:370), `gamma_over_H = collision_rate_per_efold(...)` (:177,:387, the
calibrated RTA `Γ ∝ G_F²a_s F T_γ⁴T_ν`), and `C_mono = source_scale·source_raw +
closure_strength·damping_raw` (:437). The absolute scale of the raw `C` is therefore
renormalized by a calibrated `source_scale` on this diagnostic/candidate path, which
can partially absorb the T⁴-vs-T⁵ error. So:
- The **clean core** (`dynamic_collision_core`/`dynamic_collision_driver`) consumes the
  raw `C` DIRECTLY as df/dt — this is where F-1 bites, unmasked.
- The **preflight/RTA path** re-scales `C` through a calibration layer — the error is
  partially masked there and this path is diagnostic/candidate, not the F-1 concern.

This favors making the physics correct at the source, but with awareness that fix (A)
shifts the preflight's `source_raw` (its `source_scale` calibration and any replay pins
would need regeneration). Fix (B) confines the change to the clean core and leaves the
shared operators + preflight + parity twins untouched. Both are defensible; the choice
is the physicist's.

## 4. Effect on the interpretation (why this blocks the physics claim)

With the current `C ∝ T⁴`, at fixed spectral shape the energy-relaxation rate
`Γ_relax = dQ/Δρ` is constant in `T` and `Γ_relax/H ∝ 1/T²` — it INCREASES as the
universe cools (measured probe: 0.07 at 10 MeV → 29 at 0.5 MeV), the inverse of the
standard weak decoupling where `Γ/H ∝ T³` falls through 1 near 1.5–2 MeV. After the
T-power fix, `Γ_relax ∝ G_F²T⁵` and `Γ_relax/H ∝ T³` (falling), restoring the standard
decoupling window. Until then, the clean-core collisional `N_eff ∈ [3.00, 3.15]` (Gate
B) is a numerical coincidence of the constants near ~1 MeV, **not** a physically-
interpretable decoupling result. Energy conservation is by construction and cannot
detect this.

## 4b. Symbolic / numerical verification (SageMath + first-principles, this session)

Tools available and used: WolframScript 1.14.0, SageMath 10.9, Lean (elan). The
checks below are first-principles numerical integrations of the fundamental collision
integral (CM 2-body final-state phase space with the true `|M|²(s,t)`), run against the
code. Scripts: `scratchpad/bd612/verify_totalrate.py`, `verify_Cfield.py`, `verify_const.py`.

**Result 1 — total-rate benchmark (T-power).** The thermal ν-e scattering rate from
first principles gives `⟨Γ⟩/T⁵ = 0.385011`, **constant** across T = 1, 2, 4 MeV →
the fundamental rate scales as **T⁵** exactly. (The 0.385 vs (7π/12)=1.833 coefficient
differs by the averaging convention; the scaling is the point.)

**Result 2 — collision field C(q) (T-power, decisive).** Comparing the code's
`evaluate_nue_scattering_reference` to the first-principles `C(E₁)` at matched
configuration: `C_code/C_truth × T` is **identical at T = 2 and T = 4 MeV**
(node-wise and integrated), i.e. `C_code/C_truth ∝ exactly 1/T`. The energy-transfer
moment ratio `dQ_code/dQ_truth × T = 5.74137` is bit-identical at both temperatures.
The ground truth satisfies detailed balance (C → 0 at equilibrium, ~1e-39). **So the
code is exactly one power of T short — T⁴ → T⁵ is confirmed three independent ways
(dimensional analysis, total-rate benchmark, and the exact 1/T field ratio).**

**Result 3 — pre-repair quick-truth caveat.** After removing the T-power, the
code-vs-truth magnitude ratio is `~5.7×` integrated and q-dependent (≈2–12× node-wise,
with a sign flip near C's zero-crossing). This is confounded by `|M|²` spin-sum /
normalization conventions in the quick ground truth (the `32 G_F²` prefactor, spin
averaging) and by shape sensitivity, so it does **not** by itself prove the `1/(4π³)`
constant is wrong — but it does show the constant/shape cannot be signed off from a
rough numerical reduction. A `~5.7×` constant error would shift the decoupling-window
crossing temperature by `5.7^(1/3) ≈ 1.8×`, so the constant was treated as
load-bearing.  The implemented decision in §0 resolves the internal code convention
as `1/(4π³)` for the exact reduced-`M²` polynomial currently used by the source
operators; external magnitude validation remains a separate post-repair anchor.

## 5. Implemented decisions after checkpoint

1. **T-power fix `T⁴ → T⁵`** — implemented at the source prefactor.
2. **(2π) constant** — retained as `1/(4π³)` for the code's exact reduced-`M²`
   convention.  No factor-of-two change was applied.
3. **Fix locus** — source prefactor, not clean-core consumption.  This prevents
   compensating-temperature patches from making shared operators remain
   dimensionless while their docstrings and consumers call them MeV rates.
4. **Re-pin scope** — focused parity and clean-core tests pass after the repair,
   but collisional endpoint interpretation still requires a post-repair
   normalization/Gate-B review before any `N_eff` promotion.
