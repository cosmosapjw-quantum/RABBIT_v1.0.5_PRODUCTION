# B4-PR4 — JAX characteristic collisional wiring + the unsound cross-driver reference

**Date:** 2026-07-01
**Scope:** B4-PR4 (live wiring of the calibrated-RTA collision bridge into the JAX
characteristic LRS tier-2 driver) and a measurement-driven refutation of the
originally-planned "tight cross-driver parity gate vs the SciPy collisional
characteristic driver."
**Status:** wiring landed as a CANDIDATE driver-level surface (default OFF, fenced
off the inference dispatch); the cross-driver comparison is a DOCUMENTED
DIAGNOSTIC, NOT a validation gate. Anti-drift footer applies: parity/wiring
fidelity ≠ physics validation.

---

## 1. What was wired

The JAX characteristic transport state is RAYS-ONLY
(`y = [Σ₊, Σ₋, S, T_γ, (T_νₑ, T_νₓ), X_i]`); the per-ray `I_j`/`J_j` are
reconstructed analytically from the accumulated shear `S`. There is therefore no
`dI` slice to add a collision shift to (unlike the SciPy path, which carries `I`
as state). The collision bridge's energy shift `δI` is ISOTROPIC (a single scalar
shared by all rays), so B4-PR4 promotes **one** new state DOF:

- `I_coll` — accumulated isotropic collision energy shift, `dI_coll/dN = δI_scalar`,
  with `δI_scalar` from the parity-locked `apply_gather_scatter_collision_jax`
  twin (B4-PR3, numpy-parity-locked to `apply_gather_scatter_collision`).
- `I_eff = I_reconstructed(S) + I_coll` substituted everywhere `I` is consumed
  (`extract_stress_jax`, `extract_monopole_jax`).

**Why the superposition is exact.** The transport RHS for `I` is
`dI/dN = Σ₊ P₂(μ)` — independent of `I` itself
(`transport/characteristic_rays.characteristic_transport_rhs`). So the analytic
reconstruction `I_reconstructed(S)` and the collision accumulation `I_coll`
superpose with no cross term.

`enable_collisions` is threaded as a **static** trace key through
`_char_layout → _compile_char_rhs → _char_rhs_cache_key → _get_char_rhs → _rhs_core`.
When OFF, the `I_coll` slot is absent and the compiled RHS is byte-identical to the
pre-wiring driver (no-op regression test:
`test_b4_jax_char_collision_wiring.py::test_collisions_off_reproduces_collisionless_baseline`).
Collisions are fenced to LRS tier-2 (`thermo_tier=2`, `transport_mode=characteristic`)
by a `JAXTypeICharConfig.__post_init__` guard.

## 2. The originally-planned gate premise — refuted by measurement

The plan called for a **tight cross-driver parity gate**: JAX-char-collisional
`Yp`/`DH` vs the SciPy collisional characteristic driver
(`run_full_coupled_typeI`, `TransportMode.CHARACTERISTIC`, `enable_collisions=True`,
`tier=2`) at `Σ_H ∈ {0, 0.1, 0.3}`. Direct measurement shows this gate is
**unsound** for three independent reasons.

### 2a. The SciPy collisional reference is decoupling-backbone driven

When `enable_collisions and tier ≥ 2` in characteristic mode, the SciPy driver
builds an **isotropic decoupling backbone** (`_build_decoupling_backbone` →
`solve_isotropic_decoupling`) and drives the thermo (`dT_γ/dN`, `dT_νₑ/dN`,
`dT_νₓ/dN`), the bridge temperatures, the initial conditions, AND the ν spectra
from it; the live 3T slots become passive mirrors
(`drivers/full_coupled_typeI.py:696-809, 1144-1194`). The JAX path is **live-3T**.
The two are architecturally different thermal evolutions — there is no
state-for-state correspondence to gate tightly.

### 2b. The SciPy reference reports an anomalous, shear-blind N_eff

| Σ_H | SciPy coll Yp | SciPy coll N_eff | SciPy collisionless Yp | SciPy collisionless N_eff |
|-----|---------------|------------------|------------------------|---------------------------|
| 0.0 | 0.244115 | **4.1673** | 0.241716 | 3.0345 |
| 0.1 | 0.249301 | **4.1673** | 0.241897 | 3.0345 |
| 0.3 | 0.283818 | **4.1673** | 0.243395 | 3.0344 |

The reported `N_eff = 4.1673` is the standalone `solve_isotropic_decoupling`
trajectory diagnostic (`full_coupled_typeI.py:408, 1395`); it is **shear-blind**
(identical across Σ_H — the backbone is isotropic and cached on `T_start/T_end/N_q`
only) and it does **not** enter the BBN Hubble (which uses `config.N_eff = 3.044`,
`full_coupled_typeI.py:782, 824`). It is therefore inconsistent with its own Yp:
a real ΔN_eff ≈ 1.13 would shift Yp by ~+0.015, but the measured collisional Yp
shift at Σ_H=0 is only +2.4e-3. The reported N_eff is a misleading observable, not
a validated quantity. (This is consistent with the existing maturity marker:
`backend_capabilities.py` already fences this path as a `tier="candidate"`
"transitional candidate surface ... separate from the coarse auto/scipy dispatch
key.")

### 2c. The SciPy collisional correction is implausibly large under shear

The SciPy collisional Yp correction grows to **+0.04 at Σ_H=0.3** (0.2434 → 0.2838,
a 17% increase). Incomplete-decoupling collisions are a sub-percent effect on Yp;
this is the calibrated-RTA bridge — tuned only at FLRW (`_C_RATE` calibrated to
`N_eff ≈ 3.044`) — extrapolating badly into the strong-shear regime where it has no
calibration.

## 3. The JAX live-3T collisional path is better-behaved (but still NOT validated)

| Σ_H | JAX coll Yp | JAX coll ΔYp vs free | JAX coll N_eff | JAX coll − SciPy coll ΔYp |
|-----|-------------|----------------------|----------------|---------------------------|
| 0.0 | 0.241301 | −4.1e-4 | **3.0345** | −2.8e-3 |
| 0.1 | 0.241822 | −7.4e-5 | **3.0345** | −7.5e-3 |
| 0.3 | 0.247448 | +4.1e-3 | **3.0344** | −3.6e-2 |

The JAX path keeps a **physical N_eff ≈ 3.034** (the bridge `δI` perturbs the ray
*spectral* monopole the weak rates see, not the bulk ν energy density / N_eff) and
yields **small, plausible** corrections (sub-1e-3 for Σ_H ≤ 0.1). The two drivers
**diverge by up to 3.6e-2** at Σ_H=0.3.

**Conclusion.** A tight cross-driver Yp/DH parity gate would force the
better-behaved JAX path to reproduce the anomalous SciPy reference — the opposite
of the anti-drift discipline. So:

1. The cross-driver comparison is recorded as a **DIAGNOSTIC** (this document +
   `test_b4_collisional_char_reference_unsound.py`), **not** a pass/fail validation
   gate. Neither driver's collisional path is externally validated; they are the
   same calibrated-RTA family fed by different thermal backbones.
2. The JAX collisional surface is wired as a **CANDIDATE** at the driver level only
   and is **kept off the inference dispatch** — the `canonical_forward_solver`
   ValueError for `enable_collisions` on the JAX characteristic backends is
   **retained** (a deliberate, documented deviation from the plan's "relax the
   ValueError" step, justified by this measurement).
3. The sound lock that DOES ship is **operator/RHS-level parity**: the in-driver
   `dI_coll/dN` equals the parity-locked bridge twin's isotropic `δI` at a matched
   state (`test_b4_jax_char_collision_wiring.py::test_in_driver_dI_coll_matches_the_parity_locked_bridge_twin`).
   This proves the wiring faithfully routes the calibrated operator — engineering
   fidelity, NOT physics validation.

## 4. What would unblock promotion

An **independent, finite-shear collisional** reference (external code or a
first-principles anisotropic collision integral, not a calibrated RTA) — the same
unmet need machine-encoded by the `GATE_SIGMA_H_TO_0P95` fail-closed external
anchor (PR#22). Until then the JAX characteristic collisional path stays a fenced
candidate.

---

*Anti-drift footer: this document reports measured driver outputs. No publication
claim is made. Self-consistency, cross-driver agreement, and operator parity are
NOT external validation. The collision operator is a calibrated relaxation-time
model (B4-PR1/2/3), not a first-principles collision integral.*
