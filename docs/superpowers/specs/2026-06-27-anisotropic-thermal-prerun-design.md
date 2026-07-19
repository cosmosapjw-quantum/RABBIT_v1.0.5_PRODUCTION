# Anisotropic Thermal Prerun — Design

Date: 2026-06-27
Status: approved (brainstorm), execution via subagent-driven development
Scope: augmented Type-I PSTF no-QKE BBN solver. QKE out of scope. No
public-production / publication / SMC claim.

## Context

BD596 exercised the first live-shear (nonzero Bianchi shear) endpoint run, but
only via a workaround: the accepted `phase1_thermo_prerun_flrw` neutrino start
hard-guards against shear (`augmented_continuous_ap65_rhs.py` ~1499: "FLRW
thermal neutrino start requires zero shear and zero initial A perturbation;
use supplied neutrino temperatures until the anisotropic thermal prerun is
implemented"). BD596 used `--neutrino-thermal-start-policy supplied` with the
FLRW prerun's effective T_nu and set shear directly at phase-2 start.

The accepted FLRW prerun (`_phase1_thermo_prerun_to_T0`) integrates the FLRW 3T
closure (`asymptotic_N_eff_3T_payload`, shear-free) from T=3 MeV to T_gamma0 and
returns the neutrino temperatures. It cannot represent shear or the augmented
distribution perturbation.

This feature implements the missing anisotropic thermal prerun so the FLRW
recipe carries shear (and the A-monopole offset) natively.

## Goal

A neutrino-decoupling prerun that, given primordial shear (and optional
A-monopole offset) at the phase-1 start (T=3 MeV), integrates the
**collision-coupled neutrino PSTF hierarchy** through decoupling to T_gamma0 and
returns a self-consistent phase-2 start state: decoupled `T_nu_e0/T_nu_x0`, the
shear-generated augmented modes (incl. quadrupole), and the **decayed** shear at
T_gamma0. Removes the guard.

## Decisions (from brainstorm)

- **Shear epoch:** input `sigma_plus0/sigma_minus0` is **primordial at the
  phase-1 start (3 MeV)**; the prerun decays it to T_gamma0; phase-2 starts from
  the prerun output (self-consistent).
- **A-offset:** handled too (guard fully removed).
- **Fidelity:** **full quadrupole-coupled** — the prerun evolves the neutrino
  PSTF hierarchy with shear-sourced quadrupole + anisotropic-stress
  back-reaction and collisional damping, not a leading-order H-only correction.

## Physics model

State evolved over e-folds N from T=3 MeV to T_gamma0:
`(T_gamma, T_nu_e, T_nu_x, Sigma_plus, Sigma_minus, F_{A_l}(q) per species)`.

- Expansion: `H = hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq)` (already
  `H^2 = (8 pi G/3) rho / (1 - Sigma^2)`).
- Shear: `dSigma_pm/dN = -(1 - Sigma^2) Sigma_pm + Pi_pm`, with `Pi_pm` the
  neutrino anisotropic stress (already in `augmented_nonlrs_transport`).
- Neutrino PSTF: shear couples `l <-> l +/- 2` (Liouville source
  `sigma_ab q F'_0`) with collisions (Uehling-Uhlenbeck) damping the
  quadrupole — the same dynamics the phase-2 transport already implements.
- n<->p in weak equilibrium during the prerun (no full BBN network above
  T_gamma0; reuse the existing phase-1 weak n-p prerun for Xn).

Reference: `neutrino_decoupling_PSTF_HM_generalization_ko.md` (1+3 PSTF Liouville
hierarchy with shear l<->l+/-2 coupling + HM collision),
`neutrino_large_anisotropy_exact_PSTF_collision_ko.md` (collision multipoles).

## Architecture

**Reuse, do not duplicate.** The collision-coupled anisotropic PSTF transport
already exists in the phase-2 stack: `augmented_typeI_replay`,
`augmented_collision_bridge`, `augmented_nonlrs_transport`, `solver_jax_rodas5p`.
The prerun is a thin driver that runs that transport on a neutrino-only state
from 3 MeV to T_gamma0.

- New module `src/rabbit/validation/anisotropic_thermal_prerun.py` (keeps the
  23.7k-line `augmented_continuous_ap65_rhs.py` from growing):
  `anisotropic_thermal_prerun_to_T0(...)` returning the same payload shape as
  `_phase1_thermo_prerun_to_T0` plus `Sigma_plus0_effective`,
  `Sigma_minus0_effective`, and `initial_A_modes` (incl. shear-generated
  quadrupole).
- New start policy `phase1_thermo_prerun_anisotropic` (a distinct choice, not an
  overload of `phase1_thermo_prerun_flrw`). The existing FLRW policy and its code
  path stay **bit-identical**; the guard is replaced by a dispatch to the new
  policy when `sigma!=0` or `A_offset!=0` under the anisotropic policy, and the
  FLRW policy still rejects shear (directs the user to the anisotropic policy).
- `_default_restart_kwargs` dispatches to the anisotropic prerun when `sigma!=0`
  or `A_offset!=0`; threads the prerun output (T_nu, decayed Sigma, A modes) into
  the restart state; the guard is removed.

## Staged decomposition (each a validation gate)

- **Stage A (this plan):** prerun driver skeleton reusing phase-2 transport;
  integrate `(3T + neutrino PSTF)` 3 MeV -> T_gamma0 at **sigma=0, A=0**. Gate:
  reproduces the existing FLRW prerun `T_nu_e0/T_nu_x0` within a tight tolerance
  (FLRW-limit parity). De-risks the reuse wiring before adding shear.
- **Stage B:** shear (modified H + decay + shear-sourced quadrupole via nonlrs
  Liouville). Gate: sigma->0 continuity to Stage A; quadrupole generated;
  collision number/energy closure holds.
- **Stage C:** A-offset handling + wiring (policy dispatch, restart-state
  threading, guard removal). Gate: guard gone; FLRW recipe carries shear; sigma=0
  endpoint bit-identical to BD591.
- **Stage D:** endpoint validation — redo BD596 live-shear via the new prerun;
  compare to BD596 supplied-workaround result.

## Stage B approach — RESOLVED: reuse phase-2 RHS with frozen network

Feasibility investigation (2026-06-27) selected the reuse path over
assemble-from-components. Findings:

- `--T-gamma0-MeV` (phase-2 start temperature) is CLI-configurable
  (`scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py:1229`,
  default 0.8 MeV).
- The AP65 RHS already auto-freezes the nuclear network at high T:
  `hot_weak_only` (`augmented_continuous_ap65_rhs.py:8224-8250`) replaces the
  BBN network with an exact weak n<->p equilibrium update when
  `T_gamma > activation_threshold`. This is the frozen-network regime, applied
  automatically per step.
- The AP65 RHS combines neutrino PSTF + shear (non-LRS) + collision + 3T, so
  starting phase-2 at/above neutrino decoupling with shear active runs the full
  quadrupole-coupled anisotropic decoupling continuously — no separate prerun
  module needed.
- 3T backbone consistency is guaranteed: the phase-1 FLRW prerun and the AP65
  phase-2 both use the same `coupled_3T_rhs` / `asymptotic_N_eff_3T_payload`
  engine, so AP65's own 3->0.8 MeV decoupling reproduces the prerun T_nu by
  construction.

Revised Stage B/C: rather than a separate prerun module doing the heavy
integration, raise the phase-2 start (`--T-gamma0-MeV`) toward neutrino
decoupling with equilibrium neutrino ICs (`supplied`, `T_nu = T_gamma` at the
start) and shear active; the hot weak-only mode frees the network above the
threshold. The Stage A module (`anisotropic_thermal_prerun.py`) remains the
isotropic-limit reference / parity witness.

Remaining feasibility unknown (needs a bounded compute experiment, not code
analysis): PSTF/collision/shear integration stability when phase-2 starts near
3 MeV (stiffness, the hot_weak_only -> full-network handoff, the N-span e-fold
range recalibration for the higher start). The 3T backbone is already consistent
by shared engine.

### Stage B feasibility — VALIDATED (BD597)

`_PHASE2_NETWORK_ACTIVATION_T_GAMMA_MEV = 0.08`, so phase-2 already runs
`hot_weak_only` (frozen network) from its 0.8 MeV start down to 0.08 MeV on
every endpoint run; raising the start just extends that proven window.

BD597 bounded feasibility run (`diagnostic_outputs/bd597_reuse_prerun_feasibility/`):
phase-2 from `--T-gamma0-MeV 3.0`, `--neutrino-thermal-start-policy supplied`
`T_nu = 3.0` (equilibrium), `sigma=0`, collision on, short span stopping near
0.8 MeV. Result: integrated stably to `completed_hot_endpoint` at
`T_gamma = 0.6769544542566014` MeV. Final T_nu vs the FLRW 3T-closure reference
at the same T_gamma:

| quantity | AP65 (3 MeV start) | FLRW 3T ref | rel. delta |
| --- | ---: | ---: | ---: |
| T_nu_e_MeV | 0.6705979669733102 | 0.6705864865392114 | 1.7e-5 |
| T_nu_x_MeV | 0.6697511143227399 | 0.6697438323129494 | 1.1e-5 |

The AP65 collision-coupled decoupling reproduces the analytic FLRW prerun T_nu to
~1e-5 (delta = full-collision dQ vs analytic energy transfer; collision source
active, `collision_dT_nu_e_source_dN ~ 8e-4`). Cold-projection
`N_eff_3T_asymptotic = 3.03492` matches BD591's `3.03481` to ~3e-5.

Conclusion: the reuse path is feasible and validated. The anisotropic thermal
prerun is realized by running phase-2 from a raised `--T-gamma0-MeV` with
equilibrium neutrino ICs (and shear for the full quadrupole-coupled case); the
existing `hot_weak_only` handles the frozen-network decoupling. No new heavy
integrator is required. The Stage A module remains the isotropic-limit parity
witness.

### Remaining work

- **Stage C (recipe/wiring):** package the raised-start recipe (a convenience
  policy or documented CLI: `--T-gamma0-MeV 3.0` + supplied equilibrium T_nu),
  and a full-endpoint N-span recalibration (extend the ladder to ~6.1 e-folds,
  shift the activation window up by `ln(3.0/0.8) ~ 1.32` e-folds for the 3 MeV
  start). The FLRW shear guard becomes unnecessary on this path (shear runs
  through the AP65 RHS, as BD596 already showed from 0.8 MeV).
- **Stage D (endpoint validation):** full live-shear endpoint from 3 MeV
  (sigma != 0, recalibrated N-span) reaching T < 0.01 MeV; compare to BD596 and
  the FLRW baseline.

## Constraints (anti-drift)

- New policy is opt-in; `sigma=0, A=0` behavior and all defaults unchanged.
- FLRW prerun path bit-identical for the isotropic limit.
- Reuse existing transport; no physics duplication.
- Preserve raw negative/nonfinite/decoupling evidence; no clipping.
- Each PR reports the cost line; small reviewable commits.

## Validation

- Unit (Stage A): FLRW-limit parity of prerun `T_nu` vs
  `_phase1_thermo_prerun_to_T0` within tolerance; reaches T_gamma0; positive
  finite temperatures; energy bookkeeping.
- Stage gates as above.
- Endpoint (Stage D): a focused live-shear span-ladder run reaching T<0.01 MeV.

## Out of scope

- QKE, flavor/helicity coherence.
- Public dispatch / SMC / publication.
- The collisionless `driver_typeI_full_boltzmann` ray path (not reused).
