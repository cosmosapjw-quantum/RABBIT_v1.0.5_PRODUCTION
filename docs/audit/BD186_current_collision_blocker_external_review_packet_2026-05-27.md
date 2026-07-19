# BD186 Current Collision Blocker Review Packet

Date: 2026-05-27

This document is the focused external-review packet for the current blocker.  It
summarizes the recent development process, what worked, what failed, and what
an external auditor should inspect next.  It intentionally includes an
overengineering critique because the project has made many real PRs but fewer
qualitative breakthroughs than desired.

## One-Sentence Problem Statement

The strict isotropic private full-BBN path now reaches `T_gamma < 0.01 MeV`, and
non-collision non-LRS rows can reach the endpoint, but high-q dynamic
non-LRS neutrino collision rows still fail before the full endpoint, most likely
because the dynamic collision source and its missing/approximate Jacobian
coupling produce a resource/stiffness cliff.

## Current Architecture To Preserve Unless Falsified

- Rodas5P remains the host stiff solver.
- Phase-2 BBN network remains a split implicit BE/BDF2/Newton corrector.
- Dynamic neutrino collision remains diagonal/no-QKE.
- CPU-JAX remains the repeated-run backend target.
- No public production claims.
- No output clipping of negative abundances or `Y_p`.

Do not move the activated 9-species network back into the host Rodas stage
algebra as a first response.  That earlier architecture produced trace-stage
pathologies and no-burn behavior.  The current split corrector is expensive but
not the q14 collision row killer.

## Recent Development History By Theme

### 1. Phase-2 network solver

Earlier direct host-stage `X_phase2` evolution failed around activation.  It
created large trace-domain rejections and negative internal stage/candidate
states.  Several interventions followed:

- trace/positive coordinates and raw negative evidence preservation;
- event-like activation handling;
- removing phase-2 from the Rodas RHS/Jacobian block;
- replacing positive directional-extent updates with coupled BE/BDF2/Newton in
  `Y = X/A_mass`;
- analytic network Jacobian support and AB-style Newton initial guesses.

Success:

- strict isotropic endpoint now completes in the private full-BBN path;
- no-burn behavior was replaced by actual He4/D evolution;
- raw negative candidate evidence is still recorded rather than hidden.

Failure or remaining risk:

- the corrector is still expensive;
- older SciPy LRS weak/network ell paths can fail with nonfinite `X`;
- background compression remains an accuracy question for dynamic rows.

### 2. q transport and Laguerre grids

The dynamic q-Laguerre path exposed high-q drift.  Manufactured q-profile tests
identified large tail/boundary sensitivity in the legacy 3-point q derivative.
The q derivative was upgraded to a local wider stencil on sufficiently resolved
grids while keeping legacy comparison support.

Success:

- q differentiation defect was made testable and partly repaired;
- raw Laguerre and energy-moment weights are now separated explicitly.

Failure or remaining risk:

- q transport is still not a conservative/SBP finite-volume operator;
- high-q collision source concentration can still amplify q-grid errors;
- q transport is likely an amplifier, not the first collision-on row killer.

### 3. Freedom attribution

A prior q14/q15 attribution split showed:

- non-LRS-only rows complete;
- non-LRS+weak rows complete with similar host counters;
- non-LRS+collision rows were killed or failed.

Success:

- weak-rate correction is demoted as the main runtime blocker;
- dynamic collision source path is promoted as the dominant blocker.

Failure or remaining risk:

- all-freedom rows still do not reach final endpoint;
- collision/source/Jacobian attribution is still too broad.

### 4. Dynamic collision payload and source handling

The dynamic collision source path originally risked kill/no-artifact behavior.
Recent PRs added:

- cache bounds for source factories and radial grids;
- compact metadata policies;
- source budget telemetry;
- component policies such as dQ-only/dA-only/full source probes;
- optional collision diagonal Jacobian approximation;
- full-JVP defaulting for dynamic q-Laguerre collision rows.

Success:

- `bd185_q14_collision_N4p8_cache_bound_probe.json` writes an artifact rather
  than dying before output;
- payload build attempts and wall time are visible;
- the blocker moved from opaque resource kill to observable solver/runtime
  failure.

Failure or remaining risk:

- q14 collision full endpoint still fails;
- `collision.dA_modes` is still not represented by a full structured derivative;
- the diagonal damping Jacobian is heuristic and incomplete;
- source construction/caching is still entangled with artifact metadata.

### 5. Angular/ell representation

The codebase contains:

- generic angular decomposition contracts;
- LRS ell convergence utilities;
- current non-LRS live full-BBN path with fixed three diagonal `A_modes`;
- S2 angular-grid resolution knobs (`N_mu`, `N_phi`).

Fresh current evidence:

- strict isotropic full-BBN endpoint: success;
- non-LRS no-collision S2 angular-grid pair: endpoint success but `Sigma_H`
  not angular-grid converged;
- LRS collisionless ell ladder: converged at `ell_max=4` on a short span;
- LRS 3T weak/network ell ladder: failed with nonfinite `X`;
- Teff ablation is DEPRECATED and excluded.

Remaining risk:

Full dynamic non-LRS collision full-BBN does not yet have a generic increasing
`ell_max` endpoint ladder.  An external reviewer should not treat the fixed
three-mode diagonal live path as full ell convergence.

### 6. Charge-neutrality collision-source contract repair

During BD186 packet verification, the focused collision-feedback artifact test
exposed a concrete contract bug: the LRS collision-feedback artifact builder did
not pass `electron_chemical_potential_mode` into the weak-network solve, and the
collision thermo source wrapper always injected a precomputed internal
`_electron_chemical_potential_MeV`.  That made charge-neutrality mode risk
consuming a stale fixed value inside source evaluations instead of using the
current charge-neutrality state.

BD186 repairs that contract:

- fixed mode still receives the supplied electron chemical potential;
- charge-neutrality mode uses the explicit evolved charge-asymmetry density when
  provided, otherwise recomputes from the current `X`, `T_gamma`, and `eta`
  payload;
- private `_electron_chemical_potential_MeV` can only be reused through a
  closure-local sentinel stamped by the source wrapper itself, so inbound stale
  private values cannot spoof the current charge-neutrality state.

This is a runtime physics contract repair, not a new readiness/manifest gate.

## Current Failure Mechanics

The most relevant current failure artifact is:

`diagnostic_outputs/bd186_external_audit/bd185_q14_collision_N4p8_cache_bound_probe.json`

Observed facts:

- no public production claim;
- private diagnostic artifact written;
- full-BBN ready false;
- collision payload build attempts total: 344;
- selected wall seconds total: about 137.7 s;
- selected host steps total: 343;
- selected full-JVP Jacobian evaluations total: 342.

The row no longer primarily looks like a pure Python metadata/OOM failure.  It
looks like a dynamic collision source/Jacobian/h-step failure after cache bounds
made the run finite enough to report telemetry.

## Current Blocker Hypotheses

### Hypothesis A: collision `dA_modes` needs a structured Jacobian

The host Jacobian can include frozen-source JAX full JVP and analytic A-mode
transport/shear blocks.  Collision `dA_modes` is added directly to `dA`, but its
dependence on `A`, q-node occupancy, and collision moments is not fully in the
implicit solve.  The optional collision diagonal policy estimates damping only
where `A*dA < 0`, which can miss growth, cross-mode, cross-q, and source-shift
structure.

External audit question:

Would a local-q/species/mode block Jacobian, finite-difference source response,
or source predictor plausibly remove the h-collapse without overfitting?

### Hypothesis B: collision source construction/caching still dominates

Cache bounds prevented kill/no-artifact, but source construction is still
expensive and tightly tied to runtime metadata.  If the solver repeatedly builds
or serializes payloads at high q, the failure can masquerade as numerical
stiffness.

External audit question:

Is runtime still retaining too much source metadata, or is the source itself
expensive even when metadata is compact?

### Hypothesis C: q-Laguerre high-q source concentration amplifies stiffness

The q grid uses energy weights for moments and raw weights for Laguerre
quadrature.  This is conceptually right, but high q nodes can dominate collision
budgets.  q derivative repair improved a real defect, but dynamic collision can
still couple high-q source concentration to stiff geometry/shear evolution.

External audit question:

Are q weights, radial collision source normalization, and p4 interpolation
consistent enough for high-q dynamic collision rows?

### Hypothesis D: the representation boundary itself is too narrow

The full-BBN non-LRS live path is fixed to three diagonal modes.  Generic ell
contracts exist elsewhere.  Dynamic collision could be driving modes or angular
structures outside the current fixed representation, causing wrong stress
feedback or source projection.

External audit question:

Is the current three-mode diagonal projection sufficient as an intermediate
endpoint, or must the live dynamic path move to a real ell/m hierarchy before
the collision terms can be physically interpreted?

## Overengineering And Development Drag

The code has moved real physics blockers, but the surface area is now high enough
to slow breakthroughs.

### High-priority overengineering concerns

1. `augmented_continuous_ap65_rhs.py` and
   `augmented_continuous_ap65_full_bbn_span_ladder.py` are large multi-purpose
   modules.  They mix solver mechanics, physics equations, artifact schemas,
   claim wording, and test-driven policy defaults.
2. Evidence plumbing has accumulated faster than blocker-removing runtime
   decomposition.  Some policies exist mainly to explain previous failures.
3. Artifact schema and runtime payloads are too tightly coupled.  This already
   mattered for dynamic collision metadata/caching.
4. The branch has too many active knobs.  More knobs can improve attribution,
   but they also make it easier to avoid a decisive solver/physics change.
5. "ell" language is overloaded across generic contracts, LRS utilities, and
   the fixed three-mode non-LRS live path.

### What to simplify now

- Freeze nonessential policy axes for the next collision PR.
- Extract dynamic collision source construction and source-response Jacobian
  into a separate module with a small API.
- Move artifact summarization out of hot runtime payload objects.
- Rename the fixed non-LRS live path as fixed diagonal three-mode unless/until a
  real dynamic ell/m ladder is implemented.
- Keep only one default collision metadata mode for long rows: compact summaries.

### What not to simplify yet

- Do not remove raw state diagnostics that preserve negative evidence.
- Do not collapse the phase-2 network corrector back into host Rodas stages.
- Do not remove the freedom attribution matrix until collision-on rows complete.
- Do not convert the whole project to another language before profiling the
  collision source and Jacobian path after source-response fixes.

## Files To Attach For External Review

Core docs:

- `AGENTS.md`
- `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
- `docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md`
- `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- `docs/audit/BD186_augmented_typeI_noqke_external_audit_overview_2026-05-27.md`
- `docs/audit/BD186_current_collision_blocker_external_review_packet_2026-05-27.md`
- `docs/audit/BD186_external_audit_prompt_2026-05-27.md`

Runtime/source files:

- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- `src/rabbit/jax/augmented_typeI_replay.py`
- `src/rabbit/jax/solver_jax_rodas5p.py`
- `src/rabbit/jax/nonlinear_transport.py`
- `src/rabbit/transport/augmented_collision_bridge.py`
- `src/rabbit/transport/augmented_nonlrs_transport.py`
- `src/rabbit/transport/angular_decomposition.py`
- `src/rabbit/transport/augmented_typeI_weak_network.py`
- `src/rabbit/collisions/pstf_contractions.py`
- `src/rabbit/collisions/pstf_process_catalog.py`
- `src/rabbit/collisions/deterministic_reference.py`
- `src/rabbit/network/abundances_standard.py`
- `src/rabbit/weak/live_rates.py`
- `src/rabbit/jax/weak_live_jax.py`
- `src/rabbit/jax/nudec_coupled_jax.py`

Scripts and tests:

- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `scripts/run_augmented_3t_convergence_artifact.py`
- `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `tests/test_augmented_continuous_ap65_rhs.py`
- `tests/test_jax_augmented_typeI_replay.py`
- `tests/test_angular_decomposition.py`
- `tests/test_augmented_convergence.py`
- `tests/test_standard_network.py`

Artifacts:

- `diagnostic_outputs/bd186_external_audit/artifact_summary.json`
- `diagnostic_outputs/bd186_external_audit/isotropic_sigma0_equalT_full_bbn_N4p8_max512.json`
- `diagnostic_outputs/bd186_external_audit/isotropic_full_bbn_N4p8_max512.json`
- `diagnostic_outputs/bd186_external_audit/nonlrs_angular_ablation_N4p8.json`
- `diagnostic_outputs/bd186_external_audit/lrs_collisionless_ell_ablation_sigma0p02_N0p2.json`
- `diagnostic_outputs/bd186_external_audit/bd184_q14_collision_N4p8_auto_chain_probe2.json`
- `diagnostic_outputs/bd186_external_audit/bd185_q14_collision_N4p8_cache_bound_probe.json`
- `diagnostic_outputs/bd186_external_audit/git_log_last100.txt`

## Recommended External Review Questions

1. Is the current diagnosis correct that dynamic collision source/Jacobian
   coupling dominates over weak-rate history and phase-2 network solve?
2. Is the three-mode diagonal non-LRS representation defensible for a staging
   full-BBN endpoint, or must a true ell/m dynamic path precede collision claims?
3. What minimal collision source-response Jacobian would be mathematically and
   numerically defensible?
4. Are q Laguerre raw/energy weight contracts and PSTF radial moment projections
   consistent in the attached code?
5. Is the current failure more likely numerical stiffness, source normalization,
   Python metadata/runtime overhead, or representation insufficiency?
6. Which existing artifact/claim plumbing should be deleted or split because it
   is overengineering rather than science preservation?
