# BD186 Augmented Type-I No-QKE External Audit Overview

Date: 2026-05-27

This document is a self-contained overview for an external reviewer.  It describes
what the current RABBIT branch is trying to build, what the code actually does,
which paths have run, where it fails, and why the current blocker is believed to
be concentrated in the dynamic neutrino collision path rather than the weak-rate
or phase-2 BBN network corrector.

Claim status vocabulary follows `AGENTS.md`: IMPLEMENTED, VALIDATED, DERIVED,
SPECIFIED, PROPOSED, SPECULATIVE, DEPRECATED, FORBIDDEN.

## Scope Boundary

- QKE is FORBIDDEN/out of scope for this branch.
- Public production support is FORBIDDEN to claim.
- CPU-JAX with the in-tree Rodas5P path is the repeated-run backend target.
- The current artifacts are private diagnostic artifacts, not public production
  promotion artifacts.
- Negative abundance and negative `Y_p` evidence must not be hidden by output
  truncation.  The solver records raw candidates and corrected candidates.

## What This Project Is

RABBIT is a Python research code for BBN and anisotropic Bianchi Type-I neutrino
transport.  The branch under audit is the augmented Type-I PSTF no-QKE program:

1. Evolve diagonal Type-I geometry, shear, photon/neutrino temperatures, neutrino
   angular/radial distribution data, and BBN abundances.
2. Keep the large host system on CPU-JAX/Rodas5P.
3. Keep the activated 9-species BBN nuclear network out of the host Rodas stage
   algebra and apply it as a split, coupled implicit BE/BDF2/Newton corrector.
4. Add a dynamic diagonal no-QKE neutrino collision source built from AP65-style
   angular/PSTF radial contractions.
5. Produce reproducible diagnostic artifacts that expose raw solver state,
   telemetry, and failure modes.

The current target is not a polished public endpoint.  It is to make full BBN
runs to `T_gamma < 0.01 MeV` possible under the relevant freedom combinations:
isotropic/no-collision, anisotropic non-LRS transport, weak-rate correction,
dynamic neutrino collision terms, and all freedoms together.

## Main Runtime Data Flow

The current private full-BBN path is centered in:

- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- `src/rabbit/jax/augmented_typeI_replay.py`
- `src/rabbit/jax/solver_jax_rodas5p.py`

The high-level flow is:

1. The artifact runner builds a row or ladder case with a chosen freedom set.
2. Restart kwargs are converted into a JAX live-source replay state.
3. `augmented_typeI_replay._build_live_source_grid_jax` builds the q grid,
   angular grid, projection matrix, and raw/energy Laguerre weights.
4. `_live_source_rhs_vector` reconstructs the neutrino distribution from
   `A_modes`, computes live weak rates from the monopoles, computes `H`, computes
   stress feedback, applies q/mu/phi transport derivatives, and adds
   `collision.dA_modes` when dynamic collision is active.
5. `_run_step_cap_row` advances the host with Rodas5P, using frozen-source JAX
   JVP/full-JVP policies plus optional analytic split blocks.
6. After a host candidate is accepted, the phase-2 network corrector advances the
   9 species with a network-only implicit BE/BDF2/Newton solve in
   `Y = X / A_mass`, not by putting `X_phase2` back inside Rodas stages.

## Physics And Numerical Representation

### Geometry and neutrino distribution

The current repeated-run non-LRS live path represents the neutrino distribution
through `A_modes` with shape `(species, 3, N_q)`.  These are the diagonal Type-I
plus/minus modes projected through the S2 basis, not a generic arbitrary
`ell_max` live full-BBN hierarchy.  The codebase also contains generic angular
decomposition contracts (`AngularDecompositionSpec`, `active_lrs_modes`,
`active_non_lrs_diagonal_modes`) and LRS ell convergence utilities, but the
current dynamic non-LRS full-BBN path is the fixed three-mode diagonal path.

The distribution is reconstructed as a nodal angular distribution over
`(q, mu, phi)` and then projected back to mode space.  q is usually represented
by Gauss-Laguerre nodes.  The code now carries raw Laguerre weights separately
from energy-moment weights:

- raw weights integrate `exp(-q) g(q)`;
- energy weights use `w * exp(q) * q^3` for energy-density moments.

### Collisionless transport

The live-source transport term computes q, mu, and phi derivatives of nodal
`A` and projects the result back into the three diagonal modes.  q differentiation
currently uses a local finite-difference operator in
`src/rabbit/jax/nonlinear_transport.py`.  The legacy 3-point operator was exposed
by manufactured q-profile checks as high-order Laguerre-tail sensitive.  The
current operator uses a local 5-point polynomial stencil for q grids with at
least nine nodes, while preserving the 3-point operator for comparison.

### Weak rates

Weak rates are live rates computed from reconstructed neutrino/antineutrino
monopoles and photon/neutrino temperatures.  They affect neutron/proton
conversion and therefore `Y_p`, but recent attribution rows show weak-rate
correction is not the primary runtime cliff: non-LRS-only and non-LRS+weak rows
complete with similar host counters, while collision-on rows are the first to
hit the dynamic runtime/stiffness blocker.

### Phase-2 BBN network

The 9-species network uses the standard species
`n, p, D, T, He3, He4, Li7, Be7, Li6` and up to 31 PRIMAT AC2024 reactions.
The accepted activated-network architecture is:

- host Rodas5P does not directly evolve the stiff `X_phase2` block;
- the activated network is a post-accepted-step implicit corrector;
- the residual is in abundance-per-baryon variables `Y = X / A_mass`;
- the network residual is
  `R(Y*) = Y* - Y0 - h_over_H * dY_dN(Y*)`;
- analytic network Jacobian support is implemented, with finite-difference
  fallback.

This architecture replaced earlier direct linear-`X` Rodas-stage attempts that
created trace-domain stage negativity and no-burn behavior.

### Dynamic neutrino collision source

The dynamic collision source is the current dominant blocker.  The source is
built in `_dynamic_collision_source_payload_from_restart_state_np` and calls
`evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source`, which
combines angular and PSTF radial collision source construction.  It returns
thermo energy-transfer terms and `dA_modes`, which `_live_source_rhs_vector`
adds directly to the transport `dA`.

The collision algebra is diagonal/no-QKE.  It uses the six-monomial Pauli
gain-loss contraction and PSTF radial source construction.  Known risk areas
are not the top-level six-monomial sign pattern, but implementation contracts:
q/radial normalization, p4 interpolation, angular/PSTF projection, dynamic
cache lifetime, and whether collision `dA_modes` is sufficiently represented
in the host Jacobian.

## Current Evidence From Fresh BD186 Runs

All commands below were run from the repository root with `PYTHONPATH=src` and
`JAX_PLATFORMS=cpu`.

### Strict isotropic full-BBN endpoint

VALIDATED for the private diagnostic path:

Artifact:
`diagnostic_outputs/bd186_external_audit/isotropic_sigma0_equalT_full_bbn_N4p8_max512.json`

Configuration:

- `sigma_plus0 = 0`
- `sigma_minus0 = 0`
- `T_gamma0 = T_nu_e0 = T_nu_x0 = 0.8 MeV`
- freedom composition case `[]`
- `N_span_end = 4.8`
- `h_max = 0.1`
- `max_steps = 512`
- phase-2 split corrector enabled
- `phase2_network_background_policy = effective_midpoint`

Result:

- `passed = true`
- `physical_full_bbn_span_ready = true`
- `T_final_MeV = 0.00914138906397148`
- `Yp = 0.16412585886748188`
- `D/H = 2.123358549828114e-05`
- `Sigma_H = 4.642446311648653e-15`
- `N_eff_3T = 3.1148623525203636`

Interpretation:

The strict isotropic private path can reach the full-BBN endpoint under the
current diagnostic solver settings.  This is not a public production claim and
does not validate the all-freedom dynamic collision path.

### Default-shear empty-freedom full-BBN endpoint

VALIDATED as a sheared no-freedom diagnostic, not as isotropic:

Artifact:
`diagnostic_outputs/bd186_external_audit/isotropic_full_bbn_N4p8_max512.json`

This earlier artifact used default nonzero shear initial values, so it must not
be called isotropic.  It completed:

- `T_final_MeV = 0.009139193937586743`
- `Yp = 0.16353752122474216`
- `D/H = 2.095132800108486e-05`
- `Sigma_H = 0.0778663693510014`
- `N_eff_3T = 2.9933985109712378`

### Non-LRS angular-grid ablation to full endpoint

IMPLEMENTED/EXECUTED for two non-collision, non-LRS angular-grid endpoint rows;
not VALIDATED as an angular-grid-converged result:

Artifact:
`diagnostic_outputs/bd186_external_audit/nonlrs_angular_ablation_N4p8.json`

Rows:

| row | T_final_MeV | Yp | D/H | Sigma_H | N_eff_3T |
| --- | ---: | ---: | ---: | ---: | ---: |
| `N_mu=3,N_phi=5` | 0.009139194191255893 | 0.16354612970385612 | 2.097928641626068e-05 | 0.13130812848785328 | 2.993397794780868 |
| `N_mu=5,N_phi=7` | 0.009139194188014276 | 0.16354679459234078 | 2.0984465324172533e-05 | 0.14346334013390802 | 2.99339779635095 |

Result:

- both rows completed the full endpoint;
- abundance deltas are small at this pair: `delta Yp ~= 6.65e-7`,
  `delta D/H ~= 5.18e-9`;
- `Sigma_H` differs by `0.012155211646054737`, so the artifact correctly fails
  resolution tolerance.

Interpretation:

The abundance observables are not very sensitive to this two-row angular-grid
change, but the shear observable is not angular-grid converged.  This is an
anisotropic transport validation gap.  It is also not an ell-max ladder: the
live full-BBN non-LRS path uses fixed diagonal three-mode `A_modes`, and varies
S2 angular quadrature resolution here.

### Current ell hierarchy diagnostic, excluding deprecated Teff

Teff ablation is DEPRECATED and excluded from this audit as evidence.

The current executable ell diagnostic used here is:
`diagnostic_outputs/bd186_external_audit/lrs_collisionless_ell_ablation_sigma0p02_N0p2.json`

Scope:

- LRS collisionless neutrino distribution ell ladder;
- not Teff;
- not dynamic collision;
- not full-BBN endpoint;
- `Sigma_plus0 = 0.02`, `N_span = [0, 0.2]`, `ell_max = 2,4,6`.

Result:

- `converged = true`
- `converged_ell_max = 4`

Observables:

| ell_max | Sigma_plus_final | Pi_plus_final | A2_rms |
| ---: | ---: | ---: | ---: |
| 2 | 0.01776594499704123 | 0.014592288571443304 | 0.039397101945647155 |
| 4 | 0.017765888378877794 | 0.014591142053128807 | 0.03939643599143601 |
| 6 | 0.01776588838017448 | 0.014591142091570126 | 0.03939643599943977 |

Adjacent relative deltas:

- `(2,4)`: `A2_rms ~= 1.69e-5`, `Pi_plus ~= 7.86e-5`,
  `Sigma_plus ~= 3.19e-6`
- `(4,6)`: `A2_rms ~= 2.03e-10`, `Pi_plus ~= 2.63e-9`,
  `Sigma_plus ~= 7.30e-11`

An attempted LRS 3T weak/network ell ladder over the same short span failed
before writing an artifact:

- command target: `lrs_3t_ell_ablation_sigma0p02_N0p2.json`
- failure: `ValueError: X must contain only finite values`
- location: SciPy Radau dense numerical Jacobian probing through
  `run_augmented_lrs_collisionless_weak_network_3T_solve`

Interpretation:

The implemented collisionless LRS ell hierarchy converges at this short span.
The 3T weak/network ell ladder remains brittle in the SciPy path.  The current
dynamic non-LRS collision full-BBN path still lacks a generic ell-max ladder.

### Dynamic collision q14 full-span blocker

Current blocker artifact:
`diagnostic_outputs/bd186_external_audit/bd185_q14_collision_N4p8_cache_bound_probe.json`

Result:

- `passed = false`
- `physical_full_bbn_span_ready = false`
- `selected_dynamic_collision_payload_build_attempts_total = 344`
- `selected_wall_seconds_total = 137.69426292800927`
- `selected_step_count_total = 343`
- `selected_frozen_source_jax_full_jvp_jacobian_evaluations_total = 342`
- artifact is written; this is progress over earlier kill/no-artifact behavior.

Interpretation:

BD185 moved the dynamic collision path past the prior resource/OOM cliff enough
to write an artifact, but the high-order q14 collision-on path still does not
reach the full endpoint.  The current failure has moved from "process killed
before artifact" toward "solver/runtime h-collapse with finite telemetry."

## What Has Worked

IMPLEMENTED/VALIDATED or strongly supported by artifacts:

- Keeping Rodas5P as host solver while splitting the activated 9-species network
  into a network-only implicit corrector.
- BE/BDF2/Newton in `Y = X/A_mass`, with analytic network Jacobian support.
- Removing the phase-2 nuclear network from host Rodas stage algebra, which
  avoids the earlier trace-domain stage negativity failure.
- Event-style activation handling and log/positive coordinates for trace species
  in the host path.
- Raw Laguerre-vs-energy weight separation.
- q derivative manufactured tests and local q-stencil repair.
- Dynamic collision payload/cache bounds, which changed the q14 collision row
  from kill/no-artifact to finite failed artifact.
- Compact collision payload metadata modes, reducing evidence-retention memory
  pressure.
- BD186 verification repaired the electron-chemical-potential source contract:
  fixed mode keeps the supplied value, while charge-neutrality mode recomputes
  from the explicit evolved charge-asymmetry density or the current collision
  source payload instead of consuming a stale inbound internal
  `_electron_chemical_potential_MeV`.

## What Has Not Worked Or Is Not Enough

- Default `max_steps=64` is too small for full endpoint staging; strict isotropic
  endpoint needed `max_steps=512` in the fresh run.
- Full dynamic collision q14/q15 rows still fail before final endpoint.
- The collision `dA_modes` source is still largely a frozen additive source in
  the host linearization, with only a damping-style diagonal approximation
  available as an optional policy.
- Full non-LRS generic `ell_max` convergence is not wired into the current
  full-BBN live-source endpoint.  The live path uses fixed three diagonal modes.
- Some older LRS 3T weak/network SciPy ell paths are brittle and can generate
  nonfinite `X` during Radau numerical Jacobian probing.
- Artifact and validation plumbing has grown large enough to slow changes and
  obscure the active physics failure.

## Current Best Blocker Ranking

1. Dynamic collision source/Jacobian coupling.
2. Dynamic collision source construction cost and cache/metadata retention.
3. High-q Laguerre collision-source concentration and q-transport amplification.
4. Non-LRS angular/ell representation gap: full-BBN live path is fixed three-mode
   diagonal, while generic ell contracts exist elsewhere.
5. Phase-2 network cost; currently expensive but not the observed collision row
   killer.
6. Weak-rate correction history; physically important but not the current runtime
   cliff.

## Is This A Python-Native Limit?

Not primarily, based on current evidence.

Python orchestration and JSON-safe payload retention were real enough to cause
resource pressure before cache/metadata bounds were added.  However, after BD185
the q14 collision row writes an artifact and the remaining failure looks more
like numerical stiffness/source-coupling plus expensive collision contractions
than a generic "Python is too slow" limit.

The likely cost centers are:

- repeated dynamic PSTF radial source construction;
- high-q collision-source concentration;
- host JVP/Jacobian and LU/factorization volume;
- missing structured derivative of `collision.dA_modes` with respect to `A`;
- artifact plumbing retaining too much in hot paths unless compact mode is used.

Recommended language strategy:

- Do not rewrite the whole codebase in another language now.
- Keep Python for orchestration, artifact production, and science iteration.
- Move only proven hot kernels after profiling: collision radial contractions,
  source budget reductions, and maybe q-transport/collision Jacobian kernels.
- Prefer JAX-static kernels, Numba/Cython, Rust, or C++ only after a bounded
  profile demonstrates that Python/NumPy dispatch remains dominant after the
  source/Jacobian design is corrected.

## Overengineering Audit

This branch has a genuine evidence-honesty requirement, so some artifact and
claim-boundary structure is justified.  But the current code shows meaningful
overengineering risk.

### Finding 1: Two modules are too large and mix responsibilities

Symptom: `augmented_continuous_ap65_rhs.py` is about 16.8k lines and
`augmented_continuous_ap65_full_bbn_span_ladder.py` is about 9.4k lines.  They
mix physics RHS, solver policy, activation logic, corrector logic, artifact
shape, diagnostics, and claim-boundary text.

Source: Fowler, Refactoring - Long Method / Large Class; Ousterhout, A Philosophy
of Software Design - shallow modules and information leakage.

Consequence: a physics change often requires understanding artifact contracts,
runtime policy auto-resolution, and test fixtures at the same time.  This slows
breakthrough PRs and encourages local-minimum evidence plumbing.

Remedy: split by domain boundary, not by "helper" naming:

- host Rodas step/controller;
- phase-2 network corrector;
- dynamic collision payload/source;
- artifact summarization;
- validation case matrix.

### Finding 2: Artifact evidence schema has leaked into runtime design

Symptom: runtime functions carry many fields whose main purpose is artifact
explanation or claim-boundary wording.  Full metadata and JSON-safe conversion
were implicated in the collision payload resource cliff before cache bounds and
compact summaries.

Source: Hunt and Thomas, The Pragmatic Programmer - DRY decision duplication;
Winters et al., Software Engineering at Google - Hyrum's Law.

Consequence: it becomes difficult to tell whether a row is slow because of
physics, JAX/Rodas work, collision kernels, or evidence-retention overhead.

Remedy: hard-separate runtime payloads from audit payloads.  Runtime should carry
arrays and compact counters; artifact writers should expand only summaries,
checksums, and one-shot debug dumps when explicitly requested.

### Finding 3: Too many policy axes exist before the dominant blocker is solved

Symptom: rows can vary h ladders, retry factors, source-refresh policies,
Jacobian policies, q-grid policies, metadata policies, background policies,
collision component policies, and freedom matrices.

Source: Brooks, The Mythical Man-Month - second-system effect; McConnell, Code
Complete - design in construction.

Consequence: the branch can spend PRs adding knobs that produce more attribution
surfaces without reducing the collision-on endpoint failure.

Remedy: freeze most axes for the next blocker PRs.  Vary only one of:
collision source component, collision Jacobian structure, q operator, or payload
metadata.  Require each PR to move an existing runtime blocker.

### Finding 4: The current live path and generic ell contracts are not unified

Symptom: generic ell contracts exist, LRS ell convergence exists, and the current
full-BBN non-LRS live path uses fixed three diagonal modes.  This can lead
external readers to overinterpret "ell ladder" language.

Source: Evans, Domain-Driven Design - ubiquitous language; Ousterhout -
information hiding.

Consequence: anisotropic validation status is easy to overclaim.  A reviewer
cannot infer full non-LRS ell convergence from the current live-source endpoint.

Remedy: state the distinction in artifacts.  Either implement a real live
non-LRS ell ladder or rename the current full-BBN path as a fixed diagonal
three-mode S2-projection path.

## Recommended Next Implementation Direction

1. Add a structured collision Jacobian/source response pilot for `dA_modes`:
   start diagonal/block-local in q/species/mode, then compare against the
   current frozen-source and collision-diag policies.
2. Add collision source budget compression that is runtime-first:
   per-q/per-mode norms and checksums only, full dumps only for one-shot probes.
3. Add a bounded q14/q15 dynamic collision source-component matrix:
   payload-only, dQ-only, dA-only, full scaled alpha, full alpha=1.
4. Implement real full-BBN anisotropic mode/ell convergence only after the
   fixed three-mode dynamic collision path can reach the endpoint.
5. Keep the split phase-2 network corrector and Rodas5P host architecture.

## Files External Auditors Should Read First

1. `src/rabbit/jax/augmented_typeI_replay.py`
2. `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
3. `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
4. `src/rabbit/transport/augmented_collision_bridge.py`
5. `src/rabbit/transport/augmented_typeI_weak_network.py`
6. `src/rabbit/transport/augmented_nonlrs_transport.py`
7. `src/rabbit/jax/nonlinear_transport.py`
8. `src/rabbit/collisions/pstf_contractions.py`
9. `src/rabbit/collisions/pstf_process_catalog.py`
10. `src/rabbit/network/abundances_standard.py`
11. `src/rabbit/jax/solver_jax_rodas5p.py`

## Packet Artifacts

The external audit packet is intended to include:

- this overview;
- the focused collision-blocker review document;
- the external review prompt;
- current source files listed above;
- focused tests;
- fresh BD186 JSON artifacts;
- recent git history.
