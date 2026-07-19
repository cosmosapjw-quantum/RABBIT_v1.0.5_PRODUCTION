# BD612 Current-Code Hostile Audit Prompt

Date: 2026-07-08
Repository: `/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION`
Observed branch during prompt drafting: `feature/bianchi-i-full-nonperturbative`

Use this as a ready-to-send prompt for an external auditor or a fresh high-reasoning
agent that can inspect this repository in place. It supersedes broad Bianchi-BBN
audit prompts when the target is the current post-clean-core, post-AP65-deflation
code surface.

Prompt-engineering stance used here:

- Ground every claim in repository files, commands, or generated artifacts.
- Prefer explicit success criteria, output contracts, and falsification tests over
  open-ended "think hard" instructions.
- Do not ask the auditor to reveal hidden chain-of-thought. Require concise
  verification traces, evidence tables, and decision records instead.
- Separate implementation parity, internal consistency, and physical validation.
- Make "done" concrete: files inspected, commands run or explicitly skipped,
  claim statuses assigned, severe findings ranked, and next patch bounded.

```text
[TITLE]
Journal-Grade Hostile Current-Code Audit — RABBIT Bianchi-I / No-QKE BBN,
Clean FLRW Decoupling Core, Collision Bridges, Claim Gates, and Post-Deflation
Capability Honesty

[ROLE]
You are an external adversarial reviewer for early-universe cosmology, BBN,
relativistic kinetic theory, anisotropic Bianchi cosmology, stiff numerical ODEs,
and scientific software verification.

Your job is not to encourage the project. Your job is to decide, from live code
and executable evidence, what this repository currently implements, what it only
parity-locks, what it only internally checks, what is stale after deletion, and
what must not be claimed.

Do not infer implementation from README wording, module names, prior commit
messages, or prior audit conclusions. Treat those as claims to verify.

[REPOSITORY]
Work in place:

  /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION

Assume no prior conversation context. Reconstruct the current state from the
repo itself.

[FIRST READS — REQUIRED BEFORE CLAIMS]
Read these before making any positive claim:

1. `AGENTS.md`
2. `bbn_codex_anti_drift_cost_effective_policy.md`
3. `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
4. `docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`
5. `docs/audit/INDEX.md`
6. `docs/audit/BD205_qlaguerre_collision_root_cause_2026-07-01.md`
7. `docs/audit/B4_PR4_collisional_char_reference_unsound_2026-07-01.md`
8. `docs/audit/BD610_external_clean_core_decoupling_and_deflation_audit_and_roadmap_prompt_2026-07-01.md`
9. `docs/audit/BD611_clean_core_development_direction_meta_plan_2026-07-02.md`
10. The latest `git log --oneline -30`

[HARD SCOPE]
Audit the current code surface. Do not revive broad AP65 audit-only scope unless
live imports, tests, capability registries, or scripts still reference it.

In scope:

1. Clean dynamic-collision FLRW decoupling core:
   - `src/rabbit/collisions/dynamic_collision_core.py`
   - `src/rabbit/collisions/dynamic_collision_driver.py`
   - `src/rabbit/collisions/deterministic_reference.py`
   - `tests/test_dynamic_collision_core.py`
   - `tests/test_dynamic_collision_driver.py`

2. Collision-operator parity and JAX characteristic candidate wiring:
   - `src/rabbit/collisions/projected_operator.py`
   - `src/rabbit/transport/teff_collision_bridge.py`
   - `src/rabbit/jax/collision_operator_jax.py`
   - `src/rabbit/jax/teff_collision_bridge_jax.py`
   - `src/rabbit/jax/driver_typeI_char.py`
   - `tests/test_physical_collision_operator_reference.py`
   - `tests/test_jax_collision_operator_parity.py`
   - `tests/test_jax_teff_collision_bridge_parity.py`
   - `tests/test_b4_jax_char_collision_wiring.py`
   - `tests/test_b4_collisional_char_reference_unsound.py`

3. Surviving Bianchi-I / non-LRS substrate after AP65 deflation:
   - `src/rabbit/transport/augmented_pstf_distribution.py`
   - `src/rabbit/transport/augmented_nonlrs_transport.py`
   - `src/rabbit/transport/augmented_typeI_nonlrs_collisionless.py`
   - `src/rabbit/transport/augmented_typeI_observables.py`
   - `src/rabbit/jax/characteristic_rays_jax.py`
   - `src/rabbit/jax/characteristic_rays_nonlrs_jax.py`
   - related `tests/test_b5_*`, `tests/test_augmented_*`, and characteristic tests

4. Claim, capability, and promotion surfaces:
   - `src/rabbit/config/claim_gates.py`
   - `src/rabbit/config/backend_capabilities.py`
   - `src/rabbit/config/feature_capabilities.py`
   - `SUPPORTED_CAPABILITIES.md`
   - `PROMOTION_GATES.md`
   - `README.md`
   - `STATUS.md`
   - `scripts/promotion_check.py`
   - `tests/test_claim_gates.py`
   - `tests/test_promotion_check_skip_is_not_pass.py`

5. Deletion and runtime hygiene after AP65 deflation:
   - active `scripts/*.py`
   - active tests under `tests/`
   - imports from deleted AP65 validation/weak/transport/jax audit surfaces
   - stale docs or capability strings that still advertise deleted capabilities

6. Public/canonical forward paths:
   - `src/rabbit/inference/forward_likelihood.py`
   - `src/rabbit/drivers/full_coupled_typeI.py`
   - `src/rabbit/jax/driver_typeI.py`
   - `src/rabbit/jax/driver_typeI_char.py`
   - `src/rabbit/jax/driver_typeI_full_boltzmann.py`

Out of scope unless a live code path contradicts it:

- QKE support.
- Public-production or publication-ready claims.
- New readiness/hash/manifest/figure-gate proposals.
- Whole-rewrite recommendations not backed by measured same-physics blockers.

[NON-NEGOTIABLE REPO RULES]
Use these as hard audit law:

- QKE is out of scope.
- Public production, production SMC, and publication-ready support must not be
  claimed.
- CPU-JAX and in-tree Rodas5P remain the repeated-run/backend target unless a
  measured blocker proves otherwise.
- Preserve raw failed states: negative abundances, negative `Y_p`, NaNs,
  rejected solver attempts, non-endpoint trajectories, and failed artifacts.
- Internal consistency, self-consistency, parity-to-a-model, and smoke tests are
  not physical validation.
- Segment-only benchmarks must be labeled segment-only.
- Do not recommend adding another gate, wrapper, manifest, hash relay, readiness
  page, or figure surface unless it deletes/consolidates older surface and
  directly moves a physics, solver, endpoint, or performance blocker.

[CLAIM STATUS VOCABULARY — USE EXACTLY]
Use only these labels where applicable:

- IMPLEMENTED: code exists and has been executed/tested.
- VALIDATED: independently checked by test, benchmark, derivation, external
  anchor, or reproducible artifact appropriate to the claim.
- DERIVED: mathematically derived in the document, with assumptions stated.
- SPECIFIED: defined in a design/spec document but not yet implemented.
- PROPOSED: plausible research direction, not yet derived or implemented.
- SPECULATIVE: physically/mathematically interesting but unsupported.
- DEPRECATED: superseded and should not guide implementation.
- FORBIDDEN: explicitly disallowed pattern or assumption.

Important calibration:

- "Parity-locked to numpy" can be IMPLEMENTED or internally VALIDATED for
  engineering fidelity, but it is not physical validation of collision physics.
- "Energy conserved by construction" can be IMPLEMENTED, but it does not prove
  the collision rate, transfer quadrature, or physical `N_eff` is correct.
- "N_eff close to 3.044" is not validation by itself.
- "Clean core executes to endpoint" is an engineering milestone; it is not
  Bianchi-I full-BBN validation.

[PRIVATE METHOD POLICY]
Use private scratch work as needed, but do not expose hidden chain-of-thought.
In the final output, show:

- decisions,
- concise reasoning summaries,
- evidence paths,
- command outputs or skipped-command reasons,
- independent verification questions and answers,
- falsification attempts and results.

Do not output long stream-of-consciousness reasoning.

[RECOMMENDED FIRST COMMANDS]
Start with a low-cost current-state pass:

```bash
git status --short
git log --oneline -30
python --version
find src/rabbit scripts tests -type f -name '*.py' -print0 |
  xargs -0 wc -l | sort -n | tail -40
rg -n "publication-grade|publication ready|publication-ready|validated collision physics|public production|QKE|full-BBN|AP65|FB-[0-9]+|readiness|manifest|hash relay" \
  README.md STATUS.md SUPPORTED_CAPABILITIES.md PROMOTION_GATES.md docs src/rabbit tests scripts
```

If running tests, use the repo venv and CPU JAX:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dynamic_collision_core.py \
  tests/test_dynamic_collision_driver.py \
  tests/test_deterministic_collision_reference.py

env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_physical_collision_operator_reference.py \
  tests/test_jax_collision_operator_parity.py \
  tests/test_jax_teff_collision_bridge_parity.py \
  tests/test_b4_jax_char_collision_wiring.py \
  tests/test_b4_collisional_char_reference_unsound.py

env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_claim_gates.py \
  tests/test_promotion_check_skip_is_not_pass.py \
  tests/test_finite_shear_external_anchor.py

env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/promotion_check.py --status
```

If a test is slow, mark it SKIPPED with a reason. Never report validation unless
the command actually ran and passed.

[AUDIT HYPOTHESES TO CONTRAST]
Evaluate these hypotheses explicitly. More than one may be partly true.

H1. The clean FLRW decoupling core is a real engineering repair: it fixes the
old q-Laguerre temperature-amplifier collapse and reaches endpoint with
fail-closed guards, but it remains internally anchored.

H2. The clean core is self-consistent but may still have a load-bearing physics
normalization or transfer-accuracy blocker, especially in deterministic collision
rate scaling, comoving/thermal interpolation, or `1/(2*pi^2)`/degeneracy
bookkeeping.

H3. The B4 JAX characteristic collision path is faithful wiring of a calibrated
RTA/gather-scatter model, not validated anisotropic collision physics.

H4. AP65 deflation deleted a large stale audit surface, but capability docs,
scripts, tests, or registries may still advertise deleted non-LRS/full-BBN/weak
network/AP65 capabilities.

H5. The repo's claim firewall is stronger than before but may still miss
overclaims outside curated files or allow "validated" wording to creep through
via registries, generated docs, or tests.

[PHASE 0 — CURRENT TARGET RECONSTRUCTION]
Produce a table:

| Track | Claimed purpose | Live files | Tests/gates | Actually reaches production path? | Claim status |
|---|---|---|---|---|---|

Use at least these tracks:

1. Clean FLRW decoupling core.
2. Deterministic collision reference.
3. B4 calibrated-RTA collision twins.
4. JAX characteristic LRS tier-2 collision candidate wiring.
5. B5 non-LRS operator substrate.
6. Canonical/public forward solver and inference dispatch.
7. Claim gates and capability registries.
8. Post-deflation script/test hygiene.

State the source of truth for each track: executable output, unit test, audit
note, registry text, or commit claim.

[PHASE 1 — CONTRACT / INTERFACE AUDIT]
Reconstruct these contracts before judging:

1. Clean driver frame contract:
   - state is fixed comoving `Y` distribution plus `T_gamma`;
   - `z = a*T_gamma`;
   - massless free-streaming gives `df/dN|_Y = C/H`;
   - collisionless endpoint should recover the analytic entropy limit within
     the stated EOS convention.

2. Energy-transfer contract:
   - `G = int Y^3 df/dN dY`;
   - frame factor `(T_gamma^4 / z^4)/(2*pi^2)`;
   - degeneracy factors `2` for `nu_e + anti-nu_e`, `4` for `nu_x` bank;
   - plasma loss is defined from the same evolved `df`.

3. Deterministic collision-reference contract:
   - Pauli six-monomial statistical factor;
   - detailed-balance null;
   - heating sign for colder neutrinos;
   - rate dimensions and `T` scaling;
   - `G_F^2 T^4` vs expected weak-rate scaling.

4. Calibrated RTA / gather-scatter bridge contract:
   - numpy reference meaning;
   - JAX parity meaning;
   - `delta_I` / `I_coll` semantics;
   - LRS tier-2 candidate scope;
   - non-LRS collision closure still unpromoted.

5. Claim-gate contract:
   - missing tests are red;
   - skipped tests are not pass;
   - external finite-shear anchor is fail-closed;
   - promotion check must not silently green stale/deleted nodes.

If any contract is ambiguous, mark the ambiguity as a finding.

[PHASE 2 — PHYSICS / MATH AUDIT]
For each item, record status, evidence, and residual risk:

1. Does collisionless clean-driver `N_eff` test the intended frame, or can it
pass while a collisional frame/interpolation error remains?
2. Does energy conservation by construction hide an inaccurate continuum energy
transfer?
3. Is the deterministic reference collision rate dimensionally and physically
plausible?
4. Does detailed balance hold for the actual callable collision field, not only
for isolated algebraic factors?
5. Are degeneracy factors counted once and only once?
6. Are clipped `f in [0,1]`, `H_safe`, and `max_clip_excursion` diagnostics
strong enough to prevent silent repair of bad states?
7. Does the B4 `I_coll` scalar represent only isotropic energy shift, and what
finite-shear/non-LRS physics does it necessarily miss?
8. Is the surviving B5 non-LRS substrate an active driver capability or a useful
operator-level island?
9. Are any docs or registries upgrading internal consistency to validation?

[PHASE 3 — EQUATION-TO-CODE MAPPING]
For each load-bearing equation or convention, point to the exact implementation
file and function. Include:

- comoving-to-thermal resampling and thermal-to-comoving collision-field mapping;
- `spectral_energy_moment`;
- `plasma_prefactor`;
- `_collision_field`;
- `collision_bank_energy_sources_per_efold`;
- `integrate_flrw_decoupling`;
- JAX `apply_gather_scatter_collision_jax`;
- driver `I_coll` construction and handoff;
- canonical dispatch guards that keep candidate collision paths out of public
  inference;
- claim-gate definitions and promotion-check behavior.

Mark each mapping:

- exact,
- approximate with documented regime,
- approximate but under-documented,
- disconnected,
- stale/deleted.

[PHASE 4 — NUMERICAL / PIPELINE AUDIT]
Audit:

1. stiff solver suitability and endpoint fail-closed behavior;
2. tolerance and `n_q` convergence meaning;
3. interpolation and quadrature error;
4. rate normalization and decoupling-window plausibility;
5. underflow/overflow/cancellation in tails;
6. raw state preservation vs clipping;
7. JAX x64 assumptions;
8. cache/state leaks;
9. default-off/candidate path fences;
10. runtime impact of remaining scripts and tests after AP65 deflation.

Do not treat a green smoke test as numerical validation.

[PHASE 5 — TEST / DOCS / REGISTRY HONESTY]
Classify tests into:

- import/smoke,
- unit algebra,
- parity-to-reference,
- internal consistency,
- numerical regression,
- external validation,
- claim firewall,
- deletion/hygiene.

Then answer:

1. Which tests would catch a wrong collision-rate normalization?
2. Which tests would catch a wrong comoving/thermal transfer?
3. Which tests would catch a stale capability registry after deletion?
4. Which tests would catch a public-dispatch leak of candidate collisions?
5. Which tests are false-green, schema-only, or self-consistency locks?
6. Which slow tests are required before any stronger physics claim?

Search docs and registries for stale AP65 / full-BBN / public-production /
publication-grade / validated-collision-physics wording. A single stale
generated paragraph can be a high-severity claim-surface finding if it would
mislead a reader about current capability.

[PHASE 6 — ADVERSARIAL VERIFICATION]
For your provisional conclusion, generate 8-12 independent verification
questions. Answer each using code, tests, or explicit "not run / not present".

Required questions:

1. What is the first positive claim that breaks if all docs are ignored and only
   executed code is considered?
2. What is the strongest current engineering contribution that survives a hostile
   read?
3. Is any parity-to-model result being described as physics validation?
4. Is any internal consistency result being described as external validation?
5. Did AP65 deflation silently remove a capability while registries still
   advertise it?
6. Does `promotion_check.py --status` fail closed where it should?
7. Does `enable_collisions=True` leak into canonical inference/public dispatch?
8. Does `N_eff` closeness to a known value influence any claim beyond what tests
   justify?
9. Are raw failed states preserved rather than hidden?
10. What one cheap executable test would most likely overturn your conclusion?

[PHASE 7 — FINDINGS]
Return 5-12 severe findings. For each:

Severity: P0 / P1 / P2 / P3
Status: IMPLEMENTED / VALIDATED / DERIVED / SPECIFIED / PROPOSED / SPECULATIVE / DEPRECATED / FORBIDDEN
Symptom:
Source:
Evidence:
Consequence:
Minimal remedy:
Proof needed:

Severity guide:

- P0: false physics/publication claim, public dispatch leak, hidden failed state,
  or deleted capability still advertised as current.
- P1: load-bearing physics/numerical uncertainty that can invalidate current
  `N_eff`, collision, or Bianchi capability interpretation.
- P2: maintainability/test/registry issue that can cause future drift or
  false-green review.
- P3: cleanup or wording issue with limited behavioral risk.

[PHASE 8 — MINIMAL PATCH PLAN]
Propose at most 5 patches. Each patch must be small and reviewable.

For each patch:

| Patch | Files | Net LOC estimate | Blocker moved | Why now | Test/command | Risk | Cost-effectiveness verdict |
|---|---|---:|---|---|---|---|---|

Allowed patch types:

- exact-transfer vs df-derived plasma coupling probe;
- collision-rate dimension / `Gamma/H` scaling probe;
- collisionless EOS split probe;
- public/candidate dispatch fence hardening;
- stale registry/doc deflation;
- deletion/lazy-import hygiene;
- one targeted external-anchor hook that fails closed.

Disfavored patch types:

- new broad readiness wrappers;
- new figure or manifest surfaces;
- another claim ledger without deleting stale surfaces;
- broad rewrites;
- optimizing a cheap segment while physics normalization is unresolved.

[PHASE 9 — FINAL VERDICT]
Choose exactly one:

- CRITICAL_BLOCKER
- PARTIAL_ENGINEERING_PASS_PHYSICS_BLOCKED
- INTERNAL_CONSISTENCY_PASS_EXTERNAL_VALIDATION_MISSING
- CLAIM_SURFACE_STALE_AFTER_DELETION
- READY_FOR_NEXT_MEASUREMENT_PR
- INCONCLUSIVE_NEEDS_COMMANDS

Also provide:

1. One-line reason.
2. Strongest surviving contribution.
3. Most dangerous overclaim.
4. Next single patch.
5. One thing that must not be touched yet.
6. Commands run.
7. Commands skipped and why.

[OUTPUT FORMAT — EXACT ORDER]
1. Current target reconstruction table
2. Contract/interface audit
3. Physics/math audit ledger
4. Equation-to-code mapping audit
5. Numerical/pipeline audit
6. Test/docs/registry honesty audit
7. Adversarial verification questions and answers
8. Ranked findings P0-P3
9. Minimal patch plan
10. Final verdict
11. Changed-files risk note if you propose edits

[DO NOT]
- Do not call self-consistency validation.
- Do not call parity-to-calibrated-RTA validation of collision physics.
- Do not call endpoint completion Bianchi-I full-BBN validation.
- Do not infer from commit messages.
- Do not hide skipped commands.
- Do not recommend broad rewrites without same-physics measurements.
- Do not add wrapper/gate/readiness work unless it deletes stale surface and
  moves a blocker.
- Do not claim QKE, public production, production SMC, or publication-ready
  support.
```
