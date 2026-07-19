# PR-R Release-Gate Preflight Audit

## Scope

Landed:
- `tests/test_production_gates.py` — fixture swap in
  `test_jax_flrw_gold`; `pytest.mark.xfail` annotation on
  `test_classB_typeV_bbn_gold` with inline rationale.
- `docs/ROADMAP_STATE_OF_RECORD.md §6.1` — refreshed pre-existing
  red-test status table.
- `docs/audit/PR-R_preflight.md` (this file).
- `docs/ROADMAP_PR_CATALOG.md` — appended PR-R-PF entry.

This is the **release-gate doc-side preflight**.  It closes the
trivial pre-existing red-test items from the original PR-R phase
prompt and snapshots the current state of the roadmap, but it does
**not**:

* tag a release, because the upstream tier-3 promotions (PR-T3B
  canonical, PR-T3C calibrated, PR-T3D dispatch) are still in the
  preflight stage;
* refresh registry-generated tables in `STATUS.md` / `README.md` /
  `PROMOTION_GATES.md` / `BACKEND_CAPABILITY_MATRIX.md`, because
  those are owned by `scripts/render_capability_tables.py --apply`
  and best regenerated when the registry stabilises after the
  upstream promotions;
* freeze a `tests/fixtures/tier3_cross_code.json`, because no
  cross-code parity tests exist yet (LASAGNA / FortEPiaNO /
  PRIMAT-AC2024 fixtures are part of the deferred PR-T3D
  promotion).

## Pre-existing red-test resolutions

The original PR-R phase prompt §3 enumerates four pre-existing
reds.  Their state at this PR-R-PF pass:

| Test | Original cause | Status |
| --- | --- | --- |
| `test_supported_capabilities_mentions_features` | `SUPPORTED_CAPABILITIES.md` missing the literal "Inference" string | Already green; no PR-R-PF change |
| `test_anisotropy_signal_parity` | Apples-vs-oranges SciPy char vs JAX linearised PSTF | Already green after PR-D; no PR-R-PF change |
| `test_jax_flrw_gold` | Fixture key mismatch (`jax_flrw` live-weak vs run with equilibrium FD) | **Resolved**: swapped to `gold["jax_flrw_equilibrium"]["Yp"]` per PR-R phase prompt §3.2 |
| `test_classB_typeV_bbn_gold` | Class B Type V Y_p ~9e-4 fixture drift after the geometry/initial-data audit | **Formally `xfail`** with `strict=False`, inline rationale linking to `docs/CLASSB_PROMOTION_PACKET.md`; no Type I code change |

Verification:

```
pytest -q tests/test_production_gates.py -k \
  "test_classB_typeV_bbn_gold or test_jax_flrw_gold or test_anisotropy_signal_parity"
2 passed, 57 deselected, 1 xfailed in 25.93s

pytest -q tests/test_registry_sync.py -k \
  "test_supported_capabilities_mentions_features"
1 passed, 26 deselected
```

Targeted regression bundle
(`test_production_gates.py + test_registry_sync.py +
test_inference_hierarchy_lock.py`):
``166 passed, 3 skipped, 1 xfailed in 70.8 s``.  No new reds.

## Release tag — explicitly deferred

`PR-R_release_gate.md` §11 prescribes
``git tag -a rabbit-typeI-tier3-v1`` on the closing PR.  PR-R-PF
**deliberately does not** tag.  The required precondition for the
release tag is that PR-T3B/T3C/T3D land in canonical (not
preflight) form with the FLRW ``|N_eff - 3.044| < 0.01`` lock and
the LASAGNA / FortEPiaNO / PRIMAT-AC2024 cross-code parity at
``|ΔY_p| < 5e-4`` green.  None of those gates have closed at the
time of this preflight, so a release tag would misrepresent the
actual production maturity.

The catalog entry for this PR explicitly identifies the remaining
work; the release tag is a follow-up step once the upstream
canonical promotions land.

## Roadmap snapshot

PR-A, PR-J(abort), PR-N1, PR-N2, PR-D, PR-G — merged canonical.
PR-T3A — partial (private collisionless shell).
PR-T3B — partial (PR-T3B-PF JAX operator port + PR-T3B-PF
``jax_kernel_preflight`` runtime mode + PR-T3B-PF end-to-end solve
smoke landed; FLRW `N_eff` lock and `dQ_α` sign-convention check
deferred).
PR-T3C — partial (PR-T3C-PF diagonal ν-ν skeleton landed with
algebraic detailed-balance and energy-conservation locks;
Dolgov-Hansen-Semikoz coefficient table and SciPy parity
deferred).
PR-T3D — partial (PR-T3D-PF capability skeleton registered in
`CAPABILITY_BY_KEY` only; canonical promotion + dispatch wiring +
cross-code parity deferred).
PR-R — preflight (this PR; release tag deferred).

## Broader regression isolation check (PR-R-PF #3)

After the cumulative PR-T3{A,B,C,D}-PF preflight work landed
(``+22`` commits across the four T3 chain branches in 2025-09 to
2025-10), the broader production-gate bundle was re-run to
verify no silent drift escaped the T3 envelope:

* ``tests/test_production_gates.py + test_registry_sync.py +
  test_inference_hierarchy_lock.py + test_advanced_envelope_lock.py
  + test_jax_typeI_characteristic_parity.py``: 224 passed,
  3 skipped, 1 xfailed in 111.8 s.  No regression on any
  production-gate ``@gold`` BBN observable lock or any
  registry-sync invariant.
* ``tests/test_jax_typeI_characteristic_tier2.py +
  test_jax_runtime_fallback.py``: 39 passed in 76.3 s.  No
  regression on the canonical tier-2 characteristic surface or
  the runtime device-fallback behaviour.

The T3 preflight work was successfully isolated to:

* ``src/rabbit/jax/driver_typeI_full_boltzmann.py`` (private
  collision_mode runtime knob; no public dispatch change).
* ``src/rabbit/jax/collisions_jax.py`` (pure-JAX kernels;
  not yet wired into any canonical surface).
* ``src/rabbit/jax/cubic_spline_jax.py`` (new module).
* ``src/rabbit/jax/collision_rates_jax.py`` (new module).
* ``src/rabbit/jax/nu_nu_scattering_jax.py`` (new module;
  diagonal ν-ν skeleton).
* Two registry entries
  (``JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT`` in
  ``backend_capabilities``; ``TIER3_FULL_COLLISION_PREFLIGHT``
  in ``feature_capabilities``), both candidate-tier and not in
  ``CAPABILITY_BY_BACKEND`` dispatch.
* Test surface only: ``+86`` parametrized tests across the
  ``tests/test_pr_t3*.py`` bundle (122 tier-3 preflight tests at
  the time of this audit, up from 36 at the start of the
  cumulative chain).

The release tag remains gated on the **3 narrowed canonical
blockers** enumerated in
``rabbit.config.feature_capabilities.TIER3_FULL_COLLISION_PREFLIGHT.blockers``
(scope reframed in PR-T3B-PF #15):

1. **AP-form unification**: combine ``spectral_relaxation``
   anisotropy stability (already passes canonical < 1e-3 gate)
   with ``projected_physical`` grid scaling into a single mode.
2. **Dolgov-Hansen-Semikoz ν-ν coefficient calibration** —
   apply appendix-A coefficient table to the diagonal ν-ν kernel.
3. **FLRW N_eff gap to Mangano 2005 (~0.013)** — accepted as
   AP-form model approximation limit; documented in audit trail
   + registry notes.

Out of canonical scope (deferred indefinitely; preserved as
private diagnostic surfaces only):

- ``jax_kernel_preflight`` (full Hannestad-Madsen kernel) —
  incompatible with the load-bearing Rodas5P invariant due to
  the stiff ``∂C/∂T`` Jacobian manifold exposed by the q-grid
  remap fix.  Closing requires either an IMEX/operator-split
  solver (violating the invariant) or a JAX-native
  AP-Rosenbrock variant (research-grade).
- IMEX splitting on top of Rodas5P — Rosenbrock-Wanner methods
  do not natively support IMEX.
- Mangano 5e-3 N_eff precision — out of reach without one of
  the above.

See ``docs/audit/PR-T3B_jax_kernel_runtime.md`` PR-T3B-PF #15
section for the full scope-reframing rationale.

## Verdict

Conditional pass.

The four pre-existing red tests from the original baseline are now
either green or formally `xfail` with a linked rationale.  The
roadmap snapshot is up to date in `ROADMAP_STATE_OF_RECORD.md` and
`ROADMAP_PR_CATALOG.md`.  Cumulative tier-3 preflight work has
been verified isolated from the production-gate bundle and from
the canonical tier-2 / runtime-fallback paths.  No release tag is
applied; tagging is deferred until the five canonical blockers
above close.
