# PR-T3D Capability Skeleton — Preflight Audit

## Scope

Landed:
- `src/rabbit/config/backend_capabilities.py` — adds the
  `JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT` capability and
  registers it in `CAPABILITY_BY_KEY`.
- `tests/test_inference_hierarchy_lock.py` — adds
  ``"jax_typeI_full_boltzmann_tier3_preflight"`` to
  `EXPECTED_CATALOG_KEYS`.

This is a **strictly additive capability skeleton**.  It registers
the introspection-level identity for the bounded tier-3
full-Boltzmann work landed across the prior PR-T3A/T3B/T3C
preflight slices, **without** promoting any path to canonical and
**without** adding a `CAPABILITY_BY_BACKEND` dispatch entry or a
`canonical_forward_solver` branch.  Following the existing
`JAX_EXTENDED_PSTF` pattern, the capability is discoverable via
`CAPABILITY_BY_KEY` for status-reporting / catalog-completeness
locks but is not selectable through the inference dispatch.

## Why candidate, not canonical

The full PR-T3D phase prompt promotes the path to canonical and
locks it against external cross-codes (LASAGNA / FortEPiaNO /
PRIMAT-AC2024) at ``|ΔY_p| < 5e-4``.  The current state of the
upstream preflight slices does not yet meet those gates:

* **PR-T3A** — collisionless full-phase-space shell is private
  candidate only; high-shear coarse-grid reduction degrades, and
  ``(N_μ, N_q, Σ) = (12, 20, 0.3)`` exceeded the bounded CPU audit
  budget.
* **PR-T3B** — pure-JAX nu-e + pair operators are wired through
  the bank-core dispatcher as a private
  ``collision_mode="jax_kernel_preflight"``.  End-to-end Rodas5P
  smoke at bounded shear succeeds; the FLRW
  ``|N_eff - 3.044| < 0.01`` lock against Mangano 2005 is **not
  yet** enforced and the off-grid pair-process PCHIP-vs-cubic
  parity gap remains at ``~8% rel``.
* **PR-T3C** — diagonal ν-ν is landed as a JAX-native skeleton
  with the symmetric ν-e-style matrix-element form rather than the
  Dolgov-Hansen-Semikoz appendix-A coefficient table.  Detailed
  balance is bounded at ``~7e-23`` (locked at ``< 1e-20``) and
  energy conservation at ``~1.8% rel`` (locked at ``< 5%``); no
  driver wiring, no SciPy reference, no FLRW ``N_eff`` lock.

Promoting the unified path to canonical without first calibrating
the absolute matrix-element prefactor and closing the FLRW
``N_eff = 3.044 ± 0.005`` gate would mislead consumers.  The
preflight-level capability registration documents the work-in-flight
without making promises the underlying surface cannot keep.

## Capability metadata

```python
JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT = BackendCapability(
    key="jax_typeI_full_boltzmann_tier3_preflight",
    backend="jax",
    tier="candidate",
    surface_class="candidate",
    physics_scope="TypeI_tier3_full_collision_preflight",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=2,
    validated_default=False,
    readiness_scope_contract="bounded_jax_tier3_full_boltzmann_preflight_v1",
    transport_scope_contract="full_phase_space_ray_v1",
    thermo_scope_contract="private_tier2_collision_moment_3T_v1",
    collision_scope_contract=
        "jax_kernel_plus_diagonal_nu_nu_skeleton_preflight_v1",
    ...
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar", "nux", "nuxbar"),
)
```

The contract strings are deliberately preflight-specific; promoting
to canonical will require new ``_canonical_v1`` variants once the
upstream gates close.

## Verification

- `pytest tests/test_inference_hierarchy_lock.py
  tests/test_registry_doc_sync.py
  tests/test_advanced_envelope_lock.py -q` →
  ``128 passed in 63.0 s``.  The catalog-size lock
  ``len(CAPABILITY_BY_KEY) == len(EXPECTED_CATALOG_KEYS)`` flips
  from 20 to 21 cleanly.
- `pytest tests/test_pr_t3a_collisionless_driver.py
  tests/test_pr_t3b_collision_preflight.py
  tests/test_pr_t3b_jax_operator_parity.py
  tests/test_pr_t3c_nu_nu_preflight.py
  tests/test_inference_hierarchy_lock.py -q` →
  ``135 passed in 186.4 s`` (no regression on the upstream T3*
  preflights).

## Per-cross-code diagnostic split (PR-T3D-PF #4)

The single composite gap test
``test_jax_kernel_preflight_flrw_gap_to_cross_codes`` was split
into per-cross-code parametrized variants so each gap is
independently visible in the ``pytest`` report and individual
drifts can be traced to a single reference:

- ``test_flrw_n_eff_gap_to_cross_code[code]`` for ``code`` in
  ``{LASAGNA, FortEPiaNO, PRIMAT-AC2024, Mangano 2005}`` — 4
  parametrized runs locking the per-code ``|N_eff - target|``
  bound at ``< 6e-2``.
- ``test_flrw_yp_gap_to_cross_code[code]`` for ``code`` in
  ``{LASAGNA, FortEPiaNO, PRIMAT-AC2024}`` (Mangano omits ``Y_p``)
   — 3 parametrized runs locking the per-code
  ``|Y_p - target|`` bound at ``< 6e-3``.
- ``test_flrw_baseline_measurement_invariant`` — single absolute
  lock on the FLRW measurement values
  (``Yp ≈ 0.2417042168``, ``N_eff ≈ 2.993427``) so that even a
  lockstep drift in every cross-code reference would be caught.

A new module-scoped pytest fixture
``flrw_jax_kernel_result`` runs the
``run_full_boltzmann_jax(jax_kernel_preflight, ...)`` solve once
and shares the result across all 8 parametrized test cases, so
the additional test surface costs only one extra solve relative
to the previous single composite test.

The canonical PR-T3D target tolerances (``5e-3`` for
``N_eff``, ``5e-4`` for ``Y_p``) are now exposed as named
module constants ``_CANONICAL_N_EFF_TOL`` /
``_CANONICAL_Y_P_TOL`` and surfaced in every parametrized
test's failure message, so the canonical destination is always
visible alongside the current preflight bound.

## Cross-code preflight bound tightening (PR-T3D-PF #3)

After the PR-T3C-PF #3 / #4 PCHIP -> JAX cubic spline swap inside
``pair_collision_jax`` and ``nu_nu_scattering_jax`` landed, the
FLRW cross-code gap was re-measured on
``jax_kernel_preflight``:

- ``Y_p = 0.2417042168``
- ``N_eff = 2.993427``

Both numbers are unchanged at machine precision compared to the
pre-swap measurements.  This is expected: the cubic-spline swap
only tightens parity for *off-grid* interpolations, while the
bank-core dispatch in ``_collision_jax_kernel_bank_core_jax``
uses matched Laguerre grids (``y_3 == q_nodes``,
``y_2 == q_nodes`` for the matched-grid configuration).  The
swap therefore does not change the production-runtime ``df/dN``
values at the bank-core level.

The cross-code skeleton tests
(``test_jax_kernel_preflight_flrw_gap_to_cross_codes``) are
tightened to

- ``|N_eff - target| < 6e-2`` (was ``< 1e-1``; measured ``~5.1e-2``)
- ``|Y_p - target| < 6e-3``   (was ``< 1e-2``; measured ``~5.3e-3``)

These are still preflight bounds (canonical PR-T3D targets are
``5e-3`` and ``5e-4`` respectively).  The tighter locks ensure
that any future drift — e.g., from the deferred PR-T3B q-grid
remap fix targeting the ``dQ_α`` sign convention — is flagged
explicitly rather than silently widening the gap.

## Cross-code fixture skeleton (PR-T3D-PF #2)

Landed:

- `tests/fixtures/tier3_cross_code.json` — published reference
  values for LASAGNA (Escudero 2019), FortEPiaNO (Froustey,
  Pitrou, Volpe 2020), PRIMAT-AC2024 (Pitrou 2024) and the
  Mangano 2005 SM benchmark.  Format-version-locked to ``v1``
  with a metadata block carrying the canonical tolerance targets
  (``|ΔY_p| < 5e-4``, ``|N_eff - 3.044| < 5e-3``) and an explicit
  preflight-status note pointing back to the audit docs.
- `tests/test_pr_t3d_cross_code_skeleton.py` — 4 tests:
  - fixture has the four required cross-code entries
    (LASAGNA / FortEPiaNO / PRIMAT-AC2024 / Mangano 2005);
  - fixture metadata fields (canonical tolerance targets,
    preflight-status note) are well-formed;
  - cross-code N_eff entries cluster within ``5e-3`` (3.043 vs
    3.044 vs 3.044 — Mangano matches at 3.044);
  - the FLRW (Σ=0) preflight result matches every cross-code
    entry to **loose preflight bounds** (N_eff gap ``< 0.10``,
    Y_p gap ``< 0.01``).  These are deliberately loose: the
    canonical PR-T3D parity tests will reject a 0.05 N_eff gap;
    the loose lock here is a *baseline* that flags any further
    widening over future preflight refactors and gives the
    canonical work a measurable starting point.

These tests are **not** the canonical PR-T3D parity gate.  They
are diagnostic / reporting tests that snapshot the current
preflight gap (FLRW N_eff ≈ 2.993 → ≈ 0.05 gap to 3.044)
explicitly.

## Adversarial self-audit

- **Surface-class consistency** (``test_s2_surface_class_consistency``
  in `tests/test_registry_doc_sync.py`): pass — the new capability
  has ``tier == surface_class == "candidate"``.
- **Tier alignment** (``test_s3_tier_surface_alignment``): pass —
  candidate tier matches candidate surface_class.
- **Catalog completeness**
  (`tests/test_inference_hierarchy_lock.py::test_in_catalog`):
  pass — the new key is in `EXPECTED_CATALOG_KEYS` and in
  `CAPABILITY_BY_KEY`.
- **No dispatch leak**: the new capability is **not** in
  `CAPABILITY_BY_BACKEND`, so the inference dispatch tests
  (`TestInvalidBackendRejection`) continue to reject the unsupported
  backend string.  Verified by ``test_advanced_envelope_lock.py``
  which still locks ``len(CAPABILITY_BY_BACKEND) == 10``.
- **No silent promotion**: the notes string explicitly documents
  preflight status and the open gates (DH-S calibration, FLRW
  ``N_eff`` lock, cross-code parity), so any reader of the registry
  can immediately see this is not a usable canonical path.

## Verdict

Conditional pass.

The PR-T3D capability skeleton is now landed at the introspection
level, providing a single registry key under which the prior
PR-T3A/T3B/T3C preflight slices can be referenced.  The path is
**not** promoted to canonical and is **not** dispatchable through
the inference layer.  Catalog-size, surface-class and tier-alignment
locks continue to pass.

What remains for the full PR-T3D promotion (deferred):

* close the upstream PR-T3B FLRW ``N_eff`` lock against Mangano
  2005;
* land the Dolgov-Hansen-Semikoz appendix-A coefficient table for
  PR-T3C and lock detailed balance + energy conservation at the
  strict ``1e-12`` / ``1e-14`` phase-prompt targets;
* wire the unified tier-3 path through bank-core as a single
  ``collision_mode="full_collision_tier3"`` (or equivalent) and
  build the cross-code fixture
  (`tests/fixtures/tier3_cross_code.json`);
* add the LASAGNA / FortEPiaNO / PRIMAT-AC2024 parity tests with
  ``|ΔY_p| < 5e-4`` and ``|N_eff - 3.044| < 0.005``;
* register a public dispatch key ``"jax_full_collision_tier3"`` in
  `CAPABILITY_BY_BACKEND` and add the corresponding branch in
  `canonical_forward_solver`;
* promote the capability `tier` from ``"candidate"`` to
  ``"canonical"`` and update the contract strings to ``_canonical_v1``
  variants;
* refresh the registry-generated tables in `STATUS.md` /
  `README.md` / `BACKEND_CAPABILITY_MATRIX.md` /
  `PROMOTION_GATES.md` via
  `scripts/render_capability_tables.py --apply`.
