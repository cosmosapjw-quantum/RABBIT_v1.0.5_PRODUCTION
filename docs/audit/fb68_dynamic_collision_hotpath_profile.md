# FB68 Dynamic Collision Hot-Path Profile

Date: 2026-05-20

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/profile_augmented_dynamic_collision_hotpath.py \
  --output diagnostic_outputs/fb68_dynamic_collision_hotpath_profile.json \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --warm-repetitions 2 \
  --cprofile-top 5
```

Result:

- contract: `augmented_nonlrs_dynamic_collision_payload_hotpath_profile_fb68_v1`
- passed: `true`
- violations: `[]`
- n_species: `4`
- cache-disabled cold median: `2.3705820268951356 s`
- shared-cache cold-miss median: `1.6804822400445119 s`
- shared-cache warm-hit median: `0.00948077195789665 s`
- shared-cache cold-to-warm speedup factor: `177.2516254485815`
- source-factory cache entries after profile: `1`
- radial-grid cache entries after profile: `18`
- first warm cache-hit observed: `true`

Top cumulative samples in the cold-miss row identify
`build_augmented_nonlrs_pstf_radial_moment_thermo_source(...)` as the dominant
constructor-side cost.  The warm hit row shows the existing source-factory/radial
grid caches already make repeated payload refresh cheap.  The next optimization
target is therefore AP65/AP6 radial source factory or radial-grid pretabulation
for first-use/cold-miss cost, not another broad chain-level profile.

Timing medians above exclude cProfile instrumentation; cProfile is collected in
separate calls for call-stack attribution only.

## BD7 Hot-Loop Follow-Up

The BD7 non-LRS full-collision endpoint pass found that changing only
`T_nu_e_MeV` rebuilt the AP6 radial provider and repeatedly paid the same
universal angular-geometry contraction cost.  The runtime fix keeps that
geometry table in a bounded internal cache keyed only by the validated PSTF
basis, angular weights, and direction vectors; radial energy grids, process
momenta, finite-mass electron baths, and source moments remain rebuilt or
looked up through their existing radial/source-factory keys.  The inner
per-geometry table cache is also bounded, so opt-in callable momentum-delta
models such as `radial_gaussian` cannot grow it without limit.

Focused CPU-JAX/Rodas5P smoke after the fix:

- same-state cold call: `1.632584 s`
- same-state warm call: `0.007366 s`
- `T_nu_e_MeV`-changed radial rebuild: `0.037700 s`
- radial-grid cache entries after the three calls: `27`
- internal geometry-cache entries after the three calls: `1`
- internal per-geometry table entries after the three calls: `4`

A short private FB70 non-LRS full-collision span probe to `N_span_end=0.4`
then completed with `passed=true`, `violations=[]`,
`wall_seconds_total=15.338262253906578`, `source_evaluations_total=110`,
`frozen_source_jax_jacobian_evaluations_total=13`, and
`best_T_final_MeV=0.5419225510211213`.  This is still hot-endpoint evidence,
not full-BBN endpoint support below `0.01 MeV`.

Scope boundary: this is diagnostic CPU-JAX/Rodas5P hot-path evidence only.  It
does not promote public dispatch, production SMC validation, QKE support,
continuous AP65 collision evaluation inside the JAX RHS, or publication-ready
all-freedom full-BBN support.
