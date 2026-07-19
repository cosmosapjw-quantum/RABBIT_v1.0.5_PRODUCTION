# PR-AP4 Combined Full-Span Physical Preview Preset

Date: 2026-05-17

## Scope

This note records the AP4/AP65 combined angular+`pstf_radial` full-span
`physical_preview` preset.  The preset is an opt-in, SciPy-first diagnostic
path for longer finite 3T/network evolution after the combined source, AP6
radial closure, shared radial-grid cache, charge-neutral electron-bath, and
scalar-QED routing surfaces had already landed.

## Implementation

- `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`
  accepts `--preset physical_preview`.
- The preset runs spans `(0, 1e-6)`, `(0, 1e-4)`, and `(0, 1e-3)`.
- The source policy is `frozen_initial_state`; the solver is `Radau`.
- The preset uses `max_pstf_radial_source_evaluations=64` and
  `max_nfev=200000`.
- Dry-run JSON records the resolved solver method and separates
  `routine_numeric_gate_spans=[(0,1e-6),(0,1e-4)]` from
  `isolated_diagnostic_spans=[(0,1e-3)]` so the order-sensitive long row is not
  misread as routine pass/fail coverage.
- CLI overrides of preset-defining fields are relabeled `custom`, and direct
  spec construction rejects `span_ladder_preset="physical_preview"` unless the
  fixed spans, frozen source policy, `Radau` method, and budgets match.

## Numeric Evidence

A real isolated CLI run of the preset passed in `elapsed_s=29.111393796047196`.
The routine regression now keeps the frozen-source Radau numeric gate on the
stable `1e-6` and `1e-4` rows because the `1e-3` frozen row was later found to
be order-sensitive after source-refresh solves in the same process.  The
`1e-3` value below remains recorded isolated diagnostic evidence, not a routine
pass/fail stability claim.

| N_span | T_gamma_final | H_rate_s_final | Xn_final | nfev | pair energy residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1e-6` | `0.7999992142768074` | `0.4315478525579576` | `0.13000006723216642` | `22` | `9.571472303973594e-20` |
| `1e-4` | `0.7999214316323131` | `0.4314627401457326` | `0.1300112257070123` | `420` | `9.571472303973594e-20` |
| `1e-3` | `0.7992146754386976` | `0.4306897631857426` | `0.1927605271823484` | `10634` | `9.571472303973594e-20` |

## Boundaries

This does not promote the AP4/AP65 path to public production support.  The
collision source is frozen at the initial state, so this is not live-RHS
full-BBN evidence.  QKE, production SMC validation, promotion-tolerance
convergence, and public dispatch remain out of scope.

## Negative Evidence

- A frozen-source `Radau` probe at `N_span=1e-2` returned a non-success result
  with overflow warning and unusable terminal values.
- After a `piecewise_frozen` source-refresh solve in the same process, the
  frozen-source `Radau` `N_span=1e-3` row can fail with non-finite network
  values.  Long routine evidence therefore moved to the nonuniform
  `piecewise_frozen` `N_span=1e-3` gate.
- A CPU live-RHS probe at `N_span=1e-6` was timeout-level in smoke settings.
