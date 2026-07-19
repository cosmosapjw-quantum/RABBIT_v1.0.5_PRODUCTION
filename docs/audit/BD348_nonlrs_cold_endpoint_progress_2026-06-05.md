# BD348 Non-LRS Cold Endpoint Progress

Date: 2026-06-05

Scope: private augmented Type-I PSTF no-QKE AP65 q4 diagnostic run on CPU-JAX/Rodas5P.
QKE remains out of scope.  This is not public-production or publication-ready
validation.

## Blocker Moved

BD346 left the cold FLRW-limit non-LRS case blocked in the activation span
`N=[2,3]`: the pairwise cold run completed only case1 rows `[0,1]` and `[1,2]`
before SIGTERM at 41:56.  BD348 reran the non-LRS case as a single-case cold
endpoint attempt with span-row checkpoints and split the blocked activation row
into four subspans.

The run reached the cold endpoint:

- final span: `[4.0, 4.8]`
- `T_gamma_MeV=0.009141354331964295`
- `Sigma_H=5.30703811115845e-31`
- `N_eff_3T=3.1149609726834173`
- `Yp=0.16368860367173316`
- `/usr/bin/time -v` elapsed: `23:19.94`
- max RSS: `9246264 KB`
- final JSON exit status: `0`

Top-level `passed=false` is expected for this artifact because it is a
single-case non-LRS endpoint attempt, not the completed LRS/non-LRS pairwise
parity artifact.  It therefore moves `NONLRS_COLD_BLOCKED` but does not close
PR-B parity or default-on optimization blockers.

## Exact Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint/bd348_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint/bd348_nonlrs_cold_checkpoint_endpoint.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd301_detailed_profile_q4/bd299_q4_activation_probe_replay_case.json \
  --weak-correction-level 0 \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --N-span-end-ladder 1.0,2.0,2.25,2.5,2.75,3.0,4.0,4.8 \
  --max-steps 4000 \
  --wall-time-budget-seconds 7200 \
  --phase2-network-background-policy auto_dynamic_effective_midpoint \
  --phase2-network-newton-initial-guess-policy ab2_rhs_predictor \
  --chain-restart-handoff \
  --chain-h-max-policy first_rejection_or_recovered_h_ceiling \
  --chain-max-steps-policy recovered_max_steps_floor \
  --stop-at-T-gamma-MeV 0.01 \
  --stage-collision-payload-policy thermo_state_tolerance_reuse \
  --source-refresh-stage-collision-payload-reuse-state-rtol 1e-2 \
  --source-refresh-stage-collision-payload-reuse-state-atol 0.0 \
  --span-row-checkpoint-dir diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint/jax_cache \
  --progress-jsonl
```

## Row Table

| row | span | wall s | builds | source evals | rejected | dominant error blocks |
|---:|---|---:|---:|---:|---:|---|
| 0 | `[0.0,1.0]` | 119.304 | 654 | 1601 | 0 | geometry_thermo: 200 |
| 1 | `[1.0,2.0]` | 66.272 | 202 | 1609 | 0 | geometry_thermo: 201 |
| 2 | `[2.0,2.25]` | 17.147 | 52 | 409 | 0 | geometry_thermo: 51 |
| 3 | `[2.25,2.5]` | 17.101 | 52 | 409 | 0 | geometry_thermo: 51 |
| 4 | `[2.5,2.75]` | 374.504 | 120 | 1191 | 41 | geometry_thermo: 112; phase2_conservative_extent_corrector: 41 |
| 5 | `[2.75,3.0]` | 324.369 | 135 | 1206 | 21 | geometry_thermo: 132; phase2_conservative_extent_corrector: 21 |
| 6 | `[3.0,4.0]` | 256.222 | 802 | 6402 | 0 | geometry_thermo: 800 |
| 7 | `[4.0,4.8]` | 203.932 | 643 | 5130 | 0 | geometry_thermo: 641 |

Activation split result:

- BD346 blocked at case1 `[2,3]` without a completed row.
- BD348 completed `[2,3]` as four checkpointed rows in `733.1205865918892 s`.
- The heavy activation pocket is localized to `[2.5,3.0]`, especially
  phase2-corrector dominated rejections: 62 total rejected attempts across rows
  4 and 5.

## Component Wall Attribution

`scripts/summarize_perf_artifacts.py diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint`
and `scripts/check_component_wall_attribution.py diagnostic_outputs/bd348_nonlrs_cold_checkpoint_endpoint`
were run after completion.

Attribution summary:

- total span wall: `1378.8500986339059 s`
- attributed wall: `947.0529759676429 s`
- residual unattributed wall: `431.797122666263 s` (`31.315740782414675%`)
- phase2 corrector: `682.854213352548 s`
- payload: `189.71787243813742 s`
- host Jacobian/JVP: `72.84095573495142 s`
- outer linear system: `1.6399344420060515 s`
- JAX compile/runtime: unavailable as separate values, with explicit reason
  recorded in every span row.
- component attribution checker: `PASS component wall attribution`

## Interpretation

IMPLEMENTED and VALIDATED in this artifact:

- The non-LRS cold case can reach `T_gamma <= 0.01 MeV` at q4 when the blocked
  activation span is split and checkpointed.
- The dominant performance pocket is not all of `[2,3]`; it is concentrated in
  `[2.5,3.0]`.
- The rejected attempts in `[2.5,3.0]` are phase2-corrector dominated, matching
  the BD346 hypothesis that the non-LRS cold blocker is a phase2/rejected-replay
  performance problem rather than an anisotropic physics failure.

PARTIAL / not yet closed:

- PR-B LRS/non-LRS parity is still open.  BD348 is non-LRS single-case evidence,
  not a matched pairwise artifact.
- q4 `Yp` remains under-resolved and must not be used as physics-grade BBN
  evidence.
- `N_eff_3T` remains a 3T diagnostic proxy, not a classical no-QKE final result.

Next PR target:

Run the same split ladder as a matched LRS/non-LRS pairwise cold artifact.  If
that completes, use it to close the root-level pairwise final-artifact false
green and evaluate the cold `N_eff_3T >= 3.0` floor.  If it fails, resume from
the latest span checkpoint and isolate `[2.5,3.0]` phase2/rejected-replay
deflation.
