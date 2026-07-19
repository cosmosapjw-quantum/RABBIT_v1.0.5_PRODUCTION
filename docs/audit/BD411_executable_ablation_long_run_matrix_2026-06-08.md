# BD411 Executable Ablation And Long-Run Matrix

Date: 2026-06-08

Scope: augmented Type-I PSTF no-QKE AP65/BBN solver, CPU-JAX with in-tree
Rodas5P/AP65. QKE, q9/q10, public-production claims, and new runtime gates are
out of scope.

This document records executable ablations run at current head
`b07eed3 BD407: audit phase2 controller and payload reuse`. It is evidence
generation only. It does not validate the solver and does not promote any
optimization default-on.

## Preconditions

Mandatory pre-read was checked before running:

- `AGENTS.md`
- `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
- `bbn_codex_anti_drift_cost_effective_policy.md`

Dirty worktree before this document: untracked external reports and previous
audit docs were present. They were not modified or reverted.

Validation smoke before ablations:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_staged_bbn.py \
  tests/test_bridge_vs_table_dq_harness.py \
  tests/test_payload_reuse_parity_signoff.py \
  tests/test_summarize_perf_artifacts.py --tb=short
```

Result: `35 passed in 9.88s`.

## Matrix Boundary

The executable matrix was split into:

1. Cheap/medium harnesses:
   - bridge-vs-table dQ harness at q4/q8/q16;
   - staged standard-phase2 prototype to `N=2.75`.
2. q4 activation OFAT ablations:
   - payload reuse policy;
   - rejection ceiling policy;
   - phase-2 displacement guard;
   - phase-2 pair controller;
   - phase-2 Newton Jacobian refresh.
3. endpoint long runs:
   - default/current cold endpoint pairwise LRS/non-LRS run, which artifact
     controls identify as `phase2_network_newton_jacobian_refresh_policy =
     every_iteration`;
   - q4-aligned cold endpoint pairwise LRS/non-LRS run with explicit
     `--phase2-network-newton-jacobian-refresh-policy periodic
     --phase2-network-newton-jacobian-refresh-interval 4`.

Not run: the full Cartesian product of all ablation axes to cold endpoint. It is
technically executable but not tractable as a single cost-effective evidence
step, and several axes are diagnostic-only or already q4-falsified. Those
skips are listed explicitly below; they should not be read as pass/fail.

## Artifact Root

All new run outputs are under:

```text
diagnostic_outputs/bd411_all_executable_ablation_matrix/
```

Important subdirectories:

- `bridge/`
- `staged/`
- `q4_activation/`
- `cold_endpoint_best_current/`
- `cold_endpoint_periodic_reuse/`

Each q4/cold run directory contains command, stdout/stderr, `/usr/bin/time -v`,
exit code, perf summary, and component-wall checker output.

## Bridge-vs-Table Harness

Command output:

```text
diagnostic_outputs/bd411_all_executable_ablation_matrix/bridge/bridge_vs_table_dq_q4_q8_q16.json
diagnostic_outputs/bd411_all_executable_ablation_matrix/bridge/time_v.txt
```

`/usr/bin/time -v`: elapsed `16:07.19`, max RSS `3,555,180 KB`, exit `0`.

Top-level result:

- `max_closed_effective_abs_excess_fraction = 0.0`
- solver-consumed path:
  `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source->collision_temperature_source_policy->coupled_3T_rhs_from_collision_moments`
- source composition:
  `radial_standard_3t_dQ_plus_angular_dA`

Interpretation:

- The raw standalone radial moment is intentionally preserved and is not solver
  evidence. Its excess is large at q8/q16, and even q4 has large native excess.
- The `standard_3t_plasma` closure arm exactly closes against the table in the
  harness at q4/q8/q16.
- Therefore the bridge result supports the current distinction between raw
  moment evidence and solver-consumed closed-source evidence. It does not
  explain the high endpoint `N_eff_3T`.

Representative rows:

| q | normalization | native/paired excess | solver-consumed dT excess | closed effective excess |
|---:|---|---:|---:|---:|
| 4 | raw | `nue=-10.86`, `nux=-11.68` | `dT_nue=5.10e-05`, `dT_nux=5.83e-06` | n/a |
| 4 | standard_3t_plasma | paired raw carries the raw excess | `0.0` | `0.0` |
| 8 | raw | `nue=1.0466e4`, `nux=1.0897e4` | `dT_gamma=2.94e-02`, `dT_nue=-8.27e-02`, `dT_nux=-1.84e-02` | n/a |
| 8 | standard_3t_plasma | paired raw carries the raw excess | `0.0` | `0.0` |
| 16 | raw | `nue=1.2494e4`, `nux=1.3396e4` | `dT_gamma=2.98e-02`, `dT_nue=-8.31e-02`, `dT_nux=-1.90e-02` | n/a |
| 16 | standard_3t_plasma | paired raw carries the raw excess | `0.0` | `0.0` |

## Staged Prototype Harness

Command output:

```text
diagnostic_outputs/bd411_all_executable_ablation_matrix/staged/staged_standard_phase2_N2p75.json
diagnostic_outputs/bd411_all_executable_ablation_matrix/staged/time_v.txt
```

`/usr/bin/time -v`: elapsed `0:03.90`, max RSS `83,304 KB`, exit `0`.

Result:

- `network_success = true`
- `network_nfev = 20539`
- solver message: reached integration interval end.

Interpretation:

- The staged prototype remains executable and cheap.
- It is not AP65 endpoint evidence and not a replacement for the AP65 span
  ladder.
- The high `network_nfev` means the staged route still has conditioning/startup
  cost to understand before treating it as a production path.

## q4 Activation OFAT Ablations

Base command used the BD299/BD301 q4 replay case:

```text
diagnostic_outputs/bd301_detailed_profile_q4/bd299_q4_activation_probe_replay_case.json
```

Common options included:

- `--weak-correction-level 0`
- `--enabled-freedoms weak_rate_corrections,neutrino_collision_terms`
- `--phase2-network-background-policy auto_dynamic_effective_midpoint`
- `--phase2-network-newton-initial-guess-policy ab2_rhs_predictor`
- `--chain-restart-handoff`
- `--chain-h-max-policy first_rejection_or_recovered_h_ceiling`
- `--chain-max-steps-policy recovered_max_steps_floor`
- `--progress-jsonl`

Every q4 run ended with AP65 process exit `1` because the bounded q4 replay is
not a full endpoint/promoted ladder. Each produced parseable row artifacts,
`perf_summary.json`, and `component_wall_check_exit_code=0`.

| case | elapsed | max RSS KB | span wall s | payload s | phase2 s | host jac s | linear s | residual s | terminal `N_eff_3T` | checker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_reuse_periodic` | 8:44.63 | 3,400,104 | 506.20 | 262.45 | 187.90 | 14.72 | 0.43 | 40.69 | 3.319663678 | 0 |
| `payload_auto_small_collision_reuse` | 8:54.09 | 3,535,040 | 515.91 | 277.49 | 186.48 | 14.61 | 0.45 | 36.88 | 3.319655170 | 0 |
| `payload_current_state` | 8:57.21 | 3,542,340 | 519.03 | 278.88 | 187.93 | 14.78 | 0.46 | 36.98 | 3.319655170 | 0 |
| `payload_state_tolerance_reuse` | 8:57.91 | 3,534,592 | 519.02 | 275.64 | 186.79 | 14.65 | 0.45 | 41.49 | 3.319655175 | 0 |
| `payload_step_base_reuse` | 4:39.79 | 1,191,676 | 266.99 | 35.54 | 186.53 | 14.70 | 0.35 | 29.88 | 3.319751818 | 0 |
| `phase2_ab2_trust_displacement` | 8:48.72 | 3,400,924 | 509.52 | 272.58 | 178.26 | 15.20 | 0.50 | 42.98 | 3.319663678 | 0 |
| `phase2_jacobian_chord_once` | 6:40.35 | 3,409,200 | 381.76 | 254.75 | 61.57 | 17.51 | 0.49 | 47.44 | status mix; drifted | 0 |
| `phase2_jacobian_every_iteration` | 9:16.43 | 3,405,748 | 537.03 | 276.14 | 201.29 | 15.28 | 0.52 | 43.81 | 3.319663678 | 0 |
| `phase2_pair_coarse_only` | 5:44.07 | 3,342,996 | 325.57 | 263.63 | 8.24 | 14.01 | 0.46 | 39.24 | 3.319663686 | 0 |
| `rejection_ceiling_off` | 12:09.80 | 3,434,076 | 710.95 | 272.07 | 378.97 | 15.55 | 0.50 | 43.87 | status mix; drifted | 0 |

q4 conclusions:

1. Payload policy matters, but only `step_base_reuse` produces a large q4 wall
   reduction. It also changes q4 terminal `N_eff_3T` by about `8.8e-05` against
   the q4 periodic baseline, so it is not default-on evidence.
2. `current_state`, `auto_small_collision_reuse`, and `state_tolerance_reuse`
   are slower than the q4 periodic baseline in this replay.
3. Disabling the rejection ceiling is bad: q4 wall grows to `710.95s` and phase2
   grows to `378.97s`.
4. `every_iteration` is slower than q4 `periodic` in the bounded q4 replay.
5. `chord_once` and `coarse_only` are diagnostic-only in this matrix. They show
   what could be saved by removing work, but they are not safe physical
   optimizations without a much stronger parity/error argument.

## Cold Endpoint Long Run A: Default/Current Refresh Policy

Directory:

```text
diagnostic_outputs/bd411_all_executable_ablation_matrix/cold_endpoint_best_current/
```

This command did not explicitly set the phase-2 Newton Jacobian refresh policy.
The final artifact records controls with:

```text
phase2_network_newton_jacobian_refresh_policy = every_iteration
phase2_network_newton_jacobian_refresh_interval = 1
```

Command status:

- AP65 exit code: `0`
- summarizer exit code: `0`
- component wall checker exit code: `0`
- `/usr/bin/time -v`: elapsed `35:06.85`, max RSS `15,913,780 KB`

Top-level artifact:

- `physical_full_bbn_span_ready = true`
- `summary.execution_passed = true`
- `summary.failed_or_exception_rows = 0`
- top-level `passed = false`
- `promotion_decision = not_promoted`
- blocker: `controlled_flrw_lrs_nonlrs_default_on_blocker_status =
  blocked_neff_floor_failed`

Endpoint parity/floor:

| metric | LRS | non-LRS | abs delta |
|---|---:|---:|---:|
| `T_final_MeV` | 0.0091413861 | 0.0091413625 | 2.36e-08 |
| `N_eff_3T` | 3.1148708234 | 3.1149383651 | 6.75e-05 |
| `Yp` | 0.1636771949 | 0.1636887163 | 1.15e-05 |
| `D/H` | 2.1273877937e-05 | 2.1274533271e-05 | 6.55e-10 |
| `Sigma_H` | 3.10e-27 | 5.31e-31 | 3.10e-27 |

Parity passed at tolerance `5e-4`; `N_eff_3T` floor/band failed because both
endpoint rows are above the current `3.00..3.06` band. The final non-LRS shear
is tiny (`5.31e-31`), and the LRS final shear is also tiny (`3.10e-27`), but the
model-level `N_eff_3T≈3.115` is the current endpoint physics blocker.

Component wall attribution:

| component | wall s | fraction of row wall |
|---|---:|---:|
| total row wall | 2064.45 | 100.0% |
| phase2 corrector | 1043.57 | 50.6% |
| payload | 618.76 | 30.0% |
| host Jacobian/JVP | 113.61 | 5.5% |
| outer linear system | 2.65 | 0.1% |
| JAX compile/runtime | 0.00 | unavailable/not separately measured |
| residual unattributed | 285.86 | 13.8% |

Overlapping subtimers, not to be summed into exclusive attribution:

- `phase2_step_attempt = 969.38s`
- `phase2_coarse_step_attempt = 327.65s`
- `phase2_refined_step_attempt = 641.73s`
- `phase2_step_attempt_bookkeeping = 391.23s`
- `phase2_newton = 186.04s`
- `phase2_newton_jacobian = 36.20s`
- `phase2_newton_residual = 16.05s`
- `phase2_ab2_rhs_predictor = 143.45s`
- `phase2_ab2_residual_guard = 108.41s`
- `rejected_step_replay = 12.14s`

Row wall/checkpoint summary:

| freedom | row | span | `T_gamma` | `N_eff_3T` | `Yp` | wall s | builds | rejected |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| LRS | 0 | 0.0-1.0 | 0.308539 | 9.578401 | 1e-30 | 113.00 | 1595 | 0 |
| LRS | 1 | 1.0-2.0 | 0.132042 | 5.232642 | 1e-30 | 106.96 | 1581 | 0 |
| LRS | 2 | 2.0-2.25 | 0.107762 | 4.339347 | 1e-30 | 27.34 | 397 | 0 |
| LRS | 3 | 2.25-2.5 | 0.087357 | 3.696565 | 1e-30 | 26.33 | 383 | 0 |
| LRS | 4 | 2.5-2.75 | 0.069888 | 3.319664 | 0.123243 | 214.82 | 524 | 1 |
| LRS | 5 | 2.75-3.0 | 0.055098 | 3.161257 | 0.163537 | 300.25 | 491 | 1 |
| LRS | 6 | 3.0-4.0 | 0.020345 | 3.114871 | 0.163676 | 152.04 | 2320 | 0 |
| LRS | 7 | 4.0-4.8 | 0.009141 | 3.114871 | 0.163677 | 144.87 | 3019 | 0 |
| non-LRS | 0 | 0.0-1.0 | 0.308539 | 9.578487 | 1e-30 | 125.78 | 1167 | 0 |
| non-LRS | 1 | 1.0-2.0 | 0.132042 | 5.232740 | 1e-30 | 40.81 | 202 | 0 |
| non-LRS | 2 | 2.0-2.25 | 0.107762 | 4.339431 | 1e-30 | 10.28 | 52 | 0 |
| non-LRS | 3 | 2.25-2.5 | 0.087357 | 3.696640 | 1e-30 | 10.20 | 52 | 0 |
| non-LRS | 4 | 2.5-2.75 | 0.069888 | 3.319733 | 0.123256 | 201.83 | 116 | 1 |
| non-LRS | 5 | 2.75-3.0 | 0.055098 | 3.161325 | 0.163548 | 303.06 | 194 | 1 |
| non-LRS | 6 | 3.0-4.0 | 0.020344 | 3.114938 | 0.163687 | 160.38 | 802 | 0 |
| non-LRS | 7 | 4.0-4.8 | 0.009141 | 3.114938 | 0.163689 | 126.51 | 643 | 0 |

## Cold Endpoint Long Run B: q4-Aligned Periodic Refresh

Directory:

```text
diagnostic_outputs/bd411_all_executable_ablation_matrix/cold_endpoint_periodic_reuse/
```

Additional flags:

```bash
--phase2-network-newton-jacobian-refresh-policy periodic
--phase2-network-newton-jacobian-refresh-interval 4
```

Command status:

- AP65 exit code: `0`
- summarizer exit code: `0`
- component wall checker exit code: `0`
- `/usr/bin/time -v`: elapsed `36:09.04`, max RSS `15,912,396 KB`

Top-level artifact:

- `physical_full_bbn_span_ready = true`
- `summary.execution_passed = true`
- `summary.failed_or_exception_rows = 0`
- top-level `passed = false`
- `promotion_decision = not_promoted`
- same default-on blocker: `blocked_neff_floor_failed`

Endpoint parity/floor matches the default/every-iteration run:

- LRS endpoint `N_eff_3T = 3.1148708234`
- non-LRS endpoint `N_eff_3T = 3.1149383651`
- parity delta `6.754e-05`, parity passed
- floor/band failed because endpoint `N_eff_3T` remains above `3.06`

Component wall attribution:

| component | wall s | fraction of row wall |
|---|---:|---:|
| total row wall | 2123.99 | 100.0% |
| phase2 corrector | 1062.46 | 50.0% |
| payload | 635.77 | 29.9% |
| host Jacobian/JVP | 118.30 | 5.6% |
| outer linear system | 3.09 | 0.1% |
| JAX compile/runtime | 0.00 | unavailable/not separately measured |
| residual unattributed | 304.36 | 14.3% |

Overlapping subtimers, not to be summed:

- `phase2_step_attempt = 982.85s`
- `phase2_coarse_step_attempt = 330.20s`
- `phase2_refined_step_attempt = 652.65s`
- `phase2_step_attempt_bookkeeping = 399.85s`
- `phase2_newton = 179.07s`
- `phase2_newton_jacobian = 15.87s`
- `phase2_newton_residual = 16.95s`
- `phase2_ab2_rhs_predictor = 148.22s`
- `phase2_ab2_residual_guard = 111.97s`
- `rejected_step_replay = 11.99s`

Endpoint refresh-policy conclusion:

- In q4, `periodic` was faster than `every_iteration`.
- At cold endpoint pairwise long-run scale, explicit `periodic` was slower:
  `2123.99s` vs `2064.45s`.
- `periodic` reduced the reported Newton-Jacobian subtimer
  (`15.87s` vs `36.20s`) but increased total phase2/payload/residual enough
  that it did not improve endpoint wall.
- Therefore q4-only phase2 Jacobian refresh conclusions do not generalize to
  cold endpoint and should not drive a default optimization.

## Cross-Run Findings

1. Full endpoint LRS/non-LRS parity now runs to `T_gamma<=0.01 MeV` under q4
   controls. This is real execution progress.
2. The default-on blocker is no longer endpoint reachability or LRS/non-LRS
   parity. It is the `N_eff_3T` floor/band failure:
   endpoint `N_eff_3T≈3.115` for both LRS and non-LRS.
3. The final non-LRS shear is machine-tiny (`~5e-31`), and LRS final shear is
   also tiny (`~3e-27`). This argues against final shear drift as the current
   endpoint blocker in these runs.
4. Long-run component attribution is stable: phase2 corrector is about half the
   wall, payload about 30%, residual about 14%, host Jacobian about 5-6%, dense
   outer linear solve about 0.1%.
5. Dense LU is still not the target at q4: `W/J = [62,62]` in all summarized
   rows.
6. Non-LRS geometry caused very large RSS growth to about `15.9 GB`, even though
   `W/J` stayed `[62,62]` and final shear was tiny. This is a real memory
   blocker separate from linear-solver size.
7. q4 `step_base_reuse` is the largest cheap wall reduction, but endpoint
   evidence for it is still missing. It remains an opt-in candidate, not a
   default.
8. q4 `coarse_only` and `chord_once` are diagnostic lower bounds, not physical
   optimizations.
9. The bridge closure ablation rules out one class of raw-vs-closed-source
   confusion but does not explain the endpoint `N_eff_3T≈3.115`.

## Skipped Or Incomplete Ablations

These are explicitly not completed by BD411:

- Full Cartesian endpoint runs over every q4 ablation axis. Reason: one
  endpoint pairwise run costs about 35-36 minutes and 16 GB RSS; several axes
  are diagnostic-only or q4-falsified. This should be scheduled as a smaller
  next matrix, not hidden as completed.
- q9/q10 runs. Reason: still outside current authorized scope and not needed to
  decide q4 endpoint blockers.
- QKE. Reason: out of scope.
- Public-production/readiness/manifest/hash/figure gates. Reason: forbidden
  drift for this line.
- Endpoint `step_base_reuse`. Reason: q4 suggests large wall benefit, but this
  still needs a targeted endpoint run before any performance claim.
- Endpoint `coarse_only` and `chord_once`. Reason: diagnostic-only modes that
  would need explicit non-promotion wording if run.

## Next Evidence Steps

Highest value next runs, in order:

1. Endpoint `step_base_reuse` LRS/non-LRS pairwise run with the same cold ladder
   and raw observables preserved. This directly tests the biggest q4 payload
   wall reduction.
2. Endpoint no-reuse/current-state contrast at current head, using the same
   cold ladder, if memory budget permits. This isolates payload reuse causality
   at endpoint scale.
3. Source/closure physics audit for endpoint `N_eff_3T≈3.115`, including:
   dQ normalization, `C_RATE`, standard 3T tail mapping, and whether the
   no-QKE classical Boltzmann target expected by the current closure is really
   the `3.00..3.06` band encoded in the floor tripwire.
4. Non-LRS memory profile: why a [62,62] solve with machine-tiny shear reaches
   ~16 GB RSS.
5. Phase2 step-attempt bookkeeping/predictor/residual-guard audit, because
   endpoint phase2 is now the largest exclusive bucket.

## Anti-Drift Self-Audit

- `real_blocker_moved`: yes, by evidence. Full cold endpoint pairwise runs now
  exist and show parity pass but `N_eff_3T` floor/band failure.
- `gate_removed_or_consolidated`: no code/gate changes were made.
- `raw_state_preserved`: yes. Artifacts, progress logs, stderr, checkpoints,
  non-promoted status, and raw high `N_eff_3T` are preserved.
- `verification`: pytest smoke passed; q4 summaries/checks passed; two endpoint
  AP65 runs exited 0; endpoint summaries/checks exited 0.
- `remaining_blocker`: endpoint `N_eff_3T≈3.115` model/closure target mismatch,
  endpoint phase2 wall, payload reuse endpoint ablation, and non-LRS memory.

## BD412 Follow-Up: Endpoint Step-Base And Profiling Toggle Ablations

Date: 2026-06-08.

Artifact roots:

- `diagnostic_outputs/bd412_activation_timing_ablation_matrix/cold_endpoint_step_base_reuse/`
- `diagnostic_outputs/bd412_activation_timing_ablation_matrix/q4_followup/`
- `diagnostic_outputs/bd412_activation_timing_ablation_matrix/q4_case_overrides/`
- `diagnostic_outputs/bd412_activation_timing_ablation_matrix/q4_case_override_runs/`
- `diagnostic_outputs/bd412_activation_timing_ablation_matrix/q4_source_split_valid/`

### Endpoint Step-Base Reuse Long Run

BD412 ran the missing endpoint `step_base_reuse` LRS/non-LRS pair using the same
cold endpoint ladder style as BD411. AP65 exited 0; summarizer exited 0; the
component attribution checker exited 0. This is an execution/performance
artifact only, not a solver-validation or default-on promotion.

| endpoint pair | stage payload policy | elapsed | max RSS KB | component total s | payload s | phase2 s | residual s | parity | floor |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BD411 best current | `thermo_state_tolerance_reuse` | 35:06.85 | 15,913,780 | 2064.45 | 618.76 | 1043.57 | 285.86 | pass | fail |
| BD412 step-base | `step_base_reuse` | 28:34.39 | 6,026,028 | 1685.42 | 228.98 | 1083.38 | 253.68 | pass | fail |

Measured change from BD411 best-current endpoint:

- elapsed wall: `2106.85s -> 1714.39s`, about `-18.6%`
- max RSS: `15,913,780 KB -> 6,026,028 KB`, about `-62.1%`
- component total wall: `2064.45s -> 1685.42s`, about `-18.4%`
- payload wall: `618.76s -> 228.98s`, about `-63.0%`
- phase2 wall: `1043.57s -> 1083.38s`, about `+3.8%`
- residual unattributed wall: `285.86s -> 253.68s`, about `-11.3%`

Endpoint observables remain blocked by the same floor/band issue:

- BD412 LRS `N_eff_3T = 3.114956756570601`
- BD412 non-LRS `N_eff_3T = 3.114956756570601`
- `N_eff_3T` parity delta `0.0`, parity passed
- `Yp` parity delta `1.1412241613328877e-05`
- floor/band failed because endpoint `N_eff_3T` is still above the encoded band

Interpretation:

- `step_base_reuse` is now the strongest measured endpoint performance lever.
- It moved wall and memory substantially, but it did not move the physics
  blocker.
- It must remain opt-in until PR-B parity/floor policy is resolved and raw-state
  preservation has been reviewed.
- After this run, endpoint wall is no longer primarily payload. Phase2 is the
  dominant exclusive component.

### Replay Case Override Hazard

The first BD412 q4 follow-up tried to change `rhs_trace_policy` and
`collision_source_component_policy` with CLI flags while reusing
`bd299_q4_activation_probe_replay_case.json`. The resulting artifact rows still
reported:

- `rhs_trace_policy = full`
- `collision_source_component_policy = full`

The replay case contains per-case values for both fields. In this path, those
per-case values override the CLI fields. Therefore the initial CLI-only
`source_zero`, `dQ_only`, `dA_only`, and `rhs_trace_boundary` attempts are not
valid physics/profiling ablations. They are preserved as false-ablation evidence.

Practical rule for future profiling:

- If a control is present in the replay case, change the case JSON or explicitly
  change override semantics before treating the run as an ablation.
- Always read the artifact row control fields before comparing timings.

### q4 Profiling Toggle Timing

Valid q4 baseline for this section is BD411 `baseline_reuse_periodic`.

| run | actual artifact controls | valid as ablation | elapsed | max RSS KB | component total s | payload s | phase2 s | residual s | final row |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| BD411 q4 baseline | `rhs_trace=full`, `source=full` | yes | 8:44.63 | 3,400,104 | 506.20 | 262.45 | 187.90 | 40.69 | pass |
| BD412 CLI `source_zero` | artifact still `source=full` | no | 9:16.20 | 3,403,484 | 536.56 | 281.94 | 193.50 | 45.04 | pass |
| BD412 CLI `rhs_trace_boundary` | artifact still `rhs_trace=full` | no | 9:10.38 | 3,403,688 | 531.35 | 276.77 | 194.34 | 44.27 | pass |
| BD412 `progress_jsonl_off` | same physics controls, no progress stream | yes | 9:09.52 | 3,402,552 | 530.59 | 276.64 | 194.38 | 43.74 | pass |
| BD412 valid `rhs_trace_boundary` | `rhs_trace=boundary`, `source=full` | yes | 8:35.97 | 3,390,544 | 498.35 | 268.21 | 189.81 | 24.87 | pass |

Timing interpretation:

- `rhs_trace_policy=boundary` gives a small but real q4 reduction in this run:
  elapsed `524.63s -> 515.97s` (`-1.6%`) and component total
  `506.20s -> 498.35s` (`-1.6%`).
- The main visible improvement is residual attribution, `40.69s -> 24.87s`.
  Payload and phase2 do not decrease meaningfully, so this is not a major
  runtime optimization.
- Turning `--progress-jsonl` off did not speed up this workload. The run was
  slower than baseline (`8:44.63 -> 9:09.52`), so progress streaming is not a
  priority target. Keep it enabled for long-run observability unless stderr size
  itself becomes a problem.
- Routine performance profiling should use `rhs_trace=boundary` when per-call
  provenance hashes are not required. Full trace should be reserved for
  provenance/debug runs.

### q4 Collision Source Component Split

After the replay-case override issue was found, BD412 created explicit case JSON
files for `zero`, `dQ_only`, and `dA_only` source policies and reran q4.

| run | actual source policy | elapsed | max RSS KB | component total s | payload s | phase2 s | host JVP s | residual s | row walls s | payload builds | final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline | `full` | 8:44.63 | 3,400,104 | 506.20 | 262.45 | 187.90 | 14.72 | 40.69 | 118.66, 111.65, 56.40, 219.49 | 1595, 1581, 778, 607 | pass |
| source zero | `zero` | 9:32.22 | 1,908,516 | 557.15 | 99.19 | 406.81 | 9.29 | 41.40 | 94.41, 27.93, 14.75, 420.06 | 1167, 202, 102, 158 | fail |
| dQ only | `dQ_only` | 5:49.17 | 1,851,740 | 334.12 | 96.95 | 188.31 | 9.77 | 38.67 | 96.91, 28.67, 14.19, 194.36 | 1167, 202, 102, 116 | pass |
| dA only | `dA_only` | 8:09.45 | 3,221,628 | 472.45 | 233.59 | 185.57 | 13.58 | 39.32 | 116.22, 107.08, 48.56, 200.59 | 1601, 1565, 681, 261 | pass |

Final q4 observables:

| run | `T_gamma` MeV | `N_eff_3T` | `Yp` | `D/H` | `Sigma_H` | final status |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.0698879293 | 3.3196636782 | 0.1232433502 | 0.00253890999 | 3.420e-26 | pass |
| dQ only | 0.0698877628 | 3.3197329515 | 0.1232558662 | 0.00253895836 | 5.009e-31 | pass |
| dA only | 0.0699117419 | 3.3099263922 | 0.1228393967 | 0.00254531892 | 4.521e-26 | pass |
| source zero | unavailable in final row | progress tail reached `N_eff_3T=3.4905430552` at `T_gamma=0.078949922` | progress tail `7.63157e-05` | progress tail `0.000562781` | 4.960e-31 | fail: phase2 wall budget |

The valid source-zero final row failed with:

`continuous AP65 phase-2 conservative extent corrector did not provide an accepted update: phase2 BE/Newton solve exceeded wall_time_budget_seconds.`

Source split interpretation:

- Removing angular `dA` while keeping thermal `dQ` (`dQ_only`) is the major q4
  speed lever in this diagnostic matrix: component wall `506.20s -> 334.12s`
  (`-34.0%`), payload `262.45s -> 96.95s` (`-63.1%`), max RSS
  `3.40GB -> 1.85GB`.
- Keeping angular `dA` while removing thermal `dQ` (`dA_only`) remains much
  closer to full cost: component wall `472.45s`, payload `233.59s`, RSS
  `3.22GB`.
- Therefore the expensive front-half payload/error-control behavior is tied
  primarily to the angular distribution source path, not the thermal dQ path.
- Turning off all collision source components is not a valid performance
  direction. It reduces payload but destabilizes the activation row into a
  phase2 wall-budget failure.
- `dQ_only` is a diagnostic ablation, not a proposed physical default. It
  suppresses angular feedback and changes shear/A-mode physics, even though its
  q4 scalar endpoint observables happen to stay close to baseline.

### Updated Ablation Priorities

Based on BD412, the next useful ablations are:

1. Angular-source deflation that preserves physics: cache or reduce the
   expensive `dA` path rather than disabling it. Candidate probes: angular
   moment response reuse, active-column pruning, tolerance-gated angular source
   refresh, and N_mu/N_phi convergence for nonzero shear.
2. Endpoint `step_base_reuse` review and guarded opt-in hardening. It is the
   largest measured long-run wall/RSS improvement but still does not solve the
   `N_eff_3T` floor/band blocker.
3. Endpoint source split is not recommended yet. q4 source-zero already shows
   a phase2 wall-budget failure, and dQ/dA splits are diagnostic physics
   ablations rather than valid solver configurations.
4. Use `rhs_trace=boundary` for routine timing runs, but do not spend a PR on
   it as an optimization. It is a small profiling hygiene improvement.
5. Do not disable `--progress-jsonl` for speed. It did not improve q4 runtime
   and it removes useful live evidence.

### BD412 Anti-Drift Self-Audit

- `real_blocker_moved`: yes. A real endpoint long run shows `step_base_reuse`
  reduces endpoint wall/RSS, and valid q4 source split identifies angular `dA`
  as the dominant source-side cost path.
- `gate_removed_or_consolidated`: no new gate was added; no code gate changed.
- `raw_state_preserved`: yes. Failed source-zero row, partial progress physics,
  missing final observables, stderr, and checker outputs are preserved.
- `verification`: AP65/summarizer/checker statuses were recorded for every run;
  bounded q4 exit 1 is expected for nonendpoint or failed rows, while the
  summarizer/checker exits were 0 for valid artifact attribution.
- `remaining_blocker`: endpoint `N_eff_3T≈3.115` floor/band mismatch, endpoint
  phase2 dominance after payload reduction, angular `dA` source cost, and
  replay-case CLI override hazards.
