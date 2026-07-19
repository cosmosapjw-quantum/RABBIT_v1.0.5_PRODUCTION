# BD425 PR-B Current-Head Parity Evidence

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 cold-endpoint
evidence.  QKE remains out of scope.  This note does not claim public
production, publication readiness, Bianchi validity, or default-on optimization.

## Question

Does the current head, after PR-N1 thermal neutrino start and PR-N3
distribution diagnostics, pass the PR-B zero-shear FLRW LRS/non-LRS cold
endpoint `N_eff_3T` floor and parity blocker for the collision-off q4 case?

## Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd425_pr_b_current_head_parity_q4/bd425_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd425_pr_b_current_head_parity_q4/bd425_q4_thermal_start_lrs_nonlrs_collision_off_parity.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd419_phase2_policy_endpoint_ablation/bd419_q4_lrs_endpoint_thermal_collision_off_case.json \
  --freedom-composition \
  --freedom-composition-cases-json '[["weak_rate_corrections"],["weak_rate_corrections","non_lrs_geometry"]]' \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry \
  --weak-correction-level 0 \
  --sigma-plus0 0.0 \
  --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun \
  --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --stop-at-T-gamma-MeV 0.01 \
  --progress-jsonl
```

Companion commands:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd425_pr_b_current_head_parity_q4 \
  > diagnostic_outputs/bd425_pr_b_current_head_parity_q4/bd425_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd425_pr_b_current_head_parity_q4 \
  > diagnostic_outputs/bd425_pr_b_current_head_parity_q4/bd425_component_wall_check.txt
```

## Result

`/usr/bin/time -v` exit status was `0`.  Elapsed wall was `21:58.36`;
maximum RSS was `947720 KB`.  The artifact's top-level `passed` field is
`false` because this one-resolution composition artifact is not a
resolution-convergence or publication-readiness artifact.  The executable
span rows passed, and the PR-B parity status under `.summary` is the blocker
evidence:

```text
controlled_flrw_lrs_nonlrs_default_on_blocker_status =
  passed_pr_b_neff_floor_and_lrs_nonlrs_parity
controlled_flrw_lrs_nonlrs_default_on_blocker_passed = true
controlled_flrw_lrs_nonlrs_neff_floor_passed = true
controlled_flrw_lrs_nonlrs_neff_parity_passed = true
controlled_flrw_lrs_nonlrs_neff_delta = 0.0
controlled_flrw_lrs_nonlrs_controls_identical_except_geometry = true
```

Endpoint rows:

| freedom key | T_final_MeV | N_eff_3T | Yp | D/H | Sigma_H |
|---|---:|---:|---:|---:|---:|
| `weak_rate_corrections` | 0.009139629979893203 | 3.034764564535386 | 0.24202352662323537 | 2.4930396074188933e-05 | 3.328675699807655e-31 |
| `weak_rate_corrections+non_lrs_geometry` | 0.009139629979893203 | 3.034764564535386 | 0.24202352662323537 | 2.4930396074188933e-05 | 3.328675699807655e-31 |

Both endpoint rows report `N_eff_3T_floor_passed=true` and
`N_eff_3T_floor_cold_endpoint_applicable=true`.

The `bd425_progress.jsonl` sidecar begins with the JAX ROCm plugin warning
emitted on this CPU-only host before the JSON progress events.  The JSON event
rows are preserved after that warning prelude, but the sidecar is not a
line-1-clean JSONL file.

## Performance Context

The real run artifact component summary reports:

| component | wall seconds |
|---|---:|
| total span wall | 1300.8046574838227 |
| phase2 corrector | 1052.420285879518 |
| host Jacobian | 68.24282051745104 |
| outer linear system | 4.483260634937324 |
| AP65 collision payload | 0.0 |
| residual unattributed | 175.6582904519164 |

`check_component_wall_attribution.py` output: `PASS component wall attribution`.
The non-LRS zero-shear branch matched the LRS branch at the displayed precision;
activation rows also had matching wall cost:

| span | LRS wall s | non-LRS zero-shear wall s |
|---|---:|---:|
| `[2.5, 2.75]` | 212.63984029297717 | 212.34785339795053 |
| `[2.75, 3.0]` | 241.41003618197283 | 241.0487851849757 |

## Interpretation

This moves the PR-B default-on blocker for the private q4 collision-off FLRW
limit: current-head thermal-start LRS/non-LRS zero-shear parity and the
`N_eff_3T >= 3.0` cold floor both pass in a real long endpoint run.

This does not promote any optimization default-on by itself.  Collision-on and
nonzero-shear/Bianchi evidence remain separate blockers.  Phase-2 corrector
wall remains the dominant performance target after the physics-start fix.

## Cost-Effectiveness Self-Audit

```text
added_lines: 127
deleted_lines: 0
net_lines: 127
files_touched: 1 tracked audit note, 1 tracked validation ledger row
token_use_exact: UNAVAILABLE
token_use_basis: harness did not expose an exact token counter
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: yes
blocker_movement_ratio: 0.75
blocker_movement_scope: private q4 collision-off FLRW PR-B parity/floor evidence only; not resolution convergence, promotion readiness, collision-on parity, or Bianchi validity
validation_strengthened: yes
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```
