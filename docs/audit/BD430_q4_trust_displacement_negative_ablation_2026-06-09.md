# BD430 q4 Trust-Displacement Negative Ablation

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 collision-on
performance ablation. QKE remains out of scope. This is not a solver validation,
public-production claim, or default-on promotion.

## Question

After BD429, the largest measured wall bucket is phase2 corrector:

```text
phase2 corrector = 1316.303678 s
payload = 657.815197 s
residual unattributed = 632.683240 s
```

BD419 collision-off LRS evidence suggested the opt-in
`phase2_network_ab2_initial_guess_residual_guard_policy=trust_displacement`
could reduce phase2 wall by skipping the AB2 residual guard. BD430 tested the
same idea in the current q4 collision-on LRS/non-LRS pairwise endpoint setting.

## Command

The BD430 run used the BD416 q4 collision-on thermal-start case, current code's
automatic BD429 thermo reuse absolute floor, and the opt-in trust-displacement
policy. It intentionally did not pass the BD429 CLI `--source-refresh...atol`
override, so the resolver path is exercised.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd430_q4_trust_displacement_collision_on/bd430_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd430_q4_trust_displacement_collision_on/bd430_q4_thermal_start_lrs_nonlrs_collision_on_trust_displacement.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd416_pr_n2_endpoint_ab/q4_pairwise_collision_on_thermal_case.json \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --weak-correction-level 0 --sigma-plus0 0.0 --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --phase2-network-ab2-initial-guess-residual-guard-policy trust_displacement \
  --stop-at-T-gamma-MeV 0.01 --progress-jsonl
```

Result: exit `0`, elapsed `47:55.84`, max RSS `3131816 KB`. Summarizer and
component attribution checker both exited `0`; checker output was `PASS
component wall attribution`.

## Result

Against BD429 strict residual guard:

| metric | BD429 strict | BD430 trust-displacement | delta |
|---|---:|---:|---:|
| elapsed | 47:11.96 | 47:55.84 | +43.88 s |
| max RSS KB | 3136476 | 3131816 | -4660 |
| total row wall | 2770.427630 s | 2812.808832 s | +42.381202 s |
| phase2 corrector | 1316.303678 s | 1270.817936 s | -45.485741 s |
| payload | 657.815197 s | 696.685853 s | +38.870656 s |
| residual unattributed | 632.683240 s | 673.874947 s | +41.191706 s |
| AB2 residual guard wall | 121.639050 s | 0.0 s | -121.639050 s |
| Newton solve-call wall | 400.713894 s | 536.512083 s | +135.798189 s |
| Newton residual wall | 23.843111 s | 155.343132 s | +131.500021 s |
| step-attempt bookkeeping wall | 453.516636 s | 373.526293 s | -79.990343 s |

Counters show the failure mode:

| counter | delta BD430 - BD429 |
|---|---:|
| AB2 residual guard count | -590919 |
| AB2 residual guard skipped | +590961 |
| Newton residual evaluations | +590927 |
| Newton iterations | 0 |
| Newton Jacobian evaluations | 0 |
| payload builds | 0 |
| stage reuse | 0 |

Skipping the guard removed the guard wall, but it shifted the work into Newton
residual evaluations and did not improve the endpoint run.

## Endpoint / Raw State

The run reached both endpoint rows and preserved raw observables. Summary:

```text
execution_passed = true
controlled_flrw_lrs_nonlrs_default_on_blocker_status = passed_pr_b_neff_floor_and_lrs_nonlrs_parity
controlled_flrw_lrs_nonlrs_neff_delta = 7.701442438445838e-06
```

Endpoint observables matched BD429 at the recorded precision for the blocker
status. Top-level `passed=false` remains expected for a one-resolution artifact
without resolution-convergence readiness.

## Decision

Do not promote `trust_displacement` for q4 collision-on pairwise endpoint work.
It is a useful negative ablation: the next phase2 target should not be "skip the
residual guard blindly." A better candidate must reduce Newton residual work or
step-attempt count, not just move work from guard checks into Newton.

## Cost-Effectiveness Line

```text
added_lines: 125
deleted_lines: 0
net_lines: 125
files_touched: 2
token_use_exact: UNAVAILABLE
token_use_basis: Codex harness did not expose exact per-turn token accounting in this workspace.
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no
blocker_movement_ratio: 0.25
validation_strengthened: yes
cost_effectiveness_verdict: FAILURE_MODE_RELOCATION
```

The run did not improve wall time. It strengthened the decision surface by
falsifying a plausible opt-in phase2 shortcut under the current collision-on
endpoint workload.
