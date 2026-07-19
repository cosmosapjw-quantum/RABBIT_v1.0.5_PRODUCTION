# BD431 q4 Adaptive-Trust Negative Ablation

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 collision-on
performance ablation. QKE remains out of scope. This is not a solver validation,
public-production claim, or default-on promotion.

## Question

BD430 showed that blindly skipping the AB2 initial-guess residual guard
(`trust_displacement`) worsens the q4 collision-on endpoint workload by moving
work into Newton residual evaluations. BD431 tested the less aggressive
`adaptive_trust_after_acceptance` policy, which keeps strict guard checks during
warm-up and skips them only after repeated accepted checks.

## Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd431_q4_adaptive_trust_collision_on/bd431_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd431_q4_adaptive_trust_collision_on/bd431_q4_thermal_start_lrs_nonlrs_collision_on_adaptive_trust.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd416_pr_n2_endpoint_ab/q4_pairwise_collision_on_thermal_case.json \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --weak-correction-level 0 --sigma-plus0 0.0 --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --phase2-network-ab2-initial-guess-residual-guard-policy adaptive_trust_after_acceptance \
  --stop-at-T-gamma-MeV 0.01 --progress-jsonl
```

Result: exit `0`, elapsed `49:21.46`, max RSS `3135044 KB`. Summarizer and
component attribution checker both exited `0`; checker output was `PASS
component wall attribution`.

## Result

Against BD429 strict residual guard:

| metric | BD429 strict | BD431 adaptive trust | delta |
|---|---:|---:|---:|
| elapsed | 47:11.96 | 49:21.46 | +129.50 s |
| max RSS KB | 3136476 | 3135044 | -1432 |
| total row wall | 2770.427630 s | 2895.464348 s | +125.036718 s |
| phase2 corrector | 1316.303678 s | 1309.943212 s | -6.360465 s |
| payload | 657.815197 s | 714.335047 s | +56.519850 s |
| residual unattributed | 632.683240 s | 696.478071 s | +63.794830 s |
| AB2 residual guard wall | 121.639050 s | 30.079740 s | -91.559310 s |
| Newton solve-call wall | 400.713894 s | 510.885425 s | +110.171531 s |
| Newton residual wall | 23.843111 s | 125.243497 s | +101.400386 s |
| step-attempt bookkeeping wall | 453.516636 s | 401.668647 s | -51.847989 s |

Counters:

| counter | delta BD431 - BD429 |
|---|---:|
| AB2 residual guard count | -453515 |
| AB2 residual guard skipped | +453515 |
| Newton residual evaluations | +453515 |
| Newton iterations | 0 |
| Newton Jacobian evaluations | 0 |
| payload builds | 0 |
| stage reuse | 0 |

Adaptive trust still moves too much work from pre-Newton residual checks into
Newton residual evaluations. It does not improve this endpoint workload.

## Endpoint / Raw State

The run reached both endpoint rows and preserved raw observables. Summary:

```text
execution_passed = true
controlled_flrw_lrs_nonlrs_default_on_blocker_status = passed_pr_b_neff_floor_and_lrs_nonlrs_parity
controlled_flrw_lrs_nonlrs_neff_delta = 7.701442438445838e-06
```

Top-level `passed=false` remains expected for a one-resolution artifact without
resolution-convergence readiness.

## Decision

Do not promote `adaptive_trust_after_acceptance` for the q4 collision-on
pairwise endpoint path. Together, BD430 and BD431 close the simple
"skip AB2 residual guard" family for this workload. The next phase2 candidate
must reduce Newton residual work or accepted phase2 step-attempt count rather
than shifting residual checks into Newton.

## Cost-Effectiveness Line

```text
added_lines: 112
deleted_lines: 0
net_lines: 112
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

This is evidence work, not an implementation promotion. It prevents a plausible
but harmful phase2 policy from consuming further PR cycles.
