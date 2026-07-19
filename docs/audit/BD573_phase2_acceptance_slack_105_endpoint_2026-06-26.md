# BD573 Phase-2 Acceptance-Slack 1.05 Endpoint Ablation

Date: 2026-06-26

## Scope

BD573 reruns the BD563 q4 endpoint matrix recipe with the existing opt-in
`--phase2-network-max-refined-acceptance-slack 1.05` flag.  This is an endpoint
performance ablation, not a default-on optimization, not QKE work, and not a
publication/public-production claim.

The goal is to test whether a narrow refined/coarse acceptance slack can retain
BD563 endpoint observables while capturing any material fraction of the BD571
`coarse_only_diagnostic` phase-2 wall reduction.

## Exact Command

```bash
mkdir -p diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/checkpoints \
  diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/jax_cache

PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_q4_acceptance_slack_105_endpoint.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd416_pr_n2_endpoint_ab/q4_pairwise_collision_on_thermal_case.json \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --weak-correction-level 0 \
  --sigma-plus0 0.0 \
  --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun \
  --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --phase2-network-ab2-initial-guess-residual-guard-policy adaptive_trust_after_acceptance \
  --phase2-network-newton-jacobian-refresh-policy periodic \
  --phase2-network-newton-jacobian-refresh-interval 4 \
  --phase2-network-max-refined-acceptance-slack 1.05 \
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/jax_cache \
  --progress-jsonl \
  > diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_run.log 2>&1

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint \
  > diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint \
  > diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_component_check.txt
```

## Artifact Paths

- Final JSON:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_q4_acceptance_slack_105_endpoint.json`
- Perf summary:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_perf_summary.json`
- Component checker:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_component_check.txt`
- `/usr/bin/time -v`:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_time_v.txt`
- Progress log:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/bd573_run.log`
- Span checkpoints:
  `diagnostic_outputs/bd573_phase2_acceptance_slack_105_endpoint/checkpoints/`

## Command Results

| Command | Result |
| --- | --- |
| Endpoint run | PASS, exit status 0 |
| Summarizer | PASS, JSON valid |
| Component checker | PASS, `PASS component wall attribution` |

Resource summary from `/usr/bin/time -v`:

| Metric | Value |
| --- | ---: |
| Elapsed wall | `42:37.43` |
| User time | `2549.28 s` |
| System time | `31.68 s` |
| CPU | `100%` |
| Max RSS | `4566936 KB` |
| Exit status | `0` |

## Same-Recipe Comparison: BD563 vs BD573

BD563 is the current accepted-RHS-reuse + periodic4 endpoint baseline.  BD573 is
the same matrix recipe plus `--phase2-network-max-refined-acceptance-slack
1.05`.

| Metric | BD563 | BD573 | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | `42:15.68` | `42:37.43` | `+21.75 s` |
| `/usr/bin/time` elapsed seconds | 2535.68 | 2557.43 | +21.75 (+0.86%) |
| Max RSS KB | 4572180 | 4566936 | -5244 (-0.11%) |
| Selected wall s | 2499.378611 | 2520.285897 | +20.907287 (+0.84%) |
| Phase2 corrector wall s | 1221.549889 | 1224.667073 | +3.117185 (+0.26%) |
| Payload build wall s | 823.276482 | 834.421257 | +11.144775 (+1.35%) |
| Source nonpayload overhead wall s | 68.368362 | 70.851486 | +2.483124 (+3.63%) |
| Host Jacobian wall s | 173.619921 | 174.975918 | +1.355997 (+0.78%) |
| Linear system wall s | 6.949313 | 7.234464 | +0.285151 (+4.10%) |
| Residual unattributed wall s | 205.614644 | 208.135699 | +2.521054 (+1.23%) |
| Source evaluations | 87840 | 87840 | 0 |
| Stage source evaluations | 76846 | 76846 | 0 |
| Payload builds | 12198 | 12198 | 0 |
| Step count | 10972 | 10972 | 0 |
| Sum `n_rejected` over nested span rows | 6 | 6 | 0 |

Interpretation: the slack policy preserves the BD563 controller trajectory and
raw observables, but also preserves the BD563 cost profile.  The small wall
increase is run noise plus payload/source overhead drift, not an endpoint-wall
improvement.

## Case-Level Endpoint Rows

The command runs the same two freedom cases as BD563.  Case-level wall is the
sum of live span-row checkpoint wall values.

| Case | BD563 wall s | BD573 wall s | Delta | BD563 `Yp` | BD573 `Yp` | BD563 `D/H` | BD573 `D/H` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pairwise `weak_rate_corrections+neutrino_collision_terms` | 1224.311017 | 1235.886942 | +11.575926 | 0.242000235664 | 0.242000235664 | 2.492936049701e-05 | 2.492936049701e-05 |
| all `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | 1275.067594 | 1284.398955 | +9.331361 | 0.242016521945 | 0.242016521945 | 2.493028169465e-05 | 2.493028169465e-05 |

Both cases preserve step and rejection counts: 5486 steps and 3 rejected steps
per case in both BD563 and BD573.

## Raw Observable Delta

The selected row is the zero-shear non-LRS all-freedoms endpoint row.

| Metric | BD563 | BD573 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | 0.00913961404501975 | 0.00913961404501975 | 0 |
| `Yp` | 0.24201652194490023 | 0.24201652194490023 | 0 |
| `D/H` | 2.493028169464549e-05 | 2.493028169464549e-05 | 0 |
| `N_eff_3T` | 3.0348087179727026 | 3.0348087179727026 | 0 |
| `Sigma_H` | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 0 |
| AB2 raw negative count | 8 | 8 | 0 |
| AB2 raw negative min | -1.927373191598319e-06 | -1.927373191598319e-06 | 0 |
| AB2 initial-guess rejected total | 33233 | 33233 | 0 |
| AB2 displacement-guard rejected total | 33203 | 33203 | 0 |
| AB2 residual-guard rejected total | 22 | 22 | 0 |
| Phase2 corrector rejected total | 0 | 0 | 0 |

Raw negative AB2 predictor evidence remains serialized and visible.  BD573 does
not clip, hide, or sanitize terminal observables.

## Component Wall Attribution

| Component | BD563 wall s | BD573 wall s | Delta |
| --- | ---: | ---: | ---: |
| Phase2 corrector | 1221.549889 | 1224.667073 | +3.117185 |
| Payload | 823.276482 | 834.421257 | +11.144775 |
| Source nonpayload overhead | 68.368362 | 70.851486 | +2.483124 |
| Host Jacobian | 173.619921 | 174.975918 | +1.355997 |
| Outer linear system | 6.949313 | 7.234464 | +0.285151 |
| Residual unattributed | 205.614644 | 208.135699 | +2.521054 |
| Total selected wall | 2499.378611 | 2520.285897 | +20.907287 |

The component checker passed.  No negative or NaN residual was observed.

## Decision

Reject `phase2_network_max_refined_acceptance_slack=1.05` as an endpoint-speed
candidate:

- It preserves BD563 raw observables and AB2 counters exactly.
- It does not reduce phase-2 corrector wall, payload builds, source evaluations,
  step count, or rejected steps.
- It increases selected endpoint wall by `20.91 s` (`0.84%`).
- It does not capture the BD571 phase-2 upper-bound speedup.

BD571 showed that skipping refined work is a large speed upper bound but changes
raw observables.  BD572 showed that blind max-refined cap reduction is worse.
BD573 shows that a loose acceptance slack at `1.05` is too conservative to move
the endpoint blocker.  The next endpoint-facing PR should switch away from
cap/slack-only phase-2 knobs and either:

1. implement a selective refined/coarse controller with a stronger observable
   or local-error proxy than acceptance slack alone, or
2. target payload/provider factory deflation where the same-recipe endpoint
   payload wall remains about `834 s`.

No optimization should be default-on before PR-B LRS/non-LRS parity and the cold
`N_eff_3T >= 3.0` floor tripwire pass.

## Cost-Effectiveness

Line cost: documentation-only in tracked files for this PR.  Diagnostic outputs
are generated artifacts and are not committed.

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

Blocker movement ratio: 0.30.  BD573 did not improve endpoint wall, but it
eliminated the plausible narrow acceptance-slack path with a full endpoint run
and preserved exact raw-state evidence.

Cost verdict: REJECT_CANDIDATE_ACCEPT_EVIDENCE.  The candidate failed as a
speedup, but the evidence prevents another local-minimum phase-2 knob PR.
