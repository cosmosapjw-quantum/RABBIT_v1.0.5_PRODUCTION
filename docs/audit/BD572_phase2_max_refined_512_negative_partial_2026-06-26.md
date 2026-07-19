# BD572 Phase-2 Max-Refined 512 Negative Partial Endpoint Run

Date: 2026-06-26

## Scope

BD572 tested the existing opt-in
`--phase2-conservative-extent-corrector-max-refined-substeps 512` knob against
the BD563 endpoint recipe.  This was intended as a middle point between the
BD563 full step-doubling controller and the BD571 `coarse_only_diagnostic`
upper-bound run.

The run was interrupted deliberately after it became a clear negative
performance candidate and exceeded the BD563 max RSS.  Raw checkpoints, logs,
traceback, partial final JSON, and `/usr/bin/time -v` output are preserved.

This is a partial long-run negative result, not an endpoint success and not a
default-on optimization.

## Exact Command

```bash
mkdir -p diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/checkpoints \
  diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/jax_cache

PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_q4_max_refined_512_endpoint.json \
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
  --phase2-conservative-extent-corrector-max-refined-substeps 512 \
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/jax_cache \
  --progress-jsonl \
  > diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_run.log 2>&1
```

After row6 completed and row7 started, the process was interrupted with SIGINT
to avoid losing the partial artifacts to an OOM-style failure.  The shell
session returned code `130`.

## Artifact Paths

- Partial final JSON:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_q4_max_refined_512_endpoint.json`
- Partial perf summary:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_perf_summary.json`
- Component checker:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_component_check.txt`
- `/usr/bin/time -v`:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_time_v.txt`
- Progress/traceback log:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/bd572_run.log`
- Span checkpoints:
  `diagnostic_outputs/bd572_phase2_max_refined_512_endpoint/checkpoints/`

## Command Results

| Command | Result |
| --- | --- |
| Endpoint run | INTERRUPTED, signal 2, shell session code 130 |
| Summarizer | PASS on partial final JSON, but attribution row count is 0 |
| Component checker | FAIL, exit 1, `component attribution has no rows` |

Resource summary from `/usr/bin/time -v`:

| Metric | Value |
| --- | ---: |
| Elapsed wall before SIGINT | `21:25.60` |
| User time | `1275.76 s` |
| System time | `26.10 s` |
| CPU | `101%` |
| Max RSS | `5160708 KB` |
| Termination | `Command terminated by signal 2` |

The checker failure is expected and correct: the interrupted final JSON contains
only a partial wrapper, so a pass would be false-green.

## Partial Checkpoint Comparison

BD572 completed only the pairwise case through row6.  That is already enough to
reject the cap=512 candidate because the partial wall through row6 exceeds the
full BD563 pairwise endpoint wall.

| Pairwise row | `N_span` | BD563 wall s | BD572 wall s | BD563 steps/rej | BD572 steps/rej |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `[0.0, 1.0]` | 160.595283 | 159.258118 | 801 / 0 | 801 / 0 |
| 1 | `[1.0, 2.0]` | 89.878004 | 89.250287 | 801 / 0 | 801 / 0 |
| 2 | `[2.0, 2.25]` | 22.390293 | 22.915517 | 200 / 0 | 200 / 0 |
| 3 | `[2.25, 2.5]` | 22.627492 | 22.659704 | 200 / 0 | 200 / 0 |
| 4 | `[2.5, 2.75]` | 202.035375 | 213.372524 | 202 / 3 | 214 / 4 |
| 5 | `[2.75, 3.0]` | 315.118550 | 286.594380 | 401 / 0 | 491 / 1 |
| 6 | `[3.0, 4.0]` | 229.909783 | 460.376713 | 1601 / 0 | 3200 / 0 |
| 7 | `[4.0, 4.8]` | 181.756238 | not completed | 1280 / 0 | not completed |

Partial pairwise totals:

| Metric | BD563 rows 0-6 | BD572 rows 0-6 | Delta |
| --- | ---: | ---: | ---: |
| Wall s | 1042.554779 | 1254.427242 | +211.872463 |
| Step count | 4206 | 5907 | +1701 |
| Rejected steps | 3 | 5 | +2 |

BD563 full pairwise endpoint wall is `1224.311017 s`; BD572 already reached
`1254.427242 s` before row7.  Therefore cap=512 cannot be an endpoint wall
improvement for this recipe.

## Traceback / Interruption Site

The SIGINT landed inside dynamic collision payload construction:

```text
src/rabbit/validation/dynamic_collision_runtime.py:54
src/rabbit/jax/augmented_typeI_replay.py:2214
src/rabbit/jax/augmented_typeI_replay.py:2886
src/rabbit/transport/augmented_collision_bridge.py:3908
src/rabbit/transport/augmented_collision_bridge.py:3452
src/rabbit/transport/augmented_collision_bridge.py:2156
KeyboardInterrupt
```

This does not indicate a physics exception.  It records that the long-running
negative cap=512 candidate was interrupted while building/evaluating the dynamic
collision payload.

## Decision

Reject `max_refined_substeps=512` as the next endpoint-speed path:

- It does not preserve the BD563 wall profile.
- It increases activation/cold step counts and rejected steps.
- It exceeds the full BD563 pairwise endpoint wall before completing pairwise
  row7.
- It exceeds BD563 max RSS (`5160708 KB` vs `4572180 KB`).
- It does not produce a valid final component-attribution table; the checker
  correctly fails.

Next PR should not reduce the max-refined cap blindly.  The useful signal from
BD571 remains: skipping refined work is fast but changes raw observables.  The
next experiment must be a selective policy based on a real correctness proxy
or a payload/provider deflation that avoids changing phase-2 mathematical
control.

## Cost-Effectiveness

Line cost: documentation-only in tracked files for this PR.  Diagnostic outputs
are generated artifacts and are not committed.

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

Blocker movement ratio: 0.40.  The run did not improve endpoint wall, but it
eliminated a plausible cap-reduction path with partial long-run evidence and
protected the branch from promoting a worse knob.

Cost verdict: REJECT_CANDIDATE_ACCEPT_EVIDENCE.  The candidate failed, but the
evidence is useful and should redirect BD573 away from blind cap reduction.
