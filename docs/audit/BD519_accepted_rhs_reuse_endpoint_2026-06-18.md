# BD519 Accepted RHS Reuse Endpoint Review

Date: 2026-06-18

## Scope

BD519 is a final review of the current opt-in q4 collision-on endpoint recipe
after BD517 and BD518.  The retained runtime change reuses the accepted Newton
RHS from one BDF2 substep as the next substep's AB2 predictor input when the
phase-2 background is not changing between substeps.  The path is disabled for
explicit background-node substeps.

This is not a new default, not a public-production claim, and not a QKE claim.
Raw negative evidence remains serialized.

## Exact Endpoint Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint/bd519_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint/bd519_q4_accepted_rhs_reuse_periodic4_endpoint.json \
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
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint/jax_cache \
  --progress-jsonl
```

Follow-up checks:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint \
  > diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint/bd519_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd519_accepted_rhs_reuse_periodic4_endpoint
```

## Artifact Comparison

BD517 is the prior positive opt-in recipe:
threshold-4 adaptive AB2 residual guard plus periodic Newton-Jacobian refresh
interval 4.

BD518 repeats that recipe while normalizing explicit-zero Newton refresh counts
in artifact tooling.  The BD518 endpoint process started before the BD518
commit completed, so BD518 is useful as a repeat timing artifact but not as
runtime validation of the source-level refresh-count alias patch.

BD519 is the fresh endpoint artifact after accepted-RHS reuse.

| metric | BD517 | BD518 | BD519 | delta BD519-BD518 | delta BD519-BD517 |
|---|---:|---:|---:|---:|---:|
| `/usr/bin/time` elapsed | 52:32.00 | 51:41.05 | 49:48.29 | n/a | n/a |
| max RSS KB | 5367808 | 5366368 | 5364384 | -1984 | -3424 |
| exit status | 0 | 0 | 0 | 0 | 0 |
| total wall s | 3082.342632 | 3033.020096 | 2920.393145 | -112.626951 | -161.949487 |
| phase2 wall s | 1275.945168 | 1265.577255 | 1168.783388 | -96.793868 | -107.161780 |
| payload wall s | 897.917638 | 879.334121 | 876.197773 | -3.136348 | -21.719865 |
| source nonpayload wall s | 512.265861 | 499.740607 | 491.534741 | -8.205866 | -20.731120 |
| host Jacobian wall s | 174.983878 | 172.680752 | 171.321974 | -1.358777 | -3.661903 |
| outer linear wall s | 8.477717 | 7.920874 | 7.484541 | -0.436332 | -0.993176 |
| residual unattributed wall s | 212.752369 | 207.766487 | 205.070728 | -2.695760 | -7.681642 |
| AB2 RHS predictor wall s | 165.660944 | 164.903700 | 4.321430 | -160.582270 | -161.339513 |
| AB2 residual guard wall s | 24.897236 | 24.448804 | 25.250784 | +0.801980 | +0.353547 |
| Newton solve-call wall s | 483.271162 | 480.351277 | 512.663333 | +32.312056 | +29.392171 |
| phase2 coarse step wall s | 366.760979 | 364.714758 | 334.797153 | -29.917606 | -31.963826 |
| phase2 refined step wall s | 699.164805 | 696.091517 | 632.699705 | -63.391812 | -66.465100 |
| rejected attempt wall s | 0.594444 | 0.563295 | 0.561081 | -0.002214 | -0.033362 |
| rejected replay wall s | 0.594444 | 0.563295 | 0.561081 | -0.002214 | -0.033362 |
| source evaluation count | 108324 | 108324 | 108324 | 0 | 0 |
| dynamic payload build count | 14762 | 14762 | 14762 | 0 | 0 |
| Newton Jacobian eval count | 46805 | 46805 | 46805 | 0 | 0 |
| Newton Jacobian refresh count | 46805 | 46805 | 46805 | 0 | 0 |
| Newton Jacobian reuse count | 63372 | 63372 | 63372 | 0 | 0 |

Interpretation: BD519 removes almost all measured AB2 predictor RHS wall
(`~165 s -> 4.3 s`) without changing source-evaluation counts, payload build
counts, rejected-step counts, or final raw observables.  The Newton solve-call
wall rises by about `29-32 s`, so the net endpoint gain is smaller than the
subcomponent gain but remains positive: `-3.71%` versus BD518 and `-5.25%`
versus BD517.

## Raw Observable Delta

| metric | BD517 | BD518 | BD519 | delta BD519-BD518 | delta BD519-BD517 |
|---|---:|---:|---:|---:|---:|
| `T_final_MeV` | 0.0091396140450197508 | 0.0091396140450197508 | 0.0091396140450197508 | 0 | 0 |
| `N_eff_3T` | 3.0348087179727026 | 3.0348087179727026 | 3.0348087179727026 | 0 | 0 |
| `Yp` | 0.24201652194490023 | 0.24201652194490023 | 0.24201652194490023 | 0 | 0 |
| `D/H` | 2.4930281694645491e-05 | 2.4930281694645491e-05 | 2.4930281694645491e-05 | 0 | 0 |
| `Sigma_H` | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 0 | 0 |
| AB2 raw negative count | 8 | 8 | 8 | 0 | 0 |
| AB2 raw negative min | -1.9273731915983189e-06 | -1.9273731915983189e-06 | -1.9273731915983189e-06 | 0 | 0 |
| Newton raw trial negative count | 0 | 0 | 0 | 0 | 0 |
| raw candidate negative abs max | 0 | 0 | 0 | 0 | 0 |
| raw candidate negative event count | 0 | 0 | 0 | 0 | 0 |
| raw candidate negative value count | 0 | 0 | 0 | 0 | 0 |

The raw observable deltas are exactly zero for this same-recipe comparison.
The retained raw negative AB2 predictor evidence is unchanged and not clipped.

## AB2 and Rejection Counters

The final JSON has two nested freedom-composition rows and sixteen nested span
rows.  Rejected counts below are summed from nested span rows, not from the
top-level summary fields.

| metric | BD517 | BD518 | BD519 |
|---|---:|---:|---:|
| nested span rows | 16 | 16 | 16 |
| sum `n_rejected` | 6 | 6 | 6 |
| sum adaptive rejected-attempt count | 6 | 6 | 6 |
| selected AB2 initial-guess rejected total | 33233 | 33233 | 33233 |
| selected AB2 displacement-guard rejected total | 33203 | 33203 | 33203 |
| selected AB2 residual-guard rejected total | 22 | 22 | 22 |
| selected phase2 corrector rejected total | 0 | 0 | 0 |

This supports the narrow claim that the BD519 wall improvement is from avoiding
redundant RHS reconstruction for the next AB2 predictor input, not from changing
the adaptive-controller acceptance path.

## Code Review Notes

The runtime change is confined to the phase-2 BDF2/Newton substep loop:

- `_phase2_backward_euler_newton_solve_Y` reconstructs the accepted RHS from
  the accepted state, the residual, and `h_over_H`.
- `_phase2_bdf2_newton_network_step_attempt` consumes that accepted RHS as the
  next substep's current RHS seed.
- The seed is cleared whenever explicit background nodes are present, because
  the background sample can change between substeps.
- No CLI flag, default policy, promotion gate, readiness gate, manifest gate,
  hash gate, or figure gate was added.

The artifact-tooling part is deliberately limited to one false-green fix:
explicit-zero Newton refresh totals are treated as the evaluation count when
the evaluation count is positive.  This keeps BD517-family summaries from
misreporting refresh count `0` on periodic-refresh runs.

## Decision

BD519 is a retained endpoint-positive implementation PR.  It moves the measured
endpoint wall by `-112.626951 s` versus BD518 and `-161.949487 s` versus BD517,
while preserving final physical readouts and raw negative evidence.

It remains opt-in because the recipe includes
`adaptive_trust_after_acceptance` and periodic Newton-Jacobian refresh.  PR-B
parity and the `N_eff_3T >= 3.0` floor tripwire remain default-on blockers.

Next PR should target one of the two remaining large endpoint blockers:

1. Phase2 refined/coarse warm-start, with the same endpoint recipe and a wall
   comparison against BD519.
2. Payload/provider factory deflation, with the same endpoint recipe and a
   wall comparison against BD519.

Monolith reduction should only be done when it removes or isolates one of those
performance or physics blockers.  Pure cleanup that does not move endpoint wall
or physics correctness is deferred.

## Cost-Effectiveness

The exact token count for this PR is `UNAVAILABLE`; the harness did not expose a
per-PR token counter.  Line counts should be taken from `git diff --numstat` at
commit time.

Cost-effectiveness verdict: `ACCEPT_WITH_LIMITS`.

Rationale: the patch changes runtime behavior and produces a same-recipe
endpoint wall reduction, but total endpoint wall improves by less than 10%.
The main blocker moved is the measured AB2 predictor RHS reconstruction
subcomponent, which dropped by about 97%.
