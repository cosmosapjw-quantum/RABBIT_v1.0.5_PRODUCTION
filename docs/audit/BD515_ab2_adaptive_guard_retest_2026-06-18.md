# BD515 AB2 Adaptive Residual-Guard Retest

Date: 2026-06-18

## Scope

BD515 retests the existing opt-in
`--phase2-network-ab2-initial-guess-residual-guard-policy adaptive_trust_after_acceptance`
on the current BD511 q4 thermal-start LRS/non-LRS collision-on endpoint
workload.  This is not a new default and not a solver-promotion claim.

The immediately preceding controller experiment, BD514, temporarily bounded
the internal phase-2 refined-substep growth.  BD514 completed but worsened the
same endpoint wall, so that patch was reverted and is not retained.

## Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint/bd515_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint/bd515_q4_ab2_adaptive_guard_endpoint.json \
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
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint/jax_cache \
  --progress-jsonl
```

Follow-up checks:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint \
  > diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint/bd515_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd515_ab2_adaptive_guard_endpoint
```

BD517 then tested the retained BD516 threshold-4 adaptive guard together with
the existing opt-in phase-2 Newton-Jacobian periodic refresh policy:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd517_ab2_adaptive_threshold4_periodic4_endpoint/bd517_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd517_ab2_adaptive_threshold4_periodic4_endpoint/bd517_q4_ab2_adaptive_threshold4_periodic4_endpoint.json \
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
  --span-row-checkpoint-dir diagnostic_outputs/bd517_ab2_adaptive_threshold4_periodic4_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd517_ab2_adaptive_threshold4_periodic4_endpoint/jax_cache \
  --progress-jsonl
```

## Results

BD515 exited `0`; `/usr/bin/time` elapsed was `54:31.44`, max RSS
`5366608 KB`, and component attribution checker reported `PASS`.

BD516 then lowered the opt-in adaptive trust threshold from 8 accepted residual
guards to 4 and reran the same command.  BD516 exited `0`; elapsed was
`53:31.40`, max RSS `5365940 KB`, and component attribution checker reported
`PASS`.

BD517 kept the BD516 threshold-4 behavior and added periodic Newton-Jacobian
refresh interval 4.  BD517 exited `0`; elapsed was `52:32.00`, max RSS
`5367808 KB`, and component attribution checker reported `PASS`.

Current-head comparison against the strict BD511 baseline:

| artifact | total wall s | phase2 s | payload s | source nonpayload s | residual s | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BD511 strict baseline | 3292.924846 | 1456.232442 | 917.821310 | 517.888655 | 215.214722 | 56:04.47 |
| BD514 bounded refinement | 3326.609065 | 1461.732573 | 924.354480 | 532.286930 | 220.071247 | 56:39.07 |
| BD515 adaptive guard | 3199.537191 | 1334.029096 | 920.663254 | 534.649711 | 221.690522 | 54:31.44 |
| BD516 adaptive guard threshold 4 | 3141.515915 | 1315.910702 | 908.091448 | 519.239724 | 213.935444 | 53:31.40 |
| BD517 threshold 4 + periodic4 Jacobian refresh | 3082.342632 | 1275.945168 | 897.917638 | 512.265861 | 212.752369 | 52:32.00 |

BD515 improves total component wall by `-93.387655 s` versus BD511
(`-2.84%`) and phase-2 wall by `-122.203346 s` (`-8.39%`).  Payload and source
nonpayload do not improve, so this is specifically a phase-2 policy win.

BD516 improves total component wall by `-151.408931 s` versus BD511 (`-4.60%`)
and by `-58.021276 s` versus BD515 (`-1.81%`).  The retained code change affects
only the opt-in `adaptive_trust_after_acceptance` policy; strict remains the
default residual-guard policy.

BD517 improves total component wall by `-210.582215 s` versus BD511 (`-6.39%`)
and by `-59.173284 s` versus BD516 (`-1.88%`).  The improvement is mainly phase
2 (`-180.287274 s` versus BD511, `-39.965534 s` versus BD516), with smaller
payload/source-overhead movement that is within workload variance and reuse
feedback effects.  This does not contradict BD512: periodic4 alone was endpoint
negative on the strict guard baseline, while periodic4 becomes endpoint positive
when combined with the threshold-4 adaptive guard.

Activation row 4 comparison:

| artifact | row4 wall s | phase2 s | AB2 residual guard s | Newton s | bookkeeping s |
| --- | ---: | ---: | ---: | ---: | ---: |
| BD511 strict baseline | 245.863661 | 217.787934 | 17.744830 | 70.296645 | 69.529305 |
| BD515 adaptive guard | 236.577357 | 207.182975 | 7.630985 | 76.813233 | 62.098364 |
| BD516 adaptive guard threshold 4 | 232.220211 | 204.134565 | 7.181103 | 76.081858 | 61.133671 |
| BD517 threshold 4 + periodic4 Jacobian refresh | 219.547883 | 191.634085 | 7.099560 | 64.465142 | 60.689363 |

The BD515 AB2 residual guard wall drops by `-10.113845 s` on row 4.  Newton
wall increases by `+6.516589 s`, but the net row wall still improves by
`-9.286304 s`.  BD516 lowers row 4 wall by another `-4.357146 s` versus BD515.
BD517 lowers row 4 wall by another `-12.672328 s` versus BD516, mostly by
reducing Newton wall (`76.081858 -> 64.465142 s`) and Newton-Jacobian subwall
(`24.159430 -> 9.229740 s` on the LRS activation row).

The controlled zero-shear LRS/non-LRS endpoint pair still reports
`default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`.
Top-level `passed=false` remains expected for a one-resolution artifact without
resolution-convergence readiness.  QKE remains out of scope and public dispatch
remains false.

## Relation to BD431

BD431 previously found `adaptive_trust_after_acceptance` negative on an earlier
BD429-family workload: total wall worsened `2770.427630 -> 2895.464348 s`.
BD515 does not invalidate BD431; it shows that after later payload/source/phase2
changes and the BD511 baseline, the same opt-in policy is now positive on the
current endpoint workload.  Treat the policy as workload- and implementation-
state-dependent.

## Decision

`adaptive_trust_after_acceptance` with threshold 4 is the current retained
phase-2 code improvement.  The current best measured opt-in runtime recipe is
threshold-4 adaptive guard plus periodic Newton-Jacobian refresh interval 4.
This recipe remains opt-in; no default optimization is promoted.  Do not
continue bounded refined-substep growth as a PR line; BD514 falsified that path
on endpoint wall.

Next useful work:

1. Repeat BD517 once or run a BD516/BD517 pair with warm cache controls to
   estimate variance and confirm the composition effect.
2. Inspect why adaptive guard is positive now while BD431 was negative:
   compare Newton solve-call, residual wall, and guard counts across BD431,
   BD511, BD515, BD516, and BD517.
3. If repeat evidence holds, expose the threshold-4 adaptive guard plus
   periodic4 Jacobian refresh as a recommended opt-in runbook recipe, not a
   default.
