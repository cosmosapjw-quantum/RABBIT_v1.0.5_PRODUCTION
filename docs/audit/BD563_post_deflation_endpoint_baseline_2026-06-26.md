# BD563 Post-Deflation Endpoint Baseline

Date: 2026-06-26

## Scope

BD563 reruns the retained BD519 q4 endpoint recipe at current head after the
BD558-BD562 deflation/review sequence.  This is an endpoint evidence PR, not a
new optimization, not a default-promotion claim, not public-production support,
and not QKE validation.

The purpose is to establish the current same-recipe baseline before selecting
BD564.  Raw observables and raw negative AB2 evidence are preserved.

## Exact Command

```bash
mkdir -p diagnostic_outputs/bd563_post_bd562_endpoint_baseline/checkpoints \
  diagnostic_outputs/bd563_post_bd562_endpoint_baseline/jax_cache

PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_q4_accepted_rhs_reuse_periodic4_endpoint.json \
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
  --span-row-checkpoint-dir diagnostic_outputs/bd563_post_bd562_endpoint_baseline/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd563_post_bd562_endpoint_baseline/jax_cache \
  --progress-jsonl \
  > diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_run.log 2>&1

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd563_post_bd562_endpoint_baseline \
  > diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd563_post_bd562_endpoint_baseline \
  > diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_component_check.txt
```

## Artifact Paths

- Final JSON:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_q4_accepted_rhs_reuse_periodic4_endpoint.json`
- Perf summary:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_perf_summary.json`
- Component checker:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_component_check.txt`
- `/usr/bin/time -v`:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_time_v.txt`
- Progress log:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/bd563_run.log`
- Span checkpoints:
  `diagnostic_outputs/bd563_post_bd562_endpoint_baseline/checkpoints/`

## Command Results

| Command | Result |
| --- | --- |
| Endpoint run | PASS, exit status 0 |
| Summarizer | PASS, JSON valid, 160890 bytes |
| Component checker | PASS, `PASS component wall attribution` |

Resource summary from `/usr/bin/time -v`:

| Metric | Value |
| --- | ---: |
| Elapsed wall | `42:15.68` |
| User time | `2520.94 s` |
| System time | `36.24 s` |
| CPU | `100%` |
| Max RSS | `4572180 KB` |
| Exit status | `0` |

## Same-Recipe Comparison: BD519 vs BD563

BD519 is the prior retained accepted-RHS-reuse + periodic4 endpoint evidence.
BD563 is the same recipe at current head, with only output/cache/checkpoint paths
changed.

| Metric | BD519 | BD563 | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | `49:48.29` | `42:15.68` | `-7:32.61` |
| Max RSS KB | 5364384 | 4572180 | -792204 |
| Selected wall s | 2920.393145 | 2499.378611 | -421.014534 |
| Phase2 corrector wall s | 1168.783388 | 1221.549889 | +52.766501 |
| Payload build wall s | 876.197773 | 823.276482 | -52.921291 |
| Source nonpayload overhead wall s | 491.534741 | 68.368362 | -423.166379 |
| Linear system wall s | 7.484541 | 6.949313 | -0.535228 |
| AB2 RHS predictor wall s | 4.321430 | 4.330565 | +0.009135 |
| Newton solve-call wall s | 512.663333 | 528.399827 | +15.736494 |
| Coarse step-attempt wall s | 334.797153 | 347.952574 | +13.155422 |
| Refined step-attempt wall s | 632.699705 | 665.520985 | +32.821280 |
| Step-attempt bookkeeping wall s | 415.310032 | 380.527078 | -34.782954 |
| Source evaluations | 87840 | 87840 | 0 |
| Stage source evaluations | 76846 | 76846 | 0 |
| Payload builds | 12198 | 12198 | 0 |
| Step count | 10972 | 10972 | 0 |
| Sum `n_rejected` over nested span rows | 6 | 6 | 0 |

Interpretation: current head is materially faster than the BD519 retained
baseline for the same endpoint recipe, and the endpoint observables/counters
match exactly.  The dominant wall reduction is currently reported as source
nonpayload overhead, so BD563 should be treated as the current baseline rather
than proof that BD558-BD562 cleanup caused the improvement.  BD564 must compare
against BD563, not BD519.

## Component Wall Attribution

| Component | BD519 wall s | BD563 wall s | Delta |
| --- | ---: | ---: | ---: |
| Phase2 corrector | 1168.783388 | 1221.549889 | +52.766501 |
| Payload | 876.197773 | 823.276482 | -52.921291 |
| Source nonpayload overhead | 491.534741 | 68.368362 | -423.166379 |
| Host Jacobian | 171.321974 | 173.619921 | +2.297947 |
| Outer linear system | 7.484541 | 6.949313 | -0.535228 |
| Residual unattributed | 205.070728 | 205.614644 | +0.543917 |
| Total selected wall | 2920.393145 | 2499.378611 | -421.014534 |

The component checker passed.  No negative or NaN residual was observed.

## Raw Observable Delta

| Metric | BD519 | BD563 | Delta |
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

Raw negative AB2 predictor evidence remains visible and unchanged.  No final
observable clipping or hiding is introduced.

## BD564 Selection

BD563 changes the next-PR target ordering:

1. Phase2 remains the largest named component at current head
   (`1221.55 s`) and worsened relative to BD519.  A phase2 refined/coarse
   orchestration improvement is still the higher-value candidate.
2. Payload is still large (`823.28 s`) but improved relative to BD519 without a
   build-count change.  Provider/factory deflation remains useful, but previous
   runtime-mass/static-bundle attempts were negative or inconclusive.
3. Source nonpayload overhead is now much smaller (`68.37 s`) and should not be
   the next target unless a regression reappears.

Recommended BD564: implement a single phase2 refined/coarse orchestration
change, compare the exact BD563 endpoint recipe before/after, and keep it
opt-in.  If the code change cannot be made small enough without altering
physics semantics, switch BD564 to a payload/provider factory deflation probe
only after documenting why phase2 was rejected.

## Cost-Effectiveness

Line cost: documentation-only in tracked files for this PR.  Diagnostic outputs
are generated artifacts and are not committed.

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

Blocker movement ratio: 0.35.  BD563 does not change solver behavior, but it
does move the planning blocker by replacing stale BD519 timing with a fresh
current-head endpoint baseline and by proving raw-observable equality under the
same recipe.

Cost verdict: ACCEPT.  The PR produces the baseline required for an endpoint
wall-improving BD564 and avoids another cleanup-only local minimum.
