# BD591 Post-Deflation Endpoint Recheck

Date: 2026-06-26

Purpose: rerun the exact BD563 endpoint recipe after BD583-BD590 deflation to
check for runtime/physics drift and refresh endpoint wall evidence.  This is
not an optimization PR and does not default-enable any new policy.

## Command

Same as BD563 with only output/checkpoint/cache paths changed to
`diagnostic_outputs/bd591_post_deflation_endpoint_recheck/`.

Artifact paths:

- final JSON:
  `diagnostic_outputs/bd591_post_deflation_endpoint_recheck/bd591_q4_accepted_rhs_reuse_periodic4_endpoint.json`
- run log:
  `diagnostic_outputs/bd591_post_deflation_endpoint_recheck/bd591_run.log`
- `/usr/bin/time -v`:
  `diagnostic_outputs/bd591_post_deflation_endpoint_recheck/bd591_time_v.txt`
- summary:
  `diagnostic_outputs/bd591_post_deflation_endpoint_recheck/bd591_perf_summary.json`
- component check:
  `diagnostic_outputs/bd591_post_deflation_endpoint_recheck/bd591_component_check.txt`

## Result

The endpoint run completed with exit status 0.  Both the pairwise and selected
all-freedom paths reached `T_gamma < 0.01 MeV`; the selected final
`T_final_MeV` is exactly the BD563 value to emitted precision.

| Metric | BD563 | BD591 | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | `42:15.68` | `41:16.81` | `-58.87 s` |
| max RSS KB | `4572180` | `4564456` | `-7724` |
| selected wall s | `2499.378611` | `2441.611232` | `-57.767378` |
| selected payload wall s | `823.276482` | `802.054217` | `-21.222265` |
| selected phase2 wall s | `1221.549889` | `1196.285249` | `-25.264639` |
| selected linear-system wall s | `6.949313` | `6.664489` | `-0.284824` |
| selected source nonpayload overhead s | `68.368362` | `66.084046` | `-2.284315` |
| selected steps | `10972` | `10972` | `0` |
| selected source evaluations | `87840` | `87840` | `0` |
| selected stage source evaluations | `76846` | `76846` | `0` |
| selected payload builds | `12198` | `12198` | `0` |

Interpretation: the wall improvement is evidence that the post-deflation tree
does not regress endpoint runtime under the same recipe, but it should not be
claimed as a causal optimization speedup.  The executable counters are
unchanged, so the wall delta is compatible with normal run/cache/environment
variance plus removed inactive Python surface.

## Raw Observables

| Observable | BD563 | BD591 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | `0.00913961404501975` | `0.00913961404501975` | `0.0` |
| `Yp` | `0.24201652194490023` | `0.24201652194490023` | `0.0` |
| `D/H` | `2.493028169464549e-05` | `2.493028169464549e-05` | `0.0` |
| `N_eff_3T` | `3.0348087179727026` | `3.0348087179727026` | `0.0` |
| `Sigma_H` | `3.3286755172789884e-31` | `3.3286755172789884e-31` | `0.0` |

Raw AB2 negative evidence is preserved:

- `selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_raw_negative_count_total = 8`
- `selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_raw_negative_min = -1.927373191598319e-06`
- AB2 used/rejected/residual-guard rejected counts match BD563 exactly:
  `590895`, `33233`, `22`.

## Component Wall Attribution

`scripts/check_component_wall_attribution.py` returned
`PASS component wall attribution`.

| Component | BD563 wall s | BD591 wall s | Delta |
| --- | ---: | ---: | ---: |
| payload | `823.276482` | `802.054217` | `-21.222265` |
| phase2 corrector | `1221.549889` | `1196.285249` | `-25.264639` |
| host Jacobian | `173.619921` | `170.215108` | `-3.404813` |
| outer linear system | `6.949313` | `6.664489` | `-0.284824` |
| source nonprobe nonpayload overhead | `68.368362` | `66.084046` | `-2.284315` |
| residual unattributed | `205.614644` | `200.308123` | `-5.306521` |

`jax_compile` and `jax_runtime` remain `0.0` in this attribution table because
this path records CPU-JAX host timers at the span-row attribution depth; no
claim is made that compile/runtime is separately solved.

## Cost Line

- added_lines: 142 total; 0 production
- deleted_lines: 2 total; 0 production
- net_lines: +140 total; 0 production
- files_touched: 2 audit documents
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, endpoint drift risk after deflation was checked
- blocker_movement_ratio: 0.35
- validation_strengthened: yes
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

## Anti-Drift Self-Audit

- real_blocker_moved: partially.  This PR refreshes endpoint evidence and
  proves the deflation block did not change raw endpoint observables or
  executable counters.
- gate_removed_or_consolidated: no new readiness/manifest/hash/figure gate.
- raw_state_preserved: yes; raw AB2 negative evidence is unchanged.
- verification: endpoint command, summarizer, and component checker all exited
  0.
- remaining_blocker: phase2/payload wall remains the endpoint performance
  target; PR-B parity and cold `N_eff_3T >= 3.0` remain default-on blockers.
