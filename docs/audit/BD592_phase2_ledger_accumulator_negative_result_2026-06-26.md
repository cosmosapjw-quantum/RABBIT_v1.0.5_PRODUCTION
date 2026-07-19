# BD592 Phase-2 Ledger Accumulator Negative Result

Date: 2026-06-26

## Scope

BD592 tested a narrow phase-2 bookkeeping deflation hypothesis: keep
`Phase2AttemptLedger.attempt_payloads` intact for raw/debug evidence, but change
the hot accumulator path in `src/rabbit/validation/ap65_phase2_corrector.py` to
only scan scalar/mapping keys consumed by phase-2 summaries and attribution.

This was not a physics/controller change.  It did not change the phase-2
step-doubling controller, AB2 predictor, Newton/Jacobian policy, collision
payload policy, CLI defaults, raw observable handling, QKE scope, or any
default optimization gate.

## Result

Rejected.  The same-recipe endpoint run completed and preserved raw observables
and executable counters, but it did not reduce the measured blocker:

- selected wall regressed from `2441.611232 s` to `2470.209654 s`
  (`+28.598421 s`);
- phase-2 corrector wall was effectively unchanged, `1196.285249 s` to
  `1196.485721 s` (`+0.200472 s`);
- the targeted step-attempt bookkeeping wall regressed from `375.011428 s` to
  `378.636462 s` (`+3.625034 s`);
- payload wall also increased, `802.054217 s` to `822.709169 s`
  (`+20.654952 s`), consistent with run noise and/or indirect overhead rather
  than a successful phase-2 change.

The experiment patch was saved and reverted:

- `diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_reverted_code_experiment.diff`

## Commands

Unit and caller validation before the endpoint run:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_ap65_phase2_corrector.py
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py
python -m py_compile src/rabbit/validation/ap65_phase2_corrector.py src/rabbit/validation/augmented_continuous_ap65_rhs.py
```

Observed results:

- `tests/test_ap65_phase2_corrector.py`: `13 passed`
- `tests/test_augmented_continuous_ap65_rhs.py`: `315 passed, 3 warnings`
- `py_compile`: pass

Endpoint command:

```bash
mkdir -p diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/checkpoints diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/jax_cache
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_q4_phase2_ledger_accumulator_endpoint.json \
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
  --span-row-checkpoint-dir diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/jax_cache \
  --progress-jsonl \
  > diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_run.log 2>&1
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint \
  > diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_perf_summary.json
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint \
  > diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_component_check.txt
```

Observed endpoint result:

- endpoint run: PASS, exit status `0`
- elapsed: `41:46.99`
- max RSS: `4561928 KB`
- component checker: `PASS component wall attribution`

Note: another unrelated long-running Python process was active during the run,
so elapsed wall has extra scheduler noise.  The selected wall and component
wall values below are artifact-derived and remain the decision basis.

## Same-Recipe Comparison

| Metric | BD591 baseline | BD592 experiment | Delta |
| --- | ---: | ---: | ---: |
| selected wall seconds | `2441.611232` | `2470.209654` | `+28.598421` |
| attributed wall seconds | `2241.303109` | `2266.195416` | `+24.892306` |
| residual unattributed wall seconds | `200.308123` | `204.014238` | `+3.706115` |
| payload wall seconds | `802.054217` | `822.709169` | `+20.654952` |
| phase2 corrector wall seconds | `1196.285249` | `1196.485721` | `+0.200472` |
| host Jacobian wall seconds | `170.215108` | `171.975452` | `+1.760344` |
| outer linear wall seconds | `6.664489` | `6.934631` | `+0.270143` |
| nonpayload source overhead seconds | `66.084046` | `68.090442` | `+2.006396` |
| step-attempt wall seconds | `994.315824` | `1007.302174` | `+12.986350` |
| adaptive pair wall seconds | `1036.967501` | `1034.015152` | `-2.952348` |
| step-attempt bookkeeping seconds | `375.011428` | `378.636462` | `+3.625034` |

Executable counters from the 16 checkpoint rows were unchanged:

| Counter | BD591 | BD592 | Delta |
| --- | ---: | ---: | ---: |
| step count | `10972` | `10972` | `0` |
| rejected steps | `6` | `6` | `0` |
| source evaluations | `87840` | `87840` | `0` |
| stage source evaluations | `76846` | `76846` | `0` |
| dynamic payload builds | `12198` | `12198` | `0` |
| AB2 used count | `590895` | `590895` | `0` |
| AB2 rejected count | `33233` | `33233` | `0` |
| AB2 raw-negative count | `8` | `8` | `0` |
| AB2 raw-negative min | `-9.639915568538192e-07` | `-9.639915568538192e-07` | `0` |

Raw endpoint observables were unchanged to emitted precision:

| Observable | BD591 | BD592 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | `0.00913961404501975` | `0.00913961404501975` | `0` |
| `Yp` | `0.24201652194490023` | `0.24201652194490023` | `0` |
| `D/H` | `2.493028169464549e-05` | `2.493028169464549e-05` | `0` |
| `N_eff_3T` | `3.0348087179727026` | `3.0348087179727026` | `0` |
| `Sigma_H` | `3.3286755172789884e-31` | `3.3286755172789884e-31` | `0` |

## Interpretation

The whitelist accumulator hypothesis did not move the measured blocker.  It
preserved raw state and counters, but added code surface and increased the
targeted bookkeeping timer.  The correct action is to reject and revert the
patch, not to tune the whitelist further.

BD593 should pivot away from phase-2 ledger accumulator deflation.  The best
next endpoint-facing candidate is the payload/provider finite-mass factory
shape: add an opt-in exact runtime mass-scale provider mode that removes the
instantaneous `T_nu_e` value from the source-factory cache key while passing the
live mass scale through existing runtime kwargs.  It must remain default-off
before PR-B parity/floor gates and must be accepted only with same-recipe
endpoint wall plus raw-observable/counter equality evidence.

## Cost-Effectiveness

- added_lines: documentation only after revert
- deleted_lines: none in retained code
- production_code_delta: `0`
- token_use_exact: `UNAVAILABLE`
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no retained runtime behavior change
- physics_behavior_changed: no
- raw_state_preserved: yes
- blocker_movement_ratio: `0.0`
- cost_effectiveness_verdict: `REJECTED_EXPERIMENT_REVERTED`
