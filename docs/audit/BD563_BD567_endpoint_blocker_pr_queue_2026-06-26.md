# BD563-BD593 Endpoint/Deflation PR Queue

Date: 2026-06-26

Scope: pivot from bounded span-ladder deflation to endpoint-facing blocker
movement.  Every PR in this queue must either quantify a same-recipe endpoint
baseline, implement a single endpoint-facing optimization or physics-readout
fix, or review the sequence for real blocker movement.

BD566 pivot: BD564-BD565 showed that endpoint-facing cache/warm-start candidates
can add surface without causal payload or phase-2 wins.  BD566-BD570 therefore
switch to net-negative deflation PRs before another runtime knob is attempted.

Anti-drift constraints remain active:

- QKE remains out of scope.
- No public-production or publication-ready claim.
- No whole-language rewrite.
- No optimization default-on before PR-B LRS/non-LRS parity and cold
  `N_eff_3T >= 3.0` floor tripwire.
- Preserve raw negative/nonfinite evidence; do not clip final observables.
- No new readiness/manifest/hash/figure gate.
- Segment-only timing must be labeled segment-only and cannot be reported as
  endpoint progress.

| PR | Target | Acceptance | Status |
| --- | --- | --- | --- |
| BD563 | Recover and run, or explicitly skip with exact reason, the current safest same-recipe endpoint/bounded endpoint baseline. | Records exact command, artifact path, wall/RSS, raw observable deltas, AB2 counters, rejected counts, `N_eff_3T`, `Yp`, `D/H`, and whether it is endpoint or bounded/segment-only. | done |
| BD564 | Implement one endpoint-facing improvement selected from BD563 evidence: phase2 refined/coarse warm-start or payload/provider factory deflation. | Same recipe before/after comparison reports endpoint wall or bounded endpoint wall delta; default remains off unless PR-B blockers are cleared. | done-rejected |
| BD565 | Switch to payload/provider factory deflation with a directly measured endpoint comparison, or explicitly reject the candidate with same-recipe evidence. | Reports raw observable deltas, endpoint reach, wall/RSS, counters, and component wall table if available. | done-rejected |
| BD566 | Deflate oversized span-ladder test fixture and telemetry preservation assertions without weakening behavior checks. | Net-negative line count; focused and full span-ladder test file pass. | done |
| BD567 | Continue net-negative test deflation on the AP65 RHS telemetry fixtures. | Net-negative line count; focused and full AP65 RHS test file pass or exact skip reason. | done |
| BD568 | Deflate `augmented_ap65_trace_summary.py` table/field plumbing without changing emitted keys. | Net-negative production line count; trace-summary focused tests pass. | done |
| BD569 | Deflate span-ladder production row/attempt field propagation without changing artifact schema. | Net-negative production line count; span-ladder focused tests pass. | done |
| BD570 | `/review` checkpoint for BD566-BD569 deflation. | Records net line delta, tests, drift risks, and whether endpoint work may resume. | done |
| BD571 | Run the existing opt-in `coarse_only_diagnostic` phase-2 pair-controller endpoint ablation against the BD563 recipe. | Reports endpoint wall/RSS, raw observable deltas, AB2 counters, rejected counts, component wall table, and whether the policy can be defaulted. | done |
| BD572 | Test the existing opt-in `max_refined_substeps=512` phase-2 cap as a possible midpoint between BD563 and BD571. | Endpoint or partial long-run evidence must decide whether blind cap reduction is viable. | done-rejected |
| BD573 | Test the existing opt-in `phase2_network_max_refined_acceptance_slack=1.05` as a narrow correctness-preserving phase-2 proxy. | Full endpoint evidence must show whether the slack captures any BD571 phase-2 wall gain without raw observable drift. | done-rejected |
| BD574 | Deflate phase-2 wall-summary serialization in `augmented_ap65_trace_summary.py` without changing emitted telemetry semantics. | Net-negative production line count; trace-summary and phase2 telemetry tests pass. | done |
| BD575 | Deflate source/host wall-summary serialization in `augmented_ap65_trace_summary.py` without changing emitted telemetry semantics. | Net-negative production line count; trace-summary and span-ladder telemetry tests pass. | done |
| BD576 | `/review` checkpoint for BD571-BD575, including endpoint ablations and deflation sequence. | Records net line delta, pass/fail validation, real blocker movement, and whether to continue deflation or return to endpoint runtime work. | done |
| BD577 | Compactly deflate span-ladder telemetry tests without weakening runtime-linked assertions. | Net-negative test line count; focused and full span-ladder test file pass; no standalone long audit doc. | done |
| BD578 | Deflate resolution-case selected metric forwarding without changing emitted keys or fallback logic. | Net-negative production line count; focused and full span-ladder tests pass; compact queue-only reporting. | done |
| BD579 | Deflate phase2 terminal wall key lists by deriving count/total/max keys from one base table. | Net-negative production line count; focused and full span-ladder tests pass; compact queue-only reporting. | done |
| BD580 | Deflate wall-attribution terminal wall key lists by deriving count/total/max keys from one base table. | Net-negative production line count; focused and full span-ladder tests pass; compact queue-only reporting. | done |
| BD581 | Final compact deflation PR before `/review`: reuse existing source-subwall/test assertion helpers without schema drift. | Net-negative code/test line count, compact queue-only reporting, and focused/full relevant tests pass. | done |
| BD582 | `/review` checkpoint for BD577-BD581 deflation. | Records aggregate line delta, validation, blocker movement, and whether to continue deflation or return to endpoint runtime work. | done |
| BD583 | Continue deflation only if the candidate is clearly net-negative and removes active endpoint-analysis surface; otherwise return to same-recipe endpoint runtime work. | Avoid another marginal cleanup PR: require roughly `<= -50` tracked net lines or a measured endpoint wall/blocker result. | done |
| BD584 | Continue only with a larger active-surface deletion candidate, or pivot to endpoint runtime evidence if no such deletion is found. | Net-negative production/test diff with focused/full tests, or same-recipe endpoint wall evidence. | done |
| BD585 | Continue deflation only if it deletes active endpoint-analysis surface with material net-negative diff; otherwise switch to endpoint runtime evidence. | Net-negative production/test diff with focused/full tests, or same-recipe endpoint wall evidence. | done |
| BD586 | Stop mechanical count-summary cleanup unless one more material deletion candidate exists; otherwise return to endpoint runtime evidence. | Net-negative production/test diff with focused/full tests, or same-recipe endpoint wall evidence. | done |
| BD587 | Choose between one final material deflation candidate and endpoint runtime evidence; do not continue marginal summary cleanup. | Net-negative production/test diff with focused/full tests, or same-recipe endpoint wall evidence. | done |
| BD588 | `/review` checkpoint for BD583-BD587 deflation sequence. | Records aggregate line delta, validation, blocker movement, and whether to stop deflation and resume endpoint runtime work. | done |
| BD589 | Continue only with structural active-code deflation or endpoint runtime evidence; avoid another local summary cleanup. | Either delete a larger active span-ladder surface with tests, or produce same-recipe endpoint wall/blocker evidence. | done |
| BD590 | Stop one-off summary helper harvesting unless a substantially larger active-code deletion is found; otherwise return to endpoint runtime evidence. | Net-negative production/test diff with focused/full tests, or same-recipe endpoint wall evidence with raw observable deltas. | done |
| BD591 | Return to endpoint-facing evidence or remove only proven-dead private surface found by static reference scan. | Endpoint wall evidence or net-negative dead-code deletion with compile and focused/full affected tests. | done |
| BD592 | Use BD591 evidence to choose a real endpoint-facing implementation target, not another local deflation PR. | Select phase2 or payload/provider work from component wall, change one executable path, and report same-recipe endpoint or bounded-endpoint wall/counter deltas. | done-rejected |
| BD593 | Pivot to payload/provider finite-mass factory shape after BD592 rejected phase-2 ledger accumulator deflation. | Implement only an opt-in exact runtime mass-scale provider mode, keep defaults unchanged before PR-B, and accept only with same-recipe endpoint wall plus raw-observable/counter equality evidence. | done-rejected |
| BD594 | Resume endpoint-facing work from BD593 evidence. Prefer an existing phase2 refined/coarse warm-start or provider deflation candidate only if it removes measured wall on the same endpoint recipe; otherwise run a narrower diagnostic that directly separates phase2/cold-row work. | Must report same-recipe endpoint or bounded-endpoint wall delta, raw observable/counter deltas, line cost, and whether code was accepted or reverted. | done (passed=true via 2-row ladder) |

## BD563 Result

See `docs/audit/BD563_post_deflation_endpoint_baseline_2026-06-26.md`.

Fresh same-recipe endpoint baseline:

- endpoint run: PASS, exit status 0
- elapsed: `42:15.68`
- max RSS: `4572180 KB`
- component checker: PASS
- selected wall: `2499.378611 s`
- final observables: `T_final_MeV=0.00913961404501975`,
  `Yp=0.24201652194490023`, `D/H=2.493028169464549e-05`,
  `N_eff_3T=3.0348087179727026`, `Sigma_H=3.3286755172789884e-31`
- raw AB2 negative evidence unchanged from BD519:
  count `8`, min `-1.927373191598319e-06`

BD564 should compare against BD563, not stale BD519 timing.  Current ordering:
phase2 refined/coarse orchestration first; payload/provider factory deflation
second if phase2 cannot be safely changed without physics semantics drift.

## BD564 Result

See `docs/audit/BD564_phase2_refined_guard_seed_negative_result_2026-06-26.md`.

BD564 attempted a narrow phase-2 refined-attempt AB2 residual-guard seed from a
clean coarse attempt.  The endpoint run completed and preserved final raw
observables exactly, but selected row wall regressed from `2499.378611 s` to
`2522.151432 s` and phase-2 corrector wall increased from `1221.549889 s` to
`1238.468794 s`.  Counts were unchanged: rejected steps `6`, source evals
`87840`, dynamic payload builds `12198`.

The experimental code was saved as
`diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/bd564_reverted_code_experiment.diff`
and rolled back.  BD565 should not carry forward this trust-counter path; move
to a candidate that removes actual endpoint work.

## BD565 Result

See `docs/audit/BD565_provider_static_bundle_negative_result_2026-06-26.md`.

BD565 enabled the existing PSTF radial provider static-bundle cache in the live
combined collision source factory.  The same-recipe endpoint run completed and
preserved final raw observables exactly.  Selected row wall improved from
`2499.378611 s` to `2485.903417 s`, but the targeted payload wall regressed
from `823.276482 s` to `839.577442 s`, static bundle cache hits stayed `0`, and
source/payload work counts were unchanged.

The experimental code was saved as
`diagnostic_outputs/bd565_provider_static_bundle_endpoint/bd565_reverted_code_experiment.diff`
and rolled back.  BD566 should not report BD565 as endpoint progress; it should
either find a cache key/reuse pattern that actually hits while remaining opt-in,
or switch away from provider-bundle caching.

## BD568 Result

BD568 folded the AP65 dynamic-collision wall-summary key expansion in
`src/rabbit/validation/augmented_ap65_trace_summary.py` into one prefix table
and one summary-field merge helper.  It does not change emitted JSON keys,
physics state, endpoint behavior, or runtime policy.

Cost line:

- added_lines: 62
- deleted_lines: 140
- net_lines: -78
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active production trace-summary surface reduced
- blocker_movement_ratio: 0.25
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

Validation:

- `python -m py_compile src/rabbit/validation/augmented_ap65_trace_summary.py`
  passed.
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_totals_pstf_provider_subwalls_from_compact_payload
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_extracted_builder_matches_legacy_wrapper
  tests/test_augmented_continuous_ap65_rhs.py::test_bd546_dynamic_collision_payload_build_fields_filter_samples
  tests/test_augmented_continuous_ap65_rhs.py::test_bd546_stage_collision_payload_reuse_fields_normalize_policy_stats
  tests/test_augmented_continuous_ap65_rhs.py::test_bd547_linear_system_summary_fields_combine_subtimers`
  passed: 5 tests.
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q
  tests/test_augmented_continuous_ap65_rhs.py` passed: 315 tests, 3 known
  truncation-guard warnings.

BD569 should continue deflation only where it deletes or consolidates active
span-ladder field propagation; otherwise endpoint-facing work should resume
after the BD570 review checkpoint.

## BD569 Result

BD569 folded simple `h_refinement_*_total` integer attempt aggregation in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` into
one explicit field table and one assignment loop.  It leaves float wall-time
max/count fields, text fields, limiting-field selection, adaptive-controller
diagnostics, emitted artifact keys, runtime policy, and physics state unchanged.

Cost line:

- added_lines: 99
- deleted_lines: 231
- net_lines: -132
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active span-ladder propagation surface reduced
- blocker_movement_ratio: 0.25
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_preserves_fb69_adaptive_step_telemetry
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_refines_failed_span_h_max_and_preserves_attempt_telemetry
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_preserves_linear_system_telemetry`
  passed: 3 tests.
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py` passed: 304
  tests, 1 skipped, 2 known deterministic-reference warnings.

BD570 is now the required review checkpoint for BD566-BD569 before returning to
endpoint-facing optimization or physics readout work.

## BD570 Review Checkpoint

Scope: BD566-BD569 deflation sequence, compared against `98971ce`.

Aggregate line cost:

- overall: added_lines 848, deleted_lines 1617, net_lines -769
- production code: added_lines 161, deleted_lines 371, net_lines -210
- tests: added_lines 596, deleted_lines 1243, net_lines -647
- queue document: added_lines 91, deleted_lines 3, net_lines +88
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter

Verification evidence:

- BD566: full span-ladder test file passed, 304 passed, 1 skipped.
- BD567: full AP65 RHS test file passed, 315 passed.
- BD568: trace-summary focused tests passed, then full AP65 RHS test file
  passed, 315 passed.
- BD569: h-refinement focused tests passed, then full span-ladder test file
  passed, 304 passed, 1 skipped.
- `git diff --check` passed for BD568 and BD569 before commit.

Adversarial findings:

- Real blocker moved: partial.  Active production/test surface shrank
  materially, but no endpoint wall, cold reach, or physics-readout blocker
  moved in BD566-BD569.  This is acceptable only because BD564-BD565 had just
  rejected endpoint-facing candidates and the user explicitly requested
  aggressive deflation.
- Runtime-linked telemetry: preserved.  BD568 and BD569 consolidate existing
  key expansion/aggregation; they do not add schema-only telemetry and do not
  change runtime linkage.
- False-green tooling removed: partial.  Large synthetic fixtures and repeated
  field plumbing were reduced, but the branch still has a large span-ladder
  monolith.  Further deflation should target only code that directly impedes
  endpoint work.
- Raw state preserved: yes.  No raw observable clipping, no NaN/negative
  suppression, no solver convention change, and no default optimization.
- Tests meaningful, not count-locks: yes for the touched paths.  The tests
  assert behavior and aggregation values, not just line counts or schema
  presence.

Decision:

- Do not keep harvesting deflation-only PRs indefinitely.
- Resume endpoint-facing work after this checkpoint unless the next deflation
  deletes more active monolith surface than it adds and directly reduces agent
  analysis cost for the next endpoint blocker.
- Next preferred PR: endpoint-facing phase-2/payload candidate with same-recipe
  wall, raw observable deltas, AB2 counters, rejected counts, `N_eff_3T`, `Yp`,
  and `D/H` comparison.  Keep it opt-in until PR-B parity and the cold
  `N_eff_3T >= 3.0` floor tripwire pass.

## BD571 Result

See `docs/audit/BD571_phase2_coarse_only_endpoint_ablation_2026-06-26.md`.

BD571 ran the same BD563 endpoint matrix recipe with the existing opt-in
`--phase2-network-pair-controller-policy coarse_only_diagnostic` flag.  The
endpoint run completed with exit status 0, the summarizer passed, and the
component attribution checker passed.

Measured result:

- `/usr/bin/time` elapsed improved from `42:15.68` to `26:25.24`
  (`-950.44 s`, `-37.48%`).
- selected wall improved from `2499.378611 s` to `1548.009806 s`
  (`-951.368805 s`, `-38.06%`).
- phase2 corrector wall improved from `1221.549889 s` to `254.994739 s`
  (`-966.555150 s`, `-79.13%`).
- payload wall regressed slightly from `823.276482 s` to `834.099025 s`;
  payload builds remained `12198`.
- source evaluations, stage source evaluations, step count, and rejected count
  were unchanged.
- selected endpoint raw observables changed:
  `Yp +1.67381130417e-06`, `D/H +1.49391537486e-08`,
  `N_eff_3T 0`, `Sigma_H 0`, `T_final_MeV 0`.
- raw AB2 negative evidence remains visible: count changed from `8` to `4`,
  min stayed `-1.927373191598319e-06`.

Decision:

- BD571 is a measured phase-2 endpoint-wall upper bound, not a valid default
  optimization.
- The next phase-2 PR must target a selective refined/coarse controller that
  preserves BD563 endpoint observables within an explicit same-recipe delta
  budget while retaining a material fraction of the BD571 phase-2 wall savings.
- PR-B LRS/non-LRS parity and cold `N_eff_3T >= 3.0` remain default-on blockers.

## BD572 Result

See `docs/audit/BD572_phase2_max_refined_512_negative_partial_2026-06-26.md`.

BD572 tested `--phase2-conservative-extent-corrector-max-refined-substeps 512`
as a possible middle point between BD563 full step-doubling and BD571
coarse-only.  The run was interrupted with SIGINT after pairwise row6 because it
was already a clear negative candidate and max RSS exceeded the BD563 endpoint
baseline.  Raw checkpoints, log, traceback, partial JSON, and `/usr/bin/time -v`
output were preserved.

Measured partial result:

- elapsed before SIGINT: `21:25.60`
- max RSS: `5160708 KB`, above BD563 `4572180 KB`
- completed pairwise rows: 0-6
- component checker: FAIL, exit 1, `component attribution has no rows`, expected
  because the final JSON is partial
- pairwise rows 0-6 wall: BD563 `1042.554779 s`, BD572 `1254.427242 s`,
  delta `+211.872463 s`
- BD563 full pairwise endpoint wall: `1224.311017 s`; BD572 exceeded this
  before row7 completed
- pairwise rows 0-6 steps: BD563 `4206`, BD572 `5907`, delta `+1701`
- pairwise rows 0-6 rejected steps: BD563 `3`, BD572 `5`

Decision:

- Reject blind max-refined cap reduction.  It increases retries/steps and wall
  before endpoint completion.
- BD573 should not pursue another cap-only knob.  The remaining options are a
  selective phase-2 controller with a real correctness proxy, or a
  payload/provider deflation that leaves phase-2 control math unchanged.

## BD573 Result

See `docs/audit/BD573_phase2_acceptance_slack_105_endpoint_2026-06-26.md`.

BD573 tested `--phase2-network-max-refined-acceptance-slack 1.05` as a narrow
correctness-preserving proxy between the full refined BD563 controller and the
BD571 `coarse_only_diagnostic` upper-bound run.  The endpoint run completed
with exit status 0, the summarizer passed, and the component attribution checker
passed.

Measured result:

- `/usr/bin/time` elapsed regressed from `42:15.68` to `42:37.43`
  (`+21.75 s`, `+0.86%`).
- selected wall regressed from `2499.378611 s` to `2520.285897 s`
  (`+20.907287 s`, `+0.84%`).
- phase2 corrector wall regressed slightly from `1221.549889 s` to
  `1224.667073 s` (`+3.117185 s`, `+0.26%`).
- payload wall regressed from `823.276482 s` to `834.421257 s`; payload builds
  remained `12198`.
- source evaluations `87840`, stage source evaluations `76846`, step count
  `10972`, and rejected count `6` were unchanged.
- selected endpoint raw observables were identical to BD563:
  `T_final_MeV=0.00913961404501975`,
  `Yp=0.24201652194490023`, `D/H=2.493028169464549e-05`,
  `N_eff_3T=3.0348087179727026`, `Sigma_H=3.3286755172789884e-31`.
- raw AB2 negative evidence and guard counters were identical to BD563:
  raw negative count `8`, min `-1.927373191598319e-06`, total rejected `33233`,
  displacement rejected `33203`, residual rejected `22`.

Decision:

- Reject acceptance-slack `1.05` as an endpoint-speed candidate.  It preserves
  raw state, but it also preserves the baseline phase-2 cost.
- Do not continue with another cap/slack-only phase-2 knob unless it is tied to
  a stronger local-error or endpoint-observable proxy.
- BD574 should either implement a genuinely selective refined/coarse controller
  or target payload/provider deflation, where the same-recipe endpoint payload
  wall remains roughly `834 s`.

## BD574 Result

See `docs/audit/BD574_phase2_wall_summary_deflation_2026-06-26.md`.

BD574 folded the phase-2 wall-time serialization block in
`src/rabbit/validation/augmented_ap65_trace_summary.py` into one prefix table and
the existing `_wall_seconds_summary_fields()` helper.  It does not change
runtime policy, endpoint behavior, raw observables, physics state, or artifact
semantics.

Cost line:

- added_lines: 32
- deleted_lines: 482
- net_lines: -450
- files_touched: 1 production file, 2 audit docs
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active phase-2 telemetry surface reduced
- blocker_movement_ratio: 0.35
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile src/rabbit/validation/augmented_ap65_trace_summary.py`
  passed.
- AP65 RHS targeted trace-summary tests passed: 5 tests.
- span-ladder phase2 telemetry targeted tests passed: 2 tests.
- full AP65 RHS file passed: 315 tests, 3 known truncation-guard warnings.
- full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

BD575 should remain deflation-focused only if it can delete active AP65
endpoint-analysis surface.  If the next deflation candidate is not clearly
net-negative and endpoint-relevant, return to payload/provider or selective
phase2 endpoint work.

## BD575 Result

See `docs/audit/BD575_source_host_wall_summary_deflation_2026-06-26.md`.

BD575 folded remaining source/host wall-time serialization blocks in
`src/rabbit/validation/augmented_ap65_trace_summary.py` into one prefix table
and the existing `_wall_seconds_summary_fields()` helper.  It preserves
non-wall provenance fields, runtime policy, endpoint behavior, raw observables,
physics state, and artifact semantics.

Cost line:

- added_lines: 19
- deleted_lines: 179
- net_lines: -160
- files_touched: 1 production file, 2 audit docs
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active source/host telemetry surface reduced
- blocker_movement_ratio: 0.30
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

Validation:

- `python -m py_compile src/rabbit/validation/augmented_ap65_trace_summary.py`
  passed.
- AP65 RHS targeted trace-summary tests passed: 5 tests.
- span-ladder targeted telemetry tests passed: 2 tests.
- full AP65 RHS file passed: 315 tests, 3 known truncation-guard warnings.
- full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

BD576 is now the required `/review` checkpoint for BD571-BD575.

## BD576 Review Checkpoint

See `docs/audit/BD576_review_BD571_BD575_2026-06-26.md`.

Range `f055745..HEAD`:

- production code: added `51`, deleted `661`, net `-610`
- audit docs / queue: added `978`, deleted `2`, net `+976`
- total tracked: added `1029`, deleted `663`, net `+366`
- token_use_exact: UNAVAILABLE
- blocker movement: partial

Review decision:

- BD571-BD573 closed three phase2 knob paths with endpoint or partial long-run
  evidence, but no default-safe endpoint speedup landed.
- BD574-BD575 achieved real production deflation in active telemetry code.
- The documentation surface expanded too much relative to minor deflation PRs.
  Future small deflation PRs should use compact queue entries rather than
  standalone long docs unless they contain endpoint/physics evidence.
- BD577 may continue deflation only if strongly net-negative and endpoint
  relevant; otherwise return to payload/provider or stronger selective phase2
  endpoint work.

## BD577 Result

BD577 folded repeated synthetic span-ladder telemetry rows, source subwall
assertions, and PSTF provider subwall assertions in
`tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py` into two local
test helpers: `_paired_rows()` and `_assert_mapping_values()`.  It does not
touch production code, runtime policy, physics state, artifact keys, or raw
observable handling.

Cost line:

- added_lines: 193
- deleted_lines: 315
- net_lines: -122
- files_touched: 1 test file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, span-ladder telemetry test surface reduced
- blocker_movement_ratio: 0.15
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

Validation:

- `python -m py_compile
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py` passed.
- Focused span-ladder telemetry tests passed: 5 tests.
- Full span-ladder test file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.
- `git diff --check` passed.

## BD578 Result

BD578 folded simple `_selected_metric(key, as_int=...)` resolution-case
forwarders in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` into
integer/float key tables plus one `selected_metric_summary` dict.  It leaves
payload build means, stage-reuse max/limiting-field fallbacks, collision active
column fallbacks, runtime policy, physics state, artifact keys, and raw
observable handling unchanged.

Cost line:

- added_lines: 75
- deleted_lines: 276
- net_lines: -201
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active resolution-case selected-summary surface reduced
- blocker_movement_ratio: 0.25
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused span-ladder selected-summary tests passed: 4 tests.
- Full span-ladder test file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.
- `git diff --check` passed.

## BD579 Result

BD579 replaced duplicated phase-2 terminal wall `*_count`, `*_total`, and
`*_max` key lists in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` with
one `_PHASE2_TERMINAL_WALL_BASE_KEYS` table and derived key expansion.  It
does not change emitted terminal keys, runtime policy, physics state, artifact
schema, or raw observable handling.

Cost line:

- added_lines: 29
- deleted_lines: 115
- net_lines: -86
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, phase2 terminal telemetry key duplication reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused span-ladder phase2 telemetry tests passed: 4 tests.
- Full span-ladder test file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.
- `git diff --check` passed.

## BD580 Result

BD580 replaced duplicated wall-attribution terminal wall `*_count`,
`*_total`, and `*_max` key lists in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` with
one `_WALL_ATTRIBUTION_TERMINAL_WALL_BASE_KEYS` table and derived key
expansion.  Non-wall provider counters remain explicit.  It does not change
emitted keys, runtime policy, physics state, artifact schema, or raw observable
handling.

Cost line:

- added_lines: 65
- deleted_lines: 162
- net_lines: -97
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, wall-attribution terminal telemetry duplication reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused wall-attribution/provider telemetry tests passed: 4 tests.
- Full span-ladder test file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.
- `git diff --check` passed.

## BD581 Result

BD581 derived source-subwall wall base names from `_SOURCE_EVALUATION_SUBWALL_BASE_KEYS`
and reused `_assert_mapping_values()` in live-progress/payload-summary tests.  Cost: `+120/-131=-11`,
token_use_exact `UNAVAILABLE`, blocker_movement_ratio `0.05`,
cost-effectiveness `ACCEPT_WITH_LIMITS`; no runtime/physics/schema/raw-state
change.  Validation: touched `py_compile`, 6 focused tests, full span-ladder
`304 passed, 1 skipped`, and `git diff --check`.

## BD582 Review Checkpoint

Scope: BD577-BD581 deflation sequence, compared against `087b6a1`.

Brooks-style findings:

- No must-fix correctness issue found in the changed code.  Symptom: the
  sequence only rewrites tests/key-list expansion and selected-summary
  forwarding.  Source: Ousterhout, Information Hiding.  Consequence: runtime
  policy, physics state, schema, and raw observables remain unchanged.  Remedy:
  no corrective patch required for correctness.
- Warning: do not repeat BD581-sized cleanup.  Symptom: BD581 delivered only
  `-1` tracked line overall and `-11` code/test lines excluding the queue
  document.  Source: Brooks, communication overhead; Ousterhout, tactical
  programming.  Consequence: PR overhead can exceed the deleted surface.
  Remedy: the next deflation PR must be materially net-negative, roughly
  `<= -50` tracked lines, or it should be replaced by endpoint runtime work.
- Warning: queue prose still costs lines.  Symptom: BD577-BD581 deleted
  production/test surface but added `+151` queue-document lines.  Source:
  Ousterhout, information leakage across audit prose.  Consequence: doc
  inflation weakens the user-visible deflation result.  Remedy: future minor
  deflation PRs get single compact queue entries; standalone long docs are
  reserved for endpoint/physics evidence.

Aggregate line cost:

- total tracked: added `580`, deleted `946`, net `-366`
- production code: added `162`, deleted `553`, net `-391`
- tests: added `265`, deleted `391`, net `-126`
- queue document: added `153`, deleted `2`, net `+151`
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter

Verification evidence:

- BD577: touched test `py_compile`, 5 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- BD578: touched production `py_compile`, 4 focused tests, full span-ladder
  file `304 passed, 1 skipped`.
- BD579: touched production `py_compile`, 4 focused tests, full span-ladder
  file `304 passed, 1 skipped`.
- BD580: touched production `py_compile`, 4 focused tests, full span-ladder
  file `304 passed, 1 skipped`.
- BD581: touched production/test `py_compile`, 6 focused tests, full
  span-ladder file `304 passed, 1 skipped`.
- Fresh range check: `git diff --check` over the current tree passed.

Self-audit:

- real_blocker_moved: partial.  Active production and test surface shrank, but
  no endpoint wall, cold reach, or physics-readout blocker moved.
- gate_removed_or_consolidated: no new gate added; no standalone
  readiness/manifest/hash/figure wrapper added.
- raw_state_preserved: yes.  No raw observable clipping, no negative/nonfinite
  hiding, no physics convention change, and no default optimization.
- verification: focused plus full span-ladder tests were run on each code PR;
  BD582 itself is docs-only and uses `git diff --check`.
- remaining_blocker: endpoint phase-2 wall and payload/provider work still
  need same-recipe endpoint evidence; PR-B parity and the cold
  `N_eff_3T >= 3.0` floor remain default-on blockers.

Decision:

- BD577-BD581 are acceptable as a bounded deflation batch.
- BD581 is the lower bound of usefulness; do not continue one-digit deletion
  PRs.
- BD583 should either delete a larger active endpoint-analysis duplication or
  return to endpoint runtime evidence with same-recipe wall, raw observable
  deltas, AB2 counters, rejected counts, `N_eff_3T`, `Yp`, and `D/H`.

## BD583 Result

BD583 deflated h-refinement wall telemetry aggregation in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`.
It added small row-sum/max helpers and base-key tables, then replaced repeated
payload and linear-system total/max/count blocks in row metadata and summary
assembly.  Emitted keys, fallback precedence, runtime policy, physics state,
raw observables, and default optimization state are unchanged.

Cost line:

- added_lines: 168
- deleted_lines: 359
- net_lines: -191
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active span-ladder wall aggregation surface reduced
- blocker_movement_ratio: 0.25
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused payload/linear-system/h-refinement telemetry tests passed: 5 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## BD584 Result

BD584 reused the BD583 row-sum/max helpers for selected source-evaluation,
wall-clock, rejected-step, host-Jacobian, and selected linear-system wall
summaries in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`.
It intentionally left mixed integer-count blocks alone where existing
`int(row.get(...))` vs `or 0` semantics differ.  Emitted keys, fallback
precedence for the changed float/count wall fields, runtime policy, physics
state, raw observables, and default optimization state are unchanged.

Cost line:

- added_lines: 39
- deleted_lines: 145
- net_lines: -106
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active selected wall-summary surface reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused selected wall/payload/linear-system telemetry tests passed: 4 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## BD585 Result

BD585 added a strict integer row-sum helper and reused it for selected and
h-refinement-backed count summaries in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`.
The strict helper preserves the prior `int(row.get(...))` behavior instead of
coalescing falsey values, while the older forgiving helper remains for fields
that already used `or 0`.  Emitted keys, runtime policy, physics state, raw
observables, and default optimization state are unchanged.

Cost line:

- added_lines: 162
- deleted_lines: 289
- net_lines: -127
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active count-summary surface reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused count/wall summary telemetry tests passed: 4 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## BD586 Result

BD586 reused the strict integer row-sum helper for frozen-source/JAX and
collision-relaxation count summaries in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`.
It did not touch the A-response any/all logic or matrix max reductions, where
boolean filtering and `_max_finite` semantics differ from simple integer sums.
Emitted keys, runtime policy, physics state, raw observables, and default
optimization state are unchanged.

Cost line:

- added_lines: 68
- deleted_lines: 168
- net_lines: -100
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active frozen-source/collision-relaxation summary
  surface reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused frozen-source/JAX and collision-component tests passed: 6 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## BD587 Result

BD587 reused the existing row-sum/max helpers for resolution-level selected
wall/source/payload summaries in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`.
It leaves `_max_finite` reductions, text/limiting-field selection, and boolean
logic untouched.  Emitted keys, runtime policy, physics state, raw observables,
and default optimization state are unchanged.

Cost line:

- added_lines: 62
- deleted_lines: 182
- net_lines: -120
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active resolution selected-summary surface reduced
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused resolution selected-summary tests passed: 2 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## BD588 Review Checkpoint

Scope: BD583-BD587 deflation sequence, compared against `4d2827c`.

Brooks-style findings:

- No must-fix correctness issue found in the changed code.  Symptom: the
  sequence replaces repeated row/reduction expressions with local helpers
  while preserving key names and fallback paths.  Source: Ousterhout,
  Information Hiding.  Consequence: runtime policy, physics state, schema, and
  raw observables remain unchanged.  Remedy: no corrective patch required for
  correctness.
- Warning: endpoint blocker did not move.  Symptom: BD583-BD587 cut active
  span-ladder production surface but did not run an endpoint recipe or change
  phase-2/payload work.  Source: Brooks, Conceptual Integrity; Ousterhout,
  tactical programming.  Consequence: continued deflation can become a local
  minimum even when line count improves.  Remedy: BD589 must either delete a
  larger structural active-code surface or return to same-recipe endpoint
  runtime evidence.
- Warning: queue prose still offsets code deletion.  Symptom: the five-PR block
  deleted `-644` production lines but added `+175` queue-document lines.
  Source: Ousterhout, information leakage.  Consequence: audit bookkeeping
  weakens visible deflation.  Remedy: future deflation entries stay compact;
  long standalone docs require endpoint or physics evidence.

Aggregate line cost:

- total tracked: added `676`, deleted `1145`, net `-469`
- production code: added `499`, deleted `1143`, net `-644`
- queue document: added `177`, deleted `2`, net `+175`
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter

Verification evidence:

- BD583: `py_compile`, 5 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- BD584: `py_compile`, 4 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- BD585: `py_compile`, 4 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- BD586: `py_compile`, 6 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- BD587: `py_compile`, 2 focused tests, full span-ladder file
  `304 passed, 1 skipped`.
- Fresh range check: `git diff --check` over the current tree passed.

Self-audit:

- real_blocker_moved: partial.  Cognitive/code-size blocker moved materially;
  endpoint wall, cold reach, and physics-readout blockers did not.
- gate_removed_or_consolidated: no new readiness/manifest/hash/figure wrapper
  added.
- raw_state_preserved: yes.  No raw observable clipping, no negative/nonfinite
  hiding, no physics convention change, and no default optimization.
- verification: full span-ladder suite was run for every production PR in the
  block.
- remaining_blocker: endpoint phase-2 wall and payload/provider work still need
  same-recipe endpoint evidence; PR-B parity and cold `N_eff_3T >= 3.0` remain
  default-on blockers.

Decision:

- BD583-BD587 are acceptable as a bounded, material production deflation block.
- Do not continue small mechanical summary cleanup.
- BD589 must either remove a larger structural active span-ladder surface or
  pivot back to endpoint runtime evidence with raw observable deltas, AB2
  counters, rejected counts, `N_eff_3T`, `Yp`, and `D/H`.

## BD589 Result

BD589 deflated `_h_refinement_attempt_summary` in
`src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py` by
replacing repeated row count/wall field copies with narrow key tables and two
conversion helpers.  The emitted artifact keys, conversion defaults, runtime
policy, physics state, raw observables, and default optimization state are
unchanged.

Cost line:

- added_lines: 128 total; 83 production
- deleted_lines: 174 total; 172 production
- net_lines: -46 total; -89 production
- files_touched: 1 production file, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, active h-refinement attempt summary surface
  reduced
- blocker_movement_ratio: 0.15
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT_WITH_LIMITS

Validation:

- `python -m py_compile
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  passed.
- Focused h-refinement attempt and host-lagged Jacobian tests passed: 4 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

Decision:

- BD589 is acceptable because it clears the material deletion threshold in
  active artifact code.
- BD590 should not continue one-off key-copy cleanup.  It must either remove a
  larger active-code structure or return to same-recipe endpoint runtime
  evidence.

## BD590 Result

BD590 deleted zero-use private helpers found by a source/test/script reference
scan:

- unused PSTF contraction invariant helpers in `pstf_contractions.py`;
- an unused radial-provider cache-key wrapper in
  `augmented_collision_bridge.py`;
- unused TEFF/ν-e/JAX numerical aliases;
- the unused span-ladder source-refresh wrapper.

Cost line:

- added_lines: 50 total; 0 production
- deleted_lines: 126 total; 124 production
- net_lines: -76 total; -124 production
- files_touched: 7 production files, 1 queue document
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes, dead private physics/transport/JAX surface
  removed
- blocker_movement_ratio: 0.20
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT

Validation:

- zero-use reference scan for deleted names returned no source/test/script
  references.
- `python -m py_compile` passed for all touched production modules.
- Focused suites passed: PSTF contractions 26, JAX weak/runtime fallback 30,
  augmented collision bridge cache tests 6, species-tagged bridge 5,
  source-refresh tests 2, deterministic collision/rate parity 46.
- Full augmented collision bridge file passed: 90 tests.
- Full span-ladder file passed: 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.
- `git diff --check` passed.

Decision:

- BD590 is a valid deflation PR because it deletes private dead code rather
  than moving key-copy boilerplate around.
- BD591 should prefer endpoint runtime evidence unless another static scan
  finds similarly proven-dead private surface.

## BD591 Result

See `docs/audit/BD591_post_deflation_endpoint_recheck_2026-06-26.md`.

BD591 reran the exact BD563 q4 endpoint recipe after BD583-BD590 deflation.
The run completed with exit status 0; summarizer and component-attribution
checker also exited 0.  Raw endpoint observables, AB2 negative evidence, steps,
source evaluations, and payload build counts match BD563 exactly.

Key comparison:

- elapsed: BD563 `42:15.68`, BD591 `41:16.81`
- max RSS KB: BD563 `4572180`, BD591 `4564456`
- selected wall s: BD563 `2499.378611`, BD591 `2441.611232`
- raw `Yp`, `D/H`, `N_eff_3T`, `T_final_MeV`, `Sigma_H`: exact emitted
  equality with BD563
- component checker: `PASS component wall attribution`

Decision:

- Treat the wall decrease as no-regression endpoint evidence, not as a claimed
  causal optimization speedup.
- BD592 should now resume endpoint-facing implementation.  The largest
  remaining measured targets are phase2 corrector wall and payload/provider
  wall; PR-B parity and cold `N_eff_3T >= 3.0` remain default-on blockers.

## BD592 Result

See `docs/audit/BD592_phase2_ledger_accumulator_negative_result_2026-06-26.md`.

BD592 tested and rejected a phase-2 attempt-ledger accumulator whitelist.  The
experiment preserved raw endpoint observables and executable counters exactly,
but selected wall regressed and the targeted bookkeeping wall did not improve.
The code experiment was saved in
`diagnostic_outputs/bd592_phase2_ledger_accumulator_endpoint/bd592_reverted_code_experiment.diff`
and reverted.

Key comparison against BD591:

- elapsed: BD591 `41:16.81`, BD592 `41:46.99`
- max RSS KB: BD591 `4564456`, BD592 `4561928`
- selected wall s: BD591 `2441.611232`, BD592 `2470.209654`
- phase2 corrector wall s: BD591 `1196.285249`, BD592 `1196.485721`
- step-attempt bookkeeping wall s: BD591 `375.011428`, BD592 `378.636462`
- payload wall s: BD591 `802.054217`, BD592 `822.709169`
- raw `Yp`, `D/H`, `N_eff_3T`, `T_final_MeV`, `Sigma_H`: exact emitted
  equality with BD591
- executable counters unchanged: steps `10972`, rejected steps `6`, source
  evaluations `87840`, stage source evaluations `76846`, dynamic payload
  builds `12198`, AB2 raw-negative count `8`
- component checker: `PASS component wall attribution`

Decision:

- Do not continue tuning phase-2 ledger accumulator filtering.
- BD593 should pivot to the payload/provider finite-mass source-factory shape:
  exact runtime mass-scale provider reuse, opt-in only, no default-on behavior
  before PR-B parity/floor tripwires, and accepted only with same-recipe
  endpoint wall plus raw-observable/counter equality evidence.

## BD593 Result

See `docs/audit/BD593_lrs_runtime_mass_scale_negative_result_2026-06-26.md`.

BD593 tested and rejected an opt-in LRS PSTF radial provider
`runtime_dynamic_exact` mass-scale mode.  The mode was runtime-linked and the
same-recipe endpoint run completed with exit status 0, but selected endpoint
wall regressed from BD591 `2441.611232 s` to BD593 `2505.162387 s`.

Key comparison against BD591:

- elapsed: BD591 `41:16.81`, BD593 `42:20.30`
- max RSS KB: BD591 `4564456`, BD593 `4250584`
- selected wall s: BD591 `2441.611232`, BD593 `2505.162387`
- payload wall s: BD591 `802.054217`, BD593 `811.037835`
- phase2 corrector wall s: BD591 `1196.285249`, BD593 `1223.461576`
- host JVP/Jacobian wall s: BD591 `170.215108`, BD593 `191.420645`
- raw `Yp`, `D/H`, `N_eff_3T`, `T_final_MeV`, `Sigma_H`: exact emitted
  equality with BD591
- executable counters unchanged: steps `10972`, source evaluations `87840`,
  stage source evaluations `76846`, dynamic payload builds `12198`, stage
  payload reuse `75642`, AB2 raw-negative count `8`
- component checker: `PASS component wall attribution`
- final artifact top-level `passed` is `false` despite row pass and
  `physical_full_bbn_span_ready=true`; summary blocker remains
  `tighten_resolution_or_solver_tolerance_until_terminal_deltas_converge`

The experimental diff was saved in
`diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_rejected_code_experiment.diff`
and reverted.  BD594 should not carry forward the runtime mass-scale provider
mode unless a later, narrower experiment shows a real payload reduction without
host/phase2 regression.

## BD594 Result

BD594 targeted the top-level `passed=false` blocker.  PR1 (committed `c63aee1`)
fixed the misleading diagnostic: `_resolution_blocking_next_step` now reports
`add_second_resolution_case_to_form_terminal_delta_comparison` for the
single-resolution-case path instead of
`tighten_resolution_or_solver_tolerance_until_terminal_deltas_converge`, since a
single case forms zero adjacent comparisons and has no terminal deltas to
converge.  Full span-ladder suite: `305 passed, 1 skipped`.

PR2 ran the designed two-row resolution ladder (q3 + the BD591 q4 case; only the
q-grid axis differs) to exercise the terminal-delta tolerance for real.

- artifact: `diagnostic_outputs/bd594_two_row_resolution_ladder/bd594_q3_q4_two_row_resolution_endpoint.json`
- elapsed `1:23:23` (two endpoint solves), max RSS `4847496 KB`, exit 0
- `passed=true`, `physical_full_bbn_span_ready=true`, `violations=[]`
- `resolution_ladder_case_count=2`, `adjacent_comparison_count=1`,
  `resolution_tolerance_ready=true`, `resolution_terminal_delta_violations=[]`,
  `resolution_axis_delta_kinds=['q_grid']`
- q3->q4 adjacent comparison `within_tolerance=true`; abs deltas
  `Yp=0`, `D/H=0`, `N_eff_3T=0`, `T_final_MeV=0`, `Sigma_H=3.147e-31`
  (all far inside tolerances `Yp 5e-3`, `D/H 5e-7`, `N_eff 5e-4`, `Sigma_H 5e-4`)
- both rows reach `T_final_MeV=0.00913961404501975`, `Yp=0.24201652194490023`,
  `D/H=2.493028169464549e-05`, `N_eff_3T=3.0348087179727026`, matching BD591
- new honest blocker: `extend_default_matrix_resolution_to_angular_grid_convergence`

Conclusion: the top-level `passed=false` was a single-resolution-row artifact,
not a solver-tolerance or physics failure.  Endpoint observables are q-converged
(q3 and q4 identical to emitted precision).  No defaults changed, no physics
change, raw evidence preserved.  Next endpoint blocker is angular-grid
convergence, then the phase2 corrector wall (49% of selected wall) via a
selective refined/coarse controller, kept opt-in until PR-B parity and the cold
`N_eff_3T >= 3.0` floor pass.

## Forward Plan (post-BD594)

BD595 (done): angular-grid convergence ladder, retired the post-BD594 blocker
`extend_default_matrix_resolution_to_angular_grid_convergence`.  Two-row ladder,
`q_laguerre_order=4` fixed, angular axis only: `N_mu=4/N_phi=6` baseline vs
`N_mu=6/N_phi=8` fine.  Artifacts:
`diagnostic_outputs/bd595_angular_convergence_ladder/`.

Result: `passed=true`, `resolution_tolerance_ready=true`, axis `angular_grid`,
zero terminal-delta violations.  Adjacent comparison within tolerance; abs deltas
`Yp/D-H/N_eff/T_final = 0`, `Sigma_H = 3.147e-31` (far inside tolerance).  Both
rows reach the BD591 observables exactly.  Elapsed `1:44:59`; max RSS
`13516864 KB` -- fine angular (`N_mu=6/N_phi=8`) is ~2.8x the q-ladder RSS, so
angular refinement is memory-bound.  New gate blocker:
`extend_default_matrix_resolution_to_q_and_angular_grid_convergence` (a combined
q+angular matrix).

Status: endpoint observables are now confirmed resolution-converged on both the
q axis (BD594) and the angular axis (BD595) individually -- they do not move with
refinement.  The gate's combined-matrix escalation is a thoroughness ladder, not
a physics signal (observables already invariant); further matrix rows give
diminishing returns at rising memory cost.  The next real blockers are PR-B
LRS/non-LRS parity (live shear `sigma != 0`, currently untested at endpoint) and
the phase2 wall lever below.

Phase2 wall lever (deferred; biggest perf target, ~49% of selected wall): a
selective refined/coarse controller is not yet safely implementable.  The
step-doubling local error `X_refined - X_coarse` requires computing the refined
step, and no per-step coarse-only proxy is surfaced (artifact `local_norm`
occurrences `= 0`).  Required sequence before any opt-in skip:

1. Add read-only per-step phase2 `local_norm` plus a candidate coarse-only proxy
   (coarse Newton displacement or AB2 predictor-corrector delta); no default
   change, telemetry only justified because it unblocks the measured wall lever.
2. One endpoint run; extract the per-step local-error vs proxy distribution.
3. Only if a proxy reliably predicts `target_met`, implement opt-in
   `selective_refined_on_local_error` with bit-identical fallback (`skip_tol=0`),
   accepted only with same-recipe endpoint wall plus raw-observable/counter
   parity, and kept default-off before PR-B parity and the cold
   `N_eff_3T >= 3.0` floor.

PR4 (drop `nested_gap_analysis`) was rejected: it is a live
negative-nested-gap validation guard in `scripts/summarize_perf_artifacts.py`,
not dead reporting.

## BD596 Result -- PR-B live-shear endpoint (first nonzero-shear evidence)

Retires the PR-B live-shear gap BD425 flagged as still open ("collision-on and
nonzero-shear/Bianchi evidence remain separate blockers").  The accepted FLRW
thermal-prerun start hard-guards against shear -- the anisotropic thermal prerun
is unimplemented (`augmented_continuous_ap65_rhs.py` ~1499: "FLRW thermal
neutrino start requires zero shear ... until the anisotropic thermal prerun is
implemented").  Worked around with `--neutrino-thermal-start-policy supplied`
plus the FLRW prerun's effective `T_nu_e0=0.7950627910399561`,
`T_nu_x0=0.794193458963342` (= BD591 IC), so BD591 is the matched `sigma=0`
control.

Run: q4, collision-on, non-LRS, initial `Sigma_H0=1e-3` (`sigma_sq0=1e-6`,
ratio `-2`).  Artifact `diagnostic_outputs/bd596_live_shear_parity/`.  Wall
`1:04:03`, max RSS `9818212 KB`, exit 0.

- row reaches `full_bbn_completed`, `span_ladder_passed=true`,
  `physical_full_bbn_span_ready=true`, `violations=[]`
- `controlled_flrw_lrs_nonlrs_parity` available / N_eff parity / floor pair /
  cold ready all `true`
- final `Sigma_H = 9.377e-5` -- shear persists physically (~10x decay from
  `1e-3`, not numerical-zero), so the shear dynamics are genuinely live
- abundance parity vs BD591 (FLRW `sigma=0`), all far inside terminal
  tolerances: `dYp=-3.2e-9`, `dD/H=-3.5e-12`, `dN_eff_3T=-1.35e-5`,
  `dT_final=+4.9e-9`

Conclusion: the live-shear non-LRS collision-on endpoint path is exercised for
the first time and reaches `T_gamma < 0.01 MeV`; at `Sigma_H0=1e-3` the FLRW
abundance limit is robust (anisotropy persists in the geometry but does not
imprint on `Yp/D-H/N_eff` within tolerance).  `passed=false` is the single-row
artifact (no adjacent comparison; BD594 semantics).  No defaults changed; raw
shear history preserved.  Remaining PR-B work: implement the anisotropic thermal
prerun so the FLRW recipe can carry shear natively, and/or a shear-amplitude
ladder to map the abundance-imprint threshold.
