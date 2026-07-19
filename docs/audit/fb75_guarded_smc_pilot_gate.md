# FB75 Guarded SMC Pilot Gate

Date: 2026-05-20

## Scope

FB75 adds a fail-closed diagnostic gate for guarded SMC pilot inputs.  It
validates current AP72/FB60/FB66/FB70/FB72/FB74 diagnostic products, requires
file-backed source hashes, records an AP69 SMC schema snapshot, and preserves
the closed public dispatch, production SMC, and QKE boundaries.

This does not run a new SMC sampler, register candidate or public dispatch,
claim production SMC validation, add QKE, or remove the continuous AP65
full-BBN span blocker.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/build_augmented_guarded_smc_pilot_gate.py \
  --ap72-smc-validation-artifact diagnostic_outputs/augmented_dynamic_e2e_bbn_current_figure_inputs/inputs/ap72_smc_validation.json \
  --full-bbn-suite-artifact diagnostic_outputs/fb60_full_bbn_diagnostic_suite/fb60_full_bbn_diagnostic_suite_manifest.json \
  --freedom-sweep-artifact diagnostic_outputs/fb66_freedom_ladder_full_bbn_sweep/fb66_freedom_ladder_full_bbn_sweep.json \
  --continuous-ap65-span-ladder-artifact diagnostic_outputs/fb70_continuous_ap65_full_bbn_span_ladder_fd.json \
  --weak-rate-bridge-artifact diagnostic_outputs/fb72_full_bbn_weak_rate_bridge_smoke/fb72_full_bbn_weak_rate_bridge.json \
  --figure-bundle-qa-artifact figures/augmented_publication_v2_current_qa/fb74_publication_figure_bundle_qa.json \
  --output diagnostic_outputs/fb75_guarded_smc_pilot_gate/fb75_guarded_smc_pilot_gate.json
```

Output:

- `contract=augmented_guarded_smc_pilot_gate_fb75_v1`
- `artifact_payload_sha256=18841d947067979eb5cdfddeef1a4c55656fbc62e92257a6e63197820bfea352`
- manifest file SHA256 `6087af94215ff25628c18e7a5fa3fd9a22ae2981166ec0ae467d6c036e661922`
- `passed=true`
- `guarded_smc_pilot_input_ready=true`
- `validated_full_bbn_product_inputs_ready=true`
- `statistical_pilot_input_ready=true`
- `runs_new_smc_sampler=false`
- `source_hashes_checked=true`
- `public_dispatch_ready=false`
- `production_smc_validation_ready=false`
- `candidate_dispatch_registered=false`
- `qke_scope=out_of_scope`
- `promotion_decision=not_promoted`
- `pilot_blockers=[continuous_ap65_full_bbn_span_not_ready]`

Key physical summaries:

- AP72 full-chain physical smoke completed 2 windows with repeated-run BBN
  readout source `rodas5p_repeated_run`.
- FB60 endpoint coverage is
  `T_final_MeV=0.004996944944314105--0.005000010963688484`.
- FB66 contributed 8 completed full-BBN freedom-ladder rows with zero failed
  and zero guarded rows.
- FB72 contributed 4 required/passed weak-rate pairs and 8 rows reaching the
  full-BBN endpoint.
- FB74 contributed 4 QA-checked current-artifact PNGs.
- FB70 remains hot-endpoint continuous-AP65 evidence with
  `physical_full_bbn_span_ready=false`, `rows_reaching_endpoint=0`,
  `rows_full_bbn_completed=0`, and
  `T_final_MeV=0.7999999999214282--0.7999999999607141`.

## Claim Boundary

Allowed claim:

- The current diagnostic full-BBN products are hash-checked and ready as
  guarded statistical-pilot inputs.

Forbidden claims:

- Public production support.
- Production SMC validation.
- Public or canonical dispatch.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Continuous AP65 live-source full-span physical completion.
