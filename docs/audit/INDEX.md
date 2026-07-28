# Audit Notes Index

Superseded one-off `BD###`/`PR-###`/dated audit notes were pruned (provenance
remains in `git log` / `git show`). The notes that remain here are the ones still
linked from canonical docs. Start from these anchors:

- Endpoint blocker / deflation queue:
  [BD563_BD567_endpoint_blocker_pr_queue_2026-06-26.md](BD563_BD567_endpoint_blocker_pr_queue_2026-06-26.md)
- Current accepted endpoint evidence:
  [BD591_post_deflation_endpoint_recheck_2026-06-26.md](BD591_post_deflation_endpoint_recheck_2026-06-26.md),
  [BD592_phase2_ledger_accumulator_negative_result_2026-06-26.md](BD592_phase2_ledger_accumulator_negative_result_2026-06-26.md),
  [BD593_lrs_runtime_mass_scale_negative_result_2026-06-26.md](BD593_lrs_runtime_mass_scale_negative_result_2026-06-26.md)
- Latest code-precision audit:
  [BD594_external_code_precision_audit_report_2026-06-27.md](BD594_external_code_precision_audit_report_2026-06-27.md)
- D-044 external breakthrough intake closeout:
  [BD622_D044_open_breakthrough_intake_closeout_2026-07-24.md](BD622_D044_open_breakthrough_intake_closeout_2026-07-24.md)
- D-045/D-046 owner grants + OWNER-A orbit-chart discriminator closeout:
  [BD622_D045_D046_ownera_r6_orbit_chart_closeout_2026-07-27.md](BD622_D045_D046_ownera_r6_orbit_chart_closeout_2026-07-27.md)
- D-052/D-056 gate-matrix completion (row-9 closure, metrology, independent trajectory):
  [BD622_D056_gate_matrix_completion_2026-07-28.md](BD622_D056_gate_matrix_completion_2026-07-28.md)
  (contracts: [BD622_D053_row9_closure_contract_2026-07-28.md](BD622_D053_row9_closure_contract_2026-07-28.md),
  [BD622_D054_covariance_metrology_contract_2026-07-28.md](BD622_D054_covariance_metrology_contract_2026-07-28.md),
  [BD622_D055_covariance_metrology_contract_r2_2026-07-28.md](BD622_D055_covariance_metrology_contract_r2_2026-07-28.md),
  [BD622_D056_independent_trajectory_contract_2026-07-28.md](BD622_D056_independent_trajectory_contract_2026-07-28.md))
- D-057 adversarial correction of the claimed blocker closeout:
  [BD622_D057_blocker_resolution_adversarial_audit_2026-07-28.md](BD622_D057_blocker_resolution_adversarial_audit_2026-07-28.md)
- D-058–D-064 remedy DAG completion (lease repair, rebinding, covariance
  r3/r4, trajectory r2/r3, gate reconsideration):
  [BD622_D064_remedy_dag_completion_2026-07-29.md](BD622_D064_remedy_dag_completion_2026-07-29.md)
  (contracts: [BD622_D060_covariance_metrology_contract_r3_2026-07-28.md](BD622_D060_covariance_metrology_contract_r3_2026-07-28.md),
  [BD622_D061_covariance_metrology_contract_r4_2026-07-28.md](BD622_D061_covariance_metrology_contract_r4_2026-07-28.md),
  [BD622_D062_independent_trajectory_r2_contract_2026-07-28.md](BD622_D062_independent_trajectory_r2_contract_2026-07-28.md),
  [BD622_D063_independent_trajectory_r3_contract_2026-07-28.md](BD622_D063_independent_trajectory_r3_contract_2026-07-28.md))
- D-049/D-051 lint fix, regression gate PASS, OWNER-C closure acceptance:
  [BD622_D051_ownerc_closure_acceptance_2026-07-27.md](BD622_D051_ownerc_closure_acceptance_2026-07-27.md)
  (contract: [BD622_D050_ownerc_row6_closure_contract_2026-07-27.md](BD622_D050_ownerc_row6_closure_contract_2026-07-27.md))
- D-047 OWNER-B replay contract (frozen) and D-048 result:
  [BD622_D047_ownerb_target_replay_contract_2026-07-27.md](BD622_D047_ownerb_target_replay_contract_2026-07-27.md),
  [BD622_D048_ownerb_target_replay_result_2026-07-27.md](BD622_D048_ownerb_target_replay_result_2026-07-27.md)
- Anti-drift rules: `../TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`,
  `../../bbn_codex_anti_drift_cost_effective_policy.md`

To recover a pruned note: `git log --diff-filter=D --name-only -- 'docs/audit/*'`
then `git show <commit>^:docs/audit/<file>`.
