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
- D-047 OWNER-B replay contract (frozen) and D-048 result:
  [BD622_D047_ownerb_target_replay_contract_2026-07-27.md](BD622_D047_ownerb_target_replay_contract_2026-07-27.md),
  [BD622_D048_ownerb_target_replay_result_2026-07-27.md](BD622_D048_ownerb_target_replay_result_2026-07-27.md)
- Anti-drift rules: `../TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`,
  `../../bbn_codex_anti_drift_cost_effective_policy.md`

To recover a pruned note: `git log --diff-filter=D --name-only -- 'docs/audit/*'`
then `git show <commit>^:docs/audit/<file>`.
