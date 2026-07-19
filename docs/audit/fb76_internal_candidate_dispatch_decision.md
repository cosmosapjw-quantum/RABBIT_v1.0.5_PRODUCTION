# FB76 Internal Candidate Dispatch Decision

Date: 2026-05-20

## Scope

FB76 records whether the validated FB75 pilot-input gate warrants an internal
candidate-dispatch surface.  It hash-checks the file-backed FB75 payload and
rechecks every FB75 nested source file SHA before making the decision.  It is a
decision artifact only.  It does not add a backend alias, change
`canonical_forward_solver`, run SMC, register candidate or public dispatch,
claim production SMC validation, add QKE, or remove the continuous AP65
full-BBN span blocker.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/build_augmented_internal_candidate_dispatch_decision.py \
  --guarded-smc-pilot-gate-artifact diagnostic_outputs/fb75_guarded_smc_pilot_gate/fb75_guarded_smc_pilot_gate.json \
  --output diagnostic_outputs/fb76_internal_candidate_dispatch_decision/fb76_internal_candidate_dispatch_decision.json
```

Output:

- `contract=augmented_internal_candidate_dispatch_decision_fb76_v1`
- `artifact_payload_sha256=d361d9dab63dde7b54fed1656b6d7f61f5a10ec2ffdb2e6467dbb4d3f3b09518`
- manifest file SHA256 `2e871e1ec920371779bb22570e69ec4925369e0ef51d3a4f005cbb466604eac3`
- `passed=true`
- `internal_candidate_dispatch_decision=defer`
- `internal_candidate_dispatch_warranted=false`
- `registers_dispatch=false`
- `candidate_dispatch_registered=false`
- `canonical_forward_solver_registered=false`
- `public_dispatch_ready=false`
- `production_smc_validation_ready=false`
- `qke_scope=out_of_scope`
- `promotion_decision=not_promoted`
- `decision_blockers=[continuous_ap65_full_bbn_span_not_ready]`

Registry snapshot:

- `capability_key=jax_typeI_augmented_pstf_noqke_staging`
- `in_capability_by_key=true`
- `in_capability_by_backend=false`
- `feature_tier=substrate`
- `surface_class=diagnostic`
- `validation_mode=diagnostic`

## Claim Boundary

Allowed claim:

- The current FB75 pilot-input evidence was evaluated and the internal
  candidate-dispatch decision is deferred until the continuous AP65 full-BBN
  span blocker is cleared.

Forbidden claims:

- Public production support.
- Canonical/public dispatch registration.
- Production SMC validation.
- New SMC execution.
- QKE support.
- Continuous AP65 live-source full-span physical completion.
