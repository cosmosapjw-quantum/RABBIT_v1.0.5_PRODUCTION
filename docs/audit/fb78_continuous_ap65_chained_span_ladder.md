# FB78 Continuous AP65 Chained Span Ladder

Date: 2026-05-20

## Scope

FB78 extends the private FB69/FB70 continuous AP65 diagnostic path with an
explicit restart-handoff chain.  FB69 now accepts supplied restart kwargs and
emits JSON-safe restart kwargs from each finite terminal state.  FB70 can then
run each `N_span_end` rung as a consecutive window, using the previous window's
terminal restart state as the next window's initial state.

This is still private diagnostic evidence.  FB78 does not reroute public
CPU-JAX/Rodas5P dispatch, change `canonical_forward_solver`, run SMC, claim
production SMC validation, add QKE, or make the all-freedom full-BBN path
publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/fb78_continuous_ap65_chained_span_ladder.json \
  --N-span-end-ladder 5e-11,1e-10,2e-10,5e-10 \
  --h-max 5e-11 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference \
  --chain-restart-handoff
```

Output:

- `contract=augmented_continuous_ap65_full_bbn_span_ladder_fb70_v1`
- `artifact_payload_sha256=463418cba619ef8199b642debcd3425f54a3fd21f24b62038a85ecba5f1e46b9`
- manifest file SHA256 `f3c22071c252c990041aea33471db0b52ecb2da59cddf59872300ba84bdc36fa`
- `passed=true`
- `physical_full_bbn_span_ready=false`
- `span_rows=4`
- `rows_passed=4`
- `rows_reaching_endpoint=0`
- `rows_full_bbn_completed=0`
- `T_final_MeV_min=0.799999999607141`
- `T_final_MeV_max=0.7999999999607141`
- `max_span_length=3e-10`
- `chain_restart_handoff_enabled=true`
- `restart_handoff_ready_rows=4`
- `source_evaluations_total=588`
- `stage_source_evaluations_total=70`
- `step_count_total=10`
- `terminal_completion_class=completed_hot_endpoint`

The row spans were chained as `(0,5e-11)`, `(5e-11,1e-10)`,
`(1e-10,2e-10)`, and `(2e-10,5e-10)`.  Rows after the first used the
previous window's terminal restart kwargs as their initial state.  Every row
remained a hot endpoint near `0.8 MeV`; this does not yet reach the `0.01 MeV`
full-BBN endpoint.

## Claim Boundary

Allowed claim:

- The private continuous-AP65 span classifier now has a real consecutive-window
  restart-handoff path, and the smoke ladder can propagate terminal state
  across four tiny windows without unphysical terminal observables.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
