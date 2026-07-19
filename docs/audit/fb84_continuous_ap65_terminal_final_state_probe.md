# FB84 Continuous AP65 Terminal Final-State Probe

Date: 2026-05-20

## Scope

FB84 enriches the FB70 continuous-AP65 span rows with a terminal final-state
tail probe.  FB69 already carries `final_state_vector`; FB84 records the
packed replay-state `X_phase2[-9:]` tail in FB70 rows and compares
`X_phase2[5]` against the terminal BBN `Yp` observable.  This makes the FB83
source localization check direct: terminal `Yp` can now be compared against
the accepted final state, not only against the last attempted pre-step state.

This is diagnostic provenance only.  It does not relax positivity, truncate or
repair abundances, change public CPU-JAX/Rodas5P dispatch, alter
`canonical_forward_solver`, run SMC, claim production SMC validation, add QKE,
or make the all-freedom full-BBN path publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_failure_triage.py \
  --output diagnostic_outputs/fb84_refresh_fb82_continuous_ap65_failure_triage.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- nested contract `augmented_continuous_ap65_failure_triage_fb82_v1`
- nested artifact payload SHA256 `47efcd214cc16b0810797d19d59baca5ab0a1e965ab169416ac2cdb3fe486609`
- manifest file SHA256 `81cfb5fc61419c14306f703326d333135cc34d0a4d172bc545cb27195d065acb`
- `passed=true`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `first_failing_N_span_end=1.5e-09`
- `terminal_final_state_probe.available=true`
- `terminal_final_state_probe.x_phase2_tail_start=41`
- `terminal_final_state_probe.he4_tail_index=46`
- `terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`
- `terminal_final_state_probe.terminal_observable_Yp=-1.2294890184644955e-30`
- `terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`
- `terminal_final_state_probe.terminal_y_p_matches_final_state_tail=true`

## Interpretation

The FB82 strict-sign failure is not an observable extraction mismatch.  The
terminal BBN `Yp` exactly matches `He4=X_phase2[5]` in the accepted FB69 final
state tail.  Combined with FB83, this localizes the crossing between the last
attempted pre-step state and the accepted terminal state, at a scale far below
the current abundance-bound tolerance.

## Claim Boundary

Allowed claim:

- FB70 now preserves terminal final-state `X_phase2` tail evidence, and the
  current FB82 first-failure terminal `Yp` matches final-state `He4` exactly.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by truncating, flooring, or otherwise repairing `Y_p`.
