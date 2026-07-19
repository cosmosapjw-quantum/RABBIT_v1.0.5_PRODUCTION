# FB85 Continuous AP65 Adaptive Step Acceptance

Date: 2026-05-20

## Scope

FB85 fixes the private FB69 continuous-AP65 host Rodas5P prototype so the
embedded error estimate is used as an actual step-acceptance gate.  Before
FB85, `_host_rodas5p_step` returned `err_norm`, but `_run_step_cap_row`
recorded it and advanced the state even when `err_norm > 1`.  FB85 mirrors the
repo-local Rodas5P convention: rejected attempts leave `N` and `y` unchanged,
shrink `h`, increment `n_rejected`, and retry until the step is accepted or
the private diagnostic budget fails closed.

The row artifact now records `attempt_count`, `n_rejected`,
accepted/rejected step-size samples, rejected error samples, and the local
adaptive-controller policy.  FB70 span rows preserve that telemetry.

This is a solver-control correction for a private diagnostic artifact.  It
does not truncate, floor, or repair abundances; it does not add QKE; it does
not promote public dispatch or production SMC support.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_failure_triage.py \
  --output diagnostic_outputs/fb85_adaptive_fb82_continuous_ap65_failure_triage.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- nested contract `augmented_continuous_ap65_failure_triage_fb82_v1`
- nested artifact payload SHA256 `9a0e0fe58cf8e318777b6b2a3cadae4cc367dd3424b6df75da178ba4a41b04dd`
- manifest file SHA256 `9a5d0cd620a4036ed1dc65c20842f6efc068d6de18fcd4091baffb9fad4ebee5`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `first_failing_N_span_end=1.5e-09`
- `first_failure_row.step_count=2`
- `first_failure_row.attempt_count=2`
- `first_failure_row.n_rejected=0`
- `first_failure_row.error_norm_max=3.021584391530104e-14`
- `first_failure_row.rejected_error_norm_max=0.0`
- `first_failure_row.terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`
- `first_failure_row.terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`

## Interpretation

The current first strict-`Y_p` failure is not explained by an accepted
`err_norm > 1` host step in this tiny finite-difference ladder.  The adaptive
controller records zero rejected attempts and a very small maximum accepted
error estimate.  The blocker remains the sub-tolerance final-state `He4`
sign crossing localized by FB83/FB84, not a known rejected-step acceptance
bug in the current smoke configuration.

## Claim Boundary

Allowed claim:

- The private continuous-AP65 host stepper now rejects `err_norm > 1` attempts
  and preserves accept/reject telemetry through FB70 rows.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by truncating, flooring, or otherwise repairing `Y_p`.
