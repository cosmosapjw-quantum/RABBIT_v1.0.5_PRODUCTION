# FB83 Continuous AP65 Yp Source Probe

Date: 2026-05-20

## Scope

FB83 localizes the first FB82 strict-`Y_p` failure by comparing the terminal
BBN readout against the packed FB69 last-attempted replay state.  The current
live-source replay contract packs `X_phase2` as the final 9 entries of the
state vector, and `Yp/He4` is `X_phase2[5]`, so the probe reads
`last_attempted_state_vector[-9 + 5]` and records that value separately from
the terminal observable.

This is diagnostic source-localization evidence only.  It does not relax
positivity, truncate or repair abundances, change public CPU-JAX/Rodas5P
dispatch, alter `canonical_forward_solver`, run SMC, claim production SMC
validation, add QKE, or make the all-freedom full-BBN path publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_y_p_source_probe.py \
  --output diagnostic_outputs/fb83_continuous_ap65_y_p_source_probe.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- `contract=augmented_continuous_ap65_y_p_source_probe_fb83_v1`
- `artifact_payload_sha256=bf8c39c5947c063cb800c2f2b34f75bb3a70ac311aa3a388d6578a07a6692bb1`
- manifest file SHA256 `234d89567922ddad92d0c3750c35522f0c205616bfb3dbb3f47f8885e936d3d5`
- `passed=true`
- `classification=terminal_y_p_sign_crossing_below_tolerance_after_positive_last_stage_he4`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_terminal_Yp=-1.2294890184644955e-30`
- `last_passing_terminal_Yp=8.116150311829752e-31`
- `terminal_Yp_delta=-2.0411040496474707e-30`
- `abundance_bound_tolerance=1e-18`
- `first_failing_last_attempted_He4=2.2765668298302704e-32`
- `last_passing_last_attempted_He4=1.2963142013342297e-30`
- `last_attempted_He4_delta=-1.273548533035927e-30`
- `terminal_sign_transition=positive_to_nonpositive`
- `x_phase2_tail_start=41`
- `he4_tail_index=46`
- `physical_scale_assessment=sub_tolerance_terminal_sign_crossing`
- `physical_full_bbn_span_ready=false`

## Interpretation

The first strict-sign failure is currently localized to a sub-tolerance
terminal sign crossing.  The last attempted pre-step `He4` readout in the
failing row is still positive (`2.2765668298302704e-32`), while the terminal
BBN observable extracted after the step is negative
(`-1.2294890184644955e-30`).  Both values are tiny compared with the existing
`1e-18` abundance-bound tolerance.  This narrows the next debugging target to
the final step/update and strict-sign gate interaction, not a resolved
macroscopic helium excursion.

## Claim Boundary

Allowed claim:

- The current private chained continuous-AP65 first failure is a sub-tolerance
  terminal `Y_p` sign crossing after a positive last-attempted `He4` state.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by truncating, flooring, or otherwise repairing `Y_p`.
