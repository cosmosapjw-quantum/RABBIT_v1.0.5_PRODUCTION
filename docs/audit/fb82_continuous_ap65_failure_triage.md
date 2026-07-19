# FB82 Continuous AP65 Failure Triage

Date: 2026-05-20

## Scope

FB82 triages the first FB81 refined-span failure without changing the
underlying physical gate.  It reruns the private FB81 chained continuous-AP65
bracket, extracts the first failing row, and records strict `Y_p` positivity,
abundance-bound tolerance, BBN observables, restart-handoff state, and
source-evaluation metadata separately.

This is diagnostic first-failure evidence only.  It does not relax positivity,
truncate or repair abundances, change public CPU-JAX/Rodas5P dispatch, alter
`canonical_forward_solver`, run SMC, claim production SMC validation, add QKE,
or make the all-freedom full-BBN path publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_failure_triage.py \
  --output diagnostic_outputs/fb82_continuous_ap65_failure_triage.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- `contract=augmented_continuous_ap65_failure_triage_fb82_v1`
- `artifact_payload_sha256=c64bf7175a6935b39859ae521a05fadd6548fcfa5e2326d3faff9da1e9f9a783`
- manifest file SHA256 `6eed5354131696519b92f3e7ba4c2132cf5f37f9b88e4911ebcabb7012649b0b`
- `passed=true`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `h_max=2.5e-10`
- `largest_passing_N_span_end=1e-09`
- `first_failing_N_span=[1e-09,1.5e-09]`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_T_final_MeV=0.7999999988214215`
- `Yp=-1.2294890184644955e-30`
- `abs_Yp=1.2294890184644955e-30`
- `abundance_bound_tolerance=1e-18`
- `abundance_bounds_ok=true`
- `bound_tolerance_masks_strict_sign=true`
- `DH=2.5844839174694797e-13`
- `Xn=0.1300000000856175`
- `Xp=0.869999999913933`
- `N_eff_3T=11.084874967851695`
- `Sigma_H=0.015620499328388281`
- `source_evaluation_count=118`
- `stage_source_evaluation_count=14`
- `step_count=2`
- `physical_full_bbn_span_ready=false`

## Interpretation

The first refined-span failure is not currently evidence for a large negative
abundance excursion.  It is a strict-sign failure: the raw `Y_p` is negative,
but its magnitude is much smaller than the existing abundance-bound tolerance.
FB82 preserves that distinction as a blocker instead of using it to relax the
gate.  The next physics/debugging step is to locate why the helium-4 readout
crosses through zero at the hot `0.8 MeV` endpoint before attempting larger
span extension.

## Claim Boundary

Allowed claim:

- The current private chained continuous-AP65 surface first fails at
  `N_span_end=1.5e-9` because strict `Y_p > 0` fails while the same row remains
  inside the existing abundance-bound tolerance.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by truncating, flooring, or otherwise repairing `Y_p`.
