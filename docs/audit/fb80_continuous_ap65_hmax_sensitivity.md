# FB80 Continuous AP65 h_max Sensitivity

Date: 2026-05-20

## Scope

FB80 wraps the private FB70 continuous-AP65 span ladder in a one-control
`h_max` refinement diagnostic.  It holds the target span fixed at
`N_span_end=1e-9`, runs the same physics and solver controls across a strictly
decreasing `h_max` ladder, and classifies whether the FB79 first failure is
recovered by smaller internal steps.

This is diagnostic sensitivity evidence only.  It does not change public
CPU-JAX/Rodas5P dispatch, alter `canonical_forward_solver`, run SMC, claim
production SMC validation, add QKE, or make the all-freedom full-BBN path
publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_hmax_sensitivity.py \
  --output diagnostic_outputs/fb80_continuous_ap65_hmax_sensitivity.json \
  --N-span-end-ladder 1e-9 \
  --h-max-ladder 1e-9,5e-10,2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- `contract=augmented_continuous_ap65_hmax_sensitivity_fb80_v1`
- `artifact_payload_sha256=84d6aac41fc673889320ebc5802fa78049977917da193ad9154fc487048558e4`
- manifest file SHA256 `93a3dc65aa573f0217063fc947084caa97f6c5409e6d592cf8a0d8005a927912`
- `passed=true`
- `classification=h_max_refinement_recovers_observable_failure`
- `target_N_span_end=1e-09`
- `coarsest_h_max=1e-09`
- `largest_failing_h_max=1e-09`
- `first_passing_h_max_after_failure=5e-10`
- `smallest_passing_h_max=2.5e-10`
- `rows_failed=1`
- `rows_passed=2`
- `physical_full_bbn_span_ready=false`

The coarse `h_max=1e-9` row fails above the full-BBN endpoint with raw
`Yp_nonpositive` evidence.  The same target endpoint passes at `h_max=5e-10`
and `h_max=2.5e-10`.  This classifies the FB79 first failure as a solver
step-size/stiffness sensitivity at the current smoke scale, not as proof that
the underlying physics endpoint is impossible.

## Claim Boundary

Allowed claim:

- At `N_span_end=1e-9`, the current private continuous-AP65 FB70 surface
  recovers the coarse `Y_p` failure when `h_max` is refined from `1e-9` to
  `5e-10` or below.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by sign truncation or observable repair.
