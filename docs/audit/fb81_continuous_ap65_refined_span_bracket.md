# FB81 Continuous AP65 Refined Span Bracket

Date: 2026-05-20

## Scope

FB81 applies the FB80 refined step policy to the private FB70 continuous-AP65
span ladder.  It holds `h_max=2.5e-10`, runs consecutive restart-handoff spans
through `N_span_end=(5e-10,1e-9,1.5e-9,2e-9)`, and records the largest passing
endpoint plus the first observed refined-hmax failure endpoint.

This is diagnostic bracket evidence only.  It does not change public
CPU-JAX/Rodas5P dispatch, alter `canonical_forward_solver`, run SMC, claim
production SMC validation, add QKE, or make the all-freedom full-BBN path
publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_refined_span_bracket.py \
  --output diagnostic_outputs/fb81_continuous_ap65_refined_span_bracket.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- `contract=augmented_continuous_ap65_refined_span_bracket_fb81_v1`
- `artifact_payload_sha256=76bf833035a9a23f7b444786d19924d7d676d23d2f79c086703faeb0ae3f212e`
- manifest file SHA256 `243e1170947e2ce33271c0410169be55f73fa5383de5ac305bb9005e051ab2f9`
- `passed=true`
- `classification=refined_span_pass_fail_bracketed`
- `h_max=2.5e-10`
- `largest_passing_N_span_end=1e-09`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_T_final_MeV=0.7999999988214215`
- `rows_passed=2`
- `rows_failed=2`
- `nested_step_count_total=6`
- `nested_source_evaluations_total=354`
- `physical_full_bbn_span_ready=false`

The first refined-hmax failing row remains above the full-BBN endpoint and
carries raw `Yp_nonpositive` evidence.  The following `2e-9` row has no restart
handoff because the `1.5e-9` row failed, so it is retained as downstream
failure evidence rather than as the primary bracket endpoint.

## Claim Boundary

Allowed claim:

- With `h_max=2.5e-10`, the current private chained continuous-AP65 surface
  passes through `N_span_end=1e-9` and first fails at `N_span_end=1.5e-9` with
  raw nonpositive `Y_p`.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by sign truncation or observable repair.
