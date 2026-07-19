# FB88 Continuous AP65 Trace-Boundary Span Extension

Date: 2026-05-20

## Scope

FB88 extends the private continuous-AP65 trace-boundary evolution policy beyond
the FB87 smoke endpoint.  It reuses FB70 with
`abundance_positivity_policy=trace_boundary` and chained restart handoff, then
records pass/fail bracket, conservation, stiffness, and solver-effort telemetry.

This remains private diagnostic evidence.  It does not change public dispatch,
does not add QKE, and does not claim full-BBN completion.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_trace_span_extension.py \
  --output diagnostic_outputs/fb88_continuous_ap65_trace_span_extension.json \
  --N-span-end-ladder 2e-9,3e-9,5e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- contract `augmented_continuous_ap65_trace_span_extension_fb88_v1`
- artifact payload SHA256 `49b2e9e858ffb87fece72c0ea2a031ed174eae9a0934db2e086c74c9997ba251`
- manifest file SHA256 `01be70a682384b58296b85a712ba4f05b9898fa95810804b8ba17ec8ca8507fb`
- `classification=trace_boundary_extension_all_requested_spans_passed`
- largest passing endpoint: `5e-09`
- rows passed/failed: `3` / `0`
- best `T_final_MeV=0.7999999960714048`
- conservation max: `8.746901892447222e-18`
- conservation limit: `1e-16`
- conservation complete rows: `3`
- step count total: `20`
- attempt count total: `20`
- rejected steps total: `0`
- `error_norm_max=0.0006360385926131681`
- solver-effort complete rows: `3`
- stiffness-telemetry complete rows: `3`
- source evaluations total: `1166`
- stage source evaluations total: `140`

## Interpretation

The FB87 trace-boundary policy remains stable over the next smoke ladder through
`N_span_end=5e-9`.  The run stays at the hot endpoint scale near `0.8 MeV`, so
it is not a physical full-BBN result.  The useful conclusion is narrower:
within this tiny ladder, the previous strict `Y_p` sign blocker does not recur,
recorded mass-fraction residuals stay below `1e-16`, and the host-stepped
finite-difference Rodas5P prototype does not reject steps.

## Claim Boundary

Allowed claim:

- The private trace-boundary continuous-AP65 ladder passes through
  `N_span_end=5e-9` with conservation and solver-effort telemetry recorded.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
