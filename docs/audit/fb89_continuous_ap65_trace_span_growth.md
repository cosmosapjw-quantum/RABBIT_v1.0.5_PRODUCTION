# FB89 Continuous AP65 Trace-Boundary Span Growth

Date: 2026-05-20

## Scope

FB89 adds a private multiplicative scout on top of FB88.  It generates a
geometric trace-boundary span ladder, runs the FB88 gate, and re-checks nested
claim boundaries plus conservation, stiffness, solver-effort, and rejection
telemetry inherited from FB88.

This remains private diagnostic evidence.  It does not change public dispatch,
does not add QKE, does not repair terminal abundances, and does not claim
full-BBN completion.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_trace_span_growth.py \
  --output diagnostic_outputs/fb89_continuous_ap65_trace_span_growth.json \
  --start-N-span-end 5e-9 \
  --span-growth-factor 2 \
  --span-rows 3 \
  --h-max 1e-9 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference \
  --max-steps 96
```

Output:

- contract `augmented_continuous_ap65_trace_span_growth_fb89_v1`
- artifact payload SHA256 `77a9d8a0dab4ef5b140622fb26e87860877059eab9ea3acb25e0ef068b1ab057`
- manifest file SHA256 `bc352e714afee26c01bfbc71298719dfd2fe91c063a27c11b03ce2427e53f9b2`
- `classification=trace_span_growth_all_requested_spans_passed`
- nested FB88 classification `trace_boundary_extension_all_requested_spans_passed`
- largest passing endpoint: `4e-08`
- requested span rows: `3`
- best `T_final_MeV=0.7999999685712307`
- conservation max: `7.782547616453054e-18`
- conservation limit: `1e-16`
- complete conservation/solver/stiffness rows: `3` / `3` / `3`
- step count total: `40`
- attempt count total: `40`
- rejected steps total: `0`
- `error_norm_max=0.0006361033936367059`
- source evaluations total: `2326`
- stage source evaluations total: `280`

## Interpretation

The trace-boundary policy remains stable over a geometric hot-endpoint scout
from the FB88 baseline `5e-9` through `4e-8`.  The run still ends near
`0.8 MeV`, so it is not a physical full-BBN result.  The useful conclusion is
narrower: the positivity-preserving trace-boundary phase-2 treatment continues
to avoid the prior strict `Y_p` sign failure over a larger tiny-span ladder, with
row-complete telemetry and no rejected Rodas5P host steps.

## Claim Boundary

Allowed claim:

- The private trace-boundary continuous-AP65 geometric scout passes through
  `N_span_end=4e-8` with complete conservation, stiffness, and solver-effort
  telemetry recorded.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
