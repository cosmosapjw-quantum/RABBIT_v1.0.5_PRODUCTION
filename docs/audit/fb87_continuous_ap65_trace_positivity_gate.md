# FB87 Continuous AP65 Trace-Boundary Positivity Gate

Date: 2026-05-20

## Scope

FB87 adds a private evolution-RHS positivity policy for the continuous-AP65
phase-2 abundance state.  The default raw network RHS remains available and is
not changed for public dispatch.  The new `trace_boundary` policy is opt-in and
scoped to private continuous AP65 diagnostics: it evaluates phase-2 nuclear
activities with trace species and `He4` lower-bound constrained, then enforces
an inward RHS cone at active trace/`He4` lower boundaries.

This is not output truncation.  The terminal `Y_p` readout still comes from the
raw solver state.  The gate also records raw and trace-boundary phase-2
mass-fraction sum residuals from the live RHS metadata and fails closed if the
trace-boundary residual exceeds the configured conservation limit.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_trace_positivity_gate.py \
  --output diagnostic_outputs/fb87_continuous_ap65_trace_positivity_gate.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- contract `augmented_continuous_ap65_trace_positivity_gate_fb87_v1`
- artifact payload SHA256 `dcdae7615088893f2bfbbece52620b8d81e60b1e775cb7ca8059c9d65a755276`
- manifest file SHA256 `99333c8747f006758fe9da2f0a2c8e633584a3e44a9d111999f8880d149f759d`
- `classification=trace_boundary_resolves_smoke_y_p_sign_failure_with_conservation_gate`
- raw first `Yp` failure at `N_span=[0.0,1.5e-09]`
- raw first-failure `T_final_MeV=0.7999999988214215`
- raw first-failure `Yp=-1.2294890184644993e-30`
- raw `Yp` failure rows: `2`
- trace-boundary `Yp` failure rows: `0`
- raw conservation max: `6.284872348663924e-18`
- trace-boundary conservation max: `8.110492019931864e-18`
- trace-boundary conservation limit: `1e-16`
- raw largest passing endpoint: `1e-09`
- trace-boundary largest passing endpoint: `2e-09`

## Interpretation

The FB86 diagnosis was actionable: an evolution-level trace-boundary positivity
policy removes the current smoke-ladder `Y_p` sign failure over the same tiny
span ladder without exceeding the smoke-scale mass-fraction sum residual gate.
The next blocker is no longer the immediate `1.5e-9` hot-endpoint sign crossing;
it is extending the trace-boundary ladder toward the real post-BBN endpoint
while monitoring whether conservation, stiffness, and solver effort remain
controlled at longer spans.

## Claim Boundary

Allowed claim:

- The private continuous-AP65 trace-boundary evolution policy resolves the
  current smoke-ladder `Y_p` sign failure through `N_span_end=2e-9` while
  keeping the recorded trace-boundary mass-fraction residual below `1e-16`.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by terminal `Y_p` truncation or readout repair.
