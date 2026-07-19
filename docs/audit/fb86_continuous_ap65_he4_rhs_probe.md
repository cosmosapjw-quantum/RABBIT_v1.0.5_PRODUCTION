# FB86 Continuous AP65 He4 RHS Boundary Probe

Date: 2026-05-20

## Scope

FB86 probes the first continuous-AP65 strict-`Y_p` sign failure at the
phase-2 nuclear network RHS boundary.  It consumes the FB82/FB85 first-failure
artifact, reads the raw terminal and last-attempted `X_phase2` tails, and
evaluates the JAX phase-2 network RHS at:

- last passing terminal state
- last passing last-attempted state
- first failure last-attempted state
- first failure terminal state
- first failure terminal state with `He4=0`
- first failure terminal state with `He4=1e-30`
- diagnostic-only nonnegative trace-species counterfactuals

The nonnegative trace counterfactual floors only trace-species indices, requires
the terminal observable `Y_p` to match the final-state `He4` tail before making
the localization claim, and is not applied to evolution or used as an abundance
repair.  It only identifies whether negative trace intermediates are feeding an
unphysical negative boundary RHS for `He4`.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_he4_rhs_probe.py \
  --output diagnostic_outputs/fb86_continuous_ap65_he4_rhs_probe.json \
  --N-span-end-ladder 5e-10,1e-9,1.5e-9,2e-9 \
  --h-max 2.5e-10 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- contract `augmented_continuous_ap65_he4_rhs_probe_fb86_v1`
- artifact payload SHA256 `ebd16b1fa3b6d4b673e33c2cff07855a075d3ca7f288230ccf2b9fe24b275fdf`
- manifest file SHA256 `b170d117620b40dcf413e94b865774b3a6f0c4dc12b2e52d8568130cf3baf201`
- `classification=he4_boundary_negative_due_to_negative_trace_intermediates`
- `first_failure_Yp=-1.2294890184644955e-30`
- `first_failure_N_span=[1e-09, 1.5e-09]`
- `first_failure_T_final_MeV=0.7999999988214215`
- `first_failure_negative_trace_indices=[3,4,6,7]`
- `first_failure_negative_core_non_he4_indices=[]`
- `first_failure_terminal_dHe4_network_rhs=-2.618301171321943e-21`
- `first_failure_he4_zero_dHe4_network_rhs=-2.618301171321943e-21`
- `first_failure_nonnegative_trace_he4_zero_dHe4_network_rhs=7.273403769914826e-286`

## Interpretation

The first strict-`Y_p` failure is not merely a final readout issue or an
accepted-step error-control issue.  The raw terminal and last-attempted states
already contain negative trace intermediates, and the JAX phase-2 network RHS
uses those raw concentrations strongly enough that `dHe4` remains negative
even at `He4=0`.

When the trace intermediates are replaced by nonnegative values in a
diagnostic-only counterfactual, `dHe4` at the same boundary becomes
nonnegative.  The next implementation target is therefore
positivity-preserving phase-2 network evolution for trace species, not
post-hoc output truncation.

## Claim Boundary

Allowed claim:

- The current continuous-AP65 first `Y_p` sign failure is localized to a
  negative trace-intermediate RHS boundary effect in the phase-2 network.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
- Physical correctness by truncating, flooring, or otherwise repairing `Y_p`.
