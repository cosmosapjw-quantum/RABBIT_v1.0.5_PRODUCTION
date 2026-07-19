# FB79 Continuous AP65 Span Bracket

Date: 2026-05-20

## Scope

FB79 wraps the private FB70 continuous-AP65 chained span ladder in a
profile-level bracket artifact.  It runs multiple chained span profiles with
the same physics and solver controls, preserves each nested FB70 summary and
failure region, and records the last passing profile plus the first observed
failing endpoint.

This is diagnostic bracket evidence only.  It does not reroute public
CPU-JAX/Rodas5P dispatch, change `canonical_forward_solver`, run SMC, claim
production SMC validation, add QKE, or make the all-freedom full-BBN path
publication-ready.

## Real Current Artifact

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_continuous_ap65_span_bracket.py \
  --output diagnostic_outputs/fb79_continuous_ap65_span_bracket.json \
  --profile fb78_smoke_pass_cap=5e-11,1e-10,2e-10,5e-10@5e-11 \
  --profile extended_failure_probe=1e-9,3e-9,1e-8@1e-9 \
  --q-nodes 0.5,1.5,3.0 \
  --q-energy-weights 0.02,0.05,0.10 \
  --jacobian-policy finite_difference
```

Output:

- `contract=augmented_continuous_ap65_span_bracket_fb79_v1`
- `artifact_payload_sha256=e1c73bdae84d013a3ac0551bff404716f78bf3fdcd37a337326f3f5740e8df35`
- manifest file SHA256 `2cb311b2779809db259d0574b26c1def21057ebbe152ddb97711fdf82d260557`
- `passed=true`
- `bracket_status=pass_fail_bracketed`
- `last_passing_profile_name=fb78_smoke_pass_cap`
- `largest_passing_N_span_end=5e-10`
- `best_passing_T_final_MeV=0.799999999607141`
- `first_failing_profile_name=extended_failure_probe`
- `first_failing_N_span_end=1e-09`
- `first_failing_T_final_MeV=0.7999999992142808`
- `physical_full_bbn_span_ready=false`
- `profiles_passed=1`
- `profiles_failed=1`

The failing profile records nested FB70 violations
`fb70_bbn_observable_physical_bounds_failed` and `fb70_span_rows_failed`; its
first failing nested region is still above the full-BBN endpoint and carries raw
`Yp_nonpositive` evidence.  This brackets the current continuous-AP65 chained
surface between a passing `5e-10` cap and a failing `1e-9` first endpoint.

## Claim Boundary

Allowed claim:

- The current private chained continuous-AP65 surface has repeatable bracket
  evidence: tiny chained profiles pass through `N=5e-10`, while the next
  extended profile fails at `N=1e-9` with raw nonpositive `Y_p`.

Forbidden claims:

- Full-BBN completion below `0.01 MeV`.
- Public or canonical dispatch support.
- Production SMC validation.
- QKE support.
- Publication-ready all-freedom full-BBN support.
