# BD445 Payload Provenance Host-dA Fast Path Negative Ablation

Date: 2026-06-10

Scope: augmented Type-I PSTF no-QKE AP65 q4 performance.  QKE remains out of
scope.  This was an implementation hypothesis test, not a solver validation or
promotion claim.

## Hypothesis

BD444 showed that payload provenance accounted for 353.01989179180237 seconds
of q4 wall.  The tested hypothesis was:

> `_collision_payload_provenance_payload(...)` is expensive because it calls
> `jax.device_get(collision.dA_modes)` even though dynamic payloads already carry
> host-side JSON-safe `dA_modes`; preferring payload `dA_modes` should avoid a
> device synchronization and reduce provenance wall.

## Local Patch Tested

The working-tree patch, not retained, changed
`src/rabbit/jax/augmented_typeI_replay.py` so
`_collision_payload_provenance_payload(...)` tried to read finite 3D
`collision_source["dA_modes"]` first and fell back to
`jax.device_get(collision.dA_modes)` only when host-side payload data was absent
or shape-invalid.

A temporary TDD regression test asserted that provenance could be produced
without calling `jax.device_get` when payload `dA_modes` was present.

## Tests

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_jax_augmented_typeI_replay.py::test_live_source_collision_payload_provenance_prefers_payload_dA_modes
```

RED before implementation: failed because `_collision_payload_provenance_payload`
called `jax.device_get(collision.dA_modes)`.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_jax_augmented_typeI_replay.py::test_live_source_collision_payload_provenance_prefers_payload_dA_modes \
  tests/test_jax_augmented_typeI_replay.py::test_live_source_collision_payload_fingerprint_uses_full_dA_modes \
  tests/test_jax_augmented_typeI_replay.py::test_live_source_collision_payload_fingerprint_ignores_wall_timer_summary
```

Result: 3 passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_boundary_trace_policy_suppresses_inner_loop_trace_without_changing_outputs \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd392_collision_relax_policy_uses_rhs_only_jax_for_stage_rhs --tb=short
```

Result: 2 passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_jax_augmented_typeI_replay.py --tb=short
```

Result: 72 passed.

## Full q4 Run

Path:

`diagnostic_outputs/bd445_q4_payload_provenance_host_dA_fast_path/`

Command:

`diagnostic_outputs/bd445_q4_payload_provenance_host_dA_fast_path/bd445_command.txt`

Result:

- process exit: 0
- `/usr/bin/time` elapsed: 49:16.41
- max RSS: 3,140,684 KB
- final JSON size: 28,192,458 bytes
- component checker: `PASS component wall attribution`
- `summary.execution_passed`: true
- failed or exception rows: 0
- PR-B blocker status:
  `passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
- controlled FLRW LRS/non-LRS `N_eff_3T` delta:
  7.701442438445838e-06
- controlled floor margin:
  0.034808717967143465

## BD444 vs BD445

| Metric | BD444 baseline | BD445 host-dA fast path | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | 48:47.31 | 49:16.41 | +29.10 s |
| component total wall | 2863.7824760479853 | 2892.876630565035 | +29.0941545170499 s |
| attributed wall | 2201.5930251205573 | 2230.027230124222 | +28.43420500366472 s |
| residual unattributed | 662.189450927428 | 662.849400440813 | +0.6599495133850148 s |
| source evaluation | 1170.6737594259903 | 1175.4818956335075 | +4.808136207517199 s |
| payload build | 689.9324971504393 | 695.211660287634 | +5.279163137194654 s |
| payload provenance | 353.01989179180237 | 352.23022159037646 | -0.7896702014259105 s |
| payload trace | 61.58063135470729 | 61.85262076015351 | +0.27198940544621884 s |
| phase-2 corrector | 1342.248983921425 | 1363.9694785784814 | +21.720494657056407 s |
| host Jacobian | 162.60417895158753 | 163.99408676300664 | +1.3899078114191087 s |

Endpoint observables matched the baseline family:

| Freedom key | T_gamma MeV | N_eff_3T | Yp | D/H | Sigma_H | W/J |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `weak_rate_corrections+neutrino_collision_terms` | 0.009139616879151824 | 3.034801016530264 | 0.24200019137645587 | 2.4929358404512697e-05 | 5.516438318504755e-31 | [62, 62] |
| `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | 0.00913961404501975 | 3.0348087179727026 | 0.24201652194550552 | 2.493028169465174e-05 | 3.3286755172789884e-31 | [62, 62] |

## Verdict

CONTRADICTED as a q4 speedup.  The patch preserved endpoint behavior but did not
move the provenance wall in a meaningful way.  Payload provenance changed by
less than one second while total wall worsened by about 29 seconds, driven
mostly by phase-2 noise/regression.

The code change was therefore not retained.  The useful result is negative
evidence: the large payload-provenance bucket is not explained by a simple
avoidable `jax.device_get(collision.dA_modes)` call on this q4 workload.

## Consequence

Do not pursue the host-dA provenance shortcut as PR material.  The next
performance work should target:

1. phase-2 corrector/step-attempt wall;
2. payload build/provider work;
3. the remaining exclusive residual.
