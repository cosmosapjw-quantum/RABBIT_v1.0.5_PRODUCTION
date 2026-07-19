# BD613 Solver Drift Lock

Date: 2026-07-08

## Problem

The production Type-I driver silently remapped the declared production
solver method (Radau, per `PRODUCTION_CONFIG`) to BDF, at two independent
sites, with zero recorded rationale:

- **Site A1** — `src/rabbit/drivers/full_coupled_typeI.py:105` (pre-fix):
  `_PRODUCTION_BDF_CONFIG = replace(PRODUCTION_CONFIG, method=SolverMethod.BDF)`.
- **Site A2** — `src/rabbit/drivers/full_coupled_typeI.py:1197` (pre-fix):
  `effective_solver = _PRODUCTION_BDF_CONFIG if (config.solver is PRODUCTION_CONFIG) else config.solver`,
  an identity check on the module-level `PRODUCTION_CONFIG` singleton that
  fires only for the default config.
- **Site B** — `src/rabbit/inference/forward_likelihood.py:2597-2609` (pre-fix):
  whenever a caller overrode `rtol`/`atol`, the code constructed a fresh
  `SolverConfig(method=SolverMethod.BDF, ...)`, hard-coding BDF and dropping
  whatever method had been requested.
- **Site C (asymmetry, no remap)** —
  `src/rabbit/drivers/classA_driver.py:293` (pre-fix): `__post_init__` set
  `self.solver = PRODUCTION_CONFIG` and honored it as-is, so classA actually
  ran Radau in production while the Type-I driver ran BDF under an
  identically-named "production" config.

Provenance: `git blame` on `full_coupled_typeI.py:105` resolves to commit
`692f394` ("initial commit", 2026-04-20), whose commit message carries no
body and no rationale. The remap has been present, unexplained, since the
repository's first commit. The driver metadata recorded both
`solver_method_requested` and `solver_method_effective` (line ~1515/1516)
the whole time, but nothing compared them — the drift was observable in the
metadata but never enforced or even flagged.

`src/rabbit/config/solver_config.py:89-94` (pre-fix) declared
`PRODUCTION_CONFIG = SolverConfig(method=SolverMethod.RADAU, rtol=1e-8,
atol=1e-10, max_step=0.1)` — a declaration that was false for the Type-I
production lane (site A) as long as the remap existed, and true only for
classA (site C).

## Decision (option a: honest declaration, zero runtime change)

Rather than change what actually runs, make the declaration match reality:

1. `PRODUCTION_CONFIG` now declares `method=SolverMethod.BDF` directly —
   this is what the Type-I production lane has always actually executed.
2. A new `PRODUCTION_RADAU_CONFIG` (same tolerances, `method=SolverMethod.RADAU`)
   is added as the Radau validation-oracle counterpart.
3. `classA_driver.__post_init__` now defaults to `PRODUCTION_RADAU_CONFIG`
   instead of `PRODUCTION_CONFIG`, preserving its historical
   Radau-at-production-tolerances behavior verbatim (site C is unaffected by
   the `PRODUCTION_CONFIG` method flip).
4. The hidden `_PRODUCTION_BDF_CONFIG` singleton and the identity-check remap
   at site A2 are deleted outright.
5. Site B's hard-coded `SolverConfig(method=SolverMethod.BDF, ...)`
   construction is replaced with `replace(PRODUCTION_CONFIG, rtol=..., atol=...)`,
   so an rtol/atol override inherits whatever method `PRODUCTION_CONFIG`
   declares instead of hard-coding one.

This is a pure declaration change: `REFERENCE_CONFIG` (tight-tolerance Radau
oracle) is untouched, `FAST_CONFIG` is untouched, and no tolerance, step, or
event configuration changed anywhere.

### Why this is safe: BD598 parity

BD598 measured Radau↔BDF endpoint parity directly: `|ΔY_p| = 8.25e-8`,
`D/H` relative difference `9.87e-6`. Both are far below any observational or
regression-lock threshold in this codebase, so re-declaring the Type-I
production method as BDF (matching what has run since the initial commit)
carries no physics risk. classA's Radau default is preserved unchanged, so
its behavior is bit-for-bit unaffected by this PR.

### Drift-guard design (seam + fail-loud raise)

`full_coupled_typeI.py` gets an explicit seam,
`_resolve_effective_solver(requested)`, which is identity by design:

```python
def _resolve_effective_solver(requested):
    """Identity by design (BD613). Any requested->effective divergence must be introduced
    here explicitly - the drift guard at the call site raises if the method changes."""
    return requested
```

The call site no longer performs any remap; it calls the seam and then
asserts the contract:

```python
effective_solver = _resolve_effective_solver(config.solver)
if effective_solver.method is not config.solver.method:
    raise RuntimeError(
        f"solver drift guard (BD613): requested={config.solver.method.value} "
        f"effective={effective_solver.method.value}; silent solver remaps are forbidden")
```

Any future change to the effective production method must be introduced
inside `_resolve_effective_solver` explicitly — and the moment it diverges
from what was requested, the driver fails loud instead of silently swapping
solver identity. This is the promotion gate for any future method change.
The `solver_method_requested` / `solver_method_effective` metadata keys are
unchanged in shape; they are now the drift telemetry that this guard backs
(and are always equal by construction going forward).

## Validation

```
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest tests/test_solver_drift_guard.py tests/test_solver_tolerance_envelope.py tests/test_audit_hardening_regressions.py tests/test_backend_parity.py -q
```

Result: `23 passed`.

`py_compile` on every touched source file:

```
venv/bin/python -m py_compile \
  src/rabbit/config/solver_config.py \
  src/rabbit/drivers/full_coupled_typeI.py \
  src/rabbit/inference/forward_likelihood.py \
  src/rabbit/drivers/classA_driver.py
```

Result: exit 0 (no output).

Grep checklist:

```
grep -rn "_PRODUCTION_BDF_CONFIG" src tests
```

Result: one hit — `tests/test_solver_drift_guard.py`, inside
`test_remap_singleton_deleted`'s `assert not hasattr(mod, "_PRODUCTION_BDF_CONFIG")`,
which is the intentional negative assertion proving the symbol is gone. No
hit in `src`.

```
grep -rn "is PRODUCTION_CONFIG" src
```

Result: empty. Line 1197 (the `full_coupled_typeI.py` identity-check remap)
was the only site; confirmed no others exist.

```
grep -rn "SolverMethod.BDF" src/rabbit/inference/
```

Result: empty. No hard-coded `SolverMethod.BDF` construction remains in
`rabbit.inference`.

### Runtime-behavior neutrality

A default-config small solve (`Sigma_H_plus=0.1, N_q=8, N_mu=8,
n_reactions=12, correction_level=0, tier=1, enable_teff=False, solver=None`)
was run before and after the source changes (before = 4 touched source files
reverted to the BD613-PR1a commit via a scoped `git stash push -- <paths>`,
matched back with `git stash pop`):

| Field | Before | After |
| --- | --- | --- |
| `Yp` | `0.24253007908662186` | `0.24253007908662186` |
| `DH` | `2.4898584350218947e-05` | `2.4898584350218947e-05` |
| `phase1_steps` | `232` | `232` |
| `phase2_steps` | `435` | `435` |
| `solver_method_requested` | `Radau` | `BDF` |
| `solver_method_effective` | `BDF` | `BDF` |

`Yp`, `DH`, and the step counts are identical to machine precision. The only
change is `solver_method_requested`, which now honestly reports what has
always run (`BDF`) instead of the false `Radau` declaration.

## Cost Line

- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes (silent solver-identity drift retired)
- blocker_movement_ratio: 0.10
- validation_strengthened: yes
- cost_effectiveness_verdict: ACCEPT
