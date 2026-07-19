# BD621 — scipy tier-2/tier-3 canonical-promotion audit: BLOCKED (internal-consistency failure)

Date: 2026-07-09

Status: **NOT PROMOTED — internal-consistency gate failed.** Also fixes 5
BD619 auto-default test regressions missed in that commit.

## Task

With scipy the sole active-canonical forward line (BD620), promote the scipy
Type-I tier-2 (per-species) and tier-3 (weak-budget) surfaces from `candidate`
to `canonical`. Verification standard (per user): physics is derived,
code-level implementation was never completed, no external comparison
literature — so internal consistency is the bar.

## Finding: promotion BLOCKED

`canonical` in this repo means "regression-locked reference or bounded promoted
surface". The scipy tier-2/tier-3 surfaces do NOT meet even the internal-
consistency bar: **their anisotropic-residual collision closure is not
resolution-converged.** This matches the code's own honest self-assessment
(the tier-2/tier-3 capability notes call them "transitional candidate ... pending
the full anisotropic-residual refactor").

Measured (scipy, Σ_H=0.03, CL3, tier=2, collisions on; single process, ~0.1 GB):

| N_q | Y_p | step \|ΔY_p/Y_p\| |
|---:|---:|---:|
| 16 | 0.24396398 | — |
| 20 | 0.24052191 | 1.43e-2 |
| 24 | 0.23934856 | 4.90e-3 |
| 32 | 0.23537249 | 1.69e-2 |

Y_p drifts ~3.5% from N_q=16→32 and does NOT settle — the N_q=32 step (1.69%)
is larger than the N_q=24 step (0.49%), i.e. no convergence (possibly
oscillatory/divergent). A canonical BBN surface needs Y_p stable to ~1e-4; this
is 100-350× worse. The derived N_eff is likewise resolution-dependent (2.95 at
N_q=16 → 2.65 at N_q=24) with a large gap to the parametric input (3.044) — the
solver itself warns on it.

Conclusion: promoting these surfaces to canonical would be a dishonest "landed"
over-claim of exactly the kind BD612 deflated. They correctly remain
`candidate`. The genuine precondition for promotion is making the residual
collision closure resolution-stable — an open physics-implementation task, not
a tier-field flip.

## What was committed

1. **Falsification lock** (`tests/test_scipy_tier2_convergence_lock.py`,
   `@slow @xfail(strict)`): asserts the N_q=24→32 Y_p convergence a canonical
   surface must have (<1e-3). It xfails today and will flip to a strict-XPASS
   failure the moment the residual closure is made convergent — signalling that
   promotion is now possible. (BD612 falsification-lock-first pattern.)
2. **No tier change** — scipy tier-2/tier-3 stay `candidate`;
   `test_scipy_capability_split_lock.py::test_scipy_subsurface_capabilities_are_candidate_not_canonical`
   stays green unchanged.
3. **BD619 regression fixes** (missed in BD619, caught here): five auto-default
   assertions across `test_scipy_capability_split_lock.py`,
   `test_inference_hierarchy_lock.py` (HIERARCHY table + test_auto_is_scipy +
   test_auto_equals_scipy_reference + test_no_candidate_is_auto), and
   `test_jax_advanced_backend.py` still asserted `auto → jax_characteristic`;
   updated to `auto → scipy_typeI_reference` (BD619 flip). BD619's grep-based
   test sweep matched only the reason-string / `== "jax_characteristic"`
   patterns and missed the `.key ==`/`.backend ==`/HIERARCHY-table forms.

## Verification

```
pytest tests/test_scipy_tier2_convergence_lock.py           -> 1 xfailed (lock active)
pytest tests/test_scipy_tier2_convergence_lock.py --runxfail -> 1 failed (convergence assertion fails, as documented)
pytest tests/test_scipy_capability_split_lock.py tests/test_jax_advanced_backend.py \
       tests/test_inference_hierarchy_lock.py -m "not slow"  -> all pass (BD619 regressions fixed)
```

## Cost line

- added_lines: ~50 (convergence lock + note); ~6 test-assertion fixes
- deleted_lines: ~6
- files_touched: 4 tests (1 new lock + 3 BD619-fix) + 1 note
- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes (tier-2/3 promotion question answered with evidence:
  BLOCKED on residual-closure resolution convergence; falsification-locked;
  BD619 regressions repaired)
- blocker_movement_ratio: 0.3
- validation_strengthened: yes (convergence falsification lock + regression repair)
- cost_effectiveness_verdict: ACCEPT
