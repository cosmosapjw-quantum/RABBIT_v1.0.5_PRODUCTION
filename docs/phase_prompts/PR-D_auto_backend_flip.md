# PR-D — Auto-Backend Promotion to Characteristic (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

`canonical_forward_solver(backend="auto")` today resolves to
`jax_typeI_liveweak_cl3_tier1` (linearised PSTF, live weak).  That
path captures only ~21 %% of the shear-induced Y_p signal that the
characteristic-ray method captures (paper §6.8).  One pre-existing
red test (`test_anisotropy_signal_parity`) formalises the gap:
SciPy `auto` → char; JAX `auto` → linearised.

Invariants: Rodas5P, CPU-preferred, float64, publication parity
5e-5.  PR-CHAR and PR-T3T have delivered publication-grade JAX
characteristic paths for tier-1 and tier-2.

Dependencies: none.  Trivial config change.

---

## 1. Phase objective

Flip `auto` → `jax_typeI_characteristic_tier1` (tier-1) or
`jax_typeI_characteristic_tier2` (tier-2, behind `jax_thermo_tier=2`).
Keep the explicit `jax` key pointing at linearised for backward
compatibility with any downstream consumer that relied on the
previous behaviour.

Acceptance: `test_anisotropy_signal_parity` and
`test_cross_backend_regression::test_direction_agreement` flip to
green (or are already green thanks to the matched-physics rewrite
in PR-T3T; verify).

---

## 2. Literature anchors

### 2.1 Internal
- `src/rabbit/config/backend_capabilities.py` —
  `CAPABILITY_BY_BACKEND`.
- `src/rabbit/inference/forward_likelihood.py` —
  `resolve_typeI_auto_backend` (or equivalent helper), and the
  `auto` branch of `canonical_forward_solver`.
- `STATUS.md`, `README.md`, `docs/BACKEND_CAPABILITY_MATRIX.md` —
  documented dispatch table.
- `docs/ROADMAP_STATE_OF_RECORD.md §2.4` — backend dispatch rows.

### 2.2 External
None.  Pure configuration.

### 2.3 Paper-equation cross-check
- [ ] Paper §6.8: explicit statement that linearised PSTF recovers
      only ~20–30 %% of the characteristic nonlinear correction.
      Confirm by reading page 18.

---

## 3. Skeleton diff

### 3.1 `backend_capabilities.py`

```python
# before
CAPABILITY_BY_BACKEND = {
    "jax": JAX_TYPEI_LIVEWEAK_CL3_TIER1,
    "jax_advanced": JAX_TYPEI_TIER3_WEAK_BUDGET,
    "jax_characteristic": JAX_TYPEI_CHARACTERISTIC_TIER1,
    "jax_characteristic_tier2": JAX_TYPEI_CHARACTERISTIC_TIER2,
    ...
    "auto": JAX_TYPEI_LIVEWEAK_CL3_TIER1,
}

# after
CAPABILITY_BY_BACKEND = {
    "jax": JAX_TYPEI_LIVEWEAK_CL3_TIER1,                # unchanged
    "jax_advanced": JAX_TYPEI_TIER3_WEAK_BUDGET,        # unchanged
    "jax_characteristic": JAX_TYPEI_CHARACTERISTIC_TIER1,
    "jax_characteristic_tier2": JAX_TYPEI_CHARACTERISTIC_TIER2,
    ...
    "auto": JAX_TYPEI_CHARACTERISTIC_TIER1,             # flipped
}
```

### 3.2 `resolve_typeI_auto_backend` (or equivalent)

Update the auto-resolution logic to dispatch to
`jax_characteristic_tier2` when `jax_thermo_tier=2` is supplied
and to `jax_characteristic` otherwise.  Preserve existing
fall-through logic for `enable_teff`, `bianchi_type`, non-LRS
shear (re-routes to PSTF if the char path does not support the
request).

```python
def resolve_typeI_auto_backend(
    *, N_q, correction_level, enable_teff, enable_collisions,
    tier, jax_thermo_tier, bianchi_type, Sigma_H_minus,
    N1_init, N2_init, N3_init, A_init, v0,
):
    # Route curved Class A/B to their dedicated backends.
    if bianchi_type != "TYPE_I":
        return "jax_classA", {"reason": "non-Type-I bianchi"}
    if any(x != 0 for x in (N1_init, N2_init, N3_init, A_init)):
        return "jax_classA", {"reason": "Class A curvature"}
    if v0 > 0:
        return "jax_tilted", {"reason": "tilted velocity"}
    # Non-LRS currently handled by PSTF auto fallback until PR-N2
    # lifts this guard.
    if abs(Sigma_H_minus) > 0:
        return "jax", {"reason": "non-LRS, char LRS-only"}
    # Teff is unsupported on the char path.
    if enable_teff:
        return "jax_advanced", {"reason": "Teff candidate surface"}
    # Collisions (Mangano momentum-averaged) handled by char-tier-2.
    if int(jax_thermo_tier) >= 2 or enable_collisions:
        return "jax_characteristic_tier2", {"reason": "tier-2 3T"}
    # Default: char tier-1.
    return "jax_characteristic", {"reason": "canonical Type-I publication path"}
```

### 3.3 Documentation

- `STATUS.md`: update the `auto → ...` dispatch table.
- `README.md`: same.
- `docs/BACKEND_CAPABILITY_MATRIX.md`: same.

---

## 4. WBS

1. Update `CAPABILITY_BY_BACKEND["auto"]`.
2. Update `resolve_typeI_auto_backend`.
3. Update `STATUS.md` / `README.md` / `BACKEND_CAPABILITY_MATRIX.md`.
4. Rerun `test_registry_sync.py` (the doc-sync tests); fix any
   string-drift triggered by the flip.
5. Rerun `test_production_gates.py::test_anisotropy_signal_parity`
   and confirm it now passes.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Grep the repo for hard-coded `jax_typeI_liveweak_cl3_tier1`
  strings outside `backend_capabilities.py` — any consumer that
  expects the old `auto` mapping.
- Read `test_registry_sync.py` to understand which doc strings the
  sync tests check.

### Stage 2 — External
Not applicable (pure config).

### Stage 3 — Self CoT
- Enumerate all downstream consumers of
  `canonical_forward_solver(backend="auto")`.  Are there any
  regression-gold fixtures that hard-code the old auto
  `Y_p` value?  If yes, update them to the new matched-physics
  value.
- Confirm that the pre-existing red tests documented in
  `ROADMAP_STATE_OF_RECORD.md §6.1` — especially #4 — are now
  green.
- If any **new** test turns red, it is almost certainly a fixture
  mismatch; decide whether to update the fixture or roll back.

Record in `docs/audit/PR-D_stage{1,2,3}.md`.

---

## 6. Self-audit checklist

- [ ] `auto` resolves to `jax_characteristic_tier1` (verify via
      `CAPABILITY_BY_BACKEND["auto"].key`).
- [ ] `test_anisotropy_signal_parity` green.
- [ ] `test_direction_agreement` green (already passes post-PR-T3T).
- [ ] All doc-sync tests green (strings aligned).
- [ ] No new test failure.

---

## 7. Adversarial audit prompt

> Audit PR-D (auto backend flip).  Verify:
> (1) `CAPABILITY_BY_BACKEND["auto"]` points to the char-tier-1
> capability;
> (2) `test_anisotropy_signal_parity` now passes;
> (3) `STATUS.md`, `README.md`, `BACKEND_CAPABILITY_MATRIX.md` all
> state the new mapping consistently (no stale `liveweak_cl3_tier1`
> references remain for `auto`);
> (4) downstream fixtures that depended on the old `auto` have been
> updated, not ignored.  Cap 200 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** delete the `jax` key.  It must continue to point at
   the linearised PSTF path so downstream callers that explicitly
   requested linearised behaviour still receive it.
2. **Do not** silently shift any regression-gold fixtures without
   documenting them in `PR_CATALOG.md`.
3. **Do not** adjust test tolerances to paper over a physics-path
   flip — the whole point is that `auto` now does more physics.

---

## 9. Hallucination prevention
- Never claim a test passes without running pytest and quoting the
  exit status.
- Never assume `resolve_typeI_auto_backend` exists — grep the repo
  for the actual helper name and call signature; adapt the
  skeleton above accordingly.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md §2.4`
Flip the `auto` row from
`JAX_TYPEI_LIVEWEAK_CL3_TIER1` to `JAX_TYPEI_CHARACTERISTIC_TIER1`.
Add a one-line rationale citing paper §6.8.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-D entry; note which pre-existing red tests flipped green.

### 10.3 Root-level docs
- `STATUS.md`
- `README.md`
- `docs/BACKEND_CAPABILITY_MATRIX.md`

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v
pytest tests/test_inference_backend_propagation.py -v
pytest tests/test_registry_sync.py -v
pytest tests/test_production_gates.py::TestGateP15_ReleaseParity -v

test -f docs/audit/PR-D.md
git add docs/audit/PR-D*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-D: promote auto backend to characteristic-ray path

Flips CAPABILITY_BY_BACKEND["auto"] from JAX_TYPEI_LIVEWEAK_CL3_TIER1
(linearised PSTF) to JAX_TYPEI_CHARACTERISTIC_TIER1 so that the
canonical_forward_solver default delivers publication-grade
nonperturbative characteristic-ray BBN, closing the ~21% signal-
recovery gap documented in paper §6.8.  `jax` backend key remains
on linearised PSTF for backward compatibility.  Tier-2 auto
dispatch now routes to jax_characteristic_tier2 when
jax_thermo_tier=2.  STATUS.md, README.md, BACKEND_CAPABILITY_MATRIX.md
updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Any regression-gold fixture breaks and cannot be cleanly updated.
- Doc-sync tests uncover stale references that require touching
  unrelated files.
- A downstream consumer (inference pipeline, SMC runner, etc.)
  depends on the old `auto` behaviour in a way that cannot be
  resolved in this PR.

Abort → `docs/audit/PR-D_abort.md`.
