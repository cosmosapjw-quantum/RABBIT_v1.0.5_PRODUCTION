# PR-R — Release Gate (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

All prior PRs (PR-A, PR-J, PR-N1, PR-N2, PR-D, PR-G, PR-T3A,
PR-T3B, PR-T3C, PR-T3D) have delivered code, tests, docs, and
audit trails.  PR-R is the **closing PR** for the roadmap: it
runs the canonical acceptance checklist, fixes any residual doc
drift, clears the four pre-existing red tests (or declares each
formally deferred with justification), and tags the release
commit.

Invariants: everything prior holds.

Dependencies: all eleven preceding PRs.

---

## 1. Phase objective

1. Run the roadmap-wide acceptance checklist.
2. Resolve the four pre-existing red tests documented in
   [ROADMAP_STATE_OF_RECORD.md §6.1](../ROADMAP_STATE_OF_RECORD.md#61-known-pre-existing-red-tests-not-caused-by-recent-work):
   - `test_supported_capabilities_mentions_features` —
     add the "Inference" keyword to
     `SUPPORTED_CAPABILITIES.md`.
   - `test_classB_typeV_bbn_gold` — update the Class B Type V
     gold fixture (if the class-B team has shipped its own
     correction; otherwise formally declare it deferred outside
     the Type I roadmap).
   - `test_jax_flrw_gold` — **test-side bug**: swap the comparison
     fixture from `jax_flrw` to `jax_flrw_equilibrium`.
   - `test_anisotropy_signal_parity` — already resolved by PR-D;
     confirm.
3. Freeze the release tag.
4. Bring `STATUS.md`, `README.md`,
   `PROMOTION_GATES.md`,
   `docs/BACKEND_CAPABILITY_MATRIX.md` in line with the new
   canonical tiers.
5. Refresh the cross-code fixture freeze for reproducibility.

---

## 2. Literature anchors

### 2.1 Internal
- All `ROADMAP_*.md` documents.
- All `docs/audit/PR-*.md` reports (must be present and complete).
- Every topic guide in `docs/`.

### 2.2 External
None new.

### 2.3 Paper-equation cross-check
Not applicable.

---

## 3. Skeleton diff

### 3.1 Fix `test_supported_capabilities_mentions_features`

```diff
 # SUPPORTED_CAPABILITIES.md
+## Inference
+
+The `rabbit.inference` subpackage exposes the ``canonical_forward_solver``
+dispatch described in ``ROADMAP_STATE_OF_RECORD.md §2.4`` and the NUTS
+/ nested-sampling helpers.  Inference is tagged as ``candidate``
+(validation_mode) pending the cross-code PE lock.
```

### 3.2 Fix `test_jax_flrw_gold`

```diff
 # tests/test_production_gates.py
     def test_jax_flrw_gold(self):
         gold = json.load(open("tests/fixtures/jax_bbn_gold.json"))
         ...
-        gold_Yp = gold["jax_flrw"]["Yp"]
+        # With use_live_weak_monopoles=False the equilibrium-FD path
+        # is used; compare to the corresponding gold entry.
+        gold_Yp = gold["jax_flrw_equilibrium"]["Yp"]
         rel = abs(r.Yp - gold_Yp) / gold_Yp
         assert rel < 1e-4, f"JAX FLRW gold mismatch: {r.Yp:.8f} vs {gold_Yp:.8f}"
```

### 3.3 Class B Type V gold

Either (a) update the Class B gold fixture after confirming the
class-B team has shipped a correction, or (b) mark
`test_classB_typeV_bbn_gold` as
`@pytest.mark.xfail(reason="Class B Type V gold deferred: owned by the Class A/B roadmap, see docs/CLASSB_PROMOTION_PACKET.md")`
and reference the Class B packet.

### 3.4 Tag the release

```bash
git tag -a rabbit-typeI-tier3-v1 -m "Type I tier-3 full-collision release"
```

---

## 4. WBS

1. **Full test sweep** — `pytest tests/ --tb=line`.  Record the
   baseline.
2. **Pre-existing red #1**: `SUPPORTED_CAPABILITIES.md` update.
3. **Pre-existing red #2**: Class B decision (fix or xfail).
4. **Pre-existing red #3**: fixture swap in the gold test.
5. **Pre-existing red #4**: confirm green after PR-D.
6. **Cross-code fixture freeze** — ensure
   `tests/fixtures/tier3_cross_code.json` is explicitly tagged
   with a release-date comment at the top of the file.
7. **Doc refresh**:
   - `STATUS.md` — regenerate canonical tier table.
   - `README.md` — capability table.
   - `PROMOTION_GATES.md` — all 10 PRs listed.
   - `docs/BACKEND_CAPABILITY_MATRIX.md` — 10 backend keys.
8. **Tag the commit.**

---

## 5. Three-stage verification

### Stage 1 — Internal
- Audit every `PR-*.md` in `docs/audit/` for completeness.
- Confirm every backend in `CAPABILITY_BY_BACKEND` has a
  matching doc entry.
- Walk the dependency graph in
  [ROADMAP_PR_WBS.md §0](../ROADMAP_PR_WBS.md#0-dependency-graph)
  and verify every prior PR landed.

### Stage 2 — External
- Not required for PR-R itself, but re-verify that the tier-3
  cross-code fixture numbers match the published sources (could
  have been updated by LASAGNA / FortEPiaNO in the interim).

### Stage 3 — Self CoT
- **Regression count.**  `pytest tests/ -q` pass count must be
  `prior_baseline + new_tests` modulo the fixed reds.  Compute
  the expected number and compare.
- **Doc round-trip.**  Pick three random claims from
  `STATE_OF_RECORD.md` and verify each still holds by grepping
  the code.
- **Release tag.**  Verify no uncommitted changes remain.

Record in `docs/audit/PR-R_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] Full test sweep passes modulo the documented four reds
      now resolved.
- [ ] All four pre-existing reds: green or formally xfail with
      justification.
- [ ] Every `ROADMAP_*.md` updated to the final state.
- [ ] Every topic guide's status table transitioned to
      "delivered".
- [ ] Cross-code fixture frozen.
- [ ] Release tag applied.

---

## 7. Adversarial audit prompt

> Audit PR-R (release gate).  Verify: (1) every PR in the roadmap
> has an entry in `docs/audit/` and `ROADMAP_PR_CATALOG.md`; (2)
> every backend key has a capability + doc entry; (3) the four
> pre-existing reds from the original baseline are either green
> or formally `xfail`-ed with a linked rationale; (4) test-suite
> pass count has not silently dropped from the published tally.
> Cap 300 words.

---

## 8. Anti-local-minimum reminders

1. **Do not** skip the third pre-existing red fix by relaxing the
   test tolerance.  It is a fixture-side bug and the fix is
   trivial.
2. **Do not** over-promise tier-4 / QKE in the release notes.
3. **Do not** merge with any `TODO`, `FIXME`, or `placeholder`
   remaining in the tier-3 code path.

---

## 9. Hallucination prevention

- Do not claim "all tests passing" without pasting the pytest
  summary line in the audit doc.
- Do not tag the release without the staged release notes and
  the clean tree.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
Final sweep:
- §2 component status: everything marked "Production" or
  "Candidate" with current reality.
- §4 parity tables: include the tier-3 cross-code row.
- §6 test count: update.
- §6.1 pre-existing reds: flip to "resolved in PR-R" or "deferred,
  see ...".

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-R entry with:
- Full changelog of resolved reds.
- Release tag name.
- Final test pass count.
- Cross-code fixture hash / date stamp.

### 10.3 Root docs
- `STATUS.md`, `README.md`, `PROMOTION_GATES.md`,
  `docs/BACKEND_CAPABILITY_MATRIX.md`: refreshed.

### 10.4 Topic guides
Every implementation guide's "what has been achieved" table
flipped to reflect the completed roadmap.

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

# 1. Full sweep baseline
pytest tests/ --tb=no -q | tee /tmp/release_baseline.log

# 2. Confirm every PR audit exists
for PR in A J N1 N2 D G T3A T3B T3C T3D; do
    test -f docs/audit/PR-${PR}.md
done

# 3. Verify zero TODO / FIXME / placeholder in tier-3 code
! grep -E "TODO|FIXME|placeholder" -r src/rabbit/jax/driver_typeI_full_boltzmann.py src/rabbit/jax/q_advection_jax.py src/rabbit/jax/collisions_jax.py

# 4. Stage and commit
test -f docs/audit/PR-R.md
git add docs/audit/PR-R*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-R: release gate — roadmap closed, tier-3 tagged

Closes the Type I + tier-3 roadmap (PR-A, PR-J, PR-N1, PR-N2,
PR-D, PR-G, PR-T3A, PR-T3B, PR-T3C, PR-T3D).  Resolves the four
pre-existing red tests:
  - test_supported_capabilities_mentions_features (doc fix)
  - test_jax_flrw_gold (fixture swap)
  - test_anisotropy_signal_parity (already green after PR-D)
  - test_classB_typeV_bbn_gold (deferred to the Class A/B roadmap)
STATUS, README, PROMOTION_GATES, BACKEND_CAPABILITY_MATRIX all
refreshed.  Cross-code tier-3 fixture frozen.  Release tag applied.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# 5. Tag
git tag -a rabbit-typeI-tier3-v1 -m "Type I tier-3 full-collision release"

git status
```

---

## 12. Abort conditions

- A PR in the roadmap is missing from `docs/audit/` or
  `ROADMAP_PR_CATALOG.md`.
- Test pass count has silently dropped from the catalogue's
  baseline.
- Any `TODO` / `FIXME` remains in the tier-3 code path.
- Any backend key in `CAPABILITY_BY_BACKEND` has no doc entry.

Abort → `docs/audit/PR-R_abort.md`.
