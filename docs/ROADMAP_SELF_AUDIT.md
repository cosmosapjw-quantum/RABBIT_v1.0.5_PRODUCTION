# RABBIT Per-PR Self-Audit + Documentation-Update Protocol

> **Historical protocol (PUB-00, 2026-07-12).**  This checklist remains useful
> as provenance but is no longer a mandatory gate and MUST NOT create a new
> per-PR audit/catalog surface.  Current PR closure follows the frozen spec,
> role separation, PR-body evidence, and existing claim/validation ledgers in
> `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`.

Companion documents: [ROADMAP_INDEX.md](ROADMAP_INDEX.md),
[ROADMAP_STATE_OF_RECORD.md](ROADMAP_STATE_OF_RECORD.md),
[ROADMAP_PR_WBS.md](ROADMAP_PR_WBS.md),
[ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md),
[IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md](IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md).

---

## 1.  Purpose and scope

The audit protocol enforces three invariants:

1. **No hallucinated physics.**  Every equation in the diff must
   reference a paper equation or a derivation that has been committed
   to the repo.  No "plausible-looking" formulas without provenance.
2. **No mock surrogates unflagged as such.**  Approximations must be
   marked explicitly (e.g.
   `production_authority="candidate_..."`) and must not be presented
   as exact.
3. **No silent documentation drift.**  Every code change that alters
   capability, parity, or dispatch must push a corresponding
   documentation change.  `STATE_OF_RECORD.md` and `PR_CATALOG.md`
   stay in sync with the codebase automatically via the template in
   §3.

The audit is deliberately *adversarial* — the person running it
should approach the diff as a third-party reviewer who has not seen
the development.

---

## 2.  Per-PR audit checklist

Fill the following for every PR.  The template is reproduced verbatim
in each `PR_CATALOG.md` completion record (see §4).

### 2.1 Physics correctness

- [ ] **Paper-equation provenance.**  For every new formula, cite the
      paper equation or a derivation file committed alongside.
      (Example: "Π_- from paper eq 14 generalised `|m|=2`", with a
      derivation in `docs/derivations/pi_minus.md`.)
- [ ] **No mocks / no hallucinations.**  Run a cold third-party
      reviewer pass (`Agent(subagent_type="general-purpose")` with an
      adversarial prompt is suitable) and record the verdict.
- [ ] **Unit-test coverage of new physics primitives.**  Every new
      pure function has at least one unit test with a known-good
      reference (paper limit, analytic formula, or existing SciPy
      kernel).

### 2.2 Numerical parity

- [ ] **Tier-1 char parity test grid unchanged.**  Run
      `pytest tests/test_jax_typeI_characteristic_parity.py` — all
      green (no regression on any Σ_H, CL combination).
- [ ] **Tier-2 char parity test grid unchanged.**
      `pytest tests/test_jax_typeI_characteristic_tier2.py` — all green.
- [ ] **Cross-backend regression.**
      `pytest tests/test_cross_backend_regression.py` — all green.
- [ ] **Scope-specific parity.**  Each PR has its own parity tests
      (see the PR's "Exit criteria" section of `ROADMAP_PR_WBS.md`);
      run those and confirm all pass.
- [ ] **Capture before/after numbers.**  Record parity numbers in the
      `PR_CATALOG.md` entry table.

### 2.3 Performance

- [ ] **No warm single-solve regression.**  Compare
      `rabbit.jax.driver_typeI_char.run_full_coupled_typeI_char_jax`
      warm-run time to the previous PR's baseline.  Regression
      threshold: 10 %.
- [ ] **Memory / VRAM unchanged or documented.**  If the PR touches
      device policy or JIT kernels, measure peak memory (for CPU
      solves) or peak VRAM (for GPU) and record.
- [ ] **JIT cache stability.**  Two consecutive solves with identical
      config must reuse the cached `_CHAR_RHS_CACHE` entry (verified
      by identical `id(rhs_fn)`).

### 2.4 Documentation and dispatch

- [ ] **`STATE_OF_RECORD.md` updated.**  Every affected section:
      §1 (if transport/tier scope changes), §2 (component status),
      §3 (design decisions), §4 (parity), §5 (file inventory),
      §6 (test counts).
- [ ] **`PR_CATALOG.md` entry committed** with the template from
      [ROADMAP_PR_CATALOG.md §0](ROADMAP_PR_CATALOG.md).
- [ ] **Topic guide(s) updated.**  If the PR implements a step from a
      topic guide, update the guide's status lines from "planned" to
      the PR number.
- [ ] **Augmented AP reuse contract preserved.**  For AP51-AP81 work,
      confirm the diff reuses landed PRIMAT network, weak bridge,
      augmented artifact/convergence reports, inference wrappers, SMC
      runners, and figure registry/cache surfaces unless the PR
      explicitly justifies a replacement.
- [ ] **Artifact/SMC/figure provenance.**  For AP51-AP81 work, every new
      artifact, SMC output, or figure must carry commit/config
      provenance, diagnostic/candidate labels, solver/source-policy
      metadata, and stale-artifact rejection tests.
- [ ] **Capability registry sync.**  If the PR adds a new backend or
      capability key, `CAPABILITY_BY_BACKEND` must be updated and
      `test_registry_sync.py` must pass.
- [ ] **Metadata audit.**  Every new result metadata key is documented
      either in a `_surface_scope_metadata` call or in
      `STATE_OF_RECORD.md §2.4`.

### 2.5 Test-suite integrity

- [ ] **Full test sweep.**  Run
      `pytest tests/ -m "not slow and not gpu" -q` and compare the
      pass/fail count against the baseline recorded in
      `PR_CATALOG.md`.  Any new failure must be explained (regression,
      intentional tightening, xfail flip).
- [ ] **No new `xfail`**.  `xfail` additions are treated as
      production-regression; any genuine need for `xfail` must be
      declared in the PR's risk register and land with an
      accompanying doc entry.
- [ ] **Pre-existing red tests unchanged.**  The four pre-existing
      reds documented in
      [STATE_OF_RECORD.md §6.1](ROADMAP_STATE_OF_RECORD.md#61-known-pre-existing-red-tests-not-caused-by-recent-work)
      must remain red (and no *new* test failure may be mistaken for
      one of these pre-existings).

---

## 3.  Documentation-update script

The per-PR docs update is captured as a deterministic procedure so
that the sync between `STATE_OF_RECORD.md`, `PR_CATALOG.md` and the
code can be verified mechanically.  The procedure below is intended
to be executed from the repo root.

### 3.1 Data-collection stage (run at end of PR)

Gather the inputs that the catalogue entry needs:

```bash
# Parity numbers (tier-1 + tier-2)
pytest tests/test_jax_typeI_characteristic_parity.py -v 2>&1 | tee /tmp/audit_tier1.log
pytest tests/test_jax_typeI_characteristic_tier2.py -v 2>&1 | tee /tmp/audit_tier2.log

# Full pass/fail sweep
pytest tests/ -m "not slow and not gpu" --tb=no -q 2>&1 | tee /tmp/audit_fast.log

# Warm-run timing
python -c "
import jax, time
jax.config.update('jax_enable_x64', True)
from rabbit.jax.driver_typeI_char import (
    JAXTypeICharConfig, run_full_coupled_typeI_char_jax,
)
cfg = JAXTypeICharConfig(Sigma_H_plus=0.1, correction_level=0,
                           N_q=20, N_mu=12, n_reactions=12)
run_full_coupled_typeI_char_jax(cfg)   # warm-up
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    run_full_coupled_typeI_char_jax(cfg)
    ts.append(time.perf_counter() - t0)
print(f'warm-min={min(ts):.3f}s mean={sum(ts)/5:.3f}s')
" 2>&1 | tee /tmp/audit_timing.log
```

### 3.2 `STATE_OF_RECORD.md` update

The sections that *must* be refreshed after every physics-affecting
PR:

- `§2.x` — capability status for the affected backend (Production /
  Candidate / Deferred).
- `§3` — if a new design decision was taken, append a subsection with
  rationale.
- `§4.1 / §4.2 / §4.3` — parity tables; paste the worst |ΔY_p| and
  |ΔN_eff| per Σ × CL cell.
- `§5` — file inventory: add any new module.
- `§6` — test counts, plus any xfail or red-test changes.

Always commit `STATE_OF_RECORD.md` **in the same PR** that changes the
underlying code.  Reviewers must reject PRs with a stale record.

### 3.3 `PR_CATALOG.md` append

At PR close, append a completion record using the template at
[ROADMAP_PR_CATALOG.md §0](ROADMAP_PR_CATALOG.md).  Fields:

```markdown
### PR-<ID>  <one-line summary>

- **Status:** merged / reverted / partial
- **Scope:** component(s) changed
- **Key files:** path/to/file.py:Lx-Ly (deltas only)
- **Physics added/changed:** paper equation refs, new primitives
- **Parity before / after:** worst-case numbers from audit §2.2
- **Performance before / after:** timing + VRAM figures
- **Known red tests:** any test the PR flipped or left in a
  pre-existing state
- **Docs updated:** list of files touched for doc sync
- **Self-audit verdict:** pass / conditional / fail with link to the
  adversarial review
```

### 3.4 Topic guide update

For every PR that implements a step in a topic guide:

| Topic guide | Status-line update |
|---|---|
| [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md) | Flip "Phase X" from "planned" to "delivered in PR-N?". |
| [IMPLEMENTATION_GUIDE_3T_THERMO.md](IMPLEMENTATION_GUIDE_3T_THERMO.md) | Flip to "delivered" at PR close (already done for the PR-T3T completion). |
| [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md) | Flip "Stage A/B/C/D" through PR-T3A → PR-T3D. |
| [IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md](IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md) | For AP51-AP81 work, update the planned row to landed/partial only when the named exit gate is implemented; keep reuse-first assets and no-QKE/no-public-dispatch boundaries current. |
| [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md) | Flip "Phase 0/1/2/3/4" through PR-A / PR-J / PR-G. |

---

## 4.  Adversarial review protocol

Every PR that changes physics (not pure docs or dispatch) must include
a third-party adversarial review.  The operational procedure:

1. **Spawn a fresh agent** (e.g. `claude-code-guide`, or
   `Agent(subagent_type="general-purpose")`) with no prior context on
   the PR.
2. **Prompt the agent** with:
   > You are a third-party physics auditor reviewing this diff.
   > Audit harshly for: hallucinated physics, mock surrogates,
   > off-by-one errors in state indexing, unit mismatches,
   > divergences from the paper, silent NaN paths, JIT/XLA
   > correctness issues.  Report issues ranked by severity.  If
   > everything is correct, say so explicitly.  Cap response at 400
   > words.
3. **Commit the agent's verdict** as a file `docs/audit/PR-<ID>.md`
   so future PRs and reviewers can inspect the provenance.
4. **If the auditor flags an issue**, resolve it or document why the
   flag is a false positive before closing the PR.

---

## 5.  Regression artefacts and their home

Persistent artefacts that should survive as regressions:

| Artefact | Location | Updated by |
|---|---|---|
| Parity-numbers table | `ROADMAP_STATE_OF_RECORD.md §4` | Every PR closer |
| Warm-run timing baseline | `ROADMAP_PR_CATALOG.md` per-PR entry | Every PR closer |
| Adversarial-audit reports | `docs/audit/PR-<ID>.md` | Per-PR adversarial reviewer |
| Capability registry | `src/rabbit/config/backend_capabilities.py` | PR adding a new backend |
| Backend list in `STATUS.md` / `README.md` | repo root | PR-D (auto-backend flip) |

---

## 6.  Pre-existing red tests (stable baseline)

The four tests listed in
[STATE_OF_RECORD.md §6.1](ROADMAP_STATE_OF_RECORD.md#61-known-pre-existing-red-tests-not-caused-by-recent-work)
are considered the **stable red baseline**.  The audit protocol tells
a reviewer:

- If **any of these four** flip to green during a PR, celebrate — the
  PR accidentally or deliberately fixed a pre-existing issue.  Note
  in the catalogue.
- If **any new test** turns red, the PR must either (a) explain it as
  a deliberate tightening, (b) fix it before close, or (c) lift it to
  a formally-documented `xfail` with justification.

No PR may silently degrade the count of `passed` in
`pytest tests/ -m "not slow and not gpu"`.

---

## 7.  Closing an audit

When every box in §2 is checked, the docs in §3 are updated, and the
adversarial-review verdict in §4 is committed:

1. Append the completion record to
   [ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md).
2. Delete any scratch logs (`/tmp/audit_*.log`) that are no longer
   needed.
3. Merge the PR.

The catalogue is the canonical history of the roadmap; the
state-of-record is the canonical snapshot.  A reader opening either
document after an arbitrary number of PRs have landed must see a
consistent picture.
