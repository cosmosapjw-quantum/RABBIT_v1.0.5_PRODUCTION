# PR-T3D — Tier-3 Full-Collision Integration + Cross-Code Lock (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

PR-T3A built the full-phase-space driver and q-advection.
PR-T3B wired ν-e elastic + pair.  PR-T3C added diagonal ν-ν.
PR-T3D integrates the three pieces into a canonical **tier-3**
backend surface and locks its numerical output against external
cross-codes (LASAGNA, FortEPiaNO, PRIMAT-AC2024).

Invariants: Rodas5P, CPU-preferred, float64.  Tier-3 becomes the
publication-grade incomplete-decoupling path for RABBIT.
Momentum-averaged tier-2 (`jax_characteristic_tier2`) is retained
as a fast-approximation surface but demoted from publication
status.

Dependencies: PR-T3A (strict), PR-T3B (strict), PR-T3C (strict),
PR-J (strongly recommended — tier-3 throughput is gated by
Jacobian cost).

---

## 1. Phase objective

1. Register `JAX_TYPEI_FULL_BOLTZMANN_TIER3` capability and
   `jax_full_collision_tier3` backend key.
2. Add the dispatch branch in
   `canonical_forward_solver(backend="jax_full_collision_tier3",
   ...)`.
3. Add a cross-code parity test suite
   (`tests/test_jax_tier3_cross_code_parity.py`).
4. Document tier-3 as the new publication-grade incomplete-
   decoupling path in `STATE_OF_RECORD.md §2.3`.
5. Add an anisotropic tier-3 stability sweep to verify
   `N_eff` insensitivity to Σ_H.
6. Update the `IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
   status matrix one last time.

---

## 2. Literature anchors

### 2.1 Internal
- All PR-T3A/B/C code and audit documents.
- `docs/ROADMAP_STATE_OF_RECORD.md §4.3` — cross-code target
  table.
- Existing capability registrations in
  `src/rabbit/config/backend_capabilities.py` — follow the
  pattern used for `JAX_TYPEI_CHARACTERISTIC_TIER2`.

### 2.2 External
- LASAGNA (Escudero 2019, `arXiv:1812.05605`) — FLRW Y_p / N_eff
  fixture.
- FortEPiaNO (Froustey 2020, `arXiv:2008.01074`) — FLRW Y_p / N_eff
  fixture.
- PRIMAT AC2024 (Pitrou 2024) — FLRW Y_p / N_eff fixture.
- Where possible, pull the exact published numbers into a
  `tests/fixtures/tier3_cross_code.json` file so the lock is
  reproducible.

### 2.3 Paper-equation cross-check
- [ ] Paper §20.2 (PRIMAT cross-validation existing baseline for
      the SciPy characteristic path).  This is the reference
      structure for how RABBIT documents cross-code parity.
- [ ] Paper §11.6 — N_eff definition in the tiered hierarchy.

---

## 3. Skeleton code

### 3.1 Capability registration

```python
# src/rabbit/config/backend_capabilities.py
JAX_TYPEI_FULL_BOLTZMANN_TIER3 = BackendCapability(
    key="jax_typeI_full_boltzmann_tier3",
    backend="jax",
    tier="canonical",
    surface_class="canonical",
    physics_scope="TypeI_tier3_full_collision",
    weak_mode="live_f0_cl0_cl3",
    max_correction_level=3,
    supports_teff=False,
    max_thermo_tier=3,           # new: tier-3 3T + collisions
    validated_default=False,
    validation_mode="full",
    readiness_scope_contract="bounded_jax_tier3_full_boltzmann_cpu_first_v1",
    transport_scope_contract="full_phase_space_ray_v1",
    thermo_scope_contract="tier3_3T_with_full_collision_source_v1",
    collision_scope_contract="full_nonperturbative_tier3_no_qke_v1",
    notes=(
        "JAX tier-3 full-Boltzmann incomplete decoupling (Paper II "
        "scope, no QKE).  Per-ray per-momentum state; Hannestad-Madsen "
        "ν-e elastic + pair process + Dolgov-Hansen-Semikoz diagonal "
        "ν-ν scattering.  FLRW N_eff = 3.044 ± 0.005 cross-checked "
        "against LASAGNA / FortEPiaNO / PRIMAT-AC2024."
    ),
    supports_live_weak_opt_in=True,
    live_weak_species=("nue", "nuebar", "nux", "nuxbar"),
)

CAPABILITY_BY_BACKEND = {
    ...
    "jax_full_collision_tier3": JAX_TYPEI_FULL_BOLTZMANN_TIER3,
    ...
}
```

### 3.2 Dispatch

```python
# src/rabbit/inference/forward_likelihood.py
if backend == "jax_full_collision_tier3":
    if enable_teff:
        raise ValueError("tier-3 path is incompatible with Teff.")
    if abs(Sigma_H_minus) > 0:
        raise NotImplementedError("tier-3 LRS-only (non-LRS tier-3 deferred).")
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig, run_full_boltzmann_jax,
    )
    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=float(Sigma_H), eta=float(eta), tau_n=float(tau_n),
        N_q=int(N_q), N_mu=12, n_reactions=int(n_reactions),
        correction_level=int(correction_level),
        thermo_tier=3,
        enable_nu_e=True,
        enable_pair=True,
        enable_nu_nu=True,
        runtime_device_policy="cpu_preferred",
    )
    result = run_full_boltzmann_jax(cfg)
    metadata = {
        "dispatch_backend": "jax_full_collision_tier3",
        "capability_key": capability.key,
        "transport_mode": "full_boltzmann_ray",
        "collision_closure_mode": "full_nonperturbative_tier3_no_qke",
        "production_authority": "paper_II_tier3_candidate",
        ...,
    }
    return BBNPrediction(Yp=result.Yp, DH=result.DH, ...)
```

### 3.3 Cross-code parity tests

```python
# tests/test_jax_tier3_cross_code_parity.py
import json
import pytest
from pathlib import Path

pytest.importorskip("jax")

_FIXTURE = Path(__file__).parent / "fixtures" / "tier3_cross_code.json"


@pytest.fixture(scope="module")
def tier3_result():
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    return canonical_forward_solver(
        Sigma_H=0.0, eta=6.104e-10, tau_n=878.4,
        correction_level=2, n_reactions=12,
        backend="jax_full_collision_tier3",
    )


@pytest.fixture(scope="module")
def cross_code_targets():
    return json.loads(_FIXTURE.read_text())


class TestTier3FLRW:
    def test_yp_matches_lasagna(self, tier3_result, cross_code_targets):
        target = cross_code_targets["lasagna"]["Y_p"]
        assert abs(tier3_result.Yp - target) < 5e-4

    def test_yp_matches_fortepiano(self, tier3_result, cross_code_targets):
        target = cross_code_targets["fortepiano"]["Y_p"]
        assert abs(tier3_result.Yp - target) < 5e-4

    def test_yp_matches_primat_ac2024(self, tier3_result, cross_code_targets):
        target = cross_code_targets["primat_ac2024"]["Y_p"]
        assert abs(tier3_result.Yp - target) < 5e-4

    def test_neff_locks_at_3044(self, tier3_result):
        assert abs(tier3_result.metadata["N_eff"] - 3.044) < 0.005


@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.3])
def test_anisotropic_neff_stability(sigma):
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    r = canonical_forward_solver(
        Sigma_H=sigma, correction_level=0, N_q=20,
        backend="jax_full_collision_tier3",
    )
    assert r.success
    # Tier-3 should have N_eff essentially independent of Σ.
    assert abs(r.metadata["N_eff"] - 3.044) < 0.005
```

### 3.4 Cross-code fixture

```json
// tests/fixtures/tier3_cross_code.json
{
  "lasagna": {
    "Y_p": 0.2470,
    "N_eff": 3.044,
    "reference": "Escudero 2019, JCAP 02:007"
  },
  "fortepiano": {
    "Y_p": 0.2470,
    "N_eff": 3.043,
    "reference": "Froustey, Pitrou, Volpe 2020, JCAP 12:015"
  },
  "primat_ac2024": {
    "Y_p": 0.24703,
    "N_eff": 3.044,
    "reference": "Pitrou 2024 (PRIMAT AC2024)"
  }
}
```

### 3.5 Demotion of `jax_characteristic_tier2`

The momentum-averaged tier-2 path stays as a fast-approximation
surface.  Update its metadata `"surface_class"` to remain
`"canonical"` but flag in `production_authority` that tier-3 is
the preferred publication surface:

```python
# Keep JAX_TYPEI_CHARACTERISTIC_TIER2 as-is but append a note:
notes=(
    "... ['production_authority' now qualified: prefer "
    "jax_full_collision_tier3 for publication runs after PR-T3D].  "
    "Retained as a CPU-fast approximation bringing N_eff ~ 3.034."
),
```

---

## 4. WBS

1. **Capability registration + dispatch wiring.**
2. **Cross-code fixture file.**
3. **Cross-code parity tests.**
4. **Anisotropic stability sweep test.**
5. **Demote tier-2 in metadata / docs.**
6. **Documentation updates**.

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read all three PR-T3A/B/C audit files (`PR-T3{A,B,C}.md`).
  Confirm every prerequisite is green.
- Grep for remaining `TODO` / `FIXME` / `placeholder` in
  `src/rabbit/jax/collisions_jax.py` and
  `driver_typeI_full_boltzmann.py`.

### Stage 2 — External
- Re-verify the exact published numbers in the JSON fixture:
  - LASAGNA: Escudero 2019 Table 1.
  - FortEPiaNO: Froustey 2020 Table 4.
  - PRIMAT-AC2024: paper section with the quoted `Y_p = 0.24703
    ± 6×10⁻⁵` and `N_eff = 3.044`.
- Record the excerpts in `docs/audit/PR-T3D_stage2.md`.

### Stage 3 — Self CoT
- **Scope boundary.**  Confirm explicitly that tier-3 excludes
  QKE / flavour oscillations.  The `live_weak_species` tuple
  `(nue, nuebar, nux, nuxbar)` is consistent with the
  Fierz-diagonal scope.
- **Demotion.**  Tier-2 is still useful as a fast CPU baseline
  (N_eff within 0.3 %% of SM); make sure downstream consumers
  that defaulted to `jax_characteristic_tier2` know about the
  new tier-3 surface.
- **Test matrix coverage.**  Confirm that the three cross-code
  points plus the anisotropic stability sweep cover the claimed
  publication scope; do not over-promise beyond what tier-3 was
  designed for.

Record in `docs/audit/PR-T3D_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] All three cross-code parity tests green.
- [ ] Anisotropic N_eff stability test green.
- [ ] No regression in tier-1 / tier-2 parity.
- [ ] Capability registry sync tests green (9 backends → 10).
- [ ] `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
      completely updated.
- [ ] PR catalogue entry appended.

---

## 7. Adversarial audit prompt

> Audit PR-T3D (tier-3 integration).  Verify: (1) dispatch wiring
> is complete; (2) all three PR-T3A/B/C audits are in place and
> green; (3) cross-code fixture numbers match published
> references (quote URLs); (4) tier-3 backend key is registered
> and discoverable; (5) no silent demotion of tier-2 that would
> surprise existing callers.  Cap 500 words.

---

## 8. Anti-local-minimum reminders
1. **Do not** loosen the 5e-4 cross-code tolerance.  If the test
   fails, the root cause is in PR-T3A/B/C and must be fixed there,
   not papered over here.
2. **Do not** delete the tier-2 backend key.  Backward compat.
3. **Do not** start tier-4 (QKE) work in this PR — that is a
   separate roadmap.

---

## 9. Hallucination prevention
- Every number in the JSON fixture must be sourced from a quoted
  paper line in `docs/audit/PR-T3D_stage2.md`.
- Do not claim "cross-code match" without running the three tests.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md`
- §2.3: add subsection for the tier-3 surface; promote to
  "publication-grade incomplete-decoupling path".
- §2.4: new row in the backend dispatch table for
  `jax_full_collision_tier3`.
- §4.3: update the cross-code row with measured RABBIT numbers.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-T3D entry.

### 10.3 `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
- Move "Phase 7 (dispatch)" and "Stage D" to "delivered in PR-T3D".
- Finalise the test-matrix section with measured cross-code
  numbers.

### 10.4 `STATUS.md` / `README.md` / `BACKEND_CAPABILITY_MATRIX.md`
Add the tier-3 row.

---

## 11. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_jax_tier3_cross_code_parity.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v
pytest tests/test_jax_typeI_characteristic_tier2.py -v
pytest tests/test_pr_t3a_collisionless_reduction.py -v
pytest tests/test_pr_t3b_jax_operator_parity.py -v
pytest tests/test_pr_t3c_nu_nu.py -v

test -f docs/audit/PR-T3D.md
test -f docs/audit/PR-T3D_stage1.md
test -f docs/audit/PR-T3D_stage2.md
test -f docs/audit/PR-T3D_stage3.md
git add docs/audit/PR-T3D*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-T3D: tier-3 full-collision integration + cross-code lock

Registers the JAX_TYPEI_FULL_BOLTZMANN_TIER3 capability and the
jax_full_collision_tier3 backend key.  Wires dispatch through
canonical_forward_solver.  Adds cross-code parity tests against
LASAGNA, FortEPiaNO and PRIMAT-AC2024, all within |ΔY_p| < 5e-4
and N_eff within 3.044 ± 0.005 at FLRW CL2.  Anisotropic
stability sweep confirms N_eff moves < 5e-4 across
Σ_H ∈ {0, 0.1, 0.3}.  Promotes tier-3 to the publication-grade
incomplete-decoupling path; tier-2 (momentum-averaged) retained as
a CPU-fast approximation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 12. Abort conditions

- Any cross-code parity test fails beyond 5e-4.
- Tier-1 / tier-2 regression.
- Anisotropic N_eff stability worse than 5e-4.
- Capability registry drift (new backend not discoverable via
  `CAPABILITY_BY_BACKEND`).

Abort → `docs/audit/PR-T3D_abort.md`.
