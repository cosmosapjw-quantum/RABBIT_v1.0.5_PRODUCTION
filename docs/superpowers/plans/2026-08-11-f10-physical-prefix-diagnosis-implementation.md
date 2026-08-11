# F-10 Physical-Prefix Diagnosis Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `diagnosis_report` with the exact source lock, retained order-60/`y_max=30` states, deterministic input and catalog manifests, prospectively sealed physical-prefix contract, and directly executed physical RHS/JVP receipts required by the approved design.

**Architecture:** Keep canonical historical/source bytes at their original paths and expose them through a root diagnosis index. A single audit module supplies deterministic serialization, input preparation, the unmodified physical collision/RHS evaluation, exact solver-bundle Arnoldi loading, receipt execution, and fail-closed verification. The Git history provides two immutable phases: a seal commit containing all code/inputs/contract, followed by a receipt-only commit that cannot alter protected bytes.

**Tech Stack:** Python 3.10+, NumPy, SciPy, `zipfile`, Git plumbing commands, pytest, JSON/JSONL, deterministic NPZ, Markdown, SHA-256.

## Global Constraints

- Branch: `diagnosis_report`; base: `f10-independent-validation-b3v2@719987d0bc5a018d57fded1df2c8ad3f0c3fc24f`.
- Do not merge to `main`; push only `origin/diagnosis_report` and verify the remote SHA.
- Do not modify `src/rabbit/decoupling/_independent_noqke.py`, `scripts/audit/_trajectory_core.py`, the Rust physics tree, public dispatch, gate registries, or shared harness context.
- Track the two exact retained research ZIPs and the exact V3a/V1-instrument provenance slices at their canonical repository paths; do not duplicate those bytes under the diagnosis directory.
- The contract must be committed before any new physical receipt byte. Any protected-path change after sealing invalidates the receipts and requires a new seal plus rerun.
- Direct JVP semantics are fixed at relative step `1e-3`, time augmentation, `m=10`, double modified Gram-Schmidt, and Arnoldi tolerance `1e-12`; no post-output fitting.
- Static receipts are not a trajectory. Keep `physical_prefix_executed=false`, `reaction_tail_authority_validated=false`, and `d071_reopen_earned=false`.
- Preserve raw failures; a completed receipt file may carry a negative or limited scientific verdict.
- Apply the exact claim labels `IMPLEMENTED`, `VALIDATED`, `DERIVED`, `SPECIFIED`, `PROPOSED`, `SPECULATIVE`, `DEPRECATED`, and `FORBIDDEN`.
- Report added/deleted/net lines, exact token use as `UNAVAILABLE` if no exact counter exists, blocker-movement ratio, and cost-effectiveness verdict.

---

## File Structure

- Create `scripts/audit/f10_physical_prefix_fixture.py`: deterministic fixture/manifest code, direct physical evaluator, exact ZIP-source Arnoldi loader, receipt runner, and validator CLI.
- Create `tests/test_f10_physical_prefix_fixture.py`: real-code regression tests for deterministic bytes, catalog identity, frozen-RHS equivalence, JVP semantics, and tamper rejection.
- Modify `.gitignore`: exact trailing exceptions for the two retained ZIPs, the diagnosis directory, V1 instrument sources, and the V3a retained evidence slice.
- Create `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/`: derived input, machine manifests, prospective contract, final receipts/indexes, and human navigation.
- Modify `README.md`: branch-local notice as the first content.
- Modify `docs/harness/{PROJECT_STATE,CLAIM_LEDGER,VALIDATION_LEDGER,DECISION_LOG,NEXT_SESSION_PROMPT}.md`: append bounded branch/fixture facts without changing generated gate surfaces.
- Track canonical retained artifacts under `.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/` and `.agent-harness/runs/run-20260805-f10-v3-campaign/` through exact ignore exceptions only.
- Track `RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip` and `RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip` byte-for-byte.

### Task 1: Deterministic Fixture Serialization

**Files:**
- Create: `tests/test_f10_physical_prefix_fixture.py`
- Create: `scripts/audit/f10_physical_prefix_fixture.py`

**Interfaces:**
- Produces: `canonical_json_bytes(value: object) -> bytes`
- Produces: `float64_le_bytes(values: np.ndarray) -> bytes`
- Produces: `sha256_bytes(data: bytes) -> str`
- Produces: `write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None`
- Produces: `load_numeric_npz(path: Path) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write deterministic-byte tests first**

```python
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from scripts.audit import f10_physical_prefix_fixture as fixture


def test_canonical_bytes_have_hand_checked_hashes():
    assert fixture.sha256_bytes(fixture.canonical_json_bytes({"b": 2, "a": 1})) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    expected = hashlib.sha256(struct.pack("<2d", 1.0, 2.0)).hexdigest()
    assert expected == "dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae"
    assert fixture.sha256_bytes(fixture.float64_le_bytes(np.array([1.0, 2.0]))) == expected


def test_deterministic_npz_repeats_and_rejects_object_arrays(tmp_path: Path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    arrays = {"z": np.array([3.0]), "a": np.array([1, 2], dtype=np.int64)}
    fixture.write_deterministic_npz(first, arrays)
    fixture.write_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    loaded = fixture.load_numeric_npz(first)
    np.testing.assert_array_equal(loaded["a"], [1, 2])
    with np.testing.assert_raises_regex(ValueError, "object dtype"):
        fixture.write_deterministic_npz(tmp_path / "bad.npz", {"x": np.array([{}], dtype=object)})
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py`

Expected: collection/import failure because `scripts.audit.f10_physical_prefix_fixture` does not exist.

- [ ] **Step 3: Implement only deterministic helpers**

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def float64_le_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object dtype is forbidden")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=9)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit the TDD slice**

```bash
git add scripts/audit/f10_physical_prefix_fixture.py tests/test_f10_physical_prefix_fixture.py
git commit -m "test(f10): lock deterministic fixture bytes"
```

### Task 2: Exact Source, Initial State, and Grid/Catalog Manifests

**Files:**
- Modify: `tests/test_f10_physical_prefix_fixture.py`
- Modify: `scripts/audit/f10_physical_prefix_fixture.py`

**Interfaces:**
- Consumes: Task 1 byte/digest helpers.
- Produces: `build_initial_arrays(setup: core.Setup) -> dict[str, np.ndarray]`
- Produces: `build_quadrature_catalog_manifest(setup: core.Setup) -> dict[str, object]`
- Produces: `build_source_bundle_manifest(repo: Path) -> dict[str, object]`
- Produces: CLI `prepare --repo ROOT --output-dir DIR`.

- [ ] **Step 1: Add failing tests for exact identities**

```python
def test_initial_and_catalog_manifest_use_frozen_order60_contract():
    setup = core.build_setup(order=60, y_max=30.0, label="f10-prefix")
    arrays = fixture.build_initial_arrays(setup)
    assert arrays["y"].shape == (182,)
    assert float(arrays["N"]) == 0.0
    assert int(arrays["order"]) == 60
    assert float(arrays["y_max"]) == 30.0
    manifest = fixture.build_quadrature_catalog_manifest(setup)
    assert manifest["grid"]["nodes"]["count"] == 60
    assert manifest["grid"]["weights"]["count"] == 60
    assert manifest["catalogs"]["self_reactions"]["count"] == 48
    assert manifest["catalogs"]["electron_reactions"]["count"] == 18
    assert manifest["catalogs"]["self_events"]["count"] == 27
    assert manifest["catalogs"]["electron_events"]["count"] == 15


def test_source_bundle_manifest_resolves_exact_archives_and_tree():
    manifest = fixture.build_source_bundle_manifest(Path.cwd())
    assert manifest["rabbit_source"]["commit"] == fixture.BASE_COMMIT
    assert len(manifest["rabbit_source"]["tree_oid"]) == 40
    assert manifest["solver_research_archive"]["sha256"] == (
        "8ffb9c34019e4bc9e431985df9fe69a347ced5da11f68308a1943187e3829fd8"
    )
    assert manifest["solver_research_archive"]["internal_history_commit"] == (
        "b8f11b03d9d59746c4ceddbb0712dfbd3f5386ab"
    )
    assert manifest["mathphysics_research_archive"]["sha256"] == (
        "bb3ca057d1ecee6b11e33bba5dbcd8325a23d95dfe925bb5a235866d05ed4fb0"
    )
```

- [ ] **Step 2: Verify the new tests fail for missing functions**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py`

Expected: failures naming `build_initial_arrays` and `build_source_bundle_manifest`.

- [ ] **Step 3: Implement stable dataclass/catalog and Git/ZIP manifests**

```python
def _catalog_entry(items: Sequence[object]) -> dict[str, object]:
    records = [asdict(item) for item in items]
    return {
        "count": len(records),
        "records": records,
        "canonical_json_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def build_initial_arrays(setup: core.Setup) -> dict[str, np.ndarray]:
    _, state = core.initial_state(setup)
    return {
        "N": np.array(0.0, dtype=np.float64),
        "order": np.array(setup.order, dtype=np.int64),
        "state_dim": np.array(setup.state_size, dtype=np.int64),
        "y": np.asarray(state, dtype=np.float64),
        "y_max": np.array(setup.y_max, dtype=np.float64),
    }
```

Use `git rev-parse`, `git ls-tree`, and `git show COMMIT:path` with `check=True`; parse the solver ZIP's internal `REPRODUCIBILITY_MANIFEST.json`; hash the internal Git bundle and exact `exprb.py`/`jvp.py` members. Reject a missing archive, unexpected internal commit, non-40-character OID, or digest mismatch.

- [ ] **Step 4: Implement `prepare` without physical evaluations**

`prepare` writes only:

- `initial_state_order60_ymax30.npz`;
- `SOURCE_BUNDLE.json`;
- `QUADRATURE_CATALOG_MANIFEST.json`;
- `PREFIX_INPUTS.json`;
- `PREFIX_CONTRACT.json` and `PREFIX_CONTRACT.sha256`.

The contract must bind the exact hashes it reads, define protected paths, set receipt output paths, set `physical_prefix_executed=false`, and include the approved `m=10`, relative-step `1e-3`, coverage/call/wall/kill/no-refit clauses. It must not call `evaluate_independent_collision_action` or create `receipts/` outputs.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py`

Expected: `4 passed`.

- [ ] **Step 6: Commit the source/input generator slice**

```bash
git add scripts/audit/f10_physical_prefix_fixture.py tests/test_f10_physical_prefix_fixture.py
git commit -m "feat(f10): generate exact prefix inputs"
```

### Task 3: Direct Physical RHS and Receipt Diagnostics

**Files:**
- Modify: `tests/test_f10_physical_prefix_fixture.py`
- Modify: `scripts/audit/f10_physical_prefix_fixture.py`

**Interfaces:**
- Produces: `PhysicalEvaluation` frozen dataclass.
- Produces: `evaluate_physical_state(setup: core.Setup, N: float, state: np.ndarray) -> PhysicalEvaluation`.
- Produces: `base_receipt(label: str, source_path: str, evaluation: PhysicalEvaluation) -> dict[str, object]`.

- [ ] **Step 1: Add a real low-order equivalence test**

The production change this test catches is a sign, chain-factor, state-layout, temperature, or energy-transfer drift between the receipt evaluator and the frozen trajectory RHS.

```python
def test_receipt_evaluator_matches_frozen_rhs_on_real_collision_state():
    setup = core.build_setup(
        order=8, y_max=8.0, incoming_polar_order=2,
        final_polar_order=2, electron_radial_order=8, label="test",
    )
    _, state = core.initial_state(setup)
    observed = fixture.evaluate_physical_state(setup, 0.0, state)
    stats = core.Stats()
    expected = core.make_rhs(setup, stats, core.Deadline(600.0))(0.0, state)
    np.testing.assert_array_equal(observed.rhs, expected)
    assert observed.occupations_strict_open
    assert observed.occupation_min > 0.0
    assert observed.occupation_max < 1.0
    assert np.isfinite(observed.first_law_residual)
```

- [ ] **Step 2: Run the one test and verify RED**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py::test_receipt_evaluator_matches_frozen_rhs_on_real_collision_state`

Expected: failure because `evaluate_physical_state` is absent.

- [ ] **Step 3: Implement the single-call physical evaluator**

Compute the collision action exactly once, then reproduce `_trajectory_core.make_rhs` using its declared equations:

```python
t_cm = setup.t_start * float(np.exp(-N))
pair_rate = 0.5 * np.stack(
    (action.total[0] + action.total[1],
     action.total[2] + action.total[3],
     action.total[4] + action.total[5])
)
dc_dN = pair_rate / (thermo.hubble_mev * ind.cloglog_chain_factor(cloglog))
eos = ind.electromagnetic_eos_adaptive(t_gamma)
dTgamma_dN = (
    -3.0 * (eos.rho + eos.pressure)
    + action.electron_bath_energy_transfer / thermo.hubble_mev
) / eos.drho_dtemperature
rhs = np.concatenate((dc_dN.ravel(), [dTgamma_dN, 1.0 / thermo.hubble_mev]))
```

Record units (`N` dimensionless, temperatures MeV, final clock derivative MeV^-1 per e-fold), finite checks, occupation extrema, first-law residual, domain rejections, matrix roundoff, equilibrium number/energy tail fractions, and last-four-node distortion. Hard-code no pass threshold for the tail authority; record `reaction_tail_authority_validated=False`.

- [ ] **Step 4: Run focused and comparator tests**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py tests/test_independent_noqke_comparator.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the physical evaluator slice**

```bash
git add scripts/audit/f10_physical_prefix_fixture.py tests/test_f10_physical_prefix_fixture.py
git commit -m "feat(f10): evaluate physical prefix receipts"
```

### Task 4: Exact Bundle Arnoldi and Direct JVP Provenance

**Files:**
- Modify: `tests/test_f10_physical_prefix_fixture.py`
- Modify: `scripts/audit/f10_physical_prefix_fixture.py`

**Interfaces:**
- Produces: `load_exact_arnoldi(solver_zip: Path) -> Callable[..., object]`.
- Produces: `run_state_arnoldi_receipt(setup: core.Setup, label: str, source_path: str, N: float, state: np.ndarray, arnoldi: Callable[..., object], relative_step: float = 1e-3, krylov_dim: int = 10, tolerance: float = 1e-12) -> tuple[dict[str, object], dict[str, np.ndarray]]`.
- Produces: CLI `run-receipts --repo ROOT --output-dir DIR --seal-commit OID`.

- [ ] **Step 1: Add failing exact-source and physical-JVP tests**

```python
def test_exact_bundle_arnoldi_runs_double_mgs_identity_case():
    arnoldi = fixture.load_exact_arnoldi(Path(fixture.SOLVER_ZIP_NAME))
    result = arnoldi(lambda vector: vector, np.array([1.0, 0.0]), max_dim=2, tolerance=1e-12)
    assert result.dimension == 1
    assert result.breakdown


def test_time_augmented_receipt_uses_direct_rhs_calls_and_fixed_rule():
    setup = core.build_setup(
        order=8, y_max=8.0, incoming_polar_order=2,
        final_polar_order=2, electron_radial_order=8, label="test",
    )
    _, state = core.initial_state(setup)
    arnoldi = fixture.load_exact_arnoldi(Path(fixture.SOLVER_ZIP_NAME))
    receipt, vectors = fixture.run_state_arnoldi_receipt(
        setup, "initial", "derived", 0.0, state, arnoldi,
        relative_step=1e-3, krylov_dim=2, tolerance=1e-12,
    )
    assert receipt["jvp_rule"]["relative_step"] == 1e-3
    assert receipt["rhs_call_accounting"]["base_calls"] == 1
    assert receipt["rhs_call_accounting"]["shifted_calls"] == len(receipt["jvp_calls"])
    assert receipt["arnoldi"]["dimension"] <= 2
    assert vectors["base_rhs"].shape == (26,)
    assert all(call["scheme"] == "forward_time_augmented" for call in receipt["jvp_calls"])
```

- [ ] **Step 2: Run both tests and verify RED**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py -k 'bundle_arnoldi or time_augmented'`

Expected: missing-function failures.

- [ ] **Step 3: Implement exact ZIP extraction/import and JVP capture**

Use `TemporaryDirectory`, extract the exact archive, prepend only its `src` directory, import `f10_solver_research.jvp.arnoldi`, and verify the loaded `jvp.py` SHA-256 against the archive manifest computed in Task 2.

For every Arnoldi operator call, evaluate:

```python
epsilon = relative_step * max(1.0, np.linalg.norm(np.r_[state, N])) / np.linalg.norm(direction)
shifted = evaluate_physical_state(
    setup,
    N + epsilon * direction[-1],
    state + epsilon * direction[:-1],
)
jvp = np.r_[(shifted.rhs - base.rhs) / epsilon, 0.0]
```

Retain `direction`, `epsilon`, shifted RHS, JVP, hashes, physical diagnostics, relative-difference signal, subtractive-condition ratio, and each full-RHS-equivalent call. Compute `||V^T V-I||_inf`; do not suppress Arnoldi failure or strict-domain failure.

- [ ] **Step 4: Implement fail-closed receipt CLI**

Before any physical call, `run-receipts` must verify:

- exact clean `HEAD == --seal-commit`;
- contract digest equals `PREFIX_CONTRACT.sha256`;
- every protected path is tracked and matches its sealed digest;
- both receipt output files do not exist.

It then runs all four states independently, writes deterministic NPZ vectors and canonical JSON receipts even if one state fails, writes `RECEIPT_RUN_LOG.json`, and exits nonzero if any state status is not `EXECUTED` or `EXECUTED_WITH_RECORDED_BREAKDOWN`.

- [ ] **Step 5: Run all focused tests and verify GREEN**

Run: `PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py`

Expected: all fixture tests pass.

- [ ] **Step 6: Commit the sealed-run implementation**

```bash
git add scripts/audit/f10_physical_prefix_fixture.py tests/test_f10_physical_prefix_fixture.py
git commit -m "feat(f10): retain direct physical JVP provenance"
```

### Task 5: Assemble and Commit the Prospective Seal

**Files:**
- Modify: `.gitignore`
- Create: pre-receipt files under `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/`
- Track: exact canonical run/archive paths declared in File Structure.

**Interfaces:**
- Consumes: `prepare` CLI and all Task 1-4 code.
- Produces: a clean Git seal commit whose SHA is the sole admissible `--seal-commit` for Task 6.

- [ ] **Step 1: Add exact trailing ignore exceptions**

Keep broad ignores, then append explicit exception hierarchies. Re-ignore sibling run entries with `*` at each opened directory and unignore only named files/directories. Add exact exceptions for:

```text
/RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip
/RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip
/00_F10_PHYSICAL_PREFIX_DIAGNOSIS/**
/.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_rhs.py
/.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_bdf.py
/.agent-harness/runs/run-20260805-f10-v3-campaign/{run_v3.py,analyse_v3.py,render_v3.py,ANALYSIS_V3.json,report_verification_output.json,r4_reference.json}
/.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/{pins_verified.json,selftest_result.json,driver.log,nohup.log}
/.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/**
```

- [ ] **Step 2: Generate only pre-receipt artifacts**

Run:

```bash
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py prepare \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS
```

Expected: source/input/grid/catalog/contract files exist; `receipts/PHYSICAL_RHS_JVP_RECEIPTS.json` and `receipts/PHYSICAL_RHS_JVP_VECTORS.npz` do not exist.

- [ ] **Step 3: Stage exact retained bytes and inspect scope**

Run exact `git add` arguments for `.gitignore`, the two ZIPs, both instrument files, the named V3 campaign files, `v3a_r2` pins/logs, its complete `domain` directory, the generator/test, and the pre-receipt diagnosis files. Inspect with:

```bash
git status --short
git diff --cached --name-status
git diff --cached --numstat
```

Reject any unrelated run, generated harness context, cache, gate registry, source physics, or public-dispatch path.

- [ ] **Step 4: Verify seal inputs before commit**

Run:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py tests/test_independent_noqke_comparator.py
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py verify-preseal --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS
git diff --cached --check
```

Expected: tests and preseal verifier pass, every required path is staged/tracked-or-staged, and no physical receipt exists.

- [ ] **Step 5: Create the prospective seal commit**

```bash
git commit -m "feat(f10): seal physical prefix fixture"
git rev-parse HEAD
git status --short
```

Record the resulting clean commit as `SEAL_COMMIT`. Do not amend it after receipt execution.

For each later shell invocation, resolve that immutable commit explicitly with:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
test -n "${f10_seal_commit}"
```

### Task 6: Execute and Preserve Physical RHS/JVP Receipts

**Files:**
- Create: `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/receipts/PHYSICAL_RHS_JVP_RECEIPTS.json`
- Create: `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/receipts/PHYSICAL_RHS_JVP_VECTORS.npz`
- Create: `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/receipts/RECEIPT_RUN_LOG.json`

**Interfaces:**
- Consumes: clean Task 5 `SEAL_COMMIT` and exact protected bytes.
- Produces: direct, source/input/contract-bound physical receipt artifacts.

- [ ] **Step 1: Recheck chronology immediately before output**

Run:

```bash
test -z "$(git status --porcelain)"
test ! -e 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/receipts/PHYSICAL_RHS_JVP_RECEIPTS.json
sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/PREFIX_CONTRACT.sha256
```

Expected: clean tree, absent outputs, valid contract digest.

- [ ] **Step 2: Run the exact physical receipt command once**

Run with the literal seal SHA returned in Task 5:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py run-receipts \
  --repo . \
  --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit "${f10_seal_commit}"
```

Expected: four state attempts, full direct-call accounting, preserved outputs. A nonzero exit is retained and reported; do not change parameters or overwrite it.

- [ ] **Step 3: Verify raw receipt bindings without editing sealed bytes**

Run:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py verify-receipts \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit "${f10_seal_commit}"
git diff --name-only "${f10_seal_commit}" -- \
  scripts/audit/f10_physical_prefix_fixture.py \
  tests/test_f10_physical_prefix_fixture.py \
  00_F10_PHYSICAL_PREFIX_DIAGNOSIS/PREFIX_CONTRACT.json \
  00_F10_PHYSICAL_PREFIX_DIAGNOSIS/PREFIX_CONTRACT.sha256 \
  00_F10_PHYSICAL_PREFIX_DIAGNOSIS/PREFIX_INPUTS.json \
  00_F10_PHYSICAL_PREFIX_DIAGNOSIS/QUADRATURE_CATALOG_MANIFEST.json \
  00_F10_PHYSICAL_PREFIX_DIAGNOSIS/initial_state_order60_ymax30.npz
```

Expected: receipt verification reports exact observed status; protected-path diff is empty.

### Task 7: External Navigation, Final Indexes, and SSOT Ledgers

**Files:**
- Create/modify: all remaining diagnosis files listed in File Structure.
- Modify: `README.md`
- Modify: `docs/harness/{PROJECT_STATE,CLAIM_LEDGER,VALIDATION_LEDGER,DECISION_LOG,NEXT_SESSION_PROMPT}.md`

**Interfaces:**
- Consumes: Task 6 immutable receipts and actual command results.
- Produces: human/machine navigation, fail-closed readiness, validation evidence, claim boundaries, and final checksums.

- [ ] **Step 1: Generate final machine indexes from actual bytes**

Run the CLI finalizer with the literal seal commit:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py finalize \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit "${f10_seal_commit}"
```

It writes `BRANCH_SCOPE.json`, `PROVENANCE_INDEX.json`, `RECEIPT_INDEX.json`, `READINESS.json`, and `SHA256SUMS`. It sets the first four readiness booleans from observed evidence, and fixes the final three to false with evidence-backed reasons. It excludes `SHA256SUMS` itself from its own digest list.

- [ ] **Step 2: Add human navigation and root-first notice**

Use `apply_patch` to create `00_F10_PHYSICAL_PREFIX_DIAGNOSIS/README.md` and `FILE_LOCATIONS.md`, with relative links to every canonical source, archive, checkpoint, raw log, receipt, contract, and validation file. Prepend `README.md` with a branch-only notice pointing to the diagnosis README before the existing title/content.

- [ ] **Step 3: Append bounded SSOT facts**

Record:

- project state: branch/fixture exists; no gate moved;
- claim ledger: artifact set `IMPLEMENTED`, direct static receipts `VALIDATED` only to executed state/hash scope, prefix execution `SPECIFIED`, reaction-tail authority and D-071 promotion `FORBIDDEN` on these bytes;
- validation ledger: exact commands, exit codes, versions, seal/contract/output hashes, failures/skips;
- decision log: source-by-reference, prospective chronology, no-main-merge decision;
- next-session prompt: preserve seal/receipt bytes; do not rerun or promote without a new owner-authorized contract.

Do not edit generated status-board blocks or registry status.

- [ ] **Step 4: Run final artifact and scientific validation**

Run fresh:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
PYTHONPATH=src:. python -m pytest -q tests/test_f10_physical_prefix_fixture.py tests/test_independent_noqke_comparator.py
PYTHONPATH=src:. python scripts/audit/f10_physical_prefix_fixture.py verify-final \
  --repo . --output-dir 00_F10_PHYSICAL_PREFIX_DIAGNOSIS \
  --seal-commit "${f10_seal_commit}"
sha256sum -c 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/SHA256SUMS
python -m json.tool 00_F10_PHYSICAL_PREFIX_DIAGNOSIS/READINESS.json >/dev/null
PYTHONPATH=src:. python -m pytest -q tests/test_claim_gates.py tests/test_surface_scope_honesty.py
git diff --check
```

Record every command actually run; do not record planned commands as passed.

- [ ] **Step 5: Calculate anti-drift cost and commit receipts/package**

Run:

```bash
f10_seal_commit="$(git log --format=%H --grep='^feat(f10): seal physical prefix fixture$' -n 1)"
git diff --numstat "${f10_seal_commit}"
git diff --stat "${f10_seal_commit}"
git status --short
```

Record added/deleted/net lines, `token_use_exact: UNAVAILABLE` with reason, blocker-movement ratio, and verdict in the diagnosis validation ledger. Stage exact intended paths, inspect the staged diff, then commit:

```bash
git commit -m "docs(f10): publish physical prefix diagnosis"
```

### Task 8: Adversarial Review and Remote Delivery

**Files:**
- No production changes expected; corrections require a new RED test when executable behavior changes.

**Interfaces:**
- Consumes: final local commit and complete validation evidence.
- Produces: reviewed `origin/diagnosis_report` with exact SHA equality.

- [ ] **Step 1: Review the final diff against all seven requirements**

Check every requirement ID in the approved design against tracked files, hashes, receipt selectors, and readiness fields. Specifically attempt to falsify:

- source ZIP/internal bundle identity;
- retained-state provenance and non-duplication;
- direct JVP versus historical observation-Jacobian confusion;
- contract-before-output chronology;
- first-law/occupation/domain/tail receipt presence;
- false tail, trajectory, gate, production, or main-merge claims;
- ignored or locally inaccessible links.

- [ ] **Step 2: Run fresh completion verification**

Repeat the final test/verifier/checksum/link/whitespace commands from Task 7 on the exact commit candidate and inspect their full exit status. Verify `git status --short` contains only the intended staged or clean state.

- [ ] **Step 3: Push only the diagnosis branch**

```bash
git push -u origin diagnosis_report:diagnosis_report
git fetch origin diagnosis_report
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/diagnosis_report)"
git merge-base --is-ancestor 719987d0bc5a018d57fded1df2c8ad3f0c3fc24f origin/diagnosis_report
```

Do not push or update `origin/main`.

- [ ] **Step 4: Report exact outcome**

Report branch/local/remote SHA, seal SHA, contract SHA-256, receipt status, changed files, commands and results, failures/skips, scientific ceiling, anti-drift metrics, and remaining risks. State explicitly that the physical prefix itself was not run and D-071 did not move.
