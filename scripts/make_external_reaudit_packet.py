#!/usr/bin/env python3
"""Build the BD281 external re-audit packet.

This script is packaging-only. It creates an external audit snapshot under
``audit_packets/`` and does not add a runtime validation gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKET_DATE = os.environ.get("RABBIT_AUDIT_PACKET_DATE", date.today().isoformat())
PACKET_NAME = f"BD281_external_reaudit_packet_{PACKET_DATE}"
PACKET_ROOT = ROOT / "audit_packets" / PACKET_NAME
ZIP_PATH = PACKET_ROOT.with_suffix(".zip")
ZIP_SHA_PATH = Path(str(ZIP_PATH) + ".sha256")

EXCLUDE_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll"}


@dataclass
class ManifestEntry:
    packet_path: str
    original_repo_path: str
    sha256: str
    size_bytes: int
    category: str
    why_included: str
    kind: str
    generated: bool = False
    artifact_source_path: str | None = None
    artifact_source_sha256: str | None = None
    extraction_command: str | None = None
    retained_fields: list[str] | None = None
    dropped_fields: list[str] | None = None
    reduction_reason: str | None = None


entries: list[ManifestEntry] = []


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive packet metadata
        return f"unavailable: {exc}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    name = path.name.lower()
    if any(token in name for token in ("credential", "secret", "token")):
        return True
    return False


def record(
    packet_rel: Path,
    original: str,
    category: str,
    why: str,
    kind: str,
    *,
    generated: bool = False,
    artifact_source_path: str | None = None,
    artifact_source_sha256: str | None = None,
    extraction_command: str | None = None,
    retained_fields: list[str] | None = None,
    dropped_fields: list[str] | None = None,
    reduction_reason: str | None = None,
) -> None:
    path = PACKET_ROOT / packet_rel
    entries.append(
        ManifestEntry(
            packet_path=packet_rel.as_posix(),
            original_repo_path=original,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            category=category,
            why_included=why,
            kind=kind,
            generated=generated,
            artifact_source_path=artifact_source_path,
            artifact_source_sha256=artifact_source_sha256,
            extraction_command=extraction_command,
            retained_fields=retained_fields,
            dropped_fields=dropped_fields,
            reduction_reason=reduction_reason,
        )
    )


def write_text(
    rel: str,
    text: str,
    category: str,
    why: str,
    kind: str = "doc",
) -> None:
    packet_rel = Path(rel)
    out = PACKET_ROOT / packet_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    record(packet_rel, "packet_generated", category, why, kind, generated=True)


def copy_file(
    src_rel: str,
    dest_rel: str | None,
    category: str,
    why: str,
    kind: str,
    *,
    mandatory: bool = True,
) -> bool:
    src = ROOT / src_rel
    if not src.exists():
        if mandatory:
            raise FileNotFoundError(src_rel)
        return False
    dest = PACKET_ROOT / (dest_rel or src_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    record(dest.relative_to(PACKET_ROOT), src_rel, category, why, kind)
    return True


def copy_tree(src_rel: str, dest_rel: str, category: str, why: str, kind: str) -> None:
    src_root = ROOT / src_rel
    if not src_root.exists():
        raise FileNotFoundError(src_rel)
    for src in sorted(src_root.rglob("*")):
        if src.is_dir() or should_exclude(src.relative_to(ROOT)):
            continue
        rel = src.relative_to(src_root)
        dest = PACKET_ROOT / dest_rel / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        record(dest.relative_to(PACKET_ROOT), src.relative_to(ROOT).as_posix(), category, why, kind)


def json_dump(rel: str, obj: Any, category: str, why: str, kind: str = "artifact", **meta: Any) -> None:
    packet_rel = Path(rel)
    out = PACKET_ROOT / packet_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record(packet_rel, "packet_generated", category, why, kind, generated=True, **meta)


def safe_get(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: row.get(k) for k in keys if k in row}


def summarize_bd278() -> None:
    src_rel = "diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.json"
    src = ROOT / src_rel
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    row_fields = [
        "resolution_case_label",
        "passed",
        "physical_full_bbn_span_ready",
        "completion_class",
        "classified_outcome",
        "N_eff_3T",
        "Yp",
        "DH",
        "Sigma_H",
        "T_final_MeV",
        "N_mu",
        "N_phi",
        "angular_grid_label",
        "radial_q_laguerre_n",
        "q_laguerre_order",
        "h_max",
        "N_span",
        "chain_restart_handoff",
        "initial_np_policy",
        "initial_A_monopole_offset",
        "phase1_prerun_T_start_MeV",
        "phase1_prerun_dN",
        "phase2_activation_validation_mode",
        "collision_source_component_policy",
        "collision_projection_policy",
        "jacobian_policy",
        "freedom_composition_cases",
    ]
    neffs = [r.get("N_eff_3T") for r in rows if isinstance(r.get("N_eff_3T"), (int, float))]
    yps = [r.get("Yp") for r in rows if isinstance(r.get("Yp"), (int, float))]
    dhs = [r.get("DH") for r in rows if isinstance(r.get("DH"), (int, float))]
    sigmas = [r.get("Sigma_H") for r in rows if isinstance(r.get("Sigma_H"), (int, float))]
    memory_field_names = sorted(
        {
            key
            for row in rows
            for key in row
            if any(token in key.lower() for token in ("rss", "vmhwm", "tracemalloc", "memory", "ru_maxrss"))
        }
    )
    policy_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if any(token in key.lower() for token in ("policy", "handoff", "projection", "initial_", "phase1", "phase2", "source"))
        }
    )
    extract = {
        "source_artifact_path": src_rel,
        "source_sha256": sha256_file(src),
        "extraction_command": "python scripts/make_external_reaudit_packet.py",
        "retained_fields": row_fields,
        "dropped_fields": ["large nested traces", "full nested row payloads", "per-step diagnostic expansions"],
        "reduction_reason": "The full BD278 shard is about 30 MB; this extract keeps endpoint observables, policy provenance, and parity-relevant fields.",
        "top_level": {
            "created_utc": data.get("created_utc"),
            "contract": data.get("contract"),
            "composition_mode": data.get("composition_mode"),
            "passed": data.get("passed"),
            "claim_scope": data.get("claim_scope"),
            "qke_scope": data.get("qke_scope"),
            "artifact_payload_sha256": data.get("artifact_payload_sha256"),
        },
        "summary": {
            "row_count": len(rows),
            "N_eff_3T_min": min(neffs) if neffs else None,
            "N_eff_3T_max": max(neffs) if neffs else None,
            "N_eff_3T_spread": (max(neffs) - min(neffs)) if neffs else None,
            "Yp_min": min(yps) if yps else None,
            "Yp_max": max(yps) if yps else None,
            "DH_min": min(dhs) if dhs else None,
            "DH_max": max(dhs) if dhs else None,
            "Sigma_H_min": min(sigmas) if sigmas else None,
            "Sigma_H_max": max(sigmas) if sigmas else None,
            "memory_fields_present": memory_field_names,
            "policy_fields_present_sample": policy_fields[:80],
        },
        "rows": [safe_get(r, row_fields) for r in rows],
    }
    json_dump(
        "artifacts/extracts/bd278_endpoint_matrix_shard_1_summary.json",
        extract,
        "artifact_extract",
        "Reduced BD278 shard evidence for N_eff_3T spread, endpoint observables, and missing memory fields.",
        artifact_source_path=src_rel,
        artifact_source_sha256=sha256_file(src),
        extraction_command="python scripts/make_external_reaudit_packet.py",
        retained_fields=row_fields,
        dropped_fields=extract["dropped_fields"],
        reduction_reason=extract["reduction_reason"],
    )


def summarize_artifact(src_rel: str, dest_rel: str, why: str) -> None:
    src = ROOT / src_rel
    data = json.loads(src.read_text(encoding="utf-8"))
    retained = [
        "artifact_path",
        "artifact_payload_sha256",
        "claim_scope",
        "contract",
        "created_utc",
        "inputs",
        "passed",
        "physical_full_bbn_span_ready",
        "qke_scope",
        "public_dispatch_ready",
        "production_smc_validation_ready",
        "summary",
        "terminal_span",
        "canonical_isotropic_high_temperature_baseline",
    ]
    out: dict[str, Any] = {
        "source_artifact_path": src_rel,
        "source_sha256": sha256_file(src),
        "extraction_command": "python scripts/make_external_reaudit_packet.py",
        "retained_fields": retained + ["rows_last_endpoint_sample", "row_count"],
        "dropped_fields": ["full rows except first/last samples", "large nested traces"],
        "reduction_reason": "Size-bounded extract for external audit packet.",
        "top_level": safe_get(data, retained),
    }
    rows = data.get("rows")
    if isinstance(rows, list):
        out["row_count"] = len(rows)
        sample_fields = [
            "N_eff_3T",
            "Yp",
            "DH",
            "Sigma_H",
            "T_final_MeV",
            "passed",
            "physical_full_bbn_span_ready",
            "initial_np_policy",
            "initial_A_monopole_offset",
            "phase2_activation_validation_mode",
            "collision_source_component_policy",
            "collision_projection_policy",
            "h_max",
            "N_span",
            "chain_restart_handoff",
        ]
        samples = []
        if rows:
            samples.append({"position": "first", **safe_get(rows[0], sample_fields)})
            samples.append({"position": "last", **safe_get(rows[-1], sample_fields)})
        out["rows_last_endpoint_sample"] = samples
    json_dump(
        dest_rel,
        out,
        "artifact_extract",
        why,
        artifact_source_path=src_rel,
        artifact_source_sha256=sha256_file(src),
        extraction_command="python scripts/make_external_reaudit_packet.py",
        retained_fields=out["retained_fields"],
        dropped_fields=out["dropped_fields"],
        reduction_reason=out["reduction_reason"],
    )


def write_helper_scripts() -> None:
    write_text(
        "scripts/extract_packet_summary.py",
        r'''
        #!/usr/bin/env python3
        """Print a compact summary of the audit packet artifacts."""

        from __future__ import annotations

        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]

        def load(rel: str):
            return json.loads((root / rel).read_text(encoding="utf-8"))

        bd278 = load("artifacts/extracts/bd278_endpoint_matrix_shard_1_summary.json")
        summary = bd278["summary"]
        print("BD278 shard 1 extract")
        print(f"  rows: {summary['row_count']}")
        print(
            "  N_eff_3T: "
            f"min={summary['N_eff_3T_min']} "
            f"max={summary['N_eff_3T_max']} "
            f"spread={summary['N_eff_3T_spread']}"
        )
        print(f"  Yp range: {summary['Yp_min']} .. {summary['Yp_max']}")
        print(f"  D/H range: {summary['DH_min']} .. {summary['DH_max']}")
        print(f"  Sigma_H range: {summary['Sigma_H_min']} .. {summary['Sigma_H_max']}")
        print(f"  memory fields present: {summary['memory_fields_present'] or 'NONE'}")
        print("  policy fields sample:")
        for name in summary["policy_fields_present_sample"][:25]:
            print(f"    - {name}")

        print("\nBD199 extracts available:")
        for path in sorted((root / "artifacts/extracts").glob("bd199_*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            top = obj.get("top_level", {})
            print(f"  {path.name}: passed={top.get('passed')} ready={top.get('physical_full_bbn_span_ready')}")
        ''',
        "packet_helper",
        "Cheap artifact summary script required by the external audit runbook.",
        "script",
    )
    write_text(
        "scripts/run_external_audit_smoke.py",
        r'''
        #!/usr/bin/env python3
        """Run or print the packet's cheap external-audit smoke commands."""

        from __future__ import annotations

        import argparse
        import subprocess
        import sys


        COMMANDS = [
            ["python", "scripts/extract_packet_summary.py"],
            ["python", "-m", "pytest", "-q", "tests/test_augmented_pstf_distribution.py"],
            [
                "python", "-m", "pytest", "-q",
                "tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_public_linear_solver_hooks_support_low_rank_payload",
                "tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_materialization_matches_dense_assembly",
                "tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense",
                "tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs",
            ],
            ["python", "-m", "pytest", "-q", "tests/test_block_sparse_jacobian.py"],
            ["python", "-m", "pytest", "-q", "tests/test_three_temperature_closure_invariants.py"],
        ]


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--run", action="store_true", help="execute commands instead of printing them")
            args = parser.parse_args()
            for cmd in COMMANDS:
                printable = "PYTHONPATH=src JAX_PLATFORMS=cpu " + " ".join(cmd)
                print("\n$ " + printable)
                if args.run:
                    env = dict(**__import__("os").environ)
                    env["PYTHONPATH"] = "src"
                    env["JAX_PLATFORMS"] = "cpu"
                    subprocess.run(cmd, check=True, env=env)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        ''',
        "packet_helper",
        "Convenience smoke runner for cheap external audit probes.",
        "script",
    )


def commands_cheap() -> str:
    return r'''
    #!/usr/bin/env bash
    set -euo pipefail

    python -m pip install -e ".[dev]"

    python scripts/extract_packet_summary.py

    PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_augmented_pstf_distribution.py

    PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_public_linear_solver_hooks_support_low_rank_payload \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_materialization_matches_dense_assembly \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs

    PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_block_sparse_jacobian.py

    if [ -f tests/test_three_temperature_closure_invariants.py ]; then
      PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_three_temperature_closure_invariants.py
    fi
    '''


def commands_medium() -> str:
    return r'''
    #!/usr/bin/env bash
    set -euo pipefail

    # Medium probes are templates. Inspect script help before running on a shared machine.
    python scripts/extract_packet_summary.py

    # Dense-LU vs low-rank/Woodbury synthetic parity already has focused unit coverage:
    PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \
      tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs

    # Optional bounded q4 dry-run/smoke template. Confirm flags against --help first:
    PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help

    # Suggested falsifier to add/run if API support exists:
    # - no-projection FLRW collision-source invariant
    # - controlled LRS/non-LRS zero-shear pair with identical h_max/N_span/q-grid/source policy
    '''


def commands_slow() -> str:
    return r'''
    #!/usr/bin/env bash
    set -euo pipefail

    # Slow optional probes. Run only with explicit time/memory budget acceptance.

    # 1. Controlled LRS-radial vs non-LRS zero-shear endpoint pair with identical:
    #    h_max, N_span, q grid, angular settings, source-composition policy,
    #    restart/handoff policy, weak correction level, and initialization metadata.
    #    Use the current runner --help to pin exact flags before execution.
    PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help

    # 2. h_max convergence sweep if the controlled parity pair remains split:
    #    h_max in {0.1, 0.025, 0.00625}.

    # 3. q4/q5/q9/q10 memory scaling with RSS/VmHWM/tracemalloc once row-level
    #    instrumentation exists. q9/q10 can exceed 20 GB based on prior logs.
    '''


def write_packet_docs(git_state: dict[str, str]) -> None:
    status_block = "\n".join(f"        {line}" for line in git_state["status"].splitlines())
    write_text(
        "README_EXTERNAL_AUDIT.md",
        f'''
        # BD281 External Re-Audit Packet

        Date: {PACKET_DATE}

        This packet is for an independent audit of the RABBIT augmented Type-I
        PSTF no-QKE BBN solver. It is not a validation gate and does not claim
        that the solver is public-production ready.

        ## Project Goal

        RABBIT is trying to compute Big Bang nucleosynthesis histories with a
        private augmented Type-I PSTF no-QKE transport model. The current target
        path is CPU-JAX plus the in-tree Rodas5P/AP65 host path, with a
        phase-2 BE/BDF2/Newton network corrector retained unless directly
        falsified. QKE is out of scope.

        "Augmented Type-I PSTF no-QKE" means the code evolves distribution
        distortions in a PSTF/angular-mode representation without solving a
        full quantum kinetic equation. Occupations use the logit convention
        `f(q,n,N) = sigmoid(-(q + A(q,n,N)))`. Collision and closure sources
        are expected to be accumulated in occupation-source space before
        conversion to the augmented `A` variables.

        ## Scope And Nonclaims

        In scope:

        - logit coordinate and occupation-space closure;
        - 3T temperature equations, heavy-flavour bank degeneracy, and
          `N_eff_3T` proxy handling;
        - LRS/non-LRS FLRW-limit parity and FLRW invariant submanifold checks;
        - AP65 dense LU versus block/low-rank/Woodbury/JVP solver paths;
        - RSS/VmHWM/tracemalloc attribution and JSON diagnostic churn;
        - architecture/test debt, including AP65 RHS/span ladder god modules.

        Out of scope:

        - public-production support claims;
        - QKE support;
        - clipping or hiding raw negative/nonfinite evidence;
        - deleting Teff in this packet. Teff is DEPRECATED but import-reachable
          and needs call-graph proof before removal.

        ## Why BD279 And BD280 Appeared To Disagree

        BD279 correctly found that its packet omitted modules needed to close
        energy/N_eff/logit questions. BD280 inspected a fuller code snapshot and
        found that the local physics core looked healthier than BD279 could
        verify. The internal re-audit says both are compatible: local algebraic
        invariants can pass while endpoint-level `N_eff_3T`, parity, solver,
        and memory questions remain unresolved.

        ## Top Unresolved Claims

        1. `N_eff_3T ~= 2.994` is not settled and is probably not q/angular
           discretization alone.
        2. LRS/non-LRS FLRW-limit parity is unresolved.
        3. AP65 endpoint host still uses dense LU even though block/low-rank/
           Woodbury pieces exist.
        4. RSS/VmHWM/tracemalloc attribution is missing from endpoint rows.
        5. AP65 RHS/span ladder and validation plumbing are development-speed
           blockers.
        6. Count-lock tests are weak physics evidence.
        7. Collisional `ell_max=2` exactness must be fenced to its valid
           collisionless/free-streaming regime.

        ## Install And Run

        From the extracted packet root:

        ```bash
        python -m pip install -e ".[dev]"
        bash COMMANDS_CHEAP.sh
        ```

        Medium and slow probes are in `COMMANDS_MEDIUM.sh` and
        `COMMANDS_SLOW_OPTIONAL.sh`. Run slow probes only after accepting their
        time and memory cost.

        Evidence means source, test, artifact, and command output tied to exact
        paths and hashes. Smoke/demo output is not research validation.

        ## Git Snapshot At Packet Creation

        - Branch: `{git_state["branch"]}`
        - HEAD: `{git_state["head"]}`
        - Status summary:

        ```text
{status_block}
        ```
        ''',
        "packet_doc",
        "Top-level external audit README required by user.",
    )

    write_text(
        "AUDIT_SCOPE_AND_NONCLAIMS.md",
        '''
        # Audit Scope And Nonclaims

        This packet prepares an external audit. It does not validate the solver.

        Claim labels:

        - IMPLEMENTED: code exists and has been executed/tested.
        - VALIDATED: independently checked by test, benchmark, derivation, or
          reproducible artifact.
        - DERIVED: mathematically derived with assumptions stated.
        - SPECIFIED: defined in a design/spec document but not yet implemented.
        - PROPOSED: plausible research direction, not yet derived or implemented.
        - SPECULATIVE: unsupported.
        - DEPRECATED: superseded and should not guide implementation.
        - FORBIDDEN: explicitly disallowed pattern or assumption.

        Hard boundaries:

        - QKE is out of scope.
        - No public-production support is claimed.
        - CPU-JAX plus in-tree Rodas5P/AP65 remains the target unless falsified.
        - The phase-2 BE/BDF2/Newton corrector stays unless directly falsified.
        - Raw negative/nonfinite evidence must be preserved.
        - `N_eff_3T` is a proxy until one physical definition is pinned.
        - Teff is DEPRECATED and import-reachable; deletion needs call-graph proof.
        - Packet manifests are external-audit packaging, not project gates.
        ''',
        "packet_doc",
        "Explicit scope and nonclaim boundary.",
    )

    write_text(
        "CLAIMS_TO_VERIFY.md",
        '''
        # Claims To Verify

        | ID | Claim | Current internal verdict | Source document | Files to inspect | Cheap command | Optional medium/slow command | Expected possible outcomes | What would falsify the claim | What would change the PR plan |
        |---|---|---|---|---|---|---|---|---|---|
        | C1 | Logit convention: `f=sigmoid(-(q+A))`, `dA/dN=-df/(f*(1-f))`. | SUPPORTED locally | BD279, BD280, internal re-audit | `src/rabbit/transport/augmented_pstf_distribution.py`, tests | `pytest -q tests/test_augmented_pstf_distribution.py` | Pass means local coordinate contract holds; fail means source mapping is suspect. | Any sign/scaling mismatch. | Move physics invariant PR before solver PR. |
        | C3 | Heavy-bank degeneracy/sign is locally correct but endpoint parity remains unresolved. | PARTIAL | BD279, BD280 | `nudec_coupled.py`, `nudec_tables.py`, bridge | `pytest -q tests/test_three_temperature_closure_invariants.py` | Local sign/degeneracy passes or fails. | Positive `dQ_nux_bank_N` cools heavy bank relative to free streaming or degeneracy enters twice/zero. | If falsified, heavy-bank repair outranks solver wiring. |
        | C4 | `N_eff_3T ~= 2.994` is not explained by q/angular discretization alone. | CONTRADICTED/PARTIAL for discretization-only | BD279, internal re-audit | BD278 extract | `python scripts/extract_packet_summary.py` | Small spread supports non-discretization explanation. | Fresh controlled q/angular sweep moves the value materially. | Reopen q/angular convergence branch. |
        | C5 | LRS/non-LRS zero-shear parity requires a controlled fresh run. | PARTIAL | BD279, internal re-audit | span ladder, AP65 RHS, artifacts | source/artifact inspection | controlled parity pair | Same result confirms parity; split attributes blocker. | Identical controlled rows agree within tolerance. | If parity passes, shift to hmax/source-policy/N_eff definition. |
        | C6 | Monopole projection may hide an unprojected FLRW collision-source bug. | UNTESTED/PARTIAL | internal re-audit red team | bridge, weak network | existing projected invariant test | no-projection FLRW source test | Structural invariant holds or projection is hiding leakage. | Unprojected anisotropic source/shear exceeds tolerance. | Move source projection repair up. |
        | C7 | AP65 host dense LU remains the endpoint solve path. | SUPPORTED | BD280, internal re-audit | AP65 RHS, JAX Rodas5P | source inspection | AP65 solver A/B probe | Dense path confirmed or contradicted. | Endpoint runner uses block/low-rank without dense W. | If already wired, PR-1 becomes profiling/parity. |
        | C8 | Low-rank/Woodbury/block-sparse pieces exist and pass unit algebra, but endpoint wiring is not proven. | SUPPORTED/PARTIAL | BD280 | `solver_jax_rodas5p.py`, `linear_solve_strategies.py`, tests | low-rank/block tests in cheap tier | AP65 endpoint A/B | Unit pass; endpoint unknown. | Unit algebra fails or endpoint wiring already verified. | Reorder solver PR. |
        | C9 | RSS/VmHWM/tracemalloc is missing from endpoint artifacts. | CONTRADICTED presence; missing fields | internal re-audit | artifacts | `python scripts/extract_packet_summary.py` | memory-instrumented q4/q9 row | Missing fields confirmed. | Existing row-level fields found. | If present, analyze before adding instrumentation. |
        | C10 | Geometry/thermo may dominate rejections; do not optimize only collision payload blindly. | PARTIAL/SUPPORTED | BD279/BD280 | artifacts, AP65 telemetry | artifact inspection | fresh q4 row with rejection telemetry | Dominance confirmed or contradicted. | Rejections dominated by collision payload only. | Reprioritize optimization target. |
        | C11 | Continuous-vs-piecewise target ambiguity must be resolved. | PARTIAL/SUPPORTED | BD280 | AP65 RHS/span ladder/docs | source/docs inspection | fresh labeled endpoint row | Current target clarified or ambiguous. | Code already clearly labels publication/diagnostic path. | Adjust architecture PR. |
        | C12 | `ell_max=2` exactness is overbroad in collisional augmented runtime. | CONTRADICTED if overbroad | BD279/BD280/internal | config/conventions, transport docs | source inspection | future ell hierarchy probe | Claim fenced or overbroad. | Evidence shows collisional higher multipoles cannot couple. | If exactness holds, downgrade ell hierarchy work. |
        | C13 | Teff is deprecated but import-reachable; deletion needs call-graph proof. | PARTIAL | BD280/internal | `teff_*`, config, tests | `rg -n "Teff|teff"` | call-graph/import deletion dry-run | Import reachability confirmed. | No imports/tests depend on Teff. | If unreachable, cleanup PR can delete sooner. |
        | C14 | Count-lock tests do not prove physics. | SUPPORTED critique | BD280/internal | span ladder tests | source inspection | replace with invariant tests | Count locks identified. | Count locks are paired with physics invariants and fail physically. | Test overhaul can be narrower. |
        | C15 | CPU-JAX/Rodas5P should not be abandoned before profiling and block/low-rank wiring. | SUPPORTED | internal | solver/perf files | cheap solver tests | RSS/profiling + endpoint low-rank A/B | In-stack options remain plausible. | Profiling shows stable residual kernel after block/low-rank wiring. | Only then consider rewrite. |
        ''',
        "packet_doc",
        "Claim verification table required by user.",
    )

    write_text(
        "REPRODUCTION_RUNBOOK.md",
        '''
        # Reproduction Runbook

        Run commands from the extracted packet root.

        ## Cheap Tier

        ```bash
        python -m pip install -e ".[dev]"
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_augmented_pstf_distribution.py
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_public_linear_solver_hooks_support_low_rank_payload tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_materialization_matches_dense_assembly tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_block_sparse_jacobian.py
        ```

        If `tests/test_three_temperature_closure_invariants.py` exists:

        ```bash
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_three_temperature_closure_invariants.py
        ```

        Cheap artifact inspection:

        ```bash
        python scripts/extract_packet_summary.py
        ```

        This reports BD278 row count, min/max/spread of `N_eff_3T`, available
        `Yp`, `D/H`, `Sigma_H`, whether RSS/VmHWM/tracemalloc fields exist, and
        policy fields if present.

        ## Medium Tier

        - Fresh q4 endpoint smoke or dry-run if a bounded command exists.
        - Dense-LU vs low-rank/Woodbury synthetic AP65-like system parity.
        - No-projection FLRW collision-source invariant if API supports it.
        - LRS/non-LRS zero-shear controlled pair only if runtime is bounded
          enough; otherwise use as slow template.

        Start with:

        ```bash
        bash COMMANDS_MEDIUM.sh
        ```

        ## Slow Optional Tier

        Run only with explicit time/memory budget:

        - controlled LRS-radial vs non-LRS zero-shear endpoint pair with
          identical solver, h-max, N-span, q grid, angular settings,
          source-composition policy, restart/handoff policy, weak correction
          level, and initialization metadata;
        - `h_max in {0.1, 0.025, 0.00625}` if parity remains split;
        - q4/q5/q9/q10 memory scaling with RSS/VmHWM/tracemalloc;
        - q9/q10 endpoints only if high memory/time cost is accepted.

        Start with:

        ```bash
        bash COMMANDS_SLOW_OPTIONAL.sh
        ```

        ## Troubleshooting

        - Use Python 3.10+.
        - Use `JAX_PLATFORMS=cpu` for repeatability.
        - `/proc/self/status:VmHWM` exists on Linux; use `resource.getrusage`
          elsewhere.
        - Do not interpret smoke output as physical validation.
        ''',
        "packet_doc",
        "Exact command tiers required by user.",
    )

    write_text(
        "EXPECTED_FINDINGS_AND_OPEN_QUESTIONS.md",
        '''
        # Expected Findings And Open Questions

        Expected supported findings:

        - local logit coordinate and `df -> dA` conversion should pass;
        - occupation-space closure should occur before `dA` conversion;
        - Python/JAX `N_eff_3T` code definitions should match locally;
        - low-rank/Woodbury/block-sparse unit algebra should pass;
        - AP65 endpoint host dense LU should still be visible in source;
        - endpoint artifacts likely lack row-level RSS/VmHWM/tracemalloc.

        Open questions:

        - Is endpoint `N_eff_3T ~= 2.994` a non-LRS parity defect, hmax/chaining
          artifact, source-policy issue, or proxy-definition issue?
        - Does unprojected FLRW collision preserve the isotropic submanifold?
        - Can block/low-rank/Woodbury be wired into AP65 endpoint rows without
          observable drift?
        - What actually causes q9/q10 memory: dense LU, JAX compile/runtime,
          collision arrays, caches, JSON diagnostics, or a mixture?
        - Should continuous single-RHS remain a primary target, or be demoted to
          diagnostic while piecewise phase-1/phase-2 is the endpoint target?
        ''',
        "packet_doc",
        "Concise expectation setting for external auditor.",
    )

    write_text(
        "EXTERNAL_AUDITOR_PROMPT.md",
        '''
        # External Re-Audit Prompt - RABBIT Augmented Type-I PSTF No-QKE BBN Solver

        You are auditing an extracted external packet for the RABBIT augmented
        Type-I PSTF no-QKE BBN solver.

        ## First Steps

        1. Extract the zip.
        2. Read `README_EXTERNAL_AUDIT.md`, `AUDIT_SCOPE_AND_NONCLAIMS.md`,
           `CLAIMS_TO_VERIFY.md`, and `REPRODUCTION_RUNBOOK.md`.
        3. Run cheap commands first.
        4. Inspect source and artifacts before making claims.

        ## Method: CRAG

        Use corrective retrieval-augmented audit:

        1. Extract claims.
        2. Retrieve exact source/test/artifact evidence.
        3. Classify each claim as SUPPORTED / CONTRADICTED / PARTIAL / STALE /
           UNTESTED / UNKNOWN.
        4. Convert unresolved hypotheses into executable probes.

        ## Method: Chain-of-Code

        For every nontrivial physics, numerics, or performance hypothesis, write
        or run the smallest executable probe feasible. Record commands, exit
        codes, outputs, and changed files. Do not present toy/demo outputs as
        research evidence.

        ## Role Subagents

        Use role subagents if available. Do not reveal hidden chain-of-thought.
        Ask agents for concise reasoning summaries, assumptions, evidence,
        commands, outputs, failures, and recommended actions only.

        Roles:

        1. Physics Invariant Auditor
        2. Numerical Solver Auditor
        3. Performance/Memory Auditor
        4. Architecture/Test Auditor
        5. Reproducibility Engineer
        6. Evidence/Artifact Auditor
        7. Red-Team Reviewer
        8. Final Verdict Editor

        Wait for all subagents before final synthesis. If subagents are
        unavailable, simulate the same roles sequentially. Do not use
        Explorer-style freeform exploration as the main method.

        ## Constraints

        - QKE remains out of scope.
        - Do not claim public-production readiness.
        - CPU-JAX plus in-tree Rodas5P/AP65 remains the target unless you
          explicitly falsify that as a hypothesis.
        - Keep phase-2 BE/BDF2/Newton corrector unless directly falsified.
        - Preserve raw negative/nonfinite evidence. Do not clip outputs.
        - Treat `N_eff_3T` as a proxy until one physical definition is pinned.
        - Teff is DEPRECATED and import-reachable; do not delete without
          call-graph proof.
        - Do not recommend a language rewrite unless profiling falsifies the
          in-stack block/low-rank/JAX path.

        ## Main Questions

        1. Are the internal re-audit's top claims correct?
        2. Did the packet include enough files to close BD279's prior
           energy/N_eff/logit material gap?
        3. Does local physics algebra pass independent checks?
        4. Is endpoint-level `N_eff_3T` still unresolved?
        5. Is LRS/non-LRS FLRW-limit parity the right next falsification target?
        6. Is AP65 dense LU truly still the endpoint path?
        7. Are block/low-rank/Woodbury pieces actually available and testable?
        8. Is memory attribution absent, and what is the smallest useful
           instrumentation?
        9. Are architecture/test problems real development blockers?
        10. Is the six-PR acceleration plan correctly prioritized, or should it
            be reordered?

        ## Required Final Auditor Output

        - Executive verdict: PASS / MAJOR REVISION / BLOCKED / INCONCLUSIVE.
        - One-page summary.
        - Claim ledger.
        - Commands run and outputs.
        - Falsification matrix.
        - Source-level findings with file/line references.
        - Artifact-level findings with paths and hashes.
        - Performance/memory attribution status.
        - Red-team objections.
        - Revised PR plan, max 6 PRs.
        - Missing files or missing evidence.
        - Exact recommended next commands.
        ''',
        "packet_doc",
        "Self-contained prompt for external auditor agents.",
    )


def write_command_files() -> None:
    for name, content in [
        ("COMMANDS_CHEAP.sh", commands_cheap()),
        ("COMMANDS_MEDIUM.sh", commands_medium()),
        ("COMMANDS_SLOW_OPTIONAL.sh", commands_slow()),
    ]:
        write_text(name, content, "packet_command", f"{name} required command tier.", "script")
        os.chmod(PACKET_ROOT / name, 0o755)


def write_indexes() -> None:
    def listing(base: str, title: str) -> str:
        root = PACKET_ROOT / base
        files = sorted(p.relative_to(PACKET_ROOT).as_posix() for p in root.rglob("*") if p.is_file())
        body = "\n".join(f"- `{f}`" for f in files)
        return f"# {title}\n\n{body}\n"

    write_text("source_snapshot/FILE_LIST.md", listing("src", "Source Snapshot File List"), "source_index", "Source listing for root src/ copy.")
    write_text("tests_snapshot/FILE_LIST.md", listing("tests", "Tests Snapshot File List"), "test_index", "Test listing for root tests/ copy.")
    write_text("docs_snapshot/FILE_LIST.md", listing("docs_snapshot", "Docs Snapshot File List"), "doc_index", "Docs listing.")


def write_skipped_artifacts_note() -> None:
    candidates = [
        (
            "diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.json",
            "reduced to artifacts/extracts/bd278_endpoint_matrix_shard_1_summary.json",
        ),
        (
            "audit_outputs/BD280_external_full_code_audit_minimal_2026-06-01/evidence/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.json",
            "duplicate of the BD278 full shard; reduced extract included from diagnostic_outputs",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_on_laguerre_q4_q5_N4_wt240.json",
            "reduced to artifacts/extracts/bd199_collision_on_laguerre_q4_q5_N4_wt240_extract.json",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_on_radial_q3_wt240.json",
            "reduced to artifacts/extracts/bd199_collision_on_radial_q3_wt240_extract.json",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_off_standard_anchor_wt240.json",
            "reduced to artifacts/extracts/bd199_collision_off_standard_anchor_wt240_extract.json",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/BD199_flrw_collision_external_audit_packet_2026-05-27.zip",
            "nested zip skipped; prompt/docs/logs and reduced extracts are included instead",
        ),
        (
            "audit_outputs/BD280_external_full_code_audit_minimal_2026-06-01/evidence/",
            "full prior packet evidence tree skipped to avoid duplicate large artifacts; curated raw logs and extracts included",
        ),
    ]
    rows = [
        "# Skipped Or Reduced Large Artifacts",
        "",
        "This packet is size-bounded. Large raw artifacts are either reduced with",
        "provenance in `artifacts/extracts/` or omitted when they duplicate included",
        "evidence. Virtual environments, caches, bytecode, build products, nested",
        "zips, and private-looking credential/token files are excluded.",
        "",
        "| Source path | Size bytes | Action |",
        "|---|---:|---|",
    ]
    for rel, action in candidates:
        path = ROOT / rel
        size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
        rows.append(f"| `{rel}` | {size} | {action} |")
    write_text(
        "artifacts/SKIPPED_LARGE_ARTIFACTS.md",
        "\n".join(rows) + "\n",
        "artifact_note",
        "Documents size-bounded artifact omissions and reductions.",
        "doc",
    )


def write_manifest() -> None:
    # First write payload entries. Manifest self-hashes are intentionally recorded
    # as self-referential notes because a file cannot contain its own stable hash.
    payload_entries = [asdict(e) for e in entries]
    manifest_note = {
        "packet_name": PACKET_NAME,
        "created_date": PACKET_DATE,
        "manifest_self_reference_note": (
            "PACKET_MANIFEST.md and PACKET_MANIFEST.json are generated control "
            "files. Per-file hashes are provided for payload files; verify the "
            "whole submitted archive with the adjacent .zip.sha256 file."
        ),
        "entries": payload_entries,
    }
    manifest_json_rel = Path("PACKET_MANIFEST.json")
    manifest_md_rel = Path("PACKET_MANIFEST.md")
    (PACKET_ROOT / manifest_json_rel).write_text(json.dumps(manifest_note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        "# Packet Manifest",
        "",
        f"Packet: `{PACKET_NAME}`",
        "",
        "Manifest files are generated control files; verify the full archive with the adjacent `.zip.sha256`.",
        "",
        "| Packet path | Original path | SHA256 | Size | Category | Kind | Why |",
        "|---|---|---|---:|---|---|---|",
    ]
    for entry in entries:
        rows.append(
            f"| `{entry.packet_path}` | `{entry.original_repo_path}` | `{entry.sha256}` | {entry.size_bytes} | "
            f"{entry.category} | {entry.kind} | {entry.why_included.replace('|', '/')} |"
        )
    rows.append("")
    rows.append("## Artifact Extract Provenance")
    rows.append("")
    rows.append("| Packet path | Source artifact | Source SHA256 | Retained fields | Dropped fields |")
    rows.append("|---|---|---|---|---|")
    for entry in entries:
        if entry.artifact_source_path:
            rows.append(
                f"| `{entry.packet_path}` | `{entry.artifact_source_path}` | `{entry.artifact_source_sha256}` | "
                f"{', '.join(entry.retained_fields or [])} | {', '.join(entry.dropped_fields or [])} |"
            )
    (PACKET_ROOT / manifest_md_rel).write_text("\n".join(rows) + "\n", encoding="utf-8")


def zip_packet() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKET_ROOT.rglob("*")):
            if path.is_file():
                arc = Path(PACKET_NAME) / path.relative_to(PACKET_ROOT)
                zf.write(path, arc.as_posix())
    ZIP_SHA_PATH.write_text(f"{sha256_file(ZIP_PATH)}  {ZIP_PATH.name}\n", encoding="utf-8")


def main() -> int:
    if PACKET_ROOT.exists():
        shutil.rmtree(PACKET_ROOT)
    PACKET_ROOT.mkdir(parents=True, exist_ok=True)

    git_state = {
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["log", "-1", "--oneline"]),
        "status": run_git(["status", "--short"]) or "(clean)",
    }

    # Root/environment files.
    for rel in ["AGENTS.md", "README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md", "PROMOTION_GATES.md", "pyproject.toml"]:
        copy_file(rel, rel, "governance_environment", "Root governance, status, or install context.", "doc")

    # Full runnable code/test snapshot, excluding caches and bytecode.
    copy_tree("src", "src", "source", "Full source snapshot to avoid prior missing-module gaps.", "source")
    copy_tree("tests", "tests", "tests", "Relevant and surrounding tests for cheap probes and audit inspection.", "test")
    copy_tree("scripts", "scripts", "project_scripts", "Project scripts including endpoint runner and packet builder.", "script")

    docs = [
        "docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md",
        "docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md",
        "docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md",
        "docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md",
        "docs/audit/BD186_current_collision_blocker_external_review_packet_2026-05-27.md",
        "docs/audit/BD186_external_audit_report1_collision_relaxation_2026-05-27.md",
        "docs/audit/BD186_external_audit_report2_radial_coordinate_2026-05-27.md",
        "docs/audit/BD199_recent_development_and_packet_manifest_2026-05-27.md",
        "docs/audit/BD199_flrw_laguerre_collision_drift_external_audit_prompt_2026-05-27.md",
        "docs/audit/BD277_endpoint_matrix_sharding_runtime_enabler_2026-06-01.md",
        "docs/audit/BD278_endpoint_matrix_shard_runtime_threading_2026-06-01.md",
    ]
    for rel in docs:
        copy_file(rel, f"docs_snapshot/{rel}", "docs", "Guardrail, future plan, provenance, or audit context.", "doc", mandatory=False)

    audit_docs = [
        "BD279_external_audit_report_2026-06-02.md",
        "BD280_external_full_code_audit_report_2026-06-02.md",
        "BD279_external_audit_packet_2026-06-01.md",
        "BD280_external_full_code_audit_prompt_2026-06-01.md",
        "internal_reaudit_report.md",
        "hypothesis_falsification_matrix.md",
        "skill_and_subagent_usage.md",
        "pr_acceleration_plan.md",
    ]
    for rel in audit_docs:
        copy_file(rel, f"docs_snapshot/audits/{rel}", "prior_internal_audits", "Prior or internal audit document.", "doc", mandatory=False)

    # Curated artifact copies and extracts.
    summarize_bd278()
    for src_rel, dest, why in [
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_off_standard_anchor_wt240.json",
            "artifacts/extracts/bd199_collision_off_standard_anchor_wt240_extract.json",
            "Reduced BD199 collision-off FLRW anchor evidence.",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_on_radial_q3_wt240.json",
            "artifacts/extracts/bd199_collision_on_radial_q3_wt240_extract.json",
            "Reduced BD199 q3 radial collision-on evidence.",
        ),
        (
            "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_on_laguerre_q4_q5_N4_wt240.json",
            "artifacts/extracts/bd199_collision_on_laguerre_q4_q5_N4_wt240_extract.json",
            "Reduced BD199 q4/q5 laguerre collision-on evidence.",
        ),
    ]:
        if (ROOT / src_rel).exists():
            summarize_artifact(src_rel, dest, why)

    for rel in [
        "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_audit_summary.json",
        "diagnostic_outputs/bd199_flrw_collision_audit/flrw_collision_q4_q5_cases.json",
        "diagnostic_outputs/bd199_flrw_collision_audit/logs/flrw_collision_off_standard_anchor_wt240.log",
        "diagnostic_outputs/bd199_flrw_collision_audit/logs/flrw_collision_on_radial_q3_wt240.log",
        "diagnostic_outputs/bd199_flrw_collision_audit/logs/flrw_collision_on_laguerre_q4_q5_N4_wt240.log",
        "diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.stderr.log",
        "diagnostic_outputs/bd278_endpoint_matrix_shards/bd278_endpoint_matrix_shard_1_of_4.stdout.json",
        "audit_outputs/BD279_external_audit_packet_2026-06-01/BD279_artifact_summary.json",
    ]:
        copy_file(rel, f"artifacts/raw/{rel}", "artifact_raw_small", "Small curated raw artifact/log for independent inspection.", "artifact", mandatory=False)

    json_dump(
        "artifacts/internal_reaudit_probe_output_summary.json",
        {
            "source": "internal_reaudit_report.md and hypothesis_falsification_matrix.md",
            "commands_reported": [
                "tests/test_three_temperature_closure_invariants.py: 3 passed",
                "low-rank Rodas5P targeted tests: 4 passed",
                "tests/test_block_sparse_jacobian.py: 11 passed",
                "PSTF distribution plus selected collision bridge tests: 14 passed",
            ],
            "note": "This file summarizes prior internal probe output. External auditors should rerun cheap commands.",
        },
        "artifact_extract",
        "Small summary of internal re-audit command outputs.",
    )

    write_packet_docs(git_state)
    write_helper_scripts()
    write_command_files()
    write_indexes()
    write_skipped_artifacts_note()
    write_manifest()
    zip_packet()

    print(f"packet_dir={PACKET_ROOT}")
    print(f"zip_path={ZIP_PATH}")
    print(f"sha256={sha256_file(ZIP_PATH)}")
    print(f"entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
