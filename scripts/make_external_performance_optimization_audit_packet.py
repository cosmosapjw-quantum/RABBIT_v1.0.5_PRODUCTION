#!/usr/bin/env python3
"""Build the BD282 external performance optimization audit packet.

This script is packaging-only.  It does not add a runtime validation gate and
does not claim solver validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKET_NAME = "BD282_external_performance_optimization_audit_packet_2026-06-02"
PACKET_DIR = Path("audit_packets") / PACKET_NAME
ZIP_PATH = PACKET_DIR.with_suffix(".zip")
SHA_PATH = Path(str(ZIP_PATH) + ".sha256")

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive packet metadata.
        return f"unavailable: {type(exc).__name__}: {exc}"


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    ensure_parent(path)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SKIP_DIR_NAMES}

    shutil.copytree(src, dst, ignore=ignore)


def safe_load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


PERF_ROW_KEYS = {
    "resolution_case_label",
    "completion_class",
    "span_ladder_passed",
    "physical_full_bbn_span_ready",
    "T_final_MeV",
    "N_eff_3T",
    "Yp",
    "DH",
    "Sigma_H",
    "q_laguerre_order",
    "q_node_count",
    "N_mu",
    "N_phi",
    "N_span",
    "N_span_end",
    "N_span_end_ladder",
    "h_max",
    "requested_h_max",
    "max_steps",
    "rtol",
    "atol",
    "wall_time_budget_seconds",
    "selected_wall_seconds_total",
    "selected_dynamic_collision_payload_builds_total",
    "selected_dynamic_collision_payload_build_wall_seconds_total",
    "selected_dynamic_collision_payload_build_attempts_total",
    "selected_dynamic_collision_payload_build_attempt_wall_seconds_total",
    "selected_source_evaluations_total",
    "selected_stage_source_evaluations_total",
    "selected_stage_collision_payload_reuse_total",
    "selected_linear_system_factorizations_total",
    "selected_linear_system_solves_total",
    "selected_frozen_source_jax_full_jvp_jacobian_evaluations_total",
    "selected_frozen_source_jax_block_jvp_jacobian_evaluations_total",
    "selected_frozen_source_finite_difference_jacobian_evaluations_total",
    "selected_host_lagged_jacobian_eligible_total",
    "selected_host_lagged_jacobian_reuse_total",
    "selected_host_lagged_jacobian_refresh_total",
    "selected_host_lagged_jacobian_forced_refresh_total",
    "selected_phase2_conservative_extent_corrector_step_count_total",
    "selected_phase2_conservative_extent_corrector_newton_iteration_count_total",
    "selected_phase2_conservative_extent_corrector_newton_linear_solve_count_total",
    "selected_phase2_conservative_extent_corrector_newton_residual_evaluation_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_attempt_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_used_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_rejected_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_displacement_guard_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_displacement_guard_rejected_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_throttle_count_total",
    "selected_phase2_conservative_extent_corrector_ab2_newton_initial_guess_throttle_skipped_count_total",
    "selected_phase2_conservative_extent_raw_candidate_negative_event_count_total",
    "selected_phase2_conservative_extent_raw_candidate_negative_value_count_total",
    "jacobian_policy",
    "linear_system_backend",
    "linear_system_backend_source",
    "linear_system_is_dense",
    "linear_system_factorization_policy",
    "linear_system_low_rank_active",
    "linear_system_low_rank_rank",
    "W_shape",
    "J_shape",
    "linear_system_matrix_shape",
    "jacobian_shape",
    "linear_system_dtype",
    "ru_maxrss_kb",
    "vmhwm_kb",
    "tracemalloc_peak_bytes",
    "tracemalloc_current_bytes",
    "memory_telemetry_error",
    "memory_telemetry_platform",
    "collision_projection_policy",
    "chain_h_max_policy",
    "chain_restart_handoff",
    "collision_source_component_policy",
    "source_composition_policy",
    "stage_collision_payload_policy",
    "initial_np_policy",
    "initial_A_monopole_offset",
    "phase2_conservative_extent_corrector_enabled",
    "phase2_network_background_policy",
    "phase2_network_newton_initial_guess_policy",
}


def reduced_row(row: dict[str, Any]) -> dict[str, Any]:
    reduced = {key: row.get(key) for key in PERF_ROW_KEYS if key in row}
    source_refresh = row.get("source_refresh_config")
    if isinstance(source_refresh, dict):
        reduced["source_refresh_config"] = {
            key: source_refresh.get(key)
            for key in (
                "collision_projection_policy",
                "collision_source_composition_policy",
                "angular_distribution_source_energy_policy",
                "angular_logit_conditioning_policy",
                "phase2_window_reference_policy",
                "payload_metadata_policy",
                "pstf_radial_energy_normalization",
                "qed_correction_model",
            )
            if key in source_refresh
        }
        if "collision_projection_policy" not in reduced and "collision_projection_policy" in source_refresh:
            reduced["collision_projection_policy"] = source_refresh["collision_projection_policy"]
    span_summary = row.get("span_summary")
    if isinstance(span_summary, dict):
        reduced["span_summary_subset"] = {
            key: span_summary.get(key)
            for key in (
                "adaptive_attempt_count_total",
                "adaptive_rejected_attempt_count_total",
                "adaptive_acceptance_ratio_min",
                "adaptive_rejection_ratio_max",
                "attempt_wall_seconds_max",
                "best_T_final_MeV",
                "blocking_next_step",
            )
            if key in span_summary
        }
    span_rows = row.get("span_rows")
    if isinstance(span_rows, list):
        compact_span_rows: list[dict[str, Any]] = []
        for span_row in span_rows:
            if not isinstance(span_row, dict):
                continue
            compact_span_rows.append(
                {
                    key: span_row.get(key)
                    for key in (
                        "N_start",
                        "N_end",
                        "T_gamma_final_MeV",
                        "T_final_MeV",
                        "N_eff_3T",
                        "Yp",
                        "DH",
                        "Sigma_H",
                        "full_bbn_completed",
                        "passed",
                        "wall_seconds",
                        "attempt_count",
                        "n_rejected",
                        "dominant_block",
                    )
                    if key in span_row
                }
            )
        reduced["span_rows_subset"] = compact_span_rows
    return reduced


def extract_artifact(src: Path, dst: Path) -> dict[str, Any]:
    data = safe_load_json(src)
    source_sha = sha256_file(src)
    extract: dict[str, Any] = {
        "source_path": str(src),
        "source_sha256": source_sha,
        "extraction_command": f"python scripts/make_external_performance_optimization_audit_packet.py extract {src}",
        "retained_fields": sorted(PERF_ROW_KEYS),
        "dropped_fields": "large nested diagnostics, raw arrays, and repeated trace details unless already scalar counters",
        "reason_for_reduction": "Keep wall/collision/solver/corrector/memory/provenance/endpoint fields while avoiding multi-MB trace duplication.",
    }
    if isinstance(data, dict):
        extract["top_level"] = {
            key: data.get(key)
            for key in (
                "created_utc",
                "artifact_path",
                "artifact_payload_sha256",
                "passed",
                "physical_full_bbn_span_ready",
                "public_dispatch_ready",
                "production_smc_validation_ready",
                "qke_scope",
                "claim_scope",
                "implementation_stage",
            )
            if key in data
        }
        rows = data.get("rows")
        if isinstance(rows, list):
            extract["row_count"] = len(rows)
            extract["rows"] = [reduced_row(row) for row in rows if isinstance(row, dict)]
        summary = data.get("summary")
        if isinstance(summary, dict):
            extract["summary_subset"] = {
                key: summary.get(key)
                for key in (
                    "classified_rows",
                    "blocking_next_step",
                    "full_bbn_completed_row_count",
                    "rows_full_bbn_completed",
                    "N_eff_3T_min",
                    "N_eff_3T_max",
                    "T_final_MeV_min",
                    "T_final_MeV_max",
                )
                if key in summary
            }
    else:
        extract["parse_status"] = "json parse failed or unsupported root type"
    write_text(dst, json_dump(extract))
    return {
        "source_artifact_path": str(src),
        "source_sha256": source_sha,
        "extraction_command": extract["extraction_command"],
        "retained_fields": extract["retained_fields"],
        "dropped_fields": extract["dropped_fields"],
        "reason_for_reduction": extract["reason_for_reduction"],
    }


def build_docs(missing_inputs: list[str]) -> dict[str, str]:
    missing_text = "\n".join(f"- `{item}`" for item in missing_inputs) or "- None detected during packet build."
    return {
        "README_EXTERNAL_PERFORMANCE_AUDIT.md": f"""
        # External Performance Optimization Audit Packet

        Packet: `{PACKET_NAME}`

        Build date: 2026-06-02

        Git branch: `{run_git(['branch', '--show-current'])}`

        Git HEAD: `{run_git(['rev-parse', '--short', 'HEAD'])} {run_git(['log', '-1', '--format=%s'])}`

        ## What This Project Computes

        RABBIT is an augmented Type-I PSTF no-QKE BBN solver.  The current line
        couples a Type-I geometric/transport state, momentum/angular neutrino
        distribution modes, three-temperature neutrino/plasma thermodynamics,
        weak rates, a phase-2 nuclear network corrector, and an in-tree
        Rodas5P/AP65 host path.  QKE is out of scope.

        ## Why This Performance Audit Exists

        The current performance concern is not yet proven to be Python language
        overhead.  The working hypothesis is that the slowdown comes from live
        collision payload construction, phase-2 BE/BDF2/Newton corrector work,
        host Rodas5P orchestration, rejected-step coupling, dense AP65 linear
        solves, and diagnostic/JSON churn.  A full rewrite to C++/Rust/Julia is
        therefore not the default recommendation.

        Recent q4 FLRW collision activation claims to verify from shipped
        artifacts include a partial run to about `N=2.75`, `T_gamma ~= 0.07 MeV`,
        total wall about 752s, dynamic collision payload build wall about 307s,
        about 5471 payload builds, about 5520 linear solves, about 24595
        phase-2 Newton iterations, and zero full endpoint rows.

        ## Benchmark Context Warning

        AlterBBN, PArthENoPE, PRIMAT, PRyMordial, and LINX provide useful scale
        context for standard BBN.  They are not apples-to-apples workloads for a
        live augmented no-QKE neutrino collision/source evolution path.  Standard
        BBN runtimes must not be used as rewrite proof unless workload
        differences are stated explicitly.

        Baseline context for the external auditor to independently verify:

        - AlterBBN is a C standard-BBN code family and is useful for fast
          network/rate scale context.
        - PArthENoPE is a Fortran standard-BBN code family and is useful for
          established endpoint-output comparison context.
        - PRIMAT is an accuracy-focused standard-BBN reference and can be slower
          depending on settings; it is not a live Type-I collision workload.
        - PRyMordial targets precision BBN/CMB-era calculations but is still not
          the same live augmented collision/source path.
        - LINX shows that Python/JAX can be a fast BBN stack after compilation,
          but its standard workflows do not prove this repo's live collision
          payload and phase-2 corrector orchestration is efficient.

        If any benchmark number is used outside this audit, verify the citation
        and state the workload difference explicitly.

        ## Suspected Bottlenecks To Audit

        1. dynamic collision payload build/rebuild;
        2. phase-2 BE/BDF2/Newton corrector iteration count;
        3. host Rodas5P orchestration and rejected-step coupling;
        4. dense AP65 host LU solve;
        5. full-JVP/block-JVP routing gap;
        6. JAX compile/runtime separation;
        7. diagnostic JSON/trace construction in hot loops;
        8. retained caches and q-dependent collision tensors;
        9. missing or incomplete row-level memory telemetry in older artifacts;
        10. unresolved endpoint physics parity, so speedups must not become
            defaults before correctness guards.

        ## Key Warning

        Do not make the solver wrong faster.

        ## Missing Inputs At Build Time

        {missing_text}

        ## How To Start

        Run:

        ```bash
        python -m pip install -e ".[dev]"
        bash COMMANDS_CHEAP.sh
        ```

        Then read `PROFILING_AND_BENCHMARK_RUNBOOK.md` before any medium run.
        Treat cheap probes as smoke/profiling evidence only, not solver
        validation.
        """,
        "AUDIT_SCOPE_AND_NONCLAIMS.md": """
        # Audit Scope And Nonclaims

        ## In Scope

        - Performance attribution for the augmented Type-I PSTF no-QKE BBN path.
        - CPU-JAX plus in-tree Rodas5P/AP65 host behavior.
        - Dynamic collision payload construction, phase-2 corrector, solver
          linear algebra, JAX compile/runtime separation, and diagnostic churn.
        - Advice on selective compiled kernels only after profiling evidence.

        ## Out Of Scope

        - QKE implementation.
        - Public production or publication-ready support claims.
        - q9/q10 slow sweeps by default.
        - Output clipping to hide raw negative/nonfinite evidence.
        - Full rewrite recommendations without reproducible profiling evidence.

        ## Nonclaims

        This packet does not validate the solver, endpoint physics, or public
        readiness.  It only prepares an external performance/optimization audit.
        `N_eff_3T` remains a diagnostic proxy until a physical definition is
        pinned and justified.
        """,
        "PERFORMANCE_CLAIMS_TO_VERIFY.md": """
        # Performance Claims To Verify

        | ID | Claim | Current status | Source / artifact | Files to inspect | Cheap command | Medium command | Slow optional command | What would support the claim | What would falsify the claim | What optimization follows if supported | What not to do if unsupported |
        |---|---|---|---|---|---|---|---|---|---|---|---|
        | P1 | The current slowdown is not primarily Python language overhead. | UNTESTED/PARTIAL | user-provided conclusion and internal audit | `src/rabbit/validation/augmented_continuous_ap65_rhs.py`, profiling scripts | `python scripts/run_diagnostic_payload_churn_probe.py` | q4 bounded run with `/usr/bin/time -v` | numerical kernels/corrector/payload dominate over JSON/object churn | object churn dominates wall after attribution | optimize algorithmic hot paths first | do not rewrite language based on intuition |
        | P2 | q4 activation spent about 752s total and about 307s in dynamic payload build. | SUPPORTED if artifact extract matches | `artifacts/extracts/bd299_q4_activation_probe_extract.json` | `artifacts/raw/diagnostic_outputs/bd299_q4_activation_probe.json` | `python scripts/summarize_perf_artifacts.py artifacts/` | fresh q4 partial-span | same counters appear in artifact | counters absent or materially different | payload rebuild/reuse PR | do not cite number as validated |
        | P3 | Dynamic collision payload builds occur thousands of times in partial runs. | SUPPORTED by bd299 extract if present | bd299 q4 artifact | span ladder/RHS/collision bridge | summarize script | fresh q4 partial-span | build count >1000 | build count small and wall elsewhere | cache/reuse/current-state semantics audit | do not optimize collision first |
        | P4 | Phase-2 Newton/corrector iterations may be a major multiplier. | SUPPORTED by bd299 extract if present | bd299 q4 artifact | AP65 RHS phase-2 corrector code | summarize script | fresh q4 partial-span | Newton iterations/linear solves large | counters small | corrector residual/Jacobian caching | do not delete corrector without falsification |
        | P5 | Host Rodas5P rejected-step coupling amplifies payload/corrector work. | PARTIAL | span counters and host code | AP65 RHS, span rows | summarize script | q4 fresh row | rejected attempts correlate with payload/corrector repeats | no rejected-step correlation | stage reuse/replay/caching policy | do not blame collision alone |
        | P6 | AP65 endpoint path still uses dense LU. | SUPPORTED by source | AP65 RHS and JAX solver | `src/rabbit/validation/augmented_continuous_ap65_rhs.py`, `src/rabbit/jax/solver_jax_rodas5p.py` | `pytest ...test_woodbury...` | synthetic dense-vs-Woodbury probe | `W = I/(gamma*h)-J` and LU path active | structured solve reaches endpoint | A/B endpoint routing scaffold | do not switch default backend |
        | P7 | Dense LU is not row-logged unless BD282 telemetry already landed. | STALE/PARTIAL | current HEAD includes BD282 telemetry fields | validation utils and span ladder | summarize script | fresh telemetry row | old artifacts lack fields, current rows include fields | old rows already complete | preserve telemetry | do not add new gate |
        | P8 | Low-rank/Woodbury/block-sparse pieces exist but endpoint solve routing is not proven. | SUPPORTED | tests and source | JAX solver, linear strategies | low-rank/block tests | A/B synthetic stage | unit parity passes but endpoint default unchanged | endpoint structured solve active | PR for opt-in route | do not claim acceleration landed |
        | P9 | JAX compile time and runtime are not cleanly separated in existing artifacts. | PARTIAL | artifact schema | JAX solver/profiling scripts | probe scripts | fresh q4 with explicit timing | compile/runtime fields absent or mixed | fields already separate | add timing around compiled kernels | do not compare raw wall across modes |
        | P10 | Diagnostic JSON/row shaping is inside or near hot loops and may cause object churn. | PARTIAL | source search | validation utils, AP65 RHS, replay code | diagnostic churn probe | diagnostic-lite A/B if available | JSON/fingerprint in inner loops or high serialization cost | negligible object churn | move diagnostics to row boundary | do not treat as physics evidence |
        | P11 | q-dependent collision grids/tensors, not augmented state size, likely drive memory. | UNTESTED | user concern | collision bridge/collisions/JAX arrays | memory probe | q4/q5 memory run | memory scales with q/collision tensors | memory dominated by Python rows | kernel/cache memory PR | do not run q9/q10 blind |
        | P12 | `tracemalloc` captures Python-object share but not all JAX/XLA buffers. | SUPPORTED conceptually | Python behavior | validation utils | memory probe | q4 with ru/VmHWM/tracemalloc | tracemalloc < process RSS | tracemalloc equals process RSS unexpectedly | use all three memory views | do not rely on tracemalloc alone |
        | P13 | `ru_maxrss`/`VmHWM` capture process-level memory but need platform units. | SUPPORTED | platform behavior | validation utils | memory probe | time -v run | fields include units/platform | ambiguous units | preserve unit-tagged telemetry | do not compare across OS blindly |
        | P14 | q9/q10 should be deferred until row-level memory/backend telemetry exists. | SUPPORTED policy | BD281/internal audits | artifacts and telemetry fields | summarize script | none | missing/partial memory fields in older rows | complete telemetry already exists | run q4/q5 first | do not launch q9/q10 by default |
        | P15 | Whole-language rewrite is premature. | PROPOSED | performance conclusion | all hot paths | cheap probes | q4 attribution | no stable >50-70% kernel or correctness unresolved | one stable kernel dominates and parity holds | selective port later | do not rewrite whole project |
        | P16 | Selective compiled kernels are plausible only after stable bottleneck ID. | PROPOSED | decision frame | collision/corrector/solver kernels | probes | fresh profile | stable shapes and contracts | dynamic semantics unresolved | port one kernel behind policy | do not port moving target |
        | P17 | Optimization must be parity-gated against raw physics observables and diagnostics. | SUPPORTED policy | guardrails | tests/artifacts | existing invariant tests | q4 A/B | Yp/DH/N_eff/Sigma/raw failures preserved | raw fields hidden or drift | add parity tests | do not accept speed-only PR |
        | P18 | Speedup before LRS/non-LRS plus `N_eff_3T >= 3.0` tripwire can entrench wrong physics. | SUPPORTED policy | BD281/internal audit | parity plan | none | controlled pair | floor/parity unresolved | floor is resolved | keep speedups opt-in | do not default optimize |
        """,
        "OPTIMIZATION_HYPOTHESES.md": """
        # Optimization Hypotheses

        ## A. Collision Payload

        - Repeated payload build is amplified by rejected steps.
        - Dynamic shape or policy changes may defeat caching.
        - Radial/angular source composition can force redundant computation.
        - q-dependent pairwise tensors may dominate memory.
        - Payload reuse can preserve physics only if current-state semantics are
          honored and raw source diagnostics remain available.

        ## B. Solver / Jacobian

        - Dense `W = I/(gamma*h) - J` LU dominates memory/wall at larger q.
        - Block-JVP assembly can still densify if structured solve is not wired.
        - Woodbury/low-rank stage solve can match dense solve in synthetic tests.
        - Structured endpoint solve routing is the missing implementation piece.
        - Full-JVP over A-modes is misallocated work if errors are geometry/thermo
          dominated.

        ## C. Phase-2 Corrector

        - Newton iteration count multiplies RHS/collision work.
        - The corrector should be retained unless direct falsification shows it
          is wrong.
        - Residual/Jacobian caching or analytic blocks may reduce iterations.

        ## D. JAX / Host Orchestration

        - Compile/runtime separation is incomplete in artifacts.
        - Small host-level solves or Python callbacks may dominate.
        - Dynamic payload structures may prevent JIT reuse.
        - Benchmarks need explicit `block_until_ready` where JAX arrays are used.

        ## E. Diagnostics / Artifacts

        - JSON-safe conversion, fingerprinting, and trace rows may be too close
          to hot loops.
        - Large per-step traces create Python object churn.
        - Endpoint row summaries should be separated from hot-loop diagnostics.

        ## F. Language / Runtime

        A full C++/Rust/Julia rewrite is a last resort.  Selective compiled
        candidates are pairwise collision quadrature/source assembly,
        q-dependent tensor construction, phase-2 residual/Jacobian, structured
        linear solve, diagnostic-free AP65 RHS kernel, and JAX Pallas/custom call
        only after stable shape and parity.
        """,
        "PROFILING_AND_BENCHMARK_RUNBOOK.md": """
        # Profiling And Benchmark Runbook

        ## Cheap Tier

        ```bash
        python -m pip install -e ".[dev]"

        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs

        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_block_sparse_jacobian.py

        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \\
          tests/test_augmented_pstf_distribution.py \\
          tests/test_three_temperature_closure_invariants.py

        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_memory_telemetry_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_dense_vs_woodbury_ab_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_row_serialization_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_diagnostic_payload_churn_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_collision_payload_accounting_probe.py artifacts/
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/summarize_perf_artifacts.py artifacts/
        ```

        Cheap probes are not endpoint physics validation.

        ## Medium Tier

        Before any fresh run:

        ```bash
        PYTHONPATH=src JAX_PLATFORMS=cpu \\
          python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help
        ```

        Then use a bounded q4 partial-span or q4 endpoint smoke only if exact
        flags are discoverable.  Do not guess flags.  Prefer `/usr/bin/time -v`
        when available.  Capture stdout/stderr and JSON artifacts.  A/B only one
        variable at a time: diagnostic mode, dense vs Woodbury synthetic stage,
        payload reuse policy, or full-JVP vs block-JVP if an opt-in exists.

        ## Slow Optional Tier

        q5 and q9/q10 are opt-in only.  q9/q10 require row-level memory/backend
        telemetry first: `ru_maxrss`, `VmHWM`, `tracemalloc_peak`,
        `linear_system_backend`, W/J dimensions, payload build count/wall,
        source eval count, accepted/rejected steps, and phase-2 Newton counts.
        """,
        "LANGUAGE_REWRITE_DECISION_FRAME.md": """
        # Language Rewrite Decision Frame

        ## Default Verdict

        Do not rewrite the whole project now.

        ## Selective Porting Threshold

        Selective porting becomes justified only if all hold:

        1. q4/q5 endpoint physics and parity are stable.
        2. One residual kernel is repeatedly measured at more than 50-70% wall.
        3. The kernel has stable shapes and a clear numerical contract.
        4. Python-object and JAX-compile overhead are separated from numerical
           runtime.
        5. Dense LU/block-low-rank/JAX routing options have been tested.
        6. Parity tests against raw observables pass.
        7. The port is isolated behind a backend policy with reference fallback.
        8. The auditor can reproduce before/after speed and memory wins.

        ## Candidate Port Targets

        - pairwise nu-nu collision quadrature/source assembly;
        - collision radial/angular tensor construction;
        - phase-2 network Newton residual/Jacobian;
        - dense/structured linear solve backend;
        - AP65 RHS hot-loop kernel;
        - diagnostic-free source assembly;
        - JAX Pallas/custom call if shapes are stable;
        - Numba/Cython/C++/Rust for pinned kernels only;
        - Julia only if solver-host replacement is proven better and the physics
          contract is stable.

        ## Rejection Criteria For Full Rewrite

        Reject a full rewrite if endpoint physics is unresolved, the
        `N_eff_3T < 3.0` floor/parity issue is unresolved, profiler telemetry is
        missing, speedup evidence is toy-only, raw observable parity is absent,
        or the rewrite would reproduce dense-LU/dynamic-payload architecture.
        """,
        "SOURCE_RUNTIME_SPINE.md": """
        # Source Runtime Spine

        ```text
        CLI
          scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py
        -> span ladder and row assembly
          src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py
        -> AP65 RHS, restart, phase-1/phase-2, host solver config
          src/rabbit/validation/augmented_continuous_ap65_rhs.py
        -> live Type-I weak/network/collision source assembly
          src/rabbit/transport/augmented_typeI_weak_network.py
        -> thermo/collision/weak/network blocks
          src/rabbit/thermo/
          src/rabbit/collisions/
          src/rabbit/weak/
          src/rabbit/network/
        -> in-tree Rodas5P/AP65 and linear solve strategies
          src/rabbit/jax/solver_jax_rodas5p.py
          src/rabbit/jax/linear_solve_strategies.py
          src/rabbit/solver/rodas5p.py
        ```

        Counter locations to inspect include dynamic collision payload counters,
        source/stage evaluation counts, host accepted/rejected attempts,
        dense-LU metadata, phase-2 corrector/Newton counters, and JSON-safe
        artifact shaping.
        """,
        "BENCHMARK_MATRIX.md": """
        # Benchmark Matrix

        | ID | Command | Target layer | Runtime class | Wall fields | Memory fields | Physics parity fields | Allowed interpretation | Disallowed interpretation |
        |---|---|---|---|---|---|---|---|---|
        | B0 | `python -m pip install -e ".[dev]"` | setup | cheap | install time | none | none | environment smoke | solver validation |
        | B1 | `pytest ...test_woodbury_stage_linear_solve_matches_dense ...test_low_rank_rodas_step_matches_dense_step_for_linear_rhs` | solver algebra | cheap | pytest runtime | none | residual parity | dense/Woodbury algebra check | endpoint acceleration |
        | B2 | `pytest -q tests/test_block_sparse_jacobian.py` | block sparse assembly | cheap | pytest runtime | none | algebraic parity | assembly check | structured endpoint solve |
        | B3 | `python scripts/run_memory_telemetry_probe.py` | memory telemetry | cheap | probe runtime | ru/VmHWM/tracemalloc | none | field semantics | q4 memory attribution |
        | B4 | `python scripts/run_diagnostic_payload_churn_probe.py` | diagnostic churn | cheap | serialization time | Python object size | none | overhead estimate | physics evidence |
        | B5 | `python scripts/run_collision_payload_accounting_probe.py artifacts/` | artifact accounting | cheap | artifact counters | artifact memory fields | endpoint fields if present | counter extraction | fresh run validation |
        | B6 | q4 partial-span current path | integrated runtime | medium | wall/payload/corrector/solve | ru/VmHWM/tracemalloc | Yp/DH/N_eff/Sigma/raw failures | bounded runtime evidence | q9 extrapolation |
        | B7 | q4 diagnostic-lite/hot-loop mode if implemented | diagnostics A/B | medium | wall/object churn | memory fields | same row parity | diagnostic overhead | physics validation |
        | B8 | q4 endpoint smoke if feasible | endpoint | medium/slow | full row wall | memory fields | endpoint observables | bounded endpoint evidence | publication readiness |
        | B9 | q5 endpoint optional | endpoint | slow | full row wall | memory fields | endpoint observables | scaling evidence | default requirement |
        | B10 | q9/q10 memory scaling optional | memory scaling | slow | full row wall | required process memory | endpoint observables | high-q attribution | blind slow sweep |
        | B11 | phase-2 Newton residual/Jacobian microprobe | corrector | cheap/medium | Newton counts | none | mass/charge/raw failures | corrector hotspot evidence | delete corrector |
        | B12 | JAX compile/runtime microbench | JAX | cheap/medium | compile/runtime split | JAX/process memory | none | timing separation | endpoint validation |
        | B13 | external BBN code context | benchmark context | external | reported wall | reported memory | standard BBN only | scale context | apples-to-apples proof |

        ## External BBN Code Context

        AlterBBN, PArthENoPE, PRIMAT, PRyMordial, and LINX should be treated as
        context, not direct competitors for this workload.  The useful question
        is not "is RABBIT slower than a standard BBN code?" but "after live
        collision/source work, phase-2 corrector coupling, and diagnostics are
        attributed, which residual kernel remains too slow?"  The auditor should
        independently verify any published runtime citation before using it in a
        public report.
        """,
        "EXPECTED_FINDINGS_AND_OPEN_QUESTIONS.md": """
        # Expected Findings And Open Questions

        ## Expected Findings

        - Current q4 artifacts should show collision payload and phase-2
          corrector work as major multipliers.
        - Existing Woodbury/low-rank/block-sparse tests should pass locally but
          should not prove endpoint routing.
        - Older artifacts may lack complete memory/backend telemetry, while
          current HEAD includes BD282 telemetry helpers.
        - Whole-language rewrite should remain unproven without a stable
          dominant numerical kernel.

        ## Open Questions

        - What fraction of q4 wall is payload build, corrector Newton, dense LU,
          JAX compile/runtime, and JSON/object churn?
        - Why is `selected_stage_collision_payload_reuse_total` zero under the
          current q4 dynamic case?
        - Does dense LU dominate only at q9/q10, or already at q4/q5?
        - Can phase-2 residual/Jacobian caching reduce the 24k+ Newton linear
          solves without changing raw abundances?
        - Can diagnostics be moved out of inner loops without losing raw
          nonfinite/negative evidence?
        """,
        "EXTERNAL_OPTIMIZATION_AUDITOR_PROMPT.md": """
        # External Performance Optimization Audit Prompt - RABBIT Augmented Type-I PSTF No-QKE BBN Solver

        You are an independent external performance / optimization auditor for
        the RABBIT augmented Type-I PSTF no-QKE BBN solver.

        Your task is not to rubber-stamp a rewrite and not to optimize blindly.
        Your task is to determine, from code and executable probes, why the
        current solver is slow and what should be accelerated first.

        Read these first:

        1. `README_EXTERNAL_PERFORMANCE_AUDIT.md`
        2. `AUDIT_SCOPE_AND_NONCLAIMS.md`
        3. `PERFORMANCE_CLAIMS_TO_VERIFY.md`
        4. `OPTIMIZATION_HYPOTHESES.md`
        5. `PROFILING_AND_BENCHMARK_RUNBOOK.md`
        6. `LANGUAGE_REWRITE_DECISION_FRAME.md`
        7. `SOURCE_RUNTIME_SPINE.md`
        8. `AGENTS.md`
        9. `docs_snapshot/docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
        10. prior external audits and internal re-audit docs in `docs_snapshot/audits/`

        Use CRAG:

        1. Extract performance and optimization claims.
        2. Retrieve exact source/artifact/test evidence.
        3. Classify each claim as SUPPORTED / CONTRADICTED / PARTIAL / STALE /
           UNTESTED / UNKNOWN.
        4. Convert unresolved claims into the smallest executable probe.
        5. Do not present toy/demo output as research evidence.

        Use Chain-of-Code:

        - For each nontrivial performance hypothesis, write or run the smallest
          executable probe feasible.
        - Record exact command, exit code, runtime, output summary, artifact path,
          and files touched.
        - Distinguish import/test smoke, synthetic solver algebra, partial-span
          runtime evidence, and endpoint evidence.
        - Never treat synthetic benchmarks as endpoint physics validation.

        Use role subagents if available:

        1. Runtime Spine Profiler
        2. Collision Payload Auditor
        3. Phase-2 Corrector / Newton Auditor
        4. Solver / Linear Algebra Auditor
        5. JAX Compile / Runtime Auditor
        6. Memory Attribution Auditor
        7. Hot-Loop Architecture Surgeon
        8. Language / Runtime Portability Reviewer
        9. Reproducibility Engineer
        10. Red-Team Reviewer
        11. Final Verdict Editor

        Wait for all subagents before final synthesis.  If subagents are
        unavailable, simulate the same roles sequentially.  Do not use
        Explorer-style open-ended exploration as the main method.

        Main questions:

        1. Is the current slowdown primarily Python language overhead, or
           algorithmic/orchestration overhead?
        2. Can the reported q4 partial-run numbers be reproduced from shipped
           logs/artifacts or fresh bounded runs?
        3. How much wall time is attributable to dynamic collision payload build,
           Rodas5P RHS stages, finite-difference or JVP Jacobian probes, dense LU,
           phase-2 Newton/corrector, rejected-step replay, JAX compile/runtime,
           and Python diagnostic/JSON/object churn?
        4. Which costs scale with q, angular grid, step count, rejected steps, and
           Newton iterations?
        5. Are dense LU and block/low-rank/Woodbury paths actually wired to the
           endpoint?
        6. Is block-JVP merely an assembly policy that still densifies, or does
           any structured solve reach endpoint?
        7. Does diagnostic payload construction occur inside hot loops?
        8. Is there enough evidence to defer q9/q10 until telemetry exists?
        9. Is a full C++/Rust/Julia rewrite justified now?
        10. If not, which selective compiled kernel would be justified later, and
            under what threshold?
        11. Which optimization PR should be first?
        12. What tests/parity gates must guard any speedup so it does not make
            wrong physics faster?

        Required commands:

        ```bash
        python -m pip install -e ".[dev]"

        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs

        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_block_sparse_jacobian.py

        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_memory_telemetry_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_dense_vs_woodbury_ab_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/summarize_perf_artifacts.py artifacts/
        ```

        Then inspect endpoint flags:

        ```bash
        PYTHONPATH=src JAX_PLATFORMS=cpu \\
          python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help
        ```

        Only if exact bounded flags are clear, run one medium q4 partial-span or
        smoke.  Do not guess flags.  Do not run q9/q10 unless telemetry exists
        and the user explicitly accepts slow/high-memory execution.

        Required final report:

        1. Executive verdict: FULL_REWRITE_NOW / SELECTIVE_PORT_LATER /
           STAY_PYTHON_JAX_AND_OPTIMIZE / INCONCLUSIVE.
        2. One-page summary.
        3. Claim ledger.
        4. Commands run and outputs.
        5. Profiling attribution table.
        6. Memory attribution table.
        7. Runtime spine bottleneck map.
        8. Solver path verdict.
        9. Collision payload verdict.
        10. Phase-2 corrector verdict.
        11. JAX compile/runtime verdict.
        12. Diagnostic JSON/object-churn verdict.
        13. Language/runtime decision tree.
        14. Top 5 optimizations ranked by expected speedup, correctness risk,
            implementation difficulty, and test coverage.
        15. Recommended PR sequence, max 6 PRs.
        16. Missing evidence/files.
        17. Exact next commands.
        18. Red-team objections.

        Output discipline:

        - Be blunt about uncertainty.
        - Do not recommend a rewrite without evidence.
        - Do not claim endpoint physics is fixed by speed probes.
        - Do not hide raw negative/nonfinite evidence.
        - Do not treat standard BBN package benchmarks as apples-to-apples unless
          workload differences are stated explicitly.
        - Do not present toy/demo outputs as research evidence.
        """,
    }


def script_memory_probe() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json
    import os
    import platform
    import resource
    import sys
    import tracemalloc

    def vmhwm_kb():
        path = "/proc/self/status"
        if not os.path.exists(path):
            return None, "proc_status_unavailable"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        return int(parts[1]), "kilobytes"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return None, "VmHWM_missing"

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()
    blob = bytearray(8 * 1024 * 1024)
    blob[0] = 1
    after = tracemalloc.get_traced_memory()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_unit = "kilobytes_on_linux_bytes_on_macos"
    vm, vm_unit = vmhwm_kb()
    print(json.dumps({
        "platform": platform.platform(),
        "python": sys.version,
        "allocation_bytes": len(blob),
        "ru_maxrss_raw": ru,
        "ru_maxrss_unit_note": ru_unit,
        "vmhwm_kb": vm,
        "vmhwm_unit_or_reason": vm_unit,
        "tracemalloc_before_bytes": before,
        "tracemalloc_after_bytes": after,
        "interpretation": "Smoke probe only; tracemalloc excludes most native/JAX/XLA buffers.",
    }, indent=2, sort_keys=True))
    '''


def script_dense_vs_woodbury() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json
    import numpy as np

    rng = np.random.default_rng(282)
    n = 24
    rank = 3
    gamma_h_inv = 17.0
    base_diag = gamma_h_inv + np.linspace(0.2, 2.0, n)
    base = np.diag(base_diag)
    u = rng.normal(size=(n, rank)) * 0.03
    v = rng.normal(size=(n, rank)) * 0.03
    w_dense = base - u @ v.T
    rhs = rng.normal(size=n)

    dense = np.linalg.solve(w_dense, rhs)
    base_inv_rhs = rhs / base_diag
    base_inv_u = u / base_diag[:, None]
    middle = np.eye(rank) - v.T @ base_inv_u
    woodbury = base_inv_rhs + base_inv_u @ np.linalg.solve(middle, v.T @ base_inv_rhs)

    abs_err = float(np.max(np.abs(dense - woodbury)))
    rel_err = float(abs_err / max(1.0, np.max(np.abs(dense))))
    print(json.dumps({
        "n": n,
        "rank": rank,
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
        "passes": bool(abs_err < 1e-10),
        "interpretation": "Synthetic algebra probe only; does not prove AP65 endpoint routing.",
    }, indent=2, sort_keys=True))
    '''


def script_row_serialization_probe() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json

    row = {
        "linear_system_backend": "dense_scipy_lu",
        "linear_system_backend_source": "synthetic_probe_not_runtime_evidence",
        "linear_system_matrix_shape": [42, 42],
        "jacobian_shape": [42, 42],
        "linear_system_dtype": "float64",
        "linear_system_is_dense": True,
        "linear_system_low_rank_active": False,
        "ru_maxrss_kb": 123456,
        "vmhwm_kb": None,
        "tracemalloc_peak_bytes": 98765,
        "collision_projection_policy": "flrw_monopole_only",
        "chain_h_max_policy": "first_rejection_or_recovered_h_ceiling",
        "h_max": 0.005,
        "N_span": [0.0, 1.0],
        "Yp": -0.01,
        "raw_negative_evidence_preserved": True,
    }
    encoded = json.dumps(row, sort_keys=True)
    required = [
        "linear_system_backend",
        "ru_maxrss_kb",
        "collision_projection_policy",
        "chain_h_max_policy",
        "Yp",
    ]
    print(json.dumps({
        "json_bytes": len(encoded.encode("utf-8")),
        "required_present": {key: key in row for key in required},
        "row": row,
        "interpretation": "JSON-safety probe only; negative Yp is intentionally synthetic and unclipped.",
    }, indent=2, sort_keys=True))
    '''


def script_collision_accounting() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json
    import sys
    from pathlib import Path

    KEYS = [
        "selected_wall_seconds_total",
        "selected_dynamic_collision_payload_builds_total",
        "selected_dynamic_collision_payload_build_wall_seconds_total",
        "selected_source_evaluations_total",
        "selected_stage_source_evaluations_total",
        "selected_stage_collision_payload_reuse_total",
        "selected_linear_system_solves_total",
        "selected_phase2_conservative_extent_corrector_newton_iteration_count_total",
    ]

    def rows(data):
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return [row for row in data["rows"] if isinstance(row, dict)]
        if isinstance(data, dict) and isinstance(data.get("rows"), dict):
            return [data["rows"]]
        return []

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
    out = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows(data):
            found = {key: row.get(key) for key in KEYS if key in row}
            if found:
                out.append({"path": str(path), "fields": found})
    print(json.dumps({"artifact_root": str(root), "matches": out[:50], "match_count": len(out)}, indent=2, sort_keys=True))
    '''


def script_diagnostic_churn() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json
    import sys
    import time
    import tracemalloc

    rows = []
    for idx in range(2000):
        rows.append({
            "step": idx,
            "payload_fingerprint": f"fake-{idx:06d}",
            "q_nodes": [float(i) for i in range(8)],
            "diagnostics": {f"k{j}": float(idx * j) for j in range(20)},
            "raw_negative_candidate_min": -1e-12 if idx % 101 == 0 else 0.0,
        })
    tracemalloc.start()
    t0 = time.perf_counter()
    payload = json.dumps(rows, sort_keys=True)
    wall = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    print(json.dumps({
        "row_count": len(rows),
        "json_bytes": len(payload.encode("utf-8")),
        "json_dump_wall_seconds": wall,
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
        "interpretation": "Diagnostic overhead estimate only; not physics evidence.",
    }, indent=2, sort_keys=True))
    '''


def script_summarize_perf_artifacts() -> str:
    return r'''
    #!/usr/bin/env python3
    from __future__ import annotations

    import json
    import sys
    from pathlib import Path

    FIELDS = [
        "selected_wall_seconds_total",
        "selected_dynamic_collision_payload_builds_total",
        "selected_dynamic_collision_payload_build_wall_seconds_total",
        "selected_source_evaluations_total",
        "selected_stage_source_evaluations_total",
        "selected_stage_collision_payload_reuse_total",
        "selected_linear_system_factorizations_total",
        "selected_linear_system_solves_total",
        "selected_phase2_conservative_extent_corrector_newton_iteration_count_total",
        "selected_phase2_conservative_extent_corrector_newton_linear_solve_count_total",
        "ru_maxrss_kb",
        "vmhwm_kb",
        "tracemalloc_peak_bytes",
        "linear_system_backend",
        "linear_system_matrix_shape",
        "W_shape",
        "J_shape",
        "N_eff_3T",
        "Yp",
        "DH",
        "Sigma_H",
        "T_final_MeV",
        "completion_class",
        "physical_full_bbn_span_ready",
    ]

    def iter_rows(data):
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            for row in data["rows"]:
                if isinstance(row, dict):
                    yield row
        elif isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
    summaries = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in iter_rows(data):
            fields = {key: row.get(key) for key in FIELDS if key in row}
            if fields:
                summaries.append({"path": str(path), "fields": fields})
    print(json.dumps({
        "artifact_root": str(root),
        "summary_count": len(summaries),
        "summaries": summaries[:100],
        "note": "Artifact summary only; fresh profiling requires bounded run commands.",
    }, indent=2, sort_keys=True))
    '''


def make_command_scripts() -> dict[str, str]:
    return {
        "COMMANDS_CHEAP.sh": """
        #!/usr/bin/env bash
        set -euo pipefail
        python -m pip install -e ".[dev]"
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_woodbury_stage_linear_solve_matches_dense \\
          tests/test_j04_jax_rodas5p.py::TestSolverProperties::test_low_rank_rodas_step_matches_dense_step_for_linear_rhs
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_block_sparse_jacobian.py
        PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q \\
          tests/test_augmented_pstf_distribution.py \\
          tests/test_three_temperature_closure_invariants.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_memory_telemetry_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_dense_vs_woodbury_ab_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_row_serialization_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_diagnostic_payload_churn_probe.py
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_collision_payload_accounting_probe.py artifacts/
        PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/summarize_perf_artifacts.py artifacts/
        """,
        "COMMANDS_MEDIUM.sh": """
        #!/usr/bin/env bash
        set -euo pipefail
        PYTHONPATH=src JAX_PLATFORMS=cpu \\
          python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --help
        cat <<'MSG'
        Inspect --help before any medium q4 run.
        Do not guess flags. Use /usr/bin/time -v if available.
        Suggested pattern after exact flags are chosen:
          /usr/bin/time -v env PYTHONPATH=src JAX_PLATFORMS=cpu \\
            python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \\
            <bounded q4 partial-span flags> \\
            --output-dir diagnostic_outputs/external_perf_q4
        MSG
        """,
        "COMMANDS_SLOW_OPTIONAL.sh": """
        #!/usr/bin/env bash
        set -euo pipefail
        cat <<'MSG'
        Slow optional commands are not provided as runnable defaults.
        q5 or q9/q10 runs require explicit auditor/user acceptance and row-level:
          ru_maxrss, VmHWM, tracemalloc_peak, linear_system_backend, W/J shape,
          payload build count/wall, source eval count, accepted/rejected steps,
          and phase-2 Newton iteration counters.
        MSG
        """,
    }


def manifest_entries(root: Path, artifact_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in {"PACKET_MANIFEST.md", "PACKET_MANIFEST.json"}:
            entries.append(
                {
                    "packet_path": rel,
                    "original_repo_path": None,
                    "sha256": None,
                    "size_bytes": path.stat().st_size,
                    "category": "manifest",
                    "reason_included": "Self-referential packet manifest; hash is omitted to avoid an impossible fixed point.",
                    "kind": "generated",
                }
            )
            continue
        meta = artifact_meta.get(rel, {})
        entries.append(
            {
                "packet_path": rel,
                "original_repo_path": meta.get("original_repo_path"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "category": meta.get("category", infer_category(rel)),
                "reason_included": meta.get("reason", infer_reason(rel)),
                "kind": meta.get("kind", infer_kind(rel)),
                **({"artifact_extract": meta["artifact_extract"]} if "artifact_extract" in meta else {}),
            }
        )
    return entries


def infer_category(rel: str) -> str:
    if rel.startswith("src/"):
        return "source"
    if rel.startswith("tests/"):
        return "tests"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("docs_snapshot/") or rel.startswith("docs/"):
        return "docs"
    if rel.startswith("artifacts/"):
        return "artifacts"
    if rel.endswith(".md"):
        return "packet_doc"
    return "environment"


def infer_kind(rel: str) -> str:
    if rel.startswith("artifacts/"):
        return "artifact"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("src/"):
        return "source"
    if rel.startswith("scripts/"):
        return "generated" if rel.startswith("scripts/run_") and rel.endswith("_probe.py") else "source"
    if rel.endswith(".md") or rel.endswith(".json"):
        return "doc"
    return "source"


def infer_reason(rel: str) -> str:
    if rel.startswith("src/"):
        return "Runnable source snapshot for performance and code-path inspection."
    if rel.startswith("tests/"):
        return "Relevant test snapshot for cheap probes and parity checks."
    if rel.startswith("artifacts/"):
        return "Curated performance/runtime evidence or reduced extract."
    if rel.startswith("scripts/"):
        return "Runbook or packet-only profiling helper."
    return "Packet documentation or environment metadata."


def markdown_manifest(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Packet Manifest",
        "",
        "Self-referential manifest file hashes are recorded as null by design.",
        "",
        "| packet path | original repo path | sha256 | size bytes | category | kind | reason |",
        "|---|---|---|---:|---|---|---|",
    ]
    for entry in entries:
        reason = str(entry["reason_included"]).replace("\n", " ")
        lines.append(
            "| {packet_path} | {original_repo_path} | {sha256} | {size_bytes} | {category} | {kind} | {reason} |".format(
                packet_path=entry["packet_path"],
                original_repo_path=entry.get("original_repo_path") or "",
                sha256=entry.get("sha256") or "",
                size_bytes=entry["size_bytes"],
                category=entry["category"],
                kind=entry["kind"],
                reason=reason,
            )
        )
    return "\n".join(lines) + "\n"


def zip_packet() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(PACKET_DIR.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, path.relative_to(PACKET_DIR.parent))
    digest = sha256_file(ZIP_PATH)
    SHA_PATH.write_text(f"{digest}  {ZIP_PATH.as_posix()}\n", encoding="utf-8")


def main() -> int:
    if PACKET_DIR.exists():
        shutil.rmtree(PACKET_DIR)
    PACKET_DIR.mkdir(parents=True)
    meta: dict[str, dict[str, Any]] = {}
    missing_inputs: list[str] = []

    # Runnable source and tests.
    for src_dir in (Path("src"), Path("tests")):
        if src_dir.exists():
            copy_tree(src_dir, PACKET_DIR / src_dir)
        else:
            missing_inputs.append(str(src_dir))

    selected_scripts = [
        "scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py",
        "scripts/profile_augmented_dynamic_collision_hotpath.py",
        "scripts/profile_rodas5p_batch_linear_solver.py",
        "scripts/run_augmented_continuous_ap65_span_experiment.py",
        "scripts/run_augmented_continuous_ap65_failure_triage.py",
    ]
    for item in selected_scripts:
        src = Path(item)
        if src.exists():
            copy_file(src, PACKET_DIR / item)
        else:
            missing_inputs.append(item)

    for item in ("pyproject.toml", "README.md", "STATUS.md", "SUPPORTED_CAPABILITIES.md", "PROMOTION_GATES.md", "AGENTS.md"):
        src = Path(item)
        if src.exists():
            copy_file(src, PACKET_DIR / item)
        else:
            missing_inputs.append(item)

    docs_to_copy = [
        "docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md",
        "docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md",
        "docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md",
        "docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md",
        "docs/audit/BD186_current_collision_blocker_external_review_packet_2026-05-27.md",
        "docs/audit/BD186_external_audit_report1_collision_relaxation_2026-05-27.md",
        "docs/audit/BD186_external_audit_report2_radial_coordinate_2026-05-27.md",
        "docs/audit/BD199_flrw_laguerre_collision_drift_external_audit_prompt_2026-05-27.md",
        "docs/audit/fb68_dynamic_collision_hotpath_profile.md",
    ]
    for item in docs_to_copy:
        src = Path(item)
        if src.exists():
            copy_file(src, PACKET_DIR / "docs_snapshot" / item)
            if item == "docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md":
                copy_file(src, PACKET_DIR / item)
        else:
            missing_inputs.append(item)

    audit_docs = [
        "BD281_external_reaudit_report_2026-06-02.md",
        "BD279_external_audit_report_2026-06-02.md",
        "BD280_external_full_code_audit_report_2026-06-02.md",
        "internal_reaudit_report.md",
        "hypothesis_falsification_matrix.md",
        "skill_and_subagent_usage.md",
        "pr_acceleration_plan.md",
    ]
    for item in audit_docs:
        src = Path(item)
        if src.exists():
            copy_file(src, PACKET_DIR / "docs_snapshot" / "audits" / item)
        else:
            missing_inputs.append(item)
            write_text(
                PACKET_DIR / "docs_snapshot" / "audits" / f"MISSING_{item}",
                f"# Missing Input\n\n`{item}` was requested but was not present in the repository at packet build time.\n",
            )

    source_readme = """
    # Source Snapshot Layout

    The runnable source snapshot is copied at packet root under `src/`.
    This directory is intentionally a lightweight pointer to avoid duplicating
    the full source tree.  The manifest lists each copied source file.
    """
    tests_readme = """
    # Tests Snapshot Layout

    The runnable test snapshot is copied at packet root under `tests/`.
    This directory is intentionally a lightweight pointer to avoid duplicating
    the full test tree.  The manifest lists each copied test file.
    """
    write_text(PACKET_DIR / "source_snapshot" / "README.md", source_readme)
    write_text(PACKET_DIR / "tests_snapshot" / "README.md", tests_readme)

    # Packet docs and helper scripts.
    for rel, content in build_docs(missing_inputs).items():
        write_text(PACKET_DIR / rel, content)
    for rel, content in make_command_scripts().items():
        write_text(PACKET_DIR / rel, content, executable=True)
    write_text(PACKET_DIR / "scripts" / "run_memory_telemetry_probe.py", script_memory_probe(), executable=True)
    write_text(PACKET_DIR / "scripts" / "run_dense_vs_woodbury_ab_probe.py", script_dense_vs_woodbury(), executable=True)
    write_text(PACKET_DIR / "scripts" / "run_row_serialization_probe.py", script_row_serialization_probe(), executable=True)
    write_text(PACKET_DIR / "scripts" / "run_collision_payload_accounting_probe.py", script_collision_accounting(), executable=True)
    write_text(PACKET_DIR / "scripts" / "run_diagnostic_payload_churn_probe.py", script_diagnostic_churn(), executable=True)
    write_text(PACKET_DIR / "scripts" / "summarize_perf_artifacts.py", script_summarize_perf_artifacts(), executable=True)

    # Curated artifacts: include bounded q4 probes raw plus reduced extracts.
    artifact_sources = [
        "diagnostic_outputs/bd299_q4_activation_probe.json",
        "diagnostic_outputs/bd299_window_q4_activation_probe.json",
        "diagnostic_outputs/bd299_damped_q4_activation_probe.json",
        "diagnostic_outputs/bd298_q4_activation_probe.json",
    ]
    artifact_extract_meta: list[dict[str, Any]] = []
    for item in artifact_sources:
        src = Path(item)
        if not src.exists():
            missing_inputs.append(item)
            continue
        raw_dst = PACKET_DIR / "artifacts" / "raw" / item
        copy_file(src, raw_dst)
        extract_dst = PACKET_DIR / "artifacts" / "extracts" / f"{src.stem}_extract.json"
        extract_meta = extract_artifact(src, extract_dst)
        artifact_extract_meta.append(extract_meta)
        meta[extract_dst.relative_to(PACKET_DIR).as_posix()] = {
            "original_repo_path": item,
            "category": "artifact_extract",
            "kind": "artifact",
            "reason": "Reduced performance extract from q4 activation probe.",
            "artifact_extract": extract_meta,
        }
        meta[raw_dst.relative_to(PACKET_DIR).as_posix()] = {
            "original_repo_path": item,
            "category": "artifact_raw",
            "kind": "artifact",
            "reason": "Small bounded q4 performance artifact included raw for independent parsing.",
        }

    large_lines = [
        "# Skipped Large Artifacts",
        "",
        "Large generated diagnostics are not copied raw.  The packet includes bounded q4 raw probes and extracts instead.",
        "",
        "| size bytes | path | reason |",
        "|---:|---|---|",
    ]
    try:
        candidates = []
        for path in Path("diagnostic_outputs").rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt"}:
                size = path.stat().st_size
                if size > 3_000_000 and str(path) not in artifact_sources:
                    candidates.append((size, path))
        for size, path in sorted(candidates, reverse=True)[:40]:
            large_lines.append(f"| {size} | `{path}` | skipped to keep packet size bounded; use local repo for slow artifact archaeology |")
    except Exception as exc:
        large_lines.append(f"| 0 | diagnostic_outputs scan | unavailable: {type(exc).__name__}: {exc} |")
    write_text(PACKET_DIR / "artifacts" / "SKIPPED_LARGE_ARTIFACTS.md", "\n".join(large_lines) + "\n")
    write_text(PACKET_DIR / "artifacts" / "ARTIFACT_EXTRACT_METADATA.json", json_dump(artifact_extract_meta))

    role_usage = {
        "skills": {
            "using-superpowers": "used",
            "codebase-recon": "used for repo vitals",
            "superpowers:systematic-debugging": "used as claim/probe discipline",
            "superpowers:test-driven-development": "available; packet scripts are generated helpers, not runtime code",
            "brooks-audit": "used for architecture/runtime-spine framing",
            "brooks-debt": "used for deletion/consolidation framing",
            "brooks-test": "available; existing test quality issues are documented",
        },
        "subagents": {
            "requested": True,
            "available_tool": "multi_agent_v1",
            "status": "spawn failed because agent thread limit was reached; roles simulated locally",
            "simulated_roles": [
                "Performance Packet Curator",
                "Baseline / Benchmark Context Auditor",
                "Runtime Spine Profiler",
                "Collision Payload Auditor",
                "Phase-2 Corrector / Newton Auditor",
                "Numerical Solver / Linear Algebra Auditor",
                "JAX Compilation / Runtime Auditor",
                "Memory Attribution Auditor",
                "Hot-Loop Architecture Surgeon",
                "Language / Runtime Portability Reviewer",
                "Reproducibility Engineer",
                "External Prompt Editor",
                "Red-Team Reviewer",
            ],
        },
        "build": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_status_short": run_git(["status", "--short"]),
            "git_log_oneline_30": run_git(["log", "--oneline", "-30"]),
            "missing_inputs": missing_inputs,
        },
    }
    write_text(PACKET_DIR / "ROLE_SKILL_USAGE_AND_FALLBACKS.json", json_dump(role_usage))

    entries = manifest_entries(PACKET_DIR, meta)
    write_text(PACKET_DIR / "PACKET_MANIFEST.json", json_dump(entries))
    write_text(PACKET_DIR / "PACKET_MANIFEST.md", markdown_manifest(entries))
    # Recompute so final manifest files appear with the self-referential exception.
    entries = manifest_entries(PACKET_DIR, meta)
    write_text(PACKET_DIR / "PACKET_MANIFEST.json", json_dump(entries))
    write_text(PACKET_DIR / "PACKET_MANIFEST.md", markdown_manifest(entries))

    zip_packet()
    print(json.dumps({"packet_dir": str(PACKET_DIR), "zip_path": str(ZIP_PATH), "sha256": sha256_file(ZIP_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
