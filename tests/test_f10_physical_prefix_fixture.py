"""Regression tests for the F-10 physical-prefix provenance fixture."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.audit import f10_physical_prefix_fixture as fixture
from scripts.audit import _trajectory_core as core


def test_canonical_bytes_have_hand_checked_hashes():
    """Catch key-order or float-endianness drift in sealed manifests."""

    assert fixture.sha256_bytes(fixture.canonical_json_bytes({"b": 2, "a": 1})) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    expected = hashlib.sha256(struct.pack("<2d", 1.0, 2.0)).hexdigest()
    assert expected == "dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae"
    assert fixture.sha256_bytes(
        fixture.float64_le_bytes(np.array([1.0, 2.0]))
    ) == expected


def test_deterministic_npz_repeats_and_rejects_object_arrays(tmp_path: Path):
    """Catch timestamp-dependent archives and unsafe pickle-bearing fixtures."""

    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    arrays = {"z": np.array([3.0]), "a": np.array([1, 2], dtype=np.int64)}
    fixture.write_deterministic_npz(first, arrays)
    fixture.write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    loaded = fixture.load_numeric_npz(first)
    np.testing.assert_array_equal(loaded["a"], [1, 2])
    with pytest.raises(ValueError, match="object dtype"):
        fixture.write_deterministic_npz(
            tmp_path / "bad.npz", {"x": np.array([{}], dtype=object)}
        )


def test_initial_and_catalog_manifest_use_frozen_order60_contract():
    """Catch wrong resolution, state layout, catalogue count, or value hashing."""

    setup = core.build_setup(order=60, y_max=30.0, label="f10-prefix")
    arrays = fixture.build_initial_arrays(setup)
    assert arrays["y"].shape == (182,)
    assert float(arrays["N"]) == 0.0
    assert int(arrays["order"]) == 60
    assert int(arrays["state_dim"]) == 182
    assert float(arrays["y_max"]) == 30.0

    manifest = fixture.build_quadrature_catalog_manifest(setup)
    assert manifest["grid"]["nodes"]["count"] == 60
    assert manifest["grid"]["weights"]["count"] == 60
    assert manifest["catalogs"]["self_reactions"]["count"] == 48
    assert manifest["catalogs"]["electron_reactions"]["count"] == 18
    assert manifest["catalogs"]["self_events"]["count"] == 27
    assert manifest["catalogs"]["electron_events"]["count"] == 15


def test_source_bundle_manifest_resolves_exact_archives_and_tree():
    """Catch branch/tree drift or substitution of either retained archive."""

    manifest = fixture.build_source_bundle_manifest(Path.cwd())
    assert manifest["rabbit_source"]["commit"] == fixture.BASE_COMMIT
    assert len(manifest["rabbit_source"]["tree_oid"]) == 40
    assert manifest["solver_research_archive"]["sha256"] == (
        "8ffb9c34019e4bc9e431985df9fe69a347ced5da11f68308a1943187e3829fd8"
    )
    assert manifest["solver_research_archive"]["internal_history_commit"] == (
        "b8f11b03d9d59746c4ceddbb0712dfbd3f5386ab"
    )
    assert manifest["solver_research_archive"]["internal_bundle_verified"]
    assert manifest["solver_research_archive"]["internal_bundle_heads"] == [
        {
            "commit": "b8f11b03d9d59746c4ceddbb0712dfbd3f5386ab",
            "ref": "refs/heads/research/solver-algorithm-loop",
        }
    ]
    assert manifest["mathphysics_research_archive"]["sha256"] == (
        "bb3ca057d1ecee6b11e33bba5dbcd8325a23d95dfe925bb5a235866d05ed4fb0"
    )


def test_prepare_writes_only_prospective_inputs_before_receipts():
    """Catch any physical output or unsealed receipt created by preparation."""

    repo = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".f10-fixture-test-", dir=repo) as raw:
        output = Path(raw)
        fixture.prepare_fixture(repo, output)

        contract = fixture.read_json(output / "PREFIX_CONTRACT.json")
        assert contract["contract_status"] == "PROSPECTIVE_UNEXECUTED"
        assert contract["static_receipt_discriminator"]["jvp"]["relative_step"] == 1e-3
        assert contract["static_receipt_discriminator"]["arnoldi"]["krylov_dimension"] == 10
        assert not (output / "receipts/PHYSICAL_RHS_JVP_RECEIPTS.json").exists()
        assert not (output / "receipts/PHYSICAL_RHS_JVP_VECTORS.npz").exists()
        assert (output / "PREFIX_CONTRACT.sha256").read_text().endswith(
            "  " + (output / "PREFIX_CONTRACT.json").relative_to(repo).as_posix() + "\n"
        )


def test_contract_protects_runtime_sources_and_complete_v3a_domain():
    """Catch a seal that binds checkpoints but omits their raw provenance."""

    repo = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".f10-fixture-test-", dir=repo) as raw:
        output = Path(raw)
        fixture.prepare_fixture(repo, output)
        contract = fixture.read_json(output / "PREFIX_CONTRACT.json")
        protected = {entry["path"] for entry in contract["protected_paths"]}
        expected_domain = {
            path.relative_to(repo).as_posix()
            for path in (
                repo
                / ".agent-harness/runs/run-20260805-f10-v3-campaign/"
                "v3a_r2/domain"
            ).iterdir()
            if path.is_file()
        }

        assert set(fixture.PREFIX_SOURCE_PATHS) <= protected
        assert expected_domain
        assert expected_domain <= protected


def test_cli_advertises_every_presealed_command():
    """Catch sealing a runner that cannot verify, execute, or finalize artifacts."""

    result = subprocess.run(
        [sys.executable, str(Path(fixture.__file__)), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        "prepare",
        "verify-preseal",
        "run-receipts",
        "verify-receipts",
        "finalize",
        "verify-final",
    ):
        assert command in result.stdout


def test_checksum_manifest_covers_nested_and_external_bytes(tmp_path: Path):
    """Catch a final digest list that omits payloads or accepts later mutation."""

    repo = tmp_path / "repo"
    output = repo / fixture.DIAGNOSIS_DIR_NAME
    (output / "receipts").mkdir(parents=True)
    (output / "payload.json").write_text("{}\n", encoding="utf-8")
    (output / "receipts/vector.bin").write_bytes(b"vector")
    (repo / "external.zip").write_bytes(b"archive")

    count = fixture.write_sha256sums(repo, output, ("external.zip",))
    assert count == 3
    assert fixture.verify_sha256sums(repo, output) == 3
    assert fixture.DIAGNOSIS_DIR_NAME + "/SHA256SUMS" not in (
        output / "SHA256SUMS"
    ).read_text(encoding="utf-8")

    (repo / "external.zip").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fixture.verify_sha256sums(repo, output)


def test_static_receipt_validator_requires_all_four_direct_diagnostics():
    """Catch readiness inferred from a file that lacks a required state receipt."""

    states = []
    for label in fixture.REQUIRED_STATE_LABELS:
        states.append(
            {
                "label": label,
                "status": "EXECUTED",
                "source_path": "fixture",
                "base": {
                    "first_law_residual": 0.0,
                    "rhs_sha256": "b" * 64,
                    "collision_total_sha256": "c" * 64,
                    "occupation": {
                        "strict_open": True,
                        "minimum": 0.1,
                        "maximum": 0.9,
                    },
                    "domain": {
                        "whole_reaction_rejections": 0,
                        "matrix_roundoff_corrections": 0,
                        "largest_matrix_roundoff_correction": 0.0,
                    },
                    "tail": {
                        "equilibrium_number_fraction_beyond_ymax": 1e-9,
                        "equilibrium_energy_fraction_beyond_ymax": 2e-9,
                        "last_four_node_relative_distortion_max": 0.0,
                        "last_four_node_occupation_max": 1e-9,
                        "reaction_tail_authority_validated": False,
                    },
                },
                "arnoldi": {"source_sha256": "a" * 64},
                "jvp_calls": [{"scheme": "forward_time_augmented", "epsilon": 1e-3}],
                "rhs_call_accounting": {"full_rhs_equivalent_calls": 2},
            }
        )
    receipt = {
        "results": {
            "state_count": 4,
            "states": states,
            "historical_observation_jacobians_used": False,
            "physical_prefix_executed": False,
            "reaction_tail_authority_validated": False,
            "d071_reopen_earned": False,
        }
    }

    summary = fixture.validate_static_receipt_payload(receipt)
    assert summary["all_states_executed"]
    assert summary["all_required_diagnostics_present"]
    assert summary["direct_jvp_provenance_present"]
    assert not summary["reaction_tail_authority_validated"]

    receipt["results"]["states"] = states[:-1]
    with pytest.raises(ValueError, match="state labels"):
        fixture.validate_static_receipt_payload(receipt)


def test_receipt_evaluator_matches_frozen_rhs_on_real_collision_state():
    """Catch sign, chain, layout, temperature, or energy-transfer RHS drift."""

    setup = core.build_setup(
        order=8,
        y_max=8.0,
        incoming_polar_order=2,
        final_polar_order=2,
        electron_radial_order=8,
        label="test",
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
    assert observed.equilibrium_tail_number_fraction > 0.0
    assert observed.equilibrium_tail_energy_fraction > 0.0
    assert not observed.reaction_tail_authority_validated


def test_exact_bundle_arnoldi_runs_double_mgs_identity_case():
    """Catch substitution of the retained solver bundle or its Arnoldi source."""

    arnoldi = fixture.load_exact_arnoldi(Path(fixture.SOLVER_ZIP_NAME))
    result = arnoldi(
        lambda vector: vector,
        np.array([1.0, 0.0]),
        max_dim=2,
        tolerance=1e-12,
    )

    assert result.dimension == 1
    assert result.breakdown


def test_time_augmented_receipt_uses_direct_rhs_calls_and_fixed_rule():
    """Catch use of a stored observation Jacobian or a different JVP rule."""

    setup = core.build_setup(
        order=8,
        y_max=8.0,
        incoming_polar_order=2,
        final_polar_order=2,
        electron_radial_order=8,
        label="test",
    )
    _, state = core.initial_state(setup)
    arnoldi = fixture.load_exact_arnoldi(Path(fixture.SOLVER_ZIP_NAME))
    receipt, vectors = fixture.run_state_arnoldi_receipt(
        setup,
        "initial",
        "derived",
        0.0,
        state,
        arnoldi,
        relative_step=1e-3,
        krylov_dim=2,
        tolerance=1e-12,
    )

    assert receipt["jvp_rule"]["relative_step"] == 1e-3
    assert receipt["rhs_call_accounting"]["base_calls"] == 1
    assert receipt["rhs_call_accounting"]["shifted_calls"] == len(
        receipt["jvp_calls"]
    )
    assert receipt["arnoldi"]["dimension"] <= 2
    assert vectors["base_rhs"].shape == (26,)
    assert all(
        call["scheme"] == "forward_time_augmented"
        for call in receipt["jvp_calls"]
    )
    assert all(
        call["subtractive_condition_ratio"] is None
        or np.isfinite(call["subtractive_condition_ratio"])
        for call in receipt["jvp_calls"]
    )


def test_protected_path_validation_detects_digest_substitution():
    """Catch a working-tree input that no longer matches the prospective contract."""

    repo = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".f10-fixture-test-", dir=repo) as raw:
        output = Path(raw)
        fixture.prepare_fixture(repo, output)
        contract = fixture.read_json(output / "PREFIX_CONTRACT.json")
        assert fixture.verify_protected_paths(repo, contract) == len(
            contract["protected_paths"]
        )
        contract["protected_paths"][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="protected-path digest mismatch"):
            fixture.verify_protected_paths(repo, contract)


def test_receipt_set_preserves_a_state_failure_without_erasing_success():
    """Catch all-or-nothing output loss when one physical state is invalid."""

    setup = core.build_setup(
        order=8,
        y_max=8.0,
        incoming_polar_order=2,
        final_polar_order=2,
        electron_radial_order=8,
        label="test",
    )
    _, valid = core.initial_state(setup)
    invalid = valid.copy()
    invalid[0] = np.nan
    arnoldi = fixture.load_exact_arnoldi(Path(fixture.SOLVER_ZIP_NAME))

    payload, vectors = fixture.run_receipt_set(
        setup,
        [
            ("valid", "derived", 0.0, valid),
            ("invalid", "derived-invalid", 0.0, invalid),
        ],
        arnoldi,
        relative_step=1e-3,
        krylov_dim=1,
        tolerance=1e-12,
    )

    assert payload["overall_status"] == "EXECUTED_WITH_RETAINED_FAILURES"
    assert payload["states"][0]["status"].startswith("EXECUTED")
    assert payload["states"][1]["status"] == "ERROR_RETAINED"
    assert payload["states"][1]["error_type"] == "ValueError"
    assert any(name.startswith("valid__") for name in vectors)


def test_seal_verification_uses_real_git_commit_and_protected_bytes(tmp_path: Path):
    """Catch a non-HEAD seal or post-seal protected-file mutation."""

    repo = tmp_path / "repo"
    diagnosis = repo / fixture.DIAGNOSIS_DIR_NAME
    diagnosis.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "diagnosis_report", repo], check=True)
    subprocess.run(
        ["git", "-C", repo, "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo, "config", "user.name", "Fixture Test"], check=True
    )
    payload_path = repo / "payload.txt"
    payload_path.write_text("sealed\n", encoding="utf-8")
    contract_path = diagnosis / "PREFIX_CONTRACT.json"
    contract_path.write_text(
        json.dumps(
            {
                "protected_paths": [
                    {
                        "path": "payload.txt",
                        "sha256": fixture.sha256_path(payload_path),
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (diagnosis / "PREFIX_CONTRACT.sha256").write_text(
        f"{fixture.sha256_path(contract_path)}  "
        f"{fixture.DIAGNOSIS_DIR_NAME}/PREFIX_CONTRACT.json\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-m", "seal"], check=True)
    seal = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    verified = fixture.verify_seal(repo, diagnosis, seal, require_clean=True)
    assert verified["seal_commit"] == seal
    assert verified["protected_path_count"] == 1
    payload_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protected-path digest mismatch"):
        fixture.verify_seal(repo, diagnosis, seal, require_clean=False)


def test_receipt_artifacts_bind_contract_seal_and_numeric_vectors():
    """Catch wrong seal metadata, unsafe NPZ content, or detached receipt vectors."""

    repo = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".f10-fixture-test-", dir=repo) as raw:
        output = Path(raw)
        fixture.prepare_fixture(repo, output)
        contract_sha = fixture.sha256_path(output / "PREFIX_CONTRACT.json")
        payload = {
            "overall_status": "EXECUTED",
            "state_count": 1,
            "failure_count": 0,
            "full_rhs_equivalent_calls": 1,
            "states": [{"label": "tiny", "status": "EXECUTED"}],
        }
        vectors = {"tiny__base_rhs": np.array([1.0, 2.0])}
        seal = "a" * 40
        fixture.write_receipt_artifacts(
            output,
            payload,
            vectors,
            seal_commit=seal,
            contract_sha256=contract_sha,
            started_utc="2026-08-11T00:00:00Z",
            finished_utc="2026-08-11T00:00:01Z",
            wall_seconds=1.0,
        )

        summary = fixture.verify_receipt_artifacts(output, seal, contract_sha)
        assert summary["state_count"] == 1
        assert summary["vector_count"] == 1
        with pytest.raises(ValueError, match="receipt seal commit"):
            fixture.verify_receipt_artifacts(output, "b" * 40, contract_sha)


def test_preseal_and_state_loader_bind_initial_plus_three_retained_states():
    """Catch an omitted checkpoint, wrong label, or receipt emitted before sealing."""

    repo = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".f10-fixture-test-", dir=repo) as raw:
        output = Path(raw)
        fixture.prepare_fixture(repo, output)

        summary = fixture.verify_preseal(repo, output)
        states = fixture.load_receipt_states(repo, output)
        assert summary["receipt_outputs_absent"]
        assert [state[0] for state in states] == [
            "initial",
            "creep_1200",
            "creep_2000",
            "creep_3000",
        ]
        assert [state[3].shape for state in states] == [(182,)] * 4
        assert states[0][2] == 0.0
        assert states[1][2] == pytest.approx(0.16286930247517223)
