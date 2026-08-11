"""Regression tests for the F-10 physical-prefix provenance fixture."""

from __future__ import annotations

import hashlib
import struct
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
