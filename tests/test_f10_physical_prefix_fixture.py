"""Regression tests for the F-10 physical-prefix provenance fixture."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.audit import f10_physical_prefix_fixture as fixture


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
