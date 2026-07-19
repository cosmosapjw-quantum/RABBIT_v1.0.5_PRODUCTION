"""Bianchi rollout registry integrity tests."""
from __future__ import annotations

import pytest

from rabbit.config.bianchi_cells import (
    BianchiOrientation,
    canonical_bianchi_cells,
    exceptional_bianchi_cells,
    full_bianchi_gate_node_ids,
    tilted_bianchi_gate_node_ids,
)
from rabbit.config.claim_gates import (
    GATE_FULL_BIANCHI_BBN,
    GATE_TILTED_OUTSIDE_TYPE_I_SCALAR,
)
from rabbit.config.conventions import BianchiType


@pytest.mark.production
@pytest.mark.release_smoke
class TestBianchiCellRegistry:
    def test_eleven_families_times_two_modes(self):
        cells = canonical_bianchi_cells()
        families = {cell.bianchi_type for cell in cells}
        assert len(families) == 11
        assert len(cells) == 22

        for family in families:
            modes = {cell.orientation for cell in cells if cell.bianchi_type is family}
            assert modes == {BianchiOrientation.ORTHOGONAL, BianchiOrientation.TILTED}

    def test_vi_minus_one_ninth_is_exceptional_not_family_count(self):
        canonical = {cell.bianchi_type for cell in canonical_bianchi_cells()}
        exceptional = exceptional_bianchi_cells()

        assert BianchiType.TYPE_VI_M19 not in canonical
        assert len(exceptional) == 2
        assert {cell.orientation for cell in exceptional} == {
            BianchiOrientation.ORTHOGONAL,
            BianchiOrientation.TILTED,
        }
        assert {cell.specialization_of for cell in exceptional} == {BianchiType.TYPE_VIH}

    def test_full_bianchi_gate_uses_exact_22_canonical_cells(self):
        assert GATE_FULL_BIANCHI_BBN.required_test_node_ids == full_bianchi_gate_node_ids()
        assert len(GATE_FULL_BIANCHI_BBN.required_test_node_ids) == 22
        assert all("typeVI_m19" not in node for node in GATE_FULL_BIANCHI_BBN.required_test_node_ids)

    def test_tilted_gate_targets_non_type_i_families_plus_vector_parity(self):
        tilted_nodes = tilted_bianchi_gate_node_ids(include_type_i=False)
        gate_nodes = GATE_TILTED_OUTSIDE_TYPE_I_SCALAR.required_test_node_ids

        assert len(tilted_nodes) == 10
        for node in tilted_nodes:
            assert node in gate_nodes
        assert "tests/gold/test_typeI_tilted_bbn.py" not in gate_nodes
        assert any("tilt_vector_scalar_parity" in node for node in gate_nodes)
