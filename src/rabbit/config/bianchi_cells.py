"""Bianchi family promotion cells for the orthogonal/tilted rollout.

This module keeps the mathematical taxonomy used by the claim gates separate
from the implementation enum.  The enum contains ``TYPE_VI_M19`` as an
exceptional validation slice, but the canonical Bianchi-family count is the
standard eleven families:

I, II, III, IV, V, VI_0, VI_h, VII_0, VII_h, VIII, IX.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from rabbit.config.conventions import BianchiType


class BianchiOrientation(str, Enum):
    """Promotion modes for each canonical Bianchi family."""

    ORTHOGONAL = "orthogonal"
    TILTED = "tilted"


@dataclass(frozen=True)
class BianchiPromotionCell:
    """One claim-promotion cell in the Bianchi rollout grid."""

    bianchi_type: BianchiType
    orientation: BianchiOrientation
    wave: int
    class_kind: str
    family_label: str
    file_stem: str
    gold_node_id: str
    specialization_of: BianchiType | None = None


_FILE_STEM = {
    BianchiType.TYPE_I: "typeI",
    BianchiType.TYPE_II: "typeII",
    BianchiType.TYPE_III: "typeIII",
    BianchiType.TYPE_IV: "typeIV",
    BianchiType.TYPE_V: "typeV",
    BianchiType.TYPE_VI0: "typeVI0",
    BianchiType.TYPE_VIH: "typeVIh",
    BianchiType.TYPE_VII0: "typeVII0",
    BianchiType.TYPE_VIIH: "typeVIIh",
    BianchiType.TYPE_VIII: "typeVIII",
    BianchiType.TYPE_IX: "typeIX",
    BianchiType.TYPE_VI_M19: "typeVI_m19",
}


# Ordered by the intended promotion waves, not by enum declaration order.
CANONICAL_BIANCHI_FAMILIES: Tuple[tuple[BianchiType, str, int, str], ...] = (
    (BianchiType.TYPE_I, "I", 0, "A"),
    (BianchiType.TYPE_II, "II", 1, "A"),
    (BianchiType.TYPE_V, "V", 1, "B"),
    (BianchiType.TYPE_IV, "IV", 2, "B"),
    (BianchiType.TYPE_VII0, "VII_0", 3, "A"),
    (BianchiType.TYPE_III, "III", 4, "B"),
    (BianchiType.TYPE_VIIH, "VII_h", 4, "B"),
    (BianchiType.TYPE_VIH, "VI_h", 4, "B"),
    (BianchiType.TYPE_VI0, "VI_0", 5, "A"),
    (BianchiType.TYPE_VIII, "VIII", 5, "A"),
    (BianchiType.TYPE_IX, "IX", 7, "A"),
)


EXCEPTIONAL_BIANCHI_SLICES: Tuple[tuple[BianchiType, str, int, str, BianchiType], ...] = (
    (BianchiType.TYPE_VI_M19, "VI_{-1/9}", 6, "B", BianchiType.TYPE_VIH),
)


def _cell(
    bianchi_type: BianchiType,
    family_label: str,
    wave: int,
    class_kind: str,
    orientation: BianchiOrientation,
    specialization_of: BianchiType | None = None,
) -> BianchiPromotionCell:
    stem = _FILE_STEM[bianchi_type]
    return BianchiPromotionCell(
        bianchi_type=bianchi_type,
        orientation=orientation,
        wave=int(wave),
        class_kind=class_kind,
        family_label=family_label,
        file_stem=stem,
        gold_node_id=f"tests/gold/test_{stem}_{orientation.value}_bbn.py",
        specialization_of=specialization_of,
    )


def canonical_bianchi_cells() -> Tuple[BianchiPromotionCell, ...]:
    """Return the 11-family x 2-mode rollout grid."""
    cells: list[BianchiPromotionCell] = []
    for bianchi_type, label, wave, class_kind in CANONICAL_BIANCHI_FAMILIES:
        for orientation in BianchiOrientation:
            cells.append(_cell(bianchi_type, label, wave, class_kind, orientation))
    return tuple(cells)


def exceptional_bianchi_cells() -> Tuple[BianchiPromotionCell, ...]:
    """Return validation cells that are not counted as separate families."""
    cells: list[BianchiPromotionCell] = []
    for bianchi_type, label, wave, class_kind, specialization_of in EXCEPTIONAL_BIANCHI_SLICES:
        for orientation in BianchiOrientation:
            cells.append(
                _cell(
                    bianchi_type,
                    label,
                    wave,
                    class_kind,
                    orientation,
                    specialization_of=specialization_of,
                )
            )
    return tuple(cells)


def full_bianchi_gate_node_ids(*, include_exceptional: bool = False) -> Tuple[str, ...]:
    """Gold node IDs required for the full 11-family Bianchi claim."""
    cells = list(canonical_bianchi_cells())
    if include_exceptional:
        cells.extend(exceptional_bianchi_cells())
    return tuple(cell.gold_node_id for cell in cells)


def tilted_bianchi_gate_node_ids(
    *,
    include_type_i: bool = False,
    include_exceptional: bool = False,
) -> Tuple[str, ...]:
    """Gold node IDs for tilted Bianchi promotion beyond the scalar Type-I slice."""
    cells = [
        cell for cell in canonical_bianchi_cells()
        if cell.orientation is BianchiOrientation.TILTED
        and (include_type_i or cell.bianchi_type is not BianchiType.TYPE_I)
    ]
    if include_exceptional:
        cells.extend(
            cell for cell in exceptional_bianchi_cells()
            if cell.orientation is BianchiOrientation.TILTED
        )
    return tuple(cell.gold_node_id for cell in cells)


__all__ = [
    "BianchiOrientation",
    "BianchiPromotionCell",
    "CANONICAL_BIANCHI_FAMILIES",
    "EXCEPTIONAL_BIANCHI_SLICES",
    "canonical_bianchi_cells",
    "exceptional_bianchi_cells",
    "full_bianchi_gate_node_ids",
    "tilted_bianchi_gate_node_ids",
]
