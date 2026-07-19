from __future__ import annotations

ONSET_MAX_STATE_INDEX = 40
POSBULK_TAIL_CUT_V1 = 0.8245908567511172
CONTRACT_VERSION = "cut40_tail8245908567511172_v1"

CLUSTER_STATUS = {
    "neg": "disabled",
    "pos_onset": "validated_case_holdout",
    "pos_bulk_hightail": "validated_case_holdout",
    "pos_bulk_lowtail": "undercovered_shadow_only",
}

def classify_cluster(*, sigma_plus: float, state_index: int, tail_last5_share: float | None = None) -> str:
    if sigma_plus < 0.0:
        return "neg"
    if state_index <= ONSET_MAX_STATE_INDEX:
        return "pos_onset"
    if tail_last5_share is None:
        return "pos_bulk_hightail"
    return "pos_bulk_lowtail" if tail_last5_share <= POSBULK_TAIL_CUT_V1 else "pos_bulk_hightail"

def production_allows(cluster: str) -> bool:
    return False

def research_bank_allows(cluster: str) -> bool:
    return cluster in ("pos_onset", "pos_bulk_hightail")

def shadow_only(cluster: str) -> bool:
    return cluster == "pos_bulk_lowtail"
