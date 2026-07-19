from __future__ import annotations

RAW_AUTHORITATIVE_PATH = "raw_characteristic"
REDUCED_MODAL_MODE = "offline_only"

ONSET_MAX_STATE_INDEX = 40

ACTIVE_RESEARCH_CLUSTERS = ("pos_onset", "pos_bulk")
DISABLED_CLUSTERS = ("neg",)

def cluster_label(*, sigma_plus: float, state_index: int, onset_max_state_index: int = ONSET_MAX_STATE_INDEX) -> str:
    if sigma_plus < 0.0:
        return "neg"
    return "pos_onset" if state_index <= onset_max_state_index else "pos_bulk"

def production_allows_reduced_modal(*, cluster: str) -> bool:
    return False

def research_allows_cluster_bank(*, cluster: str) -> bool:
    return cluster in ACTIVE_RESEARCH_CLUSTERS
