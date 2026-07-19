from __future__ import annotations

RESEARCH_BANK_VERSION = "cut40_v1"
ONSET_MAX_STATE_INDEX = 40

RESEARCH_BANK_STATUS = {
    "neg": "disabled",
    "pos_onset": "validated_case_holdout",
    "pos_bulk": "provisional_case_holdout",
}

RESEARCH_BANK_THRESHOLDS = {
    "pos_onset": {"max_worst_resid": 0.05, "min_worst_cos": 0.998},
    "pos_bulk": {"max_worst_resid": 0.20, "min_worst_cos": 0.98},
}

def cluster_label(*, sigma_plus: float, state_index: int, onset_max_state_index: int = ONSET_MAX_STATE_INDEX) -> str:
    if sigma_plus < 0.0:
        return "neg"
    return "pos_onset" if state_index <= onset_max_state_index else "pos_bulk"

def research_cluster_enabled(cluster: str) -> bool:
    return cluster in ("pos_onset", "pos_bulk")

def production_cluster_enabled(cluster: str) -> bool:
    return False
