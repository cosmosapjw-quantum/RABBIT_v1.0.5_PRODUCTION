from rabbit.debug.research_bank_contract import (
    RESEARCH_BANK_VERSION,
    ONSET_MAX_STATE_INDEX,
    RESEARCH_BANK_STATUS,
    RESEARCH_BANK_THRESHOLDS,
    cluster_label,
    research_cluster_enabled,
    production_cluster_enabled,
)

def test_contract_basics():
    assert RESEARCH_BANK_VERSION == "cut40_v1"
    assert ONSET_MAX_STATE_INDEX == 40
    assert RESEARCH_BANK_STATUS["neg"] == "disabled"
    assert RESEARCH_BANK_STATUS["pos_onset"] == "validated_case_holdout"
    assert RESEARCH_BANK_STATUS["pos_bulk"] == "provisional_case_holdout"

def test_cluster_label():
    assert cluster_label(sigma_plus=-1e-3, state_index=0) == "neg"
    assert cluster_label(sigma_plus=+1e-3, state_index=40) == "pos_onset"
    assert cluster_label(sigma_plus=+1e-3, state_index=41) == "pos_bulk"

def test_enablement():
    assert research_cluster_enabled("pos_onset") is True
    assert research_cluster_enabled("pos_bulk") is True
    assert research_cluster_enabled("neg") is False
    assert production_cluster_enabled("pos_onset") is False
    assert production_cluster_enabled("pos_bulk") is False

def test_thresholds():
    assert RESEARCH_BANK_THRESHOLDS["pos_onset"]["max_worst_resid"] == 0.05
    assert RESEARCH_BANK_THRESHOLDS["pos_bulk"]["max_worst_resid"] == 0.20
