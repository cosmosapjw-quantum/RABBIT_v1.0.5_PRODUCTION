from rabbit.debug.posbulk_tailsplit_contract import (
    ONSET_MAX_STATE_INDEX,
    POSBULK_TAIL_CUT_V1,
    CONTRACT_VERSION,
    CLUSTER_STATUS,
    classify_cluster,
    production_allows,
    research_bank_allows,
    shadow_only,
)

def test_constants():
    assert ONSET_MAX_STATE_INDEX == 40
    assert abs(POSBULK_TAIL_CUT_V1 - 0.8245908567511172) < 1e-15
    assert CONTRACT_VERSION == "cut40_tail8245908567511172_v1"

def test_cluster_status():
    assert CLUSTER_STATUS["neg"] == "disabled"
    assert CLUSTER_STATUS["pos_onset"] == "validated_case_holdout"
    assert CLUSTER_STATUS["pos_bulk_hightail"] == "validated_case_holdout"
    assert CLUSTER_STATUS["pos_bulk_lowtail"] == "undercovered_shadow_only"

def test_classify():
    assert classify_cluster(sigma_plus=-1.0, state_index=0, tail_last5_share=0.9) == "neg"
    assert classify_cluster(sigma_plus=+1.0, state_index=0, tail_last5_share=0.9) == "pos_onset"
    assert classify_cluster(sigma_plus=+1.0, state_index=40, tail_last5_share=0.9) == "pos_onset"
    assert classify_cluster(sigma_plus=+1.0, state_index=41, tail_last5_share=0.7) == "pos_bulk_lowtail"
    assert classify_cluster(sigma_plus=+1.0, state_index=41, tail_last5_share=0.9) == "pos_bulk_hightail"

def test_enablement():
    assert production_allows("pos_onset") is False
    assert production_allows("pos_bulk_hightail") is False
    assert research_bank_allows("pos_onset") is True
    assert research_bank_allows("pos_bulk_hightail") is True
    assert research_bank_allows("pos_bulk_lowtail") is False
    assert shadow_only("pos_bulk_lowtail") is True
