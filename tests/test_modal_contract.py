from rabbit.debug.modal_contract import (
    RAW_AUTHORITATIVE_PATH,
    REDUCED_MODAL_MODE,
    ONSET_MAX_STATE_INDEX,
    cluster_label,
    production_allows_reduced_modal,
    research_allows_cluster_bank,
)

def test_contract_constants():
    assert RAW_AUTHORITATIVE_PATH == "raw_characteristic"
    assert REDUCED_MODAL_MODE == "offline_only"
    assert ONSET_MAX_STATE_INDEX == 40

def test_cluster_label():
    assert cluster_label(sigma_plus=-1e-3, state_index=0) == "neg"
    assert cluster_label(sigma_plus=+1e-3, state_index=0) == "pos_onset"
    assert cluster_label(sigma_plus=+1e-3, state_index=40) == "pos_onset"
    assert cluster_label(sigma_plus=+1e-3, state_index=41) == "pos_bulk"

def test_policy():
    assert production_allows_reduced_modal(cluster="neg") is False
    assert production_allows_reduced_modal(cluster="pos_onset") is False
    assert production_allows_reduced_modal(cluster="pos_bulk") is False
    assert research_allows_cluster_bank(cluster="neg") is False
    assert research_allows_cluster_bank(cluster="pos_onset") is True
    assert research_allows_cluster_bank(cluster="pos_bulk") is True
