from rabbit.transport.even_ladder_freezeout_bridge import (
    freezeout_bridge_row,
    handoff_metadata_from_freezeout_row,
)


def test_handoff_metadata_aliases_match_row_values() -> None:
    row = freezeout_bridge_row(
        lmax=6,
        sigma_h=0.3,
        n_q=20,
        T_start_MeV=2.0,
        T_handoff_MeV=0.08,
        T_decay_final_MeV=0.06,
    )
    meta = handoff_metadata_from_freezeout_row(row)
    assert meta['selection_contract'] == 'finite_even_ladder_freezeout_bridge_proxy_v1'
    assert abs(meta['phase1_handoff_T'] - row['T_handoff_MeV']) < 1.0e-15
    assert abs(meta['phase1_handoff_T_gamma'] - row['phase1_handoff_T_gamma']) < 1.0e-15
    assert abs(meta['phase1_handoff_T_nu_e'] - row['T_nu_handoff_MeV']) < 1.0e-15
    assert abs(meta['phase1_handoff_lambda_np'] - row['lambda_np_handoff']) < 1.0e-15
    assert abs(meta['phase1_handoff_lambda_pn'] - row['lambda_pn_handoff']) < 1.0e-15
    assert abs(meta['phase1_handoff_Xn'] - row['Xn_handoff']) < 1.0e-15
    assert abs(meta['Xn_freeze'] - row['Xn_handoff']) < 1.0e-15
    assert abs(meta['Yp_decay_proxy'] - row['Yp_decay_proxy']) < 1.0e-15
    assert 'proxy' in meta['honesty_note']


def test_handoff_metadata_converges_with_lmax() -> None:
    rows = {
        lmax: freezeout_bridge_row(
            lmax=lmax,
            sigma_h=0.5,
            n_q=40,
            T_start_MeV=2.0,
            T_handoff_MeV=0.08,
            T_decay_final_MeV=0.06,
        )
        for lmax in (2, 4, 6, 8)
    }
    metas = {lmax: handoff_metadata_from_freezeout_row(row) for lmax, row in rows.items()}
    ref = metas[8]
    d2 = abs(metas[2]['phase1_handoff_lambda_np'] - ref['phase1_handoff_lambda_np'])
    d4 = abs(metas[4]['phase1_handoff_lambda_np'] - ref['phase1_handoff_lambda_np'])
    d6 = abs(metas[6]['phase1_handoff_lambda_np'] - ref['phase1_handoff_lambda_np'])
    assert d6 < d4 < d2
