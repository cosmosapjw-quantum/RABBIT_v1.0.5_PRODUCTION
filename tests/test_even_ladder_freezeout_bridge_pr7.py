from __future__ import annotations

from rabbit.transport.even_ladder_freezeout_bridge import freezeout_bridge_row


def test_freezeout_bridge_flrw_independent_of_lmax() -> None:
    rows = [
        freezeout_bridge_row(lmax=lmax, sigma_h=0.0, n_q=20, T_start_MeV=2.0, T_handoff_MeV=0.08, T_decay_final_MeV=0.06)
        for lmax in (2, 4, 6, 8)
    ]
    ref = rows[-1]
    for row in rows[:-1]:
        assert abs(row['Xn_handoff'] - ref['Xn_handoff']) < 1.0e-12
        assert abs(row['Yp_decay_proxy'] - ref['Yp_decay_proxy']) < 1.0e-12


def test_freezeout_bridge_high_shear_converges_to_l8() -> None:
    rows = {
        lmax: freezeout_bridge_row(lmax=lmax, sigma_h=0.5, n_q=40, T_start_MeV=2.0, T_handoff_MeV=0.08, T_decay_final_MeV=0.06)
        for lmax in (2, 4, 6, 8)
    }
    ref = rows[8]['Yp_decay_proxy']
    d2 = abs(rows[2]['Yp_decay_proxy'] - ref)
    d4 = abs(rows[4]['Yp_decay_proxy'] - ref)
    d6 = abs(rows[6]['Yp_decay_proxy'] - ref)
    assert d2 > d4 > d6
