from __future__ import annotations

import math

from rabbit.transport.even_ladder_analysis import integrate_even_ladder_final_state
from rabbit.transport.even_ladder_observable_bridge import (
    energy_density_ratio_from_state,
    observable_snapshot_from_state,
)


def test_energy_density_ratio_is_one_in_flrw():
    state = integrate_even_ladder_final_state(lmax=8, sigma_h=0.0, n_q=20)
    ratio = energy_density_ratio_from_state(state)
    assert math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_observable_snapshot_flrw_independent_of_lmax():
    state2 = integrate_even_ladder_final_state(lmax=2, sigma_h=0.0, n_q=20)
    state8 = integrate_even_ladder_final_state(lmax=8, sigma_h=0.0, n_q=20)
    snap2 = observable_snapshot_from_state(state2, T_gamma_MeV=0.8)
    snap8 = observable_snapshot_from_state(state8, T_gamma_MeV=0.8)
    for key in ('rho_ratio', 'N_eff_like', 'lambda_np', 'lambda_pn', 'Xn_eq_proxy', 'Yp_proxy'):
        assert math.isclose(snap2[key], snap8[key], rel_tol=0.0, abs_tol=1e-12)
