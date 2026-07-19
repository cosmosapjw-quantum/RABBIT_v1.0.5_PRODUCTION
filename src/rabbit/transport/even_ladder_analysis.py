from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_analysis import fit_power_law, q_weighted_l2
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_even_ladder_hierarchy import active_even_ells, compute_hierarchy_rhs_typeI_even_ladder


ALL_EVEN_LS = (0, 2, 4, 6, 8)


def integrate_even_ladder_final_state(lmax: int, sigma_h: float, n_q: int, n_end: float = 0.5, *, rtol: float = 1.0e-8, atol: float = 1.0e-10) -> AxisymmetricHierarchyState:
    grid = MomentumGrid(N_q=n_q)
    active = active_even_ells(lmax)
    state0 = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active, n_species=1)

    def rhs(_n: float, y: np.ndarray) -> np.ndarray:
        st = AxisymmetricHierarchyState.from_flat(y, grid, active, n_species=1)
        return compute_hierarchy_rhs_typeI_even_ladder(st, Sigma_H=sigma_h)

    sol = solve_ivp(rhs, (0.0, n_end), state0.to_flat(), method='RK45', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed for lmax={lmax}, sigma={sigma_h}, N_q={n_q}: {sol.message}')
    return AxisymmetricHierarchyState.from_flat(sol.y[:, -1], grid, active, n_species=1)


def summarize_even_ladder_state(state: AxisymmetricHierarchyState, *, species_idx: int = 0) -> dict[str, float]:
    grid = state.grid
    fd = fermi_dirac(grid.nodes)
    out: dict[str, float] = {
        'F0_L2': q_weighted_l2(state.moment(species_idx, 0), grid),
        'deltaF0_L2': q_weighted_l2(state.moment(species_idx, 0) - fd, grid),
    }
    for ell in state.active_ells:
        if ell == 0:
            continue
        out[f'F{ell}_L2'] = q_weighted_l2(state.moment(species_idx, ell), grid)
    return out


def compare_even_ladder_to_reference(state: AxisymmetricHierarchyState, ref_state: AxisymmetricHierarchyState, *, species_idx: int = 0) -> dict[str, float]:
    if state.grid.N_q != ref_state.grid.N_q:
        raise ValueError('state and ref_state must share the same N_q')
    grid = state.grid
    out: dict[str, float] = {}
    out['deltaF0_diff_to_ref_L2'] = q_weighted_l2(
        state.moment(species_idx, 0) - ref_state.moment(species_idx, 0),
        grid,
    )
    for ell in ALL_EVEN_LS[1:]:
        if ell in state.active_ells and ell in ref_state.active_ells:
            out[f'F{ell}_diff_to_ref_L2'] = q_weighted_l2(
                state.moment(species_idx, ell) - ref_state.moment(species_idx, ell),
                grid,
            )
    return out


def fit_even_ladder_small_shear(rows: list[dict[str, Any]], *, n_q: int, lmax: int, sigma_max: float = 0.1) -> list[dict[str, float | int | str]]:
    subset = [r for r in rows if r['N_q'] == n_q and r['lmax'] == lmax and 0.0 < r['Sigma_H'] <= sigma_max]
    subset = sorted(subset, key=lambda r: r['Sigma_H'])
    sigmas = [float(r['Sigma_H']) for r in subset]
    metrics = ['deltaF0_L2'] + [f'F{ell}_L2' for ell in active_even_ells(lmax) if ell != 0]
    out: list[dict[str, float | int | str]] = []
    for metric in metrics:
        fit = fit_power_law(sigmas, [float(r.get(metric, math.nan)) for r in subset])
        out.append({'N_q': n_q, 'lmax': lmax, 'metric': metric, **fit})
    return out
