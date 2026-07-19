from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_stageA_hierarchy import STAGE_A_ELLS, compute_hierarchy_rhs_typeI_stageA, q_d_dq
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS, compute_hierarchy_rhs_typeI_stageB

StageName = Literal['A', 'B']


def q_weighted_l2(values: np.ndarray, grid: MomentumGrid) -> float:
    quad_weight = grid.weights * np.exp(grid.nodes)
    quad_norm = float(np.sum(quad_weight))
    return float(np.sqrt(np.sum(quad_weight * values**2) / max(quad_norm, 1.0e-30)))


def integrate_stage_final_state(stage: StageName, sigma_h: float, n_q: int, n_end: float = 0.5, *, rtol: float = 1.0e-8, atol: float = 1.0e-10) -> AxisymmetricHierarchyState:
    grid = MomentumGrid(N_q=n_q)
    if stage == 'A':
        active_ells = STAGE_A_ELLS
        rhs_fn = compute_hierarchy_rhs_typeI_stageA
    elif stage == 'B':
        active_ells = STAGE_B_ELLS
        rhs_fn = compute_hierarchy_rhs_typeI_stageB
    else:
        raise ValueError(stage)
    state0 = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active_ells, n_species=1)

    def rhs(_n: float, y: np.ndarray) -> np.ndarray:
        st = AxisymmetricHierarchyState.from_flat(y, grid, active_ells, n_species=1)
        return rhs_fn(st, Sigma_H=sigma_h)

    sol = solve_ivp(rhs, (0.0, n_end), state0.to_flat(), method='RK45', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed for stage={stage}, sigma={sigma_h}, N_q={n_q}: {sol.message}')
    return AxisymmetricHierarchyState.from_flat(sol.y[:, -1], grid, active_ells, n_species=1)


@dataclass(frozen=True)
class StageBClosureResidual:
    sigma_h: float
    keep_l2: float
    omit_self_l2: float
    omit_f4_l2: float
    omit_total_l2: float
    ratio_self_to_keep: float
    ratio_f4_to_keep: float
    ratio_total_to_keep: float


def compute_stageB_closure_residual(state_b: AxisymmetricHierarchyState, sigma_h: float, *, species_idx: int = 0) -> StageBClosureResidual:
    if tuple(state_b.active_ells) != STAGE_B_ELLS:
        raise ValueError('Stage-B state with active ells (0,2,4) required')
    grid = state_b.grid
    q = grid.nodes
    F0 = state_b.moment(species_idx, 0)
    F2 = state_b.moment(species_idx, 2)
    F4 = state_b.moment(species_idx, 4)
    keep = sigma_h * q_d_dq(F0, q)
    omit_self = sigma_h * q_d_dq((2.0/7.0) * F2, q) + 1.5 * sigma_h * ((2.0/7.0) * F2)
    omit_f4 = sigma_h * q_d_dq((2.0/7.0) * F4, q) + 1.5 * sigma_h * ((20.0/21.0) * F4)
    omit_total = omit_self + omit_f4
    keep_l2 = q_weighted_l2(keep, grid)
    omit_self_l2 = q_weighted_l2(omit_self, grid)
    omit_f4_l2 = q_weighted_l2(omit_f4, grid)
    omit_total_l2 = q_weighted_l2(omit_total, grid)
    denom = max(keep_l2, 1.0e-30)
    return StageBClosureResidual(
        sigma_h=float(sigma_h),
        keep_l2=keep_l2,
        omit_self_l2=omit_self_l2,
        omit_f4_l2=omit_f4_l2,
        omit_total_l2=omit_total_l2,
        ratio_self_to_keep=float(omit_self_l2 / denom),
        ratio_f4_to_keep=float(omit_f4_l2 / denom),
        ratio_total_to_keep=float(omit_total_l2 / denom),
    )


def fit_power_law(xs: list[float], ys: list[float]) -> dict[str, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return {'slope': math.nan, 'intercept': math.nan, 'n_used': int(np.sum(mask))}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    return {'slope': float(slope), 'intercept': float(intercept), 'n_used': int(np.sum(mask))}


def summarize_stage_state(state: AxisymmetricHierarchyState, *, species_idx: int = 0) -> dict[str, float]:
    grid = state.grid
    fd = fermi_dirac(grid.nodes)
    out = {'F0_L2': q_weighted_l2(state.moment(species_idx, 0), grid), 'deltaF0_L2': q_weighted_l2(state.moment(species_idx, 0) - fd, grid)}
    if 2 in state.active_ells:
        out['F2_L2'] = q_weighted_l2(state.moment(species_idx, 2), grid)
    if 4 in state.active_ells:
        out['F4_L2'] = q_weighted_l2(state.moment(species_idx, 4), grid)
    return out
