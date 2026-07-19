from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from rabbit.decoupling.grid import DecouplingGrid
from rabbit.decoupling.rhs import (
    decoupling_layout,
    initial_decoupling_state,
    isotropic_decoupling_observables,
    isotropic_decoupling_rhs,
)
from rabbit.solver.ivp_outcome import classify_solve_ivp_result


@dataclass(frozen=True)
class IsotropicDecouplingConfig:
    T_start: float = 3.0
    T_end: float = 0.05
    n_y: int = 41
    y_min: float = 0.02
    y_max: float = 20.0
    rtol: float = 1.0e-6
    atol: float = 1.0e-9
    max_efolds: float = 10.0
    method: str = "BDF"


@dataclass
class IsotropicDecouplingResult:
    success: bool
    solver_diagnostics: dict
    T_gamma_final: float
    T_nu_e_final: float
    T_nu_x_final: float
    N_eff_final: float
    grid: DecouplingGrid
    y_final: np.ndarray
    f_nue_final: np.ndarray
    f_nuebar_final: np.ndarray
    f_nux_final: np.ndarray
    trajectory: dict[str, np.ndarray]


def _decoupling_jac_sparsity(grid: DecouplingGrid) -> csr_matrix:
    layout = decoupling_layout(grid)
    mask = np.zeros((layout.n_total, layout.n_total), dtype=bool)

    # T_gamma row depends on the full state through the energy moments.
    mask[layout.i_Tg, :] = True

    # nue / nuebar rows depend on T_gamma and the electron-flavor pair block.
    e_cols = {layout.i_Tg}
    e_cols.update(range(layout.i_nue.start, layout.i_nue.stop))
    e_cols.update(range(layout.i_nuebar.start, layout.i_nuebar.stop))
    mask[layout.i_nue, list(e_cols)] = True
    mask[layout.i_nuebar, list(e_cols)] = True

    # nux rows depend on T_gamma and the nux block.
    x_cols = {layout.i_Tg}
    x_cols.update(range(layout.i_nux.start, layout.i_nux.stop))
    mask[layout.i_nux, list(x_cols)] = True
    return csr_matrix(mask)


def solve_isotropic_decoupling(config: IsotropicDecouplingConfig = IsotropicDecouplingConfig()) -> IsotropicDecouplingResult:
    grid = DecouplingGrid(n_y=config.n_y, y_min=config.y_min, y_max=config.y_max)
    layout = decoupling_layout(grid)
    y0 = initial_decoupling_state(grid, T_gamma_init=config.T_start)

    def stop_at_T_end(N, y):
        return float(y[layout.i_Tg] - config.T_end)

    stop_at_T_end.terminal = True
    stop_at_T_end.direction = -1

    sol = solve_ivp(
        fun=lambda N, y: isotropic_decoupling_rhs(N, y, grid=grid),
        t_span=(0.0, float(config.max_efolds)),
        y0=y0,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        jac_sparsity=_decoupling_jac_sparsity(grid),
        events=stop_at_T_end,
    )

    target_reached = bool(getattr(sol, "t_events", None)) and len(sol.t_events[0]) > 0
    diag = classify_solve_ivp_result(sol, target_reached=target_reached, phase="isotropic_decoupling")

    state_history = np.asarray(sol.y.T, dtype=np.float64)
    rhs_history = np.asarray(
        [isotropic_decoupling_rhs(float(N_i), state_i, grid=grid) for N_i, state_i in zip(sol.t, state_history)],
        dtype=np.float64,
    )
    obs_history = [
        isotropic_decoupling_observables(float(N_i), state_i, grid=grid)
        for N_i, state_i in zip(sol.t, state_history)
    ]

    N_hist = np.asarray(sol.t, dtype=np.float64)
    T_gamma_hist = np.asarray(sol.y[layout.i_Tg], dtype=np.float64)
    T_nu_e_hist = np.asarray([row["T_nu_e"] for row in obs_history], dtype=np.float64)
    T_nu_x_hist = np.asarray([row["T_nu_x"] for row in obs_history], dtype=np.float64)
    N_eff_hist = np.asarray([row["N_eff"] for row in obs_history], dtype=np.float64)

    if N_hist.size >= 2:
        dT_nu_e_hist = np.gradient(T_nu_e_hist, N_hist, edge_order=1)
        dT_nu_x_hist = np.gradient(T_nu_x_hist, N_hist, edge_order=1)
    else:
        dT_nu_e_hist = np.zeros_like(T_nu_e_hist)
        dT_nu_x_hist = np.zeros_like(T_nu_x_hist)

    y_final = np.asarray(sol.y[:, -1], dtype=np.float64)
    obs = isotropic_decoupling_observables(float(sol.t[-1]), y_final, grid=grid)
    return IsotropicDecouplingResult(
        success=bool(target_reached and sol.success),
        solver_diagnostics=diag,
        T_gamma_final=float(obs["T_gamma"]),
        T_nu_e_final=float(obs["T_nu_e"]),
        T_nu_x_final=float(obs["T_nu_x"]),
        N_eff_final=float(obs["N_eff"]),
        grid=grid,
        y_final=y_final,
        f_nue_final=np.asarray(y_final[layout.i_nue], dtype=np.float64),
        f_nuebar_final=np.asarray(y_final[layout.i_nuebar], dtype=np.float64),
        f_nux_final=np.asarray(y_final[layout.i_nux], dtype=np.float64),
        trajectory={
            "N": N_hist,
            "state": state_history,
            "T_gamma": T_gamma_hist,
            "T_nu_e": T_nu_e_hist,
            "T_nu_x": T_nu_x_hist,
            "N_eff": N_eff_hist,
            "dT_gamma_dN": np.asarray(rhs_history[:, layout.i_Tg], dtype=np.float64),
            "dT_nu_e_dN": np.asarray(dT_nu_e_hist, dtype=np.float64),
            "dT_nu_x_dN": np.asarray(dT_nu_x_hist, dtype=np.float64),
            "f_nue": np.asarray(state_history[:, layout.i_nue], dtype=np.float64),
            "f_nuebar": np.asarray(state_history[:, layout.i_nuebar], dtype=np.float64),
            "f_nux": np.asarray(state_history[:, layout.i_nux], dtype=np.float64),
        },
    )
