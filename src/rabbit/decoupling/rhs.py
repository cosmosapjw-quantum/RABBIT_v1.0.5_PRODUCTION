from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rabbit.decoupling.collisions_diagonal import evaluate_diagonal_collision_rhs
from rabbit.decoupling.grid import DecouplingGrid
from rabbit.decoupling.moments import (
    effective_neff_from_distributions,
    effective_nue_pair_temperature,
    effective_nux_temperature,
    fermi_dirac_comoving,
)
from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier2
from rabbit.thermo.nudec_coupled import hubble_3T


@dataclass(frozen=True)
class DecouplingLayout:
    i_Tg: int
    i_nue: slice
    i_nuebar: slice
    i_nux: slice
    n_total: int


def decoupling_layout(grid: DecouplingGrid) -> DecouplingLayout:
    p = 0
    i_Tg = p
    p += 1
    i_nue = slice(p, p + grid.n_y)
    p += grid.n_y
    i_nuebar = slice(p, p + grid.n_y)
    p += grid.n_y
    i_nux = slice(p, p + grid.n_y)
    p += grid.n_y
    return DecouplingLayout(i_Tg=i_Tg, i_nue=i_nue, i_nuebar=i_nuebar, i_nux=i_nux, n_total=p)


def initial_decoupling_state(grid: DecouplingGrid, *, T_gamma_init: float, a_init: float = 1.0) -> np.ndarray:
    layout = decoupling_layout(grid)
    y0 = np.zeros(layout.n_total, dtype=np.float64)
    y0[layout.i_Tg] = float(T_gamma_init)
    z_gamma = float(T_gamma_init * a_init / 0.5109989500)
    f_eq = fermi_dirac_comoving(grid.nodes, z_gamma)
    y0[layout.i_nue] = f_eq
    y0[layout.i_nuebar] = f_eq
    y0[layout.i_nux] = f_eq
    return y0


def isotropic_decoupling_rhs(
    N: float,
    y: np.ndarray,
    *,
    grid: DecouplingGrid,
) -> np.ndarray:
    layout = decoupling_layout(grid)
    T_gamma = float(y[layout.i_Tg])
    if T_gamma < 1.0e-6:
        return np.zeros_like(y)

    a_scale = float(np.exp(N))
    f_nue = np.clip(np.asarray(y[layout.i_nue], dtype=np.float64), 0.0, 1.0)
    f_nuebar = np.clip(np.asarray(y[layout.i_nuebar], dtype=np.float64), 0.0, 1.0)
    f_nux = np.clip(np.asarray(y[layout.i_nux], dtype=np.float64), 0.0, 1.0)

    T_nu_e = effective_nue_pair_temperature(f_nue, f_nuebar, grid, a_scale)
    T_nu_x = effective_nux_temperature(f_nux, grid, a_scale)
    H_MeV = float(hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=0.0))

    coll_nue = evaluate_diagonal_collision_rhs(
        species="nue",
        f_state=f_nue,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_MeV=H_MeV,
        a_scale=a_scale,
    )
    coll_nuebar = evaluate_diagonal_collision_rhs(
        species="nuebar",
        f_state=f_nuebar,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_MeV=H_MeV,
        a_scale=a_scale,
    )
    coll_nux = evaluate_diagonal_collision_rhs(
        species="nux",
        f_state=f_nux,
        grid=grid,
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_MeV=H_MeV,
        a_scale=a_scale,
    )

    dQ_nu_gain = (
        coll_nue.energy_gain_per_efold
        + coll_nuebar.energy_gain_per_efold
        + 4.0 * coll_nux.energy_gain_per_efold
    )
    dT_gamma = dT_gamma_dN_tier2(T_gamma, dQ_nu_to_plasma_per_dN=-dQ_nu_gain)

    dy = np.zeros_like(y)
    dy[layout.i_Tg] = dT_gamma
    dy[layout.i_nue] = coll_nue.df_dN
    dy[layout.i_nuebar] = coll_nuebar.df_dN
    dy[layout.i_nux] = coll_nux.df_dN
    return dy


def isotropic_decoupling_observables(N: float, y: np.ndarray, *, grid: DecouplingGrid) -> dict[str, float]:
    layout = decoupling_layout(grid)
    a_scale = float(np.exp(N))
    f_nue = np.asarray(y[layout.i_nue], dtype=np.float64)
    f_nuebar = np.asarray(y[layout.i_nuebar], dtype=np.float64)
    f_nux = np.asarray(y[layout.i_nux], dtype=np.float64)
    T_gamma = float(y[layout.i_Tg])
    T_nu_e = effective_nue_pair_temperature(f_nue, f_nuebar, grid, a_scale)
    T_nu_x = effective_nux_temperature(f_nux, grid, a_scale)
    N_eff = effective_neff_from_distributions(
        f_nue,
        f_nuebar,
        f_nux,
        grid,
        a_scale=a_scale,
        T_gamma=T_gamma,
    )
    return {
        "T_gamma": T_gamma,
        "T_nu_e": T_nu_e,
        "T_nu_x": T_nu_x,
        "N_eff": float(N_eff),
    }
