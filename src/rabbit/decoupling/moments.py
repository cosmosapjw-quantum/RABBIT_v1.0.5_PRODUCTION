from __future__ import annotations

import numpy as np

from rabbit.decoupling.grid import DecouplingGrid


_M_ELECTRON_MEV = 0.5109989500
_TWO_PI2 = 2.0 * np.pi**2
_FD_CLIP = 500.0


def fermi_dirac_comoving(y_nodes: np.ndarray, z_scale: float) -> np.ndarray:
    """FD distribution on the comoving y = p a / m_e grid."""
    z = max(float(z_scale), 1.0e-30)
    y = np.asarray(y_nodes, dtype=np.float64)
    return 1.0 / (np.exp(np.minimum(y / z, _FD_CLIP)) + 1.0)


def _fd_state_energy_moment_ref(grid: DecouplingGrid) -> float:
    fd_ref = fermi_dirac_comoving(grid.nodes, 1.0)
    return grid.integrate(fd_ref, power=3)


def state_comoving_temperature_scale(f_state: np.ndarray, grid: DecouplingGrid) -> float:
    """Infer z = T a / m_e from the fourth-root energy moment."""
    m3 = max(grid.integrate(np.asarray(f_state, dtype=np.float64), power=3), 0.0)
    ref = _fd_state_energy_moment_ref(grid)
    if ref <= 1.0e-300:
        return 0.0
    return float((m3 / ref) ** 0.25)


def state_energy_density_per_efold(df_dN: np.ndarray, grid: DecouplingGrid, a_scale: float) -> float:
    """Collision-sourced energy gain per e-fold for one neutrino state [MeV^4]."""
    a = max(float(a_scale), 1.0e-30)
    pref = (_M_ELECTRON_MEV**4) / (_TWO_PI2 * a**4)
    return float(pref * grid.integrate(np.asarray(df_dN, dtype=np.float64), power=3))


def effective_nue_pair_temperature(
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    grid: DecouplingGrid,
    a_scale: float,
) -> float:
    ref = _fd_state_energy_moment_ref(grid)
    if ref <= 1.0e-300:
        return 0.0
    m3_pair = max(
        grid.integrate(np.asarray(f_nue, dtype=np.float64), power=3)
        + grid.integrate(np.asarray(f_nuebar, dtype=np.float64), power=3),
        0.0,
    )
    z_pair = (m3_pair / (2.0 * ref)) ** 0.25
    return float(_M_ELECTRON_MEV * z_pair / max(float(a_scale), 1.0e-30))


def effective_nux_temperature(
    f_nux: np.ndarray,
    grid: DecouplingGrid,
    a_scale: float,
) -> float:
    z_nux = state_comoving_temperature_scale(f_nux, grid)
    return float(_M_ELECTRON_MEV * z_nux / max(float(a_scale), 1.0e-30))


def effective_neff_from_distributions(
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    f_nux: np.ndarray,
    grid: DecouplingGrid,
    *,
    a_scale: float,
    T_gamma: float,
) -> float:
    T_std = float(T_gamma) * (4.0 / 11.0) ** (1.0 / 3.0)
    if T_std < 1.0e-30:
        return 3.044
    T_nue_pair = effective_nue_pair_temperature(f_nue, f_nuebar, grid, a_scale)
    T_nux = effective_nux_temperature(f_nux, grid, a_scale)
    return float((T_nue_pair / T_std) ** 4 + 2.0 * (T_nux / T_std) ** 4)
