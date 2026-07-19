from __future__ import annotations

import math
from typing import Any

import numpy as np

from rabbit.thermo.incomplete_decoupling import T_nu_from_T_gamma_tier1
from rabbit.transport.even_ladder_analysis import integrate_even_ladder_final_state
from rabbit.transport.stageAB_analysis import fit_power_law
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.weak.live_rates import compute_live_weak_rates


def q_weighted_integral(values: np.ndarray, q: np.ndarray, power: int) -> float:
    return float(np.trapezoid((q ** power) * values, q))


def energy_density_ratio_from_state(state: AxisymmetricHierarchyState, *, species_idx: int = 0) -> float:
    q = state.grid.nodes
    f0 = state.moment(species_idx, 0)
    fd = fermi_dirac(q)
    num = q_weighted_integral(f0, q, power=3)
    den = q_weighted_integral(fd, q, power=3)
    if den <= 0.0:
        return math.nan
    return num / den


def observable_snapshot_from_state(
    state: AxisymmetricHierarchyState,
    T_gamma_MeV: float,
    *,
    species_idx: int = 0,
    N_eff_baseline: float = 3.044,
    correction_level: int = 0,
) -> dict[str, float]:
    q = state.grid.nodes
    f0 = np.asarray(state.moment(species_idx, 0), dtype=float)
    T_nu = T_nu_from_T_gamma_tier1(T_gamma_MeV)
    weak = compute_live_weak_rates(
        f_nue_monopole=f0,
        f_nuebar_monopole=f0,
        q_nodes=q,
        T_gamma_MeV=T_gamma_MeV,
        T_nu_MeV=T_nu,
        compute_iso_reference=True,
        correction_level=correction_level,
        use_exact_fd=False,
    )
    rho_ratio = energy_density_ratio_from_state(state, species_idx=species_idx)
    n_eff_like = N_eff_baseline * rho_ratio
    xn_eq = weak.equilibrium_Xn
    yp_proxy = 2.0 * xn_eq
    return {
        'T_gamma_MeV': float(T_gamma_MeV),
        'T_nu_MeV': float(T_nu),
        'rho_ratio': float(rho_ratio),
        'N_eff_like': float(n_eff_like),
        'lambda_np': float(weak.lambda_np),
        'lambda_pn': float(weak.lambda_pn),
        'delta_np': float(weak.delta_np),
        'delta_pn': float(weak.delta_pn),
        'Xn_eq_proxy': float(xn_eq),
        'Yp_proxy': float(yp_proxy),
    }


def compare_snapshot_to_reference(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ('rho_ratio', 'N_eff_like', 'lambda_np', 'lambda_pn', 'Xn_eq_proxy', 'Yp_proxy'):
        out[f'{key}_diff_to_ref_abs'] = abs(float(row[key]) - float(ref[key]))
    return out


def fit_small_shear_observable_diffs(
    rows: list[dict[str, Any]],
    *,
    n_q: int,
    T_gamma_MeV: float,
    lmax: int,
    sigma_max: float = 0.1,
) -> list[dict[str, float | int | str]]:
    subset = [
        r for r in rows
        if r['N_q'] == n_q and abs(float(r['T_gamma_MeV']) - T_gamma_MeV) < 1e-12 and r['lmax'] == lmax and 0.0 < r['Sigma_H'] <= sigma_max
    ]
    subset = sorted(subset, key=lambda r: r['Sigma_H'])
    sigmas = [float(r['Sigma_H']) for r in subset]
    metrics = [
        'N_eff_like_diff_to_ref_abs',
        'lambda_np_diff_to_ref_abs',
        'lambda_pn_diff_to_ref_abs',
        'Xn_eq_proxy_diff_to_ref_abs',
        'Yp_proxy_diff_to_ref_abs',
    ]
    out: list[dict[str, float | int | str]] = []
    for metric in metrics:
        fit = fit_power_law(sigmas, [float(r.get(metric, math.nan)) for r in subset])
        out.append({'N_q': n_q, 'T_gamma_MeV': T_gamma_MeV, 'lmax': lmax, 'metric': metric, **fit})
    return out


def build_snapshot_row(
    *,
    lmax: int,
    sigma_h: float,
    n_q: int,
    T_gamma_MeV: float,
    n_end: float = 0.5,
) -> dict[str, float | int]:
    state = integrate_even_ladder_final_state(lmax=lmax, sigma_h=sigma_h, n_q=n_q, n_end=n_end)
    snap = observable_snapshot_from_state(state, T_gamma_MeV=T_gamma_MeV)
    return {
        'lmax': int(lmax),
        'Sigma_H': float(sigma_h),
        'N_q': int(n_q),
        **snap,
    }
