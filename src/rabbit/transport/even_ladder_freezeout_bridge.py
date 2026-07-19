from __future__ import annotations

import math
from typing import Any

import numpy as np

from rabbit.network.abundances_standard import abundance_rhs_phase1
from rabbit.thermo.incomplete_decoupling import T_nu_from_T_gamma_tier1
from rabbit.transport.even_ladder_analysis import integrate_even_ladder_final_state
from rabbit.transport.even_ladder_observable_bridge import observable_snapshot_from_state
from rabbit.transport.stageAB_analysis import fit_power_law
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState

_G_N = 6.70883e-45
_MEV_TO_S = 1.519267447e21
_PI = math.pi


def hubble_radiation_simple(T_gamma_MeV: float, T_nu_MeV: float, *, rho_ratio: float = 1.0, N_eff_baseline: float = 3.044) -> float:
    """Simple radiation Hubble rate [s^-1] for the phase-1 freeze-out bridge.

    This is a handoff-level bridge, not the full coupled thermodynamic core.
    The neutrino energy density is dressed by ``rho_ratio`` extracted from the
    ladder state.
    """
    rho_g = (_PI**2 / 15.0) * T_gamma_MeV**4
    rho_nu = N_eff_baseline * float(rho_ratio) * (7.0 / 8.0) * (_PI**2 / 15.0) * T_nu_MeV**4
    H_MeV = math.sqrt(max((8.0 * _PI * _G_N / 3.0) * (rho_g + rho_nu), 0.0))
    return H_MeV * _MEV_TO_S


def build_freezeout_profile_from_state(
    state: AxisymmetricHierarchyState,
    *,
    T_start_MeV: float = 2.0,
    T_handoff_MeV: float = 0.08,
    n_T: int = 96,
) -> dict[str, np.ndarray]:
    if T_start_MeV <= T_handoff_MeV:
        raise ValueError("T_start_MeV must exceed T_handoff_MeV")
    T_grid = np.geomspace(T_start_MeV, T_handoff_MeV, n_T)
    lambda_np = np.zeros_like(T_grid)
    lambda_pn = np.zeros_like(T_grid)
    rho_ratio = np.zeros_like(T_grid)
    n_eff_like = np.zeros_like(T_grid)
    T_nu = np.zeros_like(T_grid)
    for i, Tg in enumerate(T_grid):
        snap = observable_snapshot_from_state(state, float(Tg))
        lambda_np[i] = float(snap['lambda_np'])
        lambda_pn[i] = float(snap['lambda_pn'])
        rho_ratio[i] = float(snap['rho_ratio'])
        n_eff_like[i] = float(snap['N_eff_like'])
        T_nu[i] = float(snap['T_nu_MeV'])
    H = np.array([
        hubble_radiation_simple(float(Tg), float(Tnu), rho_ratio=float(rr))
        for Tg, Tnu, rr in zip(T_grid, T_nu, rho_ratio)
    ], dtype=float)
    return {
        'T_gamma_MeV': T_grid,
        'T_nu_MeV': T_nu,
        'lambda_np': lambda_np,
        'lambda_pn': lambda_pn,
        'rho_ratio': rho_ratio,
        'N_eff_like': n_eff_like,
        'H_s': H,
    }


def _interp_descending(x_desc: np.ndarray, y_desc: np.ndarray, x_eval: float) -> float:
    x_asc = x_desc[::-1]
    y_asc = y_desc[::-1]
    return float(np.interp(x_eval, x_asc, y_asc))


def integrate_phase1_freezeout_profile(profile: dict[str, np.ndarray], *, Xn_init: float | None = None) -> dict[str, float]:
    T = np.asarray(profile['T_gamma_MeV'], dtype=float)
    lam_np = np.asarray(profile['lambda_np'], dtype=float)
    lam_pn = np.asarray(profile['lambda_pn'], dtype=float)
    H = np.asarray(profile['H_s'], dtype=float)
    if Xn_init is None:
        total0 = max(float(lam_np[0] + lam_pn[0]), 1.0e-100)
        Xn = float(lam_pn[0] / total0)
    else:
        Xn = float(Xn_init)
    x_grid = np.log(T)

    def dXdx(x: float, xn: float) -> float:
        lnp = _interp_descending(x_grid, lam_np, x)
        lpn = _interp_descending(x_grid, lam_pn, x)
        h = max(_interp_descending(x_grid, H, x), 1.0e-100)
        return -abundance_rhs_phase1(xn, lnp, lpn) / h

    for i in range(len(x_grid) - 1):
        x0 = float(x_grid[i])
        x1 = float(x_grid[i + 1])
        dx = x1 - x0
        k1 = dXdx(x0, Xn)
        k2 = dXdx(x0 + 0.5 * dx, Xn + 0.5 * dx * k1)
        k3 = dXdx(x0 + 0.5 * dx, Xn + 0.5 * dx * k2)
        k4 = dXdx(x1, Xn + dx * k3)
        Xn = float(np.clip(Xn + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0))

    total_f = max(float(lam_np[-1] + lam_pn[-1]), 1.0e-100)
    Xn_eq_handoff = float(lam_pn[-1] / total_f)
    return {
        'Xn_init': float(Xn_init if Xn_init is not None else lam_pn[0] / max(lam_np[0] + lam_pn[0], 1.0e-100)),
        'Xn_handoff': float(Xn),
        'Xn_eq_handoff': float(Xn_eq_handoff),
        'Yp_handoff_proxy': float(2.0 * Xn),
        'T_start_MeV': float(T[0]),
        'T_handoff_MeV': float(T[-1]),
    }


def neutron_decay_interval_seconds(profile: dict[str, np.ndarray], *, T_final_MeV: float = 0.06) -> float:
    T_h = float(profile['T_gamma_MeV'][-1])
    if T_final_MeV >= T_h:
        return 0.0
    T_ext = np.geomspace(T_h, T_final_MeV, 48)
    H_ext = np.array([
        hubble_radiation_simple(float(Tg), float(T_nu_from_T_gamma_tier1(Tg)), rho_ratio=float(profile['rho_ratio'][-1]))
        for Tg in T_ext
    ], dtype=float)
    x = np.log(T_ext)
    # dt = -d ln T / H
    dt = np.trapezoid(-1.0 / np.maximum(H_ext, 1.0e-100), x)
    return float(max(dt, 0.0))


def apply_neutron_decay_proxy(Xn_handoff: float, *, delta_t_s: float, tau_n_s: float = 878.4) -> dict[str, float]:
    survival = math.exp(-max(delta_t_s, 0.0) / max(tau_n_s, 1.0e-100))
    xn_decayed = float(np.clip(Xn_handoff * survival, 0.0, 1.0))
    return {
        'decay_interval_s': float(delta_t_s),
        'neutron_survival_factor': float(survival),
        'Xn_decay_proxy': float(xn_decayed),
        'Yp_decay_proxy': float(2.0 * xn_decayed),
    }


def freezeout_bridge_row(
    *,
    lmax: int,
    sigma_h: float,
    n_q: int,
    n_end: float = 0.5,
    T_start_MeV: float = 2.0,
    T_handoff_MeV: float = 0.08,
    T_decay_final_MeV: float = 0.06,
    n_T: int = 96,
) -> dict[str, float | int]:
    state = integrate_even_ladder_final_state(lmax=lmax, sigma_h=sigma_h, n_q=n_q, n_end=n_end)
    profile = build_freezeout_profile_from_state(state, T_start_MeV=T_start_MeV, T_handoff_MeV=T_handoff_MeV, n_T=n_T)
    p1 = integrate_phase1_freezeout_profile(profile)
    decay = apply_neutron_decay_proxy(p1['Xn_handoff'], delta_t_s=neutron_decay_interval_seconds(profile, T_final_MeV=T_decay_final_MeV))
    T_nu_h = float(profile['T_nu_MeV'][-1])
    lambda_np_h = float(profile['lambda_np'][-1])
    lambda_pn_h = float(profile['lambda_pn'][-1])
    Xn_h = float(p1['Xn_handoff'])
    return {
        'lmax': int(lmax),
        'Sigma_H': float(sigma_h),
        'N_q': int(n_q),
        'transport_N_sample': float(n_end),
        'T_start_MeV': float(T_start_MeV),
        'T_handoff_MeV': float(T_handoff_MeV),
        'T_decay_final_MeV': float(T_decay_final_MeV),
        'T_nu_handoff_MeV': T_nu_h,
        'rho_ratio_handoff': float(profile['rho_ratio'][-1]),
        'N_eff_like_handoff': float(profile['N_eff_like'][-1]),
        'lambda_np_handoff': lambda_np_h,
        'lambda_pn_handoff': lambda_pn_h,
        # schema-aligned aliases for downstream honesty-preserving handoff reporting
        'phase1_handoff_T': float(T_handoff_MeV),
        'phase1_handoff_T_gamma': float(T_handoff_MeV),
        'phase1_handoff_T_nu_e': T_nu_h,
        'phase1_handoff_T_nu_x': T_nu_h,
        'phase1_handoff_lambda_np': lambda_np_h,
        'phase1_handoff_lambda_pn': lambda_pn_h,
        'phase1_handoff_Xn': Xn_h,
        'Xn_freeze': Xn_h,
        **p1,
        **decay,
    }


def handoff_metadata_from_freezeout_row(
    row: dict[str, Any],
    *,
    correction_level: int = 0,
    weak_mode: str | None = None,
) -> dict[str, Any]:
    """Return an honesty-preserving handoff metadata dict aligned with driver schema names.

    This is a bridge/proxy metadata record, not a promoted production solver result.
    Keys intentionally mirror the live driver handoff names where the underlying
    quantity is genuinely computed, and keep the word ``proxy`` where the bridge
    stops short of a full phase-2 network solve.
    """
    weak_mode = weak_mode or f'live_monopole_freezeout_bridge_cl{int(correction_level)}'
    return {
        'selection_contract': 'finite_even_ladder_freezeout_bridge_proxy_v1',
        'backend': 'finite_even_ladder_axisymmetric_typeI',
        'weak_mode': weak_mode,
        'transport_model': 'finite_even_ladder',
        'transport_N_sample': float(row['transport_N_sample']),
        'lmax': int(row['lmax']),
        'Sigma_H': float(row['Sigma_H']),
        'N_q': int(row['N_q']),
        'phase1_handoff_T': float(row['phase1_handoff_T']),
        'phase1_handoff_T_gamma': float(row['phase1_handoff_T_gamma']),
        'phase1_handoff_T_nu_e': float(row['phase1_handoff_T_nu_e']),
        'phase1_handoff_T_nu_x': float(row['phase1_handoff_T_nu_x']),
        'phase1_handoff_lambda_np': float(row['phase1_handoff_lambda_np']),
        'phase1_handoff_lambda_pn': float(row['phase1_handoff_lambda_pn']),
        'phase1_handoff_Xn': float(row['phase1_handoff_Xn']),
        'Xn_freeze': float(row['Xn_freeze']),
        'N_eff_handoff_proxy': float(row['N_eff_like_handoff']),
        'Yp_handoff_proxy': float(row['Yp_handoff_proxy']),
        'Yp_decay_proxy': float(row['Yp_decay_proxy']),
        'honesty_note': ('Bridge/proxy metadata only: phase-1 freeze-out is integrated on a handoff profile built from the finite even-ell ladder, '
                         'while phase-2 nuclear burning is represented only by a neutron-decay proxy.'),
        'deferred_features': ['full_phase2_network', 'D/H', 'Li7H', 'Li6H', 'exact_driver_promotion'],
    }


METRICS_TO_REF = (
    'rho_ratio_handoff',
    'N_eff_like_handoff',
    'lambda_np_handoff',
    'lambda_pn_handoff',
    'Xn_handoff',
    'Yp_handoff_proxy',
    'Xn_decay_proxy',
    'Yp_decay_proxy',
)


def compare_freezeout_row_to_reference(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in METRICS_TO_REF:
        out[f'{key}_diff_to_ref_abs'] = abs(float(row[key]) - float(ref[key]))
    return out


def fit_small_shear_freezeout_diffs(
    rows: list[dict[str, Any]],
    *,
    n_q: int,
    lmax: int,
    sigma_max: float = 0.1,
) -> list[dict[str, float | int | str]]:
    subset = [r for r in rows if r['N_q'] == n_q and r['lmax'] == lmax and 0.0 < r['Sigma_H'] <= sigma_max]
    subset = sorted(subset, key=lambda r: r['Sigma_H'])
    sigmas = [float(r['Sigma_H']) for r in subset]
    out: list[dict[str, float | int | str]] = []
    for metric in [f'{m}_diff_to_ref_abs' for m in METRICS_TO_REF]:
        fit = fit_power_law(sigmas, [float(r.get(metric, math.nan)) for r in subset])
        out.append({'N_q': n_q, 'lmax': lmax, 'metric': metric, **fit})
    return out
