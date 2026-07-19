"""Reference weak-rate quadrature checks.

This module intentionally does not replace the production
Gauss-Laguerre/Gauss-Legendre path.  It provides a dense fixed-grid
trapezoid evaluator, similar in spirit to LINX's JAX weak-rate
reference style, so production quadrature drift can be measured without
changing the forward solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.laguerre import laggauss

from rabbit.weak.channels import (
    _M_ELECTRON_MEV,
    _Q_DIMLESS,
    fermi_dirac,
    get_I0_born,
)
from rabbit.weak.corrections import (
    compute_I0_cl3_corrected,
    compute_I0_corrected,
    coulomb_correction_per_channel,
    radiative_correction_factor,
)
from rabbit.weak.finite_mass import finite_mass_scalar_correction
from rabbit.weak.live_rates import (
    LiveWeakResult,
    _build_distribution_fn,
    compute_live_weak_rates,
)
from rabbit.weak.quadrature import DEFAULT_WEAK_QUADRATURE, HIGHRES_WEAK_QUADRATURE


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclass(frozen=True)
class TrapezoidWeakReferenceConfig:
    """Dense fixed-grid reference settings for live weak rates."""

    n_semi_infinite: int = 1000
    n_bounded: int = 1000
    tail_width: float = 30.0
    min_tail: float = 7.0

    def __post_init__(self) -> None:
        if int(self.n_semi_infinite) < 32:
            raise ValueError("n_semi_infinite must be >= 32 for reference quadrature.")
        if int(self.n_bounded) < 32:
            raise ValueError("n_bounded must be >= 32 for reference quadrature.")
        if float(self.tail_width) <= 0.0:
            raise ValueError("tail_width must be positive.")
        if float(self.min_tail) <= 0.0:
            raise ValueError("min_tail must be positive.")


DEFAULT_TRAPEZOID_WEAK_REFERENCE = TrapezoidWeakReferenceConfig()


def _semi_grid(start: float, thermal_scale: float, cfg: TrapezoidWeakReferenceConfig) -> np.ndarray:
    span = max(float(cfg.min_tail), float(cfg.tail_width) * max(float(thermal_scale), 1.0e-30))
    return np.linspace(float(start), float(start) + span, int(cfg.n_semi_infinite))


def _bounded_grid(lo: float, hi: float, cfg: TrapezoidWeakReferenceConfig) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(cfg.n_bounded))


def _fd_e(eps_e: np.ndarray, T_e: float) -> np.ndarray:
    return fermi_dirac(np.asarray(eps_e, dtype=float), float(T_e))


def _correction_factor(channel: str, eps_e: np.ndarray, eps_nu: np.ndarray, level: int) -> np.ndarray:
    factor = np.ones_like(np.asarray(eps_e, dtype=float))
    if int(level) > 0:
        factor *= coulomb_correction_per_channel(eps_e, channel)
    # Match the production live-rate kernel: Sirlin/radiative corrections are
    # carried on the bounded neutron-decay channels, while CL3 adds the scalar
    # finite-mass/recoil/weak-magnetism factor channel-by-channel.
    if int(level) >= 2 and channel in {"c", "f"}:
        factor *= radiative_correction_factor(eps_e)
    if int(level) >= 3:
        factor *= finite_mass_scalar_correction(eps_e, eps_nu, channel)
    return factor


def _normalization_I0(level: int, tau_n: float) -> float:
    del tau_n
    if int(level) >= 3:
        return compute_I0_cl3_corrected()
    if int(level) > 0:
        return compute_I0_corrected(
            enable_coulomb=int(level) >= 1,
            enable_radiative=int(level) >= 2,
        )
    return get_I0_born()


def compute_live_weak_rates_trapezoid_reference(
    f_nue_monopole: np.ndarray,
    f_nuebar_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_gamma_MeV: float,
    T_nu_MeV: float | None = None,
    tau_n: float = 878.4,
    *,
    correction_level: int = 0,
    config: TrapezoidWeakReferenceConfig | None = None,
    compute_iso_reference: bool = False,
    use_exact_fd: bool = False,
) -> LiveWeakResult:
    """Compute live weak rates with dense trapezoid quadrature.

    This is a validation/reference path.  It uses the same distribution
    interpolation and correction hierarchy as ``compute_live_weak_rates``,
    but replaces the production Gauss rules with fixed linear grids and
    trapezoid integration over each weak channel.
    """
    if int(correction_level) not in (0, 1, 2, 3):
        raise ValueError(
            f"Unsupported correction_level={correction_level}; expected 0, 1, 2, or 3."
        )
    cfg = DEFAULT_TRAPEZOID_WEAK_REFERENCE if config is None else config
    if T_gamma_MeV < 1.0e-4:
        return LiveWeakResult(lambda_np=1.0 / tau_n, lambda_pn=0.0)
    if T_nu_MeV is None:
        T_nu_MeV = T_gamma_MeV * (4.0 / 11.0) ** (1.0 / 3.0)

    T_e = float(T_gamma_MeV) / _M_ELECTRON_MEV
    T_nu = float(T_nu_MeV) / _M_ELECTRON_MEV
    q = _Q_DIMLESS
    I0 = _normalization_I0(int(correction_level), float(tau_n))

    f_nue = _build_distribution_fn(
        np.asarray(f_nue_monopole, dtype=float),
        np.asarray(q_nodes, dtype=float),
        T_nu,
        use_exact_fd=use_exact_fd,
    )
    f_nuebar = _build_distribution_fn(
        np.asarray(f_nuebar_monopole, dtype=float),
        np.asarray(q_nodes, dtype=float),
        T_nu,
        use_exact_fd=use_exact_fd,
    )

    def _evaluate(fn_nue, fn_nuebar):
        eps_nu_a = _semi_grid(0.0, T_nu, cfg)
        eps_e_a = eps_nu_a + q
        ps_a = eps_nu_a**2 * eps_e_a * np.sqrt(np.maximum(eps_e_a**2 - 1.0, 0.0))
        I_a = float(
            _trapz(
                ps_a
                * fn_nue(eps_nu_a)
                * (1.0 - _fd_e(eps_e_a, T_e))
                * _correction_factor("a", eps_e_a, eps_nu_a, correction_level),
                eps_nu_a,
            )
        )

        eps_e_b = _semi_grid(1.0, T_e, cfg)
        eps_nu_b = eps_e_b + q
        ps_b = eps_nu_b**2 * eps_e_b * np.sqrt(np.maximum(eps_e_b**2 - 1.0, 0.0))
        I_b = float(
            _trapz(
                ps_b
                * _fd_e(eps_e_b, T_e)
                * (1.0 - fn_nuebar(eps_nu_b))
                * _correction_factor("b", eps_e_b, eps_nu_b, correction_level),
                eps_e_b,
            )
        )

        eps_e_cf = _bounded_grid(1.0, q, cfg)
        eps_nu_cf = q - eps_e_cf
        ps_cf = eps_nu_cf**2 * eps_e_cf * np.sqrt(np.maximum(eps_e_cf**2 - 1.0, 0.0))
        f_nubar_cf = fn_nuebar(eps_nu_cf)
        fd_cf = _fd_e(eps_e_cf, T_e)
        I_c = float(
            _trapz(
                ps_cf
                * (1.0 - fd_cf)
                * (1.0 - f_nubar_cf)
                * _correction_factor("c", eps_e_cf, eps_nu_cf, correction_level),
                eps_e_cf,
            )
        )
        I_f = float(
            _trapz(
                ps_cf
                * fd_cf
                * f_nubar_cf
                * _correction_factor("f", eps_e_cf, eps_nu_cf, correction_level),
                eps_e_cf,
            )
        )

        eps_e_d = _semi_grid(q, T_e, cfg)
        eps_nu_d = eps_e_d - q
        ps_d = eps_nu_d**2 * eps_e_d * np.sqrt(np.maximum(eps_e_d**2 - 1.0, 0.0))
        I_d = float(
            _trapz(
                ps_d
                * _fd_e(eps_e_d, T_e)
                * (1.0 - fn_nue(eps_nu_d))
                * _correction_factor("d", eps_e_d, eps_nu_d, correction_level),
                eps_e_d,
            )
        )

        eps_nu_e = _semi_grid(q + 1.0, T_nu, cfg)
        eps_e_e = eps_nu_e - q
        ps_e = eps_nu_e**2 * eps_e_e * np.sqrt(np.maximum(eps_e_e**2 - 1.0, 0.0))
        I_e = float(
            _trapz(
                ps_e
                * fn_nuebar(eps_nu_e)
                * (1.0 - _fd_e(eps_e_e, T_e))
                * _correction_factor("e", eps_e_e, eps_nu_e, correction_level),
                eps_nu_e,
            )
        )
        return I_a, I_b, I_c, I_d, I_e, I_f

    I_a, I_b, I_c, I_d, I_e, I_f = _evaluate(f_nue, f_nuebar)
    I_np = I_a + I_b + I_c
    I_pn = I_d + I_e + I_f
    lambda_np = max(I_np / (I0 * tau_n), 1.0 / tau_n)
    lambda_pn = max(I_pn / (I0 * tau_n), 0.0)

    lambda_np_iso = 0.0
    lambda_pn_iso = 0.0
    delta_np = 0.0
    delta_pn = 0.0
    if compute_iso_reference:
        def fd_nu(eps):
            return fermi_dirac(eps, T_nu)

        I_a_i, I_b_i, I_c_i, I_d_i, I_e_i, I_f_i = _evaluate(fd_nu, fd_nu)
        lambda_np_iso = max((I_a_i + I_b_i + I_c_i) / (I0 * tau_n), 1.0 / tau_n)
        lambda_pn_iso = max((I_d_i + I_e_i + I_f_i) / (I0 * tau_n), 0.0)
        if lambda_np_iso > 1.0e-30:
            delta_np = (lambda_np - lambda_np_iso) / lambda_np_iso
        if lambda_pn_iso > 1.0e-30:
            delta_pn = (lambda_pn - lambda_pn_iso) / lambda_pn_iso

    return LiveWeakResult(
        lambda_np=lambda_np,
        lambda_pn=lambda_pn,
        lambda_np_iso=lambda_np_iso,
        lambda_pn_iso=lambda_pn_iso,
        delta_np=delta_np,
        delta_pn=delta_pn,
        channels_np=(I_a, I_b, I_c),
        channels_pn=(I_d, I_e, I_f),
    )


def weak_rate_quadrature_drift(
    production: LiveWeakResult,
    reference: LiveWeakResult,
) -> dict[str, float]:
    """Return relative drift diagnostics between two weak-rate evaluations."""
    np_ref = max(abs(float(reference.lambda_np)), 1.0e-300)
    pn_ref = max(abs(float(reference.lambda_pn)), 1.0e-300)
    ratio_ref = float(reference.lambda_np) / max(float(reference.lambda_pn), 1.0e-300)
    ratio_prod = float(production.lambda_np) / max(float(production.lambda_pn), 1.0e-300)
    return {
        "lambda_np_rel": abs(float(production.lambda_np) - float(reference.lambda_np)) / np_ref,
        "lambda_pn_rel": abs(float(production.lambda_pn) - float(reference.lambda_pn)) / pn_ref,
        "lambda_ratio_rel": abs(ratio_prod - ratio_ref) / max(abs(ratio_ref), 1.0e-300),
    }


def scan_fd_weak_quadrature_ladder(
    T_grid_MeV,
    *,
    correction_level: int = 3,
    n_q: int = 20,
    tau_n: float = 878.4,
    reference_config: TrapezoidWeakReferenceConfig | None = None,
) -> list[dict[str, float]]:
    """Scan GL32/GL64/trapz1000 drift on an equilibrium FD weak-rate grid."""
    q_nodes, _weights = laggauss(int(n_q))
    f0 = 1.0 / (np.exp(q_nodes) + 1.0)
    records: list[dict[str, float]] = []
    for T in np.asarray(T_grid_MeV, dtype=float):
        production = compute_live_weak_rates(
            f0,
            f0,
            q_nodes,
            float(T),
            float(T),
            tau_n=float(tau_n),
            quad=DEFAULT_WEAK_QUADRATURE,
            compute_iso_reference=False,
            correction_level=int(correction_level),
            use_exact_fd=True,
        )
        validation = compute_live_weak_rates(
            f0,
            f0,
            q_nodes,
            float(T),
            float(T),
            tau_n=float(tau_n),
            quad=HIGHRES_WEAK_QUADRATURE,
            compute_iso_reference=False,
            correction_level=int(correction_level),
            use_exact_fd=True,
        )
        trapz_reference = compute_live_weak_rates_trapezoid_reference(
            f0,
            f0,
            q_nodes,
            float(T),
            float(T),
            tau_n=float(tau_n),
            correction_level=int(correction_level),
            config=reference_config,
            use_exact_fd=True,
        )
        prod_vs_validation = weak_rate_quadrature_drift(production, validation)
        validation_vs_trapz = weak_rate_quadrature_drift(validation, trapz_reference)
        records.append(
            {
                "T_gamma_MeV": float(T),
                "correction_level": float(correction_level),
                "lambda_np_gl32": float(production.lambda_np),
                "lambda_pn_gl32": float(production.lambda_pn),
                "lambda_np_gl64": float(validation.lambda_np),
                "lambda_pn_gl64": float(validation.lambda_pn),
                "lambda_np_trapz": float(trapz_reference.lambda_np),
                "lambda_pn_trapz": float(trapz_reference.lambda_pn),
                "gl32_vs_gl64_lambda_np_rel": prod_vs_validation["lambda_np_rel"],
                "gl32_vs_gl64_lambda_pn_rel": prod_vs_validation["lambda_pn_rel"],
                "gl64_vs_trapz_lambda_np_rel": validation_vs_trapz["lambda_np_rel"],
                "gl64_vs_trapz_lambda_pn_rel": validation_vs_trapz["lambda_pn_rel"],
                "gl64_vs_trapz_lambda_ratio_rel": validation_vs_trapz["lambda_ratio_rel"],
            }
        )
    return records
