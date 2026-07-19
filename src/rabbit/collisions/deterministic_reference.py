"""Deterministic collision-reference scaffolding for augmented transport."""
from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Mapping

from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
import numpy as np

from rabbit.collisions.kernels import (
    G_F_MEV,
    G_L_NUE,
    G_L_NUX,
    G_R_NUE,
    G_R_NUX,
    hm_reduced_collision_prefactor,
)
from rabbit.collisions.species import BANK_DEGENERACY, Species, as_species
from rabbit.collisions.nu_nu_scattering import nunu_diagonal_monopole_collision, weighted_mean_distribution
from rabbit.transport.angular_decomposition import CollisionMomentResult, CollisionQuadratureSpec


COLLISION_STATISTICAL_MONOMIALS: tuple[str, ...] = ("34", "12", "123", "124", "134", "234")
COLLISION_STATISTICAL_COEFFICIENTS: dict[str, float] = {
    "34": 1.0,
    "12": -1.0,
    "123": 1.0,
    "124": 1.0,
    "134": -1.0,
    "234": -1.0,
}


@dataclass(frozen=True)
class CollisionMomentBatchResult:
    """Batch collision result for staged deterministic reference kernels."""

    C: np.ndarray
    energy_transfer_by_species: Mapping[str, np.ndarray]
    detailed_balance_residual: np.ndarray
    number_residual: np.ndarray
    quadrature_contract: str

    def __post_init__(self) -> None:
        C = np.asarray(self.C, dtype=float)
        if C.ndim != 2:
            raise ValueError("C must have shape (n_batch, n_q).")
        if not np.all(np.isfinite(C)):
            raise ValueError("C must contain only finite values.")
        batch_shape = (C.shape[0],)
        detailed = np.asarray(self.detailed_balance_residual, dtype=float)
        number = np.asarray(self.number_residual, dtype=float)
        if detailed.shape != batch_shape or number.shape != batch_shape:
            raise ValueError("batch residual arrays must have shape (n_batch,).")
        if not np.all(np.isfinite(detailed)) or np.any(detailed < 0.0):
            raise ValueError("detailed_balance_residual must be finite and non-negative.")
        if not np.all(np.isfinite(number)) or np.any(number < 0.0):
            raise ValueError("number_residual must be finite and non-negative.")
        if not self.quadrature_contract:
            raise ValueError("quadrature_contract must be a non-empty string.")
        energy: dict[str, np.ndarray] = {}
        for key, value in self.energy_transfer_by_species.items():
            arr = np.asarray(value, dtype=float)
            if arr.shape != batch_shape:
                raise ValueError("energy transfer arrays must have shape (n_batch,).")
            if not np.all(np.isfinite(arr)):
                raise ValueError("energy transfer arrays must contain only finite values.")
            energy[str(key)] = arr
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "energy_transfer_by_species", energy)
        object.__setattr__(self, "detailed_balance_residual", detailed)
        object.__setattr__(self, "number_residual", number)


def _collision_statistical_factor_unchecked(
    f1: float | np.ndarray,
    f2: float | np.ndarray,
    f3: float | np.ndarray,
    f4: float | np.ndarray,
) -> float | np.ndarray:
    """Return the six-monomial Pauli factor for already validated occupancies."""

    return f3 * f4 - f1 * f2 + f1 * f2 * f3 + f1 * f2 * f4 - f1 * f3 * f4 - f2 * f3 * f4


def _positive_order(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an exact integer.")
    try:
        order = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an exact integer.") from exc
    if order <= 0:
        raise ValueError(f"{name} must be positive.")
    return order


def build_fixed_collision_quadrature(
    *,
    N_q: int,
    N_nue_y2: int,
    N_nue_y3: int,
    N_pair_y2: int,
    N_pair_leg: int,
    contract: str = "fixed_gauss_sn_v1",
) -> CollisionQuadratureSpec:
    """Build deterministic Gauss quadrature nodes for collision references."""

    n_q = _positive_order(N_q, name="N_q")
    n_y2 = _positive_order(N_nue_y2, name="N_nue_y2")
    n_y3 = _positive_order(N_nue_y3, name="N_nue_y3")
    n_pair_y2 = _positive_order(N_pair_y2, name="N_pair_y2")
    n_pair_leg = _positive_order(N_pair_leg, name="N_pair_leg")

    q_nodes, q_weights = laggauss(n_q)
    nue_y2_nodes, nue_y2_weights = laggauss(n_y2)
    nue_y3_nodes, nue_y3_weights = laggauss(n_y3)
    pair_y2_nodes, pair_y2_weights = laggauss(n_pair_y2)
    pair_leg_nodes, pair_leg_weights = leggauss(n_pair_leg)

    return CollisionQuadratureSpec(
        q_nodes=q_nodes,
        q_weights=q_weights,
        nue_y2_nodes=nue_y2_nodes,
        nue_y2_weights=nue_y2_weights,
        nue_y3_nodes=nue_y3_nodes,
        nue_y3_weights=nue_y3_weights,
        pair_y2_nodes=pair_y2_nodes,
        pair_y2_weights=pair_y2_weights,
        pair_leg_nodes=pair_leg_nodes,
        pair_leg_weights=pair_leg_weights,
        contract=contract,
    )


def _validate_collision_occupancies(
    f1: np.ndarray | float,
    f2: np.ndarray | float,
    f3: np.ndarray | float,
    f4: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate scalar occupation numbers for a diagonal no-QKE collision term."""

    f1_arr = np.asarray(f1, dtype=float)
    f2_arr = np.asarray(f2, dtype=float)
    f3_arr = np.asarray(f3, dtype=float)
    f4_arr = np.asarray(f4, dtype=float)
    for name, value in (("f1", f1_arr), ("f2", f2_arr), ("f3", f3_arr), ("f4", f4_arr)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any((value < -1.0e-15) | (value > 1.0 + 1.0e-15)):
            raise ValueError(f"{name} must lie within [0, 1] up to numerical tolerance.")
    return f1_arr, f2_arr, f3_arr, f4_arr


def collision_statistical_polynomial_terms(
    f1: np.ndarray | float,
    f2: np.ndarray | float,
    f3: np.ndarray | float,
    f4: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Return the signed six-monomial Pauli polynomial for a 2-to-2 collision.

    The fermionic gain-loss factor
    ``f3*f4*(1-f1)*(1-f2) - f1*f2*(1-f3)*(1-f4)`` has an exact quartic
    cancellation.  The surviving terms are ``34, 12, 123, 124, 134, 234``,
    with the signs in ``COLLISION_STATISTICAL_COEFFICIENTS``.  Keeping this
    representation explicit makes the deterministic reference path match the
    PSTF collision-term decomposition used by the augmented no-QKE staging line.
    """

    f1_arr, f2_arr, f3_arr, f4_arr = _validate_collision_occupancies(f1, f2, f3, f4)
    return {
        "34": f3_arr * f4_arr,
        "12": -f1_arr * f2_arr,
        "123": f1_arr * f2_arr * f3_arr,
        "124": f1_arr * f2_arr * f4_arr,
        "134": -f1_arr * f3_arr * f4_arr,
        "234": -f2_arr * f3_arr * f4_arr,
    }


def collision_statistical_factor(
    f1: np.ndarray | float,
    f2: np.ndarray | float,
    f3: np.ndarray | float,
    f4: np.ndarray | float,
) -> np.ndarray:
    """Return the Pauli-blocked 2-to-2 statistical factor."""

    f1_arr, f2_arr, f3_arr, f4_arr = _validate_collision_occupancies(f1, f2, f3, f4)
    return _collision_statistical_factor_unchecked(f1_arr, f2_arr, f3_arr, f4_arr)


def detailed_balance_residual(
    f1: np.ndarray | float,
    f2: np.ndarray | float,
    f3: np.ndarray | float,
    f4: np.ndarray | float,
) -> float:
    """Return `max(abs(S))` for the statistical factor."""

    return float(np.max(np.abs(collision_statistical_factor(f1, f2, f3, f4))))


def _fermi_dirac(y: np.ndarray | float, *, eta: float = 0.0) -> np.ndarray:
    x = np.asarray(y, dtype=float)
    return 1.0 / (np.exp(np.clip(x - float(eta), -500.0, 500.0)) + 1.0)


def _positive_temperature(T_MeV: float) -> float:
    value = float(T_MeV)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    return value


def _electron_degeneracy_parameter(electron_chemical_potential_MeV: float, T_MeV: float) -> float:
    mu = float(electron_chemical_potential_MeV)
    if not np.isfinite(mu):
        raise ValueError("electron_chemical_potential_MeV must be finite.")
    return mu / float(T_MeV)


def _nonnegative_rate(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} must be provided for deterministic references.")
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return rate


def _quadrature_arrays(quadrature: CollisionQuadratureSpec) -> dict[str, np.ndarray]:
    arrays = quadrature.arrays()
    q = arrays["q_nodes"]
    if q.ndim != 1 or q.size == 0:
        raise ValueError("quadrature q_nodes must be a non-empty 1D grid.")
    if np.any(q <= 0.0) or np.any(np.diff(q) <= 0.0):
        raise ValueError("quadrature q_nodes must be positive and strictly increasing.")
    return arrays


def _require_matching_nodes(
    q_nodes: np.ndarray,
    partner_nodes: np.ndarray,
    *,
    partner_name: str,
    process: str,
) -> None:
    if q_nodes.shape != partner_nodes.shape or not np.array_equal(q_nodes, partner_nodes):
        raise ValueError(
            f"{process} detailed-balance reference requires q_nodes to match {partner_name}. "
            "Increase both orders together or build a matched-node quadrature."
        )


def _distribution_1d(values: np.ndarray, *, q_nodes: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != q_nodes.shape:
        raise ValueError(f"{name} must have shape matching quadrature q_nodes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((arr < -1.0e-15) | (arr > 1.0 + 1.0e-15)):
        raise ValueError(f"{name} must lie within [0, 1] up to numerical tolerance.")
    return np.clip(arr, 0.0, 1.0)


def _distribution_batch(values: np.ndarray, *, q_nodes: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != q_nodes.size:
        raise ValueError(f"{name} must have shape (n_batch, n_q).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((arr < -1.0e-15) | (arr > 1.0 + 1.0e-15)):
        raise ValueError(f"{name} must lie within [0, 1] up to numerical tolerance.")
    return np.clip(arr, 0.0, 1.0)


def _interp_distribution(y: float, *, q_nodes: np.ndarray, f_values: np.ndarray) -> float:
    return float(np.clip(np.interp(float(y), q_nodes, f_values, left=f_values[0], right=0.0), 0.0, 1.0))


def _interp_distribution_array(
    y: np.ndarray,
    *,
    q_nodes: np.ndarray,
    f_values: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.interp(
            np.asarray(y, dtype=float),
            q_nodes,
            f_values,
            left=f_values[0],
            right=0.0,
        ),
        0.0,
        1.0,
    )


def _interp_distribution_batch(
    y: np.ndarray,
    *,
    q_nodes: np.ndarray,
    f_values: np.ndarray,
) -> np.ndarray:
    """Interpolate a batch of distributions on a shared sorted q grid."""

    q = np.asarray(q_nodes, dtype=float)
    f = _distribution_batch(f_values, q_nodes=q, name="f_values")
    y_arr = np.asarray(y, dtype=float)
    flat_y = y_arr.reshape(-1)
    if q.size == 1:
        interpolated = np.where(flat_y[None, :] <= q[0], f[:, :1], 0.0)
        return np.clip(interpolated.reshape((f.shape[0],) + y_arr.shape), 0.0, 1.0)

    idx_hi = np.searchsorted(q, flat_y, side="right")
    idx_lo = np.clip(idx_hi - 1, 0, q.size - 2)
    idx_next = idx_lo + 1
    denom = q[idx_next] - q[idx_lo]
    frac = (flat_y - q[idx_lo]) / denom
    interpolated = f[:, idx_lo] * (1.0 - frac[None, :]) + f[:, idx_next] * frac[None, :]
    interpolated = np.where(flat_y[None, :] < q[0], f[:, :1], interpolated)
    interpolated = np.where(flat_y[None, :] > q[-1], 0.0, interpolated)
    return np.clip(interpolated.reshape((f.shape[0],) + y_arr.shape), 0.0, 1.0)


def _laguerre_plain_weights(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights * np.exp(np.minimum(nodes, 500.0))


def _moment_integral(q_nodes: np.ndarray, q_weights: np.ndarray, C: np.ndarray, *, power: int) -> float:
    plain_weights = _laguerre_plain_weights(q_nodes, q_weights)
    return float(np.sum(plain_weights * q_nodes**power * C))


def _moment_integral_batch(q_nodes: np.ndarray, q_weights: np.ndarray, C: np.ndarray, *, power: int) -> np.ndarray:
    plain_weights = _laguerre_plain_weights(q_nodes, q_weights)
    return np.sum(plain_weights[None, :] * q_nodes[None, :] ** power * np.asarray(C, dtype=float), axis=1)


def _couplings_for_species(species: str | Species) -> tuple[Species, float, float]:
    sp = as_species(species)
    if sp in (Species.NUE, Species.NUEBAR):
        return sp, G_L_NUE, G_R_NUE
    if sp is Species.NUX:
        return sp, G_L_NUX, G_R_NUX
    raise ValueError(f"Unsupported species: {species}")


def _scattering_matrix_element(G_L: float, G_R: float, y1: float, y2: float, y3: float, y4: float) -> float:
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    return float(
        gl2_gr2 * ((y1 * y2) ** 2 + (y3 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )


def _pair_matrix_element(G_L: float, G_R: float, y1: float, y2: float, y3: float, y4: float) -> float:
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    return float(
        gl2_gr2 * ((y1 * y3) ** 2 + (y2 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )


def evaluate_nue_scattering_reference(
    f_nu: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    species: str | Species = "nue",
    electron_chemical_potential_MeV: float = 0.0,
) -> CollisionMomentResult:
    """Evaluate a deterministic monopole reference for nu-e scattering.

    This is a fixed-quadrature SciPy reference kernel for AP6 validation.
    It is not wired into a production solver path and intentionally returns
    diagnostics rather than hiding quadrature residuals.
    """

    T = _positive_temperature(T_MeV)
    electron_eta = _electron_degeneracy_parameter(electron_chemical_potential_MeV, T)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    _require_matching_nodes(q, arrays["nue_y3_nodes"], partner_name="nue_y3_nodes", process="nu-e scattering")
    qw = arrays["q_weights"]
    f = _distribution_1d(f_nu, q_nodes=q, name="f_nu")
    sp, G_L, G_R = _couplings_for_species(species)

    y2_nodes = arrays["nue_y2_nodes"]
    y2_weights = _laguerre_plain_weights(y2_nodes, arrays["nue_y2_weights"])
    y3_nodes = arrays["nue_y3_nodes"]
    y3_weights = _laguerre_plain_weights(y3_nodes, arrays["nue_y3_weights"])
    prefactor = hm_reduced_collision_prefactor(T)

    y1 = q[:, None, None]
    y3 = y3_nodes[None, :, None]
    y2 = y2_nodes[None, None, :]
    y4 = y1 + y2 - y3
    valid = y4 > 0.0
    f1 = f[:, None, None]
    f2 = _fermi_dirac(y2, eta=electron_eta)
    f3 = _interp_distribution_array(y3_nodes, q_nodes=q, f_values=f)[None, :, None]
    f4 = _fermi_dirac(y4, eta=electron_eta)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    M2 = (
        gl2_gr2 * ((y1 * y2) ** 2 + (y3 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )
    weights = y3_weights[None, :, None] * y2_weights[None, None, :]
    total = np.sum(weights * M2 * S, axis=(1, 2))
    C = prefactor * total / np.maximum(q**2, 1.0e-30)
    max_statistical_residual = float(np.max(np.abs(S))) if S.size else 0.0

    energy_transfer = _moment_integral(q, qw, C, power=3)
    number_residual = abs(_moment_integral(q, qw, C, power=2))
    return CollisionMomentResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


def evaluate_nue_scattering_reference_batch(
    f_nu: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    species: str | Species = "nue",
    electron_chemical_potential_MeV: float = 0.0,
) -> CollisionMomentBatchResult:
    """Evaluate a batch of deterministic no-QKE nu-e scattering references."""

    T = _positive_temperature(T_MeV)
    electron_eta = _electron_degeneracy_parameter(electron_chemical_potential_MeV, T)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    _require_matching_nodes(q, arrays["nue_y3_nodes"], partner_name="nue_y3_nodes", process="nu-e scattering")
    qw = arrays["q_weights"]
    f = _distribution_batch(f_nu, q_nodes=q, name="f_nu")
    sp, G_L, G_R = _couplings_for_species(species)

    y2_nodes = arrays["nue_y2_nodes"]
    y2_weights = _laguerre_plain_weights(y2_nodes, arrays["nue_y2_weights"])
    y3_nodes = arrays["nue_y3_nodes"]
    y3_weights = _laguerre_plain_weights(y3_nodes, arrays["nue_y3_weights"])
    prefactor = hm_reduced_collision_prefactor(T)

    y1 = q[None, :, None, None]
    y3 = y3_nodes[None, None, :, None]
    y2 = y2_nodes[None, None, None, :]
    y4 = y1 + y2 - y3
    valid = y4 > 0.0
    f1 = f[:, :, None, None]
    f2 = _fermi_dirac(y2, eta=electron_eta)
    f3 = _interp_distribution_batch(y3_nodes, q_nodes=q, f_values=f)[:, None, :, None]
    f4 = _fermi_dirac(y4, eta=electron_eta)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    M2 = (
        gl2_gr2 * ((y1 * y2) ** 2 + (y3 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )
    weights = y3_weights[None, None, :, None] * y2_weights[None, None, None, :]
    total = np.sum(weights * M2 * S, axis=(2, 3))
    C = prefactor * total / np.maximum(q[None, :] ** 2, 1.0e-30)
    max_statistical_residual = np.max(np.abs(S), axis=(1, 2, 3)) if S.size else np.zeros(f.shape[0])

    energy_transfer = _moment_integral_batch(q, qw, C, power=3)
    number_residual = np.abs(_moment_integral_batch(q, qw, C, power=2))
    return CollisionMomentBatchResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


def evaluate_pair_annihilation_reference(
    f_target: np.ndarray,
    f_partner: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    species: str | Species = "nue",
    electron_chemical_potential_MeV: float = 0.0,
) -> CollisionMomentResult:
    """Evaluate a deterministic monopole reference for pair processes.

    The returned collision field is for ``f_target``.  ``species`` labels
    that target bank; pass ``species='nuebar'`` only when ``f_target`` is
    the antineutrino distribution and ``f_partner`` is the neutrino partner.
    """

    T = _positive_temperature(T_MeV)
    electron_eta = _electron_degeneracy_parameter(electron_chemical_potential_MeV, T)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    _require_matching_nodes(q, arrays["pair_y2_nodes"], partner_name="pair_y2_nodes", process="pair-process")
    qw = arrays["q_weights"]
    f = _distribution_1d(f_target, q_nodes=q, name="f_target")
    fbar = _distribution_1d(f_partner, q_nodes=q, name="f_partner")
    sp, G_L, G_R = _couplings_for_species(species)

    y2_nodes = arrays["pair_y2_nodes"]
    y2_weights = _laguerre_plain_weights(y2_nodes, arrays["pair_y2_weights"])
    leg_nodes = arrays["pair_leg_nodes"]
    leg_weights = arrays["pair_leg_weights"]
    prefactor = hm_reduced_collision_prefactor(T)

    y1 = q[:, None, None]
    y2 = y2_nodes[None, :, None]
    y_sum = y1 + y2
    y3 = 0.5 * y_sum * (leg_nodes[None, None, :] + 1.0)
    y4 = y_sum - y3
    valid = (y_sum > 0.0) & (y4 > 0.0)
    f1 = f[:, None, None]
    f2 = _interp_distribution_array(y2_nodes, q_nodes=q, f_values=fbar)[None, :, None]
    f3 = _fermi_dirac(y3, eta=-electron_eta)
    f4 = _fermi_dirac(y4, eta=electron_eta)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    M2 = (
        gl2_gr2 * ((y1 * y3) ** 2 + (y2 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )
    y3_weights = 0.5 * y_sum * leg_weights[None, None, :]
    weights = y2_weights[None, :, None] * y3_weights
    total = np.sum(weights * M2 * S, axis=(1, 2))
    C = prefactor * total / np.maximum(q**2, 1.0e-30)
    max_statistical_residual = float(np.max(np.abs(S))) if S.size else 0.0

    energy_transfer = _moment_integral(q, qw, C, power=3)
    number_residual = abs(_moment_integral(q, qw, C, power=2))
    return CollisionMomentResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


def evaluate_pair_annihilation_reference_batch(
    f_target: np.ndarray,
    f_partner: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    species: str | Species = "nue",
    electron_chemical_potential_MeV: float = 0.0,
) -> CollisionMomentBatchResult:
    """Evaluate a batch of deterministic no-QKE pair-process references."""

    T = _positive_temperature(T_MeV)
    electron_eta = _electron_degeneracy_parameter(electron_chemical_potential_MeV, T)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    _require_matching_nodes(q, arrays["pair_y2_nodes"], partner_name="pair_y2_nodes", process="pair-process")
    qw = arrays["q_weights"]
    f = _distribution_batch(f_target, q_nodes=q, name="f_target")
    fbar = _distribution_batch(f_partner, q_nodes=q, name="f_partner")
    if fbar.shape != f.shape:
        raise ValueError("f_partner must match f_target batch shape.")
    sp, G_L, G_R = _couplings_for_species(species)

    y2_nodes = arrays["pair_y2_nodes"]
    y2_weights = _laguerre_plain_weights(y2_nodes, arrays["pair_y2_weights"])
    leg_nodes = arrays["pair_leg_nodes"]
    leg_weights = arrays["pair_leg_weights"]
    prefactor = hm_reduced_collision_prefactor(T)

    y1 = q[None, :, None, None]
    y2 = y2_nodes[None, None, :, None]
    y_sum = y1 + y2
    y3 = 0.5 * y_sum * (leg_nodes[None, None, None, :] + 1.0)
    y4 = y_sum - y3
    valid = (y_sum > 0.0) & (y4 > 0.0)
    f1 = f[:, :, None, None]
    f2 = _interp_distribution_batch(y2_nodes, q_nodes=q, f_values=fbar)[:, None, :, None]
    f3 = _fermi_dirac(y3, eta=-electron_eta)
    f4 = _fermi_dirac(y4, eta=electron_eta)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    M2 = (
        gl2_gr2 * ((y1 * y3) ** 2 + (y2 * y4) ** 2)
        + gl2_minus_gr2 * ((y1 * y4) ** 2 + (y2 * y3) ** 2)
    )
    y3_weights = 0.5 * y_sum * leg_weights[None, None, None, :]
    weights = y2_weights[None, None, :, None] * y3_weights
    total = np.sum(weights * M2 * S, axis=(2, 3))
    C = prefactor * total / np.maximum(q[None, :] ** 2, 1.0e-30)
    max_statistical_residual = np.max(np.abs(S), axis=(1, 2, 3)) if S.size else np.zeros(f.shape[0])

    energy_transfer = _moment_integral_batch(q, qw, C, power=3)
    number_residual = np.abs(_moment_integral_batch(q, qw, C, power=2))
    return CollisionMomentBatchResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


def evaluate_nunu_diagonal_reference(
    f_by_species: dict[str | Species, np.ndarray],
    *,
    quadrature: CollisionQuadratureSpec,
    T_nu_e: float,
    T_nu_x: float,
    gamma: float | None = None,
) -> dict[str, CollisionMomentResult]:
    """Evaluate a deterministic no-QKE diagonal nu-nu redistribution reference.

    This wraps the existing pointwise bank-redistribution helper and reports
    per-bank energy moments plus shared weighted conservation residuals.
    ``detailed_balance_residual`` is the unscaled fixed-point residual
    ``max(abs(f_s - fbar))`` for each returned bank, not a Pauli statistical
    factor.  It is a classical diagonal collision reference only; it does
    not introduce off-diagonal QKE coherence.
    """

    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    qw = arrays["q_weights"]
    T_e = _positive_temperature(T_nu_e)
    T_x = _positive_temperature(T_nu_x)
    gamma_value = _nonnegative_rate(gamma, name="gamma")
    checked: dict[str, np.ndarray] = {}
    for key, values in f_by_species.items():
        sp = as_species(key)
        if sp.value in checked:
            raise ValueError(f"duplicate species entry for {sp.value}.")
        checked[sp.value] = _distribution_1d(values, q_nodes=q, name=f"f_by_species[{sp.value}]")
    expected = {Species.NUE.value, Species.NUEBAR.value, Species.NUX.value}
    if set(checked) != expected:
        raise ValueError("f_by_species must contain exactly nue, nuebar, and nux.")

    C_by_species = nunu_diagonal_monopole_collision(
        checked,
        T_nu_e=T_e,
        T_nu_x=T_x,
        gamma=gamma_value,
    )
    fbar = weighted_mean_distribution(checked)
    weighted_C = np.zeros_like(q, dtype=float)
    energy_by_species: dict[str, float] = {}
    fixed_point_residual_by_species: dict[str, float] = {}
    for sp_key, C in C_by_species.items():
        sp = as_species(sp_key)
        C_arr = np.asarray(C, dtype=float)
        weighted_C = weighted_C + BANK_DEGENERACY[sp] * C_arr
        energy_by_species[sp_key] = _moment_integral(q, qw, C_arr, power=3)
        fixed_point_residual_by_species[sp_key] = float(np.max(np.abs(checked[sp_key] - fbar)))

    shared_number_residual = abs(_moment_integral(q, qw, weighted_C, power=2))

    return {
        sp_key: CollisionMomentResult(
            C=np.asarray(C, dtype=float),
            energy_transfer_by_species=energy_by_species,
            detailed_balance_residual=fixed_point_residual_by_species[sp_key],
            number_residual=shared_number_residual,
            quadrature_contract=quadrature.contract,
        )
        for sp_key, C in C_by_species.items()
    }


def evaluate_nunu_diagonal_twoto2_reference(
    f_alpha: np.ndarray,
    f_beta: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    gamma: float | None,
    epsilon_alpha_beta: float = 1.0,
    species: str | Species = "nue",
) -> CollisionMomentResult:
    """Evaluate a fixed-quadrature diagonal no-QKE nu-nu 2-to-2 reference.

    This is a NumPy mirror of the JAX diagonal nu-nu preflight kernel's scalar
    occupation-number algebra.  It uses the AP81 six-monomial Pauli factor and
    the transparent angular-averaged matrix element
    ``(y1*y2)**2 + (y3*y4)**2``.  The existing AP19/AP33 relaxation source is
    intentionally left as a separate staged bridge until its routing is
    revalidated against this physical reference.
    """

    T = _positive_temperature(T_MeV)
    gamma_value = _nonnegative_rate(gamma, name="gamma")
    epsilon = float(epsilon_alpha_beta)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon_alpha_beta must be positive and finite.")
    sp = as_species(species)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    qw = arrays["q_weights"]
    f_a = _distribution_1d(f_alpha, q_nodes=q, name="f_alpha")
    f_b = _distribution_1d(f_beta, q_nodes=q, name="f_beta")
    plain_weights = _laguerre_plain_weights(q, qw)
    prefactor = epsilon * gamma_value * hm_reduced_collision_prefactor(T)

    y1 = q[:, None, None]
    y3 = q[None, :, None]
    y2 = q[None, None, :]
    y4 = y1 + y2 - y3
    valid = y4 > 0.0
    f1 = f_a[:, None, None]
    f3 = f_a[None, :, None]
    f2 = f_b[None, None, :]
    f4 = _interp_distribution_array(y4, q_nodes=q, f_values=f_b)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    M2 = (y1 * y2) ** 2 + (y3 * y4) ** 2
    weights = plain_weights[None, :, None] * plain_weights[None, None, :]
    total = np.sum(weights * M2 * S, axis=(1, 2))
    C = prefactor * total / np.maximum(q**2, 1.0e-30)
    max_statistical_residual = float(np.max(np.abs(S))) if S.size else 0.0

    energy_transfer = _moment_integral(q, qw, C, power=3)
    number_residual = abs(_moment_integral(q, qw, C, power=2))
    return CollisionMomentResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


def evaluate_nunu_diagonal_twoto2_reference_batch(
    f_alpha: np.ndarray,
    f_beta: np.ndarray,
    *,
    quadrature: CollisionQuadratureSpec,
    T_MeV: float,
    gamma: float | None,
    epsilon_alpha_beta: float = 1.0,
    species: str | Species = "nue",
) -> CollisionMomentBatchResult:
    """Evaluate a batch of deterministic diagonal no-QKE nu-nu 2-to-2 references."""

    T = _positive_temperature(T_MeV)
    gamma_value = _nonnegative_rate(gamma, name="gamma")
    epsilon = float(epsilon_alpha_beta)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon_alpha_beta must be positive and finite.")
    sp = as_species(species)
    arrays = _quadrature_arrays(quadrature)
    q = arrays["q_nodes"]
    qw = arrays["q_weights"]
    f_a = _distribution_batch(f_alpha, q_nodes=q, name="f_alpha")
    f_b = _distribution_batch(f_beta, q_nodes=q, name="f_beta")
    if f_b.shape != f_a.shape:
        raise ValueError("f_beta must match f_alpha batch shape.")
    plain_weights = _laguerre_plain_weights(q, qw)
    prefactor = epsilon * gamma_value * hm_reduced_collision_prefactor(T)

    y1 = q[None, :, None, None]
    y3 = q[None, None, :, None]
    y2 = q[None, None, None, :]
    y4 = y1 + y2 - y3
    valid = y4 > 0.0
    f1 = f_a[:, :, None, None]
    f3 = f_a[:, None, :, None]
    f2 = f_b[:, None, None, :]
    f4 = _interp_distribution_batch(y4[0], q_nodes=q, f_values=f_b)
    S = np.where(valid, _collision_statistical_factor_unchecked(f1, f2, f3, f4), 0.0)
    M2 = (y1 * y2) ** 2 + (y3 * y4) ** 2
    weights = plain_weights[None, None, :, None] * plain_weights[None, None, None, :]
    total = np.sum(weights * M2 * S, axis=(2, 3))
    C = prefactor * total / np.maximum(q[None, :] ** 2, 1.0e-30)
    max_statistical_residual = np.max(np.abs(S), axis=(1, 2, 3)) if S.size else np.zeros(f_a.shape[0])

    energy_transfer = _moment_integral_batch(q, qw, C, power=3)
    number_residual = np.abs(_moment_integral_batch(q, qw, C, power=2))
    return CollisionMomentBatchResult(
        C=C,
        energy_transfer_by_species={sp.value: energy_transfer},
        detailed_balance_residual=max_statistical_residual,
        number_residual=number_residual,
        quadrature_contract=quadrature.contract,
    )


__all__ = [
    "COLLISION_STATISTICAL_COEFFICIENTS",
    "COLLISION_STATISTICAL_MONOMIALS",
    "CollisionMomentBatchResult",
    "build_fixed_collision_quadrature",
    "collision_statistical_factor",
    "collision_statistical_polynomial_terms",
    "detailed_balance_residual",
    "evaluate_nue_scattering_reference",
    "evaluate_nue_scattering_reference_batch",
    "evaluate_nunu_diagonal_reference",
    "evaluate_nunu_diagonal_twoto2_reference",
    "evaluate_nunu_diagonal_twoto2_reference_batch",
    "evaluate_pair_annihilation_reference",
    "evaluate_pair_annihilation_reference_batch",
]
