"""PSTF collision-kernel contraction helpers.

This module implements reusable pieces of the six-monomial angular
contraction described in ``neutrino_collision_term_PSTF.md``: deterministic
local angular projection, universal momentum-conservation geometry tables,
channel-specific matrix-element assembly, and a radial-grid contraction with
linear ``p4`` energy-conservation interpolation.  It remains a no-QKE
building block rather than a promoted collision-coupled BBN runtime.
"""
from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from rabbit.collisions.deterministic_reference import (
    COLLISION_STATISTICAL_COEFFICIENTS,
    COLLISION_STATISTICAL_MONOMIALS,
    collision_statistical_polynomial_terms,
)


_CONTRACTION_CONTRACT = "local_pstf_six_monomial_projection_v1"
_GEOMETRIC_CONTRACT = "universal_pstf_geometric_kernels_v1"
_CHANNEL_CONTRACT = "pstf_channel_kernel_from_geometric_table_v1"
_RADIAL_CONTRACT = "pstf_channel_radial_grid_contraction_v1"
_RADIAL_GRID_CONTRACT = "pstf_radial_channel_kernel_grid_v1"
_RADIAL_GRID_BATCH_CONTRACT = "pstf_radial_channel_kernel_grid_batch_v1"
_RADIAL_GRID_NPZ_CONTRACT = "pstf_radial_channel_kernel_grid_npz_v1"
_RADIAL_BATCH_EINSUM_PATH_CACHE_LIMIT = 64
_RADIAL_BATCH_EINSUM_PATH_CACHE: dict[tuple[object, ...], list[object]] = {}
_PARTICLE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


def _cached_radial_batch_einsum(
    subscripts: str,
    *operands: np.ndarray,
) -> np.ndarray:
    key = (
        str(subscripts),
        tuple(
            (tuple(np.shape(operand)), np.asarray(operand).dtype.str)
            for operand in operands
        ),
    )
    path = _RADIAL_BATCH_EINSUM_PATH_CACHE.get(key)
    if path is None:
        path, _ = np.einsum_path(subscripts, *operands, optimize="greedy")
        if len(_RADIAL_BATCH_EINSUM_PATH_CACHE) >= _RADIAL_BATCH_EINSUM_PATH_CACHE_LIMIT:
            _RADIAL_BATCH_EINSUM_PATH_CACHE.clear()
        _RADIAL_BATCH_EINSUM_PATH_CACHE[key] = path
    return np.einsum(subscripts, *operands, optimize=path)


def _cache_array_key(
    value: object,
    *,
    memo: MutableMapping[int, tuple[np.ndarray, tuple[object, ...]]] | None = None,
) -> tuple[object, ...]:
    array = np.ascontiguousarray(np.asarray(value))
    if memo is not None:
        memo_key = id(array)
        cached = memo.get(memo_key)
        if cached is not None and cached[0] is array:
            return cached[1]
    key = ("array", array.shape, str(array.dtype), array.tobytes())
    if memo is not None:
        memo[id(array)] = (array, key)
    return key


def _geometric_table_cache_prefix(
    basis_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    weight_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direction_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    mu_pairs: tuple[tuple[int, int], ...],
    mumu_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    memo: MutableMapping[int, tuple[np.ndarray, tuple[object, ...]]] | None = None,
) -> tuple[object, ...]:
    return (
        "universal_pstf_geometric_kernel_v1",
        tuple(_cache_array_key(value, memo=memo) for value in basis_tuple),
        tuple(_cache_array_key(value, memo=memo) for value in weight_tuple),
        tuple(_cache_array_key(value, memo=memo) for value in direction_tuple),
        tuple(tuple(pair) for pair in mu_pairs),
        tuple((tuple(first), tuple(second)) for first, second in mumu_pairs),
    )


def _validate_direction_weight_tuple(
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    if len(direction_vectors_by_particle) != 4 or len(angular_weights_by_particle) != 4:
        raise ValueError("momentum-delta geometry requires exactly four particle direction/weight arrays.")
    directions = tuple(np.asarray(value, dtype=float) for value in direction_vectors_by_particle)
    weights = tuple(np.asarray(value, dtype=float) for value in angular_weights_by_particle)
    for idx, (dirs, w) in enumerate(zip(directions, weights), start=1):
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError(f"direction_vectors_by_particle[{idx}] must have shape (n_angles, 3).")
        if w.shape != (dirs.shape[0],):
            raise ValueError(f"angular_weights_by_particle[{idx}] must match its direction axis.")
        if not np.all(np.isfinite(dirs)) or not np.all(np.isfinite(w)):
            raise ValueError("momentum-delta directions and weights must contain only finite values.")
        if np.any(w <= 0.0):
            raise ValueError("momentum-delta angular weights must be strictly positive.")
        norms = np.linalg.norm(dirs, axis=1)
        if not np.allclose(norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("momentum-delta direction vectors must be unit 3-vectors.")
    return directions, weights


def build_normalized_unit_direction_momentum_delta_weights(
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    sigma: float = 0.35,
) -> np.ndarray:
    """Return normalized four-angle weights favoring vector-momentum closure.

    This deterministic no-QKE approximation replaces the previous uniform
    angular factor in smoke-scale PSTF radial sources with a Gaussian in the
    unit-direction residual ``|e1 + e2 - e3 - e4|``.  The returned rank-4
    tensor is normalized so that its angular-quadrature integral is one, which
    preserves the radial source scale while giving the local angular
    contraction real momentum-closure structure.
    """

    directions, weights = _validate_direction_weight_tuple(
        direction_vectors_by_particle,
        angular_weights_by_particle,
    )
    width = float(sigma)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("sigma must be positive and finite.")
    residual = (
        directions[0][:, None, None, None, :]
        + directions[1][None, :, None, None, :]
        - directions[2][None, None, :, None, :]
        - directions[3][None, None, None, :, :]
    )
    raw = np.exp(-0.5 * np.sum(residual * residual, axis=-1) / (width * width))
    norm = float(np.einsum("a,b,c,d,abcd->", weights[0], weights[1], weights[2], weights[3], raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalized unit-direction momentum-delta weights have zero quadrature norm.")
    return raw / norm


def _positive_radial_momenta(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values.")
    return arr


def build_normalized_radial_momentum_delta_weight_provider(
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    p1_momenta: np.ndarray,
    p2_momenta: np.ndarray,
    p3_momenta: np.ndarray,
    sigma_fraction: float = 0.35,
    min_sigma: float = 1.0e-12,
) -> Callable[[int, int, int, float, float], np.ndarray]:
    """Build p-dependent normalized Gaussian weights for angular momentum closure.

    The returned callable matches ``build_pstf_radial_channel_kernel_grid``'s
    ``momentum_delta_weights`` protocol.  For each radial tuple it evaluates a
    Gaussian in ``|p1 e1 + p2 e2 - p3 e3 - p4 e4|`` and normalizes the rank-4
    angular tensor against the supplied quadrature weights.
    """

    directions, weights = _validate_direction_weight_tuple(
        direction_vectors_by_particle,
        angular_weights_by_particle,
    )
    p1_grid = _positive_radial_momenta(p1_momenta, name="p1_momenta")
    p2_grid = _positive_radial_momenta(p2_momenta, name="p2_momenta")
    p3_grid = _positive_radial_momenta(p3_momenta, name="p3_momenta")
    fraction = float(sigma_fraction)
    floor = float(min_sigma)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("sigma_fraction must be positive and finite.")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("min_sigma must be positive and finite.")

    def provider(i1: int, i2: int, i3: int, e4: float, p4: float) -> np.ndarray:
        del e4
        p1 = float(p1_grid[int(i1)])
        p2 = float(p2_grid[int(i2)])
        p3 = float(p3_grid[int(i3)])
        p4_value = float(p4)
        if not np.isfinite(p4_value) or p4_value < 0.0:
            raise ValueError("p4 must be finite and non-negative.")
        scale = max(p1 + p2 + p3 + p4_value, floor)
        sigma = max(fraction * scale, floor)
        residual = (
            p1 * directions[0][:, None, None, None, :]
            + p2 * directions[1][None, :, None, None, :]
            - p3 * directions[2][None, None, :, None, :]
            - p4_value * directions[3][None, None, None, :, :]
        )
        raw = np.exp(-0.5 * np.sum(residual * residual, axis=-1) / (sigma * sigma))
        norm = float(np.einsum("a,b,c,d,abcd->", weights[0], weights[1], weights[2], weights[3], raw))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("normalized radial momentum-delta weights have zero quadrature norm.")
        return raw / norm

    return provider


@dataclass(frozen=True)
class LocalPSTFStatisticalKernelTable:
    """Precomputed angular tensors for the local six-monomial contraction."""

    basis_matrix: np.ndarray
    angular_weights: np.ndarray
    projection_matrix: np.ndarray
    monomial_tensors: Mapping[str, np.ndarray]
    monomial_angular_kernels: Mapping[str, np.ndarray]
    gram_condition: float
    contraction_contract: str = _CONTRACTION_CONTRACT

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis_matrix, dtype=float)
        weights = np.asarray(self.angular_weights, dtype=float)
        projector = np.asarray(self.projection_matrix, dtype=float)
        if basis.ndim != 2:
            raise ValueError("basis_matrix must have shape (n_modes, n_angles).")
        if weights.shape != (basis.shape[1],):
            raise ValueError("angular_weights must match the basis angular axis.")
        if projector.shape != basis.shape:
            raise ValueError("projection_matrix must match basis_matrix shape.")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(weights)):
            raise ValueError("basis_matrix and angular_weights must contain only finite values.")
        if not np.all(np.isfinite(projector)):
            raise ValueError("projection_matrix must contain only finite values.")
        if np.any(weights <= 0.0):
            raise ValueError("angular_weights must be strictly positive.")
        tensors = {str(key): np.asarray(value, dtype=float) for key, value in self.monomial_tensors.items()}
        kernels = {
            str(key): np.asarray(value, dtype=float)
            for key, value in self.monomial_angular_kernels.items()
        }
        if tuple(tensors) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("monomial_tensors must follow COLLISION_STATISTICAL_MONOMIALS order.")
        if tuple(kernels) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("monomial_angular_kernels must follow COLLISION_STATISTICAL_MONOMIALS order.")
        n_modes = basis.shape[0]
        for key in ("34", "12"):
            if tensors[key].shape != (n_modes, n_modes, n_modes):
                raise ValueError(f"monomial_tensors[{key!r}] has the wrong rank.")
        for key in ("123", "124", "134", "234"):
            if tensors[key].shape != (n_modes, n_modes, n_modes, n_modes):
                raise ValueError(f"monomial_tensors[{key!r}] has the wrong rank.")
        for key, kernel in kernels.items():
            if kernel.shape != weights.shape:
                raise ValueError(f"monomial_angular_kernels[{key!r}] must match angular_weights.")
            if not np.all(np.isfinite(kernel)):
                raise ValueError(f"monomial_angular_kernels[{key!r}] must contain only finite values.")
        cond = float(self.gram_condition)
        if not np.isfinite(cond) or cond <= 0.0:
            raise ValueError("gram_condition must be positive and finite.")
        if not self.contraction_contract:
            raise ValueError("contraction_contract must be a non-empty string.")
        object.__setattr__(self, "basis_matrix", basis)
        object.__setattr__(self, "angular_weights", weights)
        object.__setattr__(self, "projection_matrix", projector)
        object.__setattr__(self, "monomial_tensors", tensors)
        object.__setattr__(self, "monomial_angular_kernels", kernels)
        object.__setattr__(self, "gram_condition", cond)


@dataclass(frozen=True)
class LocalPSTFStatisticalContractionResult:
    """Mode-space result for the local six-monomial PSTF contraction."""

    C_modes: np.ndarray
    nodal_source: np.ndarray
    nodal_occupancies: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    monomial_mode_contributions: Mapping[str, np.ndarray]
    monomial_nodal_contributions: Mapping[str, np.ndarray]
    max_occupancy_overshoot: float
    contraction_contract: str = _CONTRACTION_CONTRACT

    def __post_init__(self) -> None:
        modes = np.asarray(self.C_modes, dtype=float)
        source = np.asarray(self.nodal_source, dtype=float)
        if modes.ndim != 1:
            raise ValueError("C_modes must be a 1D mode vector.")
        if source.ndim != 1:
            raise ValueError("nodal_source must be a 1D angular vector.")
        if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(source)):
            raise ValueError("contraction result arrays must contain only finite values.")
        occupancies = tuple(np.asarray(value, dtype=float) for value in self.nodal_occupancies)
        if len(occupancies) != 4:
            raise ValueError("nodal_occupancies must contain four arrays.")
        for occupancy in occupancies:
            if occupancy.shape != source.shape:
                raise ValueError("nodal occupancy arrays must match nodal_source shape.")
            if not np.all(np.isfinite(occupancy)):
                raise ValueError("nodal occupancy arrays must contain only finite values.")
        mode_contribs = {
            str(key): np.asarray(value, dtype=float)
            for key, value in self.monomial_mode_contributions.items()
        }
        nodal_contribs = {
            str(key): np.asarray(value, dtype=float)
            for key, value in self.monomial_nodal_contributions.items()
        }
        if tuple(mode_contribs) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("monomial_mode_contributions must follow COLLISION_STATISTICAL_MONOMIALS order.")
        if tuple(nodal_contribs) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("monomial_nodal_contributions must follow COLLISION_STATISTICAL_MONOMIALS order.")
        for key, value in mode_contribs.items():
            if value.shape != modes.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"monomial_mode_contributions[{key!r}] must match C_modes.")
        for key, value in nodal_contribs.items():
            if value.shape != source.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"monomial_nodal_contributions[{key!r}] must match nodal_source.")
        overshoot = float(self.max_occupancy_overshoot)
        if not np.isfinite(overshoot) or overshoot < 0.0:
            raise ValueError("max_occupancy_overshoot must be finite and non-negative.")
        if not self.contraction_contract:
            raise ValueError("contraction_contract must be a non-empty string.")
        object.__setattr__(self, "C_modes", modes)
        object.__setattr__(self, "nodal_source", source)
        object.__setattr__(self, "nodal_occupancies", occupancies)
        object.__setattr__(self, "monomial_mode_contributions", mode_contribs)
        object.__setattr__(self, "monomial_nodal_contributions", nodal_contribs)
        object.__setattr__(self, "max_occupancy_overshoot", overshoot)


@dataclass(frozen=True)
class UniversalPSTFGeometricKernelTable:
    """Universal four-angle PSTF geometric kernels for one radial tuple."""

    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    momentum_delta_weights: np.ndarray
    output_projection_matrix: np.ndarray
    G0: Mapping[str, np.ndarray]
    G_mu: Mapping[tuple[int, int], Mapping[str, np.ndarray]]
    G_mumu: Mapping[tuple[tuple[int, int], tuple[int, int]], Mapping[str, np.ndarray]]
    geometric_contract: str = _GEOMETRIC_CONTRACT

    def __post_init__(self) -> None:
        basis_tuple = tuple(np.asarray(value, dtype=float) for value in self.basis_by_particle)
        weight_tuple = tuple(np.asarray(value, dtype=float) for value in self.angular_weights_by_particle)
        direction_tuple = tuple(np.asarray(value, dtype=float) for value in self.direction_vectors_by_particle)
        if len(basis_tuple) != 4 or len(weight_tuple) != 4 or len(direction_tuple) != 4:
            raise ValueError("geometric kernel tables require exactly four particles.")
        angle_shape = []
        for idx, (basis, weights, directions) in enumerate(zip(basis_tuple, weight_tuple, direction_tuple), start=1):
            if basis.ndim != 2:
                raise ValueError(f"basis_by_particle[{idx}] must have shape (n_modes, n_angles).")
            if weights.shape != (basis.shape[1],):
                raise ValueError(f"angular_weights_by_particle[{idx}] must match its basis angular axis.")
            if directions.shape != (basis.shape[1], 3):
                raise ValueError(f"direction_vectors_by_particle[{idx}] must have shape (n_angles, 3).")
            if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(weights)):
                raise ValueError("geometric basis and angular weights must contain only finite values.")
            if not np.all(np.isfinite(directions)):
                raise ValueError("direction vectors must contain only finite values.")
            if np.any(weights <= 0.0):
                raise ValueError("geometric angular weights must be strictly positive.")
            norms = np.linalg.norm(directions, axis=1)
            if not np.allclose(norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError("direction vectors must be unit 3-vectors.")
            angle_shape.append(basis.shape[1])
        delta = np.asarray(self.momentum_delta_weights, dtype=float)
        if delta.shape != tuple(angle_shape):
            raise ValueError("momentum_delta_weights must have shape (n_a1, n_a2, n_a3, n_a4).")
        if not np.all(np.isfinite(delta)):
            raise ValueError("momentum_delta_weights must contain only finite values.")
        projector = np.asarray(self.output_projection_matrix, dtype=float)
        if projector.shape != basis_tuple[0].shape:
            raise ValueError("output_projection_matrix must match particle-1 basis shape.")
        if not np.all(np.isfinite(projector)):
            raise ValueError("output_projection_matrix must contain only finite values.")
        G0 = {str(key): np.asarray(value, dtype=float) for key, value in self.G0.items()}
        G_mu = {
            _canonical_pair(key): {
                str(monomial): np.asarray(value, dtype=float)
                for monomial, value in tensors.items()
            }
            for key, tensors in self.G_mu.items()
        }
        G_mumu = {
            _canonical_pair_pair(key): {
                str(monomial): np.asarray(value, dtype=float)
                for monomial, value in tensors.items()
            }
            for key, tensors in self.G_mumu.items()
        }
        _validate_geometric_tensor_mapping(G0, basis_tuple, name="G0")
        for pair, tensors in G_mu.items():
            _validate_pair(pair)
            _validate_geometric_tensor_mapping(tensors, basis_tuple, name=f"G_mu[{pair!r}]")
        for pair_pair, tensors in G_mumu.items():
            _validate_pair(pair_pair[0])
            _validate_pair(pair_pair[1])
            _validate_geometric_tensor_mapping(tensors, basis_tuple, name=f"G_mumu[{pair_pair!r}]")
        if not self.geometric_contract:
            raise ValueError("geometric_contract must be a non-empty string.")
        object.__setattr__(self, "basis_by_particle", basis_tuple)
        object.__setattr__(self, "angular_weights_by_particle", weight_tuple)
        object.__setattr__(self, "direction_vectors_by_particle", direction_tuple)
        object.__setattr__(self, "momentum_delta_weights", delta)
        object.__setattr__(self, "output_projection_matrix", projector)
        object.__setattr__(self, "G0", G0)
        object.__setattr__(self, "G_mu", G_mu)
        object.__setattr__(self, "G_mumu", G_mumu)


@dataclass(frozen=True)
class PSTFBilinearMatrixElementTerm:
    """Descriptor for an ``eta * Pi_ij * Pi_kl`` matrix-element term."""

    eta: float
    first_pair: tuple[int, int]
    second_pair: tuple[int, int]

    def __post_init__(self) -> None:
        eta = float(self.eta)
        if not np.isfinite(eta):
            raise ValueError("eta must be finite.")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "first_pair", _canonical_pair(tuple(self.first_pair)))
        object.__setattr__(self, "second_pair", _canonical_pair(tuple(self.second_pair)))


@dataclass(frozen=True)
class PSTFMassMatrixElementTerm:
    """Descriptor for a ``zeta * m_e^2 c^2 * Pi_ij`` matrix-element term."""

    zeta: float
    pair: tuple[int, int]

    def __post_init__(self) -> None:
        zeta = float(self.zeta)
        if not np.isfinite(zeta):
            raise ValueError("zeta must be finite.")
        object.__setattr__(self, "zeta", zeta)
        object.__setattr__(self, "pair", _canonical_pair(tuple(self.pair)))


@dataclass(frozen=True)
class PSTFMassQuarticMatrixElementTerm:
    """Descriptor for a ``zeta * m_e^4 c^4`` matrix-element term."""

    zeta: float

    def __post_init__(self) -> None:
        zeta = float(self.zeta)
        if not np.isfinite(zeta):
            raise ValueError("zeta must be finite.")
        object.__setattr__(self, "zeta", zeta)


@dataclass(frozen=True)
class PSTFChannelKernelTable:
    """Channel-specific PSTF ``K`` tensors assembled from geometric tables."""

    K_by_monomial: Mapping[str, np.ndarray]
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...]
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...]
    energies: tuple[float, float, float, float]
    momenta: tuple[float, float, float, float]
    electron_mass_momentum: float
    coupling_prefactor: float
    radial_prefactor: float
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...] = ()
    channel_contract: str = _CHANNEL_CONTRACT

    def __post_init__(self) -> None:
        kernels = {str(key): np.asarray(value, dtype=float) for key, value in self.K_by_monomial.items()}
        if tuple(kernels) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("K_by_monomial must follow COLLISION_STATISTICAL_MONOMIALS order.")
        if not kernels:
            raise ValueError("K_by_monomial must not be empty.")
        for key, value in kernels.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"K_by_monomial[{key!r}] must contain only finite values.")
        energies = _positive_four_vector(self.energies, name="energies")
        momenta = _nonnegative_four_vector(self.momenta, name="momenta")
        electron_mass = float(self.electron_mass_momentum)
        coupling = float(self.coupling_prefactor)
        radial = float(self.radial_prefactor)
        if not np.isfinite(electron_mass) or electron_mass < 0.0:
            raise ValueError("electron_mass_momentum must be non-negative and finite.")
        if not np.isfinite(coupling):
            raise ValueError("coupling_prefactor must be finite.")
        if not np.isfinite(radial):
            raise ValueError("radial_prefactor must be finite.")
        bilinear = tuple(
            term if isinstance(term, PSTFBilinearMatrixElementTerm) else PSTFBilinearMatrixElementTerm(**term)
            for term in self.bilinear_terms
        )
        mass = tuple(
            term if isinstance(term, PSTFMassMatrixElementTerm) else PSTFMassMatrixElementTerm(**term)
            for term in self.mass_terms
        )
        mass_quartic = tuple(
            term if isinstance(term, PSTFMassQuarticMatrixElementTerm) else PSTFMassQuarticMatrixElementTerm(**term)
            for term in self.mass_quartic_terms
        )
        if not self.channel_contract:
            raise ValueError("channel_contract must be a non-empty string.")
        object.__setattr__(self, "K_by_monomial", kernels)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "momenta", momenta)
        object.__setattr__(self, "electron_mass_momentum", electron_mass)
        object.__setattr__(self, "coupling_prefactor", coupling)
        object.__setattr__(self, "radial_prefactor", radial)
        object.__setattr__(self, "bilinear_terms", bilinear)
        object.__setattr__(self, "mass_terms", mass)
        object.__setattr__(self, "mass_quartic_terms", mass_quartic)


@dataclass(frozen=True)
class PSTFChannelRadialGridContractionResult:
    """Radial-grid contraction result for one output particle-1 channel."""

    C_modes: np.ndarray
    p4_energies: np.ndarray
    p4_left_indices: np.ndarray
    p4_right_weights: np.ndarray
    valid_radial_mask: np.ndarray
    radial_contract: str = _RADIAL_CONTRACT

    def __post_init__(self) -> None:
        modes = np.asarray(self.C_modes, dtype=float)
        p4 = np.asarray(self.p4_energies, dtype=float)
        left = np.asarray(self.p4_left_indices, dtype=int)
        right_weights = np.asarray(self.p4_right_weights, dtype=float)
        valid = np.asarray(self.valid_radial_mask, dtype=bool)
        if modes.ndim != 2:
            raise ValueError("C_modes must have shape (n_p1, n_out_modes).")
        if p4.ndim != 3:
            raise ValueError("p4_energies must have shape (n_p1, n_p2, n_p3).")
        if left.shape != p4.shape or right_weights.shape != p4.shape or valid.shape != p4.shape:
            raise ValueError("p4 interpolation metadata must match p4_energies shape.")
        if modes.shape[0] != p4.shape[0]:
            raise ValueError("C_modes p1 axis must match p4_energies.")
        if not np.all(np.isfinite(modes)) or not np.all(np.isfinite(p4)):
            raise ValueError("radial contraction arrays must contain only finite values.")
        if not np.all(np.isfinite(right_weights)):
            raise ValueError("p4_right_weights must contain only finite values.")
        if np.any((right_weights < 0.0) | (right_weights > 1.0)):
            raise ValueError("p4_right_weights must lie in [0, 1].")
        if not self.radial_contract:
            raise ValueError("radial_contract must be a non-empty string.")
        object.__setattr__(self, "C_modes", modes)
        object.__setattr__(self, "p4_energies", p4)
        object.__setattr__(self, "p4_left_indices", left)
        object.__setattr__(self, "p4_right_weights", right_weights)
        object.__setattr__(self, "valid_radial_mask", valid)


@dataclass(frozen=True)
class PSTFRadialInvariantPrefactorConfig:
    """Invariant radial-measure prefactor for a 2-to-2 channel."""

    symmetry_factor: float = 1.0
    g2: float = 1.0
    g3: float = 1.0
    g4: float = 1.0
    hbar: float = 1.0
    c_light: float = 1.0

    def __post_init__(self) -> None:
        symmetry = float(self.symmetry_factor)
        g2 = float(self.g2)
        g3 = float(self.g3)
        g4 = float(self.g4)
        hbar = float(self.hbar)
        c = float(self.c_light)
        if any(not np.isfinite(value) for value in (symmetry, g2, g3, g4, hbar, c)):
            raise ValueError("radial prefactor parameters must be finite.")
        if symmetry < 0.0 or g2 < 0.0 or g3 < 0.0 or g4 < 0.0:
            raise ValueError("symmetry_factor and degeneracies must be non-negative.")
        if hbar <= 0.0 or c <= 0.0:
            raise ValueError("hbar and c_light must be positive.")
        object.__setattr__(self, "symmetry_factor", symmetry)
        object.__setattr__(self, "g2", g2)
        object.__setattr__(self, "g3", g3)
        object.__setattr__(self, "g4", g4)
        object.__setattr__(self, "hbar", hbar)
        object.__setattr__(self, "c_light", c)


@dataclass(frozen=True)
class PSTFRadialChannelKernelGrid:
    """Precomputed radial grid of channel-specific PSTF ``K`` tensors."""

    K_by_monomial: Mapping[str, np.ndarray]
    p1_energies: np.ndarray
    p2_energies: np.ndarray
    p3_energies: np.ndarray
    p4_energy_grid: np.ndarray
    p1_momenta: np.ndarray
    p2_momenta: np.ndarray
    p3_momenta: np.ndarray
    p4_momentum_grid: np.ndarray
    q2_weights: np.ndarray
    q3_weights: np.ndarray
    p4_energies: np.ndarray
    p4_momenta: np.ndarray
    p4_left_indices: np.ndarray
    p4_right_weights: np.ndarray
    valid_radial_mask: np.ndarray
    radial_prefactors: np.ndarray
    prefactor_config: PSTFRadialInvariantPrefactorConfig
    radial_quadrature_weights: np.ndarray | None = None
    kernel_grid_contract: str = _RADIAL_GRID_CONTRACT

    def __post_init__(self) -> None:
        p4 = np.asarray(self.p4_energies, dtype=float)
        p4_mom = np.asarray(self.p4_momenta, dtype=float)
        left = np.asarray(self.p4_left_indices, dtype=int)
        right_weights = np.asarray(self.p4_right_weights, dtype=float)
        valid = np.asarray(self.valid_radial_mask, dtype=bool)
        radial_prefactors = np.asarray(self.radial_prefactors, dtype=float)
        if p4.ndim != 3:
            raise ValueError("p4_energies must have shape (n_p1, n_p2, n_p3).")
        if (
            p4_mom.shape != p4.shape
            or left.shape != p4.shape
            or right_weights.shape != p4.shape
            or valid.shape != p4.shape
            or radial_prefactors.shape != p4.shape
        ):
            raise ValueError("radial kernel-grid metadata must match p4_energies shape.")
        e1 = _radial_grid(self.p1_energies, name="p1_energies", size=p4.shape[0])
        e2 = _radial_grid(self.p2_energies, name="p2_energies", size=p4.shape[1])
        e3 = _radial_grid(self.p3_energies, name="p3_energies", size=p4.shape[2])
        e4 = _strictly_increasing_grid(self.p4_energy_grid, name="p4_energy_grid")
        p1 = _radial_grid(self.p1_momenta, name="p1_momenta", size=p4.shape[0])
        p2 = _radial_grid(self.p2_momenta, name="p2_momenta", size=p4.shape[1])
        p3 = _radial_grid(self.p3_momenta, name="p3_momenta", size=p4.shape[2])
        p4_grid = _radial_grid(self.p4_momentum_grid, name="p4_momentum_grid", size=e4.size)
        q2 = _radial_grid(self.q2_weights, name="q2_weights", size=p4.shape[1])
        q3 = _radial_grid(self.q3_weights, name="q3_weights", size=p4.shape[2])
        if np.any(p1 < 0.0) or np.any(p2 < 0.0) or np.any(p3 < 0.0) or np.any(p4_grid < 0.0):
            raise ValueError("radial momenta must be non-negative.")
        if np.any(q2 < 0.0) or np.any(q3 < 0.0):
            raise ValueError("q2_weights and q3_weights must be non-negative.")
        if not np.all(np.isfinite(p4)) or not np.all(np.isfinite(p4_mom)):
            raise ValueError("p4 radial metadata must contain only finite values.")
        if not np.all(np.isfinite(right_weights)) or not np.all(np.isfinite(radial_prefactors)):
            raise ValueError("radial weights and prefactors must contain only finite values.")
        if np.any((right_weights < 0.0) | (right_weights > 1.0)):
            raise ValueError("p4_right_weights must lie in [0, 1].")
        if np.any(radial_prefactors < 0.0):
            raise ValueError("radial_prefactors must be non-negative.")
        if self.radial_quadrature_weights is None:
            radial_quadrature = valid.astype(float) * q2[None, :, None] * q3[None, None, :]
        else:
            radial_quadrature = np.asarray(self.radial_quadrature_weights, dtype=float)
            if radial_quadrature.shape != p4.shape:
                raise ValueError("radial_quadrature_weights must match p4_energies shape.")
            if not np.all(np.isfinite(radial_quadrature)) or np.any(radial_quadrature < 0.0):
                raise ValueError("radial_quadrature_weights must contain only finite non-negative values.")
        kernels = _validate_radial_kernel_tensors(
            self.K_by_monomial,
            n1=p4.shape[0],
            n2=p4.shape[1],
            n3=p4.shape[2],
            n_modes=np.asarray(next(iter(self.K_by_monomial.values()))).shape[3],
        )
        prefactor = (
            self.prefactor_config
            if isinstance(self.prefactor_config, PSTFRadialInvariantPrefactorConfig)
            else PSTFRadialInvariantPrefactorConfig(**self.prefactor_config)
        )
        if not self.kernel_grid_contract:
            raise ValueError("kernel_grid_contract must be a non-empty string.")
        object.__setattr__(self, "K_by_monomial", kernels)
        object.__setattr__(self, "p1_energies", e1)
        object.__setattr__(self, "p2_energies", e2)
        object.__setattr__(self, "p3_energies", e3)
        object.__setattr__(self, "p4_energy_grid", e4)
        object.__setattr__(self, "p1_momenta", p1)
        object.__setattr__(self, "p2_momenta", p2)
        object.__setattr__(self, "p3_momenta", p3)
        object.__setattr__(self, "p4_momentum_grid", p4_grid)
        object.__setattr__(self, "q2_weights", q2)
        object.__setattr__(self, "q3_weights", q3)
        object.__setattr__(self, "p4_energies", p4)
        object.__setattr__(self, "p4_momenta", p4_mom)
        object.__setattr__(self, "p4_left_indices", left)
        object.__setattr__(self, "p4_right_weights", right_weights)
        object.__setattr__(self, "valid_radial_mask", valid)
        object.__setattr__(self, "radial_prefactors", radial_prefactors)
        object.__setattr__(self, "prefactor_config", prefactor)
        object.__setattr__(self, "radial_quadrature_weights", radial_quadrature)


@dataclass(frozen=True)
class PSTFRadialChannelKernelGridBatch:
    """Process-leading stack of compatible PSTF radial channel grids."""

    grids: tuple[PSTFRadialChannelKernelGrid, ...]
    K_by_monomial: Mapping[str, np.ndarray]
    p4_energies: np.ndarray
    p4_left_indices: np.ndarray
    p4_right_weights: np.ndarray
    valid_radial_mask: np.ndarray
    radial_quadrature_weights: np.ndarray
    p4_grid_size: int
    stacked_kernel_nbytes: int
    batch_contract: str = _RADIAL_GRID_BATCH_CONTRACT

    def __post_init__(self) -> None:
        grids = tuple(self.grids)
        if not grids:
            raise ValueError("grids must contain at least one PSTFRadialChannelKernelGrid.")
        if any(not isinstance(grid, PSTFRadialChannelKernelGrid) for grid in grids):
            raise ValueError("grids must contain only PSTFRadialChannelKernelGrid instances.")
        p4 = np.asarray(self.p4_energies, dtype=float)
        left = np.asarray(self.p4_left_indices, dtype=int)
        right_weights = np.asarray(self.p4_right_weights, dtype=float)
        valid = np.asarray(self.valid_radial_mask, dtype=bool)
        radial_quadrature = np.asarray(self.radial_quadrature_weights, dtype=float)
        expected_radial_shape = (len(grids),) + grids[0].p4_energies.shape
        if p4.shape != expected_radial_shape:
            raise ValueError("p4_energies must have shape (n_process, n_p1, n_p2, n_p3).")
        if (
            left.shape != expected_radial_shape
            or right_weights.shape != expected_radial_shape
            or valid.shape != expected_radial_shape
            or radial_quadrature.shape != expected_radial_shape
        ):
            raise ValueError("batched radial metadata arrays must share p4_energies shape.")
        if not np.all(np.isfinite(p4)) or not np.all(np.isfinite(right_weights)):
            raise ValueError("batched p4 metadata must contain only finite values.")
        if np.any((right_weights < 0.0) | (right_weights > 1.0)):
            raise ValueError("batched p4_right_weights must lie in [0, 1].")
        if not np.all(np.isfinite(radial_quadrature)) or np.any(radial_quadrature < 0.0):
            raise ValueError("batched radial_quadrature_weights must contain only finite non-negative values.")
        p4_grid_size = int(self.p4_grid_size)
        if p4_grid_size < 2:
            raise ValueError("p4_grid_size must be at least two.")
        if np.any((left < 0) | (left >= p4_grid_size - 1)):
            raise ValueError("batched p4_left_indices are incompatible with p4_grid_size.")
        kernels = {
            str(key): np.asarray(value, dtype=float)
            for key, value in self.K_by_monomial.items()
        }
        if tuple(kernels) != COLLISION_STATISTICAL_MONOMIALS:
            raise ValueError("K_by_monomial must follow COLLISION_STATISTICAL_MONOMIALS order.")
        sample = grids[0].K_by_monomial["34"]
        if sample.ndim != 6:
            raise ValueError("radial K['34'] tensors must have rank 6 before batching.")
        n1, n2, n3, n_modes = sample.shape[:4]
        expected_low_rank = (len(grids), n1, n2, n3, n_modes, n_modes, n_modes)
        expected_high_rank = expected_low_rank + (n_modes,)
        for key in ("34", "12"):
            if kernels[key].shape != expected_low_rank:
                raise ValueError(f"K_by_monomial[{key!r}] has the wrong batched shape.")
        for key in ("123", "124", "134", "234"):
            if kernels[key].shape != expected_high_rank:
                raise ValueError(f"K_by_monomial[{key!r}] has the wrong batched shape.")
        if any(not np.all(np.isfinite(value)) for value in kernels.values()):
            raise ValueError("batched K tensors must contain only finite values.")
        stacked_nbytes = int(self.stacked_kernel_nbytes)
        if stacked_nbytes < 0:
            raise ValueError("stacked_kernel_nbytes must be non-negative.")
        if not self.batch_contract:
            raise ValueError("batch_contract must be a non-empty string.")
        object.__setattr__(self, "grids", grids)
        object.__setattr__(self, "K_by_monomial", kernels)
        object.__setattr__(self, "p4_energies", p4)
        object.__setattr__(self, "p4_left_indices", left)
        object.__setattr__(self, "p4_right_weights", right_weights)
        object.__setattr__(self, "valid_radial_mask", valid)
        object.__setattr__(self, "radial_quadrature_weights", radial_quadrature)
        object.__setattr__(self, "p4_grid_size", p4_grid_size)
        object.__setattr__(self, "stacked_kernel_nbytes", stacked_nbytes)


def _validate_basis_and_weights(
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    basis = np.asarray(basis_matrix, dtype=float)
    weights = np.asarray(angular_weights, dtype=float)
    if basis.ndim != 2:
        raise ValueError("basis_matrix must have shape (n_modes, n_angles).")
    if weights.shape != (basis.shape[1],):
        raise ValueError("angular_weights must match the basis angular axis.")
    if basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError("basis_matrix must have non-empty mode and angular axes.")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(weights)):
        raise ValueError("basis_matrix and angular_weights must contain only finite values.")
    if np.any(weights <= 0.0):
        raise ValueError("angular_weights must be strictly positive.")
    norm = float(np.sum(weights))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("angular_weights must have a positive finite sum.")
    return basis, weights, norm


def _weighted_projection_matrix(basis: np.ndarray, weights: np.ndarray, norm: float) -> tuple[np.ndarray, float]:
    gram = np.einsum("ma,na,a->mn", basis, basis, weights) / norm
    gram_condition = float(np.linalg.cond(gram))
    if not np.isfinite(gram_condition):
        raise ValueError("weighted PSTF Gram matrix is singular.")
    weighted_basis = basis * (weights / norm)[None, :]
    try:
        projection = np.linalg.solve(gram, weighted_basis)
    except np.linalg.LinAlgError as exc:
        raise ValueError("weighted PSTF Gram matrix is singular.") from exc
    return projection, gram_condition


def _ordered_monomial_kernels(
    monomial_angular_kernels: Mapping[str, np.ndarray] | None,
    *,
    n_angles: int,
) -> dict[str, np.ndarray]:
    if monomial_angular_kernels is None:
        return {key: np.ones(n_angles, dtype=float) for key in COLLISION_STATISTICAL_MONOMIALS}
    unknown = set(str(key) for key in monomial_angular_kernels) - set(COLLISION_STATISTICAL_MONOMIALS)
    if unknown:
        raise ValueError(f"unknown monomial angular kernel keys: {sorted(unknown)}")
    kernels: dict[str, np.ndarray] = {}
    for key in COLLISION_STATISTICAL_MONOMIALS:
        value = monomial_angular_kernels.get(key, np.ones(n_angles, dtype=float))
        kernel = np.asarray(value, dtype=float)
        if kernel.shape != (n_angles,):
            raise ValueError(f"monomial_angular_kernels[{key!r}] must have shape (n_angles,).")
        if not np.all(np.isfinite(kernel)):
            raise ValueError(f"monomial_angular_kernels[{key!r}] must contain only finite values.")
        kernels[key] = kernel
    return kernels


def build_local_pstf_statistical_kernel_table(
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    monomial_angular_kernels: Mapping[str, np.ndarray] | None = None,
) -> LocalPSTFStatisticalKernelTable:
    """Build local angular PSTF tensors for the six statistical monomials.

    The returned tensors implement the angular projection part of the
    ``34, 12, 123, 124, 134, 234`` gain-loss polynomial.  Optional
    monomial-specific angular kernels can encode simple deterministic angular
    weights, but full HM matrix elements and momentum-conservation kernels are
    intentionally not represented here.
    """

    basis, weights, norm = _validate_basis_and_weights(basis_matrix, angular_weights)
    projection, gram_condition = _weighted_projection_matrix(basis, weights, norm)
    kernels = _ordered_monomial_kernels(monomial_angular_kernels, n_angles=basis.shape[1])
    tensors: dict[str, np.ndarray] = {}
    for key in COLLISION_STATISTICAL_MONOMIALS:
        projector = projection * kernels[key][None, :]
        if len(key) == 2:
            tensors[key] = np.einsum("oa,ba,ca->obc", projector, basis, basis)
        else:
            tensors[key] = np.einsum("oa,ba,ca,da->obcd", projector, basis, basis, basis)
    return LocalPSTFStatisticalKernelTable(
        basis_matrix=basis,
        angular_weights=weights,
        projection_matrix=projection,
        monomial_tensors=tensors,
        monomial_angular_kernels=kernels,
        gram_condition=gram_condition,
    )


def _validate_four_particle_inputs(
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    momentum_delta_weights: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    if len(basis_by_particle) != 4 or len(angular_weights_by_particle) != 4 or len(direction_vectors_by_particle) != 4:
        raise ValueError("basis, weights, and direction inputs must each contain four particles.")
    basis_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []
    direction_list: list[np.ndarray] = []
    for idx in range(4):
        basis, weights, _norm = _validate_basis_and_weights(
            basis_by_particle[idx],
            angular_weights_by_particle[idx],
        )
        directions = np.asarray(direction_vectors_by_particle[idx], dtype=float)
        if directions.shape != (basis.shape[1], 3):
            raise ValueError("direction vectors must have shape (n_angles, 3) for each particle.")
        if not np.all(np.isfinite(directions)):
            raise ValueError("direction vectors must contain only finite values.")
        direction_norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(direction_norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("direction vectors must be unit 3-vectors.")
        basis_list.append(basis)
        weight_list.append(weights)
        direction_list.append(directions)
    delta = np.asarray(momentum_delta_weights, dtype=float)
    expected_shape = tuple(basis.shape[1] for basis in basis_list)
    if delta.shape != expected_shape:
        raise ValueError("momentum_delta_weights must have shape (n_a1, n_a2, n_a3, n_a4).")
    if not np.all(np.isfinite(delta)):
        raise ValueError("momentum_delta_weights must contain only finite values.")
    projection, _cond = _weighted_projection_matrix(
        basis_list[0],
        weight_list[0],
        float(np.sum(weight_list[0])),
    )
    return (
        (basis_list[0], basis_list[1], basis_list[2], basis_list[3]),
        (weight_list[0], weight_list[1], weight_list[2], weight_list[3]),
        (direction_list[0], direction_list[1], direction_list[2], direction_list[3]),
        delta,
        projection,
    )


def _validate_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("mu pair entries must contain two particle indices.")
    i, j = int(pair[0]), int(pair[1])
    if i == j or i < 1 or i > 4 or j < 1 or j > 4:
        raise ValueError("mu pair particle indices must be distinct values in 1..4.")
    return (i, j)


def _canonical_pair(pair: tuple[int, int]) -> tuple[int, int]:
    i, j = _validate_pair(tuple(pair))
    return (i, j) if i < j else (j, i)


def _canonical_pair_pair(
    pair_pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(pair_pair) != 2:
        raise ValueError("mu*mu entries must contain two mu pairs.")
    return (_canonical_pair(tuple(pair_pair[0])), _canonical_pair(tuple(pair_pair[1])))


def _lookup_mumu(
    table: UniversalPSTFGeometricKernelTable,
    first_pair: tuple[int, int],
    second_pair: tuple[int, int],
) -> Mapping[str, np.ndarray]:
    pair_pair = (_canonical_pair(first_pair), _canonical_pair(second_pair))
    if pair_pair in table.G_mumu:
        return table.G_mumu[pair_pair]
    reversed_pair_pair = (pair_pair[1], pair_pair[0])
    if reversed_pair_pair in table.G_mumu:
        return table.G_mumu[reversed_pair_pair]
    raise ValueError(f"G_mumu is missing required pair product {pair_pair!r}.")


def _ordered_pairs(pairs: tuple[tuple[int, int], ...] | None) -> tuple[tuple[int, int], ...]:
    raw_pairs = _PARTICLE_PAIRS if pairs is None else tuple(pairs)
    out: list[tuple[int, int]] = []
    for pair in raw_pairs:
        canonical = _canonical_pair(tuple(pair))
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


def _ordered_pair_pairs(
    pair_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if pair_pairs is None:
        return ()
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pair_pair in pair_pairs:
        canonical = _canonical_pair_pair(tuple(pair_pair))
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


def _geometric_tensor_shape(
    monomial: str,
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, ...]:
    digits = tuple(int(char) for char in monomial)
    return (basis_by_particle[0].shape[0],) + tuple(
        basis_by_particle[digit - 1].shape[0]
        for digit in digits
    )


def _zero_geometric_tensors(
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        monomial: np.zeros(_geometric_tensor_shape(monomial, basis_by_particle), dtype=float)
        for monomial in COLLISION_STATISTICAL_MONOMIALS
    }


def _validate_geometric_tensor_mapping(
    tensors: Mapping[str, np.ndarray],
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    name: str,
) -> None:
    if tuple(tensors) != COLLISION_STATISTICAL_MONOMIALS:
        raise ValueError(f"{name} must follow COLLISION_STATISTICAL_MONOMIALS order.")
    for monomial, tensor in tensors.items():
        expected = _geometric_tensor_shape(monomial, basis_by_particle)
        if tensor.shape != expected:
            raise ValueError(f"{name}[{monomial!r}] has shape {tensor.shape}, expected {expected}.")
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"{name}[{monomial!r}] must contain only finite values.")


def _mu_angle_grid(
    pair: tuple[int, int],
    direction_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    i, j = _canonical_pair(pair)
    mu_pair = direction_tuple[i - 1] @ direction_tuple[j - 1].T
    shape = [1, 1, 1, 1]
    shape[i - 1] = direction_tuple[i - 1].shape[0]
    shape[j - 1] = direction_tuple[j - 1].shape[0]
    return mu_pair.reshape(tuple(shape))


def _contract_geometric_monomial(
    angular_weight: np.ndarray,
    projection: np.ndarray,
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monomial: str,
) -> np.ndarray:
    angle_labels = ("i", "j", "k", "l")
    mode_labels = ("a", "b", "c", "d", "e")
    digits = tuple(int(char) for char in monomial)
    operands: list[np.ndarray] = [angular_weight, projection]
    terms = ["ijkl", f"{mode_labels[0]}{angle_labels[0]}"]
    output = [mode_labels[0]]
    for pos, digit in enumerate(digits, start=1):
        operands.append(basis_by_particle[digit - 1])
        terms.append(f"{mode_labels[pos]}{angle_labels[digit - 1]}")
        output.append(mode_labels[pos])
    expression = ",".join(terms) + "->" + "".join(output)
    return np.einsum(expression, *operands)


def build_universal_pstf_geometric_kernel_table(
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    momentum_delta_weights: np.ndarray,
    *,
    mu_pairs: tuple[tuple[int, int], ...] | None = None,
    mumu_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None = None,
) -> UniversalPSTFGeometricKernelTable:
    """Build universal PSTF geometry tables for ``1``, ``mu_ij``, and ``mu_ij mu_kl``.

    The caller supplies the deterministic angular quadrature and the
    momentum-conservation delta weights for one radial tuple.  The result is
    independent of the channel coefficients and can be combined later with HM
    matrix-element descriptors.
    """

    basis_tuple, weight_tuple, direction_tuple, delta, projection = _validate_four_particle_inputs(
        basis_by_particle,
        angular_weights_by_particle,
        direction_vectors_by_particle,
        momentum_delta_weights,
    )
    requested_mu_pairs = _ordered_pairs(mu_pairs)
    requested_mumu_pairs = _ordered_pair_pairs(mumu_pairs)
    G0 = _zero_geometric_tensors(basis_tuple)
    G_mu = {pair: _zero_geometric_tensors(basis_tuple) for pair in requested_mu_pairs}
    G_mumu = {pair_pair: _zero_geometric_tensors(basis_tuple) for pair_pair in requested_mumu_pairs}

    base_weight = (
        delta
        * weight_tuple[1][None, :, None, None]
        * weight_tuple[2][None, None, :, None]
        * weight_tuple[3][None, None, None, :]
    )
    mu_grid_by_pair = {
        pair: _mu_angle_grid(pair, direction_tuple)
        for pair in set(requested_mu_pairs).union(
            pair for pair_pair in requested_mumu_pairs for pair in pair_pair
        )
    }
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        G0[monomial] = _contract_geometric_monomial(
            base_weight,
            projection,
            basis_tuple,
            monomial,
        )
        for pair in requested_mu_pairs:
            G_mu[pair][monomial] = _contract_geometric_monomial(
                base_weight * mu_grid_by_pair[pair],
                projection,
                basis_tuple,
                monomial,
            )
        for pair_pair in requested_mumu_pairs:
            G_mumu[pair_pair][monomial] = _contract_geometric_monomial(
                base_weight * mu_grid_by_pair[pair_pair[0]] * mu_grid_by_pair[pair_pair[1]],
                projection,
                basis_tuple,
                monomial,
            )

    return UniversalPSTFGeometricKernelTable(
        basis_by_particle=basis_tuple,
        angular_weights_by_particle=weight_tuple,
        direction_vectors_by_particle=direction_tuple,
        momentum_delta_weights=delta,
        output_projection_matrix=projection,
        G0=G0,
        G_mu=G_mu,
        G_mumu=G_mumu,
    )


def _positive_four_vector(values: tuple[float, float, float, float], *, name: str) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values.")
    out = tuple(float(value) for value in values)
    if any((not np.isfinite(value)) or value <= 0.0 for value in out):
        raise ValueError(f"{name} must contain positive finite values.")
    return out  # type: ignore[return-value]


def _nonnegative_four_vector(values: tuple[float, float, float, float], *, name: str) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values.")
    out = tuple(float(value) for value in values)
    if any((not np.isfinite(value)) or value < 0.0 for value in out):
        raise ValueError(f"{name} must contain non-negative finite values.")
    return out  # type: ignore[return-value]


def _lookup_mumu_canonical(
    table: UniversalPSTFGeometricKernelTable,
    first_pair: tuple[int, int],
    second_pair: tuple[int, int],
) -> Mapping[str, np.ndarray]:
    pair_pair = (first_pair, second_pair)
    if pair_pair in table.G_mumu:
        return table.G_mumu[pair_pair]
    reversed_pair_pair = (second_pair, first_pair)
    if reversed_pair_pair in table.G_mumu:
        return table.G_mumu[reversed_pair_pair]
    raise ValueError(f"G_mumu is missing required pair product {pair_pair!r}.")


def _assemble_channel_kernel_arrays(
    geometric_table: UniversalPSTFGeometricKernelTable,
    *,
    energies: tuple[float, float, float, float],
    momenta: tuple[float, float, float, float],
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...],
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...],
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...],
    electron_mass_momentum: float,
    coupling_prefactor: float,
    radial_prefactor: float,
    c_light: float,
) -> dict[str, np.ndarray]:
    energy = energies
    momentum = momenta
    c2 = c_light * c_light
    c4 = c2 * c2
    electron_mass2 = electron_mass_momentum * electron_mass_momentum
    electron_mass4 = electron_mass2 * electron_mass2
    scale = radial_prefactor * coupling_prefactor
    bilinear_coeffs: list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            float,
            float,
            float,
            float,
        ]
    ] = []
    for term in bilinear_terms:
        first = term.first_pair
        second = term.second_pair
        i, j = first[0] - 1, first[1] - 1
        k, l = second[0] - 1, second[1] - 1
        bilinear_coeffs.append(
            (
                first,
                second,
                term.eta * energy[i] * energy[j] * energy[k] * energy[l] / c4,
                -term.eta * energy[i] * energy[j] * momentum[k] * momentum[l] / c2,
                -term.eta * momentum[i] * momentum[j] * energy[k] * energy[l] / c2,
                term.eta * momentum[i] * momentum[j] * momentum[k] * momentum[l],
            )
        )
    mass_coeffs: list[tuple[tuple[int, int], float, float]] = []
    for term in mass_terms:
        pair = term.pair
        i, j = pair[0] - 1, pair[1] - 1
        zeta_m2 = term.zeta * electron_mass2
        mass_coeffs.append(
            (
                pair,
                zeta_m2 * energy[i] * energy[j] / c2,
                -zeta_m2 * momentum[i] * momentum[j],
            )
        )
    mass_quartic_coeff = sum(term.zeta for term in mass_quartic_terms) * electron_mass4
    K_by_monomial: dict[str, np.ndarray] = {}
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        G0 = geometric_table.G0[monomial]
        kernel = np.zeros_like(G0, dtype=float)
        for first, second, c0, c_second, c_first, c_mumu in bilinear_coeffs:
            G_mumu = _lookup_mumu_canonical(geometric_table, first, second)
            G_first = geometric_table.G_mu.get(first)
            G_second = geometric_table.G_mu.get(second)
            if G_first is None:
                raise ValueError(f"G_mu is missing required pair {first!r}.")
            if G_second is None:
                raise ValueError(f"G_mu is missing required pair {second!r}.")
            kernel += c0 * G0
            kernel += c_second * G_second[monomial]
            kernel += c_first * G_first[monomial]
            kernel += c_mumu * G_mumu[monomial]
        for pair, c0, c_mu in mass_coeffs:
            G_pair = geometric_table.G_mu.get(pair)
            if G_pair is None:
                raise ValueError(f"G_mu is missing required pair {pair!r}.")
            kernel += c0 * G0
            kernel += c_mu * G_pair[monomial]
        if mass_quartic_coeff != 0.0:
            kernel += mass_quartic_coeff * G0
        K_by_monomial[monomial] = scale * kernel
    return K_by_monomial


def _radial_kernel_broadcast(shape: tuple[int, ...], ndim: int) -> tuple[slice | None, ...]:
    return (slice(None),) * len(shape) + (None,) * int(ndim)


def _assemble_static_delta_channel_kernel_grids(
    geometric_table: UniversalPSTFGeometricKernelTable,
    *,
    e1: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray,
    e4: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    radial_prefactors: np.ndarray,
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...],
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...],
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...],
    electron_mass_momentum: float,
    coupling_prefactor: float,
    c_light: float,
) -> dict[str, np.ndarray]:
    radial_shape = tuple(int(value) for value in np.asarray(e4).shape)
    energy = (
        np.asarray(e1, dtype=float)[:, None, None],
        np.asarray(e2, dtype=float)[None, :, None],
        np.asarray(e3, dtype=float)[None, None, :],
        np.asarray(e4, dtype=float),
    )
    momentum = (
        np.asarray(p1, dtype=float)[:, None, None],
        np.asarray(p2, dtype=float)[None, :, None],
        np.asarray(p3, dtype=float)[None, None, :],
        np.asarray(p4, dtype=float),
    )
    c2 = float(c_light) * float(c_light)
    c4 = c2 * c2
    electron_mass2 = float(electron_mass_momentum) * float(electron_mass_momentum)
    electron_mass4 = electron_mass2 * electron_mass2
    radial_scale = np.asarray(radial_prefactors, dtype=float) * float(coupling_prefactor)
    mass_quartic_coeff = (
        sum(term.zeta for term in mass_quartic_terms) * electron_mass4
    )
    radial_components: list[tuple[np.ndarray, Mapping[str, np.ndarray]]] = []
    for term in bilinear_terms:
        first = term.first_pair
        second = term.second_pair
        i, j = first[0] - 1, first[1] - 1
        k, l = second[0] - 1, second[1] - 1
        G_mumu = _lookup_mumu_canonical(geometric_table, first, second)
        G_first = geometric_table.G_mu.get(first)
        G_second = geometric_table.G_mu.get(second)
        if G_first is None:
            raise ValueError(f"G_mu is missing required pair {first!r}.")
        if G_second is None:
            raise ValueError(f"G_mu is missing required pair {second!r}.")
        radial_components.extend(
            (
                (
                    np.asarray(
                        term.eta * energy[i] * energy[j] * energy[k] * energy[l] / c4,
                        dtype=float,
                    ),
                    geometric_table.G0,
                ),
                (
                    np.asarray(
                        -term.eta * energy[i] * energy[j] * momentum[k] * momentum[l] / c2,
                        dtype=float,
                    ),
                    G_second,
                ),
                (
                    np.asarray(
                        -term.eta * momentum[i] * momentum[j] * energy[k] * energy[l] / c2,
                        dtype=float,
                    ),
                    G_first,
                ),
                (
                    np.asarray(
                        term.eta * momentum[i] * momentum[j] * momentum[k] * momentum[l],
                        dtype=float,
                    ),
                    G_mumu,
                ),
            )
        )
    for term in mass_terms:
        pair = term.pair
        i, j = pair[0] - 1, pair[1] - 1
        G_pair = geometric_table.G_mu.get(pair)
        if G_pair is None:
            raise ValueError(f"G_mu is missing required pair {pair!r}.")
        zeta_m2 = term.zeta * electron_mass2
        radial_components.extend(
            (
                (
                    np.asarray(zeta_m2 * energy[i] * energy[j] / c2, dtype=float),
                    geometric_table.G0,
                ),
                (
                    np.asarray(-zeta_m2 * momentum[i] * momentum[j], dtype=float),
                    G_pair,
                ),
            )
        )
    if mass_quartic_coeff != 0.0:
        radial_components.append(
            (np.asarray(mass_quartic_coeff, dtype=float), geometric_table.G0)
        )
    coefficient_stack = (
        None
        if not radial_components
        else np.stack(
            [
                np.full(radial_shape, float(coefficient), dtype=float)
                if coefficient.ndim == 0
                else np.broadcast_to(coefficient, radial_shape)
                for coefficient, _tables in radial_components
            ],
            axis=0,
        )
    )
    K_by_monomial: dict[str, np.ndarray] = {}
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        G0 = geometric_table.G0[monomial]
        broadcast = _radial_kernel_broadcast(radial_shape, G0.ndim)
        if coefficient_stack is None:
            kernel = np.zeros(radial_shape + tuple(G0.shape), dtype=float)
        else:
            table_stack = np.stack(
                [tables[monomial] for _coefficient, tables in radial_components],
                axis=0,
            )
            kernel = np.tensordot(coefficient_stack, table_stack, axes=(0, 0))
        kernel *= radial_scale[broadcast]
        K_by_monomial[monomial] = kernel
    return K_by_monomial


def build_pstf_channel_kernel_table(
    geometric_table: UniversalPSTFGeometricKernelTable,
    *,
    energies: tuple[float, float, float, float],
    momenta: tuple[float, float, float, float],
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...] = (),
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...] = (),
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...] = (),
    electron_mass_momentum: float = 0.0,
    coupling_prefactor: float = 1.0,
    radial_prefactor: float = 1.0,
    c_light: float = 1.0,
) -> PSTFChannelKernelTable:
    """Combine universal geometry with HM-style matrix-element descriptors."""

    if not isinstance(geometric_table, UniversalPSTFGeometricKernelTable):
        raise ValueError("geometric_table must be a UniversalPSTFGeometricKernelTable.")
    checked_energies = _positive_four_vector(energies, name="energies")
    checked_momenta = _nonnegative_four_vector(momenta, name="momenta")
    c = float(c_light)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c_light must be positive and finite.")
    electron_mass = float(electron_mass_momentum)
    coupling = float(coupling_prefactor)
    radial = float(radial_prefactor)
    if not np.isfinite(electron_mass) or electron_mass < 0.0:
        raise ValueError("electron_mass_momentum must be non-negative and finite.")
    if not np.isfinite(coupling) or not np.isfinite(radial):
        raise ValueError("coupling_prefactor and radial_prefactor must be finite.")
    checked_bilinear, checked_mass, checked_mass_quartic = _checked_channel_terms(
        bilinear_terms,
        mass_terms,
        mass_quartic_terms,
    )
    K_by_monomial = _assemble_channel_kernel_arrays(
        geometric_table,
        energies=checked_energies,
        momenta=checked_momenta,
        bilinear_terms=checked_bilinear,
        mass_terms=checked_mass,
        mass_quartic_terms=checked_mass_quartic,
        electron_mass_momentum=electron_mass,
        coupling_prefactor=coupling,
        radial_prefactor=radial,
        c_light=c,
    )
    return PSTFChannelKernelTable(
        K_by_monomial=K_by_monomial,
        bilinear_terms=checked_bilinear,
        mass_terms=checked_mass,
        mass_quartic_terms=checked_mass_quartic,
        energies=checked_energies,
        momenta=checked_momenta,
        electron_mass_momentum=electron_mass,
        coupling_prefactor=coupling,
        radial_prefactor=radial,
    )


def _radial_mode_grid(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_radial, n_modes).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty axes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _radial_mode_grid_batch(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (n_process, n_radial, n_modes).")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty axes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _radial_grid(values: np.ndarray, *, name: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D grid.")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must match its radial axis.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _strictly_increasing_grid(values: np.ndarray, *, name: str, size: int | None = None) -> np.ndarray:
    arr = _radial_grid(values, name=name, size=size)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two nodes for interpolation.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _validate_radial_kernel_tensors(
    K_by_monomial: Mapping[str, np.ndarray],
    *,
    n1: int,
    n2: int,
    n3: int,
    n_modes: int,
) -> dict[str, np.ndarray]:
    kernels = {str(key): np.asarray(value, dtype=float) for key, value in K_by_monomial.items()}
    if tuple(kernels) != COLLISION_STATISTICAL_MONOMIALS:
        raise ValueError("K_by_monomial must follow COLLISION_STATISTICAL_MONOMIALS order.")
    expected_rank2 = (n1, n2, n3, n_modes, n_modes, n_modes)
    expected_rank3 = (n1, n2, n3, n_modes, n_modes, n_modes, n_modes)
    for key in ("34", "12"):
        if kernels[key].shape != expected_rank2:
            raise ValueError(f"K_by_monomial[{key!r}] must have shape {expected_rank2}.")
    for key in ("123", "124", "134", "234"):
        if kernels[key].shape != expected_rank3:
            raise ValueError(f"K_by_monomial[{key!r}] must have shape {expected_rank3}.")
    if any(not np.all(np.isfinite(value)) for value in kernels.values()):
        raise ValueError("K_by_monomial tensors must contain only finite values.")
    return kernels


def _p4_linear_interpolation(
    energy: float,
    p4_energy_grid: np.ndarray,
) -> tuple[bool, int, float]:
    if energy < float(p4_energy_grid[0]) or energy > float(p4_energy_grid[-1]):
        return False, 0, 0.0
    if energy == float(p4_energy_grid[-1]):
        return True, p4_energy_grid.size - 2, 1.0
    right = int(np.searchsorted(p4_energy_grid, energy, side="right"))
    left = max(0, right - 1)
    e_left = float(p4_energy_grid[left])
    e_right = float(p4_energy_grid[left + 1])
    weight = (float(energy) - e_left) / (e_right - e_left)
    return True, left, float(weight)


def _checked_channel_terms(
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...],
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...],
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...],
) -> tuple[
    tuple[PSTFBilinearMatrixElementTerm, ...],
    tuple[PSTFMassMatrixElementTerm, ...],
    tuple[PSTFMassQuarticMatrixElementTerm, ...],
]:
    bilinear = tuple(
        term if isinstance(term, PSTFBilinearMatrixElementTerm) else PSTFBilinearMatrixElementTerm(**term)
        for term in bilinear_terms
    )
    mass = tuple(
        term if isinstance(term, PSTFMassMatrixElementTerm) else PSTFMassMatrixElementTerm(**term)
        for term in mass_terms
    )
    mass_quartic = tuple(
        term if isinstance(term, PSTFMassQuarticMatrixElementTerm) else PSTFMassQuarticMatrixElementTerm(**term)
        for term in mass_quartic_terms
    )
    return bilinear, mass, mass_quartic


def _required_mu_tables(
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...],
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[tuple[int, int], tuple[int, int]], ...]]:
    mu_pairs: list[tuple[int, int]] = []
    mumu_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for term in bilinear_terms:
        first = _canonical_pair(term.first_pair)
        second = _canonical_pair(term.second_pair)
        if first not in mu_pairs:
            mu_pairs.append(first)
        if second not in mu_pairs:
            mu_pairs.append(second)
        pair_pair = (first, second)
        if pair_pair not in mumu_pairs and (second, first) not in mumu_pairs:
            mumu_pairs.append(pair_pair)
    for term in mass_terms:
        pair = _canonical_pair(term.pair)
        if pair not in mu_pairs:
            mu_pairs.append(pair)
    return tuple(mu_pairs), tuple(mumu_pairs)


def _momentum_delta_weights_for_tuple(
    momentum_delta_weights: np.ndarray | Callable[[int, int, int, float, float], np.ndarray],
    *,
    i1: int,
    i2: int,
    i3: int,
    e4: float,
    p4: float,
) -> np.ndarray:
    if callable(momentum_delta_weights):
        return np.asarray(momentum_delta_weights(i1, i2, i3, e4, p4), dtype=float)
    delta = np.asarray(momentum_delta_weights, dtype=float)
    if delta.ndim == 4:
        return delta
    if delta.ndim == 7:
        return delta[i1, i2, i3]
    raise ValueError("momentum_delta_weights must have rank 4, rank 7, or be a callable.")


def build_pstf_radial_channel_kernel_grid(
    *,
    basis_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    angular_weights_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    direction_vectors_by_particle: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    momentum_delta_weights: np.ndarray | Callable[[int, int, int, float, float], np.ndarray],
    p1_energies: np.ndarray,
    p2_energies: np.ndarray,
    p3_energies: np.ndarray,
    p4_energy_grid: np.ndarray,
    p1_momenta: np.ndarray,
    p2_momenta: np.ndarray,
    p3_momenta: np.ndarray,
    p4_momentum_grid: np.ndarray,
    q2_weights: np.ndarray,
    q3_weights: np.ndarray,
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...] = (),
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...] = (),
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...] = (),
    electron_mass_momentum: float = 0.0,
    coupling_prefactor: float = 1.0,
    radial_prefactor_config: PSTFRadialInvariantPrefactorConfig | None = None,
    geometric_table_cache: MutableMapping[object, object] | None = None,
    array_key_memo: MutableMapping[int, tuple[np.ndarray, tuple[object, ...]]] | None = None,
) -> PSTFRadialChannelKernelGrid:
    """Build channel ``K`` tensors over a radial ``p1,p2,p3`` grid."""

    if len(basis_by_particle) != 4:
        raise ValueError("basis_by_particle must contain exactly four basis arrays.")
    basis_tuple = tuple(np.asarray(value, dtype=float) for value in basis_by_particle)
    n_modes = basis_tuple[0].shape[0]
    if any(basis.ndim != 2 or basis.shape[0] != n_modes for basis in basis_tuple):
        raise ValueError("all particle basis arrays must have shape (n_modes, n_angles) with a shared mode axis.")
    e1 = _radial_grid(p1_energies, name="p1_energies")
    e2 = _radial_grid(p2_energies, name="p2_energies")
    e3 = _radial_grid(p3_energies, name="p3_energies")
    e4_grid = _strictly_increasing_grid(p4_energy_grid, name="p4_energy_grid")
    p1 = _radial_grid(p1_momenta, name="p1_momenta", size=e1.size)
    p2 = _radial_grid(p2_momenta, name="p2_momenta", size=e2.size)
    p3 = _radial_grid(p3_momenta, name="p3_momenta", size=e3.size)
    p4_grid = _radial_grid(p4_momentum_grid, name="p4_momentum_grid", size=e4_grid.size)
    q2 = _radial_grid(q2_weights, name="q2_weights", size=e2.size)
    q3 = _radial_grid(q3_weights, name="q3_weights", size=e3.size)
    if np.any(e1 <= 0.0) or np.any(e2 <= 0.0) or np.any(e3 <= 0.0) or np.any(e4_grid <= 0.0):
        raise ValueError("radial energies must be positive.")
    if np.any(p1 < 0.0) or np.any(p2 < 0.0) or np.any(p3 < 0.0) or np.any(p4_grid < 0.0):
        raise ValueError("radial momenta must be non-negative.")
    if np.any(q2 < 0.0) or np.any(q3 < 0.0):
        raise ValueError("q2_weights and q3_weights must be non-negative.")
    prefactor = radial_prefactor_config or PSTFRadialInvariantPrefactorConfig()
    if not isinstance(prefactor, PSTFRadialInvariantPrefactorConfig):
        raise ValueError("radial_prefactor_config must be a PSTFRadialInvariantPrefactorConfig.")
    bilinear, mass, mass_quartic = _checked_channel_terms(bilinear_terms, mass_terms, mass_quartic_terms)
    mu_pairs, mumu_pairs = _required_mu_tables(bilinear, mass)
    # Keep this memo call-local/provider-local; reusing it across in-place
    # mutations would intentionally reuse the old content key.
    local_array_key_memo = (
        array_key_memo
        if array_key_memo is not None
        else {}
    )
    geometric_cache_prefix = (
        None
        if geometric_table_cache is None
        else _geometric_table_cache_prefix(
            basis_tuple,
            angular_weights_by_particle,
            direction_vectors_by_particle,
            mu_pairs=mu_pairs,
            mumu_pairs=mumu_pairs,
            memo=local_array_key_memo,
        )
    )

    K_by_monomial: dict[str, np.ndarray] = {}
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        rank = 3 if len(monomial) == 2 else 4
        K_by_monomial[monomial] = np.zeros((e1.size, e2.size, e3.size) + (n_modes,) * rank, dtype=float)
    p4_energies = e1[:, None, None] + e2[None, :, None] - e3[None, None, :]
    raw_left = np.searchsorted(e4_grid, p4_energies, side="right") - 1
    valid = (p4_energies >= e4_grid[0]) & (p4_energies <= e4_grid[-1])
    p4_left = np.clip(raw_left, 0, e4_grid.size - 2).astype(int)
    p4_span = e4_grid[p4_left + 1] - e4_grid[p4_left]
    p4_right_weight = np.where(valid, (p4_energies - e4_grid[p4_left]) / p4_span, 0.0)
    p4_momenta = (1.0 - p4_right_weight) * p4_grid[p4_left] + p4_right_weight * p4_grid[p4_left + 1]
    p4_momenta = np.where(valid, p4_momenta, 0.0)
    tau_hbar = 2.0 * np.pi * prefactor.hbar
    radial_prefactors = (
        prefactor.symmetry_factor
        * tau_hbar**4
        / (2.0 * e1[:, None, None])
        * (prefactor.g2 * p2[None, :, None] ** 2 / (tau_hbar**3 * 2.0 * e2[None, :, None]))
        * (prefactor.g3 * p3[None, None, :] ** 2 / (tau_hbar**3 * 2.0 * e3[None, None, :]))
        * (prefactor.g4 * p4_momenta / (tau_hbar**3 * 2.0 * prefactor.c_light**2))
    )
    radial_prefactors = np.where(valid, radial_prefactors, 0.0)
    static_delta: np.ndarray | None = None
    static_geometric_cache_key: object | None = None
    static_geometric_table: UniversalPSTFGeometricKernelTable | None = None
    if not callable(momentum_delta_weights):
        delta_candidate = np.asarray(momentum_delta_weights, dtype=float)
        if delta_candidate.ndim == 4:
            static_delta = delta_candidate
            static_geometric_cache_key = (
                None
                if geometric_cache_prefix is None
                else (
                    geometric_cache_prefix,
                    _cache_array_key(static_delta, memo=local_array_key_memo),
                )
            )

    if static_delta is not None and bool(np.any(valid)):
        if static_geometric_table is not None:
            geometric = static_geometric_table
        elif static_geometric_cache_key is not None and static_geometric_cache_key in geometric_table_cache:
            geometric = geometric_table_cache[static_geometric_cache_key]
        else:
            geometric = build_universal_pstf_geometric_kernel_table(
                basis_tuple,
                angular_weights_by_particle,
                direction_vectors_by_particle,
                static_delta,
                mu_pairs=mu_pairs,
                mumu_pairs=mumu_pairs,
            )
            static_geometric_table = geometric
            if static_geometric_cache_key is not None:
                geometric_table_cache[static_geometric_cache_key] = geometric
        K_by_monomial = _assemble_static_delta_channel_kernel_grids(
            geometric,
            e1=e1,
            e2=e2,
            e3=e3,
            e4=p4_energies,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4_momenta,
            radial_prefactors=radial_prefactors,
            bilinear_terms=bilinear,
            mass_terms=mass,
            mass_quartic_terms=mass_quartic,
            electron_mass_momentum=electron_mass_momentum,
            coupling_prefactor=coupling_prefactor,
            c_light=prefactor.c_light,
        )
    elif static_delta is None:
        for i1 in range(e1.size):
            for i2 in range(e2.size):
                for i3 in range(e3.size):
                    if not bool(valid[i1, i2, i3]):
                        continue
                    e4 = float(p4_energies[i1, i2, i3])
                    p4_value = float(p4_momenta[i1, i2, i3])
                    radial_prefactor = float(radial_prefactors[i1, i2, i3])
                    delta = _momentum_delta_weights_for_tuple(
                        momentum_delta_weights,
                        i1=i1,
                        i2=i2,
                        i3=i3,
                        e4=e4,
                        p4=p4_value,
                    )
                    geometric_cache_key = (
                        None
                        if geometric_cache_prefix is None
                        else (geometric_cache_prefix, _cache_array_key(delta))
                    )
                    if geometric_cache_key is not None and geometric_cache_key in geometric_table_cache:
                        geometric = geometric_table_cache[geometric_cache_key]
                    else:
                        geometric = build_universal_pstf_geometric_kernel_table(
                            basis_tuple,
                            angular_weights_by_particle,
                            direction_vectors_by_particle,
                            delta,
                            mu_pairs=mu_pairs,
                            mumu_pairs=mumu_pairs,
                        )
                        if geometric_cache_key is not None:
                            geometric_table_cache[geometric_cache_key] = geometric
                    channel_K = _assemble_channel_kernel_arrays(
                        geometric,
                        energies=(float(e1[i1]), float(e2[i2]), float(e3[i3]), e4),
                        momenta=(float(p1[i1]), float(p2[i2]), float(p3[i3]), p4_value),
                        bilinear_terms=bilinear,
                        mass_terms=mass,
                        mass_quartic_terms=mass_quartic,
                        electron_mass_momentum=electron_mass_momentum,
                        coupling_prefactor=coupling_prefactor,
                        radial_prefactor=radial_prefactor,
                        c_light=prefactor.c_light,
                    )
                    for monomial in COLLISION_STATISTICAL_MONOMIALS:
                        K_by_monomial[monomial][i1, i2, i3] = channel_K[monomial]

    radial_quadrature_weights = valid.astype(float) * q2[None, :, None] * q3[None, None, :]

    return PSTFRadialChannelKernelGrid(
        K_by_monomial=K_by_monomial,
        p1_energies=e1,
        p2_energies=e2,
        p3_energies=e3,
        p4_energy_grid=e4_grid,
        p1_momenta=p1,
        p2_momenta=p2,
        p3_momenta=p3,
        p4_momentum_grid=p4_grid,
        q2_weights=q2,
        q3_weights=q3,
        p4_energies=p4_energies,
        p4_momenta=p4_momenta,
        p4_left_indices=p4_left,
        p4_right_weights=p4_right_weight,
        valid_radial_mask=valid,
        radial_prefactors=radial_prefactors,
        prefactor_config=prefactor,
        radial_quadrature_weights=radial_quadrature_weights,
    )


def write_pstf_radial_channel_kernel_grid_npz(
    path: str | Path,
    grid: PSTFRadialChannelKernelGrid,
) -> Path:
    """Write a precomputed radial channel-kernel grid to a portable NPZ cache."""

    if not isinstance(grid, PSTFRadialChannelKernelGrid):
        raise ValueError("grid must be a PSTFRadialChannelKernelGrid.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "cache_contract": np.asarray(_RADIAL_GRID_NPZ_CONTRACT),
        "kernel_grid_contract": np.asarray(grid.kernel_grid_contract),
        "p1_energies": grid.p1_energies,
        "p2_energies": grid.p2_energies,
        "p3_energies": grid.p3_energies,
        "p4_energy_grid": grid.p4_energy_grid,
        "p1_momenta": grid.p1_momenta,
        "p2_momenta": grid.p2_momenta,
        "p3_momenta": grid.p3_momenta,
        "p4_momentum_grid": grid.p4_momentum_grid,
        "q2_weights": grid.q2_weights,
        "q3_weights": grid.q3_weights,
        "p4_energies": grid.p4_energies,
        "p4_momenta": grid.p4_momenta,
        "p4_left_indices": grid.p4_left_indices,
        "p4_right_weights": grid.p4_right_weights,
        "valid_radial_mask": grid.valid_radial_mask,
        "radial_prefactors": grid.radial_prefactors,
        "radial_quadrature_weights": grid.radial_quadrature_weights,
        "prefactor_config": np.asarray(
            [
                grid.prefactor_config.symmetry_factor,
                grid.prefactor_config.g2,
                grid.prefactor_config.g3,
                grid.prefactor_config.g4,
                grid.prefactor_config.hbar,
                grid.prefactor_config.c_light,
            ],
            dtype=float,
        ),
    }
    for monomial in COLLISION_STATISTICAL_MONOMIALS:
        payload[f"K_{monomial}"] = grid.K_by_monomial[monomial]
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return output


def read_pstf_radial_channel_kernel_grid_npz(path: str | Path) -> PSTFRadialChannelKernelGrid:
    """Read a radial channel-kernel grid produced by ``write_*_npz``."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        contract = str(data["cache_contract"].item())
        if contract != _RADIAL_GRID_NPZ_CONTRACT:
            raise ValueError("unsupported radial kernel-grid NPZ cache contract.")
        prefactor_values = np.asarray(data["prefactor_config"], dtype=float)
        if prefactor_values.shape != (6,):
            raise ValueError("radial kernel-grid NPZ cache has invalid prefactor metadata.")
        prefactor = PSTFRadialInvariantPrefactorConfig(
            symmetry_factor=float(prefactor_values[0]),
            g2=float(prefactor_values[1]),
            g3=float(prefactor_values[2]),
            g4=float(prefactor_values[3]),
            hbar=float(prefactor_values[4]),
            c_light=float(prefactor_values[5]),
        )
        K_by_monomial = {
            monomial: np.asarray(data[f"K_{monomial}"], dtype=float)
            for monomial in COLLISION_STATISTICAL_MONOMIALS
        }
        return PSTFRadialChannelKernelGrid(
            K_by_monomial=K_by_monomial,
            p1_energies=np.asarray(data["p1_energies"], dtype=float),
            p2_energies=np.asarray(data["p2_energies"], dtype=float),
            p3_energies=np.asarray(data["p3_energies"], dtype=float),
            p4_energy_grid=np.asarray(data["p4_energy_grid"], dtype=float),
            p1_momenta=np.asarray(data["p1_momenta"], dtype=float),
            p2_momenta=np.asarray(data["p2_momenta"], dtype=float),
            p3_momenta=np.asarray(data["p3_momenta"], dtype=float),
            p4_momentum_grid=np.asarray(data["p4_momentum_grid"], dtype=float),
            q2_weights=np.asarray(data["q2_weights"], dtype=float),
            q3_weights=np.asarray(data["q3_weights"], dtype=float),
            p4_energies=np.asarray(data["p4_energies"], dtype=float),
            p4_momenta=np.asarray(data["p4_momenta"], dtype=float),
            p4_left_indices=np.asarray(data["p4_left_indices"], dtype=int),
            p4_right_weights=np.asarray(data["p4_right_weights"], dtype=float),
            valid_radial_mask=np.asarray(data["valid_radial_mask"], dtype=bool),
            radial_prefactors=np.asarray(data["radial_prefactors"], dtype=float),
            prefactor_config=prefactor,
            radial_quadrature_weights=np.asarray(data["radial_quadrature_weights"], dtype=float),
            kernel_grid_contract=str(data["kernel_grid_contract"].item()),
        )


def build_pstf_radial_channel_kernel_grid_batch(
    grids: Sequence[PSTFRadialChannelKernelGrid],
    *,
    max_kernel_nbytes: int | None = 256 * 1024 * 1024,
) -> PSTFRadialChannelKernelGridBatch | None:
    """Stack compatible radial channel grids for process-batched contraction.

    The batch duplicates the precomputed ``K`` tensors with a process-leading
    axis.  For large ladders callers can keep the memory bound finite; if the
    stacked tensors would exceed that budget this function returns ``None`` so
    the scalar per-process contraction remains available.
    """

    grid_tuple = tuple(grids)
    if not grid_tuple:
        raise ValueError("grids must contain at least one PSTFRadialChannelKernelGrid.")
    if any(not isinstance(grid, PSTFRadialChannelKernelGrid) for grid in grid_tuple):
        raise ValueError("grids must contain only PSTFRadialChannelKernelGrid instances.")
    reference = grid_tuple[0]
    reference_kernel_shapes = {
        monomial: reference.K_by_monomial[monomial].shape
        for monomial in COLLISION_STATISTICAL_MONOMIALS
    }
    for grid in grid_tuple[1:]:
        if grid.p4_energies.shape != reference.p4_energies.shape:
            raise ValueError("all radial grids in a batch must share the same p1/p2/p3 shape.")
        if grid.p4_energy_grid.shape != reference.p4_energy_grid.shape:
            raise ValueError("all radial grids in a batch must share the same p4 grid shape.")
        for monomial in COLLISION_STATISTICAL_MONOMIALS:
            if grid.K_by_monomial[monomial].shape != reference_kernel_shapes[monomial]:
                raise ValueError("all radial grids in a batch must share K tensor shapes.")
    total_kernel_nbytes = sum(
        int(grid.K_by_monomial[monomial].nbytes)
        for grid in grid_tuple
        for monomial in COLLISION_STATISTICAL_MONOMIALS
    )
    if max_kernel_nbytes is not None:
        budget = int(max_kernel_nbytes)
        if budget < 0:
            raise ValueError("max_kernel_nbytes must be non-negative or None.")
        if total_kernel_nbytes > budget:
            return None
    K_by_monomial = {
        monomial: np.stack([grid.K_by_monomial[monomial] for grid in grid_tuple], axis=0)
        for monomial in COLLISION_STATISTICAL_MONOMIALS
    }
    return PSTFRadialChannelKernelGridBatch(
        grids=grid_tuple,
        K_by_monomial=K_by_monomial,
        p4_energies=np.stack([grid.p4_energies for grid in grid_tuple], axis=0),
        p4_left_indices=np.stack([grid.p4_left_indices for grid in grid_tuple], axis=0),
        p4_right_weights=np.stack([grid.p4_right_weights for grid in grid_tuple], axis=0),
        valid_radial_mask=np.stack([grid.valid_radial_mask for grid in grid_tuple], axis=0),
        radial_quadrature_weights=np.stack(
            [grid.radial_quadrature_weights for grid in grid_tuple],
            axis=0,
        ),
        p4_grid_size=reference.p4_energy_grid.size,
        stacked_kernel_nbytes=total_kernel_nbytes,
    )


def contract_pstf_channel_radial_grid(
    F1_modes: np.ndarray,
    F2_modes: np.ndarray,
    F3_modes: np.ndarray,
    F4_modes: np.ndarray,
    p1_energies: np.ndarray,
    p2_energies: np.ndarray,
    p3_energies: np.ndarray,
    p4_energy_grid: np.ndarray,
    q2_weights: np.ndarray,
    q3_weights: np.ndarray,
    K_by_monomial: Mapping[str, np.ndarray],
) -> PSTFChannelRadialGridContractionResult:
    """Contract precomputed channel ``K`` tensors over ``p2,p3`` with ``p4`` interpolation."""

    F1 = _radial_mode_grid(F1_modes, name="F1_modes")
    F2 = _radial_mode_grid(F2_modes, name="F2_modes")
    F3 = _radial_mode_grid(F3_modes, name="F3_modes")
    F4 = _radial_mode_grid(F4_modes, name="F4_modes")
    n1, n_modes = F1.shape
    n2 = F2.shape[0]
    n3 = F3.shape[0]
    if F2.shape[1] != n_modes or F3.shape[1] != n_modes or F4.shape[1] != n_modes:
        raise ValueError("all F*_modes arrays must share the same mode axis.")
    e1 = _radial_grid(p1_energies, name="p1_energies", size=n1)
    e2 = _radial_grid(p2_energies, name="p2_energies", size=n2)
    e3 = _radial_grid(p3_energies, name="p3_energies", size=n3)
    e4_grid = _strictly_increasing_grid(p4_energy_grid, name="p4_energy_grid", size=F4.shape[0])
    w2 = _radial_grid(q2_weights, name="q2_weights", size=n2)
    w3 = _radial_grid(q3_weights, name="q3_weights", size=n3)
    if not np.all(w2 >= 0.0) or not np.all(w3 >= 0.0):
        raise ValueError("q2_weights and q3_weights must be non-negative.")
    K = _validate_radial_kernel_tensors(
        K_by_monomial,
        n1=n1,
        n2=n2,
        n3=n3,
        n_modes=n_modes,
    )

    C = np.zeros((n1, n_modes), dtype=float)
    p4_energies = np.zeros((n1, n2, n3), dtype=float)
    p4_left = np.zeros((n1, n2, n3), dtype=int)
    p4_right_weight = np.zeros((n1, n2, n3), dtype=float)
    valid = np.zeros((n1, n2, n3), dtype=bool)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                e4 = float(e1[i1] + e2[i2] - e3[i3])
                p4_energies[i1, i2, i3] = e4
                is_valid, left, weight = _p4_linear_interpolation(e4, e4_grid)
                p4_left[i1, i2, i3] = left
                p4_right_weight[i1, i2, i3] = weight
                valid[i1, i2, i3] = is_valid
                if not is_valid:
                    continue
                F4_interp = (1.0 - weight) * F4[left] + weight * F4[left + 1]
                radial_weight = float(w2[i2] * w3[i3])
                C[i1] += radial_weight * (
                    np.einsum("ade,d,e->a", K["34"][i1, i2, i3], F3[i3], F4_interp)
                    - np.einsum("abc,b,c->a", K["12"][i1, i2, i3], F1[i1], F2[i2])
                    + np.einsum("abcd,b,c,d->a", K["123"][i1, i2, i3], F1[i1], F2[i2], F3[i3])
                    + np.einsum("abce,b,c,e->a", K["124"][i1, i2, i3], F1[i1], F2[i2], F4_interp)
                    - np.einsum("abde,b,d,e->a", K["134"][i1, i2, i3], F1[i1], F3[i3], F4_interp)
                    - np.einsum("acde,c,d,e->a", K["234"][i1, i2, i3], F2[i2], F3[i3], F4_interp)
                )
    return PSTFChannelRadialGridContractionResult(
        C_modes=C,
        p4_energies=p4_energies,
        p4_left_indices=p4_left,
        p4_right_weights=p4_right_weight,
        valid_radial_mask=valid,
    )


def _contract_pstf_channel_radial_grid_precomputed(
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    F4: np.ndarray,
    *,
    q2_weights: np.ndarray,
    q3_weights: np.ndarray,
    radial_quadrature_weights: np.ndarray | None = None,
    K: Mapping[str, np.ndarray],
    p4_energies: np.ndarray,
    p4_left: np.ndarray,
    p4_right_weight: np.ndarray,
    valid: np.ndarray,
) -> PSTFChannelRadialGridContractionResult:
    """Vectorized radial contraction for a precomputed channel grid."""

    left = np.asarray(p4_left, dtype=int)
    right_weight = np.asarray(p4_right_weight, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    F4_interp = (1.0 - right_weight[..., None]) * F4[left] + right_weight[..., None] * F4[left + 1]
    if radial_quadrature_weights is None:
        radial_weight = (
            mask.astype(float)
            * np.asarray(q2_weights, dtype=float)[None, :, None]
            * np.asarray(q3_weights, dtype=float)[None, None, :]
        )
    else:
        radial_weight = np.asarray(radial_quadrature_weights, dtype=float)
    C = (
        np.einsum("ijkade,kd,ijke,ijk->ia", K["34"], F3, F4_interp, radial_weight)
        - np.einsum("ijkabc,ib,jc,ijk->ia", K["12"], F1, F2, radial_weight)
        + np.einsum("ijkabcd,ib,jc,kd,ijk->ia", K["123"], F1, F2, F3, radial_weight)
        + np.einsum("ijkabce,ib,jc,ijke,ijk->ia", K["124"], F1, F2, F4_interp, radial_weight)
        - np.einsum("ijkabde,ib,kd,ijke,ijk->ia", K["134"], F1, F3, F4_interp, radial_weight)
        - np.einsum("ijkacde,jc,kd,ijke,ijk->ia", K["234"], F2, F3, F4_interp, radial_weight)
    )
    return PSTFChannelRadialGridContractionResult(
        C_modes=C,
        p4_energies=p4_energies,
        p4_left_indices=left,
        p4_right_weights=right_weight,
        valid_radial_mask=mask,
    )


@dataclass
class PSTFChannelRadialGridJacobian:
    """Analytic Jacobian of the precomputed channel collision moment C (leg-dropped einsums).

    The moment C[i,a] (the six-monomial einsum in
    :func:`_contract_pstf_channel_radial_grid_precomputed`) is cubic in the leg mode
    coefficients, each F appearing at most once per monomial, so dC/dF_k is the same einsum
    with leg k dropped -- machine-exact, no AD. Blocks:
      ``J1_diag[i,a,b] = dC[i,a]/dF1[i,b]`` (F1 is diagonal in the leg-1 radial index);
      ``J1_full[i,a,n,b]`` the dense form (nonzero only n==i);
      ``J2[i,a,j,c] = dC[i,a]/dF2[j,c]``; ``J3[i,a,k,d] = dC[i,a]/dF3[k,d]``;
      ``J4[i,a,n,e] = dC[i,a]/dF4[n,e]`` (through the on-shell interpolation gather).
    """

    J1_diag: np.ndarray
    J1_full: np.ndarray
    J2: np.ndarray
    J3: np.ndarray
    J4: np.ndarray


def build_pstf_channel_radial_grid_jacobian(
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    F4: np.ndarray,
    *,
    q2_weights: np.ndarray,
    q3_weights: np.ndarray,
    radial_quadrature_weights: np.ndarray | None = None,
    K: Mapping[str, np.ndarray],
    p4_energies: np.ndarray,
    p4_left: np.ndarray,
    p4_right_weight: np.ndarray,
    valid: np.ndarray,
) -> PSTFChannelRadialGridJacobian:
    """Exact analytic dC/dF_k for the precomputed channel grid (product rule, leg dropped).

    Mirrors :func:`_contract_pstf_channel_radial_grid_precomputed` (same F4 on-shell gather
    and radial weight), differentiating each of the six monomials with respect to each leg.

    Contract (identical to the forward ``_contract`` it mirrors): this low-level builder
    TRUSTS the caller -- finiteness and the p4_left/F4 energy-grid consistency are validated
    upstream by the grid wrapper, not re-checked here (an out-of-grid ``p4_left`` raises a
    loud IndexError via the same ``left+1`` gather as the forward pass).
    """
    left = np.asarray(p4_left, dtype=int)
    right_weight = np.asarray(p4_right_weight, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    F1 = np.asarray(F1, dtype=float)
    F2 = np.asarray(F2, dtype=float)
    F3 = np.asarray(F3, dtype=float)
    F4 = np.asarray(F4, dtype=float)
    F4_interp = (1.0 - right_weight[..., None]) * F4[left] + right_weight[..., None] * F4[left + 1]
    if radial_quadrature_weights is None:
        w = (mask.astype(float)
             * np.asarray(q2_weights, dtype=float)[None, :, None]
             * np.asarray(q3_weights, dtype=float)[None, None, :])
    else:
        w = np.asarray(radial_quadrature_weights, dtype=float)
    K34, K12, K123, K124, K134, K234 = (K["34"], K["12"], K["123"], K["124"], K["134"], K["234"])

    # dC[i,a]/dF1[i,b] -- F1 is "ib", so the Jacobian is diagonal in the leg-1 radial index i.
    J1_diag = (
        -np.einsum("ijkabc,jc,ijk->iab", K12, F2, w)
        + np.einsum("ijkabcd,jc,kd,ijk->iab", K123, F2, F3, w)
        + np.einsum("ijkabce,jc,ijke,ijk->iab", K124, F2, F4_interp, w)
        - np.einsum("ijkabde,kd,ijke,ijk->iab", K134, F3, F4_interp, w)
    )
    ni, na, nb = J1_diag.shape
    J1_full = np.zeros((ni, na, ni, nb))
    diag = np.arange(ni)
    J1_full[diag, :, diag, :] = J1_diag

    J2 = (
        -np.einsum("ijkabc,ib,ijk->iajc", K12, F1, w)
        + np.einsum("ijkabcd,ib,kd,ijk->iajc", K123, F1, F3, w)
        + np.einsum("ijkabce,ib,ijke,ijk->iajc", K124, F1, F4_interp, w)
        - np.einsum("ijkacde,kd,ijke,ijk->iajc", K234, F3, F4_interp, w)
    )

    J3 = (
        +np.einsum("ijkade,ijke,ijk->iakd", K34, F4_interp, w)
        + np.einsum("ijkabcd,ib,jc,ijk->iakd", K123, F1, F2, w)
        - np.einsum("ijkabde,ib,ijke,ijk->iakd", K134, F1, F4_interp, w)
        - np.einsum("ijkacde,jc,ijke,ijk->iakd", K234, F2, F4_interp, w)
    )

    # F4 enters only via the gather F4_interp; differentiate wrt F4_interp then chain.
    G = (
        +np.einsum("ijkade,kd,ijk->iajke", K34, F3, w)
        + np.einsum("ijkabce,ib,jc,ijk->iajke", K124, F1, F2, w)
        - np.einsum("ijkabde,ib,kd,ijk->iajke", K134, F1, F3, w)
        - np.einsum("ijkacde,jc,kd,ijk->iajke", K234, F2, F3, w)
    )
    nn = F4.shape[0]
    ii, jj, kk = np.meshgrid(np.arange(left.shape[0]), np.arange(left.shape[1]),
                             np.arange(left.shape[2]), indexing="ij")
    interp = np.zeros(left.shape + (nn,))
    np.add.at(interp, (ii, jj, kk, left), 1.0 - right_weight)
    np.add.at(interp, (ii, jj, kk, left + 1), right_weight)
    J4 = np.einsum("iajke,ijkn->iane", G, interp)

    return PSTFChannelRadialGridJacobian(J1_diag=J1_diag, J1_full=J1_full, J2=J2, J3=J3, J4=J4)


def pstf_channel_jacobian_jvp(
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    F4: np.ndarray,
    dF1: np.ndarray,
    dF2: np.ndarray,
    dF3: np.ndarray,
    dF4: np.ndarray,
    *,
    q2_weights: np.ndarray,
    q3_weights: np.ndarray,
    radial_quadrature_weights: np.ndarray | None = None,
    K: Mapping[str, np.ndarray],
    p4_energies: np.ndarray,
    p4_left: np.ndarray,
    p4_right_weight: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Matrix-free Jacobian-vector product dC = (dC/dF) . (dF1,dF2,dF3,dF4).

    Because the channel moment C is multilinear in the legs, the directional derivative is
    just the sum of the six monomials with ONE leg's F replaced by its perturbation -- so the
    full Jacobian is never materialized (the plan's rank-7 / N_q=80 caveat). Equivalent to
    contracting :func:`build_pstf_channel_radial_grid_jacobian`'s blocks with dF, at lower cost.
    """
    left = np.asarray(p4_left, dtype=int)
    right_weight = np.asarray(p4_right_weight, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    F1 = np.asarray(F1, dtype=float); F2 = np.asarray(F2, dtype=float); F3 = np.asarray(F3, dtype=float)
    F4 = np.asarray(F4, dtype=float)
    dF1 = np.asarray(dF1, dtype=float); dF2 = np.asarray(dF2, dtype=float); dF3 = np.asarray(dF3, dtype=float)
    dF4 = np.asarray(dF4, dtype=float)
    F4_interp = (1.0 - right_weight[..., None]) * F4[left] + right_weight[..., None] * F4[left + 1]
    dF4_interp = (1.0 - right_weight[..., None]) * dF4[left] + right_weight[..., None] * dF4[left + 1]
    if radial_quadrature_weights is None:
        w = (mask.astype(float)
             * np.asarray(q2_weights, dtype=float)[None, :, None]
             * np.asarray(q3_weights, dtype=float)[None, None, :])
    else:
        w = np.asarray(radial_quadrature_weights, dtype=float)
    K34, K12, K123, K124, K134, K234 = (K["34"], K["12"], K["123"], K["124"], K["134"], K["234"])

    # leg 1 perturbed (F1 -> dF1)
    dC = (
        -np.einsum("ijkabc,ib,jc,ijk->ia", K12, dF1, F2, w)
        + np.einsum("ijkabcd,ib,jc,kd,ijk->ia", K123, dF1, F2, F3, w)
        + np.einsum("ijkabce,ib,jc,ijke,ijk->ia", K124, dF1, F2, F4_interp, w)
        - np.einsum("ijkabde,ib,kd,ijke,ijk->ia", K134, dF1, F3, F4_interp, w)
    )
    # leg 2 perturbed (F2 -> dF2)
    dC += (
        -np.einsum("ijkabc,ib,jc,ijk->ia", K12, F1, dF2, w)
        + np.einsum("ijkabcd,ib,jc,kd,ijk->ia", K123, F1, dF2, F3, w)
        + np.einsum("ijkabce,ib,jc,ijke,ijk->ia", K124, F1, dF2, F4_interp, w)
        - np.einsum("ijkacde,jc,kd,ijke,ijk->ia", K234, dF2, F3, F4_interp, w)
    )
    # leg 3 perturbed (F3 -> dF3)
    dC += (
        +np.einsum("ijkade,kd,ijke,ijk->ia", K34, dF3, F4_interp, w)
        + np.einsum("ijkabcd,ib,jc,kd,ijk->ia", K123, F1, F2, dF3, w)
        - np.einsum("ijkabde,ib,kd,ijke,ijk->ia", K134, F1, dF3, F4_interp, w)
        - np.einsum("ijkacde,jc,kd,ijke,ijk->ia", K234, F2, dF3, F4_interp, w)
    )
    # leg 4 perturbed (F4_interp -> dF4_interp, via the same on-shell gather)
    dC += (
        +np.einsum("ijkade,kd,ijke,ijk->ia", K34, F3, dF4_interp, w)
        + np.einsum("ijkabce,ib,jc,ijke,ijk->ia", K124, F1, F2, dF4_interp, w)
        - np.einsum("ijkabde,ib,kd,ijke,ijk->ia", K134, F1, F3, dF4_interp, w)
        - np.einsum("ijkacde,jc,kd,ijke,ijk->ia", K234, F2, F3, dF4_interp, w)
    )
    return dC


def contract_pstf_radial_channel_kernel_grid(
    F1_modes: np.ndarray,
    F2_modes: np.ndarray,
    F3_modes: np.ndarray,
    F4_modes: np.ndarray,
    grid: PSTFRadialChannelKernelGrid,
) -> PSTFChannelRadialGridContractionResult:
    """Contract a precomputed radial channel-kernel grid."""

    if not isinstance(grid, PSTFRadialChannelKernelGrid):
        raise ValueError("grid must be a PSTFRadialChannelKernelGrid.")
    F1 = _radial_mode_grid(F1_modes, name="F1_modes")
    F2 = _radial_mode_grid(F2_modes, name="F2_modes")
    F3 = _radial_mode_grid(F3_modes, name="F3_modes")
    F4 = _radial_mode_grid(F4_modes, name="F4_modes")
    n1, n_modes = F1.shape
    n2 = F2.shape[0]
    n3 = F3.shape[0]
    if F2.shape[1] != n_modes or F3.shape[1] != n_modes or F4.shape[1] != n_modes:
        raise ValueError("all F*_modes arrays must share the same mode axis.")
    if grid.p1_energies.shape != (n1,) or grid.p2_energies.shape != (n2,) or grid.p3_energies.shape != (n3,):
        raise ValueError("F*_modes radial axes must match grid energy axes.")
    if grid.p4_energy_grid.shape != (F4.shape[0],):
        raise ValueError("F4_modes radial axis must match grid p4 energy axis.")
    return _contract_pstf_channel_radial_grid_precomputed(
        F1,
        F2,
        F3,
        F4,
        q2_weights=grid.q2_weights,
        q3_weights=grid.q3_weights,
        radial_quadrature_weights=grid.radial_quadrature_weights,
        K=grid.K_by_monomial,
        p4_energies=grid.p4_energies,
        p4_left=grid.p4_left_indices,
        p4_right_weight=grid.p4_right_weights,
        valid=grid.valid_radial_mask,
    )


def contract_pstf_radial_channel_kernel_grid_batch(
    F1_modes_by_process: np.ndarray,
    F2_modes_by_process: np.ndarray,
    F3_modes_by_process: np.ndarray,
    F4_modes_by_process: np.ndarray,
    batch: PSTFRadialChannelKernelGridBatch,
) -> tuple[PSTFChannelRadialGridContractionResult, ...]:
    """Contract a process-leading batch of precomputed radial channel grids."""

    if not isinstance(batch, PSTFRadialChannelKernelGridBatch):
        raise ValueError("batch must be a PSTFRadialChannelKernelGridBatch.")
    F1 = _radial_mode_grid_batch(F1_modes_by_process, name="F1_modes_by_process")
    F2 = _radial_mode_grid_batch(F2_modes_by_process, name="F2_modes_by_process")
    F3 = _radial_mode_grid_batch(F3_modes_by_process, name="F3_modes_by_process")
    F4 = _radial_mode_grid_batch(F4_modes_by_process, name="F4_modes_by_process")
    n_process, n1, n_modes = F1.shape
    n2 = F2.shape[1]
    n3 = F3.shape[1]
    if len(batch.grids) != n_process:
        raise ValueError("F*_modes_by_process process axis must match batch.grids.")
    if (
        F2.shape[0] != n_process
        or F3.shape[0] != n_process
        or F4.shape[0] != n_process
        or F2.shape[2] != n_modes
        or F3.shape[2] != n_modes
        or F4.shape[2] != n_modes
    ):
        raise ValueError("all batched F*_modes arrays must share process and mode axes.")
    if F4.shape[1] != batch.p4_grid_size:
        raise ValueError("F4_modes_by_process radial axis must match batch p4 grid size.")
    if batch.p4_energies.shape != (n_process, n1, n2, n3):
        raise ValueError("F*_modes_by_process radial axes must match batch metadata.")

    left = batch.p4_left_indices
    right_weight = batch.p4_right_weights
    process_axis = np.arange(n_process)[:, None, None, None]
    F4_interp = (
        (1.0 - right_weight[..., None]) * F4[process_axis, left]
        + right_weight[..., None] * F4[process_axis, left + 1]
    )
    radial_weight = batch.radial_quadrature_weights
    K = batch.K_by_monomial
    C = (
        _cached_radial_batch_einsum(
            "pijkade,pkd,pijke,pijk->pia",
            K["34"],
            F3,
            F4_interp,
            radial_weight,
        )
        - _cached_radial_batch_einsum(
            "pijkabc,pib,pjc,pijk->pia",
            K["12"],
            F1,
            F2,
            radial_weight,
        )
        + _cached_radial_batch_einsum(
            "pijkabcd,pib,pjc,pkd,pijk->pia",
            K["123"],
            F1,
            F2,
            F3,
            radial_weight,
        )
        + _cached_radial_batch_einsum(
            "pijkabce,pib,pjc,pijke,pijk->pia",
            K["124"],
            F1,
            F2,
            F4_interp,
            radial_weight,
        )
        - _cached_radial_batch_einsum(
            "pijkabde,pib,pkd,pijke,pijk->pia",
            K["134"],
            F1,
            F3,
            F4_interp,
            radial_weight,
        )
        - _cached_radial_batch_einsum(
            "pijkacde,pjc,pkd,pijke,pijk->pia",
            K["234"],
            F2,
            F3,
            F4_interp,
            radial_weight,
        )
    )
    return tuple(
        PSTFChannelRadialGridContractionResult(
            C_modes=C[index],
            p4_energies=batch.p4_energies[index],
            p4_left_indices=left[index],
            p4_right_weights=right_weight[index],
            valid_radial_mask=batch.valid_radial_mask[index],
        )
        for index in range(n_process)
    )


def _mode_vector(values: np.ndarray, *, n_modes: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_modes,):
        raise ValueError(f"{name} must have shape (n_modes,).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _occupancy_overshoot(values: np.ndarray) -> float:
    below = np.maximum(-np.asarray(values, dtype=float), 0.0)
    above = np.maximum(np.asarray(values, dtype=float) - 1.0, 0.0)
    return float(max(np.max(below), np.max(above)))


def _validate_nodal_occupation(values: np.ndarray, *, name: str, tolerance: float) -> np.ndarray:
    overshoot = _occupancy_overshoot(values)
    if overshoot > tolerance:
        raise ValueError(f"{name} nodal occupation lies outside [0, 1].")
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def contract_local_pstf_statistical_kernel(
    F1_modes: np.ndarray,
    F2_modes: np.ndarray,
    F3_modes: np.ndarray,
    F4_modes: np.ndarray,
    table: LocalPSTFStatisticalKernelTable,
    *,
    occupancy_tolerance: float = 1.0e-12,
) -> LocalPSTFStatisticalContractionResult:
    """Contract one local PSTF six-monomial collision source.

    ``F*_modes`` are scalar occupation-number multipoles at one fixed radial
    quadrature tuple.  The function returns the mode-space source for the
    quartic-cancelled fermion gain-loss polynomial.  QKE density matrices and
    flavour-coherence terms are out of scope by construction.
    """

    if not isinstance(table, LocalPSTFStatisticalKernelTable):
        raise ValueError("table must be a LocalPSTFStatisticalKernelTable.")
    tol = float(occupancy_tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("occupancy_tolerance must be finite and non-negative.")
    n_modes = table.basis_matrix.shape[0]
    F1 = _mode_vector(F1_modes, n_modes=n_modes, name="F1_modes")
    F2 = _mode_vector(F2_modes, n_modes=n_modes, name="F2_modes")
    F3 = _mode_vector(F3_modes, n_modes=n_modes, name="F3_modes")
    F4 = _mode_vector(F4_modes, n_modes=n_modes, name="F4_modes")

    raw_occupancies = (
        F1 @ table.basis_matrix,
        F2 @ table.basis_matrix,
        F3 @ table.basis_matrix,
        F4 @ table.basis_matrix,
    )
    max_overshoot = max(_occupancy_overshoot(values) for values in raw_occupancies)
    f1, f2, f3, f4 = (
        _validate_nodal_occupation(raw_occupancies[0], name="F1_modes", tolerance=tol),
        _validate_nodal_occupation(raw_occupancies[1], name="F2_modes", tolerance=tol),
        _validate_nodal_occupation(raw_occupancies[2], name="F3_modes", tolerance=tol),
        _validate_nodal_occupation(raw_occupancies[3], name="F4_modes", tolerance=tol),
    )

    nodal_terms = collision_statistical_polynomial_terms(f1, f2, f3, f4)
    monomial_nodal_contributions = {
        key: nodal_terms[key] * table.monomial_angular_kernels[key]
        for key in COLLISION_STATISTICAL_MONOMIALS
    }
    monomial_mode_contributions = {
        "34": COLLISION_STATISTICAL_COEFFICIENTS["34"]
        * np.einsum("ade,d,e->a", table.monomial_tensors["34"], F3, F4),
        "12": COLLISION_STATISTICAL_COEFFICIENTS["12"]
        * np.einsum("abc,b,c->a", table.monomial_tensors["12"], F1, F2),
        "123": COLLISION_STATISTICAL_COEFFICIENTS["123"]
        * np.einsum("abcd,b,c,d->a", table.monomial_tensors["123"], F1, F2, F3),
        "124": COLLISION_STATISTICAL_COEFFICIENTS["124"]
        * np.einsum("abce,b,c,e->a", table.monomial_tensors["124"], F1, F2, F4),
        "134": COLLISION_STATISTICAL_COEFFICIENTS["134"]
        * np.einsum("abde,b,d,e->a", table.monomial_tensors["134"], F1, F3, F4),
        "234": COLLISION_STATISTICAL_COEFFICIENTS["234"]
        * np.einsum("acde,c,d,e->a", table.monomial_tensors["234"], F2, F3, F4),
    }
    C_modes = sum(monomial_mode_contributions.values())
    nodal_source = sum(monomial_nodal_contributions.values())
    return LocalPSTFStatisticalContractionResult(
        C_modes=C_modes,
        nodal_source=nodal_source,
        nodal_occupancies=(f1, f2, f3, f4),
        monomial_mode_contributions=monomial_mode_contributions,
        monomial_nodal_contributions=monomial_nodal_contributions,
        max_occupancy_overshoot=max_overshoot,
    )


__all__ = [
    "LocalPSTFStatisticalContractionResult",
    "LocalPSTFStatisticalKernelTable",
    "PSTFBilinearMatrixElementTerm",
    "PSTFChannelKernelTable",
    "PSTFChannelRadialGridContractionResult",
    "PSTFMassMatrixElementTerm",
    "PSTFMassQuarticMatrixElementTerm",
    "PSTFRadialChannelKernelGrid",
    "PSTFRadialChannelKernelGridBatch",
    "PSTFRadialInvariantPrefactorConfig",
    "UniversalPSTFGeometricKernelTable",
    "build_local_pstf_statistical_kernel_table",
    "build_normalized_radial_momentum_delta_weight_provider",
    "build_normalized_unit_direction_momentum_delta_weights",
    "build_pstf_channel_kernel_table",
    "build_pstf_radial_channel_kernel_grid",
    "build_pstf_radial_channel_kernel_grid_batch",
    "build_universal_pstf_geometric_kernel_table",
    "contract_local_pstf_statistical_kernel",
    "contract_pstf_channel_radial_grid",
    "contract_pstf_radial_channel_kernel_grid",
    "contract_pstf_radial_channel_kernel_grid_batch",
    "read_pstf_radial_channel_kernel_grid_npz",
    "write_pstf_radial_channel_kernel_grid_npz",
]
