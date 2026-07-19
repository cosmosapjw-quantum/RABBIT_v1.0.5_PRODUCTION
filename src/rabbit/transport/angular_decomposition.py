"""Angular decomposition contracts for nonperturbative Type I transport.

This module defines small, deterministic data contracts for the planned
augmented PSTF / S_N Type I Boltzmann surface.  It intentionally does not
wire a solver path; the goal is to separate the new non-linear angular
distribution representation from the legacy linearised ``MultipoleSpec``
contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


AngularParity = Literal["axisymmetric", "cos", "sin"]
AngularGeometry = Literal["lrs", "non_lrs"]
AngularRepresentation = Literal[
    "axisymmetric_legendre",
    "real_spherical_harmonic",
    "sn_pstf_projection",
]
AngularClosure = Literal["zero_tail", "projected_tail", "ray_exact"]
FIXED_NON_LRS_S2_THREE_MODE_PROJECTION_CONTRACT = (
    "fixed_diagonal_three_mode_s2_projection_v1"
)


@dataclass(frozen=True, order=True)
class AngularMode:
    """Real PSTF/spherical-harmonic angular mode label.

    ``m`` is stored as a non-negative real-basis index.  For ``m > 0``,
    ``parity`` distinguishes the cosine and sine real harmonics; for
    ``m == 0`` the only allowed parity is ``axisymmetric``.
    """

    ell: int
    m: int = 0
    parity: AngularParity = "axisymmetric"

    def __post_init__(self) -> None:
        ell = int(self.ell)
        m = int(self.m)
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "m", m)

        if ell < 0:
            raise ValueError("ell must be non-negative.")
        if m < 0:
            raise ValueError("m must be stored as a non-negative real-basis index.")
        if m > ell:
            raise ValueError("m cannot exceed ell.")
        if m == 0 and self.parity != "axisymmetric":
            raise ValueError("m=0 modes must use parity='axisymmetric'.")
        if m > 0 and self.parity == "axisymmetric":
            raise ValueError("m>0 real modes must use parity='cos' or 'sin'.")


def _validate_even_ell_max(ell_max: int) -> int:
    value = int(ell_max)
    if value < 0:
        raise ValueError("ell_max must be non-negative.")
    if value % 2:
        raise ValueError("orthogonal Type I augmented PSTF ladders require even ell_max.")
    return value


def active_lrs_modes(ell_max: int) -> tuple[AngularMode, ...]:
    """Return axisymmetric even-ell modes for LRS Type I."""

    max_ell = _validate_even_ell_max(ell_max)
    return tuple(AngularMode(ell=ell) for ell in range(0, max_ell + 1, 2))


def active_non_lrs_diagonal_modes(
    ell_max: int,
    *,
    include_sine: bool = False,
) -> tuple[AngularMode, ...]:
    """Return reflection-symmetric real modes for diagonal non-LRS Type I.

    The default mode set matches an orthogonal diagonal Type I principal-axis
    frame: even ``ell`` and even ``m`` cosine modes.  Sine partners can be
    requested for diagnostic full-real-basis checks, but they are not part of
    the default no-tilt diagonal model.
    """

    max_ell = _validate_even_ell_max(ell_max)
    modes: list[AngularMode] = []
    for ell in range(0, max_ell + 1, 2):
        modes.append(AngularMode(ell=ell))
        for m in range(2, ell + 1, 2):
            modes.append(AngularMode(ell=ell, m=m, parity="cos"))
            if include_sine:
                modes.append(AngularMode(ell=ell, m=m, parity="sin"))
    return tuple(modes)


@dataclass(frozen=True)
class AngularDecompositionSpec:
    """Angular basis contract for an augmented PSTF transport state."""

    geometry: AngularGeometry
    representation: AngularRepresentation
    ell_max: int
    active_modes: tuple[AngularMode, ...]
    closure: AngularClosure = "projected_tail"
    contract: str = "angular_decomposition_spec_v1"

    def __post_init__(self) -> None:
        max_ell = _validate_even_ell_max(self.ell_max)
        modes = tuple(self.active_modes)
        contract = str(self.contract)
        if not modes:
            raise ValueError("active_modes must not be empty.")
        if any(mode.ell > max_ell for mode in modes):
            raise ValueError("active_modes cannot contain ell > ell_max.")
        if self.geometry == "lrs" and any(mode.m != 0 for mode in modes):
            raise ValueError("LRS angular decomposition only allows m=0 modes.")
        if self.geometry == "lrs" and self.representation != "axisymmetric_legendre":
            raise ValueError("LRS geometry must use representation='axisymmetric_legendre'.")
        if self.geometry == "non_lrs" and self.representation == "axisymmetric_legendre":
            raise ValueError("non-LRS geometry cannot use the axisymmetric-only representation.")
        if not contract:
            raise ValueError("contract must be a non-empty string.")

        object.__setattr__(self, "ell_max", max_ell)
        object.__setattr__(self, "active_modes", modes)
        object.__setattr__(self, "contract", contract)

    @classmethod
    def lrs(cls, ell_max: int, *, closure: AngularClosure = "projected_tail") -> "AngularDecompositionSpec":
        """Build the default LRS even-ell Legendre spec."""

        return cls(
            geometry="lrs",
            representation="axisymmetric_legendre",
            ell_max=ell_max,
            active_modes=active_lrs_modes(ell_max),
            closure=closure,
        )

    @classmethod
    def non_lrs_diagonal(
        cls,
        ell_max: int,
        *,
        closure: AngularClosure = "projected_tail",
        include_sine: bool = False,
    ) -> "AngularDecompositionSpec":
        """Build the default diagonal non-LRS real-mode spec."""

        return cls(
            geometry="non_lrs",
            representation="real_spherical_harmonic",
            ell_max=ell_max,
            active_modes=active_non_lrs_diagonal_modes(ell_max, include_sine=include_sine),
            closure=closure,
        )

    @classmethod
    def fixed_non_lrs_s2_three_mode(
        cls,
        *,
        closure: AngularClosure = "projected_tail",
    ) -> "AngularDecompositionSpec":
        """Build the live fixed diagonal S2 projection used by AP65 no-QKE runs."""

        return cls(
            geometry="non_lrs",
            representation="sn_pstf_projection",
            ell_max=2,
            active_modes=active_non_lrs_diagonal_modes(2),
            closure=closure,
            contract=FIXED_NON_LRS_S2_THREE_MODE_PROJECTION_CONTRACT,
        )

    @property
    def mode_count(self) -> int:
        return len(self.active_modes)


@dataclass(frozen=True)
class EllMaxConvergenceSpec:
    """Configuration for an ell_max convergence ladder."""

    ell_values: tuple[int, ...] = (2, 4, 6, 8)
    relative_tolerance: float = 1.0e-3
    absolute_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        values = tuple(_validate_even_ell_max(value) for value in self.ell_values)
        if len(values) < 2:
            raise ValueError("ell_values must contain at least two ladder points.")
        if tuple(sorted(values)) != values:
            raise ValueError("ell_values must be strictly increasing.")
        if len(set(values)) != len(values):
            raise ValueError("ell_values must not contain duplicates.")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.absolute_tolerance <= 0.0:
            raise ValueError("absolute_tolerance must be positive.")
        object.__setattr__(self, "ell_values", values)


def _float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class CollisionQuadratureSpec:
    """Deterministic fixed-node collision quadrature contract."""

    q_nodes: Sequence[float]
    q_weights: Sequence[float]
    nue_y2_nodes: Sequence[float]
    nue_y2_weights: Sequence[float]
    nue_y3_nodes: Sequence[float]
    nue_y3_weights: Sequence[float]
    pair_y2_nodes: Sequence[float]
    pair_y2_weights: Sequence[float]
    pair_leg_nodes: Sequence[float]
    pair_leg_weights: Sequence[float]
    contract: str = "fixed_gauss_sn_v1"

    def __post_init__(self) -> None:
        pairs = (
            ("q_nodes", "q_weights"),
            ("nue_y2_nodes", "nue_y2_weights"),
            ("nue_y3_nodes", "nue_y3_weights"),
            ("pair_y2_nodes", "pair_y2_weights"),
            ("pair_leg_nodes", "pair_leg_weights"),
        )
        for nodes_name, weights_name in pairs:
            nodes = _float_tuple(getattr(self, nodes_name), name=nodes_name)
            weights = _float_tuple(getattr(self, weights_name), name=weights_name)
            if len(nodes) != len(weights):
                raise ValueError(f"{nodes_name} and {weights_name} must have equal length.")
            object.__setattr__(self, nodes_name, nodes)
            object.__setattr__(self, weights_name, weights)
        if not self.contract:
            raise ValueError("contract must be a non-empty string.")

    def arrays(self) -> dict[str, np.ndarray]:
        """Return quadrature fields as fresh NumPy arrays."""

        return {
            name: np.asarray(getattr(self, name), dtype=float)
            for name in (
                "q_nodes",
                "q_weights",
                "nue_y2_nodes",
                "nue_y2_weights",
                "nue_y3_nodes",
                "nue_y3_weights",
                "pair_y2_nodes",
                "pair_y2_weights",
                "pair_leg_nodes",
                "pair_leg_weights",
            )
        }


@dataclass(frozen=True)
class CollisionMomentResult:
    """Shape-preserving collision moment result with audit diagnostics."""

    C: np.ndarray
    energy_transfer_by_species: Mapping[str, float]
    detailed_balance_residual: float
    number_residual: float
    quadrature_contract: str

    def __post_init__(self) -> None:
        C = np.asarray(self.C, dtype=float)
        if not np.all(np.isfinite(C)):
            raise ValueError("C must contain only finite values.")
        if self.detailed_balance_residual < 0.0:
            raise ValueError("detailed_balance_residual must be non-negative.")
        if self.number_residual < 0.0:
            raise ValueError("number_residual must be non-negative.")
        if not self.quadrature_contract:
            raise ValueError("quadrature_contract must be a non-empty string.")
        object.__setattr__(self, "C", C)
        object.__setattr__(
            self,
            "energy_transfer_by_species",
            {str(key): float(value) for key, value in self.energy_transfer_by_species.items()},
        )


__all__ = [
    "AngularClosure",
    "AngularDecompositionSpec",
    "AngularGeometry",
    "AngularMode",
    "AngularParity",
    "AngularRepresentation",
    "CollisionMomentResult",
    "CollisionQuadratureSpec",
    "EllMaxConvergenceSpec",
    "FIXED_NON_LRS_S2_THREE_MODE_PROJECTION_CONTRACT",
    "active_lrs_modes",
    "active_non_lrs_diagonal_modes",
]
