"""Private, structurally independent flat-FLRW no-QKE comparator.

This module deliberately does not import another :mod:`rabbit` module.  It is
an independent NumPy/SciPy implementation of the classical diagonal problem,
not a public backend.  The collision convention follows the target-leg
invariant phase-space formula in Hannestad--Madsen (1995), with the identical
initial-neutrino factor corrected by Dolgov--Hansen--Semikoz (1997).

Only static M0/M1 component work is currently authorized.  A trajectory or
endpoint must be approved separately after these exact source bytes pass the
frozen physics gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss, legval, legvander
from scipy.integrate import quad
from scipy.special import expit, log_expit


M_ELECTRON_MEV = 0.5109989500
G_F_MEV_MINUS_2 = 1.1663788e-11
SIN2_THETA_W = 0.23122
G_NEWTON_MEV_MINUS_2 = 6.708830746231458e-45
MEV_TO_INVERSE_SECONDS = 1.519267447878626e21

PI = float(np.pi)
TWO_PI_SQUARED = 2.0 * PI**2
SPECIES = (
    "nu_e",
    "antinu_e",
    "nu_mu",
    "antinu_mu",
    "nu_tau",
    "antinu_tau",
)
FLAVOURS = ("e", "mu", "tau")
PAIR_INDEX = {"e": 0, "mu": 1, "tau": 2}
SPECIES_INDEX = {species: index for index, species in enumerate(SPECIES)}
_EVENT_BLOCK = 4096


class IndependentNoQkeError(RuntimeError):
    """Raw fail-closed error raised by the private comparator."""


@dataclass(frozen=True)
class IndependentNoQkeGrid:
    """Affine Gauss--Legendre collocation and quadrature on ``[0, y_max]``."""

    order: int
    y_max: float
    nodes: np.ndarray
    weights: np.ndarray
    _legendre_vander: np.ndarray
    _unit_weights: np.ndarray

    def integrate(self, values: np.ndarray, *, power: int = 0) -> float:
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != self.nodes.shape:
            raise ValueError("values must have the native grid shape")
        return float(
            np.sum(
                self.weights * np.power(self.nodes, power) * values_array,
                dtype=np.float64,
            )
        )

    def legendre_coefficients(self, values: np.ndarray) -> np.ndarray:
        """Return the degree ``order-1`` interpolating Legendre coefficients."""

        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != self.nodes.shape or not np.all(np.isfinite(values_array)):
            raise IndependentNoQkeError("invalid native values for spectral interpolation")
        modes = np.arange(self.order, dtype=np.float64)
        projections = self._legendre_vander.T @ (self._unit_weights * values_array)
        return (modes + 0.5) * projections

    def interpolate(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the collocation polynomial inside the declared finite domain."""

        query = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(query)) or np.any(query < 0.0) or np.any(query > self.y_max):
            raise IndependentNoQkeError("spectral query lies outside [0, y_max]")
        x = 2.0 * query / self.y_max - 1.0
        return np.asarray(legval(x, self.legendre_coefficients(values)), dtype=np.float64)

    def modal_basis(self, y: np.ndarray | float) -> np.ndarray:
        """Evaluate the full dy-orthonormal mapped Legendre test basis."""

        query = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(query)) or np.any(query < 0.0) or np.any(query > self.y_max):
            raise IndependentNoQkeError("modal query lies outside [0, y_max]")
        x = 2.0 * query / self.y_max - 1.0
        scales = np.sqrt((2.0 * np.arange(self.order) + 1.0) / self.y_max)
        return np.asarray(legvander(x, self.order - 1) * scales, dtype=np.float64)

    def modal_coefficients(self, values: np.ndarray) -> np.ndarray:
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != self.nodes.shape or not np.all(np.isfinite(values_array)):
            raise IndependentNoQkeError("invalid native entropy variable")
        return self.modal_basis(self.nodes).T @ (self.weights * values_array)


def build_independent_grid(order: int = 48, y_max: float = 24.0) -> IndependentNoQkeGrid:
    """Build the prospectively frozen affine GL grid without a Rabbit helper."""

    if order < 8:
        raise ValueError("independent comparator requires order >= 8")
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("y_max must be finite and positive")
    unit_nodes, unit_weights = leggauss(int(order))
    nodes = 0.5 * y_max * (unit_nodes + 1.0)
    weights = 0.5 * y_max * unit_weights
    if (
        not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(weights))
        or not np.all(np.diff(nodes) > 0.0)
        or not np.all(weights > 0.0)
    ):
        raise IndependentNoQkeError("invalid affine Gauss-Legendre rule")
    return IndependentNoQkeGrid(
        order=int(order),
        y_max=float(y_max),
        nodes=np.asarray(nodes, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        _legendre_vander=np.asarray(legvander(unit_nodes, int(order) - 1), dtype=np.float64),
        _unit_weights=np.asarray(unit_weights, dtype=np.float64),
    )


def occupation_to_cloglog(f: np.ndarray | float) -> np.ndarray:
    """Encode a strict-open occupation as ``c=log(-log(1-f))``."""

    values = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise IndependentNoQkeError("occupation must be finite and strictly inside (0, 1)")
    encoded = np.log(-np.log1p(-values))
    if not np.all(np.isfinite(encoded)):
        raise IndependentNoQkeError("unrepresentable complementary-log-log coordinate")
    return np.asarray(encoded, dtype=np.float64)


def cloglog_to_occupation(c: np.ndarray | float) -> np.ndarray:
    """Decode the strict-open complementary-log-log coordinate without clipping."""

    values = np.asarray(c, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise IndependentNoQkeError("coordinate must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exponent = np.exp(values)
        decoded = -np.expm1(-exponent)
    if not np.all(np.isfinite(decoded)) or np.any(decoded <= 0.0) or np.any(decoded >= 1.0):
        raise IndependentNoQkeError("coordinate decodes outside strict occupation domain")
    return np.asarray(decoded, dtype=np.float64)


def cloglog_chain_factor(c: np.ndarray | float) -> np.ndarray:
    """Return ``df/dc=exp(c-exp(c))`` and fail if it is not representable."""

    values = np.asarray(c, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise IndependentNoQkeError("coordinate must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(values - np.exp(values))
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise IndependentNoQkeError("unrepresentable occupation chain factor")
    return np.asarray(result, dtype=np.float64)


def fermi_dirac_from_logit(logit: np.ndarray | float) -> np.ndarray:
    values = np.asarray(logit, dtype=np.float64)
    result = expit(values)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0) or np.any(result >= 1.0):
        raise IndependentNoQkeError("Fermi-Dirac occupation is not strict-open")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class IndependentElectromagneticEos:
    rho: float
    pressure: float
    drho_dtemperature: float
    rho_photon: float
    rho_electron_positron: float
    pressure_electron_positron: float


@dataclass(frozen=True)
class IndependentThermodynamics:
    number_density_neutrino: float
    energy_density_neutrino: float
    pressure_neutrino: float
    entropy_density_neutrino: float
    energy_density_electromagnetic: float
    pressure_electromagnetic: float
    entropy_density_electromagnetic: float
    energy_density_total: float
    pressure_total: float
    hubble_mev: float


@dataclass(frozen=True)
class IndependentActionMoments:
    signed_number_rate: float
    absolute_number_rate: float
    signed_energy_rate: float
    absolute_energy_rate: float


def independent_action_moments(
    *,
    grid: IndependentNoQkeGrid,
    action: np.ndarray,
    temperature_cm_mev: float,
) -> IndependentActionMoments:
    """Reduce a six-helicity target action without a Rabbit moment helper."""

    values = np.asarray(action, dtype=np.float64)
    temperature = float(temperature_cm_mev)
    if values.shape != (6, grid.order) or not np.all(np.isfinite(values)):
        raise IndependentNoQkeError("action must have shape (6, grid.order)")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise IndependentNoQkeError("T_cm must be finite and positive")
    number_weights = grid.weights * np.square(grid.nodes)
    energy_weights = number_weights * grid.nodes
    signed_number = (
        temperature**3
        * np.sum(values * number_weights[None, :], dtype=np.float64)
        / TWO_PI_SQUARED
    )
    absolute_number = (
        temperature**3
        * np.sum(np.abs(values) * number_weights[None, :], dtype=np.float64)
        / TWO_PI_SQUARED
    )
    signed_energy = (
        temperature**4
        * np.sum(values * energy_weights[None, :], dtype=np.float64)
        / TWO_PI_SQUARED
    )
    absolute_energy = (
        temperature**4
        * np.sum(np.abs(values) * energy_weights[None, :], dtype=np.float64)
        / TWO_PI_SQUARED
    )
    result = IndependentActionMoments(
        signed_number_rate=float(signed_number),
        absolute_number_rate=float(absolute_number),
        signed_energy_rate=float(signed_energy),
        absolute_energy_rate=float(absolute_energy),
    )
    if not all(np.isfinite(value) for value in result.__dict__.values()):
        raise IndependentNoQkeError("nonfinite action moment")
    return result


def independent_thermodynamics(
    *,
    grid: IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    electron_mass_mev: float = M_ELECTRON_MEV,
) -> IndependentThermodynamics:
    """Return finite-domain neutrino plus QED-off electromagnetic FLRW data."""

    temperature_cm = float(temperature_cm_mev)
    temperature_gamma = float(temperature_gamma_mev)
    if (
        not np.isfinite(temperature_cm)
        or temperature_cm <= 0.0
        or not np.isfinite(temperature_gamma)
        or temperature_gamma <= 0.0
    ):
        raise IndependentNoQkeError("thermodynamic temperatures must be positive")
    coordinates = np.asarray(pair_cloglog, dtype=np.float64)
    if coordinates.shape != (3, grid.order):
        raise IndependentNoQkeError("pair coordinates must have shape (3, grid.order)")
    occupations = cloglog_to_occupation(coordinates)
    if occupations.shape != coordinates.shape:
        raise IndependentNoQkeError("invalid pair occupation shape")

    pair_number_integral = np.sum(
        occupations * (grid.weights * np.square(grid.nodes))[None, :],
        dtype=np.float64,
    )
    pair_energy_integral = np.sum(
        occupations * (grid.weights * np.power(grid.nodes, 3))[None, :],
        dtype=np.float64,
    )
    entropy_integrand = -(
        occupations * np.log(occupations)
        + (1.0 - occupations) * np.log1p(-occupations)
    )
    pair_entropy_integral = np.sum(
        entropy_integrand * (grid.weights * np.square(grid.nodes))[None, :],
        dtype=np.float64,
    )
    # Each pair spectrum represents two explicitly enumerated helicity states.
    number_neutrino = 2.0 * temperature_cm**3 * pair_number_integral / TWO_PI_SQUARED
    energy_neutrino = 2.0 * temperature_cm**4 * pair_energy_integral / TWO_PI_SQUARED
    entropy_neutrino = 2.0 * temperature_cm**3 * pair_entropy_integral / TWO_PI_SQUARED
    pressure_neutrino = energy_neutrino / 3.0
    electromagnetic = electromagnetic_eos_adaptive(
        temperature_gamma,
        electron_mass_mev,
    )
    entropy_electromagnetic = (
        electromagnetic.rho + electromagnetic.pressure
    ) / temperature_gamma
    energy_total = energy_neutrino + electromagnetic.rho
    pressure_total = pressure_neutrino + electromagnetic.pressure
    hubble = np.sqrt(8.0 * PI * G_NEWTON_MEV_MINUS_2 * energy_total / 3.0)
    result = IndependentThermodynamics(
        number_density_neutrino=float(number_neutrino),
        energy_density_neutrino=float(energy_neutrino),
        pressure_neutrino=float(pressure_neutrino),
        entropy_density_neutrino=float(entropy_neutrino),
        energy_density_electromagnetic=float(electromagnetic.rho),
        pressure_electromagnetic=float(electromagnetic.pressure),
        entropy_density_electromagnetic=float(entropy_electromagnetic),
        energy_density_total=float(energy_total),
        pressure_total=float(pressure_total),
        hubble_mev=float(hubble),
    )
    if not all(np.isfinite(value) and value > 0.0 for value in result.__dict__.values()):
        raise IndependentNoQkeError("nonphysical independent thermodynamics")
    return result


def _fd_positive_energy(x: float) -> float:
    if x > 745.0:
        return 0.0
    return float(expit(-x))


@lru_cache(maxsize=256)
def electromagnetic_eos_adaptive(
    temperature_mev: float,
    electron_mass_mev: float = M_ELECTRON_MEV,
) -> IndependentElectromagneticEos:
    """QED-off photon plus finite-vacuum-mass e-/e+ equilibrium EOS."""

    temperature = float(temperature_mev)
    mass = float(electron_mass_mev)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise IndependentNoQkeError("electromagnetic temperature must be positive")
    if not np.isfinite(mass) or mass < 0.0:
        raise IndependentNoQkeError("electron mass must be finite and nonnegative")
    ratio = mass / temperature

    def rho_integrand(x: float) -> float:
        energy = float(np.hypot(x, ratio))
        return x * x * energy * _fd_positive_energy(energy)

    def pressure_integrand(x: float) -> float:
        energy = float(np.hypot(x, ratio))
        return x**4 * _fd_positive_energy(energy) / energy

    def derivative_integrand(x: float) -> float:
        energy = float(np.hypot(x, ratio))
        occupation = _fd_positive_energy(energy)
        return x * x * energy * energy * occupation * (1.0 - occupation)

    quad_options = {"epsabs": 2.0e-13, "epsrel": 2.0e-12, "limit": 300}
    rho_dimensionless = quad(rho_integrand, 0.0, np.inf, **quad_options)[0]
    pressure_dimensionless = quad(pressure_integrand, 0.0, np.inf, **quad_options)[0]
    derivative_dimensionless = quad(derivative_integrand, 0.0, np.inf, **quad_options)[0]

    rho_photon = PI**2 * temperature**4 / 15.0
    rho_electron = 2.0 * temperature**4 * rho_dimensionless / PI**2
    pressure_electron = 2.0 * temperature**4 * pressure_dimensionless / (3.0 * PI**2)
    drho = 4.0 * PI**2 * temperature**3 / 15.0
    drho += 2.0 * temperature**3 * derivative_dimensionless / PI**2
    values = (rho_photon, rho_electron, pressure_electron, drho)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise IndependentNoQkeError("nonphysical electromagnetic EOS result")
    return IndependentElectromagneticEos(
        rho=float(rho_photon + rho_electron),
        pressure=float(rho_photon / 3.0 + pressure_electron),
        drho_dtemperature=float(drho),
        rho_photon=float(rho_photon),
        rho_electron_positron=float(rho_electron),
        pressure_electron_positron=float(pressure_electron),
    )


def _species_flavour(species: str) -> str:
    if species.endswith("_e"):
        return "e"
    if species.endswith("_mu"):
        return "mu"
    if species.endswith("_tau"):
        return "tau"
    raise ValueError(f"unknown species {species!r}")


def _is_antineutrino(species: str) -> bool:
    return species.startswith("antinu_")


def _species(flavour: str, anti: bool) -> str:
    return ("antinu_" if anti else "nu_") + flavour


def _cp_partner(species: str) -> str:
    return _species(_species_flavour(species), not _is_antineutrino(species))


@dataclass(frozen=True)
class IndependentSelfReaction:
    target: str
    partner: str
    outgoing_3: str
    outgoing_4: str
    category: str
    row: int
    kernel: Literal["K_s", "K_t"]
    coefficient: float


def _self_row(category: str, target_flavour: str, other_flavour: str | None = None) -> int:
    if category == "same_sign_identical":
        return 1 if target_flavour == "e" else 3
    if category == "same_pair_opposite_sign_elastic":
        return 2 if target_flavour == "e" else 4
    if other_flavour is None:
        raise ValueError("cross-flavour category requires another flavour")
    includes_e = target_flavour == "e" or other_flavour == "e"
    if category == "distinct_same_sign_elastic":
        return 7 if includes_e else 3
    if category == "distinct_opposite_sign_elastic":
        return 8 if includes_e else 5
    if category == "pair_conversion":
        return 9 if includes_e else 6
    raise ValueError(f"unknown self-reaction category {category!r}")


def independent_self_reactions() -> tuple[IndependentSelfReaction, ...]:
    """Enumerate the 48 target-directed rows before pair reduction."""

    rows: list[IndependentSelfReaction] = []
    for target in SPECIES:
        flavour = _species_flavour(target)
        anti = _is_antineutrino(target)
        rows.append(
            IndependentSelfReaction(
                target=target,
                partner=target,
                outgoing_3=target,
                outgoing_4=target,
                category="same_sign_identical",
                row=_self_row("same_sign_identical", flavour),
                kernel="K_s",
                coefficient=64.0,
            )
        )
        cp = _cp_partner(target)
        rows.append(
            IndependentSelfReaction(
                target=target,
                partner=cp,
                outgoing_3=target,
                outgoing_4=cp,
                category="same_pair_opposite_sign_elastic",
                row=_self_row("same_pair_opposite_sign_elastic", flavour),
                kernel="K_t",
                coefficient=128.0,
            )
        )
        for other in FLAVOURS:
            if other == flavour:
                continue
            same_sign = _species(other, anti)
            opposite_sign = _species(other, not anti)
            rows.append(
                IndependentSelfReaction(
                    target=target,
                    partner=same_sign,
                    outgoing_3=target,
                    outgoing_4=same_sign,
                    category="distinct_same_sign_elastic",
                    row=_self_row("distinct_same_sign_elastic", flavour, other),
                    kernel="K_s",
                    coefficient=32.0,
                )
            )
            rows.append(
                IndependentSelfReaction(
                    target=target,
                    partner=opposite_sign,
                    outgoing_3=target,
                    outgoing_4=opposite_sign,
                    category="distinct_opposite_sign_elastic",
                    row=_self_row("distinct_opposite_sign_elastic", flavour, other),
                    kernel="K_t",
                    coefficient=32.0,
                )
            )
            rows.append(
                IndependentSelfReaction(
                    target=target,
                    partner=cp,
                    outgoing_3=_species(other, anti),
                    outgoing_4=_species(other, not anti),
                    category="pair_conversion",
                    row=_self_row("pair_conversion", flavour, other),
                    kernel="K_t",
                    coefficient=32.0,
                )
            )
    if len(rows) != 48:
        raise AssertionError("six-state catalogue must contain 48 target rows")
    return tuple(rows)


@dataclass(frozen=True)
class IndependentElectronReaction:
    target: str
    category: Literal["elastic_minus", "elastic_plus", "pair"]


def independent_electron_reactions() -> tuple[IndependentElectronReaction, ...]:
    rows = tuple(
        IndependentElectronReaction(target=target, category=category)
        for target in SPECIES
        for category in ("elastic_minus", "elastic_plus", "pair")
    )
    if len(rows) != 18:
        raise AssertionError("electron catalogue must contain 18 target rows")
    return rows


def independent_pair_row_fingerprint() -> tuple[int, ...]:
    """The independently enumerated physical-family grouping R1--R9."""

    return (2, 1, 6, 2, 2, 1, 4, 4, 2)


@dataclass(frozen=True)
class IndependentSelfEvent:
    legs: tuple[str, str, str, str]
    category: str
    kernel: Literal["K_s", "K_t"]
    coefficient: float


def independent_self_events() -> tuple[IndependentSelfEvent, ...]:
    """Enumerate each of the 24 reversible neutrino events exactly once."""

    events = [
        IndependentSelfEvent((s, s, s, s), "same_sign_identical", "K_s", 16.0)
        for s in SPECIES
    ]
    events += [
        IndependentSelfEvent(
            (_species(a, False), _species(a, True), _species(a, False), _species(a, True)),
            "same_pair_opposite_sign_elastic", "K_t", 64.0,
        )
        for a in FLAVOURS
    ]
    for index, a in enumerate(FLAVOURS):
        for b in FLAVOURS[index + 1 :]:
            for anti in (False, True):
                sa, sb = _species(a, anti), _species(b, anti)
                events.append(IndependentSelfEvent((sa, sb, sa, sb), "distinct_same_sign_elastic", "K_s", 16.0))
            for anti in (False, True):
                sa, sb = _species(a, anti), _species(b, not anti)
                events.append(IndependentSelfEvent((sa, sb, sa, sb), "distinct_opposite_sign_elastic", "K_t", 16.0))
            events.append(IndependentSelfEvent(
                (_species(a, False), _species(a, True), _species(b, False), _species(b, True)),
                "pair_conversion", "K_t", 32.0,
            ))
    if len(events) != 24:
        raise AssertionError("global self catalogue must contain 24 events")
    return tuple(events)


def independent_electron_events() -> tuple[IndependentElectronReaction, ...]:
    """Return the 12 elastic plus three pair semantic events."""

    events = [
        IndependentElectronReaction(target=s, category=category)
        for s in SPECIES
        for category in ("elastic_minus", "elastic_plus")
    ]
    events += [IndependentElectronReaction(_species(a, False), "pair") for a in FLAVOURS]
    return tuple(events)


@dataclass(frozen=True)
class IndependentCollisionConfig:
    """Deterministic tensor-quadrature controls for static M1 evaluation."""

    incoming_polar_order: int = 12
    final_polar_order: int = 12
    final_azimuth_order: int = 4
    electron_radial_order: int = 48
    matrix_roundoff_ulps: float = 1024.0

    def __post_init__(self) -> None:
        for name in (
            "incoming_polar_order",
            "final_polar_order",
            "final_azimuth_order",
            "electron_radial_order",
        ):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be at least two")
        if self.final_azimuth_order != 4:
            raise ValueError("the primary-source azimuth contract fixes four midpoint nodes")


@dataclass(frozen=True)
class IndependentCollisionAction:
    species_order: tuple[str, ...]
    electron: np.ndarray
    self_interaction: np.ndarray
    total: np.ndarray
    self_rows: Mapping[int, np.ndarray]
    electron_families: Mapping[str, np.ndarray]
    electron_bath_energy_transfer: float
    electron_bath_energy_by_family: Mapping[str, float]
    whole_reaction_domain_rejections: int
    matrix_roundoff_corrections: int
    largest_matrix_roundoff_correction: float
    modal_electron: np.ndarray
    modal_self_interaction: np.ndarray
    modal_total: np.ndarray
    diagnostics: Mapping[str, float]


class _SpectralLogits:
    def __init__(
        self,
        grid: IndependentNoQkeGrid,
        pair_logits: np.ndarray,
    ) -> None:
        values = np.asarray(pair_logits, dtype=np.float64)
        if values.shape != (3, grid.order) or not np.all(np.isfinite(values)):
            raise IndependentNoQkeError("pair logits must have shape (3, grid.order)")
        occupations = expit(values)
        if np.any(occupations <= 0.0) or np.any(occupations >= 1.0):
            raise IndependentNoQkeError("pair logits do not represent strict-open occupations")
        self.grid = grid
        self.values = values
        self.native_basis = grid.modal_basis(grid.nodes)
        self.coefficients = np.stack([grid.modal_coefficients(row) for row in values])

    def native(self, species: str) -> np.ndarray:
        return self.values[PAIR_INDEX[_species_flavour(species)]]

    def at(self, species: str, y: np.ndarray) -> np.ndarray:
        query = np.asarray(y, dtype=np.float64)
        if np.any(query < 0.0) or np.any(query > self.grid.y_max):
            raise IndependentNoQkeError("neutrino query outside finite spectral domain")
        result = self.grid.modal_basis(query) @ self.coefficients[
            PAIR_INDEX[_species_flavour(species)]
        ]
        if not np.all(np.isfinite(result)):
            raise IndependentNoQkeError("nonfinite spectral interpolant")
        return np.asarray(result, dtype=np.float64)


def _stable_pauli_gain_minus_loss(
    logit_1: np.ndarray | float,
    logit_2: np.ndarray,
    logit_3: np.ndarray,
    logit_4: np.ndarray,
) -> np.ndarray:
    u1 = np.asarray(logit_1, dtype=np.float64)
    u2 = np.asarray(logit_2, dtype=np.float64)
    u3 = np.asarray(logit_3, dtype=np.float64)
    u4 = np.asarray(logit_4, dtype=np.float64)
    if not all(np.all(np.isfinite(item)) for item in (u1, u2, u3, u4)):
        raise IndependentNoQkeError("nonfinite logit in Pauli factor")
    affinity = u3 + u4 - u1 - u2
    log_loss = log_expit(u1) + log_expit(u2) + log_expit(-u3) + log_expit(-u4)
    result = np.empty(np.broadcast_shapes(affinity.shape, log_loss.shape), dtype=np.float64)
    affinity_b = np.broadcast_to(affinity, result.shape)
    log_loss_b = np.broadcast_to(log_loss, result.shape)
    positive = affinity_b >= 0.0
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            result[positive] = (
                np.exp(log_loss_b[positive] + affinity_b[positive])
                * -np.expm1(-affinity_b[positive])
            )
            result[~positive] = np.exp(log_loss_b[~positive]) * np.expm1(
                affinity_b[~positive]
            )
        except FloatingPointError as error:
            raise IndependentNoQkeError("unrepresentable Pauli gain-minus-loss") from error
    if not np.all(np.isfinite(result)):
        raise IndependentNoQkeError("nonfinite Pauli gain-minus-loss")
    return result


@lru_cache(maxsize=32)
def _angular_rule(
    incoming_order: int,
    final_order: int,
    azimuth_order: int,
) -> tuple[np.ndarray, ...]:
    incoming_mu, incoming_weights = leggauss(incoming_order)
    final_mu, final_weights = leggauss(final_order)
    if azimuth_order != 4:
        raise ValueError("only the exact four-midpoint azimuth rule is authorized")
    azimuth = (np.arange(azimuth_order, dtype=np.float64) + 0.5) * (2.0 * PI / azimuth_order)
    azimuth_weights = np.full(azimuth_order, 2.0 * PI / azimuth_order, dtype=np.float64)
    return (
        incoming_mu,
        incoming_weights,
        final_mu,
        final_weights,
        azimuth,
        azimuth_weights,
    )


def _electron_half_line_rule(order: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    unit, weights = leggauss(order)
    r = 0.5 * (unit + 1.0)
    r_weights = 0.5 * weights
    momentum = temperature * r / (1.0 - r)
    dp_weights = r_weights * temperature / np.square(1.0 - r)
    return np.asarray(momentum, dtype=np.float64), np.asarray(dp_weights, dtype=np.float64)


def _kallen(s: np.ndarray, mass3_squared: float, mass4_squared: float) -> np.ndarray:
    return (
        s * s
        + mass3_squared * mass3_squared
        + mass4_squared * mass4_squared
        - 2.0 * s * mass3_squared
        - 2.0 * s * mass4_squared
        - 2.0 * mass3_squared * mass4_squared
    )


def _minkowski_dot(
    energy_a: np.ndarray | float,
    vector_a: np.ndarray,
    energy_b: np.ndarray | float,
    vector_b: np.ndarray,
) -> np.ndarray:
    return np.asarray(energy_a) * np.asarray(energy_b) - np.sum(vector_a * vector_b, axis=-1)


@dataclass(frozen=True)
class _KinematicBatch:
    support: np.ndarray
    p2: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    e4: np.ndarray
    p3_magnitude: np.ndarray
    p4_magnitude: np.ndarray
    phase_space: np.ndarray
    quadrature_weight: np.ndarray
    d12: np.ndarray
    d13: np.ndarray
    d14: np.ndarray
    d23: np.ndarray
    d24: np.ndarray
    d34: np.ndarray


def _two_body_kinematics(
    *,
    p1: float,
    p2_nodes: np.ndarray,
    p2_weights: np.ndarray,
    mass2: float,
    mass3: float,
    mass4: float,
    config: IndependentCollisionConfig,
) -> _KinematicBatch:
    if not np.isfinite(p1) or p1 <= 0.0:
        raise IndependentNoQkeError("target momentum must be finite and positive")
    (
        incoming_mu,
        incoming_weights,
        final_mu,
        final_weights,
        azimuth,
        azimuth_weights,
    ) = _angular_rule(
        config.incoming_polar_order,
        config.final_polar_order,
        config.final_azimuth_order,
    )
    p2, mu12, mu_star, phi = np.meshgrid(
        np.asarray(p2_nodes, dtype=np.float64),
        incoming_mu,
        final_mu,
        azimuth,
        indexing="ij",
    )
    w2, w12, wstar, wphi = np.meshgrid(
        np.asarray(p2_weights, dtype=np.float64),
        incoming_weights,
        final_weights,
        azimuth_weights,
        indexing="ij",
    )
    e1 = float(p1)
    e2 = np.hypot(p2, mass2)
    sin12 = np.sqrt(np.maximum(0.0, 1.0 - np.square(mu12)))
    p1_vector = np.zeros(p2.shape + (3,), dtype=np.float64)
    p1_vector[..., 2] = p1
    p2_vector = np.zeros_like(p1_vector)
    p2_vector[..., 0] = p2 * sin12
    p2_vector[..., 2] = p2 * mu12
    total_energy = e1 + e2
    total_vector = p1_vector + p2_vector
    total_magnitude = np.linalg.norm(total_vector, axis=-1)
    s = np.maximum(0.0, np.square(total_energy) - np.square(total_magnitude))
    threshold_squared = (mass3 + mass4) ** 2
    support = (s > 0.0) & (s > threshold_squared)
    sqrt_s = np.sqrt(np.where(support, s, 1.0))
    lambda_value = _kallen(s, mass3 * mass3, mass4 * mass4)
    lambda_scale = np.maximum(np.square(s), 1.0)
    lambda_roundoff = 512.0 * np.finfo(np.float64).eps * lambda_scale
    if np.any((lambda_value < -lambda_roundoff) & support):
        raise IndependentNoQkeError("negative Kallen discriminant on physical support")
    lambda_nonnegative = np.maximum(lambda_value, 0.0)
    k_star = np.where(support, np.sqrt(lambda_nonnegative) / (2.0 * sqrt_s), 0.0)
    e3_star = np.where(
        support,
        (s + mass3 * mass3 - mass4 * mass4) / (2.0 * sqrt_s),
        mass3,
    )
    beta = np.where(support, total_magnitude / total_energy, 0.0)
    gamma = np.where(support, total_energy / sqrt_s, 1.0)

    parallel = np.zeros_like(total_vector)
    nonzero_total = total_magnitude > 64.0 * np.finfo(np.float64).tiny
    parallel[nonzero_total] = total_vector[nonzero_total] / total_magnitude[nonzero_total, None]
    parallel[..., 2] = np.where(nonzero_total, parallel[..., 2], 1.0)
    transverse_x = np.zeros_like(parallel)
    transverse_x[..., 0] = parallel[..., 2]
    transverse_x[..., 2] = -parallel[..., 0]
    transverse_y = np.zeros_like(parallel)
    transverse_y[..., 1] = 1.0

    sin_star = np.sqrt(np.maximum(0.0, 1.0 - np.square(mu_star)))
    transverse = (
        (k_star * sin_star * np.cos(phi))[..., None] * transverse_x
        + (k_star * sin_star * np.sin(phi))[..., None] * transverse_y
    )
    p3_parallel = gamma * (k_star * mu_star + beta * e3_star)
    p3_vector = transverse + p3_parallel[..., None] * parallel
    e3 = gamma * (e3_star + beta * k_star * mu_star)
    p4_vector = total_vector - p3_vector
    e4 = total_energy - e3
    p3_magnitude = np.linalg.norm(p3_vector, axis=-1)
    p4_magnitude = np.linalg.norm(p4_vector, axis=-1)

    shell3 = np.abs(np.square(e3) - np.square(p3_magnitude) - mass3 * mass3)
    shell4 = np.abs(np.square(e4) - np.square(p4_magnitude) - mass4 * mass4)
    shell_scale = np.maximum.reduce((np.square(total_energy), np.ones_like(total_energy)))
    if np.any((shell3 > 2.0e-10 * shell_scale) & support) or np.any(
        (shell4 > 2.0e-10 * shell_scale) & support
    ):
        raise IndependentNoQkeError("two-body boost violates a final mass shell")
    if np.any((e3 <= 0.0) & support) or np.any((e4 <= 0.0) & support):
        raise IndependentNoQkeError("two-body boost returned nonpositive energy")

    quadrature_weight = w2 * w12 * wstar * wphi
    phase_space = np.where(support, k_star / sqrt_s, 0.0)
    return _KinematicBatch(
        support=support,
        p2=p2,
        e2=e2,
        e3=e3,
        e4=e4,
        p3_magnitude=p3_magnitude,
        p4_magnitude=p4_magnitude,
        phase_space=phase_space,
        quadrature_weight=quadrature_weight,
        d12=_minkowski_dot(e1, p1_vector, e2, p2_vector),
        d13=_minkowski_dot(e1, p1_vector, e3, p3_vector),
        d14=_minkowski_dot(e1, p1_vector, e4, p4_vector),
        d23=_minkowski_dot(e2, p2_vector, e3, p3_vector),
        d24=_minkowski_dot(e2, p2_vector, e4, p4_vector),
        d34=_minkowski_dot(e3, p3_vector, e4, p4_vector),
    )


def _strict_interpolated_logit(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    occupations = expit(logits)
    if (
        not np.all(np.isfinite(logits))
        or not np.all(np.isfinite(occupations))
        or np.any(occupations <= 0.0)
        or np.any(occupations >= 1.0)
    ):
        raise IndependentNoQkeError("spectral interpolant leaves the strict-open domain")
    return logits


def _nonnegative_matrix(
    raw: np.ndarray,
    scale: np.ndarray,
    config: IndependentCollisionConfig,
) -> tuple[np.ndarray, int, float]:
    values = np.asarray(raw, dtype=np.float64)
    scales = np.maximum(np.asarray(scale, dtype=np.float64), np.finfo(np.float64).tiny)
    tolerance = config.matrix_roundoff_ulps * np.finfo(np.float64).eps * scales
    if not np.all(np.isfinite(values)) or np.any(values < -tolerance):
        minimum = float(np.nanmin(values))
        raise IndependentNoQkeError(f"materially negative matrix element: {minimum:.17e}")
    corrected = values < 0.0
    largest = float(np.max(np.where(corrected, -values, 0.0), initial=0.0))
    return np.where(corrected, 0.0, values), int(np.count_nonzero(corrected)), largest


def _self_matrix(
    reaction: IndependentSelfReaction | IndependentSelfEvent,
    batch: _KinematicBatch,
    config: IndependentCollisionConfig,
) -> tuple[np.ndarray, int, float]:
    if reaction.kernel == "K_s":
        kernel = batch.d12 * batch.d34
    else:
        kernel = batch.d14 * batch.d23
    raw = np.where(
        batch.support,
        reaction.coefficient * G_F_MEV_MINUS_2**2 * kernel,
        0.0,
    )
    scale = np.where(
        batch.support,
        reaction.coefficient * G_F_MEV_MINUS_2**2 * np.abs(kernel),
        0.0,
    )
    return _nonnegative_matrix(raw, scale, config)


def _electron_couplings(target: str) -> tuple[float, float]:
    flavour = _species_flavour(target)
    left = (0.5 if flavour == "e" else -0.5) + SIN2_THETA_W
    return float(left), float(SIN2_THETA_W)


def _electron_matrix(
    target: str,
    category: str,
    batch: _KinematicBatch,
    electron_mass: float,
    config: IndependentCollisionConfig,
) -> tuple[np.ndarray, int, float]:
    left, right = _electron_couplings(target)
    anti = _is_antineutrino(target)
    ks = batch.d12 * batch.d34
    kt = batch.d14 * batch.d23
    ku = batch.d13 * batch.d24
    interference_13 = electron_mass**2 * batch.d13
    interference_12 = electron_mass**2 * batch.d12
    if category == "elastic_minus":
        if anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * interference_13,
        )
    elif category == "elastic_plus":
        if not anti:
            left, right = right, left
        terms = (
            left * left * ks,
            right * right * kt,
            -left * right * interference_13,
        )
    elif category == "pair":
        if anti:
            left, right = right, left
        terms = (
            left * left * kt,
            right * right * ku,
            left * right * interference_12,
        )
    else:
        raise ValueError(f"unknown electron category {category!r}")
    common = (128.0 if category == "pair" else 64.0) * G_F_MEV_MINUS_2**2
    raw = np.where(batch.support, common * (terms[0] + terms[1] + terms[2]), 0.0)
    scale = np.where(
        batch.support,
        common * (np.abs(terms[0]) + np.abs(terms[1]) + np.abs(terms[2])),
        0.0,
    )
    return _nonnegative_matrix(raw, scale, config)


def _native_pair_logits(pair_cloglog: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(pair_cloglog, dtype=np.float64)
    occupations = cloglog_to_occupation(coordinates)
    return np.log(occupations) - np.log1p(-occupations)


def pair_logits_to_cloglog(pair_logits: np.ndarray) -> np.ndarray:
    """Construct evolved comparator coordinates from three pair logits."""

    logits = np.asarray(pair_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != 3 or not np.all(np.isfinite(logits)):
        raise IndependentNoQkeError("pair logits must have shape (3, n)")
    return occupation_to_cloglog(expit(logits))


def _modal_product(rates: np.ndarray, y: np.ndarray, grid: IndependentNoQkeGrid) -> np.ndarray:
    """Fixed-block contraction; never materialize a full event-by-mode cache."""

    result = np.zeros((rates.shape[0], grid.order), dtype=np.float64)
    for start in range(0, rates.shape[1], _EVENT_BLOCK):
        stop = min(start + _EVENT_BLOCK, rates.shape[1])
        result += rates[:, start:stop] @ grid.modal_basis(y[start:stop])
    return result


def _native_action(grid: IndependentNoQkeGrid, modal: np.ndarray, temperature: float) -> np.ndarray:
    prefactor = temperature**3 / TWO_PI_SQUARED
    action = (modal @ grid.modal_basis(grid.nodes).T) / (
        prefactor * np.square(grid.nodes)[None, :]
    )
    if not np.all(np.isfinite(action)):
        raise IndependentNoQkeError("nonfinite Galerkin native action")
    return np.asarray(action, dtype=np.float64)


def _event_measure(
    batch: _KinematicBatch, p1: float, outer_weight: float, domain: np.ndarray
) -> np.ndarray:
    measure = (
        outer_weight * batch.quadrature_weight * np.square(batch.p2)
        * batch.phase_space / (batch.e2 * 256.0 * PI**4 * p1)
    )
    return np.asarray(measure[domain], dtype=np.float64)


def _self_event_row(event: IndependentSelfEvent, species: str) -> int:
    flavour = _species_flavour(species)
    others = {_species_flavour(leg) for leg in event.legs} - {flavour}
    return _self_row(event.category, flavour, next(iter(others), None))


def _assemble_self(
    grid: IndependentNoQkeGrid,
    spectra: _SpectralLogits,
    temperature: float,
    config: IndependentCollisionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    events = independent_self_events()
    modal = np.zeros((6, grid.order), dtype=np.float64)
    row_modal = np.zeros((9, 6, grid.order), dtype=np.float64)
    rejected = corrections = 0
    largest = entropy_rate = energy_residual = 0.0
    p2_nodes, p2_weights = temperature * grid.nodes, temperature * grid.weights
    basis = spectra.native_basis
    angular_size = config.incoming_polar_order * config.final_polar_order * 4

    for node_index, y1 in enumerate(grid.nodes):
        p1 = temperature * float(y1)
        outer = temperature**3 * grid.weights[node_index] * y1**2 / TWO_PI_SQUARED
        batch = _two_body_kinematics(
            p1=p1, p2_nodes=p2_nodes, p2_weights=p2_weights,
            mass2=0.0, mass3=0.0, mass4=0.0, config=config,
        )
        y3, y4 = batch.p3_magnitude / temperature, batch.p4_magnitude / temperature
        domain = batch.support & (y3 > 0.0) & (y3 < grid.y_max) & (y4 > 0.0) & (y4 < grid.y_max)
        rejected += len(events) * int(np.count_nonzero(batch.support & ~domain))
        mask, base = domain.ravel(), _event_measure(batch, p1, outer, domain)
        y3_valid, y4_valid = y3[domain], y4[domain]
        rates = np.zeros((len(events), batch.support.size), dtype=np.float64)
        matrix_cache: dict[tuple[str, float], tuple[np.ndarray, int, float]] = {}
        for event_index, event in enumerate(events):
            s1, s2, s3, s4 = event.legs
            u1 = float(spectra.native(s1)[node_index])
            u2 = np.repeat(spectra.native(s2), angular_size)[mask]
            u3, u4 = spectra.at(s3, y3_valid), spectra.at(s4, y4_valid)
            _strict_interpolated_logit(u3); _strict_interpolated_logit(u4)
            key = (event.kernel, event.coefficient)
            matrix_cache.setdefault(key, _self_matrix(event, batch, config))
            matrix, count, correction = matrix_cache[key]
            rate = base * matrix[domain] * _stable_pauli_gain_minus_loss(u1, u2, u3, u4)
            if not np.all(np.isfinite(rate)):
                raise IndependentNoQkeError("nonfinite global self-event rate")
            rates[event_index, mask] = rate
            corrections += count; largest = max(largest, correction)
            entropy_rate += float(np.sum(rate * (u1 + u2 - u3 - u4), dtype=np.float64))
            energy_residual += float(np.sum(
                rate * (p1 + batch.p2[domain] - batch.e3[domain] - batch.e4[domain]),
                dtype=np.float64,
            ))
        sums = np.sum(rates, axis=1, dtype=np.float64)
        leg_modes = (
            sums[:, None] * basis[node_index],
            rates.reshape(len(events), grid.order, angular_size).sum(axis=2) @ basis,
            _modal_product(rates[:, mask], y3_valid, grid),
            _modal_product(rates[:, mask], y4_valid, grid),
        )
        for event_index, event in enumerate(events):
            for sign, species, values in zip((1.0, 1.0, -1.0, -1.0), event.legs, leg_modes):
                index = SPECIES_INDEX[species]
                contribution = sign * values[event_index]
                modal[index] += contribution
                row_modal[_self_event_row(event, species) - 1, index] += contribution
    return modal, row_modal, {
        "whole_reaction_domain_rejections": float(rejected),
        "matrix_roundoff_corrections": float(corrections),
        "largest_matrix_roundoff_correction": float(largest),
        "neutrino_entropy_rate": float(entropy_rate),
        "event_energy_residual": float(energy_residual),
    }


def _electron_family_key(event: IndependentElectronReaction) -> str:
    if event.category == "pair":
        return f"{_species_flavour(event.target)}:pair"
    return f"{event.target}:{event.category}"


def _assemble_electron(
    grid: IndependentNoQkeGrid,
    spectra: _SpectralLogits,
    temperature_cm: float,
    temperature_gamma: float,
    electron_mass: float,
    config: IndependentCollisionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    events, basis = independent_electron_events(), spectra.native_basis
    elastic, pairs = events[:12], events[12:]
    modal = np.zeros((6, grid.order), dtype=np.float64)
    family_modal = {key: np.zeros_like(modal) for key in map(_electron_family_key, events)}
    bath_by_family = {key: 0.0 for key in family_modal}
    rejected = corrections = 0
    largest = qnu = qem = neutrino_h = electromagnetic_h = 0.0
    electron_p2, electron_weights = _electron_half_line_rule(
        config.electron_radial_order, temperature_gamma
    )
    neutrino_p2, neutrino_weights = temperature_cm * grid.nodes, temperature_cm * grid.weights
    angular_size = config.incoming_polar_order * config.final_polar_order * 4

    for node_index, y1 in enumerate(grid.nodes):
        p1 = temperature_cm * float(y1)
        outer = temperature_cm**3 * grid.weights[node_index] * y1**2 / TWO_PI_SQUARED
        batch = _two_body_kinematics(
            p1=p1, p2_nodes=electron_p2, p2_weights=electron_weights,
            mass2=electron_mass, mass3=0.0, mass4=electron_mass, config=config,
        )
        y3 = batch.p3_magnitude / temperature_cm
        domain = batch.support & (y3 > 0.0) & (y3 < grid.y_max)
        rejected += len(elastic) * int(np.count_nonzero(batch.support & ~domain))
        mask, base, y3_valid = domain.ravel(), _event_measure(batch, p1, outer, domain), y3[domain]
        rates = np.zeros((len(elastic), batch.support.size), dtype=np.float64)
        u2, u4 = -batch.e2[domain] / temperature_gamma, -batch.e4[domain] / temperature_gamma
        for event_index, event in enumerate(elastic):
            u1 = float(spectra.native(event.target)[node_index])
            u3 = spectra.at(event.target, y3_valid); _strict_interpolated_logit(u3)
            matrix, count, correction = _electron_matrix(
                event.target, event.category, batch, electron_mass, config
            )
            rate = base * matrix[domain] * _stable_pauli_gain_minus_loss(u1, u2, u3, u4)
            rates[event_index, mask] = rate
            corrections += count; largest = max(largest, correction)
            family = _electron_family_key(event)
            dqnu = float(np.sum(rate * (p1 - batch.e3[domain]), dtype=np.float64))
            dqem = float(np.sum(rate * (batch.e2[domain] - batch.e4[domain]), dtype=np.float64))
            dhnu = float(np.sum(rate * (u1 - u3), dtype=np.float64))
            dhem = float(np.sum(rate * (u2 - u4), dtype=np.float64))
            qnu += dqnu; qem += dqem; neutrino_h += dhnu; electromagnetic_h += dhem
            bath_by_family[family] += dqem
        incoming = np.sum(rates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
        outgoing = _modal_product(rates[:, mask], y3_valid, grid)
        for event_index, event in enumerate(elastic):
            index, contribution = SPECIES_INDEX[event.target], incoming[event_index] - outgoing[event_index]
            modal[index] += contribution
            family_modal[_electron_family_key(event)][index] += contribution

        batch = _two_body_kinematics(
            p1=p1, p2_nodes=neutrino_p2, p2_weights=neutrino_weights,
            mass2=0.0, mass3=electron_mass, mass4=electron_mass, config=config,
        )
        domain, mask = batch.support, batch.support.ravel()
        base = _event_measure(batch, p1, outer, domain)
        rates = np.zeros((len(pairs), batch.support.size), dtype=np.float64)
        u3, u4 = -batch.e3[domain] / temperature_gamma, -batch.e4[domain] / temperature_gamma
        for event_index, event in enumerate(pairs):
            partner = _cp_partner(event.target)
            u1 = float(spectra.native(event.target)[node_index])
            u2 = np.repeat(spectra.native(partner), angular_size)[mask]
            matrix, count, correction = _electron_matrix(
                event.target, "pair", batch, electron_mass, config
            )
            rate = base * matrix[domain] * _stable_pauli_gain_minus_loss(u1, u2, u3, u4)
            rates[event_index, mask] = rate
            corrections += count; largest = max(largest, correction)
            family = _electron_family_key(event)
            dqnu = float(np.sum(rate * (p1 + batch.p2[domain]), dtype=np.float64))
            dqem = float(np.sum(rate * (-batch.e3[domain] - batch.e4[domain]), dtype=np.float64))
            dhnu = float(np.sum(rate * (u1 + u2), dtype=np.float64))
            dhem = float(np.sum(rate * (-u3 - u4), dtype=np.float64))
            qnu += dqnu; qem += dqem; neutrino_h += dhnu; electromagnetic_h += dhem
            bath_by_family[family] += dqem
        incoming_1 = np.sum(rates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
        incoming_2 = rates.reshape(len(pairs), grid.order, angular_size).sum(axis=2) @ basis
        for event_index, event in enumerate(pairs):
            family = _electron_family_key(event)
            for species, contribution in ((event.target, incoming_1[event_index]), (_cp_partner(event.target), incoming_2[event_index])):
                index = SPECIES_INDEX[species]
                modal[index] += contribution; family_modal[family][index] += contribution
    return modal, family_modal, bath_by_family, {
        "whole_reaction_domain_rejections": float(rejected),
        "matrix_roundoff_corrections": float(corrections),
        "largest_matrix_roundoff_correction": float(largest),
        "neutrino_energy_transfer": float(qnu),
        "electromagnetic_energy_transfer": float(qem),
        "neutrino_entropy_rate": float(neutrino_h),
        "electromagnetic_entropy_rate": float(electromagnetic_h),
    }


def _relative_max_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), np.finfo(float).tiny)
    return float(np.max(np.abs(left - right)) / scale)


def evaluate_independent_collision_action(
    *,
    grid: IndependentNoQkeGrid,
    pair_cloglog: np.ndarray,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: IndependentCollisionConfig = IndependentCollisionConfig(),
    electron_mass_mev: float = M_ELECTRON_MEV,
) -> IndependentCollisionAction:
    """Assemble the private static full-degree Galerkin-Petrov M1 action."""

    temperature_cm, temperature_gamma = float(temperature_cm_mev), float(temperature_gamma_mev)
    electron_mass = float(electron_mass_mev)
    if min(temperature_cm, temperature_gamma) <= 0.0 or electron_mass < 0.0:
        raise IndependentNoQkeError("invalid collision thermodynamic input")
    spectra = _SpectralLogits(grid, _native_pair_logits(pair_cloglog))
    self_modal, row_modal, self_meta = _assemble_self(grid, spectra, temperature_cm, config)
    electron_modal, family_modal, bath_by_family, electron_meta = _assemble_electron(
        grid, spectra, temperature_cm, temperature_gamma, electron_mass, config
    )
    self_action = _native_action(grid, self_modal, temperature_cm)
    electron_action = _native_action(grid, electron_modal, temperature_cm)
    total_modal, total = self_modal + electron_modal, self_action + electron_action
    self_rows = {row + 1: _native_action(grid, values, temperature_cm) for row, values in enumerate(row_modal)}
    electron_families = {
        key: _native_action(grid, values, temperature_cm) for key, values in family_modal.items()
    }
    event_neutrino_h = self_meta["neutrino_entropy_rate"] + electron_meta["neutrino_entropy_rate"]
    node_neutrino_h = float(sum(
        np.dot(total_modal[index], spectra.coefficients[PAIR_INDEX[_species_flavour(species)]])
        for index, species in enumerate(SPECIES)
    ))
    qnu, qem = electron_meta["neutrino_energy_transfer"], electron_meta["electromagnetic_energy_transfer"]
    denominator = max(abs(qnu) + abs(qem), np.finfo(float).tiny)
    pair_total = 0.5 * np.stack((total[0] + total[1], total[2] + total[3], total[4] + total[5]))
    diagnostics = {
        "event_neutrino_energy_transfer": qnu,
        "first_law_residual": abs(qnu + qem) / denominator,
        "event_neutrino_entropy_rate": event_neutrino_h,
        "node_neutrino_entropy_rate": node_neutrino_h,
        "electromagnetic_entropy_rate": electron_meta["electromagnetic_entropy_rate"],
        "entropy_production": -(event_neutrino_h + electron_meta["electromagnetic_entropy_rate"]),
        "entropy_duality_residual": abs(node_neutrino_h - event_neutrino_h) / max(abs(node_neutrino_h) + abs(event_neutrino_h), np.finfo(float).tiny),
        "self_event_energy_residual": self_meta["event_energy_residual"],
        "charge_conjugation_residual": max(_relative_max_difference(total[2*i], total[2*i+1]) for i in range(3)),
        "mu_tau_residual": _relative_max_difference(pair_total[1], pair_total[2]),
    }
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise IndependentNoQkeError("nonfinite Galerkin diagnostic")
    return IndependentCollisionAction(
        species_order=SPECIES, electron=electron_action, self_interaction=self_action,
        total=total, self_rows=self_rows, electron_families=electron_families,
        electron_bath_energy_transfer=qem, electron_bath_energy_by_family=bath_by_family,
        whole_reaction_domain_rejections=int(self_meta["whole_reaction_domain_rejections"] + electron_meta["whole_reaction_domain_rejections"]),
        matrix_roundoff_corrections=int(self_meta["matrix_roundoff_corrections"] + electron_meta["matrix_roundoff_corrections"]),
        largest_matrix_roundoff_correction=max(self_meta["largest_matrix_roundoff_correction"], electron_meta["largest_matrix_roundoff_correction"]),
        modal_electron=electron_modal, modal_self_interaction=self_modal,
        modal_total=total_modal, diagnostics=diagnostics,
    )
