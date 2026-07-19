"""Physical process descriptors for the AP6 PSTF collision substrate.

The AP6 radial kernel builder consumes compact Hannestad-Madsen style
``Pi_ij`` descriptors.  This module maps the already-tested scalar
ultra-relativistic weak-process formulas onto those descriptors so the PSTF
path can assemble physical channel kernels without caller-authored coefficient
lists.  It is still a no-QKE process catalog, not a promoted BBN runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np

from rabbit.collisions.kernels import G_F_MEV, G_L_NUE, G_L_NUX, G_R_NUE, G_R_NUX
from rabbit.collisions.pstf_contractions import (
    PSTFBilinearMatrixElementTerm,
    PSTFChannelRadialGridContractionResult,
    PSTFMassMatrixElementTerm,
    PSTFMassQuarticMatrixElementTerm,
    PSTFRadialChannelKernelGrid,
    build_pstf_radial_channel_kernel_grid,
    contract_pstf_radial_channel_kernel_grid,
)
from rabbit.collisions.species import Species, as_species


_PROCESS_DESCRIPTOR_CONTRACT = "pstf_ur_hm_process_descriptor_v1"
_FINITE_MASS_NUE_SCATTERING_DESCRIPTOR_CONTRACT = "pstf_finite_mass_hm_nue_scattering_descriptor_v1"
_FINITE_MASS_PAIR_DESCRIPTOR_CONTRACT = "pstf_finite_mass_hm_pair_annihilation_descriptor_v1"
_PROCESS_RADIAL_SOURCE_CONTRACT = "pstf_process_radial_collision_source_v1"
IDENTICAL_NUNU_NUMBER_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT = (
    "pstf_identical_nunu_number_energy_neutral_radial_source_v1"
)
NUNU_NUMBER_NEUTRAL_RADIAL_SOURCE_CONTRACT = (
    "pstf_nunu_number_neutral_radial_source_v1"
)
NUNU_NUMBER_PAIR_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT = (
    "pstf_nunu_number_pair_energy_neutral_radial_source_v1"
)
_PROCESS_RADIAL_MOMENT_WEIGHTS_CONTRACT = "pstf_process_radial_moment_weights_v1"
_PROCESS_RADIAL_MOMENTS_CONTRACT = "pstf_process_radial_moments_v1"
_ELECTRON_MASS_MEV = 0.5109989500
_DEFAULT_CATALOG_SPECIES: tuple[Species, ...] = (
    Species.NUE,
    Species.NUEBAR,
    Species.NUX,
)


@dataclass(frozen=True)
class PSTFWeakProcessDescriptor:
    """Weak 2-to-2 process descriptor for PSTF channel-kernel assembly."""

    process: str
    species: str
    particle_labels: tuple[str, str, str, str]
    bilinear_terms: tuple[PSTFBilinearMatrixElementTerm, ...]
    mass_terms: tuple[PSTFMassMatrixElementTerm, ...] = ()
    mass_quartic_terms: tuple[PSTFMassQuarticMatrixElementTerm, ...] = ()
    partner_species: str | None = None
    electron_mass_momentum: float = 0.0
    coupling_prefactor: float = 1.0
    matrix_element_contract: str = _PROCESS_DESCRIPTOR_CONTRACT

    def __post_init__(self) -> None:
        process = str(self.process)
        species = str(self.species)
        partner = None if self.partner_species is None else str(self.partner_species)
        if not process:
            raise ValueError("process must be a non-empty string.")
        if not species:
            raise ValueError("species must be a non-empty string.")
        if len(self.particle_labels) != 4 or any(not str(label) for label in self.particle_labels):
            raise ValueError("particle_labels must contain four non-empty strings.")
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
        electron_mass = float(self.electron_mass_momentum)
        coupling = float(self.coupling_prefactor)
        if not np.isfinite(electron_mass) or electron_mass < 0.0:
            raise ValueError("electron_mass_momentum must be non-negative and finite.")
        if not np.isfinite(coupling) or coupling < 0.0:
            raise ValueError("coupling_prefactor must be non-negative and finite.")
        if not self.matrix_element_contract:
            raise ValueError("matrix_element_contract must be a non-empty string.")
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "partner_species", partner)
        object.__setattr__(self, "particle_labels", tuple(str(label) for label in self.particle_labels))
        object.__setattr__(self, "bilinear_terms", bilinear)
        object.__setattr__(self, "mass_terms", mass)
        object.__setattr__(self, "mass_quartic_terms", mass_quartic)
        object.__setattr__(self, "electron_mass_momentum", electron_mass)
        object.__setattr__(self, "coupling_prefactor", coupling)


@dataclass(frozen=True)
class PSTFProcessRadialCollisionResult:
    """Concrete process-specific PSTF radial collision source."""

    descriptor: PSTFWeakProcessDescriptor
    grid: PSTFRadialChannelKernelGrid
    contraction: PSTFChannelRadialGridContractionResult
    C_modes: np.ndarray
    process_contract: str = _PROCESS_RADIAL_SOURCE_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, PSTFWeakProcessDescriptor):
            raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
        if not isinstance(self.grid, PSTFRadialChannelKernelGrid):
            raise ValueError("grid must be a PSTFRadialChannelKernelGrid.")
        if not isinstance(self.contraction, PSTFChannelRadialGridContractionResult):
            raise ValueError("contraction must be a PSTFChannelRadialGridContractionResult.")
        modes = np.asarray(self.C_modes, dtype=float)
        if modes.shape != self.contraction.C_modes.shape:
            raise ValueError("C_modes must match contraction.C_modes.")
        if not np.all(np.isfinite(modes)):
            raise ValueError("C_modes must contain only finite values.")
        if not np.array_equal(modes, self.contraction.C_modes):
            raise ValueError("C_modes must equal contraction.C_modes.")
        if not self.process_contract:
            raise ValueError("process_contract must be a non-empty string.")
        object.__setattr__(self, "C_modes", modes)


@dataclass(frozen=True)
class PSTFProcessRadialMomentResult:
    """Number/energy moments of a process-specific radial PSTF source."""

    source: PSTFProcessRadialCollisionResult
    p1_weights: np.ndarray
    p1_energies: np.ndarray
    C_mode: np.ndarray
    mode_index: int
    number_power: float
    energy_power: float
    number_moment: float
    energy_moment: float
    max_abs_C_mode: float
    process: str
    species: str
    moment_contract: str = _PROCESS_RADIAL_MOMENTS_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.source, PSTFProcessRadialCollisionResult):
            raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
        weights = np.asarray(self.p1_weights, dtype=float)
        energies = np.asarray(self.p1_energies, dtype=float)
        mode = np.asarray(self.C_mode, dtype=float)
        if weights.ndim != 1 or energies.ndim != 1 or mode.ndim != 1:
            raise ValueError("p1_weights, p1_energies, and C_mode must be one-dimensional.")
        if weights.shape != energies.shape or mode.shape != energies.shape:
            raise ValueError("p1_weights and C_mode must match the source p1 energy grid.")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(mode)):
            raise ValueError("moment arrays must contain only finite values.")
        if np.any(weights < 0.0) or np.any(energies <= 0.0):
            raise ValueError("p1_weights must be non-negative and p1_energies must be positive.")
        mode_index = int(self.mode_index)
        number_power = float(self.number_power)
        energy_power = float(self.energy_power)
        number = float(self.number_moment)
        energy = float(self.energy_moment)
        max_abs = float(self.max_abs_C_mode)
        if mode_index < 0 or mode_index >= self.source.C_modes.shape[1]:
            raise ValueError("mode_index is out of bounds for source.C_modes.")
        if not np.array_equal(mode, self.source.C_modes[:, mode_index]):
            raise ValueError("C_mode must match source.C_modes at mode_index.")
        if not np.isfinite(number_power) or not np.isfinite(energy_power):
            raise ValueError("moment powers must be finite.")
        if not np.isfinite(number) or not np.isfinite(energy) or not np.isfinite(max_abs):
            raise ValueError("moments must be finite.")
        if max_abs < 0.0:
            raise ValueError("max_abs_C_mode must be non-negative.")
        if str(self.process) != self.source.descriptor.process or str(self.species) != self.source.descriptor.species:
            raise ValueError("process/species must match the source descriptor.")
        if not self.moment_contract:
            raise ValueError("moment_contract must be a non-empty string.")
        object.__setattr__(self, "p1_weights", weights)
        object.__setattr__(self, "p1_energies", energies)
        object.__setattr__(self, "C_mode", mode)
        object.__setattr__(self, "mode_index", mode_index)
        object.__setattr__(self, "number_power", number_power)
        object.__setattr__(self, "energy_power", energy_power)
        object.__setattr__(self, "number_moment", number)
        object.__setattr__(self, "energy_moment", energy)
        object.__setattr__(self, "max_abs_C_mode", max_abs)
        object.__setattr__(self, "process", str(self.process))
        object.__setattr__(self, "species", str(self.species))


@dataclass(frozen=True)
class PSTFProcessRadialMomentWeights:
    """Precomputed p1 radial weights for one process moment extraction."""

    p1_weights: np.ndarray
    p1_energies: np.ndarray
    number_weights: np.ndarray
    energy_weights: np.ndarray
    mode_index: int
    number_power: float
    energy_power: float
    projection_basis_n: np.ndarray | None = None
    projection_basis_e: np.ndarray | None = None
    projection_matrix_pinv: np.ndarray | None = None
    moment_weight_contract: str = _PROCESS_RADIAL_MOMENT_WEIGHTS_CONTRACT

    def __post_init__(self) -> None:
        weights = np.asarray(self.p1_weights, dtype=float)
        energies = np.asarray(self.p1_energies, dtype=float)
        number_weights = np.asarray(self.number_weights, dtype=float)
        energy_weights = np.asarray(self.energy_weights, dtype=float)
        projection_basis_n = (
            None
            if self.projection_basis_n is None
            else np.asarray(self.projection_basis_n, dtype=float)
        )
        projection_basis_e = (
            None
            if self.projection_basis_e is None
            else np.asarray(self.projection_basis_e, dtype=float)
        )
        projection_matrix_pinv = (
            None
            if self.projection_matrix_pinv is None
            else np.asarray(self.projection_matrix_pinv, dtype=float)
        )
        if weights.ndim != 1 or energies.ndim != 1:
            raise ValueError("p1_weights and p1_energies must be one-dimensional.")
        if (
            weights.shape != energies.shape
            or number_weights.shape != energies.shape
            or energy_weights.shape != energies.shape
        ):
            raise ValueError("moment weight arrays must share the p1 energy shape.")
        arrays = (weights, energies, number_weights, energy_weights)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("moment weight arrays must contain only finite values.")
        if (
            projection_basis_n is not None
            or projection_basis_e is not None
            or projection_matrix_pinv is not None
        ):
            if (
                projection_basis_n is None
                or projection_basis_e is None
                or projection_matrix_pinv is None
            ):
                raise ValueError("projection precompute arrays must be supplied together.")
            if projection_basis_n.shape != energies.shape or projection_basis_e.shape != energies.shape:
                raise ValueError("projection basis arrays must match p1_energies.")
            if projection_matrix_pinv.shape != (2, 2):
                raise ValueError("projection_matrix_pinv must have shape (2, 2).")
            if (
                not np.all(np.isfinite(projection_basis_n))
                or not np.all(np.isfinite(projection_basis_e))
                or not np.all(np.isfinite(projection_matrix_pinv))
            ):
                raise ValueError("projection precompute arrays must contain only finite values.")
        if np.any(weights < 0.0) or np.any(energies <= 0.0):
            raise ValueError("p1_weights must be non-negative and p1_energies must be positive.")
        idx = int(self.mode_index)
        number_power = float(self.number_power)
        energy_power = float(self.energy_power)
        if idx < 0:
            raise ValueError("mode_index must be non-negative.")
        if not np.isfinite(number_power) or not np.isfinite(energy_power):
            raise ValueError("moment powers must be finite.")
        if not self.moment_weight_contract:
            raise ValueError("moment_weight_contract must be a non-empty string.")
        object.__setattr__(self, "p1_weights", weights)
        object.__setattr__(self, "p1_energies", energies)
        object.__setattr__(self, "number_weights", number_weights)
        object.__setattr__(self, "energy_weights", energy_weights)
        object.__setattr__(self, "mode_index", idx)
        object.__setattr__(self, "number_power", number_power)
        object.__setattr__(self, "energy_power", energy_power)
        object.__setattr__(self, "projection_basis_n", projection_basis_n)
        object.__setattr__(self, "projection_basis_e", projection_basis_e)
        object.__setattr__(self, "projection_matrix_pinv", projection_matrix_pinv)


def _canonical_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("mu pair entries must contain two particle indices.")
    i, j = int(pair[0]), int(pair[1])
    if i == j or i < 1 or i > 4 or j < 1 or j > 4:
        raise ValueError("mu pair particle indices must be distinct values in 1..4.")
    return (i, j) if i < j else (j, i)


def _weak_couplings_for_species(species: str | Species) -> tuple[str, float, float]:
    try:
        sp = as_species(species)
    except ValueError as exc:
        raise ValueError(f"Unsupported species: {species!r}") from exc
    if sp in (Species.NUE, Species.NUEBAR):
        return sp.value, G_L_NUE, G_R_NUE
    if sp is Species.NUX:
        return sp.value, G_L_NUX, G_R_NUX
    raise ValueError(f"Unsupported species: {species!r}")


def _coupling_prefactor(include_fermi_prefactor: bool) -> float:
    return float(G_F_MEV**2 if include_fermi_prefactor else 1.0)


def pstf_process_descriptor_key(descriptor: PSTFWeakProcessDescriptor) -> str:
    """Return a stable catalog key for a process descriptor."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    if descriptor.partner_species is None:
        return f"{descriptor.process}:{descriptor.species}"
    return f"{descriptor.process}:{descriptor.species}:{descriptor.partner_species}"


def pstf_process_particle_mode_labels(descriptor: PSTFWeakProcessDescriptor) -> tuple[str, str, str, str]:
    """Return the distribution-mode labels needed for the descriptor particles."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    try:
        species = as_species(descriptor.species).value
    except ValueError as exc:
        raise ValueError("descriptor species must be a known collision species.") from exc

    if descriptor.process == "nu_e_elastic_ur":
        return (species, "e_pm", species, "e_pm")
    if descriptor.process == "nu_e_minus_elastic_finite_mass_hm":
        return (species, "e_minus", species, "e_minus")
    if descriptor.process == "nu_e_plus_elastic_finite_mass_hm":
        return (species, "e_plus", species, "e_plus")
    if descriptor.process in ("nu_nubar_to_ee_ur", "nu_nubar_to_ee_finite_mass_hm"):
        if species == Species.NUE.value:
            partner = Species.NUEBAR.value
        elif species == Species.NUEBAR.value:
            partner = Species.NUE.value
        else:
            partner = species
        return (species, partner, "e_plus", "e_minus")
    if descriptor.process == "nu_nu_diagonal_ur":
        if descriptor.partner_species is None:
            raise ValueError("nu_nu_diagonal_ur descriptors must define partner_species.")
        try:
            partner = as_species(descriptor.partner_species).value
        except ValueError as exc:
            raise ValueError("descriptor partner_species must be a known collision species.") from exc
        return (species, partner, species, partner)
    raise ValueError(f"unsupported PSTF process descriptor: {descriptor.process!r}")


def _squared_pair_terms(
    eta: float,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[PSTFBilinearMatrixElementTerm, ...]:
    return tuple(
        PSTFBilinearMatrixElementTerm(eta=float(eta), first_pair=pair, second_pair=pair)
        for pair in pairs
    )


def build_ur_nue_scattering_process_descriptor(
    species: str | Species = "nue",
    *,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return the UR ``nu + e -> nu + e`` HM descriptor used by AP6.

    In the scalar orthogonal-angle limit this reproduces the existing
    deterministic reference formula
    ``(G_L^2+G_R^2)[(y1 y2)^2+(y3 y4)^2] +
    (G_L^2-G_R^2)[(y1 y4)^2+(y2 y3)^2]``.
    """

    sp, G_L, G_R = _weak_couplings_for_species(species)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    particle_1 = "nubar_alpha" if sp == Species.NUEBAR.value else "nu_alpha"
    return PSTFWeakProcessDescriptor(
        process="nu_e_elastic_ur",
        species=sp,
        particle_labels=(particle_1, "e_pm", particle_1, "e_pm"),
        bilinear_terms=(
            *_squared_pair_terms(gl2_gr2, ((1, 2), (3, 4))),
            *_squared_pair_terms(gl2_minus_gr2, ((1, 4), (2, 3))),
        ),
        coupling_prefactor=_coupling_prefactor(include_fermi_prefactor),
    )


def build_finite_mass_nue_scattering_process_descriptor(
    species: str | Species = "nue",
    *,
    charged_lepton: str = "e_minus",
    electron_mass_momentum: float = _ELECTRON_MASS_MEV,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return the finite-mass HM ``nu + e -> nu + e`` descriptor.

    The descriptor implements the finite-mass elastic form

    ``8 [G_L^2 (s-m_e^2)^2 + G_R^2 (u-m_e^2)^2
        + G_L G_R m_e^2 (s+u-2m_e^2)]``

    using ``s-m_e^2 = 2 Pi_12`` and ``u-m_e^2 = -2 Pi_14``.  For
    ``nuebar`` and ``e_plus`` each cross the squared chiral terms; the two
    crossings cancel for ``nuebar + e_plus``.  The optional Fermi prefactor
    preserves the existing catalog convention: coefficients are expressed in
    units of ``G_F^2`` unless ``include_fermi_prefactor=True``.
    """

    sp, G_L, G_R = _weak_couplings_for_species(species)
    lepton = str(charged_lepton)
    if lepton not in ("e_minus", "e_plus"):
        raise ValueError("charged_lepton must be 'e_minus' or 'e_plus'.")
    electron_mass = float(electron_mass_momentum)
    if not np.isfinite(electron_mass) or electron_mass < 0.0:
        raise ValueError("electron_mass_momentum must be non-negative and finite.")

    antineutrino = sp == Species.NUEBAR.value
    positron = lepton == "e_plus"
    crossed = antineutrino ^ positron
    eta_12 = 32.0 * (G_R**2 if crossed else G_L**2)
    eta_14 = 32.0 * (G_L**2 if crossed else G_R**2)
    interference = 16.0 * G_L * G_R
    particle_1 = "nubar_alpha" if antineutrino else "nu_alpha"
    return PSTFWeakProcessDescriptor(
        process=f"nu_{lepton}_elastic_finite_mass_hm",
        species=sp,
        particle_labels=(particle_1, lepton, particle_1, lepton),
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(eta=eta_12, first_pair=(1, 2), second_pair=(1, 2)),
            PSTFBilinearMatrixElementTerm(eta=eta_14, first_pair=(1, 4), second_pair=(1, 4)),
        ),
        mass_terms=(
            PSTFMassMatrixElementTerm(zeta=interference, pair=(1, 2)),
            PSTFMassMatrixElementTerm(zeta=-interference, pair=(1, 4)),
        ),
        electron_mass_momentum=electron_mass,
        coupling_prefactor=_coupling_prefactor(include_fermi_prefactor),
        matrix_element_contract=_FINITE_MASS_NUE_SCATTERING_DESCRIPTOR_CONTRACT,
    )


def build_ur_pair_annihilation_process_descriptor(
    species: str | Species = "nue",
    *,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return the UR ``nu + nubar <-> e+ + e-`` HM descriptor."""

    sp, G_L, G_R = _weak_couplings_for_species(species)
    gl2_gr2 = G_L**2 + G_R**2
    gl2_minus_gr2 = G_L**2 - G_R**2
    if sp == Species.NUEBAR.value:
        particle_labels = ("nubar_alpha", "nu_alpha", "e_plus", "e_minus")
    else:
        particle_labels = ("nu_alpha", "nubar_alpha", "e_plus", "e_minus")
    return PSTFWeakProcessDescriptor(
        process="nu_nubar_to_ee_ur",
        species=sp,
        particle_labels=particle_labels,
        bilinear_terms=(
            *_squared_pair_terms(gl2_gr2, ((1, 3), (2, 4))),
            *_squared_pair_terms(gl2_minus_gr2, ((1, 4), (2, 3))),
        ),
        coupling_prefactor=_coupling_prefactor(include_fermi_prefactor),
    )


def build_finite_mass_pair_annihilation_process_descriptor(
    species: str | Species = "nue",
    *,
    electron_mass_momentum: float = _ELECTRON_MASS_MEV,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return the finite-mass HM ``nu + nubar <-> e+ + e-`` descriptor.

    The descriptor follows the in-tree HM crossing convention
    ``M2_nu_nubar_to_ee(s,t,u) = M2_nu_e_elastic(u,t,s)``.  For a neutrino
    target this uses ``s = 2 Pi_12`` and ``u - m_e^2 = -2 Pi_14``.  For an
    antineutrino target, particle 2 is the incoming neutrino, so the crossed
    term uses ``Pi_24`` while the source still acts on particle 1.
    """

    sp, G_L, G_R = _weak_couplings_for_species(species)
    electron_mass = float(electron_mass_momentum)
    if not np.isfinite(electron_mass) or electron_mass < 0.0:
        raise ValueError("electron_mass_momentum must be non-negative and finite.")

    target_antineutrino = sp == Species.NUEBAR.value
    particle_labels = (
        ("nubar_alpha", "nu_alpha", "e_plus", "e_minus")
        if target_antineutrino
        else ("nu_alpha", "nubar_alpha", "e_plus", "e_minus")
    )
    u_pair = (2, 4) if target_antineutrino else (1, 4)
    return PSTFWeakProcessDescriptor(
        process="nu_nubar_to_ee_finite_mass_hm",
        species=sp,
        particle_labels=particle_labels,
        bilinear_terms=(
            PSTFBilinearMatrixElementTerm(eta=32.0 * G_L**2, first_pair=u_pair, second_pair=u_pair),
            PSTFBilinearMatrixElementTerm(eta=32.0 * G_R**2, first_pair=(1, 2), second_pair=(1, 2)),
        ),
        mass_terms=(
            PSTFMassMatrixElementTerm(zeta=16.0 * G_L * G_R - 32.0 * G_R**2, pair=(1, 2)),
            PSTFMassMatrixElementTerm(zeta=-16.0 * G_L * G_R, pair=u_pair),
        ),
        mass_quartic_terms=(PSTFMassQuarticMatrixElementTerm(zeta=8.0 * (G_R**2 - G_L * G_R)),),
        electron_mass_momentum=electron_mass,
        coupling_prefactor=_coupling_prefactor(include_fermi_prefactor),
        matrix_element_contract=_FINITE_MASS_PAIR_DESCRIPTOR_CONTRACT,
    )


def build_ur_nunu_diagonal_process_descriptor(
    target_species: str | Species = "nue",
    partner_species: str | Species = "nux",
    *,
    epsilon_alpha_beta: float = 1.0,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return the AP81 pairwise diagonal no-QKE ``nu + nu -> nu + nu`` descriptor.

    The coefficient convention matches
    :func:`rabbit.collisions.deterministic_reference.evaluate_nunu_diagonal_twoto2_reference`:
    ``epsilon_alpha_beta * [(y1 y2)^2 + (y3 y4)^2]`` in the scalar
    orthogonal-angle limit.
    """

    try:
        target = as_species(target_species).value
        partner = as_species(partner_species).value
    except ValueError as exc:
        raise ValueError("target_species and partner_species must be known collision species.") from exc
    epsilon = float(epsilon_alpha_beta)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon_alpha_beta must be positive and finite.")
    return PSTFWeakProcessDescriptor(
        process="nu_nu_diagonal_ur",
        species=target,
        partner_species=partner,
        particle_labels=("nu_alpha", "nu_beta", "nu_alpha", "nu_beta"),
        bilinear_terms=_squared_pair_terms(epsilon, ((1, 2), (3, 4))),
        coupling_prefactor=_coupling_prefactor(include_fermi_prefactor),
    )


def build_ur_nunu_pairwise_process_descriptor(
    target_species: str | Species = "nue",
    partner_species: str | Species = "nux",
    *,
    include_fermi_prefactor: bool = False,
) -> PSTFWeakProcessDescriptor:
    """Return a pairwise no-QKE ``nu + nu -> nu + nu`` descriptor.

    The existing AP81 scalar reference exposes the identical/off-diagonal
    matrix-element difference through ``epsilon_alpha_beta``.  This helper
    makes the physical catalog choice explicit: identical target/partner banks
    use the Fierz factor 2, while distinguishable banks use factor 1.
    """

    try:
        target = as_species(target_species)
        partner = as_species(partner_species)
    except ValueError as exc:
        raise ValueError("target_species and partner_species must be known collision species.") from exc
    epsilon = 2.0 if target is partner else 1.0
    return build_ur_nunu_diagonal_process_descriptor(
        target_species=target.value,
        partner_species=partner.value,
        epsilon_alpha_beta=epsilon,
        include_fermi_prefactor=include_fermi_prefactor,
    )


def is_identical_nunu_process_descriptor(descriptor: PSTFWeakProcessDescriptor) -> bool:
    """Return whether a descriptor is same-bank diagonal ``nu-nu`` scattering."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    return (
        descriptor.process == "nu_nu_diagonal_ur"
        and descriptor.partner_species is not None
        and descriptor.partner_species == descriptor.species
    )


def is_nunu_process_descriptor(descriptor: PSTFWeakProcessDescriptor) -> bool:
    """Return whether a descriptor is a diagonal no-QKE ``nu-nu`` process."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    return descriptor.process == "nu_nu_diagonal_ur" and descriptor.partner_species is not None


def build_default_ur_weak_process_catalog(
    *,
    species_labels: tuple[str | Species, ...] = _DEFAULT_CATALOG_SPECIES,
    include_fermi_prefactor: bool = False,
    include_identical_nunu: bool = False,
) -> tuple[PSTFWeakProcessDescriptor, ...]:
    """Return the supported AP6 ultra-relativistic weak-process catalog.

    The catalog enumerates every supported species bank for ``nu-e`` elastic
    scattering and pair annihilation, plus every ordered distinct
    target/partner bank for the no-QKE pairwise diagonal ``nu-nu`` descriptor.
    The default distinct-pair set matches the current staged pairwise bridge;
    identical-bank descriptors remain available as explicit reference entries.
    It is a descriptor catalog only; runtime promotion remains gated elsewhere.
    """

    try:
        species = tuple(as_species(label) for label in species_labels)
    except ValueError as exc:
        raise ValueError("species_labels must contain only known collision species.") from exc
    if not species:
        raise ValueError("species_labels must not be empty.")
    if len(set(species)) != len(species):
        raise ValueError("species_labels must not contain duplicates.")

    descriptors: list[PSTFWeakProcessDescriptor] = []
    for target in species:
        descriptors.append(
            build_ur_nue_scattering_process_descriptor(
                target.value,
                include_fermi_prefactor=include_fermi_prefactor,
            )
        )
        descriptors.append(
            build_ur_pair_annihilation_process_descriptor(
                target.value,
                include_fermi_prefactor=include_fermi_prefactor,
            )
        )
    for target in species:
        for partner in species:
            if not include_identical_nunu and partner is target:
                continue
            descriptors.append(
                build_ur_nunu_pairwise_process_descriptor(
                    target.value,
                    partner.value,
                    include_fermi_prefactor=include_fermi_prefactor,
                )
            )
    keys = [pstf_process_descriptor_key(descriptor) for descriptor in descriptors]
    if len(set(keys)) != len(keys):
        raise ValueError("default process catalog produced duplicate descriptor keys.")
    return tuple(descriptors)


def build_default_supported_weak_process_catalog(
    *,
    species_labels: tuple[str | Species, ...] = _DEFAULT_CATALOG_SPECIES,
    electron_scattering_model: str = "finite_mass_hm",
    electron_mass_momentum: float = _ELECTRON_MASS_MEV,
    include_fermi_prefactor: bool = False,
    include_identical_nunu: bool = True,
) -> tuple[PSTFWeakProcessDescriptor, ...]:
    """Return the current supported AP6 no-QKE weak-process descriptor catalog.

    The supported default uses finite-mass HM ``nu-e`` elastic and
    ``nu-nubar <-> e+e-`` pair descriptors so the AP6 radial route no longer
    combines a finite-mass electron/positron bath with ultra-relativistic
    electromagnetic weak-process matrix elements.  Pairwise diagonal
    ``nu-nu`` descriptors remain the landed UR/no-QKE entries, including the
    identical-bank self-scattering descriptors needed for same-sector shape
    thermalization.  Callers that need the older off-diagonal-only staged
    bridge can pass ``include_identical_nunu=False`` explicitly.
    """

    model = str(electron_scattering_model)
    if model == "ur":
        return build_default_ur_weak_process_catalog(
            species_labels=species_labels,
            include_fermi_prefactor=include_fermi_prefactor,
            include_identical_nunu=include_identical_nunu,
        )
    if model != "finite_mass_hm":
        raise ValueError("electron_scattering_model must be 'finite_mass_hm' or 'ur'.")
    try:
        species = tuple(as_species(label) for label in species_labels)
    except ValueError as exc:
        raise ValueError("species_labels must contain only known collision species.") from exc
    if not species:
        raise ValueError("species_labels must not be empty.")
    if len(set(species)) != len(species):
        raise ValueError("species_labels must not contain duplicates.")

    descriptors: list[PSTFWeakProcessDescriptor] = []
    for target in species:
        for charged_lepton in ("e_minus", "e_plus"):
            descriptors.append(
                build_finite_mass_nue_scattering_process_descriptor(
                    target.value,
                    charged_lepton=charged_lepton,
                    electron_mass_momentum=electron_mass_momentum,
                    include_fermi_prefactor=include_fermi_prefactor,
                )
            )
        descriptors.append(
            build_finite_mass_pair_annihilation_process_descriptor(
                target.value,
                electron_mass_momentum=electron_mass_momentum,
                include_fermi_prefactor=include_fermi_prefactor,
            )
        )
    for target in species:
        for partner in species:
            if not include_identical_nunu and partner is target:
                continue
            descriptors.append(
                build_ur_nunu_pairwise_process_descriptor(
                    target.value,
                    partner.value,
                    include_fermi_prefactor=include_fermi_prefactor,
                )
            )
    keys = [pstf_process_descriptor_key(descriptor) for descriptor in descriptors]
    if len(set(keys)) != len(keys):
        raise ValueError("default supported process catalog produced duplicate descriptor keys.")
    return tuple(descriptors)


def _four_vector(values: tuple[float, float, float, float], *, name: str, positive: bool) -> tuple[float, float, float, float]:
    checked = tuple(float(value) for value in values)
    if len(checked) != 4 or any(not np.isfinite(value) for value in checked):
        raise ValueError(f"{name} must contain four finite values.")
    if positive and any(value <= 0.0 for value in checked):
        raise ValueError(f"{name} entries must be positive.")
    if not positive and any(value < 0.0 for value in checked):
        raise ValueError(f"{name} entries must be non-negative.")
    return checked


def _mu_value(mu_by_pair: Mapping[tuple[int, int], float], pair: tuple[int, int]) -> float:
    canonical = _canonical_pair(pair)
    value = float(mu_by_pair.get(canonical, 0.0))
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("mu_by_pair entries must be finite values in [-1, 1].")
    return value


def _canonical_mu_mapping(mu_by_pair: Mapping[tuple[int, int], float] | None) -> dict[tuple[int, int], float]:
    if mu_by_pair is None:
        return {}
    canonical: dict[tuple[int, int], float] = {}
    for pair, value in mu_by_pair.items():
        key = _canonical_pair(tuple(pair))
        if key in canonical:
            raise ValueError(f"duplicate mu_by_pair entry for pair {key!r}.")
        checked = float(value)
        if not np.isfinite(checked) or checked < -1.0 or checked > 1.0:
            raise ValueError("mu_by_pair entries must be finite values in [-1, 1].")
        canonical[key] = checked
    return canonical


def evaluate_pstf_process_matrix_element(
    descriptor: PSTFWeakProcessDescriptor,
    *,
    energies: tuple[float, float, float, float],
    momenta: tuple[float, float, float, float] | None = None,
    mu_by_pair: Mapping[tuple[int, int], float] | None = None,
    c_light: float = 1.0,
    include_coupling_prefactor: bool = True,
) -> float:
    """Evaluate a process descriptor for one local angular/radial tuple."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    e = _four_vector(energies, name="energies", positive=True)
    c = float(c_light)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c_light must be positive and finite.")
    p = _four_vector(momenta if momenta is not None else tuple(value / c for value in e), name="momenta", positive=False)
    mu = _canonical_mu_mapping(mu_by_pair)

    def pi(pair: tuple[int, int]) -> float:
        i, j = _canonical_pair(pair)
        return float(e[i - 1] * e[j - 1] / c**2 - p[i - 1] * p[j - 1] * _mu_value(mu, (i, j)))

    value = 0.0
    for term in descriptor.bilinear_terms:
        value += term.eta * pi(term.first_pair) * pi(term.second_pair)
    for term in descriptor.mass_terms:
        value += term.zeta * descriptor.electron_mass_momentum**2 * pi(term.pair)
    for term in descriptor.mass_quartic_terms:
        value += term.zeta * descriptor.electron_mass_momentum**4
    if include_coupling_prefactor:
        value *= descriptor.coupling_prefactor
    return float(value)


def build_pstf_process_radial_channel_kernel_grid(
    descriptor: PSTFWeakProcessDescriptor,
    *,
    array_key_memo: MutableMapping[int, tuple[np.ndarray, tuple[object, ...]]] | None = None,
    **radial_grid_kwargs,
) -> PSTFRadialChannelKernelGrid:
    """Build an AP6 radial channel grid directly from a process descriptor."""

    if not isinstance(descriptor, PSTFWeakProcessDescriptor):
        raise ValueError("descriptor must be a PSTFWeakProcessDescriptor.")
    reserved = {
        "bilinear_terms",
        "mass_terms",
        "mass_quartic_terms",
        "electron_mass_momentum",
        "coupling_prefactor",
    }
    overlap = reserved & set(radial_grid_kwargs)
    if overlap:
        raise ValueError(f"process descriptor supplies reserved radial-grid fields: {sorted(overlap)}")
    return build_pstf_radial_channel_kernel_grid(
        **radial_grid_kwargs,
        bilinear_terms=descriptor.bilinear_terms,
        mass_terms=descriptor.mass_terms,
        mass_quartic_terms=descriptor.mass_quartic_terms,
        electron_mass_momentum=descriptor.electron_mass_momentum,
        coupling_prefactor=descriptor.coupling_prefactor,
        array_key_memo=array_key_memo,
    )


def evaluate_pstf_process_radial_collision_source(
    descriptor: PSTFWeakProcessDescriptor,
    *,
    F1_modes: np.ndarray,
    F2_modes: np.ndarray,
    F3_modes: np.ndarray,
    F4_modes: np.ndarray,
    **radial_grid_kwargs,
) -> PSTFProcessRadialCollisionResult:
    """Build and contract a process-specific PSTF radial collision source."""

    grid = build_pstf_process_radial_channel_kernel_grid(descriptor, **radial_grid_kwargs)
    contraction = contract_pstf_radial_channel_kernel_grid(
        F1_modes,
        F2_modes,
        F3_modes,
        F4_modes,
        grid,
    )
    return PSTFProcessRadialCollisionResult(
        descriptor=descriptor,
        grid=grid,
        contraction=contraction,
        C_modes=contraction.C_modes,
    )


def compute_pstf_process_radial_moments(
    source: PSTFProcessRadialCollisionResult,
    *,
    p1_weights: np.ndarray,
    mode_index: int = 0,
    number_power: float = 2.0,
    energy_power: float = 3.0,
) -> PSTFProcessRadialMomentResult:
    """Integrate one radial source mode into number and energy moments.

    The weights are raw caller-supplied quadrature weights on the particle-1
    energy grid.  This keeps the helper independent of whether a caller uses
    Gauss-Laguerre plain-weight conversion, finite-volume bins, or another
    deterministic radial rule.
    """

    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    weights = build_pstf_process_radial_moment_weights(
        source.grid,
        p1_weights=p1_weights,
        mode_index=mode_index,
        number_power=number_power,
        energy_power=energy_power,
    )
    return compute_pstf_process_radial_moments_preweighted(source, weights)


def build_pstf_process_radial_moment_weights(
    grid: PSTFRadialChannelKernelGrid,
    *,
    p1_weights: np.ndarray,
    mode_index: int = 0,
    number_power: float = 2.0,
    energy_power: float = 3.0,
) -> PSTFProcessRadialMomentWeights:
    """Precompute p1 number/energy moment weights for a radial source grid."""

    if not isinstance(grid, PSTFRadialChannelKernelGrid):
        raise ValueError("grid must be a PSTFRadialChannelKernelGrid.")
    weights = np.asarray(p1_weights, dtype=float)
    energies = np.asarray(grid.p1_energies, dtype=float)
    if weights.ndim != 1 or weights.shape != energies.shape:
        raise ValueError("p1_weights must be one-dimensional and match grid.p1_energies.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("p1_weights must contain only finite non-negative values.")
    idx = int(mode_index)
    n_modes = int(grid.K_by_monomial["34"].shape[3])
    if idx < 0 or idx >= n_modes:
        raise ValueError("mode_index is out of bounds for the radial grid mode axis.")
    n_power = float(number_power)
    e_power = float(energy_power)
    if not np.isfinite(n_power) or not np.isfinite(e_power):
        raise ValueError("moment powers must be finite.")
    number_weights = weights * np.power(energies, n_power)
    energy_weights = weights * np.power(energies, e_power)
    f_eq = 1.0 / (np.exp(np.minimum(energies, 500.0)) + 1.0)
    projection_basis_n = f_eq * np.maximum(1.0 - f_eq, 1.0e-30)
    projection_basis_e = energies * projection_basis_n
    projection_matrix = np.asarray(
        [
            [
                float(np.sum(number_weights * projection_basis_n)),
                float(np.sum(number_weights * projection_basis_e)),
            ],
            [
                float(np.sum(energy_weights * projection_basis_n)),
                float(np.sum(energy_weights * projection_basis_e)),
            ],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(projection_matrix)):
        raise ValueError("neutral projection moment matrix must be finite.")
    return PSTFProcessRadialMomentWeights(
        p1_weights=weights,
        p1_energies=energies,
        number_weights=number_weights,
        energy_weights=energy_weights,
        mode_index=idx,
        number_power=n_power,
        energy_power=e_power,
        projection_basis_n=projection_basis_n,
        projection_basis_e=projection_basis_e,
        projection_matrix_pinv=np.linalg.pinv(projection_matrix, rcond=1.0e-14),
    )


def _project_pstf_radial_collision_source_with_moment_correction(
    source: PSTFProcessRadialCollisionResult,
    weights: PSTFProcessRadialMomentWeights,
    *,
    mode_index: int | None,
    correction_number_moment: float,
    correction_energy_moment: float,
    process_contract: str,
) -> PSTFProcessRadialCollisionResult:
    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    if not isinstance(weights, PSTFProcessRadialMomentWeights):
        raise ValueError("weights must be a PSTFProcessRadialMomentWeights.")
    if weights.p1_energies.shape != source.grid.p1_energies.shape or not np.array_equal(
        weights.p1_energies,
        source.grid.p1_energies,
    ):
        raise ValueError("precomputed moment weights must match source.grid.p1_energies.")
    idx = weights.mode_index if mode_index is None else int(mode_index)
    if idx < 0 or idx >= source.C_modes.shape[1]:
        raise ValueError("mode_index is out of bounds for source.C_modes.")
    number = float(correction_number_moment)
    energy = float(correction_energy_moment)
    if not np.isfinite(number) or not np.isfinite(energy):
        raise ValueError("projection correction moments must be finite.")
    q = np.asarray(weights.p1_energies, dtype=float)
    if (
        weights.projection_basis_n is not None
        and weights.projection_basis_e is not None
        and weights.projection_matrix_pinv is not None
    ):
        basis_n = weights.projection_basis_n
        basis_e = weights.projection_basis_e
        projection_matrix_pinv = weights.projection_matrix_pinv
    else:
        f_eq = 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)
        basis_n = f_eq * np.maximum(1.0 - f_eq, 1.0e-30)
        basis_e = q * basis_n
        mat = np.asarray(
            [
                [
                    float(np.sum(weights.number_weights * basis_n)),
                    float(np.sum(weights.number_weights * basis_e)),
                ],
                [
                    float(np.sum(weights.energy_weights * basis_n)),
                    float(np.sum(weights.energy_weights * basis_e)),
                ],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(mat)):
            raise ValueError("neutral projection moments must be finite.")
        projection_matrix_pinv = np.linalg.pinv(mat, rcond=1.0e-14)
    raw = np.asarray(source.C_modes[:, idx], dtype=float)
    rhs = np.asarray([number, energy], dtype=float)
    if not np.all(np.isfinite(rhs)):
        raise ValueError("neutral projection moments must be finite.")
    coeff = projection_matrix_pinv @ rhs
    if not np.all(np.isfinite(coeff)):
        raise ValueError("neutral projection coefficients must be finite.")
    projected_modes = np.array(source.C_modes, dtype=float, copy=True)
    projected_modes[:, idx] = raw - coeff[0] * basis_n - coeff[1] * basis_e
    projected_contraction = PSTFChannelRadialGridContractionResult(
        C_modes=projected_modes,
        p4_energies=source.contraction.p4_energies,
        p4_left_indices=source.contraction.p4_left_indices,
        p4_right_weights=source.contraction.p4_right_weights,
        valid_radial_mask=source.contraction.valid_radial_mask,
        radial_contract=source.contraction.radial_contract,
    )
    return PSTFProcessRadialCollisionResult(
        descriptor=source.descriptor,
        grid=source.grid,
        contraction=projected_contraction,
        C_modes=projected_modes,
        process_contract=process_contract,
    )


def project_pstf_radial_collision_source_number_energy_neutral(
    source: PSTFProcessRadialCollisionResult,
    weights: PSTFProcessRadialMomentWeights,
    *,
    mode_index: int | None = None,
    process_contract: str = IDENTICAL_NUNU_NUMBER_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT,
) -> PSTFProcessRadialCollisionResult:
    """Project one source mode so its number and energy moments vanish.

    Same-bank elastic ``nu-nu`` scattering may reshape the spectrum and angular
    hierarchy, but it must not inject net number or energy into that species.
    The AP6 radial quadrature is finite and asymmetric in the output-particle
    view, so this helper removes the two conserved moment components from the
    source mode before it is mapped into thermo or hierarchy feedback.
    """

    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    if not isinstance(weights, PSTFProcessRadialMomentWeights):
        raise ValueError("weights must be a PSTFProcessRadialMomentWeights.")
    idx = weights.mode_index if mode_index is None else int(mode_index)
    if idx < 0 or idx >= source.C_modes.shape[1]:
        raise ValueError("mode_index is out of bounds for source.C_modes.")
    raw = np.asarray(source.C_modes[:, idx], dtype=float)
    return _project_pstf_radial_collision_source_with_moment_correction(
        source,
        weights,
        mode_index=idx,
        correction_number_moment=float(np.sum(weights.number_weights * raw)),
        correction_energy_moment=float(np.sum(weights.energy_weights * raw)),
        process_contract=process_contract,
    )


def project_pstf_radial_collision_source_number_neutral(
    source: PSTFProcessRadialCollisionResult,
    weights: PSTFProcessRadialMomentWeights,
    *,
    mode_index: int | None = None,
    process_contract: str = NUNU_NUMBER_NEUTRAL_RADIAL_SOURCE_CONTRACT,
) -> PSTFProcessRadialCollisionResult:
    """Project one source mode so its number moment vanishes.

    Off-diagonal diagonal-``nu-nu`` elastic scattering conserves the particle
    number of each participating species while allowing energy exchange between
    banks.  This projection removes only the finite-quadrature number residual
    and preserves the raw energy moment of the output species.
    """

    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    if not isinstance(weights, PSTFProcessRadialMomentWeights):
        raise ValueError("weights must be a PSTFProcessRadialMomentWeights.")
    idx = weights.mode_index if mode_index is None else int(mode_index)
    if idx < 0 or idx >= source.C_modes.shape[1]:
        raise ValueError("mode_index is out of bounds for source.C_modes.")
    raw = np.asarray(source.C_modes[:, idx], dtype=float)
    return _project_pstf_radial_collision_source_with_moment_correction(
        source,
        weights,
        mode_index=idx,
        correction_number_moment=float(np.sum(weights.number_weights * raw)),
        correction_energy_moment=0.0,
        process_contract=process_contract,
    )


def project_pstf_radial_collision_source_energy_shift_number_preserving(
    source: PSTFProcessRadialCollisionResult,
    weights: PSTFProcessRadialMomentWeights,
    *,
    correction_energy_moment: float,
    mode_index: int | None = None,
    process_contract: str = NUNU_NUMBER_PAIR_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT,
) -> PSTFProcessRadialCollisionResult:
    """Shift one source energy moment without changing its number moment.

    Off-diagonal diagonal-``nu-nu`` rows are represented as ordered output
    species sources.  Applying equal energy-moment shifts to the two directions
    of a complete unordered pair removes the finite-quadrature total-energy
    residual while preserving each row's number moment and relative energy
    exchange.
    """

    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    if not isinstance(weights, PSTFProcessRadialMomentWeights):
        raise ValueError("weights must be a PSTFProcessRadialMomentWeights.")
    idx = weights.mode_index if mode_index is None else int(mode_index)
    if idx < 0 or idx >= source.C_modes.shape[1]:
        raise ValueError("mode_index is out of bounds for source.C_modes.")
    return _project_pstf_radial_collision_source_with_moment_correction(
        source,
        weights,
        mode_index=idx,
        correction_number_moment=0.0,
        correction_energy_moment=float(correction_energy_moment),
        process_contract=process_contract,
    )


def compute_pstf_process_radial_moments_preweighted(
    source: PSTFProcessRadialCollisionResult,
    weights: PSTFProcessRadialMomentWeights,
) -> PSTFProcessRadialMomentResult:
    """Integrate one radial source mode using precomputed p1 moment weights."""

    if not isinstance(source, PSTFProcessRadialCollisionResult):
        raise ValueError("source must be a PSTFProcessRadialCollisionResult.")
    if not isinstance(weights, PSTFProcessRadialMomentWeights):
        raise ValueError("weights must be a PSTFProcessRadialMomentWeights.")
    if weights.p1_energies.shape != source.grid.p1_energies.shape or not np.array_equal(
        weights.p1_energies,
        source.grid.p1_energies,
    ):
        raise ValueError("precomputed moment weights must match source.grid.p1_energies.")
    idx = int(weights.mode_index)
    if idx < 0 or idx >= source.C_modes.shape[1]:
        raise ValueError("mode_index is out of bounds for source.C_modes.")
    C_mode = source.C_modes[:, idx]
    number = float(np.sum(weights.number_weights * C_mode))
    energy = float(np.sum(weights.energy_weights * C_mode))
    max_abs = float(np.max(np.abs(C_mode))) if C_mode.size else 0.0
    return PSTFProcessRadialMomentResult(
        source=source,
        p1_weights=weights.p1_weights,
        p1_energies=weights.p1_energies,
        C_mode=C_mode,
        mode_index=idx,
        number_power=weights.number_power,
        energy_power=weights.energy_power,
        number_moment=number,
        energy_moment=energy,
        max_abs_C_mode=max_abs,
        process=source.descriptor.process,
        species=source.descriptor.species,
    )


__all__ = [
    "PSTFWeakProcessDescriptor",
    "PSTFProcessRadialCollisionResult",
    "PSTFProcessRadialMomentResult",
    "PSTFProcessRadialMomentWeights",
    "IDENTICAL_NUNU_NUMBER_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT",
    "NUNU_NUMBER_PAIR_ENERGY_NEUTRAL_RADIAL_SOURCE_CONTRACT",
    "NUNU_NUMBER_NEUTRAL_RADIAL_SOURCE_CONTRACT",
    "build_default_supported_weak_process_catalog",
    "build_default_ur_weak_process_catalog",
    "build_finite_mass_pair_annihilation_process_descriptor",
    "build_finite_mass_nue_scattering_process_descriptor",
    "build_pstf_process_radial_moment_weights",
    "build_pstf_process_radial_channel_kernel_grid",
    "build_ur_nue_scattering_process_descriptor",
    "build_ur_nunu_diagonal_process_descriptor",
    "build_ur_nunu_pairwise_process_descriptor",
    "build_ur_pair_annihilation_process_descriptor",
    "compute_pstf_process_radial_moments",
    "compute_pstf_process_radial_moments_preweighted",
    "evaluate_pstf_process_matrix_element",
    "evaluate_pstf_process_radial_collision_source",
    "is_identical_nunu_process_descriptor",
    "is_nunu_process_descriptor",
    "project_pstf_radial_collision_source_number_energy_neutral",
    "project_pstf_radial_collision_source_energy_shift_number_preserving",
    "project_pstf_radial_collision_source_number_neutral",
    "pstf_process_descriptor_key",
    "pstf_process_particle_mode_labels",
]
