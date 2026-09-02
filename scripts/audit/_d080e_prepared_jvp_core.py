"""Fixed-state reuse prototype for production-order static Jacobians.

D-080E does not alter the frozen collision physics.  It reuses objects that are
mathematically independent of a spectral tangent direction while the state,
``T_cm``, ``T_gamma``, quadrature configuration, and discrete support branch
remain fixed:

- the primal collision action and thermodynamics;
- two-body kinematic batches and support masks;
- weak matrix elements;
- mapped-Legendre basis evaluations.

The existing D-079 tangent assemblers remain the derivative authority.  This
module wraps those frozen functions in a scoped memoization context and exposes
an amortized multi-direction evaluator.  The direction loop is still serial;
this is a reuse/profiling prototype, not yet a vectorized production kernel and
not a solver callback.

Natural units ``hbar=c=k_B=1`` and the packed ordering
``(c_e,c_mu,c_tau,T_gamma,elapsed_time)`` are inherited unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import hashlib
from typing import Any, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit import _d079_collision_jvp as collision_jvp
from scripts.audit._d079_rhs_jvp import evaluate_c_only_rhs_jvp
from scripts.audit._d079_tangent_primitives import (
    TangentSpectralLogits,
    cloglog_chart_tangent,
    matrix,
)
from scripts.audit._d080c_tgamma_rhs import evaluate_tgamma_rhs_column
from scripts.audit._d080d_static_jacobian import rhs_block_relative

FloatArray = NDArray[np.float64]
EXPECTED_COMPARATOR_BLOB_SHA = "de44feee0aa484abe26976c7dc34c579643005b5"
EXPECTED_D079_COLLISION_BLOB_SHA = "591a64702c58a2de265fb88636f186e2d1b7e019"
EXPECTED_D079_RHS_BLOB_SHA = "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5"
EXPECTED_D080C_RHS_BLOB_SHA = "c18feacbd57c9519af14504027b7d465758eb1ef"
EXPECTED_D080D_JACOBIAN_BLOB_SHA = "c577fefaf7a83443a7531e59a283c4f15e8815e1"

_PATCH_OWNER: object | None = None


class D080EReuseError(RuntimeError):
    """Fail-closed error for invalid fixed-state reuse."""


@dataclass(frozen=True)
class FixedStateReusePolicy:
    """Research controls for direction-independent cache families."""

    cache_kinematics: bool = True
    cache_matrices: bool = True
    cache_modal_basis: bool = True
    max_modal_basis_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        if int(self.max_modal_basis_bytes) < 0:
            raise ValueError("max_modal_basis_bytes must be nonnegative")


class _CachedGrid:
    """Duck-typed comparator grid with exact-content modal-basis memoization."""

    def __init__(
        self,
        base: ind.IndependentNoQkeGrid,
        owner: "FixedStateKernelCache",
    ) -> None:
        self._base = base
        self._owner = owner
        self.order = base.order
        self.y_max = base.y_max
        self.nodes = base.nodes
        self.weights = base.weights
        self._legendre_vander = base._legendre_vander
        self._unit_weights = base._unit_weights

    @staticmethod
    def _query_key(value: ArrayLike) -> tuple[tuple[int, ...], bytes]:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        return array.shape, array.tobytes(order="C")

    def modal_basis(self, y: ArrayLike) -> FloatArray:
        cache = self._owner
        cache.modal_basis_requests += 1
        if not cache.policy.cache_modal_basis:
            cache.modal_basis_bypasses += 1
            return np.asarray(self._base.modal_basis(y), dtype=np.float64)
        key = self._query_key(y)
        existing = cache.modal_basis_cache.get(key)
        if existing is not None:
            cache.modal_basis_hits += 1
            return existing
        cache.modal_basis_misses += 1
        value = np.asarray(self._base.modal_basis(y), dtype=np.float64)
        value.setflags(write=False)
        if cache.modal_basis_bytes + value.nbytes <= cache.policy.max_modal_basis_bytes:
            cache.modal_basis_cache[key] = value
            cache.modal_basis_bytes += int(value.nbytes)
        else:
            cache.modal_basis_evictions += 1
        return value

    def modal_coefficients(self, values: ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.nodes.shape or not np.all(np.isfinite(array)):
            raise D080EReuseError("invalid native values for modal coefficients")
        return np.asarray(
            self.modal_basis(self.nodes).T @ (self.weights * array),
            dtype=np.float64,
        )

    def integrate(self, values: ArrayLike, *, power: int = 0) -> float:
        return self._base.integrate(np.asarray(values, dtype=np.float64), power=power)

    def legendre_coefficients(self, values: ArrayLike) -> FloatArray:
        return np.asarray(
            self._base.legendre_coefficients(np.asarray(values, dtype=np.float64)),
            dtype=np.float64,
        )

    def interpolate(self, values: ArrayLike, y: ArrayLike) -> FloatArray:
        return np.asarray(
            self._base.interpolate(
                np.asarray(values, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
            ),
            dtype=np.float64,
        )


class FixedStateKernelCache:
    """Scoped memoization of frozen comparator kinematics and matrix elements."""

    def __init__(
        self,
        grid: ind.IndependentNoQkeGrid,
        policy: FixedStateReusePolicy,
    ) -> None:
        self.policy = policy
        self.grid = _CachedGrid(grid, self)
        self.kinematic_cache: dict[tuple[Any, ...], Any] = {}
        self.self_matrix_cache: dict[tuple[Any, ...], tuple[FloatArray, int, float]] = {}
        self.electron_matrix_cache: dict[tuple[Any, ...], tuple[FloatArray, int, float]] = {}
        self.modal_basis_cache: dict[tuple[tuple[int, ...], bytes], FloatArray] = {}
        self.modal_basis_bytes = 0

        self.kinematic_requests = 0
        self.kinematic_hits = 0
        self.kinematic_misses = 0
        self.kinematic_bypasses = 0
        self.self_matrix_requests = 0
        self.self_matrix_hits = 0
        self.self_matrix_misses = 0
        self.self_matrix_bypasses = 0
        self.electron_matrix_requests = 0
        self.electron_matrix_hits = 0
        self.electron_matrix_misses = 0
        self.electron_matrix_bypasses = 0
        self.modal_basis_requests = 0
        self.modal_basis_hits = 0
        self.modal_basis_misses = 0
        self.modal_basis_bypasses = 0
        self.modal_basis_evictions = 0

        self._original_kinematics = ind._two_body_kinematics
        self._original_self_matrix = ind._self_matrix
        self._original_electron_matrix = ind._electron_matrix

    @staticmethod
    def _array_token(value: ArrayLike) -> tuple[tuple[int, ...], str]:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        return array.shape, digest

    @staticmethod
    def _config_token(config: ind.IndependentCollisionConfig) -> tuple[Any, ...]:
        return (
            int(config.incoming_polar_order),
            int(config.final_polar_order),
            int(config.final_azimuth_order),
            int(config.electron_radial_order),
            float(config.matrix_roundoff_ulps),
        )

    def _cached_kinematics(self, **kwargs: Any) -> Any:
        self.kinematic_requests += 1
        if not self.policy.cache_kinematics:
            self.kinematic_bypasses += 1
            return self._original_kinematics(**kwargs)
        key = (
            float(kwargs["p1"]),
            self._array_token(kwargs["p2_nodes"]),
            self._array_token(kwargs["p2_weights"]),
            float(kwargs["mass2"]),
            float(kwargs["mass3"]),
            float(kwargs["mass4"]),
            self._config_token(kwargs["config"]),
        )
        existing = self.kinematic_cache.get(key)
        if existing is not None:
            self.kinematic_hits += 1
            return existing
        self.kinematic_misses += 1
        value = self._original_kinematics(**kwargs)
        self.kinematic_cache[key] = value
        return value

    def _cached_self_matrix(
        self,
        reaction: Any,
        batch: Any,
        config: ind.IndependentCollisionConfig,
    ) -> tuple[FloatArray, int, float]:
        self.self_matrix_requests += 1
        if not self.policy.cache_matrices:
            self.self_matrix_bypasses += 1
            return self._original_self_matrix(reaction, batch, config)
        key = (
            id(batch),
            str(reaction.kernel),
            float(reaction.coefficient),
            self._config_token(config),
        )
        existing = self.self_matrix_cache.get(key)
        if existing is not None:
            self.self_matrix_hits += 1
            return existing
        self.self_matrix_misses += 1
        value = self._original_self_matrix(reaction, batch, config)
        array = np.asarray(value[0], dtype=np.float64)
        array.setflags(write=False)
        frozen = (array, int(value[1]), float(value[2]))
        self.self_matrix_cache[key] = frozen
        return frozen

    def _cached_electron_matrix(
        self,
        target: str,
        category: str,
        batch: Any,
        electron_mass: float,
        config: ind.IndependentCollisionConfig,
    ) -> tuple[FloatArray, int, float]:
        self.electron_matrix_requests += 1
        if not self.policy.cache_matrices:
            self.electron_matrix_bypasses += 1
            return self._original_electron_matrix(
                target, category, batch, electron_mass, config
            )
        key = (
            id(batch),
            str(target),
            str(category),
            float(electron_mass),
            self._config_token(config),
        )
        existing = self.electron_matrix_cache.get(key)
        if existing is not None:
            self.electron_matrix_hits += 1
            return existing
        self.electron_matrix_misses += 1
        value = self._original_electron_matrix(
            target, category, batch, electron_mass, config
        )
        array = np.asarray(value[0], dtype=np.float64)
        array.setflags(write=False)
        frozen = (array, int(value[1]), float(value[2]))
        self.electron_matrix_cache[key] = frozen
        return frozen

    @contextmanager
    def patch(self) -> Iterator[None]:
        """Temporarily route frozen private helpers through this exact cache."""

        global _PATCH_OWNER
        if _PATCH_OWNER is not None:
            raise D080EReuseError("nested or concurrent fixed-state cache patch")
        _PATCH_OWNER = self
        if (
            ind._two_body_kinematics is not self._original_kinematics
            or ind._self_matrix is not self._original_self_matrix
            or ind._electron_matrix is not self._original_electron_matrix
        ):
            _PATCH_OWNER = None
            raise D080EReuseError("frozen comparator helper was already replaced")
        ind._two_body_kinematics = self._cached_kinematics
        ind._self_matrix = self._cached_self_matrix
        ind._electron_matrix = self._cached_electron_matrix
        try:
            yield
        finally:
            ind._two_body_kinematics = self._original_kinematics
            ind._self_matrix = self._original_self_matrix
            ind._electron_matrix = self._original_electron_matrix
            _PATCH_OWNER = None

    @staticmethod
    def _collect_array_bytes(value: Any, seen: set[int]) -> int:
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            return int(value.nbytes)
        if is_dataclass(value):
            return sum(
                FixedStateKernelCache._collect_array_bytes(
                    getattr(value, field.name), seen
                )
                for field in fields(value)
            )
        if isinstance(value, dict):
            return sum(
                FixedStateKernelCache._collect_array_bytes(item, seen)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return sum(
                FixedStateKernelCache._collect_array_bytes(item, seen)
                for item in value
            )
        return 0

    def estimated_cache_bytes(self) -> int:
        seen: set[int] = set()
        return sum(
            self._collect_array_bytes(value, seen)
            for value in (
                self.kinematic_cache,
                self.self_matrix_cache,
                self.electron_matrix_cache,
                self.modal_basis_cache,
            )
        )

    def snapshot(self) -> dict[str, int]:
        matrix_hits = self.self_matrix_hits + self.electron_matrix_hits
        matrix_misses = self.self_matrix_misses + self.electron_matrix_misses
        return {
            "kinematic_requests": self.kinematic_requests,
            "kinematic_hits": self.kinematic_hits,
            "kinematic_misses": self.kinematic_misses,
            "kinematic_bypasses": self.kinematic_bypasses,
            "self_matrix_requests": self.self_matrix_requests,
            "self_matrix_hits": self.self_matrix_hits,
            "self_matrix_misses": self.self_matrix_misses,
            "electron_matrix_requests": self.electron_matrix_requests,
            "electron_matrix_hits": self.electron_matrix_hits,
            "electron_matrix_misses": self.electron_matrix_misses,
            "matrix_hits": matrix_hits,
            "matrix_misses": matrix_misses,
            "modal_basis_requests": self.modal_basis_requests,
            "modal_basis_hits": self.modal_basis_hits,
            "modal_basis_misses": self.modal_basis_misses,
            "modal_basis_bypasses": self.modal_basis_bypasses,
            "modal_basis_evictions": self.modal_basis_evictions,
            "kinematic_entries": len(self.kinematic_cache),
            "self_matrix_entries": len(self.self_matrix_cache),
            "electron_matrix_entries": len(self.electron_matrix_cache),
            "modal_basis_entries": len(self.modal_basis_cache),
            "estimated_cache_bytes": self.estimated_cache_bytes(),
        }


@dataclass
class PreparedStaticRhs:
    """Direction-independent state for repeated exact spectral JVPs."""

    raw_grid: ind.IndependentNoQkeGrid
    grid: _CachedGrid
    cache: FixedStateKernelCache
    pair_cloglog: FloatArray
    temperature_cm_mev: float
    temperature_gamma_mev: float
    electron_mass_mev: float
    config: ind.IndependentCollisionConfig
    spectra: Any
    base_action: ind.IndependentCollisionAction
    thermodynamics: ind.IndependentThermodynamics
    chain: FloatArray
    base_pair_action: FloatArray
    base_c_rate: FloatArray
    base_temperature_rate: float
    base_time_rate: float
    base_rhs: FloatArray


@dataclass(frozen=True)
class PreparedRhsJvpResult:
    base_rhs: FloatArray
    jvp: FloatArray
    direction_cloglog: FloatArray
    delta_rho_neutrino: float
    delta_hubble_over_hubble: float
    neutrino_energy_transfer: float
    electron_bath_energy_transfer: float
    first_law_tangent_residual: float
    cache_snapshot: dict[str, int]


@dataclass(frozen=True)
class PreparedStaticJacobianResult:
    base_rhs: FloatArray
    jacobian: FloatArray
    spectral_columns: FloatArray
    tgamma_column: FloatArray
    elapsed_column: FloatArray
    maximum_serial_oracle_residual: float
    cache_snapshot: dict[str, int]


def _finite_vector(name: str, value: ArrayLike, size: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise D080EReuseError(f"{name} must be a finite vector of length {size}")
    return array.copy()


def prepare_static_rhs_reuse(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
    policy: FixedStateReusePolicy = FixedStateReusePolicy(),
) -> PreparedStaticRhs:
    """Evaluate the primal operator once and populate exact fixed-state caches."""

    c = matrix("pair_cloglog", pair_cloglog, (3, grid.order))
    tcm = float(temperature_cm_mev)
    tg = float(temperature_gamma_mev)
    mass = float(electron_mass_mev)
    if not all(np.isfinite(value) for value in (tcm, tg, mass)):
        raise ValueError("thermodynamic inputs must be finite")
    if min(tcm, tg) <= 0.0 or mass < 0.0:
        raise ValueError("temperatures must be positive and mass nonnegative")

    cache = FixedStateKernelCache(grid, policy)
    cached_grid = cache.grid
    spectra = ind._SpectralLogits(cached_grid, ind._native_pair_logits(c))
    with cache.patch():
        base_action = ind.evaluate_independent_collision_action(
            grid=cached_grid,
            pair_cloglog=c,
            temperature_cm_mev=tcm,
            temperature_gamma_mev=tg,
            config=config,
            electron_mass_mev=mass,
        )
    thermo = ind.independent_thermodynamics(
        grid=cached_grid,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        electron_mass_mev=mass,
    )
    chain = ind.cloglog_chain_factor(c)
    total = np.asarray(base_action.total, dtype=np.float64)
    base_pair = 0.5 * np.stack(
        (
            total[0] + total[1],
            total[2] + total[3],
            total[4] + total[5],
        )
    )
    base_c_rate = base_pair / (thermo.hubble_mev * chain)
    eos = ind.electromagnetic_eos_adaptive(tg, mass)
    base_temperature_rate = (
        -3.0 * (eos.rho + eos.pressure)
        + base_action.electron_bath_energy_transfer / thermo.hubble_mev
    ) / eos.drho_dtemperature
    base_time_rate = 1.0 / thermo.hubble_mev
    base_rhs = np.concatenate(
        (base_c_rate.ravel(), [base_temperature_rate, base_time_rate])
    )
    if not np.all(np.isfinite(base_rhs)):
        raise D080EReuseError("nonfinite prepared base RHS")
    return PreparedStaticRhs(
        raw_grid=grid,
        grid=cached_grid,
        cache=cache,
        pair_cloglog=c,
        temperature_cm_mev=tcm,
        temperature_gamma_mev=tg,
        electron_mass_mev=mass,
        config=config,
        spectra=spectra,
        base_action=base_action,
        thermodynamics=thermo,
        chain=np.asarray(chain, dtype=np.float64),
        base_pair_action=np.asarray(base_pair, dtype=np.float64),
        base_c_rate=np.asarray(base_c_rate, dtype=np.float64),
        base_temperature_rate=float(base_temperature_rate),
        base_time_rate=float(base_time_rate),
        base_rhs=np.asarray(base_rhs, dtype=np.float64),
    )


def evaluate_prepared_c_only_rhs_jvp(
    prepared: PreparedStaticRhs,
    direction_cloglog: ArrayLike,
) -> PreparedRhsJvpResult:
    """Apply the frozen D-079 spectral derivative using fixed-state reuse."""

    grid = prepared.grid
    c = prepared.pair_cloglog
    v = matrix("direction_cloglog", direction_cloglog, c.shape)
    tangent = TangentSpectralLogits(grid, c, v)
    with prepared.cache.patch():
        self_modal, _self_energy_residual = collision_jvp._assemble_self_jvp(
            grid,
            prepared.spectra,
            tangent,
            prepared.temperature_cm_mev,
            prepared.config,
        )
        electron_modal, dqnu, dqem = collision_jvp._assemble_electron_jvp(
            grid,
            prepared.spectra,
            tangent,
            prepared.temperature_cm_mev,
            prepared.temperature_gamma_mev,
            prepared.electron_mass_mev,
            prepared.config,
        )
    self_action = ind._native_action(
        grid, self_modal, prepared.temperature_cm_mev
    )
    electron_action = ind._native_action(
        grid, electron_modal, prepared.temperature_cm_mev
    )
    total = self_action + electron_action
    total_pair = 0.5 * np.stack(
        (
            total[0] + total[1],
            total[2] + total[3],
            total[4] + total[5],
        )
    )

    df, _du, dlog_chain = cloglog_chart_tangent(c, v)
    energy_weights = grid.weights * np.power(grid.nodes, 3)
    delta_rho_neutrino = (
        2.0
        * prepared.temperature_cm_mev**4
        * np.sum(df * energy_weights[None, :], dtype=np.float64)
        / ind.TWO_PI_SQUARED
    )
    delta_hubble_over_hubble = (
        0.5
        * delta_rho_neutrino
        / prepared.thermodynamics.energy_density_total
    )
    tangent_c_rate = total_pair / (
        prepared.thermodynamics.hubble_mev * prepared.chain
    ) - prepared.base_c_rate * (
        delta_hubble_over_hubble + dlog_chain
    )

    eos = ind.electromagnetic_eos_adaptive(
        prepared.temperature_gamma_mev,
        prepared.electron_mass_mev,
    )
    base_qem = float(prepared.base_action.electron_bath_energy_transfer)
    tangent_temperature_rate = (
        dqem / prepared.thermodynamics.hubble_mev
        - (base_qem / prepared.thermodynamics.hubble_mev)
        * delta_hubble_over_hubble
    ) / eos.drho_dtemperature
    tangent_time_rate = (
        -prepared.base_time_rate * delta_hubble_over_hubble
    )
    jvp = np.concatenate(
        (
            tangent_c_rate.ravel(),
            [tangent_temperature_rate, tangent_time_rate],
        )
    )
    size = 3 * grid.order + 2
    jvp = _finite_vector("prepared c-only RHS JVP", jvp, size)
    denominator = max(abs(dqnu) + abs(dqem), np.finfo(np.float64).tiny)
    return PreparedRhsJvpResult(
        base_rhs=_finite_vector("prepared base RHS", prepared.base_rhs, size),
        jvp=jvp,
        direction_cloglog=v,
        delta_rho_neutrino=float(delta_rho_neutrino),
        delta_hubble_over_hubble=float(delta_hubble_over_hubble),
        neutrino_energy_transfer=float(dqnu),
        electron_bath_energy_transfer=float(dqem),
        first_law_tangent_residual=float(abs(dqnu + dqem) / denominator),
        cache_snapshot=prepared.cache.snapshot(),
    )


def evaluate_prepared_c_only_rhs_jvps(
    prepared: PreparedStaticRhs,
    directions_cloglog: ArrayLike,
) -> FloatArray:
    """Apply repeated exact JVPs while amortizing fixed-state data.

    The direction axis is deliberately evaluated serially in this prototype.
    D-080E measures the remaining marginal cost before a true vectorized event
    tape is considered.
    """

    directions = np.asarray(directions_cloglog, dtype=np.float64)
    expected = (3, prepared.grid.order)
    if (
        directions.ndim != 3
        or directions.shape[1:] != expected
        or directions.shape[0] <= 0
        or not np.all(np.isfinite(directions))
    ):
        raise ValueError(
            f"directions must have finite shape (k,{expected[0]},{expected[1]})"
        )
    outputs = [
        evaluate_prepared_c_only_rhs_jvp(prepared, direction).jvp
        for direction in directions
    ]
    return np.asarray(np.stack(outputs), dtype=np.float64)


def assemble_prepared_static_jacobian(
    *,
    grid: ind.IndependentNoQkeGrid,
    pair_cloglog: ArrayLike,
    temperature_cm_mev: float,
    temperature_gamma_mev: float,
    config: ind.IndependentCollisionConfig = ind.IndependentCollisionConfig(),
    electron_mass_mev: float = ind.M_ELECTRON_MEV,
    policy: FixedStateReusePolicy = FixedStateReusePolicy(),
    direction_block_size: int = 8,
    verify_serial_oracle: bool = True,
) -> PreparedStaticJacobianResult:
    """Assemble a same-physics square matrix using the reuse prototype."""

    block_size = int(direction_block_size)
    if block_size <= 0:
        raise ValueError("direction_block_size must be positive")
    prepared = prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm_mev,
        temperature_gamma_mev=temperature_gamma_mev,
        config=config,
        electron_mass_mev=electron_mass_mev,
        policy=policy,
    )
    with prepared.cache.patch():
        tgamma = evaluate_tgamma_rhs_column(
            grid=prepared.grid,
            pair_cloglog=prepared.pair_cloglog,
            temperature_cm_mev=prepared.temperature_cm_mev,
            temperature_gamma_mev=prepared.temperature_gamma_mev,
            config=prepared.config,
            electron_mass_mev=prepared.electron_mass_mev,
        )
    if rhs_block_relative(
        prepared.base_rhs, tgamma.base_rhs, grid.order
    ) > 5.0e-13:
        raise D080EReuseError("prepared and T_gamma base operators disagree")

    spectral_size = 3 * grid.order
    state_size = spectral_size + 2
    spectral_columns = np.empty((state_size, spectral_size), dtype=np.float64)
    maximum_oracle_residual = 0.0
    for start in range(0, spectral_size, block_size):
        stop = min(start + block_size, spectral_size)
        directions = np.zeros((stop - start, 3, grid.order), dtype=np.float64)
        for local, column in enumerate(range(start, stop)):
            directions[local].ravel()[column] = 1.0
        values = evaluate_prepared_c_only_rhs_jvps(prepared, directions)
        spectral_columns[:, start:stop] = values.T
        if verify_serial_oracle:
            for direction, candidate in zip(directions, values):
                reference = evaluate_c_only_rhs_jvp(
                    grid=grid,
                    pair_cloglog=prepared.pair_cloglog,
                    direction_cloglog=direction,
                    temperature_cm_mev=prepared.temperature_cm_mev,
                    temperature_gamma_mev=prepared.temperature_gamma_mev,
                    config=prepared.config,
                    electron_mass_mev=prepared.electron_mass_mev,
                )
                maximum_oracle_residual = max(
                    maximum_oracle_residual,
                    rhs_block_relative(candidate, reference.jvp, grid.order),
                )

    jacobian = np.zeros((state_size, state_size), dtype=np.float64)
    jacobian[:, :spectral_size] = spectral_columns
    jacobian[:, spectral_size] = tgamma.tgamma_column
    elapsed_column = jacobian[:, -1]
    if np.any(elapsed_column != 0.0) or not np.all(np.isfinite(jacobian)):
        raise D080EReuseError("invalid prepared square Jacobian")
    return PreparedStaticJacobianResult(
        base_rhs=_finite_vector("prepared base RHS", prepared.base_rhs, state_size),
        jacobian=np.asarray(jacobian, dtype=np.float64),
        spectral_columns=np.asarray(spectral_columns, dtype=np.float64),
        tgamma_column=np.asarray(tgamma.tgamma_column, dtype=np.float64),
        elapsed_column=np.asarray(elapsed_column, dtype=np.float64),
        maximum_serial_oracle_residual=float(maximum_oracle_residual),
        cache_snapshot=prepared.cache.snapshot(),
    )
