"""Safe public facade for the D-080E fixed-state reuse prototype.

The implementation core is kept separate so a cache-lifetime defect discovered
by the first GREEN run remains auditable.  Matrix-element entries are keyed by
the identity of retained kinematic batches.  Therefore matrix reuse is valid
only when the corresponding kinematic batches are retained for the whole
prepared-state lifetime.  The facade rejects the unsafe policy combination
instead of allowing Python object-id reuse to alias unrelated event batches.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from rabbit.decoupling import _independent_noqke as ind
from scripts.audit import _d080e_prepared_jvp_core as _core

D080EReuseError = _core.D080EReuseError
FixedStateKernelCache = _core.FixedStateKernelCache
PreparedStaticRhs = _core.PreparedStaticRhs
PreparedRhsJvpResult = _core.PreparedRhsJvpResult
PreparedStaticJacobianResult = _core.PreparedStaticJacobianResult
EXPECTED_COMPARATOR_BLOB_SHA = _core.EXPECTED_COMPARATOR_BLOB_SHA
EXPECTED_D079_COLLISION_BLOB_SHA = _core.EXPECTED_D079_COLLISION_BLOB_SHA
EXPECTED_D079_RHS_BLOB_SHA = _core.EXPECTED_D079_RHS_BLOB_SHA
EXPECTED_D080C_RHS_BLOB_SHA = _core.EXPECTED_D080C_RHS_BLOB_SHA
EXPECTED_D080D_JACOBIAN_BLOB_SHA = _core.EXPECTED_D080D_JACOBIAN_BLOB_SHA


@dataclass(frozen=True)
class FixedStateReusePolicy(_core.FixedStateReusePolicy):
    """Validated cache policy.

    Matrix entries may be cached only when kinematic batches are retained,
    because the core uses retained-batch identity as part of the exact cache
    key.  Disabling kinematics therefore requires disabling matrix reuse too.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cache_matrices and not self.cache_kinematics:
            raise ValueError(
                "matrix caching requires retained kinematic batches; "
                "set cache_matrices=False when cache_kinematics=False"
            )


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
    return _core.prepare_static_rhs_reuse(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm_mev,
        temperature_gamma_mev=temperature_gamma_mev,
        config=config,
        electron_mass_mev=electron_mass_mev,
        policy=policy,
    )


def evaluate_prepared_c_only_rhs_jvp(
    prepared: PreparedStaticRhs,
    direction_cloglog: ArrayLike,
) -> PreparedRhsJvpResult:
    return _core.evaluate_prepared_c_only_rhs_jvp(prepared, direction_cloglog)


def evaluate_prepared_c_only_rhs_jvps(
    prepared: PreparedStaticRhs,
    directions_cloglog: ArrayLike,
):
    return _core.evaluate_prepared_c_only_rhs_jvps(prepared, directions_cloglog)


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
    return _core.assemble_prepared_static_jacobian(
        grid=grid,
        pair_cloglog=pair_cloglog,
        temperature_cm_mev=temperature_cm_mev,
        temperature_gamma_mev=temperature_gamma_mev,
        config=config,
        electron_mass_mev=electron_mass_mev,
        policy=policy,
        direction_block_size=direction_block_size,
        verify_serial_oracle=verify_serial_oracle,
    )
