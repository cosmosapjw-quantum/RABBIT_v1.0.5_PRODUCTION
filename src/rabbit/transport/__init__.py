"""
rabbit.transport — PSTF momentum-resolved neutrino hierarchy.

Modules (planned)
-----------------
state.py              HierarchyState: species × q_bins × {monopole, quadrupole}.
                      Pack/unpack for flat ODE vector.
typeI_hierarchy.py    ℓ = 0, 2 hierarchy for Type I (exact: no ℓ ≥ 3 needed).
                      Shear source −(8/15)Σ on quadrupole.
projectors.py         Extract anisotropic stress π and monopole f₀(q)
                      from hierarchy state.
angular_decomposition.py
                      Data contracts for the nonperturbative augmented
                      PSTF / S_N Type I transport surface.

Phase 3 deflation: the AP65 LRS-collisionless and weak-network re-export blocks
(augmented_typeI_collisionless, augmented_typeI_weak_network) were removed with the
audit track (zero live consumers). The surviving augmented substrate — the logit
distribution (augmented_pstf_distribution), stress-moment observables, and the non-LRS
collisionless + nonlinear-transport operators backing the B5 realizability gate — is
retained below.
"""

from rabbit.transport.angular_decomposition import (
    AngularDecompositionSpec,
    AngularMode,
    CollisionMomentResult,
    CollisionQuadratureSpec,
    EllMaxConvergenceSpec,
    active_lrs_modes,
    active_non_lrs_diagonal_modes,
)
from rabbit.transport.augmented_pstf_distribution import (
    distribution_rhs_to_augmented_rhs,
    fermi_dirac_from_logit,
    gram_matrix_from_basis,
    mode_norms_from_basis,
    modes_to_nodal,
    project_nodal_to_modes,
    projection_matrix_from_basis,
    reconstruct_distribution,
)
from rabbit.transport.augmented_typeI_observables import (
    AugmentedStressMoments,
    angular_monopole,
    lrs_quadrupole_kernel,
    non_lrs_quadrupole_kernels,
    stress_moments_from_distribution,
)
from rabbit.transport.augmented_typeI_nonlrs_collisionless import (
    AugmentedNonLRSSourceCollisionlessConfig,
    AugmentedNonLRSSourceCollisionlessRHS,
    AugmentedNonLRSSourceCollisionlessSolveResult,
    NonLRSQuadrupoleSourceRHS,
    NonLRSS2Grid,
    augmented_nonlrs_source_collisionless_rhs,
    build_non_lrs_s2_grid,
    non_lrs_quadrupole_source_rhs,
    run_augmented_nonlrs_source_collisionless_solve,
)
from rabbit.transport.augmented_nonlrs_transport import (
    NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT,
    AugmentedNonLRSNonlinearTransportCoevolutionRHS,
    AugmentedNonLRSNonlinearTransportGrid,
    AugmentedNonLRSNonlinearTransportRHS,
    AugmentedNonLRSNonlinearTransportSolveResult,
    augmented_nonlrs_nonlinear_mode_rhs,
    augmented_nonlrs_nonlinear_nodal_rhs,
    augmented_nonlrs_nonlinear_transport_coevolution_rhs,
    build_nonlrs_nonlinear_transport_grid,
    run_augmented_nonlrs_nonlinear_transport_solve,
)

__all__ = [
    "AngularDecompositionSpec",
    "AngularMode",
    "AugmentedNonLRSSourceCollisionlessConfig",
    "AugmentedNonLRSSourceCollisionlessRHS",
    "AugmentedNonLRSSourceCollisionlessSolveResult",
    "AugmentedNonLRSNonlinearTransportCoevolutionRHS",
    "AugmentedNonLRSNonlinearTransportGrid",
    "AugmentedNonLRSNonlinearTransportRHS",
    "AugmentedNonLRSNonlinearTransportSolveResult",
    "AugmentedStressMoments",
    "CollisionMomentResult",
    "CollisionQuadratureSpec",
    "EllMaxConvergenceSpec",
    "NonLRSQuadrupoleSourceRHS",
    "NonLRSS2Grid",
    "NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT",
    "active_lrs_modes",
    "active_non_lrs_diagonal_modes",
    "angular_monopole",
    "augmented_nonlrs_source_collisionless_rhs",
    "augmented_nonlrs_nonlinear_mode_rhs",
    "augmented_nonlrs_nonlinear_nodal_rhs",
    "augmented_nonlrs_nonlinear_transport_coevolution_rhs",
    "build_non_lrs_s2_grid",
    "build_nonlrs_nonlinear_transport_grid",
    "distribution_rhs_to_augmented_rhs",
    "fermi_dirac_from_logit",
    "gram_matrix_from_basis",
    "lrs_quadrupole_kernel",
    "mode_norms_from_basis",
    "modes_to_nodal",
    "non_lrs_quadrupole_source_rhs",
    "non_lrs_quadrupole_kernels",
    "project_nodal_to_modes",
    "projection_matrix_from_basis",
    "reconstruct_distribution",
    "run_augmented_nonlrs_source_collisionless_solve",
    "run_augmented_nonlrs_nonlinear_transport_solve",
    "stress_moments_from_distribution",
]
