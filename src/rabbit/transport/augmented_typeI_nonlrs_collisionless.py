"""Non-LRS augmented Type I collisionless staging helpers.

This module locks the diagonal Type I quadrupole source projection on an S2
grid and provides a SciPy reference coevolution shell around that source-only
transport block.  It is a reduction gate for future nonlinear non-LRS transport
wiring, not the full nonlinear collisionless transport operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

from rabbit.transport.augmented_pstf_distribution import (
    gram_matrix_from_basis,
    project_nodal_to_modes,
    projection_matrix_from_basis,
    reconstruct_distribution,
)
from rabbit.transport.augmented_typeI_observables import (
    AugmentedStressMoments,
    non_lrs_quadrupole_kernels,
    stress_moments_from_distribution,
)


@dataclass(frozen=True)
class NonLRSS2Grid:
    """Tensor-product S2 angular grid for diagonal non-LRS Type I staging."""

    mu: np.ndarray
    phi: np.ndarray
    angular_weights: np.ndarray
    W_plus: np.ndarray
    W_minus: np.ndarray
    basis_matrix: np.ndarray
    contract: str = "tensor_product_gauss_legendre_mu_uniform_midpoint_phi_v1"

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float)
        phi = np.asarray(self.phi, dtype=float)
        weights = np.asarray(self.angular_weights, dtype=float)
        W_plus = np.asarray(self.W_plus, dtype=float)
        W_minus = np.asarray(self.W_minus, dtype=float)
        basis = np.asarray(self.basis_matrix, dtype=float)
        if mu.ndim != 1 or mu.size == 0:
            raise ValueError("mu must be a non-empty 1D array.")
        if phi.shape != mu.shape or weights.shape != mu.shape:
            raise ValueError("phi and angular_weights must match mu.")
        if W_plus.shape != mu.shape or W_minus.shape != mu.shape:
            raise ValueError("quadrupole kernels must match mu.")
        if basis.shape != (3, mu.size):
            raise ValueError("basis_matrix must have shape (three non-LRS modes, n_angles).")
        expected_basis = np.vstack((np.ones_like(mu), W_plus, W_minus))
        if not np.allclose(basis, expected_basis, rtol=1.0e-13, atol=1.0e-13):
            raise ValueError("basis_matrix must equal the {monopole, W_plus, W_minus} staging basis.")
        arrays = (mu, phi, weights, W_plus, W_minus, basis)
        if any(not np.all(np.isfinite(arr)) for arr in arrays):
            raise ValueError("non-LRS S2 grid arrays must contain only finite values.")
        if np.any(weights <= 0.0):
            raise ValueError("angular_weights must be strictly positive.")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("angular_weights must have positive total weight.")
        gram = gram_matrix_from_basis(basis, weights, max_condition=1.0e10)
        projection = projection_matrix_from_basis(basis, weights, max_condition=1.0e10)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "angular_weights", weights)
        object.__setattr__(self, "W_plus", W_plus)
        object.__setattr__(self, "W_minus", W_minus)
        object.__setattr__(self, "basis_matrix", basis)
        object.__setattr__(self, "gram_matrix", gram)
        object.__setattr__(self, "projection_matrix", projection)


@dataclass(frozen=True)
class NonLRSQuadrupoleSourceRHS:
    """Projected quadrupole source RHS for the three-mode non-LRS staging basis."""

    dA_modes: np.ndarray
    basis_matrix: np.ndarray
    angular_weights: np.ndarray
    closure: str = "quadrupole_source_projection_only_v1"


@dataclass(frozen=True)
class AugmentedNonLRSSourceCollisionlessConfig:
    """Configuration for the non-LRS source-only collisionless solve shell."""

    N_mu: int = 16
    N_phi: int = 16
    f_nu: float = 0.40520
    feedback_factor: float = 6.0


@dataclass(frozen=True)
class AugmentedNonLRSSourceCollisionlessRHS:
    """Geometry and source-projected transport derivative for one RHS call."""

    dSigma_plus: float
    dSigma_minus: float
    dA_modes: np.ndarray
    Pi_plus: float
    Pi_minus: float
    moments: AugmentedStressMoments
    closure: str = "non_lrs_quadrupole_source_coevolution_v1"


@dataclass(frozen=True)
class AugmentedNonLRSSourceCollisionlessSolveResult:
    """Result from the non-LRS source-only SciPy reference solve."""

    success: bool
    message: str
    N: np.ndarray
    Sigma_plus: np.ndarray
    Sigma_minus: np.ndarray
    A_modes_final: np.ndarray
    Pi_plus_final: float
    Pi_minus_final: float
    moments_final: AugmentedStressMoments
    q_nodes: np.ndarray
    q_energy_weights: np.ndarray
    grid: NonLRSS2Grid
    closure: str = "non_lrs_quadrupole_source_coevolution_v1"


def _exact_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer.")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an exact integer.") from exc
    return int(out)


def build_non_lrs_s2_grid(*, N_mu: int = 16, N_phi: int = 16) -> NonLRSS2Grid:
    """Build an S2 grid and the `{monopole, W_plus, W_minus}` staging basis."""

    n_mu = _exact_int(N_mu, name="N_mu")
    n_phi = _exact_int(N_phi, name="N_phi")
    if n_mu < 3:
        raise ValueError("N_mu must be at least 3 for the three-mode non-LRS basis.")
    if n_phi < 5:
        raise ValueError("N_phi must be at least 5 for the three-mode non-LRS basis.")

    mu_1d, w_mu = leggauss(n_mu)
    phi_1d = (np.arange(n_phi, dtype=float) + 0.5) * (2.0 * np.pi / n_phi)
    mu = np.broadcast_to(mu_1d[:, None], (n_mu, n_phi)).reshape(-1)
    phi = np.broadcast_to(phi_1d[None, :], (n_mu, n_phi)).reshape(-1)
    weights = np.broadcast_to(
        w_mu[:, None] * (2.0 * np.pi / n_phi),
        (n_mu, n_phi),
    ).reshape(-1)
    W_plus, W_minus = non_lrs_quadrupole_kernels(mu, phi)
    basis = np.vstack((np.ones_like(mu), W_plus, W_minus))
    return NonLRSS2Grid(
        mu=mu,
        phi=phi,
        angular_weights=weights,
        W_plus=W_plus,
        W_minus=W_minus,
        basis_matrix=basis,
    )


def _as_grid(grid: NonLRSS2Grid | None) -> NonLRSS2Grid:
    return build_non_lrs_s2_grid() if grid is None else grid


def _grid_from_config(
    grid: NonLRSS2Grid | None,
    config: AugmentedNonLRSSourceCollisionlessConfig,
) -> NonLRSS2Grid:
    if grid is not None:
        return grid
    return build_non_lrs_s2_grid(N_mu=config.N_mu, N_phi=config.N_phi)


def _validate_source_collisionless_config(
    config: AugmentedNonLRSSourceCollisionlessConfig,
) -> None:
    _exact_int(config.N_mu, name="N_mu")
    _exact_int(config.N_phi, name="N_phi")
    if not np.isfinite(float(config.f_nu)) or float(config.f_nu) < 0.0:
        raise ValueError("f_nu must be finite and non-negative.")
    if not np.isfinite(float(config.feedback_factor)):
        raise ValueError("feedback_factor must be finite.")


def _default_q_energy_grid(N_q: int) -> tuple[np.ndarray, np.ndarray]:
    n_q = _exact_int(N_q, name="N_q")
    if n_q <= 0:
        raise ValueError("N_q must be positive.")
    q, w = laggauss(n_q)
    return q, w * np.exp(q) * q**3


def non_lrs_quadrupole_source_rhs(
    Sigma_plus: float,
    Sigma_minus: float,
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    *,
    grid: NonLRSS2Grid | None = None,
) -> NonLRSQuadrupoleSourceRHS:
    """Project the diagonal Type I quadrupole source onto the S2 staging basis.

    The source-only nodal equation is

    `dA/dN = -2 q (Sigma_plus W_plus + Sigma_minus W_minus)`.

    Nonlinear angular advection and q-cascade terms are intentionally outside
    this reduction gate.
    """

    g = _as_grid(grid)
    A = np.asarray(A_modes, dtype=float)
    q = np.asarray(q_nodes, dtype=float)
    if A.ndim != 3:
        raise ValueError("A_modes must have shape (n_species, three non-LRS modes, n_q).")
    if A.shape[0] == 0:
        raise ValueError("A_modes must contain at least one species.")
    if A.shape[1] != 3:
        raise ValueError("A_modes must contain exactly three non-LRS modes.")
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_nodes must be a non-empty 1D array.")
    if q.shape != (A.shape[-1],):
        raise ValueError("q_nodes must match the q axis of A_modes.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(q)):
        raise ValueError("A_modes and q_nodes must contain only finite values.")
    if np.any(q <= 0.0):
        raise ValueError("q_nodes must be strictly positive.")
    if q.size > 1 and np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nodes must be strictly increasing.")
    if not np.isfinite(float(Sigma_plus)) or not np.isfinite(float(Sigma_minus)):
        raise ValueError("Sigma_plus and Sigma_minus must be finite.")

    source = -2.0 * q[:, None] * (
        float(Sigma_plus) * g.W_plus[None, :]
        + float(Sigma_minus) * g.W_minus[None, :]
    )
    source = np.broadcast_to(source[None, :, :], (A.shape[0], q.size, g.mu.size))
    dA = project_nodal_to_modes(
        source,
        g.basis_matrix,
        g.angular_weights,
        projection_matrix=g.projection_matrix,
    )
    return NonLRSQuadrupoleSourceRHS(
        dA_modes=np.asarray(dA, dtype=float),
        basis_matrix=g.basis_matrix,
        angular_weights=g.angular_weights,
    )


def augmented_nonlrs_source_collisionless_rhs(
    Sigma_plus: float,
    Sigma_minus: float,
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    q_energy_weights: np.ndarray,
    *,
    grid: NonLRSS2Grid | None = None,
    config: AugmentedNonLRSSourceCollisionlessConfig | None = None,
) -> AugmentedNonLRSSourceCollisionlessRHS:
    """Evaluate the non-LRS source-only augmented Type I coevolution RHS.

    The transport block intentionally uses the AP4 source projection
    ``dA/dN = -2 q (Sigma_plus W_plus + Sigma_minus W_minus)``.  The
    geometry block is live: it reconstructs the augmented distribution,
    extracts ``Pi_+`` and ``Pi_-`` stress moments, and feeds them into the
    diagonal plus/minus shear equations.  Nonlinear angular advection and the
    full non-LRS q-cascade remain outside this staged shell.
    """

    cfg = config or AugmentedNonLRSSourceCollisionlessConfig()
    _validate_source_collisionless_config(cfg)
    g = _grid_from_config(grid, cfg)
    A = np.asarray(A_modes, dtype=float)
    q = np.asarray(q_nodes, dtype=float)
    q_weights = np.asarray(q_energy_weights, dtype=float)
    source = non_lrs_quadrupole_source_rhs(
        Sigma_plus,
        Sigma_minus,
        A,
        q,
        grid=g,
    )
    if q_weights.shape != q.shape:
        raise ValueError("q_energy_weights must match q_nodes.")
    if not np.all(np.isfinite(q_weights)):
        raise ValueError("q_energy_weights must contain only finite values.")
    if np.any(q_weights <= 0.0):
        raise ValueError("q_energy_weights must be strictly positive.")

    f = reconstruct_distribution(A, q, g.basis_matrix)
    moments = stress_moments_from_distribution(
        f,
        q_weights,
        g.angular_weights,
        g.W_plus,
        W_minus=g.W_minus,
    )
    rho = np.asarray(moments.rho_weighted, dtype=float)
    pi_plus = np.asarray(moments.pi_plus_tilde, dtype=float)
    if moments.pi_minus_tilde is None:
        raise RuntimeError("non-LRS stress moments must include Pi_minus.")
    pi_minus = np.asarray(moments.pi_minus_tilde, dtype=float)
    rho_total = max(float(np.sum(rho)), 1.0e-300)
    total_pi_plus = float(np.sum(rho * pi_plus) / rho_total)
    total_pi_minus = float(np.sum(rho * pi_minus) / rho_total)
    Pi_plus = float(cfg.feedback_factor * cfg.f_nu * total_pi_plus)
    Pi_minus = float(cfg.feedback_factor * cfg.f_nu * total_pi_minus)
    sigma_plus = float(Sigma_plus)
    sigma_minus = float(Sigma_minus)
    sigma_sq = sigma_plus**2 + sigma_minus**2
    dSigma_plus = -((1.0 - sigma_sq) * sigma_plus) - Pi_plus
    dSigma_minus = -((1.0 - sigma_sq) * sigma_minus) - Pi_minus

    return AugmentedNonLRSSourceCollisionlessRHS(
        dSigma_plus=float(dSigma_plus),
        dSigma_minus=float(dSigma_minus),
        dA_modes=source.dA_modes,
        Pi_plus=Pi_plus,
        Pi_minus=Pi_minus,
        moments=moments,
    )


def run_augmented_nonlrs_source_collisionless_solve(
    Sigma_plus0: float,
    Sigma_minus0: float,
    *,
    N_span: tuple[float, float] = (0.0, 0.1),
    N_q: int = 6,
    n_species: int = 4,
    config: AugmentedNonLRSSourceCollisionlessConfig | None = None,
    grid: NonLRSS2Grid | None = None,
    q_nodes: np.ndarray | None = None,
    q_energy_weights: np.ndarray | None = None,
    initial_A_modes: np.ndarray | None = None,
    method: str = "Radau",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> AugmentedNonLRSSourceCollisionlessSolveResult:
    """Run the non-LRS source-only augmented Type I reference solve.

    This integrates the diagonal ``Sigma_+``/``Sigma_-`` geometry variables
    with the AP4 source-projected augmented distribution state.  It is a
    smoke-scale SciPy reference shell for staged validation, not a promoted
    nonlinear non-LRS transport or BBN/network runtime.
    """

    cfg = config or AugmentedNonLRSSourceCollisionlessConfig()
    _validate_source_collisionless_config(cfg)
    g = _grid_from_config(grid, cfg)
    if len(N_span) != 2 or not np.all(np.isfinite(N_span)):
        raise ValueError("N_span must contain two finite values.")
    if float(N_span[1]) <= float(N_span[0]):
        raise ValueError("N_span end must be greater than start.")
    n_sp = _exact_int(n_species, name="n_species")
    if n_sp <= 0:
        raise ValueError("n_species must be positive.")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive.")
    if not np.isfinite(float(Sigma_plus0)) or not np.isfinite(float(Sigma_minus0)):
        raise ValueError("Sigma_plus0 and Sigma_minus0 must be finite.")

    if (q_nodes is None) != (q_energy_weights is None):
        raise ValueError("q_nodes and q_energy_weights must be provided together.")
    if q_nodes is None:
        q, q_weights = _default_q_energy_grid(N_q)
    else:
        q = np.asarray(q_nodes, dtype=float)
        q_weights = np.asarray(q_energy_weights, dtype=float)

    if initial_A_modes is None:
        A0 = np.zeros((n_sp, 3, q.size), dtype=float)
    else:
        A0 = np.asarray(initial_A_modes, dtype=float)
        if A0.shape != (n_sp, 3, q.size):
            raise ValueError("initial_A_modes must have shape (n_species, three non-LRS modes, N_q).")

    def pack(sigma_plus: float, sigma_minus: float, A: np.ndarray) -> np.ndarray:
        return np.concatenate((np.array([float(sigma_plus), float(sigma_minus)]), A.ravel()))

    def unpack(y: np.ndarray) -> tuple[float, float, np.ndarray]:
        sigma_plus = float(y[0])
        sigma_minus = float(y[1])
        A = y[2:].reshape(A0.shape)
        return sigma_plus, sigma_minus, A

    def rhs(_N: float, y: np.ndarray) -> np.ndarray:
        sigma_plus, sigma_minus, A = unpack(y)
        block = augmented_nonlrs_source_collisionless_rhs(
            sigma_plus,
            sigma_minus,
            A,
            q,
            q_weights,
            grid=g,
            config=cfg,
        )
        return pack(block.dSigma_plus, block.dSigma_minus, block.dA_modes)

    sol = solve_ivp(
        rhs,
        (float(N_span[0]), float(N_span[1])),
        pack(float(Sigma_plus0), float(Sigma_minus0), A0),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    sigma_plus_final, sigma_minus_final, A_final = unpack(sol.y[:, -1])
    final_rhs = augmented_nonlrs_source_collisionless_rhs(
        sigma_plus_final,
        sigma_minus_final,
        A_final,
        q,
        q_weights,
        grid=g,
        config=cfg,
    )
    return AugmentedNonLRSSourceCollisionlessSolveResult(
        success=bool(sol.success),
        message=str(sol.message),
        N=np.asarray(sol.t, dtype=float),
        Sigma_plus=np.asarray(sol.y[0], dtype=float),
        Sigma_minus=np.asarray(sol.y[1], dtype=float),
        A_modes_final=A_final,
        Pi_plus_final=final_rhs.Pi_plus,
        Pi_minus_final=final_rhs.Pi_minus,
        moments_final=final_rhs.moments,
        q_nodes=q,
        q_energy_weights=q_weights,
        grid=g,
    )


__all__ = [
    "AugmentedNonLRSSourceCollisionlessConfig",
    "AugmentedNonLRSSourceCollisionlessRHS",
    "AugmentedNonLRSSourceCollisionlessSolveResult",
    "NonLRSQuadrupoleSourceRHS",
    "NonLRSS2Grid",
    "augmented_nonlrs_source_collisionless_rhs",
    "build_non_lrs_s2_grid",
    "non_lrs_quadrupole_source_rhs",
    "run_augmented_nonlrs_source_collisionless_solve",
]
