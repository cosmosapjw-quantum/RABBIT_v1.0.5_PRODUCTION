"""Non-LRS S2 nonlinear collisionless transport operator for augmented Type I."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import operator

import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.integrate import solve_ivp

from rabbit.jax.nonlinear_transport import build_mu_diff_matrix, build_q_diff_matrix
from rabbit.transport.augmented_pstf_distribution import (
    distribution_rhs_to_augmented_rhs,
    modes_to_nodal,
    project_nodal_to_modes,
    reconstruct_distribution,
)
from rabbit.transport.augmented_typeI_observables import (
    AugmentedStressMoments,
    stress_moments_from_distribution,
)
from rabbit.transport.augmented_typeI_nonlrs_collisionless import (
    AugmentedNonLRSSourceCollisionlessConfig,
    NonLRSS2Grid,
    build_non_lrs_s2_grid,
)


NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT = "nonlrs_s2_nonlinear_collisionless_transport_rhs_v1"
_Q_DIFF_MATRIX_CACHE_MAXSIZE = 16
_Q_DIFF_MATRIX_CACHE: OrderedDict[tuple[tuple[int, ...], str, bytes], np.ndarray] = OrderedDict()


@dataclass(frozen=True)
class AugmentedNonLRSNonlinearTransportGrid:
    """S2 tensor grid plus angular derivative matrices for AP62."""

    s2_grid: NonLRSS2Grid
    N_mu: int
    N_phi: int
    D_mu: np.ndarray
    D_phi: np.ndarray
    energy_shift_plus: np.ndarray
    energy_shift_minus: np.ndarray
    mu_drift_plus: np.ndarray
    mu_drift_minus: np.ndarray
    phi_drift_minus: np.ndarray
    contract: str = "nonlrs_s2_nonlinear_transport_grid_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.s2_grid, NonLRSS2Grid):
            raise ValueError("s2_grid must be a NonLRSS2Grid.")
        N_mu = _positive_int(self.N_mu, name="N_mu", minimum=3)
        N_phi = _positive_int(self.N_phi, name="N_phi", minimum=5)
        if self.s2_grid.mu.size != N_mu * N_phi:
            raise ValueError("N_mu*N_phi must match the S2 grid angle count.")
        D_mu = np.asarray(self.D_mu, dtype=float)
        D_phi = np.asarray(self.D_phi, dtype=float)
        factor_arrays = (
            np.asarray(self.energy_shift_plus, dtype=float),
            np.asarray(self.energy_shift_minus, dtype=float),
            np.asarray(self.mu_drift_plus, dtype=float),
            np.asarray(self.mu_drift_minus, dtype=float),
            np.asarray(self.phi_drift_minus, dtype=float),
        )
        if D_mu.shape != (N_mu, N_mu):
            raise ValueError("D_mu must have shape (N_mu, N_mu).")
        if D_phi.shape != (N_phi, N_phi):
            raise ValueError("D_phi must have shape (N_phi, N_phi).")
        angle_count = self.s2_grid.mu.size
        if any(arr.shape != (angle_count,) for arr in factor_arrays):
            raise ValueError("precomputed drift factors must match the angular axis.")
        if (
            not np.all(np.isfinite(D_mu))
            or not np.all(np.isfinite(D_phi))
            or any(not np.all(np.isfinite(arr)) for arr in factor_arrays)
        ):
            raise ValueError("angular derivative matrices and drift factors must be finite.")
        if not self.contract:
            raise ValueError("contract must be a non-empty string.")
        object.__setattr__(self, "N_mu", N_mu)
        object.__setattr__(self, "N_phi", N_phi)
        object.__setattr__(self, "D_mu", D_mu)
        object.__setattr__(self, "D_phi", D_phi)
        object.__setattr__(self, "energy_shift_plus", factor_arrays[0])
        object.__setattr__(self, "energy_shift_minus", factor_arrays[1])
        object.__setattr__(self, "mu_drift_plus", factor_arrays[2])
        object.__setattr__(self, "mu_drift_minus", factor_arrays[3])
        object.__setattr__(self, "phi_drift_minus", factor_arrays[4])
        object.__setattr__(self, "contract", str(self.contract))

    @property
    def mu(self) -> np.ndarray:
        return self.s2_grid.mu

    @property
    def phi(self) -> np.ndarray:
        return self.s2_grid.phi

    @property
    def angular_weights(self) -> np.ndarray:
        return self.s2_grid.angular_weights

    @property
    def W_plus(self) -> np.ndarray:
        return self.s2_grid.W_plus

    @property
    def W_minus(self) -> np.ndarray:
        return self.s2_grid.W_minus


_NONLINEAR_TRANSPORT_GRID_CACHE_MAXSIZE = 16
_NONLINEAR_TRANSPORT_GRID_CACHE: OrderedDict[
    tuple[int, int],
    AugmentedNonLRSNonlinearTransportGrid,
] = OrderedDict()


@dataclass(frozen=True)
class AugmentedNonLRSNonlinearTransportRHS:
    """Nodal and projected AP62 nonlinear transport RHS."""

    df_dN_nodal: np.ndarray
    dA_modes: np.ndarray
    energy_shift: np.ndarray
    mu_drift: np.ndarray
    phi_drift: np.ndarray
    grid: AugmentedNonLRSNonlinearTransportGrid
    transport_scope_contract: str = NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.df_dN_nodal, dtype=float),
            np.asarray(self.dA_modes, dtype=float),
            np.asarray(self.energy_shift, dtype=float),
            np.asarray(self.mu_drift, dtype=float),
            np.asarray(self.phi_drift, dtype=float),
        )
        if any(not np.all(np.isfinite(arr)) for arr in arrays):
            raise ValueError("non-LRS nonlinear transport RHS arrays must be finite.")
        if arrays[0].ndim != 3:
            raise ValueError("df_dN_nodal must have shape (n_species, N_q, n_angles).")
        if arrays[1].shape != (arrays[0].shape[0], 3, arrays[0].shape[1]):
            raise ValueError("dA_modes must have shape (n_species, three modes, N_q).")
        if arrays[2].shape != (arrays[0].shape[2],):
            raise ValueError("energy_shift must match the angular axis.")
        if arrays[3].shape != arrays[2].shape or arrays[4].shape != arrays[2].shape:
            raise ValueError("angular drift arrays must match the angular axis.")
        if not self.transport_scope_contract:
            raise ValueError("transport_scope_contract must be a non-empty string.")
        object.__setattr__(self, "df_dN_nodal", arrays[0])
        object.__setattr__(self, "dA_modes", arrays[1])
        object.__setattr__(self, "energy_shift", arrays[2])
        object.__setattr__(self, "mu_drift", arrays[3])
        object.__setattr__(self, "phi_drift", arrays[4])
        object.__setattr__(self, "transport_scope_contract", str(self.transport_scope_contract))


@dataclass(frozen=True)
class AugmentedNonLRSNonlinearTransportCoevolutionRHS:
    """Geometry plus nonlinear S2 transport derivative for one AP63 RHS call."""

    dSigma_plus: float
    dSigma_minus: float
    dA_modes: np.ndarray
    Pi_plus: float
    Pi_minus: float
    moments: AugmentedStressMoments
    transport: AugmentedNonLRSNonlinearTransportRHS
    closure: str = "nonlrs_s2_nonlinear_transport_coevolution_v1"
    f_species: np.ndarray | None = None


@dataclass(frozen=True)
class AugmentedNonLRSNonlinearTransportSolveResult:
    """SciPy solve result for AP63 nonlinear S2 transport coevolution."""

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
    grid: AugmentedNonLRSNonlinearTransportGrid
    method: str
    nfev: int
    closure: str = "nonlrs_s2_nonlinear_transport_coevolution_v1"


def _positive_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an exact integer.")
    try:
        checked = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an exact integer.") from exc
    if checked < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return checked


def _periodic_phi_diff_matrix(N_phi: int) -> np.ndarray:
    n_phi = _positive_int(N_phi, name="N_phi", minimum=5)
    dphi = 2.0 * np.pi / float(n_phi)
    D = np.zeros((n_phi, n_phi), dtype=float)
    for idx in range(n_phi):
        D[idx, (idx + 1) % n_phi] = 0.5 / dphi
        D[idx, (idx - 1) % n_phi] = -0.5 / dphi
    return D


def build_nonlrs_nonlinear_transport_grid(
    *,
    N_mu: int = 16,
    N_phi: int = 16,
) -> AugmentedNonLRSNonlinearTransportGrid:
    """Build the AP62 S2 grid and angular derivative matrices."""

    n_mu = _positive_int(N_mu, name="N_mu", minimum=3)
    n_phi = _positive_int(N_phi, name="N_phi", minimum=5)
    key = (n_mu, n_phi)
    cached = _NONLINEAR_TRANSPORT_GRID_CACHE.get(key)
    if cached is not None:
        _NONLINEAR_TRANSPORT_GRID_CACHE.move_to_end(key)
        return cached
    s2_grid = build_non_lrs_s2_grid(N_mu=n_mu, N_phi=n_phi)
    mu_1d = s2_grid.mu.reshape(n_mu, n_phi)[:, 0]
    mu = s2_grid.mu
    phi = s2_grid.phi
    one_minus_mu2 = 1.0 - mu * mu
    grid = AugmentedNonLRSNonlinearTransportGrid(
        s2_grid=s2_grid,
        N_mu=n_mu,
        N_phi=n_phi,
        D_mu=build_mu_diff_matrix(mu_1d),
        D_phi=_periodic_phi_diff_matrix(n_phi),
        energy_shift_plus=2.0 * s2_grid.W_plus,
        energy_shift_minus=2.0 * s2_grid.W_minus,
        mu_drift_plus=3.0 * mu * one_minus_mu2,
        mu_drift_minus=-np.sqrt(3.0) * mu * one_minus_mu2 * np.cos(2.0 * phi),
        phi_drift_minus=-np.sqrt(3.0) * np.sin(2.0 * phi),
    )
    _NONLINEAR_TRANSPORT_GRID_CACHE[key] = grid
    if len(_NONLINEAR_TRANSPORT_GRID_CACHE) > _NONLINEAR_TRANSPORT_GRID_CACHE_MAXSIZE:
        _NONLINEAR_TRANSPORT_GRID_CACHE.popitem(last=False)
    return grid


def _as_grid(
    grid: AugmentedNonLRSNonlinearTransportGrid | None,
) -> AugmentedNonLRSNonlinearTransportGrid:
    return build_nonlrs_nonlinear_transport_grid() if grid is None else grid


def _validate_q_nodes(q_nodes: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=float)
    if q.ndim != 1 or q.size < 3:
        raise ValueError("q_nodes must be a 1D array with at least three entries.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_nodes must contain only finite values.")
    if np.any(q <= 0.0):
        raise ValueError("q_nodes must be strictly positive.")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nodes must be strictly increasing.")
    return q


def _default_q_energy_grid(N_q: int) -> tuple[np.ndarray, np.ndarray]:
    n_q = _positive_int(N_q, name="N_q", minimum=3)
    q, w = laggauss(n_q)
    return np.asarray(q, dtype=float), np.asarray(w * np.exp(q) * q**3, dtype=float)


def _cached_q_diff_matrix(q_nodes: np.ndarray) -> np.ndarray:
    q = np.ascontiguousarray(q_nodes, dtype=float)
    key = (tuple(q.shape), q.dtype.str, q.tobytes())
    cached = _Q_DIFF_MATRIX_CACHE.get(key)
    if cached is not None:
        _Q_DIFF_MATRIX_CACHE.move_to_end(key)
        return cached
    D_q = np.asarray(build_q_diff_matrix(q), dtype=float)
    D_q.setflags(write=False)
    _Q_DIFF_MATRIX_CACHE[key] = D_q
    if len(_Q_DIFF_MATRIX_CACHE) > _Q_DIFF_MATRIX_CACHE_MAXSIZE:
        _Q_DIFF_MATRIX_CACHE.popitem(last=False)
    return D_q


def _validate_q_weights(q_energy_weights: np.ndarray, q_nodes: np.ndarray) -> np.ndarray:
    weights = np.asarray(q_energy_weights, dtype=float)
    if weights.shape != q_nodes.shape:
        raise ValueError("q_energy_weights must match q_nodes.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("q_energy_weights must be strictly positive and finite.")
    return weights


def _validate_nodal_distribution(
    f_species: np.ndarray,
    q_nodes: np.ndarray,
    grid: AugmentedNonLRSNonlinearTransportGrid,
) -> np.ndarray:
    f = np.asarray(f_species, dtype=float)
    if f.ndim != 3:
        raise ValueError("f_species must have shape (n_species, N_q, n_angles).")
    if f.shape[0] == 0:
        raise ValueError("f_species must contain at least one species.")
    if f.shape[1] != q_nodes.size:
        raise ValueError("f_species q axis must match q_nodes.")
    if f.shape[2] != grid.s2_grid.mu.size:
        raise ValueError("f_species angular axis must match the S2 grid.")
    if not np.all(np.isfinite(f)):
        raise ValueError("f_species must contain only finite values.")
    return f


def _validate_modes(
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
) -> np.ndarray:
    A = np.asarray(A_modes, dtype=float)
    if A.ndim != 3:
        raise ValueError("A_modes must have shape (n_species, three non-LRS modes, N_q).")
    if A.shape[0] == 0:
        raise ValueError("A_modes must contain at least one species.")
    if A.shape[1] != 3:
        raise ValueError("A_modes must contain exactly three non-LRS modes.")
    if A.shape[2] != q_nodes.size:
        raise ValueError("A_modes q axis must match q_nodes.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A_modes must contain only finite values.")
    return A


def _angular_vector_field(
    Sigma_plus: float,
    Sigma_minus: float,
    grid: AugmentedNonLRSNonlinearTransportGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_plus = float(Sigma_plus)
    sigma_minus = float(Sigma_minus)
    if not np.isfinite(sigma_plus) or not np.isfinite(sigma_minus):
        raise ValueError("Sigma_plus and Sigma_minus must be finite.")
    energy_shift = (
        sigma_plus * grid.energy_shift_plus
        + sigma_minus * grid.energy_shift_minus
    )
    mu_drift = (
        sigma_plus * grid.mu_drift_plus
        + sigma_minus * grid.mu_drift_minus
    )
    phi_drift = sigma_minus * grid.phi_drift_minus
    return energy_shift, mu_drift, phi_drift


def _mu_derivative(
    f: np.ndarray,
    grid: AugmentedNonLRSNonlinearTransportGrid,
) -> np.ndarray:
    shape = (f.shape[0], f.shape[1], grid.N_mu, grid.N_phi)
    f_grid = f.reshape(shape)
    return np.einsum("ij,sqjp->sqip", grid.D_mu, f_grid).reshape(f.shape)


def _phi_derivative(
    f: np.ndarray,
    grid: AugmentedNonLRSNonlinearTransportGrid,
) -> np.ndarray:
    shape = (f.shape[0], f.shape[1], grid.N_mu, grid.N_phi)
    f_grid = f.reshape(shape)
    return np.einsum("pj,sqij->sqip", grid.D_phi, f_grid).reshape(f.shape)


def augmented_nonlrs_nonlinear_nodal_rhs(
    f_species: np.ndarray,
    q_nodes: np.ndarray,
    *,
    Sigma_plus: float,
    Sigma_minus: float,
    grid: AugmentedNonLRSNonlinearTransportGrid | None = None,
) -> AugmentedNonLRSNonlinearTransportRHS:
    """Evaluate the AP62 S2 nonlinear collisionless transport RHS on nodes.

    This extends the LRS exact collisionless equation to the diagonal S2 basis
    through the shear scalar `2*(Sigma_plus*W_plus + Sigma_minus*W_minus)` and
    the corresponding angular gradient field.  The nodal distribution RHS is
    converted through the augmented-logit chain rule before projection into the
    staged A-mode basis.  It is a private transport operator:
    no weak/network, collision source, full-BBN solve, public dispatch, or QKE
    claim is made here.
    """

    g = _as_grid(grid)
    q = _validate_q_nodes(q_nodes)
    f = _validate_nodal_distribution(f_species, q, g)
    D_q = _cached_q_diff_matrix(q)
    energy_shift, mu_drift, phi_drift = _angular_vector_field(Sigma_plus, Sigma_minus, g)
    df_dq = np.einsum("ij,sja->sia", D_q, f)
    df_dmu = _mu_derivative(f, g)
    df_dphi = _phi_derivative(f, g)
    rhs = (
        -energy_shift[None, None, :] * q[None, :, None] * df_dq
        - mu_drift[None, None, :] * df_dmu
        - phi_drift[None, None, :] * df_dphi
    )
    dA = distribution_rhs_to_augmented_rhs(
        rhs,
        f,
        g.s2_grid.basis_matrix,
        g.s2_grid.angular_weights,
        projection_matrix=g.s2_grid.projection_matrix,
    )
    return AugmentedNonLRSNonlinearTransportRHS(
        df_dN_nodal=np.asarray(rhs, dtype=float),
        dA_modes=np.asarray(dA, dtype=float),
        energy_shift=energy_shift,
        mu_drift=mu_drift,
        phi_drift=phi_drift,
        grid=g,
    )


def augmented_nonlrs_nonlinear_mode_rhs(
    *,
    Sigma_plus: float,
    Sigma_minus: float,
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    grid: AugmentedNonLRSNonlinearTransportGrid | None = None,
) -> AugmentedNonLRSNonlinearTransportRHS:
    """Reconstruct staged S2 modes, run AP62 nodal RHS, and project back."""

    g = _as_grid(grid)
    q = _validate_q_nodes(q_nodes)
    A = _validate_modes(A_modes, q)
    return _augmented_nonlrs_nonlinear_logit_mode_rhs(
        Sigma_plus=Sigma_plus,
        Sigma_minus=Sigma_minus,
        A_modes=A,
        q_nodes=q,
        grid=g,
    )


def _augmented_nonlrs_nonlinear_logit_mode_rhs(
    *,
    Sigma_plus: float,
    Sigma_minus: float,
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    grid: AugmentedNonLRSNonlinearTransportGrid,
    f_species: np.ndarray | None = None,
) -> AugmentedNonLRSNonlinearTransportRHS:
    """Evaluate AP62 transport directly in augmented-logit coordinates."""

    q = _validate_q_nodes(q_nodes)
    A = _validate_modes(A_modes, q)
    g = _as_grid(grid)
    D_q = _cached_q_diff_matrix(q)
    energy_shift, mu_drift, phi_drift = _angular_vector_field(
        Sigma_plus,
        Sigma_minus,
        g,
    )
    nodal_A = modes_to_nodal(A, g.s2_grid.basis_matrix)
    dA_dq = np.einsum("ij,sja->sia", D_q, nodal_A)
    dA_dmu = _mu_derivative(nodal_A, g)
    dA_dphi = _phi_derivative(nodal_A, g)
    nodal_dA = (
        -energy_shift[None, None, :] * q[None, :, None] * (1.0 + dA_dq)
        - mu_drift[None, None, :] * dA_dmu
        - phi_drift[None, None, :] * dA_dphi
    )
    dA = project_nodal_to_modes(
        nodal_dA,
        g.s2_grid.basis_matrix,
        g.s2_grid.angular_weights,
        projection_matrix=g.s2_grid.projection_matrix,
    )
    if f_species is None:
        f = reconstruct_distribution(A, q, g.s2_grid.basis_matrix)
    else:
        f = np.asarray(f_species, dtype=float)
        if f.shape != nodal_dA.shape:
            raise ValueError("f_species must match the reconstructed nodal shape.")
    df_dN = -f * (1.0 - f) * nodal_dA
    return AugmentedNonLRSNonlinearTransportRHS(
        df_dN_nodal=np.asarray(df_dN, dtype=float),
        dA_modes=np.asarray(dA, dtype=float),
        energy_shift=energy_shift,
        mu_drift=mu_drift,
        phi_drift=phi_drift,
        grid=g,
    )


def augmented_nonlrs_nonlinear_transport_coevolution_rhs(
    Sigma_plus: float,
    Sigma_minus: float,
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    q_energy_weights: np.ndarray,
    *,
    grid: AugmentedNonLRSNonlinearTransportGrid | None = None,
    config: AugmentedNonLRSSourceCollisionlessConfig | None = None,
) -> AugmentedNonLRSNonlinearTransportCoevolutionRHS:
    """Evaluate AP63 nonlinear transport plus live shear-stress feedback."""

    g = _as_grid(grid)
    q = _validate_q_nodes(q_nodes)
    weights = _validate_q_weights(q_energy_weights, q)
    cfg = config or AugmentedNonLRSSourceCollisionlessConfig()
    if not np.isfinite(float(cfg.f_nu)) or float(cfg.f_nu) < 0.0:
        raise ValueError("f_nu must be finite and non-negative.")
    if not np.isfinite(float(cfg.feedback_factor)):
        raise ValueError("feedback_factor must be finite.")
    A = _validate_modes(A_modes, q)
    f = reconstruct_distribution(A, q, g.s2_grid.basis_matrix)
    transport = _augmented_nonlrs_nonlinear_logit_mode_rhs(
        Sigma_plus=Sigma_plus,
        Sigma_minus=Sigma_minus,
        A_modes=A,
        q_nodes=q,
        grid=g,
        f_species=f,
    )
    moments = stress_moments_from_distribution(
        f,
        weights,
        g.s2_grid.angular_weights,
        g.s2_grid.W_plus,
        W_minus=g.s2_grid.W_minus,
    )
    rho = np.asarray(moments.rho_weighted, dtype=float)
    pi_plus = np.asarray(moments.pi_plus_tilde, dtype=float)
    if moments.pi_minus_tilde is None:
        raise RuntimeError("non-LRS nonlinear stress moments must include Pi_minus.")
    pi_minus = np.asarray(moments.pi_minus_tilde, dtype=float)
    rho_total = max(float(np.sum(rho)), 1.0e-300)
    total_pi_plus = float(np.sum(rho * pi_plus) / rho_total)
    total_pi_minus = float(np.sum(rho * pi_minus) / rho_total)
    Pi_plus = float(cfg.feedback_factor * cfg.f_nu * total_pi_plus)
    Pi_minus = float(cfg.feedback_factor * cfg.f_nu * total_pi_minus)
    sigma_plus = float(Sigma_plus)
    sigma_minus = float(Sigma_minus)
    if not np.isfinite(sigma_plus) or not np.isfinite(sigma_minus):
        raise ValueError("Sigma_plus and Sigma_minus must be finite.")
    sigma_sq = sigma_plus**2 + sigma_minus**2
    # BD622-R2 / audit F-020 (SL-R): the augmented <fW>/<f> Pi polarity (Sigma>0 => Pi>0)
    # requires -Pi for physical damping (H-theorem); +Pi anti-damps (~1600x shear growth).
    return AugmentedNonLRSNonlinearTransportCoevolutionRHS(
        dSigma_plus=float(-((1.0 - sigma_sq) * sigma_plus) - Pi_plus),
        dSigma_minus=float(-((1.0 - sigma_sq) * sigma_minus) - Pi_minus),
        dA_modes=transport.dA_modes,
        Pi_plus=Pi_plus,
        Pi_minus=Pi_minus,
        moments=moments,
        transport=transport,
        f_species=f,
    )


def run_augmented_nonlrs_nonlinear_transport_solve(
    Sigma_plus0: float,
    Sigma_minus0: float,
    *,
    N_span: tuple[float, float] = (0.0, 0.1),
    N_q: int = 6,
    n_species: int = 4,
    N_mu: int = 16,
    N_phi: int = 16,
    config: AugmentedNonLRSSourceCollisionlessConfig | None = None,
    grid: AugmentedNonLRSNonlinearTransportGrid | None = None,
    q_nodes: np.ndarray | None = None,
    q_energy_weights: np.ndarray | None = None,
    initial_A_modes: np.ndarray | None = None,
    method: str = "Radau",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> AugmentedNonLRSNonlinearTransportSolveResult:
    """Integrate AP63 nonlinear S2 transport plus shear-stress feedback."""

    if len(N_span) != 2 or not np.all(np.isfinite(N_span)):
        raise ValueError("N_span must contain two finite values.")
    if float(N_span[1]) <= float(N_span[0]):
        raise ValueError("N_span end must exceed start.")
    if not np.isfinite(float(Sigma_plus0)) or not np.isfinite(float(Sigma_minus0)):
        raise ValueError("Sigma_plus0 and Sigma_minus0 must be finite.")
    if float(rtol) <= 0.0 or float(atol) <= 0.0:
        raise ValueError("rtol and atol must be positive.")
    if not method:
        raise ValueError("method must be a non-empty string.")
    n_sp = _positive_int(n_species, name="n_species", minimum=1)
    g = _as_grid(grid) if grid is not None else build_nonlrs_nonlinear_transport_grid(
        N_mu=N_mu,
        N_phi=N_phi,
    )
    if (q_nodes is None) != (q_energy_weights is None):
        raise ValueError("q_nodes and q_energy_weights must be provided together.")
    if q_nodes is None:
        q, q_weights = _default_q_energy_grid(N_q)
    else:
        q = _validate_q_nodes(q_nodes)
        q_weights = _validate_q_weights(q_energy_weights, q)  # type: ignore[arg-type]
    if initial_A_modes is None:
        A0 = np.zeros((n_sp, 3, q.size), dtype=float)
        A0[:, 0, :] = 1.0 / (np.exp(q) + 1.0)
    else:
        A0 = np.asarray(initial_A_modes, dtype=float)
        if A0.shape != (n_sp, 3, q.size):
            raise ValueError("initial_A_modes must have shape (n_species, three non-LRS modes, N_q).")
        if not np.all(np.isfinite(A0)):
            raise ValueError("initial_A_modes must contain only finite values.")

    cfg = config or AugmentedNonLRSSourceCollisionlessConfig(
        N_mu=g.N_mu,
        N_phi=g.N_phi,
    )

    def pack(sigma_plus: float, sigma_minus: float, A: np.ndarray) -> np.ndarray:
        return np.concatenate((np.array([float(sigma_plus), float(sigma_minus)]), A.ravel()))

    def unpack(y: np.ndarray) -> tuple[float, float, np.ndarray]:
        return float(y[0]), float(y[1]), y[2:].reshape(A0.shape)

    def rhs(_N: float, y: np.ndarray) -> np.ndarray:
        sigma_plus, sigma_minus, A = unpack(y)
        block = augmented_nonlrs_nonlinear_transport_coevolution_rhs(
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
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    )
    sigma_plus_final, sigma_minus_final, A_final = unpack(sol.y[:, -1])
    final_rhs = augmented_nonlrs_nonlinear_transport_coevolution_rhs(
        sigma_plus_final,
        sigma_minus_final,
        A_final,
        q,
        q_weights,
        grid=g,
        config=cfg,
    )
    return AugmentedNonLRSNonlinearTransportSolveResult(
        success=bool(sol.success),
        message=str(sol.message),
        N=np.asarray(sol.t, dtype=float),
        Sigma_plus=np.asarray(sol.y[0], dtype=float),
        Sigma_minus=np.asarray(sol.y[1], dtype=float),
        A_modes_final=np.asarray(A_final, dtype=float),
        Pi_plus_final=final_rhs.Pi_plus,
        Pi_minus_final=final_rhs.Pi_minus,
        moments_final=final_rhs.moments,
        q_nodes=q,
        q_energy_weights=q_weights,
        grid=g,
        method=str(method),
        nfev=int(sol.nfev),
    )


__all__ = [
    "NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT",
    "AugmentedNonLRSNonlinearTransportGrid",
    "AugmentedNonLRSNonlinearTransportCoevolutionRHS",
    "AugmentedNonLRSNonlinearTransportRHS",
    "AugmentedNonLRSNonlinearTransportSolveResult",
    "augmented_nonlrs_nonlinear_transport_coevolution_rhs",
    "augmented_nonlrs_nonlinear_mode_rhs",
    "augmented_nonlrs_nonlinear_nodal_rhs",
    "build_nonlrs_nonlinear_transport_grid",
    "run_augmented_nonlrs_nonlinear_transport_solve",
]
