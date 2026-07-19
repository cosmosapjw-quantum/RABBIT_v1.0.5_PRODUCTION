"""Augmented PSTF distribution reconstruction and projection utilities.

The functions here are intentionally basis-agnostic.  Callers provide an
angular basis matrix evaluated on an S_N grid; this module handles the
nonlinear logit distribution and the projection between nodal angular values
and augmented PSTF coefficients.
"""
from __future__ import annotations

import numpy as np


def fermi_dirac_from_logit(logit: np.ndarray | float) -> np.ndarray:
    """Return ``1 / (exp(logit) + 1)`` with overflow-safe branches."""

    x = np.asarray(logit, dtype=float)
    out = np.empty_like(x, dtype=float)

    positive = x >= 0.0
    z_pos = np.exp(-x[positive])
    out[positive] = z_pos / (1.0 + z_pos)

    z_neg = np.exp(x[~positive])
    out[~positive] = 1.0 / (1.0 + z_neg)
    return out


def _as_basis_matrix(basis_matrix: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis_matrix, dtype=float)
    if basis.ndim != 2:
        raise ValueError("basis_matrix must have shape (n_modes, n_angles).")
    if basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError("basis_matrix must have non-zero dimensions.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis_matrix must contain only finite values.")
    return basis


def _as_q_nodes(q_nodes: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_nodes must be a non-empty 1D array.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_nodes must contain only finite values.")
    return q


def _as_weights(angular_weights: np.ndarray, n_angles: int) -> np.ndarray:
    weights = np.asarray(angular_weights, dtype=float)
    if weights.shape != (n_angles,):
        raise ValueError("angular_weights must have shape (n_angles,).")
    if not np.all(np.isfinite(weights)):
        raise ValueError("angular_weights must contain only finite values.")
    return weights


def mode_norms_from_basis(
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    min_norm: float = 1.0e-30,
) -> np.ndarray:
    """Return quadrature norms ``sum_a w_a B_ma^2`` for each mode."""

    basis = _as_basis_matrix(basis_matrix)
    weights = _as_weights(angular_weights, basis.shape[1])
    norms = np.einsum("a,ma,ma->m", weights, basis, basis)
    if np.any(norms <= min_norm):
        raise ValueError("basis_matrix contains a mode with near-zero quadrature norm.")
    return norms


def gram_matrix_from_basis(
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    max_condition: float = 1.0e12,
) -> np.ndarray:
    """Return the weighted angular Gram matrix for the supplied basis."""

    basis = _as_basis_matrix(basis_matrix)
    weights = _as_weights(angular_weights, basis.shape[1])
    gram = np.einsum("a,ma,na->mn", weights, basis, basis)
    if not np.all(np.isfinite(gram)):
        raise ValueError("basis Gram matrix must contain only finite values.")
    condition = np.linalg.cond(gram)
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError("basis Gram matrix is singular or too ill-conditioned.")
    return gram


def projection_matrix_from_basis(
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    max_condition: float = 1.0e12,
) -> np.ndarray:
    """Return the linear nodal-to-mode projection matrix for a fixed basis."""

    basis = _as_basis_matrix(basis_matrix)
    weights = _as_weights(angular_weights, basis.shape[1])
    gram = gram_matrix_from_basis(basis, weights, max_condition=max_condition)
    weighted_basis = basis * weights.reshape((1, weights.size))
    projection = np.linalg.solve(gram, weighted_basis)
    if projection.shape != basis.shape or not np.all(np.isfinite(projection)):
        raise ValueError("projection matrix construction failed.")
    return projection


def modes_to_nodal(A_modes: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """Evaluate augmented coefficients on angular nodes.

    Parameters
    ----------
    A_modes
        Array with shape ``(..., n_modes, n_q)``.
    basis_matrix
        Array with shape ``(n_modes, n_angles)``.

    Returns
    -------
    ndarray
        Array with shape ``(..., n_q, n_angles)``.
    """

    A = np.asarray(A_modes, dtype=float)
    basis = _as_basis_matrix(basis_matrix)
    if A.ndim < 2:
        raise ValueError("A_modes must have shape (..., n_modes, n_q).")
    if A.shape[-2] != basis.shape[0]:
        raise ValueError("A_modes mode axis does not match basis_matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A_modes must contain only finite values.")
    return np.einsum("...mq,ma->...qa", A, basis)


def reconstruct_distribution(
    A_modes: np.ndarray,
    q_nodes: np.ndarray,
    basis_matrix: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``f(q, angle)`` from augmented PSTF coefficients."""

    q = _as_q_nodes(q_nodes)
    nodal_A = modes_to_nodal(A_modes, basis_matrix)
    if nodal_A.shape[-2] != q.size:
        raise ValueError("A_modes q axis does not match q_nodes.")
    logit = nodal_A + q.reshape((1,) * (nodal_A.ndim - 2) + (q.size, 1))
    return fermi_dirac_from_logit(logit)


def project_nodal_to_modes(
    nodal_values: np.ndarray,
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    mode_norms: np.ndarray | None = None,
    gram_matrix: np.ndarray | None = None,
    projection_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Project nodal angular values onto the supplied basis.

    ``nodal_values`` must have shape ``(..., n_q, n_angles)`` and the
    returned coefficients have shape ``(..., n_modes, n_q)``.  By default,
    this solves the full weighted Gram system, so it remains valid for S_N
    grids where finite quadrature makes the basis only approximately
    orthogonal.  Supplying ``mode_norms`` selects the faster diagonal path
    for known-orthogonal bases.
    """

    nodal = np.asarray(nodal_values, dtype=float)
    basis = _as_basis_matrix(basis_matrix)
    weights = _as_weights(angular_weights, basis.shape[1])
    if nodal.ndim < 2:
        raise ValueError("nodal_values must have shape (..., n_q, n_angles).")
    if nodal.shape[-1] != basis.shape[1]:
        raise ValueError("nodal_values angular axis does not match basis_matrix.")
    if not np.all(np.isfinite(nodal)):
        raise ValueError("nodal_values must contain only finite values.")

    supplied = sum(value is not None for value in (mode_norms, gram_matrix, projection_matrix))
    if supplied > 1:
        raise ValueError("Provide only one of mode_norms, gram_matrix, or projection_matrix.")

    if projection_matrix is not None:
        projection = np.asarray(projection_matrix, dtype=float)
        if projection.shape != basis.shape:
            raise ValueError("projection_matrix must have shape matching basis_matrix.")
        if not np.all(np.isfinite(projection)):
            raise ValueError("projection_matrix must contain only finite values.")
        coeff_qm = np.einsum("ma,...qa->...qm", projection, nodal)
        return np.moveaxis(coeff_qm, -1, -2)

    rhs = np.einsum("...qa,ma,a->...qm", nodal, basis, weights)

    if mode_norms is not None:
        norms = np.asarray(mode_norms, dtype=float)
        if norms.shape != (basis.shape[0],):
            raise ValueError("mode_norms must have shape (n_modes,).")
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise ValueError("mode_norms must be finite and positive.")
        coeff_qm = rhs / norms.reshape((1,) * (rhs.ndim - 1) + (basis.shape[0],))
    else:
        gram = gram_matrix_from_basis(basis, weights) if gram_matrix is None else np.asarray(gram_matrix, dtype=float)
        if gram.shape != (basis.shape[0], basis.shape[0]):
            raise ValueError("gram_matrix must have shape (n_modes, n_modes).")
        if not np.all(np.isfinite(gram)):
            raise ValueError("gram_matrix must contain only finite values.")
        rhs_flat = rhs.reshape(-1, basis.shape[0]).T
        coeff_flat = np.linalg.solve(gram, rhs_flat).T
        coeff_qm = coeff_flat.reshape(rhs.shape)

    return np.moveaxis(coeff_qm, -1, -2)


def distribution_rhs_to_augmented_rhs(
    df_dN: np.ndarray,
    f: np.ndarray,
    basis_matrix: np.ndarray,
    angular_weights: np.ndarray,
    *,
    eps_f: float = 1.0e-12,
    mode_norms: np.ndarray | None = None,
    gram_matrix: np.ndarray | None = None,
    projection_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Convert nodal distribution RHS values into augmented mode RHS.

    Since ``f = sigmoid(-(q + A))``,
    ``df/dA = -f(1-f)`` and therefore
    ``dA/dN = -df/dN / max(f(1-f), eps_f)`` before angular projection.
    """

    df = np.asarray(df_dN, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    if df.shape != f_arr.shape:
        raise ValueError("df_dN and f must have identical shapes.")
    if eps_f <= 0.0:
        raise ValueError("eps_f must be positive.")
    if not np.all(np.isfinite(df)) or not np.all(np.isfinite(f_arr)):
        raise ValueError("df_dN and f must contain only finite values.")
    if np.any((f_arr < -1.0e-15) | (f_arr > 1.0 + 1.0e-15)):
        raise ValueError("f must lie within [0, 1] up to numerical tolerance.")

    pauli = np.maximum(f_arr * (1.0 - f_arr), float(eps_f))
    nodal_dA = -df / pauli
    return project_nodal_to_modes(
        nodal_dA,
        basis_matrix,
        angular_weights,
        mode_norms=mode_norms,
        gram_matrix=gram_matrix,
        projection_matrix=projection_matrix,
    )


__all__ = [
    "distribution_rhs_to_augmented_rhs",
    "fermi_dirac_from_logit",
    "gram_matrix_from_basis",
    "mode_norms_from_basis",
    "modes_to_nodal",
    "project_nodal_to_modes",
    "projection_matrix_from_basis",
    "reconstruct_distribution",
]
