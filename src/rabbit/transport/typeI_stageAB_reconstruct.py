"""
rabbit.transport.typeI_stageAB_reconstruct — Diagnostic reconstruction for Stage A/B.

This module provides reconstruction operators that map q-resolved axisymmetric
Legendre moments back to a nodal distribution f(q, mu) for diagnostics only.
The evolution variable remains the moment hierarchy itself.

Reconstruction modes
--------------------
R1: direct linear Legendre reconstruction
    f(q, mu) = sum_{ell in active_ells} F_ell(q) P_ell(mu)

R2: FLRW-consistent minimum-distortion nodal reconstruction
    At each q we solve a constrained least-change problem around the isotropic
    baseline F_0(q), enforcing the same active moment coefficients on a fixed
    Gauss-Legendre mu-grid.  This is *not* positivity-preserving by design; it
    is only a diagnostic alternative to R1.

Both modes satisfy the required FLRW contract:
    R[F0, 0, 0, ...](q, mu) = F0(q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from rabbit.transport.stageAB_state import AxisymmetricHierarchyState

ReconstructionMode = Literal["R1", "R2"]


@dataclass(frozen=True)
class MuQuadrature:
    nodes: np.ndarray
    weights: np.ndarray


def gauss_legendre_mu_quadrature(n_mu: int = 17) -> MuQuadrature:
    if n_mu < 3:
        raise ValueError("n_mu must be >= 3")
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    return MuQuadrature(nodes=nodes.astype(float), weights=weights.astype(float))


def legendre_block(active_ells: Iterable[int], mu: np.ndarray) -> np.ndarray:
    rows = []
    for ell in active_ells:
        coeffs = np.zeros(ell + 1, dtype=float)
        coeffs[-1] = 1.0
        rows.append(np.polynomial.legendre.legval(mu, coeffs))
    return np.asarray(rows, dtype=float)


def reproject_active_moments(
    f_nodal: np.ndarray,
    active_ells: Iterable[int],
    quadrature: MuQuadrature,
) -> np.ndarray:
    """Reproject nodal f(q,mu) to the active Legendre coefficients.

    Parameters
    ----------
    f_nodal
        Array of shape (..., N_q, N_mu).
    active_ells
        Iterable of active ell values.
    quadrature
        Mu quadrature on [-1, 1].
    """
    active_ells = tuple(active_ells)
    P = legendre_block(active_ells, quadrature.nodes)  # (n_ell, N_mu)
    prefactors = 0.5 * (2.0 * np.asarray(active_ells, dtype=float) + 1.0)
    # Einstein summation over mu with quadrature weights.
    moments = np.einsum(
        "m,...qm,lm->...lq",
        quadrature.weights,
        f_nodal,
        P,
        optimize=True,
    )
    return prefactors[None, :, None] * moments


def reconstruct_nodal_distribution(
    state: AxisymmetricHierarchyState,
    mode: ReconstructionMode = "R1",
    quadrature: MuQuadrature | None = None,
) -> tuple[MuQuadrature, np.ndarray]:
    """Reconstruct f(q,mu) for all species from active moments.

    Returns
    -------
    quadrature, f_nodal
        quadrature object and nodal array with shape (n_species, N_q, N_mu).
    """
    quadrature = quadrature or gauss_legendre_mu_quadrature()
    active_ells = tuple(state.active_ells)
    P = legendre_block(active_ells, quadrature.nodes)  # (n_ell, N_mu)

    if mode == "R1":
        f_nodal = np.einsum("slq,lm->sqm", state.data, P, optimize=True)
        return quadrature, f_nodal

    if mode != "R2":
        raise ValueError(f"Unsupported reconstruction mode: {mode}")

    # Discrete moment operator: c_l = ((2l+1)/2) * sum_i w_i f_i P_l(mu_i)
    prefactors = 0.5 * (2.0 * np.asarray(active_ells, dtype=float) + 1.0)
    A = prefactors[:, None] * quadrature.weights[None, :] * P  # (n_ell, N_mu)
    AAT_inv = np.linalg.pinv(A @ A.T)
    projection = A.T @ AAT_inv  # (N_mu, n_ell)

    n_species, _, n_q = state.data.shape
    n_mu = quadrature.nodes.size
    f_nodal = np.zeros((n_species, n_q, n_mu), dtype=float)
    for s in range(n_species):
        F0 = state.moment(s, 0)
        for iq in range(n_q):
            baseline = np.full(n_mu, F0[iq], dtype=float)
            coeffs = state.data[s, :, iq]
            correction = projection @ (coeffs - A @ baseline)
            f_nodal[s, iq, :] = baseline + correction
    return quadrature, f_nodal


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    mode: str
    n_mu: int
    f_min: float
    f_max: float
    violation_fraction: float
    reprojection_l2_abs: float
    reprojection_l2_rel: float
    per_ell_l2_abs: dict[int, float]
    per_ell_l2_rel: dict[int, float]


def compute_reconstruction_diagnostics(
    state: AxisymmetricHierarchyState,
    mode: ReconstructionMode,
    n_mu: int = 17,
) -> ReconstructionDiagnostics:
    quadrature = gauss_legendre_mu_quadrature(n_mu=n_mu)
    quadrature, f_nodal = reconstruct_nodal_distribution(state, mode=mode, quadrature=quadrature)
    reproj = reproject_active_moments(f_nodal, state.active_ells, quadrature)
    delta = reproj - state.data

    q_weights = state.grid.weights.astype(float)
    ell_abs: dict[int, float] = {}
    ell_rel: dict[int, float] = {}
    scale_sq_total = 0.0
    err_sq_total = 0.0
    for iell, ell in enumerate(state.active_ells):
        d = delta[:, iell, :]
        s = state.data[:, iell, :]
        err_sq = float(np.sum(q_weights[None, :] * d**2))
        scale_sq = float(np.sum(q_weights[None, :] * s**2))
        ell_abs[ell] = float(np.sqrt(err_sq))
        ell_rel[ell] = float(np.sqrt(err_sq / max(scale_sq, 1.0e-30)))
        err_sq_total += err_sq
        scale_sq_total += scale_sq

    violations = (f_nodal < 0.0) | (f_nodal > 1.0)
    violation_fraction = float(np.mean(violations.astype(float)))
    return ReconstructionDiagnostics(
        mode=mode,
        n_mu=n_mu,
        f_min=float(np.min(f_nodal)),
        f_max=float(np.max(f_nodal)),
        violation_fraction=violation_fraction,
        reprojection_l2_abs=float(np.sqrt(err_sq_total)),
        reprojection_l2_rel=float(np.sqrt(err_sq_total / max(scale_sq_total, 1.0e-30))),
        per_ell_l2_abs=ell_abs,
        per_ell_l2_rel=ell_rel,
    )
