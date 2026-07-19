#!/usr/bin/env python3
"""Bounded preflight experiments for tier-3 full-collision design choices.

This script is intentionally small and CPU-safe. It probes two questions:

1. q-advection discretization:
   - inline semidiscrete RHS on the Laguerre q-grid
   - exact characteristic remap with monotone PCHIP interpolation

2. Jacobian structure for the planned gather-collide-scatter (GCS) tier-3 path:
   - materialized full transport Jacobian size
   - factorized moment-space core size/rank
   - banded q-advection block size
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm


def fd_equilibrium(q_nodes: np.ndarray) -> np.ndarray:
    q_clip = np.clip(q_nodes, -500.0, 500.0)
    return 1.0 / (np.exp(q_clip) + 1.0)


def build_centered_diff_matrix(q_nodes: np.ndarray) -> np.ndarray:
    n_q = q_nodes.shape[0]
    diff = np.zeros((n_q, n_q), dtype=np.float64)

    for i in range(1, n_q - 1):
        h_m = q_nodes[i] - q_nodes[i - 1]
        h_p = q_nodes[i + 1] - q_nodes[i]
        diff[i, i - 1] = -h_p / (h_m * (h_m + h_p))
        diff[i, i] = (h_p - h_m) / (h_m * h_p)
        diff[i, i + 1] = h_m / (h_p * (h_m + h_p))

    h0 = q_nodes[1] - q_nodes[0]
    h1 = q_nodes[2] - q_nodes[1]
    diff[0, 0] = -(2.0 * h0 + h1) / (h0 * (h0 + h1))
    diff[0, 1] = (h0 + h1) / (h0 * h1)
    diff[0, 2] = -h0 / (h1 * (h0 + h1))

    hm = q_nodes[-2] - q_nodes[-3]
    hp = q_nodes[-1] - q_nodes[-2]
    diff[-1, -3] = hp / (hm * (hm + hp))
    diff[-1, -2] = -(hm + hp) / (hm * hp)
    diff[-1, -1] = (2.0 * hp + hm) / (hp * (hm + hp))
    return diff


def build_upwind_neg_diff_matrix(q_nodes: np.ndarray) -> np.ndarray:
    """Upwind derivative for dq/dN < 0 (flow from high q toward low q)."""
    n_q = q_nodes.shape[0]
    diff = np.zeros((n_q, n_q), dtype=np.float64)

    for i in range(n_q - 1):
        h = q_nodes[i + 1] - q_nodes[i]
        diff[i, i] = -1.0 / h
        diff[i, i + 1] = 1.0 / h

    h_last = q_nodes[-1] - q_nodes[-2]
    diff[-1, -2] = -1.0 / h_last
    diff[-1, -1] = 1.0 / h_last
    return diff


def exact_fd_shift(q_nodes: np.ndarray, shift: float) -> np.ndarray:
    return fd_equilibrium(q_nodes * np.exp(shift))


def pchip_shift(f_nodes: np.ndarray, q_nodes: np.ndarray, shift: float) -> np.ndarray:
    q_depart = q_nodes * np.exp(shift)
    interp = PchipInterpolator(q_nodes, f_nodes, extrapolate=True)
    advected = interp(q_depart)
    advected = np.where(q_depart < q_nodes[0], f_nodes[0], advected)
    advected = np.where(q_depart > q_nodes[-1], 0.0, advected)
    return np.clip(advected, 0.0, 1.0)


@dataclass
class AdvectionCaseResult:
    n_q: int
    shift: float
    method: str
    rel_l2: float
    max_abs: float
    min_value: float
    max_value: float
    spectral_radius: float | None = None
    max_real_eig: float | None = None


def run_advection_case(n_q: int, shift: float) -> list[AdvectionCaseResult]:
    q_nodes, _ = laggauss(n_q)
    f0 = fd_equilibrium(q_nodes)
    f_exact = exact_fd_shift(q_nodes, shift)
    dq = np.diag(q_nodes)

    centered = dq @ build_centered_diff_matrix(q_nodes)
    upwind = dq @ build_upwind_neg_diff_matrix(q_nodes)

    f_centered = expm(shift * centered) @ f0
    f_upwind = expm(shift * upwind) @ f0
    f_pchip = pchip_shift(f0, q_nodes, shift)

    def metrics(method: str, f_num: np.ndarray, matrix: np.ndarray | None) -> AdvectionCaseResult:
        rel_l2 = float(np.linalg.norm(f_num - f_exact) / np.linalg.norm(f_exact))
        max_abs = float(np.max(np.abs(f_num - f_exact)))
        if matrix is not None:
            eigvals = np.linalg.eigvals(matrix)
            spectral_radius = float(np.max(np.abs(eigvals)))
            max_real = float(np.max(np.real(eigvals)))
        else:
            spectral_radius = None
            max_real = None
        return AdvectionCaseResult(
            n_q=n_q,
            shift=shift,
            method=method,
            rel_l2=rel_l2,
            max_abs=max_abs,
            min_value=float(np.min(f_num)),
            max_value=float(np.max(f_num)),
            spectral_radius=spectral_radius,
            max_real_eig=max_real,
        )

    return [
        metrics("centered_semidiscrete", f_centered, centered),
        metrics("upwind_semidiscrete", f_upwind, upwind),
        metrics("pchip_exact_remap", f_pchip, None),
    ]


def build_gcs_factorization(
    n_species: int = 4,
    n_rays: int = 12,
    n_q: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    mu_nodes, mu_weights = leggauss(n_rays)
    p2 = 0.5 * (3.0 * mu_nodes**2 - 1.0)

    transport_dim = n_species * n_rays * n_q
    moment_dim = 2 * n_species * n_q
    gather = np.zeros((moment_dim, transport_dim), dtype=np.float64)
    apply = np.zeros((transport_dim, moment_dim), dtype=np.float64)

    for species in range(n_species):
        for iq in range(n_q):
            i_mono = (2 * species) * n_q + iq
            i_quad = (2 * species + 1) * n_q + iq
            for ray in range(n_rays):
                i_transport = ((species * n_rays + ray) * n_q) + iq
                gather[i_mono, i_transport] = 0.5 * mu_weights[ray]
                gather[i_quad, i_transport] = 2.5 * mu_weights[ray] * p2[ray]
                apply[i_transport, i_mono] = 1.0
                apply[i_transport, i_quad] = p2[ray]

    return gather, apply


def build_upwind_transport_block(
    n_species: int = 4,
    n_rays: int = 12,
    n_q: int = 20,
) -> np.ndarray:
    q_nodes, _ = laggauss(n_q)
    diff = build_upwind_neg_diff_matrix(q_nodes)
    coeffs = np.linspace(-0.35, 0.35, n_rays, dtype=np.float64)
    ray_block = [coeff * (np.diag(q_nodes) @ diff) for coeff in coeffs]

    transport_dim = n_species * n_rays * n_q
    block = np.zeros((transport_dim, transport_dim), dtype=np.float64)
    for species in range(n_species):
        for ray in range(n_rays):
            start = (species * n_rays + ray) * n_q
            stop = start + n_q
            block[start:stop, start:stop] = ray_block[ray]
    return block


def run_factorization_probe(
    n_species: int = 4,
    n_rays: int = 12,
    n_q: int = 20,
    n_scalars: int = 15,
) -> dict[str, float]:
    gather, apply = build_gcs_factorization(n_species=n_species, n_rays=n_rays, n_q=n_q)
    moment_dim = gather.shape[0]
    transport_dim = gather.shape[1]

    rng = np.random.default_rng(20260421)
    moment_core = rng.normal(size=(moment_dim, moment_dim))
    moment_core /= np.sqrt(moment_dim)

    coll_jac = apply @ moment_core @ gather
    adv_jac = build_upwind_transport_block(n_species=n_species, n_rays=n_rays, n_q=n_q)

    factor_nnz = int(np.count_nonzero(gather) + np.count_nonzero(apply) + np.count_nonzero(moment_core))
    adv_nnz = int(np.count_nonzero(adv_jac))
    materialized_transport_entries = transport_dim * transport_dim
    total_state_dim = transport_dim + n_scalars

    return {
        "transport_dim": transport_dim,
        "moment_dim": moment_dim,
        "total_state_dim": total_state_dim,
        "materialized_transport_entries": materialized_transport_entries,
        "materialized_total_entries": total_state_dim * total_state_dim,
        "collision_factor_storage_nnz": factor_nnz,
        "advection_block_nnz": adv_nnz,
        "materialized_collision_rank": int(np.linalg.matrix_rank(coll_jac)),
        "materialized_collision_density": float(np.count_nonzero(np.abs(coll_jac) > 1e-12) / coll_jac.size),
        "factor_storage_vs_materialized_ratio": float(factor_nnz / materialized_transport_entries),
        "moment_core_vs_materialized_ratio": float(moment_dim * moment_dim / materialized_transport_entries),
    }


def main() -> None:
    advection = []
    for n_q in (20, 40):
        for shift in (0.20, 0.50):
            advection.extend(run_advection_case(n_q=n_q, shift=shift))

    factorization = run_factorization_probe()

    payload = {
        "advection": [result.__dict__ for result in advection],
        "factorization": factorization,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
