from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class RuntimeReducedBank:
    species: str
    q_nodes: np.ndarray
    q_weights: np.ndarray
    labels: List[str]
    raw_basis: np.ndarray      # (k, nq)
    ortho_basis: np.ndarray    # (k, nq), weighted-orthonormal


def _species_key(species: str) -> str:
    s = str(species).lower()
    if s in ("nuebar", "nubar_e", "nu_e_bar"):
        return "nue"
    return s


def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _interp_mode(q_src: np.ndarray, y_src: np.ndarray, q_dst: np.ndarray) -> np.ndarray:
    return np.interp(q_dst, q_src, y_src, left=y_src[0], right=y_src[-1]).astype(np.float64)


def _weighted_inner(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * a * b))


def _weighted_norm(a: np.ndarray, w: np.ndarray) -> float:
    val = _weighted_inner(a, a, w)
    return float(np.sqrt(max(val, 0.0)))


def _weighted_gram_schmidt(rows: List[np.ndarray], w: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    out: List[np.ndarray] = []
    for row in rows:
        v = np.asarray(row, dtype=np.float64).copy()
        for u in out:
            v = v - _weighted_inner(v, u, w) * u
        nrm = _weighted_norm(v, w)
        if nrm > tol:
            out.append(v / nrm)
    if not out:
        raise ValueError("No linearly independent modes survived weighted orthonormalization.")
    return np.vstack(out)


def load_runtime_bank(
    bank_path: str | Path,
    species: str,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    *,
    n_shared: int = 2,
    n_species: int = 1,
    include_fd2: bool = True,
) -> RuntimeReducedBank:
    path = Path(bank_path)
    data = json.loads(path.read_text())

    q_src = _as_array(data["q_nodes"])
    q_dst = _as_array(q_nodes)
    w_dst = _as_array(q_weights)
    w_dst = np.where(np.isfinite(w_dst), w_dst, 0.0)
    if float(np.sum(w_dst)) <= 0.0:
        raise ValueError("q_weights must have positive total weight.")

    modes: List[np.ndarray] = []
    labels: List[str] = []

    if include_fd2:
        fd2 = _interp_mode(q_src, _as_array(data["modes"]["fd2"]), q_dst)
        modes.append(fd2)
        labels.append("fd2")

    shared = data.get("modes", {}).get("shared_windowed", [])
    for i, row in enumerate(shared[: max(0, int(n_shared))]):
        modes.append(_interp_mode(q_src, _as_array(row), q_dst))
        labels.append(f"shared{i+1}")

    sp_key = _species_key(species)
    sp_modes = data.get("modes", {}).get("species_windowed", {}).get(sp_key, [])
    for i, row in enumerate(sp_modes[: max(0, int(n_species))]):
        modes.append(_interp_mode(q_src, _as_array(row), q_dst))
        labels.append(f"{sp_key}{i+1}")

    if not modes:
        raise ValueError("Requested reduced bank is empty.")

    raw_basis = np.vstack(modes)
    ortho_basis = _weighted_gram_schmidt(list(raw_basis), w_dst)

    return RuntimeReducedBank(
        species=sp_key,
        q_nodes=q_dst,
        q_weights=w_dst,
        labels=labels,
        raw_basis=raw_basis,
        ortho_basis=ortho_basis,
    )


def project_vector_onto_bank(vec: np.ndarray, bank: RuntimeReducedBank) -> Dict[str, Any]:
    y = _as_array(vec)
    B = bank.ortho_basis
    w = bank.q_weights

    coeffs = np.array([_weighted_inner(y, b, w) for b in B], dtype=np.float64)
    recon = np.sum(coeffs[:, None] * B, axis=0)

    y_norm = _weighted_norm(y, w)
    r = y - recon
    r_norm = _weighted_norm(r, w)

    cos = 0.0
    recon_norm = _weighted_norm(recon, w)
    if y_norm > 0.0 and recon_norm > 0.0:
        cos = _weighted_inner(y, recon, w) / (y_norm * recon_norm)
    cos = float(np.clip(cos, -1.0, 1.0))

    resid = float(r_norm / y_norm) if y_norm > 0.0 else 0.0

    return {
        "coeffs": coeffs,
        "recon": recon,
        "cos": cos,
        "resid": resid,
        "y_norm": float(y_norm),
        "recon_norm": float(recon_norm),
        "labels": list(bank.labels[: len(coeffs)]),
    }
