from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MacroProjectionResult:
    raw_qdot: float
    raw_n2: float
    coeff_T: float
    coeff_mu: float
    proj_qdot_T: float
    proj_qdot_Tmu: float
    orth_qdot_T: float
    orth_qdot_Tmu: float
    tail_frac_raw_last3: float
    tail_frac_orthT_last3: float
    tail_frac_orthTmu_last3: float
    signflip_index_raw: int | None
    signflip_index_orthT: int | None
    signflip_index_orthTmu: int | None


def f_eq_fd(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return 1.0 / (np.exp(np.clip(q, 0.0, 700.0)) + 1.0)


def basis_mu(q: np.ndarray) -> np.ndarray:
    f0 = f_eq_fd(q)
    return f0 * (1.0 - f0)


def basis_T(q: np.ndarray) -> np.ndarray:
    f0 = f_eq_fd(q)
    return q * f0 * (1.0 - f0)


def weighted_inner(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w * a * b))


def weighted_energy_moment(C: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    C = np.asarray(C, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # Current code convention: energy-like moment tracked through q^3 weighting.
    return float(np.sum(w * (q ** 3) * C))


def weighted_numberlike_moment(C: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    C = np.asarray(C, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w * (q ** 2) * C))


def project_T_only(C: np.ndarray, q: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    phiT = basis_T(q)
    den = weighted_inner(phiT, phiT, w)
    if abs(den) < 1e-300:
        return np.zeros_like(C), 0.0
    aT = weighted_inner(C, phiT, w) / den
    return aT * phiT, float(aT)


def project_Tmu(C: np.ndarray, q: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    # Weighted Gram-Schmidt for [phi_mu, phi_T]
    phim = basis_mu(q)
    phiT = basis_T(q)

    g11 = weighted_inner(phim, phim, w)
    if abs(g11) < 1e-300:
        return np.zeros_like(C), 0.0, 0.0

    e1 = phim / np.sqrt(g11)

    v2 = phiT - weighted_inner(phiT, e1, w) * e1
    g22 = weighted_inner(v2, v2, w)
    if abs(g22) < 1e-300:
        c1 = weighted_inner(C, e1, w)
        P = c1 * e1
        # recover coeff_mu in original basis approximately
        coeff_mu = c1 / np.sqrt(g11)
        return P, float(coeff_mu), 0.0

    e2 = v2 / np.sqrt(g22)

    c1 = weighted_inner(C, e1, w)
    c2 = weighted_inner(C, e2, w)
    P = c1 * e1 + c2 * e2

    # recover coefficients in original [phim, phiT] basis
    A = np.array([
        [weighted_inner(phim, phim, w), weighted_inner(phim, phiT, w)],
        [weighted_inner(phiT, phim, w), weighted_inner(phiT, phiT, w)],
    ], dtype=np.float64)
    b = np.array([
        weighted_inner(C, phim, w),
        weighted_inner(C, phiT, w),
    ], dtype=np.float64)

    try:
        coeff_mu, coeff_T = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_mu, coeff_T = np.nan, np.nan

    return P, float(coeff_mu), float(coeff_T)


def tail_fraction_lastk(arr: np.ndarray, k: int = 3) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    denom = float(np.sum(np.abs(arr)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(np.abs(arr[-k:])) / denom)


def first_signflip(arr: np.ndarray) -> int | None:
    arr = np.asarray(arr, dtype=np.float64)
    s = np.sign(arr)
    nz = np.where(s != 0.0)[0]
    if len(nz) < 2:
        return None
    last = s[nz[0]]
    for idx in nz[1:]:
        if s[idx] != last:
            return int(idx)
        last = s[idx]
    return None


def analyze_collision_monopole(C: np.ndarray, q: np.ndarray, w: np.ndarray) -> MacroProjectionResult:
    C = np.asarray(C, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    raw_qdot = weighted_energy_moment(C, q, w)
    raw_n2 = weighted_numberlike_moment(C, q, w)

    PT, coeff_T_only = project_T_only(C, q, w)
    O_T = C - PT

    PTmu, coeff_mu, coeff_T = project_Tmu(C, q, w)
    O_Tmu = C - PTmu

    raw_density = w * (q ** 3) * C
    orthT_density = w * (q ** 3) * O_T
    orthTmu_density = w * (q ** 3) * O_Tmu

    return MacroProjectionResult(
        raw_qdot=float(raw_qdot),
        raw_n2=float(raw_n2),
        coeff_T=float(coeff_T),
        coeff_mu=float(coeff_mu),
        proj_qdot_T=weighted_energy_moment(PT, q, w),
        proj_qdot_Tmu=weighted_energy_moment(PTmu, q, w),
        orth_qdot_T=weighted_energy_moment(O_T, q, w),
        orth_qdot_Tmu=weighted_energy_moment(O_Tmu, q, w),
        tail_frac_raw_last3=tail_fraction_lastk(raw_density, k=3),
        tail_frac_orthT_last3=tail_fraction_lastk(orthT_density, k=3),
        tail_frac_orthTmu_last3=tail_fraction_lastk(orthTmu_density, k=3),
        signflip_index_raw=first_signflip(raw_density),
        signflip_index_orthT=first_signflip(orthT_density),
        signflip_index_orthTmu=first_signflip(orthTmu_density),
    )
