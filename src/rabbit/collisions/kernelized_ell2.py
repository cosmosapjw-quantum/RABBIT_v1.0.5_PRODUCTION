"""
rabbit.collisions.kernelized_ell2 — candidate v1 ℓ=2 collision source (BD627 W1).

Flag-gated replacement for the calibrated RTA quadrupole ansatz
``C₂ = −κ₂ (Γ/H) Ψ₂`` (κ₂ = 5/3).  When the ``kernelized`` mode is selected this
module assembles ``C₂[Ψ₂](q)`` from the candidate v1 ν–e ℓ=2 kernel table.
The independent 2026-07-12 audit found that the table is electron-minus-only,
its detailed-balance gate is RED, and its convergence evidence is stale.  The
table is therefore IMPLEMENTED but NOT VALIDATED as a complete electron-sector
operator.  This path must not be used as publication evidence.

DEFAULT OFF: the production lane keeps the calibrated RTA.  This module is only
reached when ``ell2_collision_mode == "kernelized"``.

════════════════════════════════════════════════════════════════════════════════
NORMALIZATION CHAIN (the exact chain, documented per BD627 W1 requirement)
════════════════════════════════════════════════════════════════════════════════

The table stores, per (process ν_e, species, T), the linearized ℓ=2 collision
matrix in the **Γ (f-rate) basis** — the well-conditioned object the dual-route D1
gate produced (never the 1/f₀-amplified Ψ-rate; F-029/F-032 tail-dust lesson):

    nu2[T, q]        diagonal LOSS      (stored negative; = Σ −val over the integral)
    R2[T, q, q']     REDISTRIBUTION     (in-scattering)
    M2_G = -diag(nu2) + R2               f-rate matrix, units MeV
    Ĉ₂(q_i) = Σ_j M2_G[i,j] Γ₂(q_j)      f-rate amplitude dĈ₂/dt, units MeV

The SEALED Γ↔Ψ bridge (audit/spec_sectors/L2_CONVENTIONS.md, D0-GATE) is applied at
the solver boundary — state map at the source leg, rate map at the output leg, each
exactly once (never double-apply the −2):

    state map (source, per column j):  Γ₂(q_j) = −(2/√5) f₀(q_j) Ψ₂(q_j)
    rate  map (output, per row i):     dΨ₂/dt(q_i) = +√5 · Ĉ₂(q_i) / f₀(q_i)

with f₀(q) = 1/(e^{q/T}+1) the massless-FD neutrino occupancy (NOT divided by
exp(q/T) inside any table lookup — the table is already Γ-basis).  Composing:

    dΨ₂/dt(q_i) = Σ_j [ −2 f₀(q_j)/f₀(q_i) ] M2_G[i,j] Ψ₂(q_j)          [MeV]

Finally convert the MeV rate to the dimensionless per-e-fold source that the RTA C₂
it replaces uses (``C₂ = −κ₂ (Γ/H) Ψ₂`` is dΨ₂/dN = (dΨ₂/dt)/H):

    C₂(q_i) = (dΨ₂/dt)(q_i) / H_MeV,      H_MeV = H_inv_sec / 1.519267447e21

Deep-tail rows f₀(q_i/T) < 1e-6 are floored to zero (dynamically negligible; they
also carry the ill-conditioned 1/f₀(q_i) rate-map amplification, F-029).

The old magnitude comparison against calibrated `5/3` is definition- and
channel-dependent and is retained only as a diagnostic, not a validated
physical overdamping factor.

SCOPE (W1): ν_e only (nu_e process, species nue/nux).  The ν–ν process is NOT wired
(F-031: leg-4 interpolation breaks detailed balance) — NotImplementedError.  The run
q-grid MUST equal the table's N_q=20 production-inference grid (q-interpolation of the
2D kernel is out of scope for W1); a mismatch raises.  The table is a single-T object
(its extraction used one T for all occupancies, exact when T_γ≈T_ν near decoupling);
W1 interpolates and bridges at T_ν (the neutrino frame of the Ψ₂ moment).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator

# MeV → s⁻¹ (natural-units time conversion), matches projected_operator._MEV2S
_MEV2S = 1.519267447e21
_SQRT5 = np.sqrt(5.0)
_F0_TAIL_FLOOR = 1e-6

# Default promoted data asset.
_DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "ell2_kernel_tables_v1.npz"
)

# species tag → table combo key.  The v1 asset is electron-minus-only despite
# earlier ν–e± wording; PUB-00 keeps that limitation explicit.
_SPECIES_TO_COMBO = {
    "nue": "nu_e__nue",
    "nux": "nu_e__nux",
}
# tags that select the (unwired) ν–ν process — explicitly refused (F-031).
_NUNU_TAGS = {"nu_nu", "nunu", "nu_nu__nue", "nu_nu__nux"}


class _KernelTable:
    """Loaded + T-interpolated ℓ=2 kernel for one process/species combo.

    Holds PCHIP (C1 monotone, no round()/quantization — N6 lesson) interpolators
    over log-T of the Γ-basis nu2[T,q] and R2[T,q,q'] arrays.  Built once per
    (path, combo) and cached; evaluated at the run temperature per call.
    """

    def __init__(self, npz, combo):
        self.q_nodes = np.asarray(npz["q_nodes"], dtype=np.float64)
        self.T_grid = np.asarray(npz["T_grid"], dtype=np.float64)
        self._ln_T = np.log(self.T_grid)
        nu2 = np.asarray(npz[f"{combo}__nu2"], dtype=np.float64)      # (n_T, N_q)
        R2 = np.asarray(npz[f"{combo}__R2"], dtype=np.float64)        # (n_T, N_q, N_q)
        # single C1 interpolator over the T axis; evaluating returns the q / q,q' slice
        self._nu2_interp = PchipInterpolator(self._ln_T, nu2, axis=0, extrapolate=True)
        self._R2_interp = PchipInterpolator(self._ln_T, R2, axis=0, extrapolate=True)

    def M2_gamma(self, T):
        """Γ-basis f-rate matrix M2_G = -diag(nu2) + R2 interpolated (log-T) at T [MeV]."""
        ln_T = np.log(float(T))
        nu2 = np.asarray(self._nu2_interp(ln_T))       # (N_q,)
        R2 = np.asarray(self._R2_interp(ln_T))         # (N_q, N_q)
        return -np.diag(nu2) + R2


@lru_cache(maxsize=8)
def _load_table(path):
    """Load the npz once (module-level cache keyed on the resolved path)."""
    npz = np.load(path, allow_pickle=True)
    combos = {}
    for combo in _SPECIES_TO_COMBO.values():
        combos[combo] = _KernelTable(npz, combo)
    q_nodes = np.asarray(npz["q_nodes"], dtype=np.float64)
    return q_nodes, combos


def _resolve_combo(species):
    """Map a species tag to a table combo key; refuse the unwired ν–ν process."""
    tag = str(species).lower()
    if tag in _NUNU_TAGS:
        raise NotImplementedError(
            "kernelized ℓ=2 collision: the ν–ν process is not wired for W1 "
            "(F-031: leg-4 barycentric interpolation breaks detailed balance). "
            "Only the nu_e process (species 'nue'/'nux') is available."
        )
    combo = _SPECIES_TO_COMBO.get(tag)
    if combo is None:
        raise ValueError(
            f"kernelized ℓ=2 collision: unsupported species {species!r}; "
            f"expected one of {sorted(_SPECIES_TO_COMBO)} (nu_e process)."
        )
    return combo


def _c2_single(Psi2, T_nu, H_inv_sec, q_nodes, combo_table):
    """Kernelized C₂[Ψ₂](q) for a single species (1-D Ψ₂ → 1-D C₂ per e-fold)."""
    Psi2 = np.asarray(Psi2, dtype=np.float64)
    N = Psi2.shape[0]

    # Require the run grid to match the table grid exactly (W1: no q-interpolation).
    if q_nodes.shape != combo_table.q_nodes.shape or not np.allclose(
        q_nodes, combo_table.q_nodes, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "kernelized ℓ=2 collision: run q-grid does not match the table q-grid "
            f"(N_q table={combo_table.q_nodes.shape[0]}, run={q_nodes.shape[0]}). "
            "W1 requires the N_q=20 production-inference grid; q-interpolation of the "
            "2-D kernel is out of scope for W1."
        )

    H_MeV = float(H_inv_sec) / _MEV2S
    if H_MeV < 1e-50:
        return np.zeros(N)

    M2_G = combo_table.M2_gamma(T_nu)                    # (N_q, N_q), MeV, Γ-basis

    # massless-FD neutrino occupancy f₀(q/T_ν) and the deep-tail floor mask
    f0 = 1.0 / (np.exp(np.clip(q_nodes / float(T_nu), -500.0, 500.0)) + 1.0)
    tail = f0 < _F0_TAIL_FLOOR

    # state map (source columns): Γ₂(q_j) = −(2/√5) f₀(q_j) Ψ₂(q_j)
    Gamma2 = -(2.0 / _SQRT5) * f0 * Psi2
    # Γ-basis operator: Ĉ₂(q_i) = Σ_j M2_G[i,j] Γ₂(q_j)   [f-rate, MeV]
    Chat2 = M2_G @ Gamma2
    # rate map (output rows): dΨ₂/dt(q_i) = +√5 Ĉ₂(q_i) / f₀(q_i)   [MeV]
    dPsi2_dt = np.zeros(N)
    keep = ~tail
    dPsi2_dt[keep] = _SQRT5 * Chat2[keep] / f0[keep]
    # per-e-fold source (matches the RTA C₂ convention it replaces)
    return dPsi2_dt / H_MeV


def kappa2_eff(T_nu, species="nue", *, H_inv_sec=None, table_path=None):
    """Diagnostic scalar ℓ=2 ratio from the candidate v1 kernel.

    BD628 W-complete (F-033 fix).  The production tier-2 scipy CHARACTERISTIC lane
    damps ℓ=2 in ray space via a *constant* κ₂ = 5/3 (``characteristic_residual.py``
    ``delta_K = −relax·κ₂·(Γ/H)·(K−1)``).  This function returns a table-derived,
    T-dependent diagnostic that can replace that constant in the opt-in legacy
    candidate path.  It is not a channel-complete or normalization-rigorous physical
    coefficient and MUST NOT be cited as validation of the ad-hoc 5/3.

    DEFINITION (H-independent by construction).  For a representative unit-flat thermal
    quadrupole Ψ₂ ≡ 1, form the per-e-fold kernel source ``C₂[Ψ₂](q)`` (this is the same
    object as :func:`kernelized_C2_ell2`, i.e. ``(dΨ₂/dt)/H``) and the calibrated
    per-e-fold rate ``Γ/H = collision_rate_per_efold`` at the same state.  The mode-
    resolved ℓ=2 relaxation-rate ratio at momentum node q_i is

        r(q_i) = −C₂[Ψ₂](q_i) / (Γ/H).

    Both C₂ (∝ 1/H, it divides by H_MeV internally) and Γ/H (∝ 1/H) carry the SAME 1/H,
    so H cancels exactly and ``r`` — hence κ₂_eff — is independent of the Hubble rate
    used to evaluate it (see the ``H_inv_sec`` note below; verified in tests).

    κ₂_eff(T) is taken at the DOMINANT (fastest-damping) thermally-populated mode:

        κ₂_eff(T) = max over CORE nodes (f₀(q_i/T) > 1e-6) of r(q_i).

    This maximum is one legacy diagnostic projection of
    ``M₂ = −diag(nu2)+R2``.  At 2 MeV it gives the previously recorded value near
    0.40 for the electron-minus table and its unmatched scalar denominator.  Other
    channel sets and norms give materially different values, so neither 0.40 nor a
    derived overdamping factor is a validated physical claim.

    Parameters
    ----------
    T_nu : float
        Neutrino temperature [MeV].  The single-T kernel is evaluated and bridged in the
        legacy single-temperature frame.  The stored D-GATE did not pass its nominal
        detailed-balance threshold, so this frame is not described as exact.
    species : str
        'nue' / 'nuebar' (ν_e bank, ``nu_e__nue`` combo) or 'nux'
        (ν_x bank, ``nu_e__nux`` combo).  The ν–ν process raises (F-031).
    H_inv_sec : float, optional
        Hubble rate [s⁻¹] used to evaluate BOTH the kernel source and Γ/H.  κ₂_eff is
        independent of it (H cancels); defaults to a standalone thermal estimate.
    table_path : str, optional
        Override the promoted data asset path (for tests).

    Returns
    -------
    float
        Legacy diagnostic ratio for the requested species; dimensionless and clipped
        to the calibrated interval.  It is not a channel-complete physical coefficient.
    """
    from rabbit.collisions.projected_operator import (
        collision_rate_per_efold,
        _hubble_simple,
        KAPPA_ELL2,
    )
    from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX

    T = float(T_nu)
    tag = str(species).lower()
    if tag == "nuebar":          # ν̄_e shares the ν_e bank + combo
        tag = "nue"
    combo = _resolve_combo(tag)  # validates; refuses ν–ν (F-031) / unknown
    a_coupling = A_TOTAL_NUX if combo.endswith("nux") else A_TOTAL_NUE

    path = os.path.abspath(table_path or _DEFAULT_TABLE_PATH)
    q_nodes, combos = _load_table(path)

    if H_inv_sec is None:
        H_inv_sec = _hubble_simple(T, T)

    # per-e-fold kernel ℓ=2 source for a unit-flat representative thermal Ψ₂
    Psi2 = np.ones(q_nodes.shape[0])
    C2 = _c2_single(Psi2, T, H_inv_sec, q_nodes, combos[combo])

    # calibrated per-e-fold rate in the same (single-T) frame; H cancels vs C2
    GH = collision_rate_per_efold(T, T, H_inv_sec, a_coupling)
    if GH <= 0.0:
        return 0.0

    f0 = 1.0 / (np.exp(np.clip(q_nodes / T, -500.0, 500.0)) + 1.0)
    core = f0 >= _F0_TAIL_FLOOR
    if not np.any(core):
        return 0.0

    # mode-resolved ℓ=2 relaxation-rate ratio; dominant (fastest) core mode
    r = -C2[core] / GH
    val = float(np.max(r))
    # Preserve the legacy diagnostic behaviour by clipping to the calibrated interval.
    # This is not a physical bound: the electron-minus numerator and aggregate scalar
    # denominator use unmatched channel sets, and the low-temperature divergence comes
    # from the frozen single-temperature table versus the scalar electron suppression.
    # Publication paths must not use this clipped diagnostic as a collision coefficient.
    return float(np.clip(val, 0.0, KAPPA_ELL2))


def kernelized_C2_ell2(Psi2, T_gamma, T_nu, H_inv_sec, q_nodes, q_weights,
                       species, *, table_path=None):
    """Candidate v1 electron-minus ℓ=2 source C₂[Ψ₂](q), per-e-fold.

    Returns the ℓ=2 collision source in the SAME convention as the RTA C₂ it
    replaces (a dimensionless per-e-fold source on the Ψ₂ moment): for a single
    species, ``C₂ = (dΨ₂/dt)/H`` assembled from the candidate Γ-basis table with
    the sealed Γ↔Ψ bridge applied at the boundary (see module docstring).

    Parameters
    ----------
    Psi2 : ndarray, shape (N_q,) or (n_species, N_q)
        Quadrupole perturbation Ψ₂ (production Legendre convention).
    T_gamma, T_nu : float
        Photon and neutrino temperatures [MeV].  The kernel is interpolated and
        bridged at T_ν (neutrino frame; the table is a single-T object).
    H_inv_sec : float
        Hubble rate [s⁻¹].
    q_nodes, q_weights : ndarray, shape (N_q,)
        Run momentum grid; must equal the table's N_q=20 grid (raises otherwise).
        ``q_weights`` is accepted for signature parity; the loss/redistribution
        quadrature weights are already baked into the tabulated kernel.
    species : str or sequence of str
        Neutrino species tag ('nue' or 'nux').  A sequence (len n_species) is
        required when Psi2 is 2-D.  The ν–ν process raises NotImplementedError.
    table_path : str, optional
        Override the promoted data asset path (for tests).

    Returns
    -------
    ndarray, same shape as Psi2
    """
    path = os.path.abspath(table_path or _DEFAULT_TABLE_PATH)
    _, combos = _load_table(path)
    q_nodes = np.asarray(q_nodes, dtype=np.float64)
    Psi2 = np.asarray(Psi2, dtype=np.float64)

    if Psi2.ndim == 1:
        combo = _resolve_combo(species)
        return _c2_single(Psi2, T_nu, H_inv_sec, q_nodes, combos[combo])

    if Psi2.ndim == 2:
        ns = Psi2.shape[0]
        if isinstance(species, str):
            species_seq = [species] * ns
        else:
            species_seq = list(species)
        if len(species_seq) != ns:
            raise ValueError(
                f"kernelized ℓ=2 collision: species length {len(species_seq)} "
                f"!= n_species {ns} for 2-D Ψ₂."
            )
        out = np.zeros_like(Psi2)
        for s in range(ns):
            combo = _resolve_combo(species_seq[s])
            out[s] = _c2_single(Psi2[s], T_nu, H_inv_sec, q_nodes, combos[combo])
        return out

    raise ValueError(f"Psi2 must be 1-D or 2-D, got ndim={Psi2.ndim}")
