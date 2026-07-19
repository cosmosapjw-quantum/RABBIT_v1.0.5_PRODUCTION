"""
rabbit.drivers.full_coupled_typeI — Type I full-coupled driver (SciPy backend).

MATURITY: INTERIM REFERENCE
  SciPy BDF remains the default and number-of-record solver path.
  Rust AOT is the active implementation target. JAX-native Rodas5P is a
  frozen explicit numerical-method/parity oracle. G-01R remains open; neither
  this driver nor the JAX oracle carries publication or public-production
  authority.

Unified 972-DOF state vector (N_q=80, phase-2 standard network):
    geometry(2) + hierarchy(960) + thermo(1) + network(9) = 972  (tier-1 thermo)

Coupling map:
    geometry  ← π (transport projectors)
    transport ← Σ (geometry), [collision terms from thermo, future Tier 2+]
    thermo    ← [ν-e exchange from transport+collision, future Tier 2+]
    weak      ← f₀(q) monopole (transport projectors)
    network   ← live λ (weak/live_rates.py)

Independent variable: N = ln a (e-folds).
Solver: SciPy BDF default/temporary number-of-record path. JAX Rodas5P is a
frozen explicit numerical-method oracle; Rust is the SPECIFIED migration
target, G-01R remains open, and no publication authority is granted.
Fidelity: Computed per-component (geometry, transport, thermo, weak, network)
and combined via min(). Tier 1 = IMPROVED_APPROX; Tier 2 = PRODUCTION_HIERARCHY.

At Tier 1 (current): collisions are off, thermo uses entropy conservation,
transport is collisionless. This is rabbit's analog of the v3.1 full
system but with LIVE weak rate evaluation (no tables).
"""
from __future__ import annotations
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from rabbit.config.grids import MomentumGrid, MultipoleSpec, DEFAULT_GRID, DEFAULT_MULTIPOLE, DEFAULT_MULTIPOLE_GENERIC
from rabbit.config.solver_config import SolverConfig, PRODUCTION_CONFIG, SolverMethod
from rabbit.config.fidelity import FidelityLevel, combined_fidelity
from rabbit.config.conventions import BianchiType, N_SPECIES_TRANSPORT
from rabbit.config.transport_mode import TransportMode

from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.geometry.constraints import friedmann_residual_typeI

from rabbit.transport.state import HierarchyState, fermi_dirac
from rabbit.transport.typeI_hierarchy import compute_hierarchy_rhs_typeI
from rabbit.transport.projectors import extract_aniso_stress, extract_monopole_distribution
from rabbit.transport.characteristic_rays import (
    setup_ray_grid, mu_current, characteristic_transport_rhs,
    extract_stress as char_extract_stress,
    extract_monopole as char_extract_monopole,
    extract_monopole_from_background as char_extract_monopole_from_background,
    ray_diagnostics,
)
from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge
from rabbit.transport.characteristic_residual import (
    apply_species_residual_relaxation,
)
from rabbit.transport.characteristic_species import (
    SPECIES as CHAR_SPECIES,
    characteristic_species_enabled,
    species_energy_weights,
    layout_characteristic_species,
)

from rabbit.thermo.incomplete_decoupling import (
    dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1, DecouplingTier,
)
from rabbit.thermo.eos_photon_electron import rho_plasma, _RHO_GAMMA_PREFACTOR
from rabbit.validation.truncation_guards import enforce_positive_typeI_omega

from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.weak.quadrature import DEFAULT_WEAK_QUADRATURE
from rabbit.weak.teff_correction import compute_teff_weak_correction

from rabbit.network.abundances_standard import (
    abundance_rhs_phase1, abundance_rhs_phase2,
    phase1_to_phase2, N_SPECIES, mass_conservation_residual,
    SPECIES_NAMES, N_REACTIONS_FULL, N_REACTIONS_BACKBONE,
)

from rabbit.results.schema import CanonicalResult, Observables, RunParameters
from rabbit.results.diagnostics import TransportDiag, WeakDiag, ConstraintDiag
from rabbit.solver.ivp_outcome import classify_solve_ivp_result
from rabbit.solver.rodas5p_adapter import solve_ivp_rodas5p
from rabbit.debug.typeI_probe_common import monopole_debug_packet
from rabbit.decoupling.solver import (
    IsotropicDecouplingConfig,
    solve_isotropic_decoupling,
)
from rabbit.decoupling.moments import _M_ELECTRON_MEV


# ═══════════════════════════════════════════════════════════════════════
# §1. Constants
# ═══════════════════════════════════════════════════════════════════════
_G_N = 6.70883e-45
_MEV_TO_S = 1.519267447e21
_DECOUPLING_BACKBONE_CACHE: dict[tuple[float, float, int, bytes, bytes], dict] = {}
# BD622-R1 (audit F-025): minimum backbone momentum-node density n_y / y_max. The backbone
# grid must keep at least this many nodes per unit of dimensionless momentum; below ~0.5
# the tier-2 observables sit far off their own continuum limit (N_q=20 gave density 0.377,
# Y_p 2.9% low, N_eff 0.275 low vs the density-refined plateau).
_BACKBONE_NODE_DENSITY_TARGET = 1.0
_TAU_N = 878.4
_ETA = 6.104e-10
_N_EFF = 3.044
_F_NU = 0.40520


def _resolve_effective_solver(requested):
    """Identity by design (BD613). Any requested->effective divergence must be introduced
    here explicitly - the drift guard at the call site raises if the method changes."""
    return requested


# ═══════════════════════════════════════════════════════════════════════
# §2. State layout
# ═══════════════════════════════════════════════════════════════════════
# y = [Σ₊, Σ₋, Ψ_flat(960), T_γ, X₀..X₈]
#      0    1   2..961       962  963..970
# Total: 972 in phase 2 with the 9-species standard network (N_q=80, tier-1 thermo)

_I_SP = 0       # Σ₊
_I_SM = 1       # Σ₋
_I_HIER = 2     # hierarchy start
# _I_HIER_END = 2 + n_transport  (computed from grid)
# _I_TG = _I_HIER_END            (T_γ)
# _I_NET = _I_TG + 1             (network start)

def _layout(grid, multipole, tier=1):
    n_transport = multipole.transport_dof(grid)
    i_hier_end = _I_HIER + n_transport
    i_tg = i_hier_end
    if tier >= 2:
        # T_γ, T_νe, T_νx
        i_tnue = i_tg + 1
        i_tnux = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tnue = -1  # unused
        i_tnux = -1
        i_net = i_tg + 1
    n_total = i_net + N_SPECIES
    return n_transport, i_hier_end, i_tg, i_tnue, i_tnux, i_net, n_total


def _layout_characteristic(N_mu, tier=1, per_species=False):
    """State layout for characteristic transport mode.

    State: [Σ₊, Σ₋, I₁..I_N, J₁..J_N, S, T_γ, [T_νe, T_νx], X₀..X₈]
    """
    if per_species:
        idx, p = layout_characteristic_species(N_mu, start=_I_HIER)
        n_transport = p - _I_HIER
        i_hier_end = p
        i_I = -1
        i_J = -1
        i_S = idx["S"]
    else:
        n_I = N_mu
        n_J = N_mu
        n_transport = 2 * N_mu + 1  # I + J + S
        i_hier_end = _I_HIER + n_transport
        i_I = _I_HIER
        i_J = _I_HIER + n_I
        i_S = _I_HIER + n_I + n_J
    i_tg = i_hier_end
    if tier >= 2:
        i_tnue = i_tg + 1
        i_tnux = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tnue = -1
        i_tnux = -1
        i_net = i_tg + 1
    n_total = i_net + N_SPECIES

    return n_transport, i_hier_end, i_tg, i_tnue, i_tnux, i_net, n_total, i_I, i_J, i_S


def _layout_native_tier1():
    """Compact state: [Sigma+, Sigma-, S, T_gamma, X_network(9)]."""
    return 1, 3, -1, -1, 4, 13, -1, -1, 2


def _characteristic_jac_sparsity(N_mu, tier=1, per_species=False):
    """Conservative Jacobian sparsity pattern for SciPy stiff solvers.

    The mask is an over-approximation: every marked dependency may be nonzero,
    and some marked entries are structurally zero. Over-approximating preserves
    exactness while still letting Radau/BDF avoid fully dense finite differences.
    """
    (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total,
     i_I, i_J, i_S) = _layout_characteristic(N_mu, tier=tier, per_species=per_species)
    mask = np.zeros((n_total, n_total), dtype=bool)

    # Geometry rows.
    if per_species:
        idx, _ = layout_characteristic_species(N_mu, start=_I_HIER)
        geo_cols = {_I_SP, _I_SM, idx["S"]}
        for sp in CHAR_SPECIES:
            geo_cols.update(range(idx[f"I_{sp}"].start, idx[f"I_{sp}"].stop))
            geo_cols.update(range(idx[f"J_{sp}"].start, idx[f"J_{sp}"].stop))
        if tier >= 2:
            geo_cols.update([i_tg, i_tne, i_tnx])
    else:
        geo_cols = set(range(_I_HIER, i_hier_end)) | {_I_SP, _I_SM}
        if tier >= 2:
            geo_cols.update([i_tg, i_tne, i_tnx])
    mask[_I_SP, list(geo_cols)] = True
    mask[_I_SM, list(geo_cols)] = True

    if per_species:
        idx, _ = layout_characteristic_species(N_mu, start=_I_HIER)
        thermo_cols = [_I_SP, _I_SM, i_tg, i_tne, i_tnx]
        for sp in CHAR_SPECIES:
            block_cols = {_I_SP, idx["S"], i_tg, i_tne, i_tnx}
            block_cols.update(range(idx[f"I_{sp}"].start, idx[f"I_{sp}"].stop))
            block_cols.update(range(idx[f"J_{sp}"].start, idx[f"J_{sp}"].stop))
            mask[idx[f"I_{sp}"], list(block_cols)] = True
            mask[idx[f"J_{sp}"], list(block_cols)] = True
        mask[idx["S"], _I_SP] = True
        mask[i_tg, thermo_cols] = True
        mask[i_tne, thermo_cols] = True
        mask[i_tnx, thermo_cols] = True

        net_cols = {_I_SP, idx["S"], i_tg, i_tne, i_tnx}
        net_cols.update(range(i_net, i_net + N_SPECIES))
        for sp in ("nue", "nuebar"):
            net_cols.update(range(idx[f"I_{sp}"].start, idx[f"I_{sp}"].stop))
            net_cols.update(range(idx[f"J_{sp}"].start, idx[f"J_{sp}"].stop))
        mask[i_net:i_net + N_SPECIES, list(net_cols)] = True
    else:
        trans_cols = {_I_SP, i_S, i_tg}
        trans_cols.update(range(i_I, i_I + N_mu))
        trans_cols.update(range(i_J, i_J + N_mu))
        if tier >= 2:
            trans_cols.update([i_tne, i_tnx])
        mask[i_I:i_I + N_mu, list(trans_cols)] = True
        mask[i_J:i_J + N_mu, list(trans_cols)] = True
        mask[i_S, _I_SP] = True

        thermo_cols = [_I_SP, _I_SM, i_tg]
        if tier >= 2:
            thermo_cols += [i_tne, i_tnx]
        mask[i_tg, thermo_cols] = True
        if tier >= 2:
            mask[i_tne, thermo_cols] = True
            mask[i_tnx, thermo_cols] = True

        net_cols = {_I_SP, i_S, i_tg}
        net_cols.update(range(i_I, i_I + N_mu))
        net_cols.update(range(i_J, i_J + N_mu))
        net_cols.update(range(i_net, i_net + N_SPECIES))
        if tier >= 2:
            net_cols.update([i_tne, i_tnx])
        mask[i_net:i_net + N_SPECIES, list(net_cols)] = True

    return csr_matrix(mask)


def _interp_history_linear(N: float, N_hist: np.ndarray, values: np.ndarray):
    xp = np.asarray(N_hist, dtype=np.float64)
    fp = np.asarray(values, dtype=np.float64)
    if xp.size == 0:
        raise ValueError("Interpolation requested on an empty history")
    if xp.size == 1 or N <= xp[0]:
        return np.array(fp[0], dtype=np.float64, copy=True)
    if N >= xp[-1]:
        return np.array(fp[-1], dtype=np.float64, copy=True)

    hi = int(np.searchsorted(xp, float(N), side="right"))
    lo = hi - 1
    x0 = float(xp[lo])
    x1 = float(xp[hi])
    if x1 <= x0 + 1.0e-15:
        return np.array(fp[lo], dtype=np.float64, copy=True)
    alpha = (float(N) - x0) / (x1 - x0)
    return (1.0 - alpha) * fp[lo] + alpha * fp[hi]


def _build_decoupling_backbone(
    config: "FullCoupledConfig",
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
) -> dict:
    q_arr = np.asarray(q_nodes, dtype=np.float64)
    q_wt = np.asarray(q_weights, dtype=np.float64)
    cache_key = (
        float(config.T_start),
        float(config.T_end),
        int(config.N_q),
        q_arr.tobytes(),
        q_wt.tobytes(),
    )
    cached = _DECOUPLING_BACKBONE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    T_backbone_start = min(float(config.T_start), 1.5)
    N_shift = float(np.log(max(float(config.T_start), 1.0e-30) / max(T_backbone_start, 1.0e-30)))
    z_start = max(T_backbone_start / _M_ELECTRON_MEV, 1.0)
    y_max = max(20.0, 1.1 * float(np.max(np.asarray(q_nodes, dtype=np.float64))) * z_start)

    # BD622-R1 (audit F-025): n_y must scale with y_max, not only N_q. The old rule
    # n_y = max(81, 2*N_q+1) pinned n_y at 81 while y_max grew with the largest
    # Gauss-Laguerre root, so raising N_q COLLAPSED the node density and the N_q
    # promotion ladder walked the under-resolved regime (the BD621 non-convergence).
    n_y = max(
        81,
        2 * int(config.N_q) + 1,
        int(np.ceil(_BACKBONE_NODE_DENSITY_TARGET * y_max)),
    )
    if n_y % 2 == 0:
        n_y += 1

    result = solve_isotropic_decoupling(
        IsotropicDecouplingConfig(
            T_start=T_backbone_start,
            T_end=float(config.T_end),
            n_y=n_y,
            y_max=y_max,
            max_efolds=20.0,
            method="Radau",
        )
    )
    if not result.success:
        raise RuntimeError(
            "Isotropic decoupling backbone did not reach the target temperature: "
            f"{result.solver_diagnostics}"
        )

    grid_nodes = np.asarray(result.grid.nodes, dtype=np.float64)
    N_hist = np.asarray(result.trajectory["N"], dtype=np.float64)
    T_gamma_hist = np.asarray(result.trajectory["T_gamma"], dtype=np.float64)
    T_nu_e_hist = np.asarray(result.trajectory["T_nu_e"], dtype=np.float64)
    T_nu_x_hist = np.asarray(result.trajectory["T_nu_x"], dtype=np.float64)
    a_hist = np.exp(N_hist)

    z_nue_hist = np.maximum(T_nu_e_hist * a_hist / _M_ELECTRON_MEV, 1.0e-30)
    z_nux_hist = np.maximum(T_nu_x_hist * a_hist / _M_ELECTRON_MEV, 1.0e-30)
    f_nue_q = np.asarray(
        [
            np.interp(q_arr * z_nue, grid_nodes, row, left=float(row[0]), right=0.0)
            for row, z_nue in zip(result.trajectory["f_nue"], z_nue_hist)
        ],
        dtype=np.float64,
    )
    f_nuebar_q = np.asarray(
        [
            np.interp(q_arr * z_nue, grid_nodes, row, left=float(row[0]), right=0.0)
            for row, z_nue in zip(result.trajectory["f_nuebar"], z_nue_hist)
        ],
        dtype=np.float64,
    )
    f_nux_q = np.asarray(
        [
            np.interp(q_arr * z_nux, grid_nodes, row, left=float(row[0]), right=0.0)
            for row, z_nux in zip(result.trajectory["f_nux"], z_nux_hist)
        ],
        dtype=np.float64,
    )
    q3 = q_wt * q_arr**3
    q4 = q3 * q_arr
    q5 = q4 * q_arr
    f_fd = 1.0 / (np.exp(np.minimum(q_arr, 500.0)) + 1.0)
    block_fd = f_fd * (1.0 - f_fd)
    ref_m3 = float(np.sum(q3 * f_fd))
    ref_m4 = float(np.sum(q4 * f_fd))
    ref_m5 = float(np.sum(q5 * f_fd))
    ref_ell0 = ref_m4 / max(ref_m3, 1.0e-300)
    ref_ell2 = ref_m5 / max(ref_m4, 1.0e-300)
    ref_block0 = float(np.sum(q4 * block_fd))
    ref_block2 = float(np.sum(q5 * block_fd))
    ref_dist0 = float(np.sum(q4 * f_fd**2))
    ref_dist2 = float(np.sum(q5 * f_fd**2))

    def _spectral_series(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows_clip = np.clip(np.asarray(rows, dtype=np.float64), 0.0, 1.0)
        m3 = rows_clip @ q3
        m4 = rows_clip @ q4
        m5 = rows_clip @ q5
        block = rows_clip * (1.0 - rows_clip)
        block0 = (block @ q4) / max(ref_block0, 1.0e-300)
        block2 = (block @ q5) / max(ref_block2, 1.0e-300)
        hard0 = (m4 / np.maximum(m3, 1.0e-300)) / max(ref_ell0, 1.0e-300)
        hard2 = (m5 / np.maximum(m4, 1.0e-300)) / max(ref_ell2, 1.0e-300)
        ell0 = np.clip(hard0 * np.sqrt(np.maximum(block0, 0.0)), 0.5, 2.0)
        ell2 = np.clip(hard2 * np.sqrt(np.maximum(block2, 0.0)), 0.5, 2.5)
        return np.asarray(ell0, dtype=np.float64), np.asarray(ell2, dtype=np.float64)

    def _distortion_series(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows_clip = np.clip(np.asarray(rows, dtype=np.float64), 0.0, 1.0)
        delta = rows_clip - f_fd[None, :]
        dist0 = np.sqrt((delta**2) @ q4 / max(ref_dist0, 1.0e-300))
        dist2 = np.sqrt((delta**2) @ q5 / max(ref_dist2, 1.0e-300))
        ell0 = np.clip(1.0 + 1.25 * dist0, 1.0, 2.0)
        ell2 = np.clip(1.0 + 1.75 * dist2, 1.0, 2.5)
        rms = np.sqrt(0.5 * (dist0**2 + dist2**2))
        return (
            np.asarray(ell0, dtype=np.float64),
            np.asarray(ell2, dtype=np.float64),
            np.asarray(rms, dtype=np.float64),
        )

    def _mismatch_series(T_bank_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ratio = np.maximum(T_gamma_hist, 1.0e-30) / np.maximum(np.asarray(T_bank_hist, dtype=np.float64), 1.0e-30)
        logs = np.abs(np.log(ratio))
        factors = np.clip(1.0 + 0.75 * logs, 1.0, 2.0)
        return np.asarray(factors, dtype=np.float64), np.asarray(logs, dtype=np.float64)

    gamma_factor_ell0_nue, gamma_factor_ell2_nue = _spectral_series(f_nue_q)
    gamma_factor_ell0_nuebar, gamma_factor_ell2_nuebar = _spectral_series(f_nuebar_q)
    gamma_factor_ell0_nux, gamma_factor_ell2_nux = _spectral_series(f_nux_q)
    distortion_factor_ell0_nue, distortion_factor_ell2_nue, distortion_rms_nue = _distortion_series(f_nue_q)
    distortion_factor_ell0_nuebar, distortion_factor_ell2_nuebar, distortion_rms_nuebar = _distortion_series(
        f_nuebar_q
    )
    distortion_factor_ell0_nux, distortion_factor_ell2_nux, distortion_rms_nux = _distortion_series(f_nux_q)
    mismatch_factor_nue, temperature_mismatch_log_nue = _mismatch_series(T_nu_e_hist)
    mismatch_factor_nuebar, temperature_mismatch_log_nuebar = _mismatch_series(T_nu_e_hist)
    mismatch_factor_nux, temperature_mismatch_log_nux = _mismatch_series(T_nu_x_hist)
    backbone = {
        "solver_diagnostics": result.solver_diagnostics,
        "N": N_hist + N_shift,
        "T_gamma": T_gamma_hist,
        "T_nu_e": T_nu_e_hist,
        "T_nu_x": T_nu_x_hist,
        "N_eff": np.asarray(result.trajectory["N_eff"], dtype=np.float64),
        "dT_gamma_dN": np.asarray(result.trajectory["dT_gamma_dN"], dtype=np.float64),
        "dT_nu_e_dN": np.asarray(result.trajectory["dT_nu_e_dN"], dtype=np.float64),
        "dT_nu_x_dN": np.asarray(result.trajectory["dT_nu_x_dN"], dtype=np.float64),
        "f_nue": f_nue_q,
        "f_nuebar": f_nuebar_q,
        "f_nux": f_nux_q,
        "q_nodes": q_arr,
        "q_weights": q_wt,
        "gamma_factor_ell0_nue": gamma_factor_ell0_nue,
        "gamma_factor_ell2_nue": gamma_factor_ell2_nue,
        "gamma_factor_ell0_nuebar": gamma_factor_ell0_nuebar,
        "gamma_factor_ell2_nuebar": gamma_factor_ell2_nuebar,
        "gamma_factor_ell0_nux": gamma_factor_ell0_nux,
        "gamma_factor_ell2_nux": gamma_factor_ell2_nux,
        "mismatch_factor_nue": mismatch_factor_nue,
        "mismatch_factor_nuebar": mismatch_factor_nuebar,
        "mismatch_factor_nux": mismatch_factor_nux,
        "temperature_mismatch_log_nue": temperature_mismatch_log_nue,
        "temperature_mismatch_log_nuebar": temperature_mismatch_log_nuebar,
        "temperature_mismatch_log_nux": temperature_mismatch_log_nux,
        "distortion_factor_ell0_nue": distortion_factor_ell0_nue,
        "distortion_factor_ell2_nue": distortion_factor_ell2_nue,
        "distortion_factor_ell0_nuebar": distortion_factor_ell0_nuebar,
        "distortion_factor_ell2_nuebar": distortion_factor_ell2_nuebar,
        "distortion_factor_ell0_nux": distortion_factor_ell0_nux,
        "distortion_factor_ell2_nux": distortion_factor_ell2_nux,
        "distortion_rms_nue": distortion_rms_nue,
        "distortion_rms_nuebar": distortion_rms_nuebar,
        "distortion_rms_nux": distortion_rms_nux,
        "grid_n_y": int(result.grid.n_y),
        "grid_y_max": float(result.grid.y_max),
        "N_shift": N_shift,
        "T_driver_start": float(config.T_start),
        "T_backbone_start": T_backbone_start,
    }
    _DECOUPLING_BACKBONE_CACHE[cache_key] = backbone
    return backbone


def _interp_decoupling_backbone(N: float, cache: dict) -> dict:
    N_hist = cache["N"]
    if float(N) <= float(cache.get("N_shift", 0.0)):
        T_eq = float(cache["T_driver_start"] * np.exp(-float(N)))
        q_nodes = np.asarray(cache["q_nodes"], dtype=np.float64)
        f_eq = 1.0 / (np.exp(np.minimum(q_nodes, 500.0)) + 1.0)
        return {
            "T_gamma": T_eq,
            "T_nu_e": T_eq,
            "T_nu_x": T_eq,
            "N_eff": 3.044,
            "dT_gamma_dN": -T_eq,
            "dT_nu_e_dN": -T_eq,
            "dT_nu_x_dN": -T_eq,
            "f_nue": np.asarray(f_eq, dtype=np.float64),
            "f_nuebar": np.asarray(f_eq, dtype=np.float64),
            "f_nux": np.asarray(f_eq, dtype=np.float64),
            "gamma_factor_ell0_nue": 1.0,
            "gamma_factor_ell2_nue": 1.0,
            "gamma_factor_ell0_nuebar": 1.0,
            "gamma_factor_ell2_nuebar": 1.0,
            "gamma_factor_ell0_nux": 1.0,
            "gamma_factor_ell2_nux": 1.0,
            "mismatch_factor_nue": 1.0,
            "mismatch_factor_nuebar": 1.0,
            "mismatch_factor_nux": 1.0,
            "temperature_mismatch_log_nue": 0.0,
            "temperature_mismatch_log_nuebar": 0.0,
            "temperature_mismatch_log_nux": 0.0,
            "distortion_factor_ell0_nue": 1.0,
            "distortion_factor_ell2_nue": 1.0,
            "distortion_factor_ell0_nuebar": 1.0,
            "distortion_factor_ell2_nuebar": 1.0,
            "distortion_factor_ell0_nux": 1.0,
            "distortion_factor_ell2_nux": 1.0,
            "distortion_rms_nue": 0.0,
            "distortion_rms_nuebar": 0.0,
            "distortion_rms_nux": 0.0,
        }
    return {
        "T_gamma": float(_interp_history_linear(N, N_hist, cache["T_gamma"])),
        "T_nu_e": float(_interp_history_linear(N, N_hist, cache["T_nu_e"])),
        "T_nu_x": float(_interp_history_linear(N, N_hist, cache["T_nu_x"])),
        "N_eff": float(_interp_history_linear(N, N_hist, cache["N_eff"])),
        "dT_gamma_dN": float(_interp_history_linear(N, N_hist, cache["dT_gamma_dN"])),
        "dT_nu_e_dN": float(_interp_history_linear(N, N_hist, cache["dT_nu_e_dN"])),
        "dT_nu_x_dN": float(_interp_history_linear(N, N_hist, cache["dT_nu_x_dN"])),
        "f_nue": np.asarray(_interp_history_linear(N, N_hist, cache["f_nue"]), dtype=np.float64),
        "f_nuebar": np.asarray(_interp_history_linear(N, N_hist, cache["f_nuebar"]), dtype=np.float64),
        "f_nux": np.asarray(_interp_history_linear(N, N_hist, cache["f_nux"]), dtype=np.float64),
        "gamma_factor_ell0_nue": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell0_nue"])),
        "gamma_factor_ell2_nue": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell2_nue"])),
        "gamma_factor_ell0_nuebar": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell0_nuebar"])),
        "gamma_factor_ell2_nuebar": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell2_nuebar"])),
        "gamma_factor_ell0_nux": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell0_nux"])),
        "gamma_factor_ell2_nux": float(_interp_history_linear(N, N_hist, cache["gamma_factor_ell2_nux"])),
        "mismatch_factor_nue": float(_interp_history_linear(N, N_hist, cache["mismatch_factor_nue"])),
        "mismatch_factor_nuebar": float(_interp_history_linear(N, N_hist, cache["mismatch_factor_nuebar"])),
        "mismatch_factor_nux": float(_interp_history_linear(N, N_hist, cache["mismatch_factor_nux"])),
        "temperature_mismatch_log_nue": float(
            _interp_history_linear(N, N_hist, cache["temperature_mismatch_log_nue"])
        ),
        "temperature_mismatch_log_nuebar": float(
            _interp_history_linear(N, N_hist, cache["temperature_mismatch_log_nuebar"])
        ),
        "temperature_mismatch_log_nux": float(
            _interp_history_linear(N, N_hist, cache["temperature_mismatch_log_nux"])
        ),
        "distortion_factor_ell0_nue": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell0_nue"])
        ),
        "distortion_factor_ell2_nue": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell2_nue"])
        ),
        "distortion_factor_ell0_nuebar": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell0_nuebar"])
        ),
        "distortion_factor_ell2_nuebar": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell2_nuebar"])
        ),
        "distortion_factor_ell0_nux": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell0_nux"])
        ),
        "distortion_factor_ell2_nux": float(
            _interp_history_linear(N, N_hist, cache["distortion_factor_ell2_nux"])
        ),
        "distortion_rms_nue": float(_interp_history_linear(N, N_hist, cache["distortion_rms_nue"])),
        "distortion_rms_nuebar": float(_interp_history_linear(N, N_hist, cache["distortion_rms_nuebar"])),
        "distortion_rms_nux": float(_interp_history_linear(N, N_hist, cache["distortion_rms_nux"])),
    }


# ═══════════════════════════════════════════════════════════════════════
# §3. Hubble rate
# ═══════════════════════════════════════════════════════════════════════

def _hubble_invsec(T_gamma, T_nu, N_eff, Sigma_sq=0.0):
    rho_em = rho_plasma(T_gamma)
    rho_nu = N_eff * (7.0/8.0) * _RHO_GAMMA_PREFACTOR * T_nu**4
    rho_total = rho_em + rho_nu
    Omega = enforce_positive_typeI_omega(Sigma_sq, context="full_coupled_typeI._hubble_invsec", strict=True)
    H_sq = (8.0 * np.pi * _G_N / 3.0) * rho_total / Omega
    return np.sqrt(max(H_sq, 0.0)) * _MEV_TO_S


# ═══════════════════════════════════════════════════════════════════════
# §4. Coupled RHS
# ═══════════════════════════════════════════════════════════════════════

def coupled_rhs(N, y, grid, multipole, phase, tau_n, eta, N_eff, f_nu, tier=1, enable_teff=False, enable_collisions=False, n_reactions=12, correction_level=0,
                transport_mode=TransportMode.CHARACTERISTIC, N_mu=12, _char_cache=None,
                ell2_collision_mode="calibrated_rta",
                characteristic_species_mode="auto",
                allow_species_identical_research=False,
                ablate_hubble_anisotropy=False,
                ablate_weak_transport_monopole=False,
                _decoupling_backbone=None, _native_workspace=None,
                _native_buffers=None):
    """Full coupled RHS for Type I.

    Parameters
    ----------
    phase : int
        1 = weak freeze-out (only X_n evolves), 2 = full network.
    tier : int
        1 = instantaneous decoupling (parametric N_eff).
        2 = live 3-temperature (T_γ, T_νe, T_νx) evolution.
    enable_teff : bool
        If True AND transport_mode is LINEARIZED_PSTF, apply Teff spectral
        hardening correction (Channel 2). Ignored in CHARACTERISTIC mode.
    enable_collisions : bool
        If True AND tier >= 2, add the bounded incomplete-decoupling
        closure. In CHARACTERISTIC mode the bounded promoted candidate path is
        the per-species transport bridge coupled to an isotropic momentum-grid
        decoupling backbone plus anisotropic residual relaxation closure.
    transport_mode : TransportMode
        LINEARIZED_PSTF or CHARACTERISTIC.
    N_mu : int
        Number of ray quadrature points (CHARACTERISTIC mode only).
    _char_cache : dict or None
        Precomputed ray grid data for CHARACTERISTIC mode.
    """
    if _native_workspace is not None:
        derivative, monopole, aux, ray_work = _native_buffers
        _native_workspace.evaluate_into(y[:4], derivative, monopole, aux, ray_work)
        # Recheck here so a corrupted/monkeypatched bridge cannot be consumed.
        from rabbit.native_tier1 import validate_native_outputs
        validate_native_outputs(derivative, monopole, aux, ray_work)
        weak = compute_live_weak_rates(
            monopole, monopole, _char_cache["q_gl"], y[3], aux[2], tau_n,
            compute_iso_reference=False, correction_level=correction_level,
        )
        dy = np.zeros_like(y)
        dy[:4] = derivative
        if phase == 1:
            dy[4] = abundance_rhs_phase1(y[4], weak.lambda_np, weak.lambda_pn) / aux[1]
            dy[5] = -dy[4]
        else:
            dy[4:4 + N_SPECIES] = abundance_rhs_phase2(
                y[4:4 + N_SPECIES], y[3], eta, weak.lambda_np, weak.lambda_pn,
                n_reactions=n_reactions,
            ) / aux[1]
        return dy

    use_char = (transport_mode == TransportMode.CHARACTERISTIC)
    backbone_bg = None

    if use_char:
        # ── Characteristic layout ──
        (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total,
         i_I, i_J, i_S) = _layout_characteristic(
            N_mu,
            tier=tier,
            per_species=(characteristic_species_mode == "per_species"),
        )

        Sigma_plus = y[_I_SP]
        Sigma_minus = y[_I_SM]
        T_gamma = y[i_tg]
        X = y[i_net:i_net + N_SPECIES]

        if T_gamma < 1e-6:
            return np.zeros(n_total)

        Sigma_sq = Sigma_plus**2 + Sigma_minus**2
        Sigma_sq_for_hubble = 0.0 if ablate_hubble_anisotropy else Sigma_sq
        if _decoupling_backbone is not None and tier >= 2 and enable_collisions:
            backbone_bg = _interp_decoupling_backbone(float(N), _decoupling_backbone)

        # ── Ray geometry ──
        mu0, w0, X0, signs = _char_cache['mu0'], _char_cache['w0'], _char_cache['X0'], _char_cache['signs']
        if characteristic_species_mode == "per_species":
            idx = _char_cache["species_idx"]
            S_val = float(y[idx["S"]])
            mu = mu_current(X0, signs, S_val)
            T_gamma_bg = float(backbone_bg["T_gamma"]) if backbone_bg is not None else float(T_gamma)
            T_nu_e = float(backbone_bg["T_nu_e"]) if backbone_bg is not None else float(y[i_tne])
            T_nu_x = float(backbone_bg["T_nu_x"]) if backbone_bg is not None else float(y[i_tnx])
            fweights = species_energy_weights(f_nu, T_nu_e, T_nu_x)

            Pi_plus = 0.0
            for sp in CHAR_SPECIES:
                I_sp = y[idx[f"I_{sp}"]]
                J_sp = y[idx[f"J_{sp}"]]
                Pi_plus += char_extract_stress(I_sp, J_sp, mu, w0, fweights[sp])

            Omega = enforce_positive_typeI_omega(
                Sigma_sq,
                context="full_coupled_typeI.coupled_rhs[per_species_char]",
                strict=True,
                floor=0.0,
            )
            dSp, dSm = compute_typeI_geometry_rhs(Sigma_plus, Sigma_minus, Pi_plus, 0.0, Omega)

            from rabbit.thermo.nudec_coupled import hubble_3T
            H_mev = hubble_3T(T_gamma_bg, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq_for_hubble)
            H = float(H_mev * _MEV_TO_S)

            q_gl = _char_cache["q_gl"]
            q_wt = _char_cache["q_wt"]

            dy = np.zeros_like(y)
            dy[_I_SP] = dSp
            dy[_I_SM] = dSm

            dS_shared = None
            for sp in CHAR_SPECIES:
                I_sp = y[idx[f"I_{sp}"]]
                J_sp = y[idx[f"J_{sp}"]]
                dI, dJ, dS = characteristic_transport_rhs(Sigma_plus, I_sp, J_sp, mu)
                if dS_shared is None:
                    dS_shared = float(dS)

                gs = apply_species_tagged_bridge(
                    species=sp,
                    I=I_sp,
                    J=J_sp,
                    w0=w0,
                    q_nodes=q_gl,
                    q_weights=q_wt,
                    T_gamma=T_gamma_bg,
                    T_nu_e=T_nu_e,
                    T_nu_x=T_nu_x,
                    H=H,
                    record_debug=False,
                ) if backbone_bg is None else None
                if backbone_bg is not None:
                    residual = apply_species_residual_relaxation(
                        species=sp,
                        I=I_sp,
                        J=J_sp,
                        w0=w0,
                        T_gamma=T_gamma_bg,
                        T_nu_e=T_nu_e,
                        T_nu_x=T_nu_x,
                        H_inv_sec=H,
                        ell2_collision_mode=ell2_collision_mode,
                        gamma_factor_ell0=float(backbone_bg[f"gamma_factor_ell0_{sp}"]),
                        gamma_factor_ell2=float(backbone_bg[f"gamma_factor_ell2_{sp}"]),
                        mismatch_factor=float(backbone_bg[f"mismatch_factor_{sp}"]),
                        distortion_factor_ell0=float(backbone_bg[f"distortion_factor_ell0_{sp}"]),
                        distortion_factor_ell2=float(backbone_bg[f"distortion_factor_ell2_{sp}"]),
                        distortion_rms=float(backbone_bg[f"distortion_rms_{sp}"]),
                    )
                    dI = dI + residual.delta_I
                    dJ = dJ + residual.delta_J
                else:
                    dI = dI + gs.delta_I
                dy[idx[f"I_{sp}"]] = dI
                dy[idx[f"J_{sp}"]] = dJ

            dy[idx["S"]] = 0.0 if dS_shared is None else dS_shared

            if backbone_bg is not None:
                dTg = float(backbone_bg["dT_gamma_dN"])
                dTne = float(backbone_bg["dT_nu_e_dN"])
                dTnx = float(backbone_bg["dT_nu_x_dN"])
                f_nue = char_extract_monopole_from_background(
                    y[idx["I_nue"]],
                    y[idx["J_nue"]],
                    w0,
                    q_gl,
                    backbone_bg["f_nue"],
                )
                f_nuebar = char_extract_monopole_from_background(
                    y[idx["I_nuebar"]],
                    y[idx["J_nuebar"]],
                    w0,
                    q_gl,
                    backbone_bg["f_nuebar"],
                )
            else:
                from rabbit.thermo.nudec_coupled import coupled_3T_rhs
                dTg, dTne, dTnx = coupled_3T_rhs(
                    T_gamma_bg,
                    T_nu_e,
                    T_nu_x,
                    H_MeV=H_mev,
                )
                f_nue = char_extract_monopole(y[idx["I_nue"]], y[idx["J_nue"]], w0, q_gl)
                f_nuebar = char_extract_monopole(y[idx["I_nuebar"]], y[idx["J_nuebar"]], w0, q_gl)
            if ablate_weak_transport_monopole:
                f_fd = 1.0 / (np.exp(np.minimum(q_gl, 500.0)) + 1.0)
                f_nue = f_fd
                f_nuebar = f_fd.copy()
            dy[i_tg] = dTg
            dy[i_tne] = dTne
            dy[i_tnx] = dTnx

            f_nue = np.clip(f_nue, 0.0, 1.0)
            f_nuebar = np.clip(f_nuebar, 0.0, 1.0)

            weak = compute_live_weak_rates(
                f_nue, f_nuebar, q_gl,
                T_gamma_bg, T_nu_e, tau_n,
                compute_iso_reference=False,
                correction_level=correction_level,
            )

            if phase == 1:
                dX = np.zeros(N_SPECIES)
                dX[0] = abundance_rhs_phase1(X[0], weak.lambda_np, weak.lambda_pn) / max(H, 1e-100)
                dX[1] = -dX[0]
            else:
                dX = abundance_rhs_phase2(
                    X, T_gamma_bg, eta, weak.lambda_np, weak.lambda_pn,
                    n_reactions=n_reactions,
                ) / max(H, 1e-100)
            dy[i_net:i_net + N_SPECIES] = dX
            return dy

        I_vals = y[i_I:i_I + N_mu]
        J_vals = y[i_J:i_J + N_mu]
        S_val = float(y[i_S])
        mu = mu_current(X0, signs, S_val)

        # ── Stress (exact) ──
        Pi_plus = char_extract_stress(I_vals, J_vals, mu, w0, f_nu)
        Pi_minus = 0.0  # LRS

        Omega = enforce_positive_typeI_omega(Sigma_sq, context="full_coupled_typeI.coupled_rhs[char]", strict=True, floor=0.0)

        dSp, dSm = compute_typeI_geometry_rhs(
            Sigma_plus, Sigma_minus, Pi_plus, Pi_minus, Omega)

        # ── Transport RHS (characteristic) ──
        dI, dJ, dS = characteristic_transport_rhs(Sigma_plus, I_vals, J_vals, mu)

        # ── Collision via Teff gather-scatter bridge (characteristic mode) ──
        if enable_collisions and tier >= 2:
            from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision
            q_gl = _char_cache['q_gl']
            q_wt = _char_cache.get('q_wt', np.ones_like(q_gl))
            T_gamma_coll = float(backbone_bg["T_gamma"]) if backbone_bg is not None else float(T_gamma)
            T_nu_coll = (
                float(backbone_bg["T_nu_e"])
                if backbone_bg is not None
                else ((T_nu_from_T_gamma_tier1(T_gamma) if tier < 2 else y[i_tne]))
            )
            H_coll = _hubble_invsec(T_gamma_coll, T_nu_coll, N_eff, Sigma_sq_for_hubble)

            gs = apply_gather_scatter_collision(
                I_vals, J_vals, w0, q_gl, q_wt,
                T_gamma_coll, T_nu_coll, H_coll)
            # Add collision-induced energy shift to ray RHS
            dI = dI + gs.delta_I

        # ── Thermo + Hubble (shared-char branch) ──
        #
        # The char + tier=2 + no-backbone path used to fall through to
        # the LINEARIZED_PSTF section below, which crashed because
        # ``_layout(grid, multipole, tier=tier)`` returned indices for a
        # 6×n_ell×N_q hierarchy and the state was char-sized.  The fix
        # hoists the monopole/weak/network/pack/return block out of the
        # tier-1 ``else:`` branch so that it runs for tier-1 and tier-2
        # alike in char mode, and only the PSTF path falls through
        # when ``use_char`` is False.
        if tier >= 2:
            if backbone_bg is not None:
                T_gamma_bg = float(backbone_bg["T_gamma"])
                T_nu_e = float(backbone_bg["T_nu_e"])
                T_nu_x = float(backbone_bg["T_nu_x"])
                from rabbit.thermo.nudec_coupled import hubble_3T
                H_mev = hubble_3T(T_gamma_bg, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq_for_hubble)
                dTg = float(backbone_bg["dT_gamma_dN"])
                dTne = float(backbone_bg["dT_nu_e_dN"])
                dTnx = float(backbone_bg["dT_nu_x_dN"])
                H = H_mev * _MEV_TO_S
                T_nu_for_rates = T_nu_e
                T_gamma_for_rates = T_gamma_bg
            else:
                T_nu_e = y[i_tne]; T_nu_x = y[i_tnx]
                from rabbit.thermo.nudec_coupled import coupled_3T_rhs, hubble_3T
                H_mev = hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq_for_hubble)
                dTg, dTne, dTnx = coupled_3T_rhs(T_gamma, T_nu_e, T_nu_x, H_MeV=H_mev)
                H = H_mev * _MEV_TO_S
                T_nu_for_rates = T_nu_e
                T_gamma_for_rates = T_gamma
        else:
            dTg = dT_gamma_dN_tier1(T_gamma)
            T_nu = T_nu_from_T_gamma_tier1(T_gamma)
            H = _hubble_invsec(T_gamma, T_nu, N_eff, Sigma_sq_for_hubble)
            T_nu_for_rates = T_nu
            T_gamma_for_rates = T_gamma

        # ── Monopole (exact, q-dependent) → Weak rates (tier-1 and tier-2) ──
        q_gl = _char_cache['q_gl']
        f_nue = char_extract_monopole(I_vals, J_vals, w0, q_gl)
        # In collisionless transport, all 6 neutrino species have identical
        # distributions (CP-symmetric free-streaming).  For Tier 2+ with
        # live ν-e collisions, ν_e receives ~5× more heating than ν_x
        # (a_e/a_x ≈ 4.68), requiring per-species monopole tracking.
        from rabbit.transport.per_species_rays import species_identical_guard
        species_identical_guard(
            tier if enable_collisions else 1,
            strict=bool(tier >= 2 and enable_collisions and not allow_species_identical_research),
        )
        f_nuebar = f_nue.copy()
        if ablate_weak_transport_monopole:
            f_fd = 1.0 / (np.exp(np.minimum(q_gl, 500.0)) + 1.0)
            f_nue = f_fd
            f_nuebar = f_fd.copy()
        f_nue = np.clip(f_nue, 0.0, 1.0)
        f_nuebar = np.clip(f_nuebar, 0.0, 1.0)

        weak = compute_live_weak_rates(
            f_nue, f_nuebar, q_gl,
            T_gamma_for_rates, T_nu_for_rates, tau_n,
            compute_iso_reference=False,
            correction_level=correction_level)

        lnp = weak.lambda_np
        lpn = weak.lambda_pn

        # ── Network RHS ──
        if phase == 1:
            dX = np.zeros(N_SPECIES)
            dX[0] = abundance_rhs_phase1(X[0], lnp, lpn) / max(H, 1e-100)
            dX[1] = -dX[0]
        else:
            dX = abundance_rhs_phase2(X, T_gamma_for_rates, eta, lnp, lpn,
                                       n_reactions=n_reactions) / max(H, 1e-100)

        # ── Pack ──
        dy = np.zeros(n_total)
        dy[_I_SP] = dSp
        dy[_I_SM] = dSm
        dy[i_I:i_I + N_mu] = dI
        dy[i_J:i_J + N_mu] = dJ
        dy[i_S] = dS
        dy[i_tg] = dTg
        if tier >= 2:
            dy[i_tne] = dTne
            dy[i_tnx] = dTnx
        dy[i_net:i_net + N_SPECIES] = dX
        return dy

    # ═══════════════════════════════════════════════════════════════
    # LINEARIZED_PSTF mode (original code path, unchanged)
    # ═══════════════════════════════════════════════════════════════
    n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total = _layout(grid, multipole, tier=tier)

    Sigma_plus = y[_I_SP]
    Sigma_minus = y[_I_SM]
    hier_flat = y[_I_HIER:i_hier_end]
    T_gamma = y[i_tg]
    X = y[i_net:i_net + N_SPECIES]

    if T_gamma < 1e-6:
        return np.zeros(n_total)

    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    Sigma_sq_for_hubble = 0.0 if ablate_hubble_anisotropy else Sigma_sq

    # ── Unpack transport ──
    state = HierarchyState.from_flat(hier_flat, grid, multipole)

    # ── Enforce collisionless monopole invariant ──
    # In collisionless Type I, dΨ₀/dN = 0 exactly (no source term).
    # The implicit solver's dense numerical Jacobian introduces spurious
    # monopole-quadrupole coupling through Newton iterations, causing Ψ₀
    # to drift from zero. This is NOT physics — it's solver contamination.
    # We project out the spurious perturbation at each RHS evaluation.
    # When collisions are enabled, Ψ₀ evolves physically and must not be zeroed.
    if not enable_collisions:
        state.data[:, 0, :] = 0.0

    # ── Transport → π → Geometry feedback ──
    stress = extract_aniso_stress(state, grid, f_nu=f_nu)
    Omega = enforce_positive_typeI_omega(Sigma_sq, context="full_coupled_typeI.coupled_rhs", strict=True, floor=0.0)

    dSp, dSm = compute_typeI_geometry_rhs(
        Sigma_plus, Sigma_minus,
        stress.Pi_plus, stress.Pi_minus, Omega)

    # ── Geometry → Σ → Transport ──
    dhier = compute_hierarchy_rhs_typeI(state, Sigma_plus, Sigma_minus)

    # ── Thermo + Hubble: branched by tier ──
    if tier >= 2:
        T_nu_e = y[i_tne]
        T_nu_x = y[i_tnx]
        from rabbit.thermo.nudec_coupled import coupled_3T_rhs, hubble_3T
        dTg, dTne, dTnx = coupled_3T_rhs(T_gamma, T_nu_e, T_nu_x)
        H = hubble_3T(T_gamma, T_nu_e, T_nu_x, Sigma_sq=Sigma_sq_for_hubble) * _MEV_TO_S
        T_nu_for_rates = T_nu_e
        # Previously undefined; needed by compute_live_weak_rates and the
        # phase-2 network-RHS call below when tier >= 2 in the PSTF path.
        T_gamma_for_rates = T_gamma
    else:
        dTg = dT_gamma_dN_tier1(T_gamma)
        T_nu = T_nu_from_T_gamma_tier1(T_gamma)
        H = _hubble_invsec(T_gamma, T_nu, N_eff, Sigma_sq_for_hubble)
        T_nu_for_rates = T_nu
        T_gamma_for_rates = T_gamma

    # ── Collision terms (tier >= 2): adds source + damping to hierarchy ──
    if tier >= 2 and enable_collisions:
        from rabbit.collisions.projected_operator import PhysicalCollisionOperator

        # Extract 3-species monopole/quadrupole from 6-species hierarchy
        Psi0_coll = np.array([state.monopole(0), state.monopole(1), state.monopole(2)])
        Psi2_coll = np.array([state.quadrupole(0), state.quadrupole(1), state.quadrupole(2)])

        coll_op = PhysicalCollisionOperator(
            grid, n_species=3, ell2_collision_mode=ell2_collision_mode)
        coll_result = coll_op.evaluate(Psi0_coll, Psi2_coll,
                                        T_gamma, T_nu_for_rates, H)

        # Map 3-species collision output → 6-species hierarchy
        N_q = grid.N_q
        dhier_3d = dhier.reshape(state.n_species, state.n_ell, N_q)
        # Species 0 (ν_e): collision species 0
        dhier_3d[0, 0, :] += coll_result.C_ell0[0]
        dhier_3d[0, 1, :] += coll_result.C_ell2[0]
        # Species 1 (ν̄_e): collision species 1
        dhier_3d[1, 0, :] += coll_result.C_ell0[1]
        dhier_3d[1, 1, :] += coll_result.C_ell2[1]
        # Species 2-5 (ν_x): collision species 2 (same for all)
        for s in range(2, state.n_species):
            dhier_3d[s, 0, :] += coll_result.C_ell0[2]
            dhier_3d[s, 1, :] += coll_result.C_ell2[2]
        dhier = dhier_3d.ravel()

    # ── Transport → f₀ → [Teff correction] → Live weak rates ──
    if backbone_bg is not None:
        f_nue = np.asarray(backbone_bg["f_nue"], dtype=np.float64)
        f_nuebar = np.asarray(backbone_bg["f_nuebar"], dtype=np.float64)
    else:
        f_nue = extract_monopole_distribution(state, grid, species_idx=0)
        f_nuebar = extract_monopole_distribution(state, grid, species_idx=1)

    if ablate_weak_transport_monopole:
        f_nue = fermi_dirac(grid.nodes)
        f_nuebar = f_nue.copy()

    # Channel 2: Teff spectral hardening correction
    if enable_teff and abs(stress.pi_tilde_mean) > 1e-20:
        f_nue, f_nuebar, _teff_diag = compute_teff_weak_correction(
            f_nue, f_nuebar, grid.nodes,
            pi_tilde_nue=float(stress.pi_tilde_per_species[0]),
            pi_tilde_nuebar=float(stress.pi_tilde_per_species[1]),
            method='exact', N_mu=16)

    weak = compute_live_weak_rates(
        f_nue, f_nuebar, grid.nodes,
        T_gamma_for_rates, T_nu_for_rates, tau_n,
        compute_iso_reference=False,
        correction_level=correction_level)

    lnp = weak.lambda_np
    lpn = weak.lambda_pn

    # ── Network RHS ──
    if phase == 1:
        dX = np.zeros(N_SPECIES)
        dX[0] = abundance_rhs_phase1(X[0], lnp, lpn) / max(H, 1e-100)
        dX[1] = -dX[0]
    else:
        dX = abundance_rhs_phase2(X, T_gamma_for_rates, eta, lnp, lpn,
                                   n_reactions=n_reactions) / max(H, 1e-100)

    # ── Pack ──
    dy = np.zeros(n_total)
    dy[_I_SP] = dSp
    dy[_I_SM] = dSm
    dy[_I_HIER:i_hier_end] = dhier
    dy[i_tg] = dTg
    if tier >= 2:
        dy[i_tne] = dTne
        dy[i_tnx] = dTnx
    dy[i_net:i_net + N_SPECIES] = dX

    return dy


# ═══════════════════════════════════════════════════════════════════════
# §5. Driver configuration and entry point
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FullCoupledConfig:
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    tau_n: float = _TAU_N
    eta: float = _ETA
    N_eff: float = _N_EFF
    f_nu: float = _F_NU
    grid: MomentumGrid = None
    multipole: MultipoleSpec = None
    solver: SolverConfig = None
    N_q: int = 80
    tier: int = 1  # Default: instantaneous decoupling (Tier 2 available via tier=2)
    enable_teff: bool = False  # characteristic reference path forbids Teff
    enable_collisions: bool = False  # Physical collision operator (tier >= 2 only)
    # BD627 W1: ℓ=2 collision closure. "calibrated_rta" (default) = the calibrated
    # −κ₂ (Γ/H) Ψ₂ ansatz; "kernelized" = the candidate v1 electron-minus table,
    # which is default-off and not publication validation.
    ell2_collision_mode: str = "calibrated_rta"
    n_reactions: int = 12  # 12=backbone (Paper I), 31=standard with ⁶Li
    correction_level: int = 0  # 0=Born, 1=+Coulomb, 2=+Sirlin, 3=+FM/WM
    transport_mode: TransportMode = TransportMode.CHARACTERISTIC
    N_mu: int = 12  # Ray quadrature points (CHARACTERISTIC mode only)
    characteristic_species_mode: str = "auto"   # auto | per_species | legacy_shared
    requested_characteristic_species_mode: str = field(init=False, repr=False, default="auto")
    allow_legacy_shared_tier2_research: bool = False
    allow_species_identical_research: bool = False
    ablate_hubble_anisotropy: bool = False
    ablate_weak_transport_monopole: bool = False
    tier1_lrs_implementation: str = "python_reference"

    def __post_init__(self):
        if self.tier1_lrs_implementation not in {"python_reference", "rust_native"}:
            raise ValueError("tier1_lrs_implementation must be 'python_reference' or 'rust_native'")
        if self.tier1_lrs_implementation == "rust_native":
            if (self.tier != 1 or self.transport_mode != TransportMode.CHARACTERISTIC
                    or self.enable_collisions or self.enable_teff
                    or abs(self.Sigma_H_minus) > 0.0
                    or self.ablate_weak_transport_monopole):
                raise NotImplementedError(
                    "rust_native requires collisionless Tier-1 LRS characteristic mode "
                    "without Teff or weak-monopole ablation"
                )
        self.requested_characteristic_species_mode = str(self.characteristic_species_mode)
        if self.characteristic_species_mode not in {"auto", "per_species", "legacy_shared"}:
            raise ValueError(
                "characteristic_species_mode must be one of "
                "{'auto', 'per_species', 'legacy_shared'}."
            )
        if self.grid is None:
            self.grid = MomentumGrid(N_q=self.N_q)
        if self.multipole is None:
            # Auto-select: generic (n_ell=3) when Σ₋ ≠ 0, LRS (n_ell=2) otherwise
            if abs(self.Sigma_H_minus) > 0.0:
                self.multipole = DEFAULT_MULTIPOLE_GENERIC
            else:
                self.multipole = DEFAULT_MULTIPOLE
        if self.solver is None:
            self.solver = PRODUCTION_CONFIG
        if (self.tier1_lrs_implementation == "rust_native"
                and self.solver.method is SolverMethod.RODAS5P):
            raise NotImplementedError(
                "rust_native R-01 is restricted to SciPy solvers; Rodas5P remains a frozen oracle"
            )
        # ── LRS guard: characteristic transport tracks only Σ₊ ──
        if (self.transport_mode == TransportMode.CHARACTERISTIC
                and abs(self.Sigma_H_minus) > 0.0):
            raise NotImplementedError(
                f"Characteristic transport is LRS-only (Σ₋ = 0). "
                f"Received Σ₋ = {self.Sigma_H_minus}. "
                f"Use transport_mode=TransportMode.LINEARIZED_PSTF for generic Type I."
            )
        if characteristic_species_enabled(
            transport_mode=self.transport_mode,
            tier=self.tier,
            enable_collisions=self.enable_collisions,
        ):
            if self.characteristic_species_mode == "auto":
                self.characteristic_species_mode = "per_species"
        # ── Sentinel-alias guard (audit F-024 / BD622-R4) ──
        # At tier < 2 the layout sets i_tne = i_tnx = -1 (unused sentinels),
        # so per_species nu-bank temperature reads would alias y[-1] = X_8
        # (last nuclear species) and live-3T thermo would silently run on
        # garbage. Forbid explicit per_species below tier 2.
        if self.characteristic_species_mode == "per_species" and self.tier < 2:
            raise ValueError(
                "characteristic_species_mode='per_species' requires tier >= 2: "
                "at tier < 2 the T_nu_e/T_nu_x state indices are -1 sentinels, "
                "so per-species temperature reads alias y[-1] = X_8 (last "
                "nuclear species) [audit F-024 / BD622-R4]. "
                "Use tier=2 (with enable_collisions) or characteristic_species_mode='auto'."
            )
        if (
            self.characteristic_species_mode == "legacy_shared"
            and self.tier >= 2
            and self.enable_collisions
            and not self.allow_legacy_shared_tier2_research
        ):
            raise ValueError(
                "legacy_shared characteristic tier2 is deprecated for production. "
                "Use per_species or explicitly set allow_legacy_shared_tier2_research=True."
            )
        if self.transport_mode == TransportMode.CHARACTERISTIC and self.enable_teff:
            raise ValueError(
                "Characteristic reference path supersedes the legacy Teff closure. "
                "Do not enable Teff on direct SciPy characteristic runs."
            )


def run_full_coupled_typeI(config: FullCoupledConfig = None, **kw) -> CanonicalResult:
    """Run the canonical Type I full-coupled BBN solve."""
    if config is None:
        config = FullCoupledConfig(**kw)

    grid = config.grid
    multipole = config.multipole
    tier = config.tier
    use_char = (config.transport_mode == TransportMode.CHARACTERISTIC)
    use_native = (config.tier1_lrs_implementation == "rust_native")
    use_char_species = (use_char and config.characteristic_species_mode == "per_species")

    # ── P1-2: High-shear safety warning ──
    _SIGMA_SAFE_MAX = 0.75  # Ω = 1 − Σ² > 0.44; solver reliable
    if config.Sigma_H_plus > _SIGMA_SAFE_MAX:
        warnings.warn(
            f"Σ_H = {config.Sigma_H_plus:.3f} exceeds safe maximum {_SIGMA_SAFE_MAX}. "
            f"Solver may be unreliable near the Milne limit (Ω → 0). "
            f"Results for Σ_H > {_SIGMA_SAFE_MAX} should be interpreted with caution.",
            stacklevel=2)

    if use_native:
        from scipy.special import roots_laguerre
        from rabbit.native_tier1 import NativeTier1Workspace
        (n_transport, i_tg, i_tne, i_tnx, i_net, n_total,
         i_I, i_J, i_S) = _layout_native_tier1()
        ray_grid = setup_ray_grid(config.N_mu)
        q_gl, q_wgl_np = roots_laguerre(config.N_q)
        q_gl = q_gl.astype(np.float64)
        q_wgl_np = q_wgl_np.astype(np.float64)
        workspace = NativeTier1Workspace(
            ray_grid=ray_grid, momentum_grid=(q_gl, q_wgl_np),
            N_eff=config.N_eff, f_nu=config.f_nu,
            ablate_hubble_anisotropy=config.ablate_hubble_anisotropy,
        )
        native_buffers = (np.empty(4), np.empty(config.N_q), np.empty(9),
                          np.empty(5 * config.N_mu))
        char_cache = {"q_gl": q_gl, "q_wt": q_wgl_np}
        f_eq = 1.0 / (np.exp(np.minimum(q_gl, 500)) + 1.0)
    elif use_char:
        from scipy.special import roots_laguerre
        (n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total,
         i_I, i_J, i_S) = _layout_characteristic(
            config.N_mu,
            tier=tier,
            per_species=use_char_species,
        )
        # Build char_cache
        mu0, w0, X0, signs = setup_ray_grid(config.N_mu)
        q_gl_np, q_wgl_np = roots_laguerre(config.N_q)
        q_gl = q_gl_np.astype(np.float64)
        f_eq = 1.0 / (np.exp(np.minimum(q_gl, 500)) + 1.0)
        char_cache = {'mu0': mu0, 'w0': w0, 'X0': X0, 'signs': signs,
                      'q_gl': q_gl, 'q_wt': q_wgl_np.astype(np.float64)}
        if use_char_species:
            species_idx, _ = layout_characteristic_species(config.N_mu, start=_I_HIER)
            char_cache["species_idx"] = species_idx
    else:
        n_transport, i_hier_end, i_tg, i_tne, i_tnx, i_net, n_total = _layout(grid, multipole, tier=tier)
        char_cache = None
        f_eq = fermi_dirac(grid.nodes)
        q_gl = grid.nodes

    decoupling_backbone = None
    if use_char and config.enable_collisions and tier >= 2:
        decoupling_backbone = _build_decoupling_backbone(
            config,
            q_gl,
            char_cache["q_wt"] if char_cache is not None else np.ones_like(q_gl),
        )

    # ── Initial conditions ──
    if decoupling_backbone is not None:
        bg0 = _interp_decoupling_backbone(0.0, decoupling_backbone)
        T_gamma_init = float(bg0["T_gamma"])
        T_nu_init = float(bg0["T_nu_e"])
        init_f_nue = np.asarray(bg0["f_nue"], dtype=np.float64)
        init_f_nuebar = np.asarray(bg0["f_nuebar"], dtype=np.float64)
    else:
        T_gamma_init = float(config.T_start)
        T_nu_init = float(config.T_start)
        init_f_nue = f_eq
        init_f_nuebar = f_eq
    init_weak = compute_live_weak_rates(
        init_f_nue, init_f_nuebar, q_gl,
        T_gamma_init, T_nu_init, config.tau_n,
        compute_iso_reference=False)
    Xn_eq = init_weak.equilibrium_Xn

    y0 = np.zeros(n_total)
    y0[_I_SP] = config.Sigma_H_plus
    y0[_I_SM] = config.Sigma_H_minus
    if use_native:
        y0[i_S] = 0.0
    elif use_char:
        if use_char_species:
            idx = char_cache["species_idx"]
            y0[idx["S"]] = 0.0
            for sp in CHAR_SPECIES:
                y0[idx[f"J_{sp}"]] = 1.0
        else:
            # I_j = 0, S = 0 (isotropic equilibrium)
            y0[i_J:i_J + config.N_mu] = 1.0  # J_j = 1 initially
    else:
        hier0 = HierarchyState.from_isotropic(grid, multipole)
        y0[_I_HIER:i_hier_end] = hier0.to_flat()
    y0[i_tg] = T_gamma_init
    if tier >= 2:
        if decoupling_backbone is not None:
            y0[i_tne] = float(bg0["T_nu_e"])
            y0[i_tnx] = float(bg0["T_nu_x"])
        else:
            y0[i_tne] = config.T_start
            y0[i_tnx] = config.T_start
    y0[i_net] = Xn_eq
    y0[i_net + 1] = 1.0 - Xn_eq

    # ── Common RHS kwargs ──
    effective_solver = _resolve_effective_solver(config.solver)
    if effective_solver.method is not config.solver.method:
        raise RuntimeError(
            f"solver drift guard (BD613): requested={config.solver.method.value} "
            f"effective={effective_solver.method.value}; silent solver remaps are forbidden")

    rhs_kw = dict(
        grid=grid, multipole=multipole,
        tau_n=config.tau_n, eta=config.eta,
        N_eff=config.N_eff, f_nu=config.f_nu,
        tier=tier, enable_teff=config.enable_teff,
        enable_collisions=config.enable_collisions,
        ell2_collision_mode=config.ell2_collision_mode,
        n_reactions=config.n_reactions,
        correction_level=config.correction_level,
        transport_mode=config.transport_mode,
        N_mu=config.N_mu, _char_cache=char_cache,
        characteristic_species_mode=config.characteristic_species_mode,
        allow_species_identical_research=config.allow_species_identical_research,
        ablate_hubble_anisotropy=config.ablate_hubble_anisotropy,
        ablate_weak_transport_monopole=config.ablate_weak_transport_monopole,
        _decoupling_backbone=decoupling_backbone,
        _native_workspace=(workspace if use_native else None),
        _native_buffers=(native_buffers if use_native else None),
    )
    # ── Solver dispatch seam (BD615) ──
    # RODAS5P is the in-tree Rosenbrock-W solver and is NOT a scipy method; it
    # routes through the rodas5p adapter. jac_sparsity is a scipy-only Radau/BDF
    # hint and is unused on the RODAS5P lane (the Rosenbrock solver builds its
    # own dense FD Jacobian) — recorded in metadata as rodas5p_jac_sparsity_unused.
    active_phase = [1]
    is_rodas5p = effective_solver.method is SolverMethod.RODAS5P
    rodas5p_jac_sparsity_unused = bool(is_rodas5p and use_char)
    if is_rodas5p:
        solver_kw = None
    else:
        solver_kw = effective_solver.to_scipy_kwargs()
        if use_native and effective_solver.scipy_method_str in ("Radau", "BDF"):
            def _native_lrs_jac(N, y):
                base = coupled_rhs(N, y, phase=active_phase[0], **rhs_kw)
                jac = np.zeros((n_total, n_total))
                for column in range(n_total):
                    if column == _I_SM:
                        continue
                    step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(y[column]))
                    shifted = y.copy()
                    shifted[column] += step
                    jac[:, column] = (
                        coupled_rhs(N, shifted, phase=active_phase[0], **rhs_kw) - base
                    ) / step
                return jac
            solver_kw["jac"] = _native_lrs_jac
        elif use_char and effective_solver.scipy_method_str in ("Radau", "BDF"):
            solver_kw["jac_sparsity"] = _characteristic_jac_sparsity(
                config.N_mu,
                tier=tier,
                per_species=use_char_species,
            )

    def _run_phase(fun, t_span, y0_phase, event, phase):
        if use_native:
            active_phase[0] = phase
        if is_rodas5p:
            return solve_ivp_rodas5p(
                fun, t_span, y0_phase, events=event,
                rtol=effective_solver.rtol, atol=effective_solver.atol,
                max_step=effective_solver.max_step,
                first_step=effective_solver.first_step,
                dense_output=effective_solver.dense_output,
                jac_reuse_max_steps=effective_solver.jac_reuse_max_steps,
            )
        return solve_ivp(fun=fun, t_span=t_span, y0=y0_phase, events=event, **solver_kw)

    # ── Phase 1: T_start → T_handoff ──
    def stop_p1(N, y): return y[i_tg] - config.T_handoff
    stop_p1.terminal = True; stop_p1.direction = -1

    t0 = time.perf_counter()

    sol1 = _run_phase(
        lambda N, y: coupled_rhs(N, y, phase=1, **rhs_kw),
        [0.0, 50.0], y0, stop_p1, 1)

    p1_target = bool(getattr(sol1, "t_events", None)) and len(sol1.t_events[0]) > 0
    phase1_solver_diag = classify_solve_ivp_result(
        sol1, target_reached=p1_target, phase="phase1")
    if phase1_solver_diag["solver_outcome"] != "target_reached":
        raise RuntimeError(f"Phase 1 did not reach handoff target: {phase1_solver_diag}")

    # ── Handoff ──
    y_handoff = sol1.y[:, -1].copy()
    Xn_handoff = y_handoff[i_net]
    X_p2 = phase1_to_phase2(Xn_handoff)
    y_handoff[i_net:i_net + N_SPECIES] = X_p2
    N_handoff = sol1.t[-1]

    phase1_probe = {
        'phase1_handoff_N': float(N_handoff),
        'phase1_handoff_T': float(y_handoff[i_tg]),
        'phase1_handoff_Xn': float(Xn_handoff),
        'phase1_handoff_sigma_plus': float(y_handoff[_I_SP]),
        'phase1_handoff_sigma_minus': float(y_handoff[_I_SM]),
    }

    if use_native:
        derivative_h, f_nue_h, aux_h, ray_work_h = native_buffers
        workspace.evaluate_into(y_handoff[:4], derivative_h, f_nue_h, aux_h, ray_work_h)
        from rabbit.native_tier1 import validate_native_outputs
        validate_native_outputs(derivative_h, f_nue_h, aux_h, ray_work_h)
        weak_h = compute_live_weak_rates(
            f_nue_h, f_nue_h, q_gl, y_handoff[i_tg], aux_h[2], config.tau_n,
            compute_iso_reference=False, correction_level=config.correction_level,
        )
        phase1_probe.update({
            'phase1_handoff_pi_plus': float(aux_h[3]),
            'phase1_handoff_T_gamma': float(y_handoff[i_tg]),
            'phase1_handoff_T_nu_e': float(aux_h[2]),
            'phase1_handoff_T_nu_x': float(aux_h[2]),
            'phase1_handoff_lambda_np': float(weak_h.lambda_np),
            'phase1_handoff_lambda_pn': float(weak_h.lambda_pn),
            'phase1_handoff_I0': float(getattr(weak_h, 'I0', np.nan)),
            'phase1_handoff_monopole_probe': monopole_debug_packet(
                q_gl, char_cache['q_wt'], f_nue_h),
        })
    elif use_char:
        mu0 = char_cache['mu0']
        w0 = char_cache['w0']
        X0 = char_cache['X0']
        signs = char_cache['signs']
        q_gl_h = char_cache['q_gl']
        q_wt_h = char_cache.get('q_wt', np.ones_like(q_gl_h))

        if tier >= 2:
            if decoupling_backbone is not None:
                bg_h = _interp_decoupling_backbone(float(N_handoff), decoupling_backbone)
                T_gamma_h = float(bg_h["T_gamma"])
                T_nu_e_h = float(bg_h["T_nu_e"])
                T_nu_x_h = float(bg_h["T_nu_x"])
            else:
                T_gamma_h = float(y_handoff[i_tg])
                T_nu_e_h = float(y_handoff[i_tne])
                T_nu_x_h = float(y_handoff[i_tnx])
        else:
            T_gamma_h = float(y_handoff[i_tg])
            T_nu_e_h = float(T_nu_from_T_gamma_tier1(y_handoff[i_tg]))
            T_nu_x_h = T_nu_e_h

        if use_char_species:
            idx = char_cache["species_idx"]
            S_h = float(y_handoff[idx["S"]])
            mu_h = mu_current(X0, signs, S_h)
            fweights_h = species_energy_weights(config.f_nu, T_nu_e_h, T_nu_x_h)

            Pi_plus_h = 0.0
            for sp in CHAR_SPECIES:
                I_sp = y_handoff[idx[f"I_{sp}"]]
                J_sp = y_handoff[idx[f"J_{sp}"]]
                Pi_plus_h += char_extract_stress(I_sp, J_sp, mu_h, w0, fweights_h[sp])

            if decoupling_backbone is not None:
                f_nue_h = char_extract_monopole_from_background(
                    y_handoff[idx["I_nue"]],
                    y_handoff[idx["J_nue"]],
                    w0,
                    q_gl_h,
                    bg_h["f_nue"],
                )
                f_nuebar_h = char_extract_monopole_from_background(
                    y_handoff[idx["I_nuebar"]],
                    y_handoff[idx["J_nuebar"]],
                    w0,
                    q_gl_h,
                    bg_h["f_nuebar"],
                )
            else:
                f_nue_h = char_extract_monopole(
                    y_handoff[idx["I_nue"]],
                    y_handoff[idx["J_nue"]],
                    w0,
                    q_gl_h,
                )
                f_nuebar_h = char_extract_monopole(
                    y_handoff[idx["I_nuebar"]],
                    y_handoff[idx["J_nuebar"]],
                    w0,
                    q_gl_h,
                )
        else:
            I_h = y_handoff[i_I:i_I + config.N_mu]
            J_h = y_handoff[i_J:i_J + config.N_mu]
            S_h = float(y_handoff[i_S])
            mu_h = mu_current(X0, signs, S_h)
            Pi_plus_h = char_extract_stress(I_h, J_h, mu_h, w0, config.f_nu)
            if decoupling_backbone is not None:
                f_nue_h = np.asarray(bg_h["f_nue"], dtype=np.float64)
                f_nuebar_h = np.asarray(bg_h["f_nuebar"], dtype=np.float64)
            else:
                f_nue_h = char_extract_monopole(I_h, J_h, w0, q_gl_h)
                f_nuebar_h = f_nue_h.copy()

        if config.ablate_weak_transport_monopole:
            f_fd_h = 1.0 / (np.exp(np.minimum(q_gl_h, 500.0)) + 1.0)
            f_nue_h = f_fd_h
            f_nuebar_h = f_fd_h.copy()

        weak_h = compute_live_weak_rates(
            f_nue_h, f_nuebar_h, q_gl_h,
            T_gamma_h, T_nu_e_h, config.tau_n,
            compute_iso_reference=False,
            correction_level=config.correction_level,
        )

        phase1_probe.update({
            'phase1_handoff_pi_plus': float(Pi_plus_h),
            'phase1_handoff_T_gamma': float(T_gamma_h),
            'phase1_handoff_T_nu_e': float(T_nu_e_h),
            'phase1_handoff_T_nu_x': float(T_nu_x_h),
            'phase1_handoff_lambda_np': float(weak_h.lambda_np),
            'phase1_handoff_lambda_pn': float(weak_h.lambda_pn),
            'phase1_handoff_I0': float(getattr(weak_h, 'I0', np.nan)),
            'phase1_handoff_monopole_probe': monopole_debug_packet(q_gl_h, q_wt_h, f_nue_h),
        })

    # ── Phase 2: T_handoff → T_end ──
    def stop_p2(N, y): return y[i_tg] - config.T_end
    stop_p2.terminal = True; stop_p2.direction = -1

    sol2 = _run_phase(
        lambda N, y: coupled_rhs(N, y, phase=2, **rhs_kw),
        [N_handoff, N_handoff + 30.0], y_handoff, stop_p2, 2)

    wall_time = time.perf_counter() - t0

    p2_target = bool(getattr(sol2, "t_events", None)) and len(sol2.t_events[0]) > 0
    phase2_solver_diag = classify_solve_ivp_result(
        sol2, target_reached=p2_target, phase="phase2")
    if phase2_solver_diag["solver_outcome"] != "target_reached":
        raise RuntimeError(f"Phase 2 did not reach final target: {phase2_solver_diag}")

    # ── Extract results ──
    y_final = sol2.y[:, -1]
    T_final = y_final[i_tg]
    X_final = y_final[i_net:i_net + N_SPECIES]
    Sigma_final = y_final[_I_SP]

    Yp = float(X_final[5])
    DH = float(X_final[2] / (2.0 * max(X_final[1], 1e-30)))
    Li7H = float((X_final[6]/7.0 + X_final[7]/7.0) / max(X_final[1], 1e-30))
    Li6H = float(max(X_final[8], 0.0) / (6.0 * max(X_final[1], 1e-30))) if N_SPECIES > 8 else 0.0
    mc = mass_conservation_residual(X_final)

    if use_native:
        derivative_f, monopole_f, aux_f, ray_work_f = native_buffers
        workspace.evaluate_into(y_final[:4], derivative_f, monopole_f, aux_f, ray_work_f)
        from rabbit.native_tier1 import validate_native_outputs
        validate_native_outputs(derivative_f, monopole_f, aux_f, ray_work_f)
        T_nu_final = float(aux_f[2])
    else:
        T_nu_final = T_nu_from_T_gamma_tier1(T_final)
    # N_eff provenance (BD598 A-R1/A-R3) — honesty layer, value unchanged:
    # at Tier ≥ 2 / with a decoupling backbone, N_eff is DERIVED from the
    # evolved neutrino sector. At Tier 1 (parametric instantaneous decoupling)
    # the reported number is the entropy-ratio DIAGNOSTIC of the parametric
    # T_ν; it is SPECIFIED, not derived, and differs from config.N_eff
    # (N_eff_input below), which seeds ρ_ν in _hubble_invsec. The gap between
    # the two is a known convention question, surfaced (not hidden) here; any
    # *derived* N_eff claim must use tier >= 2.
    if decoupling_backbone is not None:
        N_eff_meas = float(_interp_decoupling_backbone(float(sol2.t[-1]), decoupling_backbone)["N_eff"])
        N_eff_is_derived = True
    elif tier >= 2:
        from rabbit.thermo.nudec_coupled import N_eff_from_3T
        N_eff_meas = N_eff_from_3T(T_final, y_final[i_tne], y_final[i_tnx])
        N_eff_is_derived = True
    else:
        from rabbit.thermo.incomplete_decoupling import N_eff_from_T_ratio
        N_eff_meas = N_eff_from_T_ratio(T_nu_final, T_final)
        N_eff_is_derived = False

    obs = Observables(Yp=Yp, DH=DH, N_eff=N_eff_meas, Li7H=Li7H,
                      Li6H=Li6H, Xn_freeze=float(Xn_handoff),
                      N_eff_input=float(config.N_eff),
                      N_eff_is_derived=N_eff_is_derived)

    # ── N_eff self-consistency check ──
    # Warn when the reported N_eff departs from the parametric input that seeds
    # ρ_ν in the Hubble. At Tier 1 the reported value is the entropy-ratio
    # diagnostic (SPECIFIED), so a sub-threshold gap is expected and is now
    # surfaced via Observables.N_eff_input / N_eff_is_derived rather than hidden.
    neff_gap = abs(config.N_eff - N_eff_meas)
    if neff_gap > 0.05:
        derived_word = "derived" if N_eff_is_derived else "diagnostic"
        warnings.warn(
            f"N_eff gap: input={config.N_eff:.4f}, {derived_word}={N_eff_meas:.4f}, "
            # BD622-R3 (audit H-D): dY_p/dN_eff ~ 0.013, not 13.
            f"delta={neff_gap:.4f}. This produces a Y_p offset of ~{0.013*neff_gap:.1e}. "
            f"Use Tier 2 or adjust parametric N_eff to close this gap.",
            stacklevel=2)

    # ── Compute component-level fidelity ──
    fid_geometry = FidelityLevel.REFERENCE_EXACT
    if use_char:
        # Characteristic: exact collisionless (no ℓ-truncation, no linearization)
        fid_transport = FidelityLevel.REFERENCE_EXACT
    else:
        fid_transport = (FidelityLevel.PRODUCTION_HIERARCHY if tier >= 2
                         else FidelityLevel.IMPROVED_APPROX)
    fid_thermo = (FidelityLevel.PRODUCTION_HIERARCHY if tier >= 2
                  else FidelityLevel.IMPROVED_APPROX)
    fid_weak = (
        FidelityLevel.PRODUCTION_HIERARCHY
        if int(config.correction_level) >= 2
        else FidelityLevel.IMPROVED_APPROX
    )
    fid_network = FidelityLevel.PRODUCTION_HIERARCHY
    fid_overall = combined_fidelity(
        fid_geometry, fid_transport, fid_thermo, fid_weak, fid_network)

    fid_breakdown = {
        'geometry': fid_geometry.short_label,
        'transport': fid_transport.short_label,
        'thermo': fid_thermo.short_label,
        'weak': fid_weak.short_label,
        'network': fid_network.short_label,
        'overall': fid_overall.short_label,
    }
    weak_budget_mode = {
        0: 'born',
        1: 'born+coulomb',
        2: 'born+coulomb+sirlin',
        3: 'born+coulomb+sirlin+finite_mass',
    }[int(config.correction_level)]

    # Trajectories
    N_all = np.concatenate([sol1.t, sol2.t])
    Sp_all = np.concatenate([sol1.y[_I_SP, :], sol2.y[_I_SP, :]])
    T_all = np.concatenate([sol1.y[i_tg, :], sol2.y[i_tg, :]])

    transport_meta = {'transport_mode': config.transport_mode.value}
    if use_char:
        transport_meta['N_mu'] = config.N_mu
        transport_meta['n_dof_transport'] = n_transport

    if decoupling_backbone is not None and config.enable_collisions and use_char_species:
        collision_closure_mode = 'anisotropic_residual_relaxation_v1'
        thermo_exchange_mode = 'isotropic_decoupling_backbone_v1'
        weak_background_mode = 'isotropic_decoupling_backbone_v1'
        decoupling_backbone_mode = 'isotropic_momentum_grid_v1'
    elif decoupling_backbone is not None and config.enable_collisions:
        collision_closure_mode = 'shared_anisotropic_residual_relaxation_v1'
        thermo_exchange_mode = 'isotropic_decoupling_backbone_v1'
        weak_background_mode = 'isotropic_decoupling_backbone_v1'
        decoupling_backbone_mode = 'isotropic_momentum_grid_v1'
    elif config.enable_collisions and use_char_species:
        collision_closure_mode = 'legacy_species_tagged_bridge_v4'
        thermo_exchange_mode = 'table_3T_v1'
        weak_background_mode = 'transport_monopole_v1'
        decoupling_backbone_mode = 'off'
    elif config.enable_collisions:
        collision_closure_mode = 'legacy_shared_gather_scatter_v1'
        thermo_exchange_mode = 'table_3T_v1'
        weak_background_mode = 'transport_monopole_v1'
        decoupling_backbone_mode = 'off'
    elif tier >= 2:
        collision_closure_mode = 'off'
        thermo_exchange_mode = 'table_3T_v1'
        weak_background_mode = 'transport_monopole_v1'
        decoupling_backbone_mode = 'off'
    else:
        collision_closure_mode = 'off'
        thermo_exchange_mode = 'tier1_entropy_v1'
        weak_background_mode = 'transport_monopole_v1'
        decoupling_backbone_mode = 'off'

    return CanonicalResult(
        fidelity_level=fid_overall,
        grid=grid, multipole=multipole, solver=effective_solver,
        params=RunParameters(
            Sigma_H_plus=config.Sigma_H_plus,
            Sigma_H_minus=config.Sigma_H_minus,
            eta=config.eta, tau_n=config.tau_n,
            T_start=config.T_start, T_end=config.T_end,
            bianchi_type=BianchiType.TYPE_I),
        observables=obs,
        constraint_diag=ConstraintDiag(
            mass_conservation=np.array([mc])),
        trajectory={'N': N_all, 'Sigma_plus': Sp_all, 'T_gamma': T_all},
        wall_time_s=wall_time,
        metadata={'driver': 'full_coupled_typeI',
                  'solver_method_requested': config.solver.scipy_method_str,
                  'solver_method_effective': effective_solver.scipy_method_str,
                  'rodas5p_jac_sparsity_unused': rodas5p_jac_sparsity_unused,
                  'n_dof': n_total, 'phase1_steps': len(sol1.t),
                  'phase2_steps': len(sol2.t), 'tier': tier,
                  'tier1_lrs_implementation': config.tier1_lrs_implementation,
                  'enable_teff': config.enable_teff,
                  'enable_collisions': config.enable_collisions,
                  'correction_level': config.correction_level,
                  'weak_budget_mode': weak_budget_mode,
                  'correction_budget_channels': {
                      'coulomb': int(config.correction_level) >= 1,
                      'radiative_sirlin': int(config.correction_level) >= 2,
                      'finite_mass_recoil_wm': int(config.correction_level) >= 3,
                  },
                  'N_eff_input': config.N_eff,
                  'N_eff_measured': float(N_eff_meas),
                  'N_eff_gap': float(neff_gap),
                  'mass_conservation': float(mc),
                  'fidelity_breakdown': fid_breakdown,
                  'characteristic_species_mode_requested': config.requested_characteristic_species_mode,
                  'characteristic_species_mode_resolved': config.characteristic_species_mode,
                  'transport_species_mode': ('per_species' if use_char_species else 'shared'),
                  'species_identical_approx': (False if use_char_species else True),
                  'ablation_flags': {
                      'hubble_anisotropy': bool(config.ablate_hubble_anisotropy),
                      'weak_transport_monopole': bool(config.ablate_weak_transport_monopole),
                  },
                  'production_authority': (
                      'characteristic_decoupling_backbone_residual_relaxation'
                      if (decoupling_backbone is not None and config.enable_collisions and use_char_species)
                      else ('legacy_characteristic_species_bridge'
                      if (config.enable_collisions and use_char_species)
                      else ('raw_characteristic_per_species' if use_char_species else 'raw_characteristic'))
                  ),
                  'collision_closure_mode': collision_closure_mode,
                  'residual_rate_calibration_mode': (
                      'spectrum+blocking+mismatch+distortion_v1'
                      if (decoupling_backbone is not None and config.enable_collisions)
                      else 'off'
                  ),
                  'thermo_exchange_mode': thermo_exchange_mode,
                  'weak_background_mode': weak_background_mode,
                  'decoupling_backbone_mode': decoupling_backbone_mode,
                  'decoupling_backbone_solver_diagnostics': (
                      decoupling_backbone.get('solver_diagnostics') if decoupling_backbone is not None else None
                  ),
                  'phase1_solver_diagnostics': phase1_solver_diag,
                  'phase2_solver_diagnostics': phase2_solver_diag,
                  **phase1_probe,
                  **transport_meta})
