"""Private full phase-space ray driver for Type-I BBN.

This module opens the bounded PR-T3A surface:

- explicit per-ray, per-momentum distribution state
- continuous q-advection inside the existing Rodas5P contract
- collisionless canonical path plus an audit-only collision-preflight mode
- experimental factorized Jacobian payload routed through the custom
  linear-solver hook
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, partial
import os
from typing import Callable, Optional

_ALLOW_FULL_BOLTZMANN_ACCELERATOR = (
    os.environ.get("RABBIT_FULL_BOLTZMANN_ALLOW_ACCELERATOR", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
if not _ALLOW_FULL_BOLTZMANN_ACCELERATOR:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

if not _ALLOW_FULL_BOLTZMANN_ACCELERATOR:
    jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

from rabbit.config.jax_config import enable_compilation_cache
from rabbit.jax._cache_guard import _CACHE_LOCK, register_cache_clearer
from rabbit.jax.characteristic_rays_jax import P2_jax, jacobian_jax, mu_current_jax
from rabbit.jax.q_advection_jax import apply_continuous_q_advection_jax, cached_q_advection_operators
from rabbit.jax.rhs_typeI import typeI_geometry_rhs
from rabbit.jax.solver_jax_rodas5p import (
    LowRankJacobianFactors,
    jax_rodas5p_solve,
    prepare_low_rank_linear_solver,
    solve_prepared_low_rank_linear_system,
)
from rabbit.jax.nudec_coupled_jax import (
    collision_moment_3T_source_terms_jax,
    _electron_mass_suppression_jax,
    coupled_3T_rhs_from_collision_moments_jax,
    energy_transfer_rate_nue_jax,
    energy_transfer_rate_nux_jax,
    hubble_3T_jax,
    N_eff_from_3T_jax,
    nu_nu_temperature_equilibration_sources_jax,
)
from rabbit.jax.thermo_provider_jax import (
    tier1_T_nu_from_T_gamma_jax,
    tier1_hubble_aniso_jax,
    tier1_dT_gamma_dN_jax,
    tier1_N_eff_from_T_ratio_jax,
)
from rabbit.jax.weak_jax import compute_born_rates
from rabbit.jax.weak_live_jax import (
    compute_live_rates_from_monopoles_level_specialized_jax,
)
from rabbit.jax.network_jax import (
    N_SPECIES,
    abundance_rhs_phase1_jax,
    abundance_rhs_phase2_jax,
    load_rate_table,
    mass_conservation_residual_jax,
    phase1_to_phase2_jax,
    validate_rate_table_window_jax,
)
from rabbit.solver.ivp_outcome import classify_solve_ivp_result
from rabbit.validation.truncation_guards import (
    validate_manual_network_truncation,
    validate_min_resolution,
    validate_phase_temperature_order,
    validate_typeI_initial_budget,
)
from rabbit.collisions.kernels import (
    A_TOTAL_NUE,
    A_TOTAL_NUX,
    G_L_NUE,
    G_L_NUX,
    G_R_NUE,
    G_R_NUX,
)
from rabbit.jax.collisions_jax import (
    laguerre_grid as _collisions_laguerre_grid_np,
    legendre_grid as _collisions_legendre_grid_np,
    nu_e_collision_jax as _nu_e_collision_jax_kernel,
    pair_collision_jax as _pair_collision_jax_kernel,
)
from rabbit.jax.cubic_spline_jax import cubic_spline_eval_with_clamp
from rabbit.jax.nu_nu_scattering_jax import nu_nu_diagonal_collision_tensorized_jax
from rabbit.jax.collision_rates_jax import gamma_nu_nu_over_H_jax
from rabbit.jax.collision_ell2_kernelized_jax import (
    Ell2KernelTables,
    kernelized_C2_ell2_f2_jnp as _kernelized_C2_ell2_f2_jnp,
    load_ell2_kernel_tables as _load_ell2_kernel_tables,
)
from rabbit.jax.kernel_backend import inspect_kernel_backend
from rabbit.thermo.nudec_tables import _C_RATE as _MANGANO_ENERGY_TRANSFER_C_RATE


_N_EFF = 3.044
_F_NU = 0.40520
_TAU_N = 878.4
_ETA = 6.104e-10
_MEV_TO_S = 1.519267447e21
_DIST_SPECIES = ("nue", "nuebar", "nux", "nuxbar")
_COLLISION_PREFLIGHT_C_RATE = float(_MANGANO_ENERGY_TRANSFER_C_RATE)
_AP_ACCURACY_C_RATE = 280.0
_AP_ACCURACY_ENERGY_TRANSFER_SCALE = _AP_ACCURACY_C_RATE / _COLLISION_PREFLIGHT_C_RATE
_Q_THERMAL_MEAN = 3.151374371

_FULL_RHS_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_FULL_RHS_CACHE_MAXSIZE = 16


def _full_solver_collision_backend_report(collision_kernel_backend: str | None) -> dict:
    report = inspect_kernel_backend(
        collision_kernel_backend,
        scope="full_solver_collision",
    )
    metadata = report.as_dict()
    metadata["full_solver_collision_backend_contract"] = (
        "jax_xla_only_no_external_kernel_backend"
    )
    return metadata


def _resolve_full_solver_collision_kernel_backend(
    collision_kernel_backend: str | None,
) -> tuple[str, dict]:
    metadata = _full_solver_collision_backend_report(collision_kernel_backend)
    effective = str(metadata["kernel_backend_effective"])
    if effective != "jax":
        raise NotImplementedError(
            "Full-solver collision_backend currently requires a JAX-JIT-safe "
            "kernel backend. The requested backend "
            f"{collision_kernel_backend!r} resolved to {effective!r}, which is "
            "not part of the active JAX/XLA-only runtime."
        )
    return effective, metadata


def _clear_full_boltzmann_caches():
    _FULL_RHS_CACHE.clear()


register_cache_clearer(_clear_full_boltzmann_caches)


def _ensure_full_boltzmann_runtime_optimisations() -> dict:
    import os

    default_cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "rabbit_jax_cache",
    )
    cache_dir = os.environ.get("RABBIT_JAX_CACHE_DIR", default_cache_dir)
    min_compile_time_secs = float(os.environ.get("RABBIT_JAX_CACHE_MIN_COMPILE_SECS", "0.0"))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass
    enabled = bool(enable_compilation_cache(cache_dir, min_compile_time_secs=min_compile_time_secs))
    return {
        "persistent_compilation_cache_enabled": enabled,
        "compilation_cache_dir": cache_dir,
        "persistent_cache_min_compile_time_secs": min_compile_time_secs,
    }


def _build_ray_grid_numpy(N_mu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy Gauss-Legendre ray grid + auxiliaries.

    Building the arrays in numpy and converting to jnp only at the
    boundary is necessary to avoid the
    ``UnexpectedTracerError`` that arises if the first call to a
    ``lru_cache``-decorated helper happens inside a JIT trace and
    the cached jnp arrays are later returned from a different
    trace context.  The numpy intermediate guarantees the cache
    stores fully-concrete data; ``jnp.asarray`` at call sites
    materialises a fresh ``DeviceArray`` per trace.
    """
    mu0_np, w0_np = leggauss(int(N_mu))
    mu0_np = mu0_np.astype(np.float64)
    w0_np = w0_np.astype(np.float64)
    x0_np = mu0_np**2 / np.maximum(1.0 - mu0_np**2, 1e-30)
    signs_np = np.sign(mu0_np)
    signs_np = np.where(signs_np == 0.0, 1.0, signs_np)
    return mu0_np, w0_np, x0_np, signs_np


@lru_cache(maxsize=8)
def _cached_ray_grid_numpy(N_mu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _build_ray_grid_numpy(int(N_mu))


def _cached_ray_grid(N_mu: int):
    mu0_np, w0_np, x0_np, signs_np = _cached_ray_grid_numpy(int(N_mu))
    return (
        jnp.asarray(mu0_np),
        jnp.asarray(w0_np),
        jnp.asarray(x0_np),
        jnp.asarray(signs_np),
    )


@lru_cache(maxsize=8)
def _cached_laguerre_grid_numpy(N_q: int) -> tuple[np.ndarray, np.ndarray]:
    q_np, w_np = np.polynomial.laguerre.laggauss(int(N_q))
    return q_np.astype(np.float64), w_np.astype(np.float64)


def _cached_laguerre_grid(N_q: int):
    q_np, w_np = _cached_laguerre_grid_numpy(int(N_q))
    return jnp.asarray(q_np), jnp.asarray(w_np)


@lru_cache(maxsize=8)
def _cached_energy_weight_numpy(N_q: int) -> np.ndarray:
    q_np, w_np = _cached_laguerre_grid_numpy(int(N_q))
    return w_np * np.exp(q_np) * q_np**3


def _cached_energy_weight(N_q: int):
    return jnp.asarray(_cached_energy_weight_numpy(int(N_q)))


@lru_cache(maxsize=8)
def _cached_equilibrium_distribution_numpy(N_q: int) -> np.ndarray:
    q_np, _ = _cached_laguerre_grid_numpy(int(N_q))
    return 1.0 / (np.exp(np.minimum(q_np, 500.0)) + 1.0)


def _cached_equilibrium_distribution(N_q: int):
    return jnp.asarray(_cached_equilibrium_distribution_numpy(int(N_q)))


def _full_layout(N_mu: int, N_q: int, n_network_species: int, thermo_tier: int = 1) -> dict:
    n_dist = len(_DIST_SPECIES)
    i_transport = 2
    n_transport = n_dist * int(N_mu) * int(N_q)
    i_S = i_transport + n_transport
    i_tg = i_S + 1
    if int(thermo_tier) >= 2:
        i_tne = i_tg + 1
        i_tnx = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tne = -1
        i_tnx = -1
        i_net = i_tg + 1
    n_total = i_net + int(n_network_species)
    return {
        "i_Sp": 0,
        "i_Sm": 1,
        "i_transport": i_transport,
        "n_transport": n_transport,
        "i_S": i_S,
        "i_tg": i_tg,
        "i_tne": i_tne,
        "i_tnx": i_tnx,
        "i_net": i_net,
        "n_total": n_total,
        "N_mu": int(N_mu),
        "N_q": int(N_q),
        "n_dist_species": n_dist,
        "transport_shape": (n_dist, int(N_mu), int(N_q)),
        "thermo_tier": int(thermo_tier),
    }


def _transport_active_indices(layout: dict, n_network_species: int) -> jnp.ndarray:
    active = [layout["i_Sp"], layout["i_Sm"], layout["i_S"], layout["i_tg"]]
    if int(layout.get("thermo_tier", 1)) >= 2:
        active.extend([layout["i_tne"], layout["i_tnx"]])
    active.extend(range(layout["i_net"], layout["i_net"] + int(n_network_species)))
    return jnp.asarray(active, dtype=jnp.int32)


def _unpack_transport(y: jnp.ndarray, layout: dict) -> jnp.ndarray:
    i0 = int(layout["i_transport"])
    i1 = i0 + int(layout["n_transport"])
    return y[i0:i1].reshape(layout["transport_shape"])


def _pack_transport_slice(flat: jnp.ndarray, layout: dict, value: jnp.ndarray) -> jnp.ndarray:
    return flat.at[layout["i_transport"] : layout["i_transport"] + layout["n_transport"]].set(value.reshape(-1))


@jax.jit
def _species_energy_weights_jax(
    f_nu_total: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
) -> jnp.ndarray:
    raw = jnp.stack([T_nu_e**4, T_nu_e**4, 2.0 * T_nu_x**4, 2.0 * T_nu_x**4])
    return jnp.asarray(f_nu_total, dtype=jnp.float64) * raw / jnp.maximum(jnp.sum(raw), 1.0e-300)


@jax.jit
def _extract_monopole_from_rays(
    f_rays: jnp.ndarray,
    jacobian_weights: jnp.ndarray,
    w0: jnp.ndarray,
) -> jnp.ndarray:
    return 0.5 * jnp.tensordot(w0 * jacobian_weights, f_rays, axes=((0,), (0,)))


@jax.jit
def _extract_stress_from_rays(
    f_species_rays: jnp.ndarray,
    jacobian_weights: jnp.ndarray,
    mu: jnp.ndarray,
    w0: jnp.ndarray,
    energy_weight: jnp.ndarray,
    f_nu: float,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
) -> jnp.ndarray:
    # Each transport species carries a normalized spectral shape; the physical
    # energy share is set by degeneracy × T_nu^4.  The heavy sector is stored as
    # two representative species, each standing for one heavy-flavour pair.
    species_weights = _species_energy_weights_jax(jnp.asarray(f_nu), T_nu_e, T_nu_x)
    energy_per_species_ray = jax.vmap(
        lambda f_ray: jnp.tensordot(f_ray, energy_weight, axes=((1,), (0,)))
    )(f_species_rays)
    energy_per_ray = jnp.sum(
        species_weights[:, None]
        * energy_per_species_ray
        / jnp.maximum(
            jnp.tensordot(
                _cached_equilibrium_distribution(int(f_species_rays.shape[-1])),
                energy_weight,
                axes=((0,), (0,)),
            ),
            1e-100,
        ),
        axis=0,
    )
    return jnp.sum(w0 * jacobian_weights * P2_jax(mu) * energy_per_ray)


def _separate_nue_nuebar_live_kernel(correction_level: int) -> Callable:
    level = int(correction_level)
    if level not in {0, 1, 2, 3}:
        raise ValueError(f"Unsupported correction_level={correction_level}")

    return lambda T_gamma, T_nu, tau_n, q_nodes, f_nue, f_nuebar: (
        compute_live_rates_from_monopoles_level_specialized_jax(
            T_gamma,
            T_nu,
            tau_n,
            q_nodes,
            f_nue,
            f_nuebar,
            correction_level=level,
            spline_matrix=None,
        )
    )


def _build_transport_transport_block(
    Sigma_plus: jnp.ndarray,
    S_val: jnp.ndarray,
    *,
    mu0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
    q_forward_op: jnp.ndarray,
    q_backward_op: jnp.ndarray,
    layout: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mu = mu_current_jax(x0, signs, S_val)
    coeffs = Sigma_plus * P2_jax(mu)

    n_total = int(layout["n_total"])
    J = jnp.zeros((n_total, n_total), dtype=jnp.float64)
    for sp in range(int(layout["n_dist_species"])):
        for ray in range(int(layout["N_mu"])):
            block = jnp.where(coeffs[ray] >= 0.0, coeffs[ray] * q_forward_op, coeffs[ray] * q_backward_op)
            start = int(layout["i_transport"]) + (sp * int(layout["N_mu"]) + ray) * int(layout["N_q"])
            J = J.at[start : start + int(layout["N_q"]), start : start + int(layout["N_q"])].set(block)
    return J, mu, coeffs


def _block_diag_rect(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    out = jnp.zeros(
        (a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]),
        dtype=jnp.result_type(a.dtype, b.dtype),
    )
    out = out.at[: a.shape[0], : a.shape[1]].set(a)
    out = out.at[a.shape[0] :, a.shape[1] :].set(b)
    return out


def _build_collision_bank_gather(
    *,
    jacobian_weights: jnp.ndarray,
    w0: jnp.ndarray,
    layout: dict,
) -> jnp.ndarray:
    n_mu = int(layout["N_mu"])
    n_q = int(layout["N_q"])
    state_dim = int(layout["n_total"])
    species_stride = n_mu * n_q
    eye_q = jnp.eye(n_q, dtype=jnp.float64)
    mono_weights = 0.5 * w0 * jacobian_weights
    gather = jnp.zeros((3 * n_q, state_dim), dtype=jnp.float64)
    mono_block = (mono_weights[None, :, None] * eye_q[:, None, :]).reshape(
        n_q, species_stride
    )
    i0 = int(layout["i_transport"])
    gather = gather.at[0:n_q, i0 : i0 + species_stride].set(mono_block)
    gather = gather.at[n_q : 2 * n_q, i0 + species_stride : i0 + 2 * species_stride].set(
        mono_block
    )
    gather = gather.at[2 * n_q :, i0 + 2 * species_stride : i0 + 3 * species_stride].set(
        0.5 * mono_block
    )
    gather = gather.at[2 * n_q :, i0 + 3 * species_stride : i0 + 4 * species_stride].set(
        0.5 * mono_block
    )
    return gather


def _build_collision_bank_scatter(layout: dict) -> jnp.ndarray:
    n_mu = int(layout["N_mu"])
    n_q = int(layout["N_q"])
    state_dim = int(layout["n_total"])
    species_stride = n_mu * n_q
    scatter = jnp.zeros((state_dim, 3 * n_q), dtype=jnp.float64)
    eye_q = jnp.eye(n_q, dtype=jnp.float64)
    species_block = jnp.broadcast_to(eye_q, (n_mu, n_q, n_q)).reshape(
        species_stride, n_q
    )
    i0 = int(layout["i_transport"])
    scatter = scatter.at[i0 : i0 + species_stride, 0:n_q].set(species_block)
    scatter = scatter.at[i0 + species_stride : i0 + 2 * species_stride, n_q : 2 * n_q].set(
        species_block
    )
    scatter = scatter.at[i0 + 2 * species_stride : i0 + 3 * species_stride, 2 * n_q :].set(
        species_block
    )
    scatter = scatter.at[i0 + 3 * species_stride : i0 + 4 * species_stride, 2 * n_q :].set(
        species_block
    )
    return scatter


# ═══════════════════════════════════════════════════════════════════════
# BD631 FB — resolved ℓ=2 collision (gather / apply / scatter), f2-DIRECT.
#
# Adds a resolved quadrupole (ℓ=2) collision term to the JAX full-Boltzmann
# lane on top of the monopole (ℓ=0) mechanism.  The two are independent
# low-rank updates (P₀ vs P₂ angular shapes).  See
# ``audit/reference/FB_ELL2_DESIGN.md`` §A and the mandatory devil fixes
# ``audit/telemetry/fb/fb_devil.json`` (M1 Gram-Schmidt P₂⟂P₀, M2 implicit
# Jacobian block, M3 f2-direct form, M4 N2 floor).
# ═══════════════════════════════════════════════════════════════════════

_ELL2_KERNEL_MODE = "ap_unified_ell2_kernelized_preflight"


def _mode_enables_ell2_kernel(collision_mode: str) -> bool:
    """Predicate gating the resolved ℓ=2 kernel wiring (§A/§D)."""
    return str(collision_mode).strip().lower() == _ELL2_KERNEL_MODE


def _ell2_perp_and_norm(
    mu: jnp.ndarray,
    jacobian_weights: jnp.ndarray,
    w0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gram-Schmidt-orthogonalized P₂⟂P₀ under the boosted measure + its norm.

    M1 (devil, conservation blocker for strong shear).  The raw P₂(μ) leaks
    ℓ=2 → ℓ=0 at finite N_mu (Σw0·J·P2 = 2.1e-3 @Σ=0.3); the discrete-measure
    projection onto the constant,

        P2_perp(μ) = P2(μ) - (Σ w0·J·P2)/(Σ w0·J),

    makes ``Σ w0·J·P2_perp = 0`` to machine precision at ANY N_mu/shear, so the
    ℓ=2 collision preserves number + ℓ=0 energy exactly (by construction).
    ``N2 = (5/2)·Σ w0·J·P2_perp²`` is the scatter renormalization; floored at a
    small positive value (M4) so ``df2/dN / N2`` never blows up.
    """
    wj = w0 * jacobian_weights
    sum_wj = jnp.sum(wj)
    P2 = P2_jax(mu)
    sum_wj_P2 = jnp.sum(wj * P2)
    P2_perp = P2 - sum_wj_P2 / jnp.where(sum_wj == 0.0, 1.0, sum_wj)
    N2 = 2.5 * jnp.sum(wj * P2_perp * P2_perp)
    N2 = jnp.maximum(N2, 1e-12)
    return P2_perp, N2


def _build_collision_bank_gather_ell2(
    *,
    mu: jnp.ndarray,
    jacobian_weights: jnp.ndarray,
    w0: jnp.ndarray,
    layout: dict,
) -> jnp.ndarray:
    """G2 : state → (3·N_q,) physical ℓ=2 moment ``f2`` per bank (§A.i).

    ``f2[bank](q) = (5/2)·Σ_μ (w0·J·P2_perp(μ))·f[bank,μ,q]``.  Mirrors
    :func:`_build_collision_bank_gather` (monopole) but with the ℓ=2 angular
    weight ``(5/2)·w0·J·P2_perp`` in place of the P₀ weight ``(1/2)·w0·J``.
    Bank assembly matches the monopole 4-species → 3-bank structure
    (nue, nuebar, heavy=½(nux+nuxbar)).
    """
    n_mu = int(layout["N_mu"])
    n_q = int(layout["N_q"])
    state_dim = int(layout["n_total"])
    species_stride = n_mu * n_q
    eye_q = jnp.eye(n_q, dtype=jnp.float64)
    P2_perp, _N2 = _ell2_perp_and_norm(mu, jacobian_weights, w0)
    ell2_weights = 2.5 * w0 * jacobian_weights * P2_perp        # (N_mu,)
    ell2_block = (ell2_weights[None, :, None] * eye_q[:, None, :]).reshape(
        n_q, species_stride
    )
    i0 = int(layout["i_transport"])
    gather = jnp.zeros((3 * n_q, state_dim), dtype=jnp.float64)
    gather = gather.at[0:n_q, i0 : i0 + species_stride].set(ell2_block)
    gather = gather.at[n_q : 2 * n_q, i0 + species_stride : i0 + 2 * species_stride].set(
        ell2_block
    )
    gather = gather.at[2 * n_q :, i0 + 2 * species_stride : i0 + 3 * species_stride].set(
        0.5 * ell2_block
    )
    gather = gather.at[2 * n_q :, i0 + 3 * species_stride : i0 + 4 * species_stride].set(
        0.5 * ell2_block
    )
    return gather


def _build_collision_bank_scatter_ell2(
    *,
    mu: jnp.ndarray,
    jacobian_weights: jnp.ndarray,
    w0: jnp.ndarray,
    layout: dict,
) -> jnp.ndarray:
    """S2 : (3·N_q,) ``df2/dN`` per bank → state (§A.iii).

    ``δf[bank,μ,q] += (df2/dN)[bank](q)·P2_perp(μ)/N2``.  Mirrors
    :func:`_build_collision_bank_scatter` (monopole, P₀=1) but with the ℓ=2
    angular shape ``P2_perp(μ)/N2``.  ``gather∘scatter`` is the identity on the
    quadrupole moment (N2 consistency); the ℓ=0 projection of the scatter is
    zero (M1), so it changes neither number nor ℓ=0 energy.
    """
    n_mu = int(layout["N_mu"])
    n_q = int(layout["N_q"])
    state_dim = int(layout["n_total"])
    species_stride = n_mu * n_q
    eye_q = jnp.eye(n_q, dtype=jnp.float64)
    P2_perp, N2 = _ell2_perp_and_norm(mu, jacobian_weights, w0)
    coeff = P2_perp / N2                                        # (N_mu,)
    scatter_block = (coeff[:, None, None] * eye_q[None, :, :]).reshape(
        species_stride, n_q
    )
    i0 = int(layout["i_transport"])
    scatter = jnp.zeros((state_dim, 3 * n_q), dtype=jnp.float64)
    scatter = scatter.at[i0 : i0 + species_stride, 0:n_q].set(scatter_block)
    scatter = scatter.at[i0 + species_stride : i0 + 2 * species_stride, n_q : 2 * n_q].set(
        scatter_block
    )
    scatter = scatter.at[i0 + 2 * species_stride : i0 + 3 * species_stride, 2 * n_q :].set(
        scatter_block
    )
    scatter = scatter.at[i0 + 3 * species_stride : i0 + 4 * species_stride, 2 * n_q :].set(
        scatter_block
    )
    return scatter


def _collision_bank_ell2_core_jax(
    f2_bank: jnp.ndarray,
    *,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_MeV: jnp.ndarray,
    q_nodes: jnp.ndarray,
    ell2_tables: Ell2KernelTables,
) -> jnp.ndarray:
    """Apply the kernelized ℓ=2 operator to the 3-bank ``f2`` (§A.ii, M3).

    ``df2/dN[bank] = where(f0(q/T_ν)<1e-6, 0, -2·(M2_G(T_ν)@f2[bank]))/H_MeV``.
    Banks nue/nuebar use the ``nu_e__nue`` combo at ``T_nu_e``; the heavy bank
    (nux) uses ``nu_e__nux`` at ``T_nu_x``.  Linear in ``f2`` (state-independent
    given (T_ν, H)); its Jacobian block is the analytic ``A2 = -2·M2_G/H``.
    """
    n_q = int(q_nodes.shape[0])
    f2_nue = f2_bank[0:n_q]
    f2_nuebar = f2_bank[n_q : 2 * n_q]
    f2_nux = f2_bank[2 * n_q :]
    dN_nue = _kernelized_C2_ell2_f2_jnp(
        f2_nue, T_nu_e, H_MeV, ell2_tables.lnT, ell2_tables.nue_nu2, ell2_tables.nue_R2, q_nodes
    )
    dN_nuebar = _kernelized_C2_ell2_f2_jnp(
        f2_nuebar, T_nu_e, H_MeV, ell2_tables.lnT, ell2_tables.nue_nu2, ell2_tables.nue_R2, q_nodes
    )
    dN_nux = _kernelized_C2_ell2_f2_jnp(
        f2_nux, T_nu_x, H_MeV, ell2_tables.lnT, ell2_tables.nux_nu2, ell2_tables.nux_R2, q_nodes
    )
    return jnp.concatenate([dN_nue, dN_nuebar, dN_nux])


@jax.jit
def _compute_energy_exchange_rate_jax(
    collision_C: jnp.ndarray,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_nu_MeV: jnp.ndarray,
) -> jnp.ndarray:
    integrand = q_nodes**3 * collision_C * jnp.exp(q_nodes)
    energy_integral = jnp.sum(q_weights * integrand)
    prefactor = T_nu_MeV**4 / (2.0 * jnp.pi**2)
    return -prefactor * energy_integral


@jax.jit
def _collision_shape_number_moment_jax(
    collision_C: jnp.ndarray,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(q_weights * q_nodes**2 * collision_C * jnp.exp(q_nodes))


@jax.jit
def _collision_shape_energy_moment_jax(
    collision_C: jnp.ndarray,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(q_weights * q_nodes**3 * collision_C * jnp.exp(q_nodes))


@jax.jit
def _project_number_energy_neutral_shape_jax(
    raw_C: jnp.ndarray,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Project a spectral damping shape to conserve number and energy."""
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    basis_n = f_eq * jnp.maximum(1.0 - f_eq, 1.0e-30)
    basis_e = q_nodes * basis_n

    mat = jnp.asarray(
        [
            [
                _collision_shape_number_moment_jax(basis_n, q_nodes, q_weights),
                _collision_shape_number_moment_jax(basis_e, q_nodes, q_weights),
            ],
            [
                _collision_shape_energy_moment_jax(basis_n, q_nodes, q_weights),
                _collision_shape_energy_moment_jax(basis_e, q_nodes, q_weights),
            ],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray(
        [
            _collision_shape_number_moment_jax(raw_C, q_nodes, q_weights),
            _collision_shape_energy_moment_jax(raw_C, q_nodes, q_weights),
        ],
        dtype=jnp.float64,
    )
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    valid = jnp.abs(det) > 1.0e-100
    inv_det = jnp.where(valid, 1.0 / det, 0.0)
    coeff = jnp.asarray(
        [
            (rhs[0] * mat[1, 1] - mat[0, 1] * rhs[1]) * inv_det,
            (mat[0, 0] * rhs[1] - rhs[0] * mat[1, 0]) * inv_det,
        ],
        dtype=jnp.float64,
    )
    return raw_C - coeff[0] * basis_n - coeff[1] * basis_e


@jax.jit
def _nu_nu_self_thermalization_shape_jax(
    f_mono: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_bank: jnp.ndarray,
    H_MeV: jnp.ndarray,
) -> jnp.ndarray:
    """Leading-order diagonal nu-nu spectral self-thermalization.

    Elastic same-sector nu-nu scattering can relax spectral distortions while
    conserving that species' number and energy.  This AP-form shape damping is
    only used inside the spectral nu-nu preflight mode; it is not a QKE flavour
    source.
    """
    f_clip = jnp.clip(f_mono, 0.0, 1.0)
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    gamma_over_H = gamma_nu_nu_over_H_jax(
        T_bank,
        H_MeV,
        alpha_eq_beta=True,
    )
    momentum_shape = jnp.clip(q_nodes / _Q_THERMAL_MEAN, 0.25, 8.0)
    raw = -gamma_over_H * momentum_shape * (f_clip - f_eq)
    return _project_number_energy_neutral_shape_jax(raw, q_nodes, q_weights)


@jax.jit
def _rate_enforcement_scale_jax(
    raw_plasma: jnp.ndarray,
    target_plasma: jnp.ndarray,
    min_abs: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """Positive scalar that can only reinforce a correctly signed source."""
    raw_plasma = jnp.asarray(raw_plasma, dtype=jnp.float64)
    target_plasma = jnp.asarray(target_plasma, dtype=jnp.float64)
    min_abs = jnp.maximum(jnp.asarray(min_abs, dtype=jnp.float64), 1.0e-300)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    raw_abs = jnp.abs(raw_plasma)
    target_abs = jnp.abs(target_plasma)
    raw_sign = jnp.sign(raw_plasma)
    denom = jnp.where(raw_abs < min_abs, raw_sign * min_abs, raw_plasma)
    scale_full = target_plasma / jnp.where(denom == 0.0, min_abs, denom)
    scale = jnp.clip(scale_full, lower, upper)
    same_direction = raw_plasma * target_plasma > 0.0
    active = target_abs >= min_abs
    return jnp.where(active & same_direction, scale, 0.0)


@jax.jit
def _source_scale_jax(
    source_term: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_bank: jnp.ndarray,
    target_plasma_energy_exchange: jnp.ndarray,
) -> jnp.ndarray:
    raw_plasma = _compute_energy_exchange_rate_jax(source_term, q_nodes, q_weights, T_bank)
    return _rate_enforcement_scale_jax(
        raw_plasma,
        target_plasma_energy_exchange,
        jnp.asarray(1.0e-80, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(100.0, dtype=jnp.float64),
    )


@jax.jit
def _collision_rate_per_efold_jax(
    T_gamma: jnp.ndarray,
    T_nu: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
    coupling: jnp.ndarray,
) -> jnp.ndarray:
    suppression = _electron_mass_suppression_jax(T_gamma)
    gamma_mev = (
        (_COLLISION_PREFLIGHT_C_RATE / jnp.pi**5)
        * (jnp.asarray(1.1663788e-11, dtype=jnp.float64) ** 2)
        * coupling
        * suppression
        * T_gamma**4
        * T_nu
    )
    H_mev = H_inv_sec / _MEV_TO_S
    return jnp.where(H_mev < 1.0e-50, 0.0, gamma_mev / H_mev)


def _collision_preflight_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
    closure_strength: jnp.ndarray,
) -> jnp.ndarray:
    n_q = int(q_nodes.shape[0])
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    momentum_shape = jnp.clip(q_nodes / _Q_THERMAL_MEAN, 0.0, 8.0)
    H_mev = H_inv_sec / _MEV_TO_S

    def _one_species(
        f_mono: jnp.ndarray,
        coupling: float,
        target_rate: jnp.ndarray,
        T_bank: jnp.ndarray,
    ) -> jnp.ndarray:
        f_clip = jnp.clip(f_mono, 0.0, 1.0)
        f_target = 1.0 / (
            jnp.exp(jnp.minimum(q_nodes * T_bank / jnp.maximum(T_gamma, 1.0e-30), 500.0)) + 1.0
        )
        psi_target = (f_target - f_eq) / jnp.maximum(f_eq, 1.0e-30)
        gamma_over_H = _collision_rate_per_efold_jax(
            T_gamma,
            T_bank,
            H_inv_sec,
            jnp.asarray(coupling, dtype=jnp.float64),
        )
        gamma_q = gamma_over_H * momentum_shape
        psi_actual = (f_clip - f_eq) / jnp.maximum(f_eq, 1.0e-30)
        energy_basis = gamma_q * q_nodes * jnp.maximum(1.0 - f_eq, 1.0e-30)
        source_raw = gamma_q * psi_target
        damping_raw = -gamma_q * psi_actual
        damping_plasma = _compute_energy_exchange_rate_jax(damping_raw, q_nodes, q_weights, T_bank)
        basis_plasma = _compute_energy_exchange_rate_jax(energy_basis, q_nodes, q_weights, T_bank)
        damping_raw = jnp.where(
            jnp.abs(basis_plasma) > 1.0e-80,
            damping_raw - (damping_plasma / basis_plasma) * energy_basis,
            damping_raw,
        )
        target_plasma_dQ_dN = -target_rate / jnp.maximum(H_mev, 1.0e-80)
        source_scale = _source_scale_jax(
            source_raw,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_bank=T_bank,
            target_plasma_energy_exchange=target_plasma_dQ_dN,
        )
        return source_scale * source_raw + closure_strength * damping_raw

    f_nue = bank_state[:n_q]
    f_nuebar = bank_state[n_q : 2 * n_q]
    f_nux = bank_state[2 * n_q :]
    C_nue = _one_species(f_nue, A_TOTAL_NUE, energy_transfer_rate_nue_jax(T_gamma, T_nu_e), T_nu_e)
    C_nuebar = _one_species(f_nuebar, A_TOTAL_NUE, energy_transfer_rate_nue_jax(T_gamma, T_nu_e), T_nu_e)
    C_nux = _one_species(f_nux, A_TOTAL_NUX, energy_transfer_rate_nux_jax(T_gamma, T_nu_x), T_nu_x)
    return jnp.concatenate([C_nue, C_nuebar, C_nux])


def _collision_projected_physical_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
) -> jnp.ndarray:
    """Projected physical source+damping core on the species bank state.

    This is the JAX mirror of the host-side projected physical preflight:
    state-dependent monopole damping plus a linearized thermal source term,
    returned directly as ``df/dN`` on the bank distributions.
    """
    n_q = int(q_nodes.shape[0])
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)

    def _one_species(
        f_mono: jnp.ndarray,
        coupling: float,
        T_bank: jnp.ndarray,
    ) -> jnp.ndarray:
        f_clip = jnp.clip(f_mono, 0.0, 1.0)
        gamma_over_H = _collision_rate_per_efold_jax(
            T_gamma,
            T_bank,
            H_inv_sec,
            jnp.asarray(coupling, dtype=jnp.float64),
        )
        psi_actual = (f_clip - f_eq) / jnp.maximum(f_eq, 1.0e-30)
        psi_target = (
            (T_gamma - T_bank) / jnp.maximum(T_gamma, 1.0e-30)
        ) * q_nodes * jnp.maximum(1.0 - f_eq, 1.0e-30)
        c_psi = gamma_over_H * (psi_target - psi_actual)
        return f_eq * c_psi

    f_nue = bank_state[:n_q]
    f_nuebar = bank_state[n_q : 2 * n_q]
    f_nux = bank_state[2 * n_q :]
    C_nue = _one_species(f_nue, A_TOTAL_NUE, T_nu_e)
    C_nuebar = _one_species(f_nuebar, A_TOTAL_NUE, T_nu_e)
    C_nux = _one_species(f_nux, A_TOTAL_NUX, T_nu_x)
    return jnp.concatenate([C_nue, C_nuebar, C_nux])


_GL2_GR2_NUE = G_L_NUE**2 + G_R_NUE**2
_GL2_mGR2_NUE = G_L_NUE**2 - G_R_NUE**2
_GL2_GR2_NUX = G_L_NUX**2 + G_R_NUX**2
_GL2_mGR2_NUX = G_L_NUX**2 - G_R_NUX**2


def _collision_jax_kernel_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
    temperature_frame_remap: bool = False,
    project_mangano_rate: bool = False,
) -> jnp.ndarray:
    """Pure-JAX nu-e + pair Stage-B closure on the species bank state.

    Calls ``rabbit.jax.collisions_jax.{nu_e_collision_jax, pair_collision_jax}``
    on each bank species at the plasma temperature ``T_gamma`` (the
    electron equilibrium temperature inside the operators) and converts
    the natural ``[MeV]`` rate to ``d/dN`` via ``H_MeV``.

    The bank carries three monopoles ``(f_nue, f_nuebar, f_nux)``;
    ``f_nuxbar`` is identified with ``f_nux`` for the no-asymmetry
    tier-3 limit, matching the existing bank conventions in this
    module.  Couplings: ``A_E`` (= ``G_L^2 + G_R^2`` for ν_e) on the
    ν_e / ν̄_e channels; ``A_X`` on ν_x.
    """
    n_q = int(q_nodes.shape[0])
    f_nue = bank_state[:n_q]
    f_nuebar = bank_state[n_q : 2 * n_q]
    f_nux = bank_state[2 * n_q :]
    if bool(temperature_frame_remap):
        q_nodes_nue_e = q_nodes * T_nu_e / jnp.maximum(T_gamma, 1.0e-30)
        q_nodes_nux_e = q_nodes * T_nu_x / jnp.maximum(T_gamma, 1.0e-30)
    else:
        q_nodes_nue_e = q_nodes
        q_nodes_nux_e = q_nodes

    # Quadrature grids — strictly matched to ``q_nodes`` for the
    # neutrino Laguerre grids (y3 in nu-e and y2 in pair) so that
    # detailed balance at ``f = f_FD`` collapses algebraically with no
    # interpolation noise.  Specifically:
    #   * y3 (outgoing nu in nu-e elastic) = laguerre(N_q) = q_nodes
    #   * y2 (electron in nu-e elastic)    = laguerre(N_q) = q_nodes
    #     → f_FD(y2) is computed directly; matching size N_q does not
    #       affect detailed balance for nu-e but keeps the bank-core
    #       internally consistent.
    #   * y2 (antineutrino in pair)        = laguerre(N_q) = q_nodes
    #     → PCHIP evaluation of ``f_nubar`` at ``y2_nodes`` is identity
    #       at the input nodes, restoring algebraic detailed balance.
    #   * pair y3 stays on Gauss-Legendre at fixed size 24 (it is on
    #     ``[0, y1+y2]``, not on q_nodes; FD electron there is computed
    #     directly so no neutrino interpolation enters).
    # The numpy laggauss/leggauss outputs are LRU-cached; converting to
    # jnp inline here avoids leaking JIT-traced DeviceArrays across
    # trace boundaries.
    _y3_nodes_np, _y3_weights_np = _collisions_laguerre_grid_np(int(n_q))
    _y2_nodes_np, _y2_weights_np = _collisions_laguerre_grid_np(int(n_q))
    _leg_nodes_np, _leg_weights_np = _collisions_legendre_grid_np(24)
    y3_nodes_jax = jnp.asarray(_y3_nodes_np)
    y3_weights_jax = jnp.asarray(_y3_weights_np)
    y2_nodes_jax = jnp.asarray(_y2_nodes_np)
    y2_weights_jax = jnp.asarray(_y2_weights_np)
    leg_nodes_jax = jnp.asarray(_leg_nodes_np)
    leg_weights_jax = jnp.asarray(_leg_weights_np)

    H_MeV = H_inv_sec / _MEV_TO_S
    H_floor = jnp.maximum(H_MeV, 1.0e-100)

    # nu-e elastic kernel for each species: electron always at T_gamma.
    # The species bank is stored on q_bank = p / T_nu, whereas the
    # Hannestad-Madsen kernels integrate over y = p / T_e.  Feed each
    # kernel the electron-frame nodes y_i = q_i T_nu/T_gamma while
    # keeping the same f_i values; interpolation inside the kernel then
    # evaluates f(y T_gamma/T_nu), i.e. the same physical distribution
    # on the electron-frame grid.
    C_scat_nue = _nu_e_collision_jax_kernel(
        f_nue,
        q_nodes_nue_e,
        T_gamma,
        y3_nodes=y3_nodes_jax,
        y3_weights=y3_weights_jax,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        GL2_GR2=_GL2_GR2_NUE,
        GL2_mGR2=_GL2_mGR2_NUE,
    )
    C_scat_nuebar = _nu_e_collision_jax_kernel(
        f_nuebar,
        q_nodes_nue_e,
        T_gamma,
        y3_nodes=y3_nodes_jax,
        y3_weights=y3_weights_jax,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        GL2_GR2=_GL2_GR2_NUE,
        GL2_mGR2=_GL2_mGR2_NUE,
    )
    C_scat_nux = _nu_e_collision_jax_kernel(
        f_nux,
        q_nodes_nux_e,
        T_gamma,
        y3_nodes=y3_nodes_jax,
        y3_weights=y3_weights_jax,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        GL2_GR2=_GL2_GR2_NUX,
        GL2_mGR2=_GL2_mGR2_NUX,
    )

    # Pair process: nue-nuebar pair at A_E, nux-nuxbar pair at A_X
    # (with f_nuxbar = f_nux in the no-asymmetry limit).
    C_pair_nue = _pair_collision_jax_kernel(
        f_nue,
        f_nuebar,
        q_nodes_nue_e,
        T_gamma,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        leg_nodes=leg_nodes_jax,
        leg_weights=leg_weights_jax,
        GL2_GR2=_GL2_GR2_NUE,
        GL2_mGR2=_GL2_mGR2_NUE,
    )
    C_pair_nuebar = _pair_collision_jax_kernel(
        f_nuebar,
        f_nue,
        q_nodes_nue_e,
        T_gamma,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        leg_nodes=leg_nodes_jax,
        leg_weights=leg_weights_jax,
        GL2_GR2=_GL2_GR2_NUE,
        GL2_mGR2=_GL2_mGR2_NUE,
    )
    C_pair_nux = _pair_collision_jax_kernel(
        f_nux,
        f_nux,  # f_nuxbar = f_nux in the no-asymmetry tier-3 limit
        q_nodes_nux_e,
        T_gamma,
        y2_nodes=y2_nodes_jax,
        y2_weights=y2_weights_jax,
        leg_nodes=leg_nodes_jax,
        leg_weights=leg_weights_jax,
        GL2_GR2=_GL2_GR2_NUX,
        GL2_mGR2=_GL2_mGR2_NUX,
    )

    C_nue = (C_scat_nue + C_pair_nue) / H_floor
    C_nuebar = (C_scat_nuebar + C_pair_nuebar) / H_floor
    C_nux = (C_scat_nux + C_pair_nux) / H_floor
    if bool(project_mangano_rate):
        raw_dQ_e = -(
            _compute_energy_exchange_rate_jax(C_nue, q_nodes, q_weights, T_nu_e)
            + _compute_energy_exchange_rate_jax(C_nuebar, q_nodes, q_weights, T_nu_e)
        )
        raw_dQ_x = -4.0 * _compute_energy_exchange_rate_jax(C_nux, q_nodes, q_weights, T_nu_x)
        target_dQ_e = 2.0 * energy_transfer_rate_nue_jax(T_gamma, T_nu_e) / H_floor
        target_dQ_x = 4.0 * energy_transfer_rate_nux_jax(T_gamma, T_nu_x) / H_floor
        scale_e = jnp.where(
            jnp.abs(raw_dQ_e) > 1.0e-100,
            target_dQ_e / raw_dQ_e,
            1.0,
        )
        scale_x = jnp.where(
            jnp.abs(raw_dQ_x) > 1.0e-100,
            target_dQ_x / raw_dQ_x,
            1.0,
        )
        scale_e = jnp.clip(scale_e, 0.0, 100.0)
        scale_x = jnp.clip(scale_x, 0.0, 100.0)
        C_nue = C_nue * scale_e
        C_nuebar = C_nuebar * scale_e
        C_nux = C_nux * scale_x
    # Suppress q_weights-related unused warning by the linter; the
    # weights enter via the cached y3/y2 grids inside the kernels.
    del q_weights, T_nu_e, T_nu_x
    return jnp.concatenate([C_nue, C_nuebar, C_nux])


def _collision_ap_unified_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
    enforce_rate: bool = True,
    energy_transfer_scale: float = 1.0,
) -> jnp.ndarray:
    """Canonical AP-form unification core (PR-T3B canonical #1 / #2).

    Combines the **anisotropy-stable** ``psi_target`` from
    ``spectral_relaxation_preflight`` (full Fermi-Dirac form
    accounting for the bank temperature shift, capturing
    shear-driven asymmetries) with the **grid-stable**
    ``c_psi = gamma_over_H * (psi_target - psi_actual)`` source
    structure from ``projected_physical_preflight``.

    When ``enforce_rate=True`` (default), additionally applies a
    **soft total-rate constraint**: scales ``f_eq * c_psi`` so the
    plasma-side energy exchange matches the canonical Mangano
    rate.  Differs from ``spectral_relaxation`` by using a softer
    clip (``[0.1, 5]`` vs ``[0, 100]``) and a 1% activation floor
    (``threshold = max(0.01 * |target_dQ|, 1e-80)``); this avoids
    the ill-conditioning that breaks ``spectral_relaxation`` at
    grids above ``(4, 6)`` while still recovering the
    σ-independence of the total ν-plasma energy exchange.  The scalar
    enforcement is sign-safe: a positive scale is applied only when the raw
    source energy moment already points in the canonical heating/cooling
    direction.

    Returns the rate-enforced ``f_eq * c_psi`` in ``df/dN`` units.
    """
    n_q = int(q_nodes.shape[0])
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    H_mev = H_inv_sec / _MEV_TO_S

    def _one_species(
        f_mono: jnp.ndarray,
        coupling: float,
        target_rate: jnp.ndarray,
        T_bank: jnp.ndarray,
    ) -> jnp.ndarray:
        f_clip = jnp.clip(f_mono, 0.0, 1.0)
        gamma_over_H = _collision_rate_per_efold_jax(
            T_gamma,
            T_bank,
            H_inv_sec,
            jnp.asarray(coupling, dtype=jnp.float64),
        )
        # Full-FD psi_target (anisotropy-aware): borrowed from
        # spectral_relaxation.  When T_bank deviates from T_gamma the
        # FD shape captures the entire shift, not a Sommerfeld
        # linearization.
        f_target = 1.0 / (
            jnp.exp(
                jnp.minimum(q_nodes * T_bank / jnp.maximum(T_gamma, 1.0e-30), 500.0)
            )
            + 1.0
        )
        psi_target = (f_target - f_eq) / jnp.maximum(f_eq, 1.0e-30)
        psi_actual = (f_clip - f_eq) / jnp.maximum(f_eq, 1.0e-30)
        if not enforce_rate:
            c_psi = gamma_over_H * (psi_target - psi_actual)
            return f_eq * c_psi
        # Decompose into source (drive toward target) and damping
        # (drive away from actual) so each can carry an independent
        # energy-conservation projection a la spectral_relaxation.
        source_raw = gamma_over_H * psi_target * f_eq
        damping_raw = -gamma_over_H * psi_actual * f_eq
        # Energy-conservation projection on damping (spectral's
        # mechanism): subtract from damping the part that has nonzero
        # plasma energy exchange, so the damping is internally
        # energy-neutral.  Uses an energy basis that has the
        # gamma_over_H * q-shape natural for the AP-form.
        energy_basis = gamma_over_H * q_nodes * jnp.maximum(1.0 - f_eq, 1.0e-30) * f_eq
        damping_plasma = _compute_energy_exchange_rate_jax(
            damping_raw, q_nodes, q_weights, T_bank
        )
        basis_plasma = _compute_energy_exchange_rate_jax(
            energy_basis, q_nodes, q_weights, T_bank
        )
        damping_raw = jnp.where(
            jnp.abs(basis_plasma) > 1.0e-80,
            damping_raw - (damping_plasma / basis_plasma) * energy_basis,
            damping_raw,
        )
        # Soft total-rate enforcement on the source term: scale
        # source_raw so its plasma energy moment matches the
        # canonical Mangano rate.  Uses softer clip + 1% floor than
        # spectral_relaxation to avoid the ill-conditioning that
        # breaks spectral above (4, 6).
        target_dQ_dN = -target_rate / jnp.maximum(H_mev, 1.0e-80)
        raw_dQ_dN = _compute_energy_exchange_rate_jax(
            source_raw, q_nodes, q_weights, T_bank
        )
        threshold = jnp.maximum(0.01 * jnp.abs(target_dQ_dN), 1.0e-80)
        scale = _rate_enforcement_scale_jax(
            raw_dQ_dN,
            target_dQ_dN,
            threshold,
            jnp.asarray(0.1, dtype=jnp.float64),
            jnp.asarray(5.0, dtype=jnp.float64),
        )
        return scale * source_raw + damping_raw

    f_nue = bank_state[:n_q]
    f_nuebar = bank_state[n_q : 2 * n_q]
    f_nux = bank_state[2 * n_q :]
    source_scale = jnp.asarray(energy_transfer_scale, dtype=jnp.float64)
    C_nue = _one_species(
        f_nue,
        A_TOTAL_NUE,
        source_scale * energy_transfer_rate_nue_jax(T_gamma, T_nu_e),
        T_nu_e,
    )
    C_nuebar = _one_species(
        f_nuebar,
        A_TOTAL_NUE,
        source_scale * energy_transfer_rate_nue_jax(T_gamma, T_nu_e),
        T_nu_e,
    )
    C_nux = _one_species(
        f_nux,
        A_TOTAL_NUX,
        source_scale * energy_transfer_rate_nux_jax(T_gamma, T_nu_x),
        T_nu_x,
    )
    return jnp.concatenate([C_nue, C_nuebar, C_nux])


def _remap_common_frame_to_bank_frame_jax(
    values_common: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    T_bank: jnp.ndarray,
    T_common: jnp.ndarray,
) -> jnp.ndarray:
    q_common = q_nodes * T_bank / jnp.maximum(T_common, 1.0e-30)
    return cubic_spline_eval_with_clamp(
        q_nodes,
        values_common,
        q_common,
        fill_low=values_common[0],
        fill_high=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _remap_bank_frame_to_common_frame_jax(
    f_bank: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    T_bank: jnp.ndarray,
    T_common: jnp.ndarray,
) -> jnp.ndarray:
    q_bank = q_nodes * T_common / jnp.maximum(T_bank, 1.0e-30)
    return jnp.clip(
        cubic_spline_eval_with_clamp(
            q_nodes,
            f_bank,
            q_bank,
            fill_low=f_bank[0],
            fill_high=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        0.0,
        1.0,
    )


def _collision_nu_nu_spectral_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
) -> jnp.ndarray:
    """Diagonal cross-flavour ν-ν spectral exchange on the bank monopoles.

    This is a classical Boltzmann self-scattering correction, not a QKE
    flavour-mixing term.  It exchanges energy between the electron-neutrino
    pair and the four heavy-flavour states while projecting the total
    neutrino-sector energy moment to zero.
    """
    n_q = int(q_nodes.shape[0])
    f_nue = jnp.clip(bank_state[:n_q], 0.0, 1.0)
    f_nuebar = jnp.clip(bank_state[n_q : 2 * n_q], 0.0, 1.0)
    f_nux = jnp.clip(bank_state[2 * n_q :], 0.0, 1.0)

    T_common = jnp.maximum((T_nu_e**4 + 2.0 * T_nu_x**4) / 3.0, 0.0) ** 0.25
    H_floor = jnp.maximum(H_inv_sec / _MEV_TO_S, 1.0e-100)

    f_nue_c = _remap_bank_frame_to_common_frame_jax(
        f_nue, q_nodes=q_nodes, T_bank=T_nu_e, T_common=T_common
    )
    f_nuebar_c = _remap_bank_frame_to_common_frame_jax(
        f_nuebar, q_nodes=q_nodes, T_bank=T_nu_e, T_common=T_common
    )
    f_nux_c = _remap_bank_frame_to_common_frame_jax(
        f_nux, q_nodes=q_nodes, T_bank=T_nu_x, T_common=T_common
    )

    # The kernel is evaluated in the common neutrino temperature frame.
    # Degeneracy factors:
    #   each electron species scatters from four heavy states;
    #   each heavy state scatters from ν_e and ν̄_e.
    # ``nu_nu_diagonal_collision_tensorized_jax`` uses the same structural
    # detailed-balance convention as the preflight kernel. Its sign is opposite to the
    # forward-time relaxation convention used by the full-Boltzmann driver, so
    # the runtime coupling flips the sign before projection.
    C_nue_c = -4.0 * nu_nu_diagonal_collision_tensorized_jax(
        f_nue_c,
        f_nux_c,
        q_nodes,
        T_common,
        epsilon_alpha_beta=1.0,
        matrix_coeff=1.0,
    ) / H_floor
    C_nuebar_c = -4.0 * nu_nu_diagonal_collision_tensorized_jax(
        f_nuebar_c,
        f_nux_c,
        q_nodes,
        T_common,
        epsilon_alpha_beta=1.0,
        matrix_coeff=1.0,
    ) / H_floor
    C_nux_c = -(
        nu_nu_diagonal_collision_tensorized_jax(
            f_nux_c,
            f_nue_c,
            q_nodes,
            T_common,
            epsilon_alpha_beta=1.0,
            matrix_coeff=1.0,
        )
        + nu_nu_diagonal_collision_tensorized_jax(
            f_nux_c,
            f_nuebar_c,
            q_nodes,
            T_common,
            epsilon_alpha_beta=1.0,
            matrix_coeff=1.0,
        )
    ) / H_floor

    C_nue = _remap_common_frame_to_bank_frame_jax(
        C_nue_c, q_nodes=q_nodes, T_bank=T_nu_e, T_common=T_common
    )
    C_nuebar = _remap_common_frame_to_bank_frame_jax(
        C_nuebar_c, q_nodes=q_nodes, T_bank=T_nu_e, T_common=T_common
    )
    C_nux = _remap_common_frame_to_bank_frame_jax(
        C_nux_c, q_nodes=q_nodes, T_bank=T_nu_x, T_common=T_common
    )

    dQ_nue_pair_N = -(
        _compute_energy_exchange_rate_jax(C_nue, q_nodes, q_weights, T_nu_e)
        + _compute_energy_exchange_rate_jax(C_nuebar, q_nodes, q_weights, T_nu_e)
    )
    dQ_nux_bank_N = -4.0 * _compute_energy_exchange_rate_jax(
        C_nux, q_nodes, q_weights, T_nu_x
    )
    total_nu_dQ_N = dQ_nue_pair_N + dQ_nux_bank_N
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    energy_basis = q_nodes * f_eq * jnp.maximum(1.0 - f_eq, 1.0e-30)
    basis_moment = _compute_energy_exchange_rate_jax(
        energy_basis, q_nodes, q_weights, T_nu_x
    )
    C_nux = jnp.where(
        jnp.abs(basis_moment) > 1.0e-100,
        C_nux + (total_nu_dQ_N / 4.0 / basis_moment) * energy_basis,
        C_nux,
    )
    projected_dQ_nue_pair_N = -(
        _compute_energy_exchange_rate_jax(C_nue, q_nodes, q_weights, T_nu_e)
        + _compute_energy_exchange_rate_jax(C_nuebar, q_nodes, q_weights, T_nu_e)
    )
    target_dQ_nue_pair_N, _target_dQ_nux_bank_N = nu_nu_temperature_equilibration_sources_jax(
        T_nu_e,
        T_nu_x,
        H_inv_sec / _MEV_TO_S,
    )
    moment_scale = jnp.where(
        (jnp.abs(projected_dQ_nue_pair_N) > 1.0e-100)
        & (projected_dQ_nue_pair_N * target_dQ_nue_pair_N > 0.0),
        target_dQ_nue_pair_N / projected_dQ_nue_pair_N,
        0.0,
    )
    moment_scale = jnp.clip(moment_scale, 0.0, 10.0)
    C_nue = moment_scale * C_nue
    C_nuebar = moment_scale * C_nuebar
    C_nux = moment_scale * C_nux

    C_nue = C_nue + _nu_nu_self_thermalization_shape_jax(
        f_nue,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_bank=T_nu_e,
        H_MeV=H_floor,
    )
    C_nuebar = C_nuebar + _nu_nu_self_thermalization_shape_jax(
        f_nuebar,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_bank=T_nu_e,
        H_MeV=H_floor,
    )
    C_nux = C_nux + _nu_nu_self_thermalization_shape_jax(
        f_nux,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_bank=T_nu_x,
        H_MeV=H_floor,
    )
    equal_temperature = jnp.abs(T_nu_e - T_nu_x) <= (
        1.0e-14 * jnp.maximum(jnp.maximum(jnp.abs(T_nu_e), jnp.abs(T_nu_x)), 1.0)
    )
    equal_bank = jnp.maximum(
        jnp.max(jnp.abs(f_nue - f_nux)),
        jnp.max(jnp.abs(f_nuebar - f_nux)),
    ) <= 1.0e-12
    fd_bank = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)
    equal_fd = jnp.max(jnp.abs(f_nux - fd_bank)) <= 1.0e-12
    inactive = equal_temperature & equal_bank & equal_fd
    C_nue = jnp.where(inactive, jnp.zeros_like(C_nue), C_nue)
    C_nuebar = jnp.where(inactive, jnp.zeros_like(C_nuebar), C_nuebar)
    C_nux = jnp.where(inactive, jnp.zeros_like(C_nux), C_nux)
    return jnp.concatenate([C_nue, C_nuebar, C_nux])


def _collision_bank_core_jax(
    bank_state: jnp.ndarray,
    *,
    collision_mode: str,
    collision_kernel_backend: str = "jax",
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
    closure_strength: jnp.ndarray,
) -> jnp.ndarray:
    mode = str(collision_mode).strip().lower()
    if mode == "spectral_relaxation_preflight":
        return _collision_preflight_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_strength=closure_strength,
        )
    if mode == "projected_physical_preflight":
        return _collision_projected_physical_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if mode == "ap_unified_preflight":
        return _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if mode == _ELL2_KERNEL_MODE:
        # BD631 FB. The MONOPOLE core is identical to ap_unified_preflight;
        # the resolved ℓ=2 quadrupole term is a separate, additive low-rank
        # update wired in the RHS (near the monopole scatter) and the
        # factorized Jacobian (block-diag A2), gated by
        # _mode_enables_ell2_kernel. It never enters this monopole bank core
        # nor the 3T energy exchange (ℓ=2 is traceless; §C.iii).
        return _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if mode == "ap_preconditioned_canonical":
        # Phase δ-3 (Plan §2.5). The RHS is identical to
        # ap_unified_preflight; the difference is that the Jacobian
        # construction path applies the
        # rabbit.jax.collision_ap_preconditioner_jax diagonal
        # augmentation (Option E modified-equation form). The
        # Jacobian wire-up (next phase) is detected via
        # _mode_enables_ap_preconditioner so the preconditioner-only
        # path can be added without further dispatching here.
        return _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if mode == "ap_unified_nu_nu_preflight":
        return _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if mode == "ap_unified_nu_nu_spectral_preflight":
        ap_core = _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
        nunu_core = _collision_nu_nu_spectral_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
        return ap_core + nunu_core
    if mode == "ap_unified_nu_nu_spectral_accuracy_preflight":
        ap_core = _collision_ap_unified_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            energy_transfer_scale=_AP_ACCURACY_ENERGY_TRANSFER_SCALE,
        )
        nunu_core = _collision_nu_nu_spectral_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
        return ap_core + nunu_core
    if mode == "jax_kernel_preflight":
        return _collision_jax_kernel_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
        )
    if _mode_uses_jax_kernel_remap_rhs(mode):
        return _collision_jax_kernel_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            temperature_frame_remap=True,
            project_mangano_rate=False,
        )
    if mode == "jax_kernel_projected_preflight":
        return _collision_jax_kernel_bank_core_jax(
            bank_state,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            temperature_frame_remap=True,
            project_mangano_rate=True,
        )
    return jnp.zeros((3 * int(q_nodes.shape[0]),), dtype=jnp.float64)


def _collision_bank_energy_exchange_jax(
    collision_bank_rhs: jnp.ndarray,
    *,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_q = int(q_nodes.shape[0])
    C_nue = collision_bank_rhs[:n_q]
    C_nuebar = collision_bank_rhs[n_q : 2 * n_q]
    C_nux = collision_bank_rhs[2 * n_q :]
    dQ_nue_pair_N = -(
        _compute_energy_exchange_rate_jax(C_nue, q_nodes, q_weights, T_nu_e)
        + _compute_energy_exchange_rate_jax(C_nuebar, q_nodes, q_weights, T_nu_e)
    )
    dQ_nux_bank_N = -4.0 * _compute_energy_exchange_rate_jax(C_nux, q_nodes, q_weights, T_nu_x)
    return dQ_nue_pair_N, dQ_nux_bank_N


def _mode_uses_jax_kernel_remap_rhs(collision_mode: str) -> bool:
    return str(collision_mode).strip().lower() in {
        "jax_kernel_remap_preflight",
        "jax_kernel_remap_preconditioned_preflight",
    }


def _mode_enables_kernel_jacobian_preconditioner(collision_mode: str) -> bool:
    return str(collision_mode).strip().lower() == "jax_kernel_remap_preconditioned_preflight"


def _mode_enables_ap_preconditioner(collision_mode: str) -> bool:
    """Phase δ-3 predicate. The Jacobian-build path inspects this to
    decide whether to call ``rabbit.jax.collision_ap_preconditioner_jax``
    for the bank-diagonal augmentation. The RHS dispatch above already
    routes ``ap_preconditioned_canonical`` through the same kernel as
    ``ap_unified_preflight``; the Jacobian-only difference is what
    closes the Mangano gap in Phase δ-4.
    """
    return str(collision_mode).strip().lower() == "ap_preconditioned_canonical"


def _kernel_preconditioner_bank_diag_jax(
    *,
    q_nodes: jnp.ndarray,
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_inv_sec: jnp.ndarray,
) -> jnp.ndarray:
    """Negative diagonal bank-Jacobian stabilizer for remapped kernels.

    This is a modified-Jacobian preconditioner only: the RHS value is
    unchanged.  It tells Rodas5P about the fast collision relaxation manifold
    that the raw remapped Hannestad-Madsen bank kernel otherwise exposes only
    through a poorly conditioned dense derivative.
    """
    n_q = int(q_nodes.shape[0])
    gamma_e = _collision_rate_per_efold_jax(
        T_gamma,
        T_nu_e,
        H_inv_sec,
        jnp.asarray(A_TOTAL_NUE, dtype=jnp.float64),
    )
    gamma_x = _collision_rate_per_efold_jax(
        T_gamma,
        T_nu_x,
        H_inv_sec,
        jnp.asarray(A_TOTAL_NUX, dtype=jnp.float64),
    )
    q_shape = jnp.clip(q_nodes / _Q_THERMAL_MEAN, 0.25, 8.0)
    diag_e = -jnp.clip(gamma_e * q_shape, 0.0, 1.0e8)
    diag_x = -jnp.clip(gamma_x * q_shape, 0.0, 1.0e8)
    return jnp.concatenate([diag_e, diag_e, diag_x])


def _build_collisionless_factorized_jac_fn(
    rhs_fn: Callable,
    *,
    phase: int,
    correction_level: int,
    collision_mode: str,
    collision_kernel_backend: str,
    collision_preflight_relaxation: float,
    tau_n: float,
    eta: float,
    N_eff: float,
    f_nu: float,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    layout: dict,
    n_network_species: int,
    energy_weight: jnp.ndarray,
    rate_table,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
    q_forward_op: jnp.ndarray,
    q_backward_op: jnp.ndarray,
    ell2_tables: Optional[Ell2KernelTables] = None,
) -> Callable:
    enable_collision_preflight = str(collision_mode).strip().lower() != "collisionless"
    enable_ell2 = _mode_enables_ell2_kernel(collision_mode)
    active_idx = _transport_active_indices(layout, n_network_species=n_network_species)
    n_active = int(active_idx.shape[0])
    state_dim = int(layout["n_total"])

    active_tangents = jnp.zeros((n_active, state_dim), dtype=jnp.float64)
    active_tangents = active_tangents.at[jnp.arange(n_active), active_idx].set(1.0)
    active_row_selector = jnp.zeros((state_dim, n_active), dtype=jnp.float64)
    active_row_selector = active_row_selector.at[active_idx, jnp.arange(n_active)].set(1.0)
    eye_q = jnp.eye(int(layout["N_q"]), dtype=jnp.float64)
    collision_scatter = _build_collision_bank_scatter(layout)

    def _build_transport_moment_gather(
        *,
        mu: jnp.ndarray,
        jacobian_weights: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        energy_eq = jnp.tensordot(
            _cached_equilibrium_distribution(int(layout["N_q"])),
            energy_weight,
            axes=((0,), (0,)),
        )
        thermo_tier = int(layout.get("thermo_tier", 1))
        T_gamma = y[layout["i_tg"]]
        if thermo_tier >= 2:
            T_nu_e_weight = y[layout["i_tne"]]
            T_nu_x_weight = y[layout["i_tnx"]]
        else:
            T_nu_e_weight = tier1_T_nu_from_T_gamma_jax(T_gamma)
            T_nu_x_weight = T_nu_e_weight
        species_weights = _species_energy_weights_jax(
            jnp.asarray(f_nu, dtype=jnp.float64),
            T_nu_e_weight,
            T_nu_x_weight,
        )
        if bool(enable_collision_preflight) and thermo_tier >= 2:
            gather = jnp.zeros((1 + 3 * int(layout["N_q"]), state_dim), dtype=jnp.float64)
        else:
            gather = jnp.zeros((1 + 2 * int(layout["N_q"]), state_dim), dtype=jnp.float64)
        n_mu = int(layout["N_mu"])
        n_q = int(layout["N_q"])
        species_stride = n_mu * n_q
        i_transport = int(layout["i_transport"])
        monopole_weights = 0.5 * w0 * jacobian_weights
        n_species = int(layout["n_dist_species"])
        stress_ray_weights = (
            species_weights[:, None]
            * w0[None, :]
            * jacobian_weights[None, :]
            * P2_jax(mu)[None, :]
            / jnp.maximum(energy_eq, 1e-100)
        )
        stress_block = (stress_ray_weights[:, :, None] * energy_weight[None, None, :]).reshape(
            n_species * species_stride
        )
        gather = gather.at[0, i_transport : i_transport + n_species * species_stride].set(
            stress_block
        )

        if bool(enable_collision_preflight) and thermo_tier >= 2:
            collision_gather = _build_collision_bank_gather(
                jacobian_weights=jacobian_weights,
                w0=w0,
                layout=layout,
            )
            gather = gather.at[1:, :].set(collision_gather)
        else:
            mono_block = (monopole_weights[None, :, None] * eye_q[:, None, :]).reshape(
                n_q, species_stride
            )
            gather = gather.at[1 : 1 + n_q, i_transport : i_transport + species_stride].set(
                mono_block
            )
            gather = gather.at[
                1 + n_q :, i_transport + species_stride : i_transport + 2 * species_stride
            ].set(mono_block)
        return gather

    def _extract_transport_moments(
        *,
        mu: jnp.ndarray,
        jacobian_weights: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gather = _build_transport_moment_gather(
            mu=mu,
            jacobian_weights=jacobian_weights,
            y=y,
        )
        return gather, gather @ y

    def _active_transport_response_from_moments(
        moments: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        Sigma_plus = y[layout["i_Sp"]]
        Sigma_minus = y[layout["i_Sm"]]
        T_gamma = y[layout["i_tg"]]
        thermo_tier = int(layout.get("thermo_tier", 1))
        if thermo_tier >= 2:
            T_nu_e = y[layout["i_tne"]]
            T_nu_x = y[layout["i_tnx"]]
            T_nu_weak = T_nu_e
        else:
            T_nu_e = tier1_T_nu_from_T_gamma_jax(T_gamma)
            T_nu_x = T_nu_e
            T_nu_weak = T_nu_e
        Sigma_sq = Sigma_plus**2 + Sigma_minus**2
        if thermo_tier >= 2:
            H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq)
        else:
            H_MeV = tier1_hubble_aniso_jax(T_gamma, T_nu_e, jnp.asarray(N_eff), Sigma_sq)
        H_inv_s = H_MeV * _MEV_TO_S

        if bool(enable_collision_preflight) and thermo_tier >= 2:
            bank_state = jnp.clip(moments[1:], 0.0, 1.0)
            f_nue_monopole = bank_state[: int(layout["N_q"])]
            f_nuebar_monopole = bank_state[int(layout["N_q"]) : 2 * int(layout["N_q"])]
        else:
            bank_state = None
            n_q = int(layout["N_q"])
            f_nue_monopole = jnp.clip(moments[1 : 1 + n_q], 0.0, 1.0)
            f_nuebar_monopole = jnp.clip(moments[1 + n_q : 1 + 2 * n_q], 0.0, 1.0)
        live_kernel = _separate_nue_nuebar_live_kernel(int(correction_level))
        lambda_np, lambda_pn, _I0 = live_kernel(
            T_gamma,
            T_nu_weak,
            jnp.asarray(tau_n),
            q_nodes,
            f_nue_monopole,
            f_nuebar_monopole,
        )

        active_rhs = jnp.zeros((n_active,), dtype=jnp.float64)
        active_rhs = active_rhs.at[0].set(moments[0])
        scalar_offset = 4

        if thermo_tier >= 2:
            collision_bank_rhs = _collision_bank_core_jax(
                bank_state,
                collision_mode=collision_mode,
                collision_kernel_backend=collision_kernel_backend,
                q_nodes=q_nodes,
                q_weights=q_weights,
                T_gamma=T_gamma,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
                H_inv_sec=H_inv_s,
                closure_strength=jnp.asarray(collision_preflight_relaxation, dtype=jnp.float64),
            ) if bool(enable_collision_preflight) else jnp.zeros((3 * int(layout["N_q"]),), dtype=jnp.float64)
            dQ_nue_pair_N, dQ_nux_bank_N = _collision_bank_energy_exchange_jax(
                collision_bank_rhs,
                q_nodes=q_nodes,
                q_weights=q_weights,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
            )
            if _mode_enables_nu_nu_equilibration(collision_mode):
                dQ_nue_nunu_N, dQ_nux_nunu_N = nu_nu_temperature_equilibration_sources_jax(
                    T_nu_e,
                    T_nu_x,
                    H_MeV,
                )
                dQ_nue_pair_N = dQ_nue_pair_N + dQ_nue_nunu_N
                dQ_nux_bank_N = dQ_nux_bank_N + dQ_nux_nunu_N
            dT_gamma, dT_nu_e, dT_nu_x = collision_moment_3T_source_terms_jax(
                T_gamma,
                T_nu_e,
                T_nu_x,
                dQ_nue_pair_N=dQ_nue_pair_N,
                dQ_nux_bank_N=dQ_nux_bank_N,
            )
            active_rhs = active_rhs.at[3].set(dT_gamma)
            active_rhs = active_rhs.at[4].set(dT_nu_e)
            active_rhs = active_rhs.at[5].set(dT_nu_x)
            scalar_offset = 6

        if int(phase) == 1:
            Xn = y[layout["i_net"]]
            dXn = abundance_rhs_phase1_jax(Xn, lambda_np, lambda_pn) / jnp.maximum(H_inv_s, 1e-100)
            dX = jnp.stack([dXn, -dXn])
        else:
            X = jax.lax.dynamic_slice(y, (layout["i_net"],), (int(n_network_species),))
            dX = abundance_rhs_phase2_jax(X, T_gamma, eta, lambda_np, lambda_pn, rate_table) / jnp.maximum(
                H_inv_s, 1e-100
            )
        active_rhs = active_rhs.at[scalar_offset : scalar_offset + int(n_network_species)].set(dX)
        return active_rhs

    @jax.jit
    def jac_fn(N, y):
        base, mu, _coeffs = _build_transport_transport_block(
            y[layout["i_Sp"]],
            y[layout["i_S"]],
            mu0=mu0,
            x0=x0,
            signs=signs,
            q_forward_op=q_forward_op,
            q_backward_op=q_backward_op,
            layout=layout,
        )

        f_of_y = lambda y_: rhs_fn(N, y_)
        _, lin_f = jax.linearize(f_of_y, y)
        active_cols = jax.vmap(lin_f)(active_tangents)

        jacobian_weights = jacobian_jax(x0, y[layout["i_S"]], mu)
        right_factor, transport_moments = _extract_transport_moments(
            mu=mu,
            jacobian_weights=jacobian_weights,
            y=y,
        )
        core_matrix = jax.jacfwd(
            lambda moments: _active_transport_response_from_moments(moments, y)
        )(transport_moments)

        J = base
        J = J.at[:, active_idx].set(active_cols.T)
        left_factor = active_row_selector
        right_factor = right_factor
        combined_core = core_matrix

        if bool(enable_collision_preflight):
            T_gamma = y[layout["i_tg"]]
            thermo_tier = int(layout.get("thermo_tier", 1))
            if thermo_tier >= 2:
                T_nu_e = y[layout["i_tne"]]
                T_nu_x = y[layout["i_tnx"]]
                H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, y[layout["i_Sp"]] ** 2 + y[layout["i_Sm"]] ** 2)
            else:
                T_nu_e = tier1_T_nu_from_T_gamma_jax(T_gamma)
                T_nu_x = T_nu_e
                H_MeV = tier1_hubble_aniso_jax(
                    T_gamma,
                    T_nu_e,
                    jnp.asarray(N_eff),
                    y[layout["i_Sp"]] ** 2 + y[layout["i_Sm"]] ** 2,
                )
            Sigma_sq = y[layout["i_Sp"]] ** 2 + y[layout["i_Sm"]] ** 2
            H_inv_s = H_MeV * _MEV_TO_S
            collision_gather = _build_collision_bank_gather(
                jacobian_weights=jacobian_weights,
                w0=w0,
                layout=layout,
            )
            collision_bank_state = collision_gather @ y
            collision_core = jax.jacfwd(
                lambda bank_state: _collision_bank_core_jax(
                    bank_state,
                    collision_mode=collision_mode,
                    collision_kernel_backend=collision_kernel_backend,
                    q_nodes=q_nodes,
                    q_weights=q_weights,
                    T_gamma=T_gamma,
                    T_nu_e=T_nu_e,
                    T_nu_x=T_nu_x,
                    H_inv_sec=H_inv_s,
                    closure_strength=jnp.asarray(collision_preflight_relaxation, dtype=jnp.float64),
                )
            )(collision_bank_state)
            if _mode_enables_kernel_jacobian_preconditioner(collision_mode):
                precond_diag = _kernel_preconditioner_bank_diag_jax(
                    q_nodes=q_nodes,
                    T_gamma=T_gamma,
                    T_nu_e=T_nu_e,
                    T_nu_x=T_nu_x,
                    H_inv_sec=H_inv_s,
                )
                derivative_scale = jnp.sum(jnp.abs(collision_core), axis=1)
                precond_diag = -jnp.maximum(jnp.abs(precond_diag), derivative_scale)
                precond_diag = jnp.clip(precond_diag, -1.0e8, 0.0)
                collision_core = collision_core + jnp.diag(precond_diag)
            if _mode_enables_ap_preconditioner(collision_mode):
                # Phase δ-4 (Plan §2.5): Option E modified-equation
                # preconditioner. The bank diagonal is augmented with
                # -Γ_α(q,T)/H so the implicit step controller "sees" the
                # fast equilibration manifold without microstepping. RHS
                # remains unchanged (the dispatch above routes through the
                # same _collision_ap_unified_bank_core_jax), so the
                # forward physics is identical to ap_unified_preflight;
                # the Jacobian-only difference is what closes the
                # Mangano N_eff gap. The bank stores monopoles only
                # (no μ tile), so we pass N_mu=1 to the preconditioner
                # diagonal builder; species count is fixed at 3.
                from rabbit.jax.collision_ap_preconditioner_jax import (
                    compute_ap_preconditioner_diag,
                )
                ap_diag = compute_ap_preconditioner_diag(
                    T_gamma_MeV=T_gamma,
                    H_MeV=H_MeV,
                    q_nodes=q_nodes,
                    N_mu=1,
                    n_species=3,
                    kernel_backend=collision_kernel_backend,
                )
                # Inject onto the collision_core diagonal. The
                # collision_core is the bank-block Jacobian once L and
                # R are the gather/scatter projectors.
                ap_diag_flat = ap_diag.reshape(-1)
                # Defensive: shape mismatch raises rather than silently
                # skipping (the previous behaviour hid the layout bug
                # that this fix addressed).
                if ap_diag_flat.shape[0] != collision_core.shape[0]:
                    raise ValueError(
                        f"ap_preconditioned_canonical: ap_diag_flat size "
                        f"{ap_diag_flat.shape[0]} != collision_core dim "
                        f"{collision_core.shape[0]} (bank monopole layout?)"
                    )
                # Inject raw ap_diag (already negative-clipped at
                # [-1e8, 0] by compute_ap_preconditioner_diag). Unlike
                # the kernel-remap-preconditioner, the AP-unified RHS
                # already contains the relaxation term, so we do NOT
                # additionally scale to derivative_scale — that would
                # double-count the stiffness and destabilise the
                # integrator at low resolution.
                collision_core = collision_core + jnp.diag(ap_diag_flat)
            left_factor = jnp.concatenate([active_row_selector, collision_scatter], axis=1)
            right_factor = jnp.concatenate([right_factor, collision_gather], axis=0)
            combined_core = _block_diag_rect(core_matrix, collision_core)

            if enable_ell2:
                # BD631 FB (M2): the resolved ℓ=2 A2 block enters the IMPLICIT
                # factorized Jacobian as a second, independent low-rank update
                # (P₂ shape vs the monopole P₀). A2 is linear in f2, so
                # jacfwd of the twin returns the analytic
                # A2 = -2·M2_G(T_ν)/H (tail rows zeroed) exactly — this is the
                # true ℓ=2 relaxation eigen-rate the stiff step needs
                # (rate/e-fold up to 107 @T=10; RHS-only would overshoot).
                ell2_gather = _build_collision_bank_gather_ell2(
                    mu=mu, jacobian_weights=jacobian_weights, w0=w0, layout=layout
                )
                ell2_scatter = _build_collision_bank_scatter_ell2(
                    mu=mu, jacobian_weights=jacobian_weights, w0=w0, layout=layout
                )
                ell2_bank_state = ell2_gather @ y
                ell2_core = jax.jacfwd(
                    lambda f2_bank: _collision_bank_ell2_core_jax(
                        f2_bank,
                        T_nu_e=T_nu_e,
                        T_nu_x=T_nu_x,
                        H_MeV=H_MeV,
                        q_nodes=q_nodes,
                        ell2_tables=ell2_tables,
                    )
                )(ell2_bank_state)
                left_factor = jnp.concatenate([left_factor, ell2_scatter], axis=1)
                right_factor = jnp.concatenate([right_factor, ell2_gather], axis=0)
                combined_core = _block_diag_rect(combined_core, ell2_core)

        return LowRankJacobianFactors(
            base_jacobian=J,
            left_factor=left_factor,
            core_matrix=combined_core,
            right_factor=right_factor,
        )

    return jac_fn


def _rhs_core_full_boltzmann_collisionless(
    N,
    y,
    *,
    phase: int,
    correction_level: int,
    collision_mode: str,
    collision_kernel_backend: str,
    collision_preflight_relaxation: float,
    tau_n: float,
    eta: float,
    N_eff: float,
    f_nu: float,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
    q_nodes: jnp.ndarray,
    q_weights: jnp.ndarray,
    energy_weight: jnp.ndarray,
    q_forward_op: jnp.ndarray,
    q_backward_op: jnp.ndarray,
    layout: dict,
    n_network_species: int,
    rate_table,
    ell2_tables: Optional[Ell2KernelTables] = None,
):
    del N
    enable_collision_preflight = str(collision_mode).strip().lower() != "collisionless"
    enable_ell2 = _mode_enables_ell2_kernel(collision_mode)

    Sigma_plus = y[layout["i_Sp"]]
    Sigma_minus = y[layout["i_Sm"]]
    S_val = y[layout["i_S"]]
    T_gamma = y[layout["i_tg"]]

    transport = _unpack_transport(y, layout)
    mu = mu_current_jax(x0, signs, S_val)
    jac_weights = jacobian_jax(x0, S_val, mu)
    coeffs = Sigma_plus * P2_jax(mu)
    d_transport = apply_continuous_q_advection_jax(transport, coeffs, q_forward_op, q_backward_op)

    thermo_tier = int(layout.get("thermo_tier", 1))
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    if thermo_tier >= 2:
        T_nu_e = y[layout["i_tne"]]
        T_nu_x = y[layout["i_tnx"]]
        T_nu_weak = T_nu_e
        H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq)
        dT_gamma = jnp.zeros((), dtype=jnp.float64)
        dT_nu_e = jnp.zeros((), dtype=jnp.float64)
        dT_nu_x = jnp.zeros((), dtype=jnp.float64)
    else:
        T_nu_e = tier1_T_nu_from_T_gamma_jax(T_gamma)
        T_nu_x = T_nu_e
        T_nu_weak = T_nu_e
        H_MeV = tier1_hubble_aniso_jax(T_gamma, T_nu_e, jnp.asarray(N_eff), Sigma_sq)
        dT_gamma = tier1_dT_gamma_dN_jax(T_gamma)
    H_inv_s = H_MeV * _MEV_TO_S

    Pi_plus = _extract_stress_from_rays(
        transport,
        jac_weights,
        mu,
        w0,
        energy_weight,
        f_nu,
        T_nu_e,
        T_nu_x,
    )
    Pi_minus = jnp.zeros((), dtype=jnp.float64)
    dSp, dSm = typeI_geometry_rhs(Sigma_plus, Sigma_minus, Pi_plus, Pi_minus)
    dS = Sigma_plus

    if bool(enable_collision_preflight):
        collision_gather = _build_collision_bank_gather(
            jacobian_weights=jac_weights,
            w0=w0,
            layout=layout,
        )
        collision_scatter = _build_collision_bank_scatter(layout)
        collision_bank_state = collision_gather @ y
        collision_bank_rhs = _collision_bank_core_jax(
            collision_bank_state,
            collision_mode=collision_mode,
            collision_kernel_backend=collision_kernel_backend,
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_s,
            closure_strength=jnp.asarray(collision_preflight_relaxation, dtype=jnp.float64),
        )
        d_transport = d_transport + _unpack_transport(collision_scatter @ collision_bank_rhs, layout)
        if enable_ell2:
            # BD631 FB: resolved ℓ=2 (quadrupole) collision — a SEPARATE,
            # additive low-rank update alongside the monopole term (P₂ vs P₀
            # angular shapes). f2-DIRECT (M3): gather f2, apply -2·M2_G@f2/H,
            # scatter with P2_perp/N2. It NEVER enters the 3T energy exchange
            # below (ℓ=2 is traceless; §C.iii) — its ℓ=0 moment is 0 by M1.
            ell2_gather = _build_collision_bank_gather_ell2(
                mu=mu, jacobian_weights=jac_weights, w0=w0, layout=layout
            )
            ell2_scatter = _build_collision_bank_scatter_ell2(
                mu=mu, jacobian_weights=jac_weights, w0=w0, layout=layout
            )
            ell2_bank_f2 = ell2_gather @ y
            ell2_dN = _collision_bank_ell2_core_jax(
                ell2_bank_f2,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
                H_MeV=H_MeV,
                q_nodes=q_nodes,
                ell2_tables=ell2_tables,
            )
            d_transport = d_transport + _unpack_transport(ell2_scatter @ ell2_dN, layout)
        if thermo_tier >= 2:
            dQ_nue_pair_N, dQ_nux_bank_N = _collision_bank_energy_exchange_jax(
                collision_bank_rhs,
                q_nodes=q_nodes,
                q_weights=q_weights,
                T_nu_e=T_nu_e,
                T_nu_x=T_nu_x,
            )
            if _mode_enables_nu_nu_equilibration(collision_mode):
                dQ_nue_nunu_N, dQ_nux_nunu_N = nu_nu_temperature_equilibration_sources_jax(
                    T_nu_e,
                    T_nu_x,
                    H_MeV,
                )
                dQ_nue_pair_N = dQ_nue_pair_N + dQ_nue_nunu_N
                dQ_nux_bank_N = dQ_nux_bank_N + dQ_nux_nunu_N
            dT_gamma, dT_nu_e, dT_nu_x = coupled_3T_rhs_from_collision_moments_jax(
                T_gamma,
                T_nu_e,
                T_nu_x,
                dQ_nue_pair_N=dQ_nue_pair_N,
                dQ_nux_bank_N=dQ_nux_bank_N,
            )

    f_nue_monopole = jnp.clip(
        _extract_monopole_from_rays(transport[0], jac_weights, w0), 0.0, 1.0
    )
    f_nuebar_monopole = jnp.clip(
        _extract_monopole_from_rays(transport[1], jac_weights, w0), 0.0, 1.0
    )
    live_kernel = _separate_nue_nuebar_live_kernel(int(correction_level))
    lambda_np, lambda_pn, _I0 = live_kernel(
        T_gamma,
        T_nu_weak,
        jnp.asarray(tau_n),
        q_nodes,
        f_nue_monopole,
        f_nuebar_monopole,
    )

    if int(phase) == 1:
        Xn = y[layout["i_net"]]
        dXn = abundance_rhs_phase1_jax(Xn, lambda_np, lambda_pn) / jnp.maximum(H_inv_s, 1e-100)
        dX = jnp.stack([dXn, -dXn])
    else:
        X = jax.lax.dynamic_slice(y, (layout["i_net"],), (int(n_network_species),))
        dX = abundance_rhs_phase2_jax(X, T_gamma, eta, lambda_np, lambda_pn, rate_table) / jnp.maximum(
            H_inv_s, 1e-100
        )

    dy = jnp.zeros_like(y)
    dy = dy.at[layout["i_Sp"]].set(dSp)
    dy = dy.at[layout["i_Sm"]].set(dSm)
    dy = _pack_transport_slice(dy, layout, d_transport)
    dy = dy.at[layout["i_S"]].set(dS)
    dy = dy.at[layout["i_tg"]].set(dT_gamma)
    if thermo_tier >= 2:
        dy = dy.at[layout["i_tne"]].set(dT_nu_e)
        dy = dy.at[layout["i_tnx"]].set(dT_nu_x)
    dy = jax.lax.dynamic_update_slice(dy, dX, (layout["i_net"],))
    return dy


def _full_rhs_cache_key(
    *,
    phase: int,
    correction_level: int,
    collision_mode: str,
    collision_kernel_backend: str,
    collision_preflight_relaxation: float,
    thermo_tier: int,
    N_mu: int,
    N_q: int,
    n_network_species: int,
    n_reactions: int,
    tau_n: float,
    eta: float,
    N_eff: float,
    f_nu: float,
) -> tuple:
    return (
        int(phase),
        int(correction_level),
        str(collision_mode),
        str(collision_kernel_backend),
        float(collision_preflight_relaxation),
        int(thermo_tier),
        int(N_mu),
        int(N_q),
        int(n_network_species),
        int(n_reactions),
        float(tau_n),
        float(eta),
        float(N_eff),
        float(f_nu),
    )


def _get_full_boltzmann_rhs(
    *,
    phase: int,
    correction_level: int,
    collision_mode: str,
    collision_preflight_relaxation: float,
    thermo_tier: int,
    N_mu: int,
    N_q: int,
    n_network_species: int,
    n_reactions: int,
    tau_n: float,
    eta: float,
    N_eff: float,
    f_nu: float,
    collision_kernel_backend: str = "jax",
):
    key = _full_rhs_cache_key(
        phase=phase,
        correction_level=correction_level,
        collision_mode=collision_mode,
        collision_kernel_backend=collision_kernel_backend,
        collision_preflight_relaxation=collision_preflight_relaxation,
        thermo_tier=thermo_tier,
        N_mu=N_mu,
        N_q=N_q,
        n_network_species=n_network_species,
        n_reactions=n_reactions,
        tau_n=tau_n,
        eta=eta,
        N_eff=N_eff,
        f_nu=f_nu,
    )
    with _CACHE_LOCK:
        cached = _FULL_RHS_CACHE.get(key)
        if cached is not None:
            _FULL_RHS_CACHE.move_to_end(key)
            return cached

    mu0, w0, x0, signs = _cached_ray_grid(int(N_mu))
    q_nodes, q_weights = _cached_laguerre_grid(int(N_q))
    energy_weight = _cached_energy_weight(int(N_q))
    q_forward_op, q_backward_op, _q_centered_op = cached_q_advection_operators(int(N_q))
    layout = _full_layout(int(N_mu), int(N_q), int(n_network_species), thermo_tier=int(thermo_tier))
    rate_table = load_rate_table(int(n_reactions))
    mode = str(collision_mode).strip().lower()
    # BD631 FB: load the ν–e ℓ=2 kernel table ONCE at driver-build (host numpy,
    # outside any trace) → static jnp arrays closed over the jitted RHS/Jacobian.
    # None for every non-ℓ=2 mode (the ℓ=2 code path is gated + never touched).
    ell2_tables = _load_ell2_kernel_tables() if _mode_enables_ell2_kernel(mode) else None
    if mode == "direct_kernel_preflight":
        base_rhs_fn, base_jac_fn, _base_layout, _base_mu0, _base_w0, _base_x0, _base_signs, _base_q_nodes, _base_energy = _get_full_boltzmann_rhs(
            phase=int(phase),
            correction_level=int(correction_level),
            collision_mode="collisionless",
            collision_preflight_relaxation=1.0,
            thermo_tier=int(thermo_tier),
            N_mu=int(N_mu),
            N_q=int(N_q),
            n_network_species=int(n_network_species),
            n_reactions=int(n_reactions),
            tau_n=float(tau_n),
            eta=float(eta),
            N_eff=float(N_eff),
            f_nu=float(f_nu),
            collision_kernel_backend="jax",
        )
        rhs_fn = _build_direct_kernel_runtime_rhs(
            base_rhs_fn,
            layout=layout,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
        )
        jac_fn = _build_direct_kernel_runtime_jac_fn(
            base_jac_fn,
            layout=layout,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
        )
    else:
        rhs_fn = jax.jit(
            partial(
                _rhs_core_full_boltzmann_collisionless,
                phase=int(phase),
                correction_level=int(correction_level),
                collision_mode=str(collision_mode),
                collision_kernel_backend=str(collision_kernel_backend),
                collision_preflight_relaxation=float(collision_preflight_relaxation),
                tau_n=float(tau_n),
                eta=float(eta),
                N_eff=float(N_eff),
                f_nu=float(f_nu),
                mu0=mu0,
                w0=w0,
                x0=x0,
                signs=signs,
                q_nodes=q_nodes,
                q_weights=q_weights,
                energy_weight=energy_weight,
                q_forward_op=q_forward_op,
                q_backward_op=q_backward_op,
                layout=layout,
                n_network_species=int(n_network_species),
                rate_table=rate_table,
                ell2_tables=ell2_tables,
            )
        )
        jac_fn = _build_collisionless_factorized_jac_fn(
            rhs_fn,
            phase=int(phase),
            correction_level=int(correction_level),
            collision_mode=str(collision_mode),
            collision_kernel_backend=str(collision_kernel_backend),
            collision_preflight_relaxation=float(collision_preflight_relaxation),
            tau_n=float(tau_n),
            eta=float(eta),
            N_eff=float(N_eff),
            f_nu=float(f_nu),
            q_nodes=q_nodes,
            q_weights=q_weights,
            layout=layout,
            n_network_species=int(n_network_species),
            energy_weight=energy_weight,
            rate_table=rate_table,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
            q_forward_op=q_forward_op,
            q_backward_op=q_backward_op,
            ell2_tables=ell2_tables,
        )
    payload = (rhs_fn, jac_fn, layout, mu0, w0, x0, signs, q_nodes, energy_weight)
    with _CACHE_LOCK:
        _FULL_RHS_CACHE[key] = payload
        while len(_FULL_RHS_CACHE) > _FULL_RHS_CACHE_MAXSIZE:
            _FULL_RHS_CACHE.popitem(last=False)
    return payload


@lru_cache(maxsize=16)
def _build_event_fn(i_tg: int, threshold: float) -> Callable:
    idx = int(i_tg)
    thr = float(threshold)

    def event_fn(N, y):
        del N
        return y[idx] - thr

    return event_fn


@dataclass
class JAXFullBoltzmannConfig:
    Sigma_H_plus: float = 0.0
    Sigma_H_minus: float = 0.0
    T_start: float = 10.0
    T_handoff: float = 0.08
    T_end: float = 0.005
    tau_n: float = _TAU_N
    eta: float = _ETA
    N_eff: float = _N_EFF
    f_nu: float = _F_NU
    N_mu: int = 12
    N_q: int = 20
    n_reactions: int = 12
    correction_level: int = 0
    thermo_tier: int = 1
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 2000
    event_refine_steps: int = 24
    runtime_device_policy: str = "cpu_preferred"
    q_advection_scheme: str = "upwind"
    collision_mode: str = "collisionless"
    collision_kernel_backend: str = "jax"
    collision_preflight_relaxation: float = 1.0

    def __post_init__(self):
        collision_kernel_backend, collision_kernel_metadata = (
            _resolve_full_solver_collision_kernel_backend(self.collision_kernel_backend)
        )
        self.collision_kernel_backend = collision_kernel_backend
        self._collision_kernel_metadata = collision_kernel_metadata
        validate_phase_temperature_order(
            self.T_start,
            self.T_handoff,
            self.T_end,
            context="JAX full-Boltzmann temperature schedule",
        )
        validate_min_resolution("N_mu", self.N_mu, minimum=2, strict=True)
        validate_min_resolution("N_q", self.N_q, minimum=4, strict=True)
        validate_manual_network_truncation(self.n_reactions, context="JAX full-Boltzmann network")
        validate_typeI_initial_budget(
            self.Sigma_H_plus,
            self.Sigma_H_minus,
            context="JAX full-Boltzmann initial data",
        )
        if abs(float(self.Sigma_H_minus)) > 0.0:
            raise NotImplementedError("Collisionless full-Boltzmann driver is LRS-only for now (Σ_- = 0).")
        if int(self.correction_level) not in (0, 1, 2, 3):
            raise ValueError("correction_level must be 0, 1, 2, or 3.")
        if str(self.runtime_device_policy).strip().lower() != "cpu_preferred":
            raise NotImplementedError("PR-T3A full-Boltzmann driver currently supports only runtime_device_policy='cpu_preferred'.")
        if str(self.q_advection_scheme).strip().lower() != "upwind":
            raise ValueError("q_advection_scheme must be 'upwind' for the current landed PR-T3A surface.")
        mode = str(self.collision_mode).strip().lower()
        if mode not in {
            "collisionless",
            "spectral_relaxation_preflight",
            "projected_physical_preflight",
            "direct_kernel_preflight",
            "jax_kernel_preflight",
            "jax_kernel_remap_preflight",
            "jax_kernel_remap_preconditioned_preflight",
            "jax_kernel_projected_preflight",
            "ap_unified_preflight",
            "ap_unified_nu_nu_preflight",
            "ap_unified_nu_nu_spectral_preflight",
            "ap_unified_nu_nu_spectral_accuracy_preflight",
            "ap_preconditioned_canonical",  # Phase δ-3/δ-4
            "ap_unified_ell2_kernelized_preflight",  # BD631 FB
        }:
            raise ValueError(
                "collision_mode must be 'collisionless', 'spectral_relaxation_preflight', "
                "'projected_physical_preflight', 'direct_kernel_preflight', "
                "'jax_kernel_preflight', 'jax_kernel_remap_preflight', "
                "'jax_kernel_remap_preconditioned_preflight', "
                "'jax_kernel_projected_preflight', 'ap_unified_preflight', "
                "'ap_unified_nu_nu_preflight', "
                "'ap_unified_nu_nu_spectral_preflight', "
                "'ap_unified_nu_nu_spectral_accuracy_preflight', "
                "'ap_preconditioned_canonical', or "
                "'ap_unified_ell2_kernelized_preflight'."
            )
        if mode != "spectral_relaxation_preflight" and abs(float(self.collision_preflight_relaxation) - 1.0) > 0.0:
            # Keep the hidden preflight knob honest: it has no effect unless the
            # private collision-preflight mode is explicitly enabled.
            raise ValueError(
                "collision_preflight_relaxation is only meaningful when "
                "collision_mode='spectral_relaxation_preflight'."
            )
        if int(self.thermo_tier) not in (1, 2):
            raise ValueError("thermo_tier must be 1 or 2 for the current private full-Boltzmann surface.")
        if int(self.thermo_tier) == 2 and mode == "collisionless":
            raise NotImplementedError(
                "thermo_tier=2 is currently available only with "
                "a private collision preflight mode."
            )
        if mode == "direct_kernel_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("direct_kernel_preflight is currently supported only with thermo_tier=2.")
        if mode == "jax_kernel_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("jax_kernel_preflight is currently supported only with thermo_tier=2.")
        if mode == "jax_kernel_remap_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("jax_kernel_remap_preflight is currently supported only with thermo_tier=2.")
        if mode == "jax_kernel_remap_preconditioned_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("jax_kernel_remap_preconditioned_preflight is currently supported only with thermo_tier=2.")
        if mode == "jax_kernel_projected_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("jax_kernel_projected_preflight is currently supported only with thermo_tier=2.")
        if mode == "ap_unified_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("ap_unified_preflight is currently supported only with thermo_tier=2.")
        if mode == "ap_unified_nu_nu_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("ap_unified_nu_nu_preflight is currently supported only with thermo_tier=2.")
        if mode == "ap_unified_nu_nu_spectral_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("ap_unified_nu_nu_spectral_preflight is currently supported only with thermo_tier=2.")
        if mode == "ap_unified_nu_nu_spectral_accuracy_preflight" and int(self.thermo_tier) != 2:
            raise NotImplementedError("ap_unified_nu_nu_spectral_accuracy_preflight is currently supported only with thermo_tier=2.")
        if mode == "ap_unified_ell2_kernelized_preflight":
            # BD631 FB scope guards (F-029/F-031/F-034 + devil M1/M4).
            if int(self.thermo_tier) != 2:
                raise NotImplementedError(
                    "ap_unified_ell2_kernelized_preflight is currently supported only with thermo_tier=2."
                )
            if int(self.N_q) != 20:
                raise NotImplementedError(
                    "ap_unified_ell2_kernelized_preflight requires N_q=20 (the legacy v1 "
                    "electron-minus table grid; no q-interpolation of the 2-D kernel)."
                )
            sigma_h = abs(float(self.Sigma_H_plus))
            if sigma_h >= 1.0:
                raise ValueError(
                    "ap_unified_ell2_kernelized_preflight: Sigma_H>=1 exceeds the FB "
                    "angular-quadrature resolution limit (N2 collapse; devil M4). "
                    "The legacy single-T table was only tabulated over T∈[0.1,~10] MeV; "
                    "that tabulation range is not a physics-validation domain."
                )
            if sigma_h >= 0.3 and int(self.N_mu) < 16:
                raise ValueError(
                    "ap_unified_ell2_kernelized_preflight: Sigma_H>=0.3 requires N_mu>=16 for "
                    "ℓ=2 moment accuracy (devil M1: the P₂ moment quadrature, not conservation "
                    "— conservation is exact at any N_mu via the Gram-Schmidt P₂⟂P₀)."
                )


@dataclass
class JAXFullBoltzmannResult:
    Yp: float
    DH: float
    Li7H: float
    Li6H: float
    N_eff: float
    Xn_freeze: float
    success: bool
    n_steps_p1: int
    n_steps_p2: int
    metadata: dict


def _initial_transport_state(N_mu: int, N_q: int) -> np.ndarray:
    f_eq = np.asarray(_cached_equilibrium_distribution(int(N_q)))
    transport = np.tile(f_eq[None, None, :], (len(_DIST_SPECIES), int(N_mu), 1))
    return transport.astype(np.float64)


# Modes that route through the pure-JAX bank-core path
# (`_collision_bank_core_jax`).  All of them share the same low-rank
# Jacobian layout at a given thermo_tier; only the closure name differs.
_PURE_JAX_BANK_CORE_MODES = (
    "spectral_relaxation_preflight",
    "projected_physical_preflight",
    "jax_kernel_preflight",
    "jax_kernel_remap_preflight",
    "jax_kernel_remap_preconditioned_preflight",
    "jax_kernel_projected_preflight",
    "ap_unified_preflight",
    "ap_unified_nu_nu_preflight",
    "ap_unified_nu_nu_spectral_preflight",
    "ap_unified_nu_nu_spectral_accuracy_preflight",
    "ap_preconditioned_canonical",  # Phase δ-3/δ-4
    "ap_unified_ell2_kernelized_preflight",  # BD631 FB (monopole = ap_unified + additive ℓ=2)
)


def _mode_enables_nu_nu_equilibration(collision_mode: str) -> bool:
    return str(collision_mode).strip().lower() == "ap_unified_nu_nu_preflight"


def _mode_enables_nu_nu_spectral_bank(collision_mode: str) -> bool:
    return str(collision_mode).strip().lower() in {
        "ap_unified_nu_nu_spectral_preflight",
        "ap_unified_nu_nu_spectral_accuracy_preflight",
    }


def _ap_energy_transfer_scale_for_mode(collision_mode: str) -> float:
    if str(collision_mode).strip().lower() == "ap_unified_nu_nu_spectral_accuracy_preflight":
        return float(_AP_ACCURACY_ENERGY_TRANSFER_SCALE)
    return 1.0


def _resolve_jacobian_payload_contract(collision_mode: str, thermo_tier: int) -> str:
    if collision_mode == "collisionless":
        return "collisionless_active_separate_nue_nuebar_transport_moment_low_rank_v1"
    if collision_mode == "direct_kernel_preflight":
        return "direct_kernel_transport_plus_active_plus_3T_mixed_low_rank_v1"
    closure_prefix = {
        "spectral_relaxation_preflight": "collision_preflight",
        "projected_physical_preflight": "projected_physical",
        "jax_kernel_preflight": "jax_kernel",
        "jax_kernel_remap_preflight": "jax_kernel_remap",
        "jax_kernel_remap_preconditioned_preflight": "jax_kernel_remap_preconditioned",
        "jax_kernel_projected_preflight": "jax_kernel_projected",
        "ap_unified_preflight": "ap_unified",
        "ap_unified_nu_nu_preflight": "ap_unified_nu_nu",
        "ap_unified_nu_nu_spectral_preflight": "ap_unified_nu_nu_spectral",
        "ap_unified_nu_nu_spectral_accuracy_preflight": "ap_unified_nu_nu_spectral_accuracy",
        "ap_preconditioned_canonical": "ap_preconditioned_canonical",
        "ap_unified_ell2_kernelized_preflight": "ap_unified_plus_resolved_ell2_kernelized",
    }[collision_mode]
    if collision_mode == _ELL2_KERNEL_MODE:
        # BD631 FB: monopole (ap_unified) + additive resolved-ℓ=2 low-rank block.
        return f"{closure_prefix}_transport_plus_active_plus_3T_plus_ell2_low_rank_v1"
    if int(thermo_tier) == 1:
        return f"{closure_prefix}_transport_plus_active_low_rank_v1"
    return f"{closure_prefix}_transport_plus_active_plus_3T_low_rank_v1"


def _resolve_jacobian_transport_projector_contract(
    collision_mode: str, thermo_tier: int
) -> str:
    if collision_mode == "collisionless":
        return "stress_plus_separate_nue_nuebar_monopole_q_gather_v1"
    if collision_mode == "direct_kernel_preflight":
        return (
            "stress_plus_collision_bank_active_plus_transport_bank_gather_plus_direct_active_fd_v1"
        )
    if collision_mode == _ELL2_KERNEL_MODE:
        return "stress_plus_collision_bank_active_plus_transport_bank_gather_plus_ell2_gather_v1"
    if collision_mode in _PURE_JAX_BANK_CORE_MODES:
        if int(thermo_tier) == 1:
            return "stress_plus_monopole_q_plus_collision_bank_gather_v1"
        return "stress_plus_collision_bank_active_plus_transport_bank_gather_v1"
    raise ValueError(f"unknown collision_mode for projector contract: {collision_mode!r}")


def _resolve_jacobian_low_rank_moment_dim(
    collision_mode: str, thermo_tier: int, N_q: int
) -> int:
    if collision_mode == "collisionless":
        return 1 + 2 * int(N_q)
    if collision_mode == "direct_kernel_preflight":
        return 1 + 4 * int(N_q)
    if collision_mode == _ELL2_KERNEL_MODE:
        # tier2 monopole (1 + 6·N_q) + resolved ℓ=2 gather (3·N_q).
        return 1 + 6 * int(N_q) + 3 * int(N_q)
    if collision_mode in _PURE_JAX_BANK_CORE_MODES:
        if int(thermo_tier) == 1:
            return int(N_q) + 1 + 3 * int(N_q)
        return 1 + 6 * int(N_q)
    raise ValueError(f"unknown collision_mode for moment dim: {collision_mode!r}")


def _resolve_jacobian_low_rank_apply_dim(
    collision_mode: str, thermo_tier: int, N_q: int, n_species: int
) -> int:
    if collision_mode == "collisionless":
        return 4 + int(n_species)
    if collision_mode == "direct_kernel_preflight":
        return 9 + int(n_species) + 3 * int(N_q)
    if collision_mode == _ELL2_KERNEL_MODE:
        # tier2 monopole apply (6 + n_species + 3·N_q) + resolved ℓ=2 scatter (3·N_q).
        return 6 + int(n_species) + 3 * int(N_q) + 3 * int(N_q)
    if collision_mode in _PURE_JAX_BANK_CORE_MODES:
        if int(thermo_tier) == 1:
            return 4 + int(n_species) + 3 * int(N_q)
        return 6 + int(n_species) + 3 * int(N_q)
    raise ValueError(f"unknown collision_mode for apply dim: {collision_mode!r}")


def _build_collision_preflight_update_for_state(
    y: np.ndarray,
    *,
    layout: dict,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
    closure_mode: str = "spectral_relaxation",
    spectral_damping_relax: float = 1.0,
):
    """Embed the host-side Stage-B collision preflight into the full state layout."""
    from rabbit.jax.full_boltzmann_collision_preflight import (
        build_lifted_transport_collision_preflight,
    )

    y_arr = np.asarray(y, dtype=np.float64)
    transport = np.asarray(_unpack_transport(jnp.asarray(y_arr), layout), dtype=np.float64)
    S_val = float(y_arr[layout["i_S"]])
    mu = np.asarray(mu_current_jax(x0, signs, S_val), dtype=np.float64)
    jac_weights = np.asarray(jacobian_jax(x0, S_val, mu), dtype=np.float64)
    q_nodes, q_weights = _cached_laguerre_grid(int(layout["N_q"]))

    T_gamma = float(y_arr[layout["i_tg"]])
    if int(layout.get("thermo_tier", 1)) >= 2:
        T_nu_e = float(y_arr[layout["i_tne"]])
        T_nu_x = float(y_arr[layout["i_tnx"]])
    else:
        T_nu_e = float(tier1_T_nu_from_T_gamma_jax(jnp.asarray(T_gamma)))
        T_nu_x = T_nu_e
    Sigma_plus = float(y_arr[layout["i_Sp"]])
    Sigma_minus = float(y_arr[layout["i_Sm"]])
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    if int(layout.get("thermo_tier", 1)) >= 2:
        H_MeV = float(hubble_3T_jax(jnp.asarray(T_gamma), jnp.asarray(T_nu_e), jnp.asarray(T_nu_x), jnp.asarray(Sigma_sq)))
    else:
        H_MeV = float(
            tier1_hubble_aniso_jax(
                jnp.asarray(T_gamma),
                jnp.asarray(T_nu_e),
                jnp.asarray(_N_EFF),
                jnp.asarray(Sigma_sq),
            )
        )
    H_inv_sec = H_MeV * _MEV_TO_S

    return build_lifted_transport_collision_preflight(
        f_species_rays=transport,
        jacobian_weights=jac_weights,
        w0=np.asarray(w0, dtype=np.float64),
        q_nodes=np.asarray(q_nodes, dtype=np.float64),
        q_weights=np.asarray(q_weights, dtype=np.float64),
        T_gamma=T_gamma,
        T_nu_e=T_nu_e,
        T_nu_x=T_nu_x,
        H_inv_sec=H_inv_sec,
        transport_offset=int(layout["i_transport"]),
        state_dim=int(layout["n_total"]),
        closure_mode=closure_mode,
        spectral_damping_relax=spectral_damping_relax,
    )


def _build_direct_kernel_runtime_contribution_for_state(
    y: np.ndarray,
    *,
    layout: dict,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
    fd_eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-side direct-kernel runtime contribution on the full phase-state layout.

    Returns
    -------
    state_rhs_direct, left_factor, core_matrix, right_factor, active_cols_direct
        The explicit direct-kernel collision update, its low-rank passive transport
        block, and dense columns with respect to the active state coordinates.
    """
    from rabbit.jax.full_boltzmann_collision_preflight import (
        evaluate_collision_bank_and_3t_sources_from_state,
        materialize_collision_bank_and_3t_source_jacobian,
    )

    thermo_tier = int(layout.get("thermo_tier", 1))
    if thermo_tier < 2:
        raise NotImplementedError("direct-kernel runtime contribution is currently tier-2 only.")

    q_nodes, q_weights = _cached_laguerre_grid(int(layout["N_q"]))
    q_nodes_np = np.asarray(q_nodes, dtype=np.float64)
    q_weights_np = np.asarray(q_weights, dtype=np.float64)
    n_q = int(layout["N_q"])
    state_dim = int(layout["n_total"])

    def _compute_direct_components(y_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_local = np.asarray(y_state, dtype=np.float64).reshape(-1)
        lifted = _build_collision_preflight_update_for_state(
            y_local,
            layout=layout,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
            closure_mode="direct_kernel",
            spectral_damping_relax=1.0,
        )

        T_gamma = float(y_local[layout["i_tg"]])
        T_nu_e = float(y_local[layout["i_tne"]])
        T_nu_x = float(y_local[layout["i_tnx"]])
        Sigma_plus = float(y_local[layout["i_Sp"]])
        Sigma_minus = float(y_local[layout["i_Sm"]])
        Sigma_sq = Sigma_plus**2 + Sigma_minus**2
        H_inv_sec = float(
            hubble_3T_jax(
                jnp.asarray(T_gamma),
                jnp.asarray(T_nu_e),
                jnp.asarray(T_nu_x),
                jnp.asarray(Sigma_sq),
            )
        ) * _MEV_TO_S

        bank_state = np.asarray(lifted.gather_map @ y_local, dtype=np.float64)
        combined = evaluate_collision_bank_and_3t_sources_from_state(
            bank_state,
            q_nodes=q_nodes_np,
            q_weights=q_weights_np,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode="direct_kernel",
        )
        thermo_rhs = np.asarray(combined[3 * n_q :], dtype=np.float64)
        core_matrix = materialize_collision_bank_and_3t_source_jacobian(
            bank_state,
            q_nodes=q_nodes_np,
            q_weights=q_weights_np,
            T_gamma=T_gamma,
            T_nu_e=T_nu_e,
            T_nu_x=T_nu_x,
            H_inv_sec=H_inv_sec,
            closure_mode="direct_kernel",
        )

        state_rhs = np.asarray(lifted.state_rhs, dtype=np.float64).copy()
        state_rhs[layout["i_tg"]] = thermo_rhs[0]
        state_rhs[layout["i_tne"]] = thermo_rhs[1]
        state_rhs[layout["i_tnx"]] = thermo_rhs[2]

        left_factor = np.zeros((state_dim, 3 * n_q + 3), dtype=np.float64)
        left_factor[:, : 3 * n_q] = np.asarray(lifted.factors.left_factor, dtype=np.float64)
        left_factor[layout["i_tg"], 3 * n_q + 0] = 1.0
        left_factor[layout["i_tne"], 3 * n_q + 1] = 1.0
        left_factor[layout["i_tnx"], 3 * n_q + 2] = 1.0
        right_factor = np.asarray(lifted.factors.right_factor, dtype=np.float64)
        return state_rhs, left_factor, np.asarray(core_matrix, dtype=np.float64), right_factor

    n_network_species = int(layout["n_total"] - layout["i_net"])
    active_idx = np.asarray(_transport_active_indices(layout, n_network_species=n_network_species), dtype=np.int64)
    active_cols = np.zeros((state_dim, active_idx.size), dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    state_rhs, left_factor, core_matrix, right_factor = _compute_direct_components(y_arr)

    scalar_count = 6
    for active_slot, idx in enumerate(active_idx[:scalar_count]):
        step = float(fd_eps * max(1.0, abs(y_arr[idx])))
        y_hi = y_arr.copy()
        y_lo = y_arr.copy()
        y_hi[idx] += step
        y_lo[idx] -= step
        active_cols[:, active_slot] = (_compute_direct_components(y_hi)[0] - _compute_direct_components(y_lo)[0]) / (2.0 * step)

    return state_rhs, left_factor, core_matrix, right_factor, active_cols


def _direct_kernel_runtime_state_update(
    y: np.ndarray,
    *,
    layout: dict,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
) -> np.ndarray:
    return _build_direct_kernel_runtime_contribution_for_state(
        y,
        layout=layout,
        mu0=mu0,
        w0=w0,
        x0=x0,
        signs=signs,
    )[0]


def _build_direct_kernel_runtime_rhs(
    base_rhs_fn: Callable,
    *,
    layout: dict,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
) -> Callable:
    state_dim = int(layout["n_total"])
    out_spec = jax.ShapeDtypeStruct((state_dim,), jnp.float64)

    def _host_update(y_host):
        return _direct_kernel_runtime_state_update(
            y_host,
            layout=layout,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
        )

    @jax.jit
    def rhs_fn(N, y):
        base = base_rhs_fn(N, y)
        direct = jax.pure_callback(_host_update, out_spec, y)
        return base + direct

    return rhs_fn


def _build_direct_kernel_runtime_jac_fn(
    base_jac_fn: Callable,
    *,
    layout: dict,
    mu0: jnp.ndarray,
    w0: jnp.ndarray,
    x0: jnp.ndarray,
    signs: jnp.ndarray,
) -> Callable:
    state_dim = int(layout["n_total"])
    n_q = int(layout["N_q"])
    n_network_species = int(layout["n_total"] - layout["i_net"])
    active_idx = _transport_active_indices(layout, n_network_species=n_network_species)
    active_count = int(active_idx.shape[0])

    callback_spec = (
        jax.ShapeDtypeStruct((state_dim, 3 * n_q + 3), jnp.float64),
        jax.ShapeDtypeStruct((3 * n_q + 3, 3 * n_q), jnp.float64),
        jax.ShapeDtypeStruct((3 * n_q, state_dim), jnp.float64),
        jax.ShapeDtypeStruct((state_dim, active_count), jnp.float64),
    )

    def _host_factors(y_host):
        _state_rhs, left_factor, core_matrix, right_factor, active_cols = _build_direct_kernel_runtime_contribution_for_state(
            y_host,
            layout=layout,
            mu0=mu0,
            w0=w0,
            x0=x0,
            signs=signs,
        )
        return left_factor, core_matrix, right_factor, active_cols

    @jax.jit
    def jac_fn(N, y):
        base_factors = base_jac_fn(N, y)
        direct_left, direct_core, direct_right, direct_active_cols = jax.pure_callback(
            _host_factors,
            callback_spec,
            y,
        )
        base_jacobian = base_factors.base_jacobian.at[:, active_idx].add(direct_active_cols)
        left_factor = jnp.concatenate([base_factors.left_factor, direct_left], axis=1)
        core_matrix = _block_diag_rect(base_factors.core_matrix, direct_core)
        right_factor = jnp.concatenate([base_factors.right_factor, direct_right], axis=0)
        return LowRankJacobianFactors(
            base_jacobian=base_jacobian,
            left_factor=left_factor,
            core_matrix=core_matrix,
            right_factor=right_factor,
        )

    return jac_fn


def run_full_boltzmann_jax(
    config: Optional[JAXFullBoltzmannConfig] = None,
    **kw,
) -> JAXFullBoltzmannResult:
    if config is None:
        config = JAXFullBoltzmannConfig(**kw)

    _ensure_full_boltzmann_runtime_optimisations()
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("JAX CPU backend is unavailable for runtime_device_policy='cpu_preferred'.")
    with jax.default_device(cpu_devices[0]):
        result = _run_full_boltzmann_impl(config)
    result.metadata = dict(getattr(result, "metadata", {}) or {})
    result.metadata["runtime_device_policy"] = "cpu_preferred"
    result.metadata["runtime_device_initial_platform"] = "cpu"
    result.metadata["runtime_device_final_platform"] = "cpu"
    result.metadata["runtime_device_contract"] = "cpu_preferred_exact_runtime_v1"
    result.metadata["runtime_device_fallback_applied"] = False
    return result


def _run_full_boltzmann_impl(config: JAXFullBoltzmannConfig) -> JAXFullBoltzmannResult:
    validate_rate_table_window_jax(
        config.T_handoff,
        load_rate_table(int(config.n_reactions)),
        context="JAX full-Boltzmann phase-2 handoff",
        strict=True,
    )
    validate_rate_table_window_jax(
        config.T_end,
        load_rate_table(int(config.n_reactions)),
        context="JAX full-Boltzmann phase-2 end",
        strict=True,
    )

    rhs_p1, jac_p1, layout_p1, _mu0, _w0, _x0, _signs, _q_nodes, _energy_weight = _get_full_boltzmann_rhs(
        phase=1,
        correction_level=int(config.correction_level),
        collision_mode=str(config.collision_mode),
        collision_kernel_backend=str(config.collision_kernel_backend),
        collision_preflight_relaxation=float(config.collision_preflight_relaxation),
        thermo_tier=int(config.thermo_tier),
        N_mu=int(config.N_mu),
        N_q=int(config.N_q),
        n_network_species=2,
        n_reactions=int(config.n_reactions),
        tau_n=float(config.tau_n),
        eta=float(config.eta),
        N_eff=float(config.N_eff),
        f_nu=float(config.f_nu),
    )
    rhs_p2, jac_p2, layout_p2, _mu0_2, _w0_2, _x0_2, _signs_2, _q_nodes_2, _energy_weight_2 = _get_full_boltzmann_rhs(
        phase=2,
        correction_level=int(config.correction_level),
        collision_mode=str(config.collision_mode),
        collision_kernel_backend=str(config.collision_kernel_backend),
        collision_preflight_relaxation=float(config.collision_preflight_relaxation),
        thermo_tier=int(config.thermo_tier),
        N_mu=int(config.N_mu),
        N_q=int(config.N_q),
        n_network_species=int(N_SPECIES),
        n_reactions=int(config.n_reactions),
        tau_n=float(config.tau_n),
        eta=float(config.eta),
        N_eff=float(config.N_eff),
        f_nu=float(config.f_nu),
    )

    T_start = jnp.asarray(float(config.T_start))
    T_nu_init = T_start
    lnp0, lpn0 = compute_born_rates(T_start, T_nu_init, jnp.asarray(float(config.tau_n)))
    Xn_eq = float(lpn0 / jnp.maximum(lnp0 + lpn0, 1e-100))

    y0 = np.zeros(int(layout_p1["n_total"]), dtype=np.float64)
    y0[layout_p1["i_Sp"]] = float(config.Sigma_H_plus)
    y0[layout_p1["i_Sm"]] = float(config.Sigma_H_minus)
    y0 = _pack_transport_slice(
        jnp.asarray(y0),
        layout_p1,
        jnp.asarray(_initial_transport_state(config.N_mu, config.N_q)),
    )
    y0 = np.array(y0, dtype=np.float64, copy=True)
    y0[layout_p1["i_tg"]] = float(config.T_start)
    if int(config.thermo_tier) >= 2:
        y0[layout_p1["i_tne"]] = float(config.T_start)
        y0[layout_p1["i_tnx"]] = float(config.T_start)
    y0[layout_p1["i_net"]] = Xn_eq
    y0[layout_p1["i_net"] + 1] = 1.0 - Xn_eq

    event_p1 = _build_event_fn(layout_p1["i_tg"], float(config.T_handoff))
    result_p1 = jax_rodas5p_solve(
        rhs_p1,
        jnp.asarray(y0),
        jnp.asarray([0.0, 50.0]),
        rtol=float(config.rtol),
        atol=float(config.atol),
        max_steps=int(config.max_steps),
        event_fn=event_p1,
        event_refine_steps=int(config.event_refine_steps),
        jac_fn=jac_p1,
        linear_solver_prepare_fn=prepare_low_rank_linear_solver,
        linear_solver_solve_fn=solve_prepared_low_rank_linear_system,
    )

    phase1_diag = classify_solve_ivp_result(
        result_p1,
        target_reached=bool(getattr(result_p1, "event_triggered", False)),
        phase="phase1",
    )
    if not result_p1.success or not bool(getattr(result_p1, "event_triggered", False)):
        return JAXFullBoltzmannResult(
            Yp=float("nan"),
            DH=float("nan"),
            Li7H=float("nan"),
            Li6H=float("nan"),
            N_eff=float("nan"),
            Xn_freeze=float("nan"),
            success=False,
            n_steps_p1=int(result_p1.n_steps),
            n_steps_p2=0,
            metadata={"error": "phase 1 did not reach handoff", "phase1_solver_diagnostics": phase1_diag},
        )

    y_p1 = np.asarray(result_p1.y_final, dtype=np.float64)
    N_handoff = float(result_p1.N_final)
    Xn_handoff = float(y_p1[layout_p1["i_net"]])
    X_p2 = np.asarray(phase1_to_phase2_jax(jnp.asarray(Xn_handoff)))

    y_handoff = np.zeros(int(layout_p2["n_total"]), dtype=np.float64)
    y_handoff[layout_p2["i_Sp"]] = y_p1[layout_p1["i_Sp"]]
    y_handoff[layout_p2["i_Sm"]] = y_p1[layout_p1["i_Sm"]]
    y_handoff = np.array(
        _pack_transport_slice(
            jnp.asarray(y_handoff),
            layout_p2,
            _unpack_transport(jnp.asarray(y_p1), layout_p1),
        ),
        dtype=np.float64,
        copy=True,
    )
    y_handoff[layout_p2["i_S"]] = y_p1[layout_p1["i_S"]]
    y_handoff[layout_p2["i_tg"]] = y_p1[layout_p1["i_tg"]]
    if int(config.thermo_tier) >= 2:
        y_handoff[layout_p2["i_tne"]] = y_p1[layout_p1["i_tne"]]
        y_handoff[layout_p2["i_tnx"]] = y_p1[layout_p1["i_tnx"]]
    y_handoff[layout_p2["i_net"] : layout_p2["i_net"] + int(N_SPECIES)] = X_p2

    event_p2 = _build_event_fn(layout_p2["i_tg"], float(config.T_end))
    result_p2 = jax_rodas5p_solve(
        rhs_p2,
        jnp.asarray(y_handoff),
        jnp.asarray([N_handoff, N_handoff + 30.0]),
        rtol=float(config.rtol),
        atol=float(config.atol),
        max_steps=int(config.max_steps),
        event_fn=event_p2,
        event_refine_steps=int(config.event_refine_steps),
        jac_fn=jac_p2,
        linear_solver_prepare_fn=prepare_low_rank_linear_solver,
        linear_solver_solve_fn=solve_prepared_low_rank_linear_system,
    )

    phase2_diag = classify_solve_ivp_result(
        result_p2,
        target_reached=bool(getattr(result_p2, "event_triggered", False)),
        phase="phase2",
    )

    y_final = np.asarray(result_p2.y_final, dtype=np.float64)
    X_final = y_final[layout_p2["i_net"] : layout_p2["i_net"] + int(N_SPECIES)]
    T_final = float(y_final[layout_p2["i_tg"]])
    if int(config.thermo_tier) >= 2:
        T_nu_e_final = float(y_final[layout_p2["i_tne"]])
        T_nu_x_final = float(y_final[layout_p2["i_tnx"]])
        T_nu_final = T_nu_e_final
        N_eff_meas = float(
            N_eff_from_3T_jax(
                jnp.asarray(T_final),
                jnp.asarray(T_nu_e_final),
                jnp.asarray(T_nu_x_final),
            )
        )
    else:
        T_nu_e_final = float(tier1_T_nu_from_T_gamma_jax(jnp.asarray(T_final)))
        T_nu_x_final = T_nu_e_final
        T_nu_final = T_nu_e_final
        N_eff_meas = float(tier1_N_eff_from_T_ratio_jax(jnp.asarray(T_final), jnp.asarray(T_nu_final)))

    Yp = float(X_final[5])
    DH = float(X_final[2] / (2.0 * max(X_final[1], 1e-30)))
    Li7H = float((X_final[6] / 7.0 + X_final[7] / 7.0) / max(X_final[1], 1e-30))
    Li6H = float(max(X_final[8], 0.0) / (6.0 * max(X_final[1], 1e-30))) if int(N_SPECIES) > 8 else 0.0

    transport_final = np.asarray(_unpack_transport(jnp.asarray(y_final), layout_p2))
    collision_mode = str(config.collision_mode).strip().lower()
    species_identical_max_abs = float(
        np.max(np.abs(transport_final - transport_final[:1]))
    )

    _collision_scope_map = {
        "collisionless": "collisionless_only_v1",
        "spectral_relaxation_preflight": "spectral_relaxation_preflight_v1",
        "projected_physical_preflight": "projected_physical_preflight_v1",
        "direct_kernel_preflight": "direct_kernel_preflight_v1",
        "jax_kernel_preflight": "jax_kernel_preflight_v1",
        "jax_kernel_remap_preflight": "jax_kernel_temperature_remap_preflight_v1",
        "jax_kernel_remap_preconditioned_preflight": "jax_kernel_temperature_remap_jacobian_preconditioned_v1",
        "jax_kernel_projected_preflight": "jax_kernel_temperature_remap_mangano_projected_v1",
        "ap_unified_preflight": "ap_unified_preflight_v1",
        "ap_unified_nu_nu_preflight": "ap_unified_plus_energy_conserving_nu_nu_3T_preflight_v1",
        "ap_unified_nu_nu_spectral_preflight": "ap_unified_plus_energy_projected_nu_nu_spectral_bank_preflight_v1",
        "ap_unified_nu_nu_spectral_accuracy_preflight": "ap_unified_plus_calibrated_energy_transfer_and_projected_nu_nu_spectral_bank_preflight_v1",
        "ap_preconditioned_canonical": "ap_preconditioned_canonical_modified_equation_v1",
        "ap_unified_ell2_kernelized_preflight": "ap_unified_plus_resolved_ell2_kernelized_nu_e_preflight_v1",
    }
    collision_kernel_metadata = dict(
        getattr(
            config,
            "_collision_kernel_metadata",
            _full_solver_collision_backend_report(config.collision_kernel_backend),
        )
        or {}
    )
    metadata = {
        "backend": "jax_full_boltzmann",
        "driver": "full_boltzmann_jax",
        "transport_mode": "full_boltzmann_ray",
        "transport_scope_contract": "collisionless_full_phase_space_ray_v1",
        "collision_scope_contract": _collision_scope_map[collision_mode],
        "thermo_scope_contract": (
            "tier1_single_T_nu_v1"
            if int(config.thermo_tier) == 1
            else "private_tier2_collision_moment_3T_v1"
        ),
        "q_advection_scheme": str(config.q_advection_scheme),
        "q_advection_contract": "continuous_upwind_semidiscrete_v1",
        "characteristic_oracle_contract": "pchip_exact_remap_oracle_v1",
        "thermo_tier": int(config.thermo_tier),
        "correction_level": int(config.correction_level),
        "distribution_species": list(_DIST_SPECIES),
        "distribution_species_count": len(_DIST_SPECIES),
        "weak_monopole_scope_contract": "separate_nue_nuebar_live_f0_v1",
        "weak_monopole_species": ["nue", "nuebar"],
        "stress_species_weight_contract": "temperature4_degeneracy_weighted_species_stress_v1",
        "stress_species_degeneracy": [1.0, 1.0, 2.0, 2.0],
        "species_identical_approx": collision_mode == "collisionless",
        "species_identical_max_abs_final": species_identical_max_abs,
        "transport_min_final": float(np.min(transport_final)),
        "transport_max_final": float(np.max(transport_final)),
        "state_dim_phase1": int(layout_p1["n_total"]),
        "state_dim_phase2": int(layout_p2["n_total"]),
        "transport_dim": int(layout_p2["n_transport"]),
        "N_mu": int(config.N_mu),
        "N_q": int(config.N_q),
        "n_reactions": int(config.n_reactions),
        "N_eff_measured": N_eff_meas,
        "T_nu_final": T_nu_final,
        "T_nu_e_final": T_nu_e_final,
        "T_nu_x_final": T_nu_x_final,
        "phase1_handoff_N": float(N_handoff),
        "phase1_handoff_T": float(y_p1[layout_p1["i_tg"]]),
        "phase1_handoff_Xn": float(Xn_handoff),
        "phase1_solver_diagnostics": phase1_diag,
        "phase2_solver_diagnostics": phase2_diag,
        "phase1_jax_solver_diagnostics": dict(getattr(result_p1, "diagnostics", {}) or {}),
        "phase2_jax_solver_diagnostics": dict(getattr(result_p2, "diagnostics", {}) or {}),
        "jacobian_payload_contract": _resolve_jacobian_payload_contract(
            collision_mode, int(config.thermo_tier)
        ),
        "jacobian_transport_block_contract": "analytic_q_advection_block_v1",
        "jacobian_transport_projector_contract": _resolve_jacobian_transport_projector_contract(
            collision_mode, int(config.thermo_tier)
        ),
        "jacobian_low_rank_moment_dim": _resolve_jacobian_low_rank_moment_dim(
            collision_mode, int(config.thermo_tier), int(config.N_q)
        ),
        "jacobian_low_rank_apply_dim": _resolve_jacobian_low_rank_apply_dim(
            collision_mode,
            int(config.thermo_tier),
            int(config.N_q),
            int(N_SPECIES),
        ),
        "collision_mode": collision_mode,
        "collision_kernel_backend_requested": collision_kernel_metadata.get(
            "kernel_backend_requested", str(config.collision_kernel_backend)
        ),
        "collision_kernel_backend_effective": collision_kernel_metadata.get(
            "kernel_backend_effective", str(config.collision_kernel_backend)
        ),
        "collision_kernel_backend_jit_safe": collision_kernel_metadata.get(
            "kernel_backend_jit_safe", True
        ),
        "collision_kernel_backend_grad_safe": collision_kernel_metadata.get(
            "kernel_backend_grad_safe", True
        ),
        "collision_kernel_backend_gpu_resident": collision_kernel_metadata.get(
            "kernel_backend_gpu_resident", False
        ),
        "collision_kernel_backend_host_mediated": collision_kernel_metadata.get(
            "kernel_backend_host_mediated", False
        ),
        "collision_kernel_backend_contract": collision_kernel_metadata.get(
            "full_solver_collision_backend_contract",
            "jax_xla_only_no_external_kernel_backend",
        ),
        **collision_kernel_metadata,
        "nu_nu_equilibration_enabled": _mode_enables_nu_nu_equilibration(collision_mode),
        "nu_nu_spectral_bank_enabled": _mode_enables_nu_nu_spectral_bank(collision_mode),
        "nu_nu_spectral_impl": (
            "tensorized_static_kernel_v1"
            if _mode_enables_nu_nu_spectral_bank(collision_mode)
            else "off"
        ),
        "nu_nu_spectral_self_thermalization_contract": (
            "number_energy_neutral_leading_order_shape_damping_v1"
            if _mode_enables_nu_nu_spectral_bank(collision_mode)
            else "off"
        ),
        "mangano_energy_transfer_C_rate": _COLLISION_PREFLIGHT_C_RATE,
        "mangano_energy_transfer_C_rate_effective": (
            _COLLISION_PREFLIGHT_C_RATE * _ap_energy_transfer_scale_for_mode(collision_mode)
        ),
        "ap_accuracy_energy_transfer_scale": _ap_energy_transfer_scale_for_mode(collision_mode),
        "ap_accuracy_candidate_enabled": (
            collision_mode == "ap_unified_nu_nu_spectral_accuracy_preflight"
        ),
        "ap_accuracy_candidate_contract": (
            "calibrated_no_qke_energy_transfer_scale_plus_spectral_nu_nu_v1"
            if collision_mode == "ap_unified_nu_nu_spectral_accuracy_preflight"
            else "off"
        ),
        "collision_rate_prefactor_source": "rabbit.thermo.nudec_tables._C_RATE",
        "rate_enforcement_sign_contract": "positive_scale_requires_matching_energy_moment_sign_v1",
        "kernel_jacobian_preconditioner_enabled": _mode_enables_kernel_jacobian_preconditioner(collision_mode),
        "collision_preflight_relaxation": float(config.collision_preflight_relaxation),
        "mass_conservation": float(mass_conservation_residual_jax(jnp.asarray(X_final))),
    }

    return JAXFullBoltzmannResult(
        Yp=Yp,
        DH=DH,
        Li7H=Li7H,
        Li6H=Li6H,
        N_eff=N_eff_meas,
        Xn_freeze=float(Xn_handoff),
        success=bool(result_p2.success and getattr(result_p2, "event_triggered", False)),
        n_steps_p1=int(result_p1.n_steps),
        n_steps_p2=int(result_p2.n_steps),
        metadata=metadata,
    )
