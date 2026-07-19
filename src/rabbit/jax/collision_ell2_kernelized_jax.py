"""rabbit.jax.collision_ell2_kernelized_jax — JAX twin of the ν–e ℓ=2 kernel.

BD631 FB (Track FB).  This module is the **pure-jnp twin** of the host-numpy
``rabbit.collisions.kernelized_ell2`` module.  It exists because the numpy
module uses ``scipy.interpolate.PchipInterpolator`` + ``@lru_cache`` (N6 lesson:
no python cache, no ``round()`` inside a jit trace), so it cannot be reused
inside the JAX full-Boltzmann RHS/Jacobian.  The twin loads the candidate v1
electron-minus-only table ONCE at driver-build time (host numpy, outside any trace) and
exposes:

    - :func:`load_ell2_kernel_tables` — host load → static jnp arrays.
    - :func:`m2_gamma_jnp` — in-trace ``M2_G(T) = -diag(nu2) + R2`` via
      ``jnp.interp`` on log-T (linear; **exact at the 28 T-grid nodes**).
    - :func:`kernelized_C2_ell2_f2_jnp` — the f2-DIRECT twin
      ``df2/dN = where(f0<1e-6, 0, -2·(M2_G(T)@f2)) / H_MeV`` (M3: no 1/f0).
    - :func:`a2_block_analytic_jnp` — the analytic linear operator
      ``A2 = -2·M2_G/H`` with tail rows zeroed (== ``jacfwd`` of the twin).

At table nodes the f2-DIRECT form (M3) is exact vs the numpy Ψ₂-space route: the bridge f₀'s
telescope so ``twin(f0·Ψ₂) == f0 · numpy_C2(Ψ₂)`` to machine precision at the 28
T-grid nodes (devil review, telescoping rel err ≤ 5e-16).  See
``audit/reference/FB_ELL2_DESIGN.md`` §A.ii/§B and ``audit/telemetry/fb/fb_devil.json``.

SCOPE (inherited, F-029/F-031/F-034):
    - ν_e process only (combos ``nu_e__nue`` / ``nu_e__nux``); the ν–ν channel
      is deferred (F-031, detailed-balance break) and refused by the loader.
    - Single-T frame: the table is a single-T object; ``f0(q/T_ν)`` uses the
      neutrino frame, exact when ``T_γ ≈ T_ν`` near decoupling (F-034).
    - N_q must equal the table's 20 nodes (no q-interpolation; F-029/W1).

The numpy module (``kernelized_ell2.py``) STAYS numpy/host; this file is the jnp
twin at table nodes.  NumPy PCHIP and JAX linear interpolation differ between
nodes, so this is NOT end-to-end interpolation parity or physics validation.
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

# Deep-tail occupancy floor — matches ``kernelized_ell2._F0_TAIL_FLOOR`` (F-029).
_F0_TAIL_FLOOR = 1e-6

# Promoted data asset (same file the numpy module loads).
_DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "ell2_kernel_tables_v1.npz"
)

# species/bank tag → table combo key.  ν–e process only (F-031 refuses ν–ν).
_NUE_COMBO = "nu_e__nue"
_NUX_COMBO = "nu_e__nux"
_NUNU_TAGS = {"nu_nu", "nunu", "nu_nu__nue", "nu_nu__nux"}


def refuse_nu_nu(tag: str) -> None:
    """Refuse the ν–ν ℓ=2 channel (F-031: leg-4 interp breaks detailed balance).

    The FB ℓ=2 lane wires only the ν–e process (fixed banks nue/nuebar→nu_e__nue,
    nux→nu_e__nux), so ν–ν is structurally unreachable; this guard makes the
    scope limit explicit for any caller that tries to route a ν–ν tag.
    """
    if str(tag).strip().lower() in _NUNU_TAGS:
        raise NotImplementedError(
            "FB ℓ=2 kernelized collision: the ν–ν process is not wired "
            "(F-031: leg-4 barycentric interpolation breaks detailed balance). "
            "Only the ν–e process (nu_e__nue / nu_e__nux) is available."
        )


class Ell2KernelTables(NamedTuple):
    """Static jnp arrays for the two wired ν–e ℓ=2 kernel combos.

    All fields are ``jnp`` arrays closed over the jitted RHS/Jacobian at
    driver-build time (host load, outside any trace).
    """

    q_nodes: jnp.ndarray     # (N_q,)   Gauss-Laguerre nodes
    lnT: jnp.ndarray         # (n_T,)   log of the 28-node T grid
    nue_nu2: jnp.ndarray     # (n_T, N_q)      Γ-basis loss, nu_e__nue
    nue_R2: jnp.ndarray      # (n_T, N_q, N_q) Γ-basis redistribution, nu_e__nue
    nux_nu2: jnp.ndarray     # (n_T, N_q)      Γ-basis loss, nu_e__nux
    nux_R2: jnp.ndarray      # (n_T, N_q, N_q) Γ-basis redistribution, nu_e__nux


def load_ell2_kernel_tables(path: Optional[str] = None) -> Ell2KernelTables:
    """Host-load the ν–e ℓ=2 kernel table into static jnp arrays.

    Called ONCE at driver-build time (host numpy, outside any jit trace).
    Only the wired ν–e combos are loaded; a ν–ν request raises (F-031).
    """
    resolved = os.path.abspath(path or _DEFAULT_TABLE_PATH)
    npz = np.load(resolved, allow_pickle=True)
    q_nodes = np.asarray(npz["q_nodes"], dtype=np.float64)
    T_grid = np.asarray(npz["T_grid"], dtype=np.float64)
    lnT = np.log(T_grid)
    return Ell2KernelTables(
        q_nodes=jnp.asarray(q_nodes, dtype=jnp.float64),
        lnT=jnp.asarray(lnT, dtype=jnp.float64),
        nue_nu2=jnp.asarray(npz[f"{_NUE_COMBO}__nu2"], dtype=jnp.float64),
        nue_R2=jnp.asarray(npz[f"{_NUE_COMBO}__R2"], dtype=jnp.float64),
        nux_nu2=jnp.asarray(npz[f"{_NUX_COMBO}__nu2"], dtype=jnp.float64),
        nux_R2=jnp.asarray(npz[f"{_NUX_COMBO}__R2"], dtype=jnp.float64),
    )


def _interp_cols(lt: jnp.ndarray, lnT: jnp.ndarray, table2d: jnp.ndarray) -> jnp.ndarray:
    """Interpolate each column of ``table2d`` (n_T, K) over log-T at ``lt`` → (K,)."""
    return jax.vmap(lambda col: jnp.interp(lt, lnT, col), in_axes=1, out_axes=0)(table2d)


def m2_gamma_jnp(
    T_nu: jnp.ndarray,
    lnT: jnp.ndarray,
    nu2: jnp.ndarray,
    R2: jnp.ndarray,
) -> jnp.ndarray:
    """Γ-basis f-rate matrix ``M2_G(T_ν) = -diag(nu2) + R2`` [MeV], jnp/in-trace.

    ``jnp.interp`` is linear in log-T and **exact at the 28 T-grid nodes** (it
    returns the tabulated slice there); between nodes it differs from the
    numpy module's monotone-cubic PCHIP by the documented linear-vs-PCHIP gap
    (~1e-3 rel; design open-risk #5).
    """
    lt = jnp.log(jnp.asarray(T_nu, dtype=jnp.float64))
    nu2_T = _interp_cols(lt, lnT, nu2)                       # (N_q,)
    n_q = nu2.shape[1]
    R2_T = _interp_cols(lt, lnT, R2.reshape(R2.shape[0], n_q * n_q)).reshape(n_q, n_q)
    return -jnp.diag(nu2_T) + R2_T


def kernelized_C2_ell2_f2_jnp(
    f2: jnp.ndarray,
    T_nu: jnp.ndarray,
    H_MeV: jnp.ndarray,
    lnT: jnp.ndarray,
    nu2: jnp.ndarray,
    R2: jnp.ndarray,
    q_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """f2-DIRECT ℓ=2 collision source ``df2/dN`` (M3).

    ``df2/dN(q_i) = where(f0(q_i/T_ν) < 1e-6, 0, -2·(M2_G(T_ν) @ f2)(q_i)) / H_MeV``.

    The bridge f₀'s telescope analytically (no 1/f₀ tail dust): this is the same
    numbers as the numpy Ψ₂-space route but state-independent (linear in ``f2``)
    and safe in the deep tail (F-029/F-032).  ``H_MeV`` is passed directly (no
    ``H_inv_sec``/``_MEV2S`` round-trip).
    """
    M2_G = m2_gamma_jnp(T_nu, lnT, nu2, R2)                  # (N_q, N_q)
    f0 = 1.0 / (jnp.exp(jnp.clip(q_nodes / T_nu, -500.0, 500.0)) + 1.0)
    tail = f0 < _F0_TAIL_FLOOR
    df2_dt = jnp.where(tail, 0.0, -2.0 * (M2_G @ f2))        # [MeV], f-moment rate
    return df2_dt / jnp.maximum(H_MeV, 1e-100)               # per e-fold


def a2_block_analytic_jnp(
    T_nu: jnp.ndarray,
    H_MeV: jnp.ndarray,
    lnT: jnp.ndarray,
    nu2: jnp.ndarray,
    R2: jnp.ndarray,
    q_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """Analytic ℓ=2 Jacobian block ``A2 = -2·M2_G(T_ν)/H_MeV`` (tail rows zeroed).

    Since :func:`kernelized_C2_ell2_f2_jnp` is linear in ``f2``, this equals
    ``jacfwd(kernelized_C2_ell2_f2_jnp, argnums=0)`` exactly (test T6).  The
    diagonal is ``≤ 0`` on all thermally-populated core nodes (dissipative;
    devil claim 6: 660/660 core entries ≤ 0).
    """
    M2_G = m2_gamma_jnp(T_nu, lnT, nu2, R2)
    f0 = 1.0 / (jnp.exp(jnp.clip(q_nodes / T_nu, -500.0, 500.0)) + 1.0)
    tail = f0 < _F0_TAIL_FLOOR
    A2 = (-2.0 / jnp.maximum(H_MeV, 1e-100)) * M2_G
    return jnp.where(tail[:, None], 0.0, A2)
