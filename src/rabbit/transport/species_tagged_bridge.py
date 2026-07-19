from __future__ import annotations

import os
from dataclasses import is_dataclass, replace
import numpy as np

from rabbit.collisions.species import as_species, Species, BANK_DEGENERACY
from rabbit.transport.teff_collision_bridge import apply_gather_scatter_collision
from rabbit.thermo.incomplete_decoupling import compute_energy_exchange_rate


TOTAL_G = float(
    BANK_DEGENERACY[Species.NUE]
    + BANK_DEGENERACY[Species.NUEBAR]
    + BANK_DEGENERACY[Species.NUX]
)
_SPECIES_PROFILE_CACHE: dict[tuple[str, bytes], np.ndarray] = {}
_PROFILE_MEASURE_CACHE: dict[tuple[bytes, bytes], np.ndarray] = {}


def _user_relax() -> float:
    raw = os.environ.get("RABBIT_COLLISION_BRIDGE_RELAX", "1.0")
    try:
        return float(raw)
    except Exception:
        return 1.0


def _low_q_shape(q_nodes):
    q = np.asarray(q_nodes, dtype=np.float64)
    return np.exp(-q / 3.0)


def _signed_split_shape(q_nodes):
    q = np.asarray(q_nodes, dtype=np.float64)
    base = np.exp(-q / 3.0)
    return (q / 3.0 - 1.0) * base


def _species_profile(species, q_nodes):
    """
    Shared-total-preserving profile set.

    We choose profiles such that pointwise weighted mean over species is exactly 1:
        (1*r_nue + 1*r_nuebar + 4*r_nux)/6 = 1
    """
    sp = as_species(species)
    q = np.asarray(q_nodes, dtype=np.float64)
    cache_key = (sp.value if isinstance(sp, Species) else str(sp), q.tobytes())
    cached = _SPECIES_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    phi = _low_q_shape(q)

    # v4 profile:
    # - shared low-q electron-flavor enhancement: a * phi
    # - ν_e / \barν_e antisplit: ± b * chi
    # - ν_x compensates only the shared electron-flavor enhancement
    #
    # Pointwise weighted mean is still exactly 1:
    #   (1*r_nue + 1*r_nuebar + 4*r_nux)/6 = 1
    #
    # because:
    #   +a phi + a phi + 4*(-a/2) phi = 0
    #   +b chi - b chi + 4*0 = 0
    a = 0.35
    b = 0.08
    chi = _signed_split_shape(q)
    if sp is Species.NUE:
        prof = 1.0 + a * phi + b * chi
    elif sp is Species.NUEBAR:
        prof = 1.0 + a * phi - b * chi
    elif sp is Species.NUX:
        prof = 1.0 - 0.5 * a * phi
    else:
        raise ValueError(f"Unknown species: {species}")

    prof = np.asarray(prof, dtype=np.float64)
    _SPECIES_PROFILE_CACHE[cache_key] = prof
    return prof


def _positive_measure(base_C, q_nodes, q_weights):
    key = (
        np.asarray(q_nodes, dtype=np.float64).tobytes(),
        np.asarray(q_weights, dtype=np.float64).tobytes(),
    )
    cached = _PROFILE_MEASURE_CACHE.get(key)
    if cached is None:
        q = np.asarray(q_nodes, dtype=np.float64)
        w = np.asarray(q_weights, dtype=np.float64)
        cached = w * q**3
        _PROFILE_MEASURE_CACHE[key] = cached
    C = np.asarray(base_C, dtype=np.float64)
    return np.abs(C) * cached + 1.0e-300


def _alpha_from_profile(profile, base_C, q_nodes, q_weights):
    """
    Collapse q-dependent profile to a scalar for delta_I and delta_rho_nu.
    Using a positive measure built from the shared C_monopole keeps the weighted
    species-mean exactly at 1 whenever the pointwise weighted profile mean is 1.
    """
    mu = _positive_measure(base_C, q_nodes, q_weights)
    return float(np.sum(mu * np.asarray(profile, dtype=np.float64)) / np.sum(mu))


def _finite_or_zero(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def apply_species_tagged_bridge(
    *,
    species,
    I,
    J,
    w0,
    q_nodes,
    q_weights,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    H: float,
    record_debug: bool = True,
):
    """
    Species-tagged bridge v3: shared-total-preserving split.

    IMPORTANT:
    - We no longer match tiny external Qdot targets.
    - We preserve the original shared collision strength and only split it by species.
    - The shared reference bridge uses T_nu_e, matching the original Tier-2 shared path.
    """
    sp = as_species(species)
    relax = _user_relax()

    # Shared reference bridge: preserve the original shared-path strength
    base = apply_gather_scatter_collision(
        I, J, w0, q_nodes, q_weights,
        T_gamma, float(T_nu_e), H,
        compute_tangency=record_debug,
    )

    profile = _species_profile(sp, q_nodes)
    alpha = _alpha_from_profile(profile, base.C_monopole, q_nodes, q_weights)

    scaled_C = _finite_or_zero(np.asarray(base.C_monopole, dtype=np.float64) * profile * relax)
    scaled_deltaI = _finite_or_zero(np.asarray(base.delta_I, dtype=np.float64) * alpha * relax)
    scaled_delta_rho = float(base.delta_rho_nu) * alpha * relax
    clean_f = _finite_or_zero(base.f_monopole)

    if is_dataclass(base):
        out = replace(
            base,
            delta_rho_nu=scaled_delta_rho,
            delta_I=scaled_deltaI,
            C_monopole=scaled_C,
            f_monopole=clean_f,
        )
    else:
        payload = dict(vars(base))
        payload.update(
            delta_rho_nu=scaled_delta_rho,
            delta_I=scaled_deltaI,
            C_monopole=scaled_C,
            f_monopole=clean_f,
        )
        out = type(base)(**payload)

    if record_debug:
        # debug metadata, preserving old field names for existing scripts
        q_shared = float(
            compute_energy_exchange_rate(
                np.asarray(base.C_monopole, dtype=np.float64),
                np.asarray(q_nodes, dtype=np.float64),
                np.asarray(q_weights, dtype=np.float64),
                float(T_nu_e),
            )
        )
        q_species = float(
            compute_energy_exchange_rate(
                np.asarray(scaled_C, dtype=np.float64),
                np.asarray(q_nodes, dtype=np.float64),
                np.asarray(q_weights, dtype=np.float64),
                float(T_nu_e),
            )
        )

        setattr(out, "species_tagged_relax", float(relax))
        setattr(out, "species_tagged_alpha", float(alpha))
        setattr(out, "species_tagged_amp", float(alpha))
        setattr(out, "species_tagged_qdot_shape", float(q_shared))
        setattr(out, "species_tagged_qdot_target", float(q_species))
    return out
