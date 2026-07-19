from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from rabbit.transport.reduced_modal_bank import load_runtime_bank, project_vector_onto_bank
from rabbit.transport.species_tagged_bridge import (
    apply_species_tagged_bridge as _raw_apply_species_tagged_bridge,
)
from rabbit.debug.modal_contract import RAW_AUTHORITATIVE_PATH, REDUCED_MODAL_MODE


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _energy_moment(C_monopole: np.ndarray, q_nodes: np.ndarray, q_weights: np.ndarray) -> float:
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    C = np.asarray(C_monopole, dtype=np.float64)
    return float(np.sum(w * (q ** 3) * C))


def _safe_amp(qproj: float, qraw: float, eps: float = 1e-300) -> float:
    if not np.isfinite(qraw) or abs(qraw) < eps:
        return 0.0
    amp = qproj / qraw
    if not np.isfinite(amp):
        return 0.0
    return float(amp)


def _make_result_like(base: Any, C_proj: np.ndarray, amp: float, debug: Dict[str, Any]) -> Any:
    cls = type(base)

    kwargs: Dict[str, Any] = {}
    if hasattr(base, "tangency_D2"):
        kwargs["tangency_D2"] = float(base.tangency_D2)
    if hasattr(base, "delta_rho_nu"):
        kwargs["delta_rho_nu"] = float(base.delta_rho_nu) * amp
    if hasattr(base, "delta_I"):
        kwargs["delta_I"] = np.nan_to_num(
            np.asarray(base.delta_I, dtype=np.float64) * amp,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    if hasattr(base, "C_monopole"):
        kwargs["C_monopole"] = np.nan_to_num(
            np.asarray(C_proj, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    if hasattr(base, "f_monopole"):
        kwargs["f_monopole"] = np.nan_to_num(
            np.asarray(base.f_monopole, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    if hasattr(base, "theta_per_ray"):
        kwargs["theta_per_ray"] = np.nan_to_num(
            np.asarray(base.theta_per_ray, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    out = cls(**kwargs)
    setattr(out, "bridge_debug", debug)
    setattr(out, "reduced_modal_debug", debug)
    return out


def apply_reduced_modal_species_bridge(
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
    bank_path: str | None = None,
    n_shared: int | None = None,
    n_species: int | None = None,
    include_fd2: bool | None = None,
):
    bank_path = bank_path or os.environ.get("RABBIT_REDUCED_MODAL_BANK", "")
    if not bank_path:
        return _raw_apply_species_tagged_bridge(
            species=species, I=I, J=J, w0=w0, q_nodes=q_nodes, q_weights=q_weights,
            T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H=H,
        )

    n_shared = _env_int("RABBIT_REDUCED_MODAL_SHARED_RANK", 2) if n_shared is None else int(n_shared)
    n_species = _env_int("RABBIT_REDUCED_MODAL_SPECIES_RANK", 1) if n_species is None else int(n_species)
    include_fd2 = _env_bool("RABBIT_REDUCED_MODAL_INCLUDE_FD2", True) if include_fd2 is None else bool(include_fd2)

    base = _raw_apply_species_tagged_bridge(
        species=species, I=I, J=J, w0=w0, q_nodes=q_nodes, q_weights=q_weights,
        T_gamma=T_gamma, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H=H,
    )

    C_raw = np.asarray(base.C_monopole, dtype=np.float64)
    q_nodes = np.asarray(q_nodes, dtype=np.float64)
    q_weights = np.asarray(q_weights, dtype=np.float64)

    bank = load_runtime_bank(
        bank_path=Path(bank_path),
        species=str(species),
        q_nodes=q_nodes,
        q_weights=q_weights,
        n_shared=n_shared,
        n_species=n_species,
        include_fd2=include_fd2,
    )
    proj = project_vector_onto_bank(C_raw, bank)
    C_proj = np.asarray(proj["recon"], dtype=np.float64)

    qdot_raw = _energy_moment(C_raw, q_nodes, q_weights)
    qdot_proj = _energy_moment(C_proj, q_nodes, q_weights)
    amp = _safe_amp(qdot_proj, qdot_raw)

    debug = {
        "mode_labels": list(proj["labels"]),
        "coeffs": [float(x) for x in proj["coeffs"]],
        "cos": float(proj["cos"]),
        "resid": float(proj["resid"]),
        "qdot_raw": float(qdot_raw),
        "qdot_proj": float(qdot_proj),
        "amp": float(amp),
        "bank_path": str(bank_path),
        "shared_rank": int(n_shared),
        "species_rank": int(n_species),
        "include_fd2": bool(include_fd2),
        "base_debug": getattr(base, "bridge_debug", None),
    }

    return _make_result_like(base, C_proj=C_proj, amp=amp, debug=debug)


# === POST_RETURN_GATE_WRAPPER ===
if "_orig_apply_reduced_modal_species_bridge" not in globals():
    _orig_apply_reduced_modal_species_bridge = apply_reduced_modal_species_bridge

    def apply_reduced_modal_species_bridge(*args, **kwargs):
        import inspect as _inspect
        import os as _os
        from rabbit.transport.species_tagged_bridge import apply_species_tagged_bridge as _raw_species_bridge

        res = _orig_apply_reduced_modal_species_bridge(*args, **kwargs)

        min_cos = float(_os.environ.get("RABBIT_REDUCED_MODAL_MIN_COS", "0.90"))
        max_resid = float(_os.environ.get("RABBIT_REDUCED_MODAL_MAX_RESID", "0.20"))
        strict = _os.environ.get("RABBIT_REDUCED_MODAL_STRICT", "1").lower() not in ("0", "false", "no")

        dbg = getattr(res, "bridge_debug", None)
        cos_v = dbg.get("cos", None) if isinstance(dbg, dict) else None
        resid_v = dbg.get("resid", None) if isinstance(dbg, dict) else None

        fail = False
        if strict:
            if cos_v is None or resid_v is None:
                fail = True
            else:
                fail = (float(cos_v) < min_cos) or (float(resid_v) > max_resid)

        if fail:
            sig = _inspect.signature(_orig_apply_reduced_modal_species_bridge)
            bound = sig.bind_partial(*args, **kwargs)
            params = bound.arguments

            raw = _raw_species_bridge(
                species=params["species"],
                I=params["I"],
                J=params["J"],
                w0=params["w0"],
                q_nodes=params["q_nodes"],
                q_weights=params["q_weights"],
                T_gamma=params["T_gamma"],
                T_nu_e=params["T_nu_e"],
                T_nu_x=params["T_nu_x"],
                H=params["H"],
            )

            status = {
                "reduced_modal_status": "rejected_to_raw",
                "authoritative_path": "raw_characteristic",
                "reduced_modal_reject_reason": {
                    "cos": cos_v,
                    "resid": resid_v,
                    "min_cos": min_cos,
                    "max_resid": max_resid,
                },
            }

            raw.bridge_debug = dict(getattr(raw, "bridge_debug", {}) or {})
            raw.bridge_debug.update(status)
            raw.reduced_modal_debug = dict(raw.bridge_debug)
            return raw

        status = {
            "reduced_modal_status": "accepted_research_only",
            "authoritative_path": RAW_AUTHORITATIVE_PATH,
            "surrogate_path": "reduced_modal",
            "reduced_modal_mode": REDUCED_MODAL_MODE,
            "reduced_modal_reject_reason": None,
            "reduced_modal_gate": {
                "min_cos": min_cos,
                "max_resid": max_resid,
            },
        }

        if not isinstance(getattr(res, "bridge_debug", None), dict):
            res.bridge_debug = {}
        res.bridge_debug = dict(res.bridge_debug)
        res.bridge_debug.update(status)
        res.reduced_modal_debug = dict(res.bridge_debug)
        return res
