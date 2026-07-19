"""Profile tier-3 full-Boltzmann RHS/Jacobian kernels.

The full-Boltzmann driver is CPU-locked by default to preserve the public
runtime contract.  Pass ``--allow-accelerator`` to benchmark private kernels on
the default JAX accelerator backend before importing the driver module.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any


def _block_until_ready(x: Any) -> None:
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x:
            _block_until_ready(item)
    elif hasattr(x, "__dict__"):
        for item in x.__dict__.values():
            _block_until_ready(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-accelerator", action="store_true")
    parser.add_argument("--N-mu", type=int, default=4)
    parser.add_argument("--N-q", type=int, default=6)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Collision mode to profile. May be repeated. Defaults to the tier-3 preflight set.",
    )
    args = parser.parse_args()

    os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")
    if args.allow_accelerator:
        os.environ["RABBIT_FULL_BOLTZMANN_ALLOW_ACCELERATOR"] = "1"

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from rabbit.jax.driver_typeI_full_boltzmann import (
        _get_full_boltzmann_rhs,
        _initial_transport_state,
    )

    modes = args.mode or [
        "collisionless",
        "ap_unified_preflight",
        "ap_unified_nu_nu_preflight",
        "ap_unified_nu_nu_spectral_preflight",
        "ap_unified_nu_nu_spectral_accuracy_preflight",
    ]

    def make_state(layout: dict) -> Any:
        y = np.zeros(layout["n_total"], dtype=np.float64)
        y[layout["i_Sp"]] = 0.03
        y[layout["i_S"]] = 0.01
        i0 = layout["i_transport"]
        i1 = i0 + layout["n_transport"]
        y[i0:i1] = _initial_transport_state(args.N_mu, args.N_q).reshape(-1)
        y[layout["i_tg"]] = 3.0
        y[layout["i_tne"]] = 2.9
        y[layout["i_tnx"]] = 2.8
        y[layout["i_net"] : layout["i_net"] + 2] = [0.13, 0.87]
        return jnp.asarray(y)

    rows = []
    for mode in modes:
        t0 = time.perf_counter()
        rhs_fn, jac_fn, layout, *_ = _get_full_boltzmann_rhs(
            phase=1,
            correction_level=0,
            collision_mode=mode,
            collision_preflight_relaxation=1.0,
            thermo_tier=2,
            N_mu=args.N_mu,
            N_q=args.N_q,
            n_network_species=2,
            n_reactions=12,
            tau_n=878.4,
            eta=6.104e-10,
            N_eff=3.044,
            f_nu=0.40520,
        )
        y = make_state(layout)
        build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        rhs_out = rhs_fn(jnp.asarray(0.0), y)
        _block_until_ready(rhs_out)
        rhs_compile_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        jac_out = jac_fn(jnp.asarray(0.0), y)
        _block_until_ready(jac_out)
        jac_compile_s = time.perf_counter() - t0

        for _ in range(5):
            _block_until_ready(rhs_fn(jnp.asarray(0.0), y))
            _block_until_ready(jac_fn(jnp.asarray(0.0), y))

        t0 = time.perf_counter()
        for _ in range(args.reps):
            rhs_out = rhs_fn(jnp.asarray(0.0), y)
        _block_until_ready(rhs_out)
        rhs_s = (time.perf_counter() - t0) / max(int(args.reps), 1)

        t0 = time.perf_counter()
        for _ in range(args.reps):
            jac_out = jac_fn(jnp.asarray(0.0), y)
        _block_until_ready(jac_out)
        jac_s = (time.perf_counter() - t0) / max(int(args.reps), 1)

        rows.append(
            {
                "mode": mode,
                "N_mu": int(args.N_mu),
                "N_q": int(args.N_q),
                "state_dim": int(layout["n_total"]),
                "build_s": build_s,
                "rhs_compile_s": rhs_compile_s,
                "jac_compile_s": jac_compile_s,
                "rhs_s": rhs_s,
                "jac_s": jac_s,
            }
        )

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "allow_accelerator": bool(args.allow_accelerator),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
