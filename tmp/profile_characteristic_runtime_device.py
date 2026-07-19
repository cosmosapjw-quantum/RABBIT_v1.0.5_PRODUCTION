from __future__ import annotations

import argparse
import json
import time

import jax

from rabbit.jax.driver_typeI_char import JAXTypeICharConfig, run_full_coupled_typeI_char_jax


def _build_config(case: str, policy: str) -> JAXTypeICharConfig:
    base = dict(
        correction_level=2,
        N_mu=12,
        N_q=20,
        n_reactions=12,
        runtime_device_policy=policy,
    )
    if case == "tier1_flrw":
        return JAXTypeICharConfig(
            Sigma_H_plus=0.0,
            thermo_tier=1,
            **base,
        )
    if case == "tier1_shear":
        return JAXTypeICharConfig(
            Sigma_H_plus=2.0e-3,
            thermo_tier=1,
            **base,
        )
    if case == "tier2_shear":
        return JAXTypeICharConfig(
            Sigma_H_plus=2.0e-3,
            thermo_tier=2,
            **base,
        )
    raise ValueError(f"Unknown case: {case}")


def _run_once(case: str, policy: str) -> dict[str, object]:
    cfg = _build_config(case, policy)
    started = time.perf_counter()
    result = run_full_coupled_typeI_char_jax(cfg)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "runtime_device_initial_platform": result.metadata.get("runtime_device_initial_platform"),
        "runtime_device_final_platform": result.metadata.get("runtime_device_final_platform"),
        "runtime_device_fallback_applied": result.metadata.get("runtime_device_fallback_applied"),
        "runtime_device_policy": result.metadata.get("runtime_device_policy"),
        "runtime_device_contract": result.metadata.get("runtime_device_contract"),
        "n_steps_p1": result.n_steps_p1,
        "n_steps_p2": result.n_steps_p2,
        "Yp": result.Yp,
        "N_eff": result.N_eff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--policy", action="append", dest="policies")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cases = args.cases or ["tier1_flrw", "tier1_shear", "tier2_shear"]
    policies = args.policies or ["cpu_preferred", "gpu_then_cpu_retry"]

    payload: dict[str, object] = {
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "results": {},
    }
    for case in cases:
        case_results: dict[str, object] = {}
        for policy in policies:
            case_results[policy] = [_run_once(case, policy) for _ in range(int(args.repeat))]
        payload["results"][case] = case_results
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
