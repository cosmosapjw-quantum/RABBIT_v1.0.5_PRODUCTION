from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

import rabbit.collisions.dynamic_collision_driver as driver


def fake_solve(_fun, t_span, y0, **_kwargs):
    states = np.column_stack([y0, y0.copy()])
    states[-1, -1] = 1.0e-2
    times = np.array([t_span[0], 5.0])
    return SimpleNamespace(
        success=True,
        message="fake terminal success",
        t=times,
        y=states,
        t_events=[times[-1:]],
    )


driver.solve_ivp = fake_solve
result = driver.integrate_flrw_decoupling(n_q=8, collisions=False)

payload = {}
for name in (
    "N_eff",
    "T_nu_e_over_gamma",
    "T_nu_x_over_gamma",
    "z_final",
    "T_gamma_final",
    "N_final",
    "max_clip_excursion",
    "min_hubble_MeV",
):
    payload[name] = float(getattr(result, name)).hex()
payload.update(
    reached_endpoint=result.reached_endpoint,
    solver_success=result.solver_success,
    solver_message=result.solver_message,
)
for name in ("f_nue_final", "f_nux_final"):
    value = np.asarray(getattr(result, name))
    payload[name] = {
        "dtype": value.dtype.str,
        "shape": value.shape,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
