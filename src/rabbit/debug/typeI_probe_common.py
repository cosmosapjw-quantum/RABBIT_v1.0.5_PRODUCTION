from __future__ import annotations
import numpy as np

def monopole_debug_packet(q_nodes, q_weights, f):
    q = np.asarray(q_nodes, dtype=float)
    w = np.asarray(q_weights, dtype=float)
    ff = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)

    m0 = float(np.sum(w * ff))
    m1 = float(np.sum(w * q * ff))
    m2 = float(np.sum(w * q**2 * ff))
    m3 = float(np.sum(w * q**3 * ff))
    mean_q = m1 / max(m0, 1.0e-300)

    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "mean_q": float(mean_q),
        "f_min": float(np.min(ff)),
        "f_max": float(np.max(ff)),
    }
