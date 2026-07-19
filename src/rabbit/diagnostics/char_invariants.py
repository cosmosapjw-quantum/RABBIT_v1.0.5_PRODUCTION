from __future__ import annotations
import numpy as np

def _arr(x):
    return np.asarray(x, dtype=np.float64)

def weighted_energy_terms(q_nodes, q_weights, C):
    q = _arr(q_nodes)
    w = _arr(q_weights)
    c = _arr(C)
    return w * np.exp(q) * q**3 * c

def qdot_energy(q_nodes, q_weights, C):
    return float(np.sum(weighted_energy_terms(q_nodes, q_weights, C)))

def tail_fraction(q_nodes, q_weights, C, last_k=3):
    t = weighted_energy_terms(q_nodes, q_weights, C)
    denom = float(np.sum(np.abs(t)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(t[-last_k:])) / denom)

def sign_flip_count(x):
    a = _arr(x)
    s = np.sign(a)
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] * s[:-1] < 0))

def topk_terms(q_nodes, q_weights, C, k=8):
    q = _arr(q_nodes)
    w = _arr(q_weights)
    c = _arr(C)
    t = weighted_energy_terms(q, w, c)
    idx = np.argsort(np.abs(t))[::-1][:k]
    return [
        {
            "i": int(i),
            "q": float(q[i]),
            "w": float(w[i]),
            "C": float(c[i]),
            "term": float(t[i]),
        }
        for i in idx
    ]

def summarize_profile(q_nodes, q_weights, C, last_k=3, topk=8):
    c = _arr(C)
    return {
        "qdot_energy": qdot_energy(q_nodes, q_weights, c),
        "tail_fraction_lastk": tail_fraction(q_nodes, q_weights, c, last_k=last_k),
        "C_norm": float(np.linalg.norm(c)),
        "sign_flip_count": sign_flip_count(c),
        "topk_terms": topk_terms(q_nodes, q_weights, c, k=topk),
    }
