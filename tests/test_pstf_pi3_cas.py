"""tests/test_pstf_pi3_cas.py — gate the CAS-symbolic Pi^(3) against the numerical evaluator.

Plan B.F3: the offline CAS generator (scripts/cas/gen_pstf_pi3.py, system python3 + sympy)
reduces the generalized HM angular polynomial Pi^(3)(x,y) to an exact polynomial and emits
sample values (scripts/cas/pi3_validation_samples.json). This test cross-checks those symbolic
values against the venv's numerical collisions/pstf_body_frame.angular_polynomial_n3 (the
already-validated evaluator) to MACHINE PRECISION -- the gate that licenses replacing the
numerical theta-quadrature F^(3) with the analytic x-polynomial (PR#9).

The JSON is a generated, regenerable artifact: `python3 scripts/cas/gen_pstf_pi3.py --emit-samples`.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from rabbit.collisions.pstf_body_frame import collision_body_frame, angular_polynomial_n3
from rabbit.collisions.pstf_pi3_analytic import analytic_pi3, has_analytic_pi3

_SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cas" / "pi3_validation_samples.json"


def _load():
    if not _SAMPLES.exists():
        pytest.skip(f"CAS validation samples not generated: {_SAMPLES}")
    return json.loads(_SAMPLES.read_text())


def test_cas_pi3_matches_numerical_evaluator():
    """Symbolic Pi^(3) (CAS) == numerical angular_polynomial_n3 to machine precision."""
    samples = _load()
    assert len(samples) >= 12, "expected the representative gate-case sample set"
    worst = 0.0
    for s in samples:
        p1, p2, p3 = s["p"]
        frame = collision_body_frame(E1=p1, p1=p1, E2=p2, p2=p2, E3=p3, p3=p3,
                                     m4=0.0, x=s["x"], y=s["y"])
        assert frame.valid, f"sample frame invalid: {s}"
        numeric = angular_polynomial_n3(s["L"], s["J"], tuple(s["ranks"]), tuple(s["legs"]), frame)
        worst = max(worst, abs(numeric - s["value"]))
        assert numeric == pytest.approx(s["value"], abs=1e-11), (
            f"CAS vs numerical mismatch at {s}: numeric={numeric}, cas={s['value']}")
    assert worst < 1e-11


def test_cas_samples_cover_monopole_and_genuine_trilinear():
    """The gate set must include the all-monopole, two-monopole, and 234 tripolar cases."""
    legs = {tuple(s["legs"]) for s in _load()}
    assert (1, 2, 3) in legs        # all-monopole
    assert (2, 1, 1) in legs        # two-monopole reduction (leg 2 -> P_L(x))
    assert (2, 3, 4) in legs        # genuine tripolar kernel (no leg-1 simplification)


# cases mirrored from gen_pstf_pi3._GATE_CASES (the generated artifact)
_CASES = [
    (0, 0, (0, 0, 0), (1, 2, 3)), (1, 1, (1, 0, 0), (2, 1, 1)), (2, 2, (2, 0, 0), (2, 1, 1)),
    (1, 1, (1, 0, 0), (3, 1, 1)), (2, 2, (2, 0, 0), (4, 1, 1)), (1, 1, (1, 1, 1), (2, 3, 4)),
    (2, 2, (1, 1, 2), (2, 3, 4)), (2, 0, (1, 1, 2), (2, 3, 4)), (2, 2, (1, 1, 2), (1, 3, 4)),
]


def test_analytic_pi3_matches_numerical_on_random_grid():
    """The loaded CAS analytic evaluator == numerical angular_polynomial_n3 to machine
    precision across random valid (x,y) -- the full evaluator gate (beyond the stored samples)."""
    rng = np.random.default_rng(0)
    p1, p2, p3 = 1.2, 0.9, 0.8
    checked = 0
    for (L, J, ranks, legs) in _CASES:
        assert has_analytic_pi3(L, J, ranks, legs)
        for _ in range(8):
            x = float(rng.uniform(-0.7, 0.7))
            yv = float(rng.uniform(-0.7, 0.7))
            fr = collision_body_frame(E1=p1, p1=p1, E2=p2, p2=p2, E3=p3, p3=p3, m4=0.0, x=x, y=yv)
            if not fr.valid:
                continue
            num = angular_polynomial_n3(L, J, ranks, legs, fr)
            ana = analytic_pi3(L, J, ranks, legs, x, yv, p1, p2, p3)
            assert ana == pytest.approx(num, abs=1e-11), f"{(L,J,ranks,legs)} at x={x},y={yv}"
            checked += 1
    assert checked >= 40, "too few valid grid points sampled"
