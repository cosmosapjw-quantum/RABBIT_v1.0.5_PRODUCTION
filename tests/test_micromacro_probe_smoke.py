import numpy as np

from rabbit.debug.micromacro_probe import analyze_collision_monopole


def test_micromacro_probe_finite():
    q = np.linspace(0.5, 20.0, 20)
    w = np.ones_like(q) / len(q)
    C = np.exp(-q) * (q - 3.0)

    out = analyze_collision_monopole(C, q, w)

    assert np.isfinite(out.raw_qdot)
    assert np.isfinite(out.proj_qdot_T)
    assert np.isfinite(out.proj_qdot_Tmu)
    assert np.isfinite(out.orth_qdot_T)
    assert np.isfinite(out.orth_qdot_Tmu)
    assert 0.0 <= out.tail_frac_raw_last3 <= 1.0
    assert 0.0 <= out.tail_frac_orthT_last3 <= 1.0
    assert 0.0 <= out.tail_frac_orthTmu_last3 <= 1.0
