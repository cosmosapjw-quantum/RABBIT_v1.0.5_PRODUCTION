import numpy as np
from scipy.special import roots_laguerre

from rabbit.weak.live_rates import compute_live_weak_rates


def test_scipy_cl3_is_not_cl2_noop():
    q, _ = roots_laguerre(6)
    f = 1.0 / (np.exp(q) + 1.0)
    r2 = compute_live_weak_rates(f, f, q, 1.0, 1.0, correction_level=2)
    r3 = compute_live_weak_rates(f, f, q, 1.0, 1.0, correction_level=3)
    assert abs(r3.lambda_np - r2.lambda_np) > 0.0
    assert abs(r3.lambda_pn - r2.lambda_pn) > 0.0


def test_scipy_cl3_direction_matches_fm_expectation():
    q, _ = roots_laguerre(6)
    f = 1.0 / (np.exp(q) + 1.0)
    r2 = compute_live_weak_rates(f, f, q, 1.0, 1.0, correction_level=2)
    r3 = compute_live_weak_rates(f, f, q, 1.0, 1.0, correction_level=3)
    assert r3.lambda_np < r2.lambda_np
    assert r3.lambda_pn < r2.lambda_pn
