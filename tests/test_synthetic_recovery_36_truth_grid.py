"""Frozen provenance for the 36-point recovery gate deferred to B-05.

The historical grid remains documented, but no coverage result is produced
through the non-traceable host-forward NUTS wrapper.  The test locks an
explicit RED boundary instead of skipping or fabricating a gradient.
"""

from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════
# §1. Truth grid (36 points = 4 × 3 × 3)
# ═══════════════════════════════════════════════════════════════════════

# Plan §4.2 truth grid (36 points)
_SIGMA_TRUTH_GRID = (0.0, 0.05, 0.10, 0.20)
_ETA10_TRUTH_GRID = (6.0, 6.104, 6.20)
_TAUN_TRUTH_GRID = (877.4, 878.4, 879.4)

TRUTH_GRID = [
    (s, e, t)
    for s in _SIGMA_TRUTH_GRID
    for e in _ETA10_TRUTH_GRID
    for t in _TAUN_TRUTH_GRID
]
assert len(TRUTH_GRID) == 36


# ═══════════════════════════════════════════════════════════════════════
# §2. Per-truth NUTS-3D recovery
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.expensive
@pytest.mark.slow
@pytest.mark.production
class TestSyntheticRecovery36:
    """The grid is specified but has no accepted sampler implementation."""

    def test_36_truth_coverage_gate_is_blocked_until_b05(self):
        from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE
        from rabbit.inference.joint_3d_inference import run_bbn_nuts_3d

        with pytest.raises(RuntimeError) as exc_info:
            run_bbn_nuts_3d()
        assert str(exc_info.value) == BBN_JAX_SAMPLER_UNAVAILABLE
