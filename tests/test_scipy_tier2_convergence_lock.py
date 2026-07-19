"""BD621 — falsification lock for scipy tier-2 resolution convergence.

The scipy Type-I tier-2 per-species surface (and the tier-3 weak-budget surface
on top of it) is a *candidate*, not canonical, because its anisotropic-residual
collision closure is NOT resolution-converged. This is the internal-consistency
gate that blocks promotion: a canonical BBN surface needs Y_p stable to ~1e-4,
but tier-2 drifts ~3.5% in Y_p from N_q=16 to N_q=32 with no sign of settling
(the N_q=32 step is larger than the N_q=24 step). See
docs/audit/BD621_scipy_tier2_tier3_promotion_internal_consistency_2026-07-09.md.

This is a strict-xfail falsification lock (BD612 pattern): it asserts the
convergence a canonical surface must have. It fails today (residual closure not
converged) and will flip green the moment the closure is made resolution-stable,
which is the precondition for promoting these surfaces to canonical.
"""
from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="BD621: scipy tier-2 residual collision closure is not resolution-"
    "converged (~3.5% Y_p drift N_q=16->32); blocks canonical promotion.",
)
def test_scipy_tier2_yp_resolution_converged():
    def yp(nq):
        return canonical_forward_solver(
            Sigma_H=0.03, backend="scipy", correction_level=3,
            N_q=nq, tier=2, enable_collisions=True,
        ).Yp

    yp24, yp32 = yp(24), yp(32)
    rel = abs(yp32 - yp24) / yp24
    # a canonical surface must be converged to <1e-3 between N_q=24 and 32.
    assert rel < 1.0e-3, f"tier-2 Y_p not converged: |dYp/Yp|(24->32)={rel:.2e}"
