"""tests/test_li7h_identifiability.py — S1 (break the BD599 INF-1 rank ceiling).

With only (Y_p, D/H) the BBN Fisher is rank <= 2 for 3 parameters, so Σ_H is never
data-identified. Adding Li7/H as an opt-in 3rd observable lifts the Fisher to rank
3 off-null (cond ~1e5), making Σ_H formally identifiable there. Li7/H is opt-in,
NOT a default constraint — the unresolved lithium problem forbids a face-value Li7
likelihood (see the S1 spec / module caveats).
"""

from __future__ import annotations

import pytest

from rabbit.inference.fisher import bbn_fisher
from rabbit.inference.joint_3d_inference import (
    JointInference3DConfig,
    sigma_h_identifiability,
)


@pytest.mark.slow
def test_default_two_observable_is_rank2_offnull():
    """Without Li7/H the off-null Fisher stays rank 2 (the INF-1 ceiling)."""
    _, diag = bbn_fisher(sigma_H_fid=0.2, include_li7h=False)
    assert diag.rank == 2


@pytest.mark.slow
def test_li7h_lifts_fisher_to_rank3_offnull():
    """Li7/H as a 3rd observable lifts the off-null Fisher to rank 3, well-conditioned."""
    _, diag = bbn_fisher(sigma_H_fid=0.2, include_li7h=True)
    assert diag.rank == 3
    assert diag.condition_number < 1.0e8


@pytest.mark.slow
def test_identifiability_informative_reachable_with_li7h_offnull():
    """With Li7/H the identifiability diagnostic can report Σ_H informative off-null."""
    cfg = JointInference3DConfig(backend="scipy", N_q=20, correction_level=2)
    d = sigma_h_identifiability(cfg, sigma_H_fid=0.2, include_li7h=True)
    assert d.degenerate is False
    assert d.informative is True
