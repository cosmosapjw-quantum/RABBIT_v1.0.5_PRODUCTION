"""tests/test_sigma_h_identifiability.py — BD598 PR-C (D-R1/D-R2/D-R5).

The BBN likelihood is rank-deficient in the Σ_H direction because Y_p enters the
Friedmann equation as Σ_H². This pins the honest contract:

  * the *robust* identifiability signal is the likelihood rank / condition
    number, NOT the central-FD marginal (which is ε-noise for an even response
    at the null — D-R5);
  * under default BBN observational errors the data does not independently
    identify Σ_H from (η, τ_n) even off the null, so the runtime flag must say
    so rather than letting a prior-dominated marginal read as data-driven.

The flag *logic* (when it would say "informative") is unit-tested directly so the
positive branch is covered without an expensive forward solve.
"""

from __future__ import annotations

import pytest

from rabbit.inference.joint_3d_inference import (
    JointInference3DConfig,
    sigma_h_identifiability,
    _sigma_h_informative,
)


# ── flag logic (fast, no forward solve) ──────────────────────────────────

def test_informative_requires_rank3_and_shrinkage():
    """The positive branch fires only with full rank AND a tightened marginal."""
    assert _sigma_h_informative(rank_likelihood=3, shrinkage=0.5) is True
    # rank-deficient: never informative regardless of (noisy) shrinkage
    assert _sigma_h_informative(rank_likelihood=2, shrinkage=1e-7) is False
    # full rank but marginal ≈ prior: not data-driven
    assert _sigma_h_informative(rank_likelihood=3, shrinkage=0.95) is False


# ── real likelihood at the FLRW null (forward solves) ────────────────────

@pytest.fixture(scope="module")
def flrw_null_diag():
    cfg = JointInference3DConfig(backend="scipy", N_q=20, correction_level=2)
    return sigma_h_identifiability(cfg, sigma_H_fid=0.0)


def test_flrw_null_likelihood_is_rank_deficient(flrw_null_diag):
    """At Σ_H=0 the likelihood-only Fisher loses the Σ_H direction (rank 2)."""
    assert flrw_null_diag.rank_likelihood == 2
    assert flrw_null_diag.degenerate is True


def test_flrw_null_sigma_h_is_not_informative(flrw_null_diag):
    """At the null Σ_H is not data-identified — informative must be False.

    The huge likelihood condition number is the fingerprint of the degeneracy;
    posterior_sigma here is a central-FD artifact (D-R5), not a data constraint.
    """
    d = flrw_null_diag
    assert d.informative is False
    assert d.cond_likelihood > 1.0e8


@pytest.mark.slow
def test_bbn_does_not_independently_identify_sigma_h_off_null():
    """Honest finding: even at Σ_H=0.25, BBN (Yp,DH) under default obs errors
    does not independently identify Σ_H from (η, τ_n) — the likelihood stays
    rank-deficient, so the runtime flag correctly refuses to call it informative.
    Demonstrating an informative regime needs the robust-Fisher work (D-R5),
    deferred.
    """
    cfg = JointInference3DConfig(backend="scipy", N_q=20, correction_level=2)
    d = sigma_h_identifiability(cfg, sigma_H_fid=0.25)
    assert d.degenerate is True
    assert d.informative is False
