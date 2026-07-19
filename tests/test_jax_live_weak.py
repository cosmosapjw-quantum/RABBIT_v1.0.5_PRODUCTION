from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX required")

import jax.numpy as jnp
from numpy.polynomial.laguerre import laggauss

from rabbit.jax.weak_live_jax import (
    LIVE_WEAK_FAST_CANDIDATE_QUADRATURE_SPEC,
    LIVE_WEAK_QUADRATURE_SPEC,
    LIVE_WEAK_VALIDATION_QUADRATURE_SPEC,
    compute_live_born_rates_from_monopoles,
    compute_live_rates_from_monopoles_level_specialized_jax,
)
from rabbit.jax.weak_live_fused import (
    compute_live_born_rates_fused,
    compute_live_rates_from_monopoles_fused_jax,
    compute_live_rates_from_shared_monopole_cl0_fused_jax,
    compute_live_rates_from_shared_monopole_cl1_fused_jax,
    compute_live_rates_from_shared_monopole_cl2_fused_jax,
    compute_live_rates_from_shared_monopole_cl3_fused_jax,
)
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.weak.quadrature import (
    DEFAULT_WEAK_QUADRATURE,
    FAST_INFERENCE_WEAK_QUADRATURE,
    HIGHRES_WEAK_QUADRATURE,
    weak_quadrature_for_mode,
)
from rabbit.jax.rhs_typeI import extract_monopole_distributions


def test_live_weak_quadrature_specs_are_locked_to_shared_registry():
    assert LIVE_WEAK_QUADRATURE_SPEC.mode == "production"
    assert LIVE_WEAK_QUADRATURE_SPEC.N_laguerre == DEFAULT_WEAK_QUADRATURE.N_laguerre == 32
    assert LIVE_WEAK_QUADRATURE_SPEC.N_legendre == DEFAULT_WEAK_QUADRATURE.N_legendre == 32
    assert LIVE_WEAK_VALIDATION_QUADRATURE_SPEC.N_laguerre == HIGHRES_WEAK_QUADRATURE.N_laguerre == 64
    assert LIVE_WEAK_VALIDATION_QUADRATURE_SPEC.N_legendre == HIGHRES_WEAK_QUADRATURE.N_legendre == 64
    assert LIVE_WEAK_FAST_CANDIDATE_QUADRATURE_SPEC.N_laguerre == FAST_INFERENCE_WEAK_QUADRATURE.N_laguerre == 24
    assert weak_quadrature_for_mode("production") == DEFAULT_WEAK_QUADRATURE
    assert weak_quadrature_for_mode("validation") == HIGHRES_WEAK_QUADRATURE
    assert weak_quadrature_for_mode("inference_fast") == FAST_INFERENCE_WEAK_QUADRATURE


@pytest.mark.parametrize("N_q", [6, 12])
def test_live_weak_matches_scipy_live_rates_when_monopoles_are_fd(N_q):
    q_nodes_np, _ = laggauss(N_q)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)

    lnp_live, lpn_live = compute_live_born_rates_from_monopoles(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0
    )
    scipy_ref = compute_live_weak_rates(
        np.array(f0), np.array(f0), np.array(q_nodes), 1.0, 1.0, 878.4,
        compute_iso_reference=False, correction_level=0,
    )

    # Stable CL0 live-weak candidate: narrow parity only in the explicitly
    # locked N_q={6,12} FD window. Full matched-physics parity remains future work.
    assert abs(float(lnp_live) - scipy_ref.lambda_np) / scipy_ref.lambda_np < 1.0e-10
    assert abs(float(lpn_live) - scipy_ref.lambda_pn) / scipy_ref.lambda_pn < 1.0e-10




@pytest.mark.parametrize("N_q", [6, 8, 10, 12, 20])
@pytest.mark.parametrize("T", [10.0, 3.0, 1.0, 0.8])
def test_live_weak_fd_reconstruction_matches_direct_born_rates(N_q, T):
    from rabbit.jax.weak_jax import compute_born_rates

    q_nodes_np, _ = laggauss(N_q)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)

    lnp_live, lpn_live = compute_live_born_rates_from_monopoles(
        jnp.array(T), jnp.array(T), jnp.array(878.4), q_nodes, f0, f0
    )
    lnp_ref, lpn_ref = compute_born_rates(jnp.array(T), jnp.array(T), jnp.array(878.4))

    rel_np = abs(float(lnp_live) - float(lnp_ref)) / float(lnp_ref)
    rel_pn = abs(float(lpn_live) - float(lpn_ref)) / float(lpn_ref)
    assert rel_np < 1.0e-3
    assert rel_pn < 1.0e-3


@pytest.mark.parametrize("N_q", [6, 12, 20])
def test_fused_live_born_rates_match_unfused_logit_residual_path(N_q):
    q_nodes_np, _ = laggauss(N_q)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    distorted = jnp.clip(f0 * (1.0 + 0.03 * jnp.exp(-q_nodes / 2.0)), 0.0, 1.0)

    ref = compute_live_born_rates_from_monopoles(
        jnp.array(1.0), jnp.array(0.97), jnp.array(878.4), q_nodes, distorted, f0
    )
    fused = compute_live_born_rates_fused(
        jnp.array(1.0), jnp.array(0.97), jnp.array(878.4), q_nodes, distorted, f0
    )

    np.testing.assert_allclose(
        np.asarray(jnp.stack(fused)),
        np.asarray(jnp.stack(ref)),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("correction_level", [0, 1, 2, 3])
def test_fused_live_weak_cl0_cl3_match_unfused_level_specialized_path(correction_level):
    q_nodes_np, _ = laggauss(20)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    f_nue = jnp.clip(f0 * (1.0 + 0.02 * jnp.exp(-q_nodes / 2.0)), 0.0, 1.0)
    f_nuebar = jnp.clip(f0 * (1.0 - 0.015 * jnp.exp(-q_nodes / 2.5)), 0.0, 1.0)

    ref = compute_live_rates_from_monopoles_level_specialized_jax(
        jnp.array(1.0),
        jnp.array(0.97),
        jnp.array(878.4),
        q_nodes,
        f_nue,
        f_nuebar,
        correction_level=correction_level,
    )
    fused = compute_live_rates_from_monopoles_fused_jax(
        jnp.array(1.0),
        jnp.array(0.97),
        jnp.array(878.4),
        q_nodes,
        f_nue,
        f_nuebar,
        correction_level=correction_level,
    )

    np.testing.assert_allclose(
        np.asarray(jnp.stack(fused)),
        np.asarray(jnp.stack(ref)),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    "correction_level,kernel",
    [
        (0, compute_live_rates_from_shared_monopole_cl0_fused_jax),
        (1, compute_live_rates_from_shared_monopole_cl1_fused_jax),
        (2, compute_live_rates_from_shared_monopole_cl2_fused_jax),
        (3, compute_live_rates_from_shared_monopole_cl3_fused_jax),
    ],
)
def test_fused_shared_live_weak_matches_two_monopole_fused_path(correction_level, kernel):
    q_nodes_np, _ = laggauss(20)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    distorted = jnp.clip(f0 * (1.0 + 0.01 * jnp.sin(q_nodes) * jnp.exp(-q_nodes / 3.0)), 0.0, 1.0)

    shared = kernel(
        jnp.array(1.0), jnp.array(0.97), jnp.array(878.4), q_nodes, distorted
    )
    two_species = compute_live_rates_from_monopoles_fused_jax(
        jnp.array(1.0),
        jnp.array(0.97),
        jnp.array(878.4),
        q_nodes,
        distorted,
        distorted,
        correction_level=correction_level,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.stack(shared)),
        np.asarray(jnp.stack(two_species)),
        rtol=0.0,
        atol=0.0,
    )


def test_live_weak_responds_to_monopole_distortion():
    q_nodes_np, _ = laggauss(20)
    q_nodes = jnp.array(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    distorted = jnp.clip(f0 * (1.0 + 0.05 * jnp.exp(-q_nodes / 3.0)), 0.0, 1.0)

    lnp_base, lpn_base = compute_live_born_rates_from_monopoles(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0
    )
    lnp_dist, lpn_dist = compute_live_born_rates_from_monopoles(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, distorted, f0
    )

    assert float(lnp_dist) > float(lnp_base)
    assert float(lpn_dist) >= 0.0


def test_extract_monopole_distributions_returns_physical_occupations():
    q_nodes_np, _ = laggauss(6)
    q_nodes = jnp.array(q_nodes_np)
    psi_flat = jnp.zeros(6 * 2 * 6)
    f_nue, f_nuebar = extract_monopole_distributions(psi_flat, 6, 2, q_nodes)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    assert np.allclose(np.array(f_nue), np.array(f0))
    assert np.allclose(np.array(f_nuebar), np.array(f0))
