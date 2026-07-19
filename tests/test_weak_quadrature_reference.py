from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss

from rabbit.weak import live_rates
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.weak.quadrature import DEFAULT_WEAK_QUADRATURE, HIGHRES_WEAK_QUADRATURE
from rabbit.weak.reference_quadrature import (
    TrapezoidWeakReferenceConfig,
    compute_live_weak_rates_trapezoid_reference,
    scan_fd_weak_quadrature_ladder,
    weak_rate_quadrature_drift,
)


def _fd_monopoles(n_q: int = 20):
    q_nodes, _weights = laggauss(n_q)
    f0 = 1.0 / (np.exp(q_nodes) + 1.0)
    return q_nodes, f0


@pytest.mark.production
@pytest.mark.parametrize("correction_level", [0, 2, 3])
def test_dense_trapezoid_reference_agrees_with_highres_gauss(correction_level: int) -> None:
    q_nodes, f0 = _fd_monopoles()
    highres = compute_live_weak_rates(
        f0,
        f0,
        q_nodes,
        1.0,
        1.0,
        quad=HIGHRES_WEAK_QUADRATURE,
        compute_iso_reference=False,
        correction_level=correction_level,
        use_exact_fd=True,
    )
    reference = compute_live_weak_rates_trapezoid_reference(
        f0,
        f0,
        q_nodes,
        1.0,
        1.0,
        correction_level=correction_level,
        config=TrapezoidWeakReferenceConfig(n_semi_infinite=1000, n_bounded=1000),
        use_exact_fd=True,
    )

    drift = weak_rate_quadrature_drift(highres, reference)
    assert drift["lambda_np_rel"] < 2.0e-5
    assert drift["lambda_pn_rel"] < 2.0e-5
    assert drift["lambda_ratio_rel"] < 2.0e-8


@pytest.mark.production
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_production_gauss_laguerre_is_not_current_weak_rate_bottleneck(
    temperature: float,
) -> None:
    q_nodes, f0 = _fd_monopoles()
    production = compute_live_weak_rates(
        f0,
        f0,
        q_nodes,
        temperature,
        temperature,
        quad=DEFAULT_WEAK_QUADRATURE,
        compute_iso_reference=False,
        correction_level=3,
        use_exact_fd=True,
    )
    validation = compute_live_weak_rates(
        f0,
        f0,
        q_nodes,
        temperature,
        temperature,
        quad=HIGHRES_WEAK_QUADRATURE,
        compute_iso_reference=False,
        correction_level=3,
        use_exact_fd=True,
    )
    reference = compute_live_weak_rates_trapezoid_reference(
        f0,
        f0,
        q_nodes,
        temperature,
        temperature,
        correction_level=3,
        use_exact_fd=True,
    )

    prod_vs_validation = weak_rate_quadrature_drift(production, validation)
    validation_vs_trapz = weak_rate_quadrature_drift(validation, reference)
    assert prod_vs_validation["lambda_np_rel"] < 1.0e-5
    assert prod_vs_validation["lambda_pn_rel"] < 1.0e-5
    assert validation_vs_trapz["lambda_np_rel"] < 3.0e-5
    assert validation_vs_trapz["lambda_pn_rel"] < 3.0e-5


def test_trapezoid_reference_iso_diagnostics_recover_fd_null() -> None:
    q_nodes, f0 = _fd_monopoles()
    result = compute_live_weak_rates_trapezoid_reference(
        f0,
        f0,
        q_nodes,
        1.0,
        1.0,
        correction_level=3,
        compute_iso_reference=True,
        use_exact_fd=True,
    )

    assert result.lambda_np == pytest.approx(result.lambda_np_iso, rel=0.0, abs=1.0e-15)
    assert result.lambda_pn == pytest.approx(result.lambda_pn_iso, rel=0.0, abs=1.0e-15)
    assert result.delta_np == pytest.approx(0.0, abs=1.0e-15)
    assert result.delta_pn == pytest.approx(0.0, abs=1.0e-15)


def test_live_weak_channel_kernel_cache_uses_compact_quadrature_signature() -> None:
    q_nodes, f0 = _fd_monopoles()
    live_rates._weak_kernel_cache.clear()

    compute_live_weak_rates(
        f0,
        f0,
        q_nodes,
        1.0,
        1.0,
        quad=DEFAULT_WEAK_QUADRATURE,
        compute_iso_reference=False,
        correction_level=3,
        use_exact_fd=True,
    )

    assert live_rates._weak_kernel_cache
    for key in live_rates._weak_kernel_cache:
        assert all(not isinstance(part, (bytes, bytearray)) for part in key)


def test_live_weak_bounded_channel_geometry_is_reused_across_temperatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_nodes, f0 = _fd_monopoles()
    live_rates._weak_kernel_cache.clear()
    if hasattr(live_rates, "_weak_static_channel_cache"):
        live_rates._weak_static_channel_cache.clear()
    original_coulomb = live_rates.coulomb_correction_per_channel
    channel_calls: dict[str, int] = {}

    def counting_coulomb(eps: np.ndarray, channel: str) -> np.ndarray:
        channel_calls[channel] = channel_calls.get(channel, 0) + 1
        return original_coulomb(eps, channel)

    monkeypatch.setattr(live_rates, "coulomb_correction_per_channel", counting_coulomb)

    for temperature in (0.9, 1.1):
        compute_live_weak_rates(
            f0,
            f0,
            q_nodes,
            temperature,
            temperature,
            quad=DEFAULT_WEAK_QUADRATURE,
            compute_iso_reference=False,
            correction_level=3,
            use_exact_fd=True,
        )

    assert channel_calls["c"] == 1
    assert channel_calls["f"] == 1


def test_fd_quadrature_ladder_scan_reports_gl_and_trapz_drifts() -> None:
    records = scan_fd_weak_quadrature_ladder(
        [0.7, 1.0],
        correction_level=3,
        reference_config=TrapezoidWeakReferenceConfig(
            n_semi_infinite=1000,
            n_bounded=1000,
        ),
    )

    assert [row["T_gamma_MeV"] for row in records] == [0.7, 1.0]
    for row in records:
        assert row["lambda_np_gl32"] > 0.0
        assert row["lambda_pn_gl32"] > 0.0
        assert row["gl32_vs_gl64_lambda_np_rel"] < 1.0e-5
        assert row["gl32_vs_gl64_lambda_pn_rel"] < 1.0e-5
        assert row["gl64_vs_trapz_lambda_np_rel"] < 3.0e-5
        assert row["gl64_vs_trapz_lambda_pn_rel"] < 3.0e-5
