"""PR-T3B-PF #11: AP-form FLRW ``N_eff`` grid-convergence sweep.

Documents the convergence behaviour of the
``projected_physical_preflight`` AP-form collision mode as the
``(N_mu, N_q)`` quadrature grid is refined.  The locked finding:
the AP-form is **fully grid-converged** at all three grids
``(4, 6)``, ``(8, 12)`` and ``(12, 20)`` — the residual gap to
Mangano 2005's ``N_eff = 3.044`` is bounded by the model
approximation itself, not the quadrature resolution.

This is the calibration signal for canonical PR-T3B: the
``~0.013`` gap will only close when the AP-form is upgraded
(e.g., AP/IMEX hybrid with the full Hannestad-Madsen kernel for
the non-equilibrium correction, or higher-order asymptotic
expansion of the relaxation operator).  Increasing grid
resolution alone is not the path forward.

The ``spectral_relaxation_preflight`` mode is excluded from the
finer grids because its damping closure exceeds the bounded
``max_steps`` budget when the kernel becomes more resolved; this
is recorded as a separate documentation note rather than a test
regression.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def projected_physical_results() -> dict:
    """One projected_physical FLRW run per grid, shared across the
    parametrized convergence tests."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )

    results: dict = {}
    for (N_mu, N_q) in [(4, 6), (8, 12), (12, 20)]:
        cfg = JAXFullBoltzmannConfig(
            Sigma_H_plus=0.0,
            N_mu=N_mu,
            N_q=N_q,
            correction_level=0,
            n_reactions=12,
            collision_mode="projected_physical_preflight",
            thermo_tier=2,
            rtol=1e-6,
            atol=1e-8,
            max_steps=512,
            event_refine_steps=12,
        )
        r = run_full_boltzmann_jax(cfg)
        assert r.success, (
            f"projected_physical FLRW failed at (N_mu={N_mu}, N_q={N_q})"
        )
        results[(N_mu, N_q)] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
        }
    return results


@pytest.mark.parametrize("grid", [(4, 6), (8, 12), (12, 20)])
def test_projected_physical_grid_converged_n_eff(
    projected_physical_results: dict, grid: tuple
) -> None:
    """Each grid produces ``N_eff ≈ 3.0307`` within ``5e-3``.

    Locked baselines from measurement:
    * ``(4, 6)``  → ``N_eff = 3.030738``
    * ``(8, 12)`` → ``N_eff = 3.030755``
    * ``(12, 20)``→ ``N_eff = 3.030729``

    The variance across grids is ``< 3e-5``, far below the
    ``5e-3`` lock window — the AP-form is **fully grid-converged**.
    """
    n_eff = projected_physical_results[grid]["N_eff"]
    assert abs(n_eff - 3.0307) < 5e-3, (
        f"projected_physical at {grid}: N_eff = {n_eff:.6f} "
        f"drifted from baseline 3.0307"
    )


def test_projected_physical_grid_convergence_envelope(
    projected_physical_results: dict,
) -> None:
    """The spread across the three grids is bounded by ``1e-4``,
    which is the **calibration signal** for canonical PR-T3B:
    the ``~0.013`` gap to Mangano 2005's ``3.044`` is dominated by
    the AP-form *model* approximation, not the quadrature
    resolution.  Increasing grid resolution alone will not close
    the gap; the canonical fix must upgrade the AP-form
    (AP/IMEX hybrid with the Hannestad-Madsen kernel, or
    higher-order relaxation expansion)."""
    n_effs = [v["N_eff"] for v in projected_physical_results.values()]
    spread = max(n_effs) - min(n_effs)
    assert spread < 1e-4, (
        f"projected_physical N_eff spread across grids "
        f"((4,6) / (8,12) / (12,20)) widened: {spread:.6f}"
    )


def test_projected_physical_gap_to_mangano_is_model_dominated(
    projected_physical_results: dict,
) -> None:
    """At the most-resolved grid ``(12, 20)`` the gap to Mangano
    2005 (``N_eff = 3.044``) is locked at ``~0.013`` — the
    canonical-PR-T3B target reduction (``< 5e-3``)."""
    MANGANO = 3.044
    n_eff = projected_physical_results[(12, 20)]["N_eff"]
    gap = abs(n_eff - MANGANO)
    # Locked at ~0.013 (measured 0.01327)
    assert 0.012 < gap < 0.015, (
        f"projected_physical (12, 20) gap to Mangano drifted: "
        f"|N_eff - 3.044| = {gap:.4f}"
    )
