"""PR-T3 full CL envelope lock for the tier-3 AP-unified candidate.

The PR-T3B milestone landed at CL0; this regression locks the entire
CL ladder (CL0, CL1, CL2, CL3) on the frozen local full-Boltzmann
component oracle. F06 retired its former public dispatch.

All measurements use the frozen component defaults (``rtol=1e-8``,
``atol=1e-10``, ``max_steps=2000``, ``event_refine_steps=24``) so
the direct ``run_full_boltzmann_jax`` component call is reproducible.

Locks (η = 6.104e-10, τ_n = 878.4 s, FLRW unless otherwise noted,
``collision_mode='ap_unified_preflight'``, ``thermo_tier=2``):

§1. **Per-CL FLRW + anisotropic envelope at (N_mu=4, N_q=6).**
    Each (CL, σ) cell is locked to its measured Y_p anchor.

§2. **Per-CL grid scaling.**  Each CL must pass the canonical
    PR-T3D §5 grid gate ``< 1e-4`` across
    ``(N_mu, N_q) ∈ {(4, 6), (8, 12), (12, 20)}``.

§3. **Per-CL anisotropy stability.**  Each CL must pass the
    canonical PR-T3D §5 anisotropy gate ``< 1e-3`` across
    ``Σ_H ∈ {0, 0.05, 0.10}``.

§4. **CL ordering.**  Y_p must increase monotonically with the
    correction level: CL0 < CL1 < CL2 < CL3 at every σ.

§5. **N_eff is CL-invariant (FLRW).**  All CL collapse onto the
    AP-form FLRW ``N_eff ≈ 3.0345`` benchmark; the documented
    Mangano gap is the same for every CL.

§6. **Authority fence.** The former ``jax_ap_unified_tier3`` public
    endpoint is rejected, while the full-Boltzmann component capability
    remains catalogued as a frozen local oracle.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jax")


# ───────────────────────────────────────────────────────────────────
# Gold-locked envelope (captured 2026-04-29 from the canonical
# AP-unified surface at dispatch-default tolerances; tolerance
# accommodates cross-platform float reduction noise).
# ───────────────────────────────────────────────────────────────────

# (CL, σ) → measured Y_p at (N_mu=4, N_q=6) at dispatch defaults.
GOLD_FLRW_PLUS_ANISO = {
    (0, 0.00): 0.2413912333,
    (0, 0.05): 0.2376974663,
    (0, 0.10): 0.2334759957,
    (1, 0.00): 0.2452896005,
    (1, 0.05): 0.2403693063,
    (1, 0.10): 0.2351073017,
    (2, 0.00): 0.2469861143,
    (2, 0.05): 0.2420543272,
    (2, 0.10): 0.2367810324,
    (3, 0.00): 0.2474967478,
    (3, 0.05): 0.2425466634,
    (3, 0.10): 0.2372550219,
}

GRIDS = [(4, 6), (8, 12), (12, 20)]

YP_TOL = 5e-4
N_EFF_TOL_VS_BENCHMARK = 5e-4
GRID_GATE = 1e-4
ANISO_GATE = 1e-3
N_EFF_BENCHMARK = 3.0345  # AP-unified FLRW (Mangano gap +0.0095 documented)


@pytest.fixture(scope="module")
def jax_setup():
    import jax
    jax.config.update("jax_enable_x64", True)


def _solve_full_boltzmann(cl: int, sigma: float, N_mu: int, N_q: int):
    """Direct ``run_full_boltzmann_jax`` call with the same default
    tolerances the public dispatch uses."""
    from rabbit.jax.driver_typeI_full_boltzmann import (
        JAXFullBoltzmannConfig,
        run_full_boltzmann_jax,
    )
    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=sigma, N_mu=N_mu, N_q=N_q,
        correction_level=cl, n_reactions=12,
        collision_mode="ap_unified_preflight",
        thermo_tier=2,
    )
    return run_full_boltzmann_jax(cfg)


@pytest.fixture(scope="module")
def envelope_results(jax_setup):
    """One solve per (CL, σ) cell at (N_mu=4, N_q=6); shared across §1
    + §3 + §4 tests."""
    results = {}
    for (cl, sigma) in GOLD_FLRW_PLUS_ANISO.keys():
        r = _solve_full_boltzmann(cl, sigma, 4, 6)
        assert r.success, f"AP-unified solve failed at (CL={cl}, σ={sigma})"
        results[(cl, sigma)] = {
            "Yp": float(r.Yp),
            "N_eff": float(r.metadata["N_eff_measured"]),
        }
    return results


@pytest.fixture(scope="module")
def grid_scan_results(jax_setup):
    """One solve per (CL, grid) FLRW cell; shared across §2 tests."""
    results = {}
    for cl in (0, 1, 2, 3):
        for (N_mu, N_q) in GRIDS:
            r = _solve_full_boltzmann(cl, 0.0, N_mu, N_q)
            assert r.success, f"AP-unified solve failed at (CL={cl}, grid=({N_mu},{N_q}))"
            results[(cl, N_mu, N_q)] = {
                "Yp": float(r.Yp),
                "N_eff": float(r.metadata["N_eff_measured"]),
            }
    return results


# ───────────────────────────────────────────────────────────────────
# §1. Per-CL FLRW + anisotropic envelope lock
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", list(GOLD_FLRW_PLUS_ANISO.keys()),
                         ids=[f"cl{c}_sigma{s:.2f}" for c, s in GOLD_FLRW_PLUS_ANISO.keys()])
def test_yp_envelope_anchor(envelope_results, cell):
    """Each (CL, σ) cell must reproduce its gold Y_p within tolerance."""
    measured = envelope_results[cell]["Yp"]
    gold = GOLD_FLRW_PLUS_ANISO[cell]
    assert abs(measured - gold) < YP_TOL, (
        f"(CL={cell[0]}, σ={cell[1]}) Y_p drifted: measured={measured:.10f} "
        f"vs gold={gold:.10f}, |Δ|={abs(measured-gold):.4e} > tol={YP_TOL}"
    )


# ───────────────────────────────────────────────────────────────────
# §2. Per-CL grid scaling (canonical PR-T3D §5 gate)
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cl", [0, 1, 2, 3])
def test_grid_n_eff_canonical_gate(grid_scan_results, cl):
    """Per-CL FLRW N_eff spread across the canonical 3-grid sweep
    must pass the < 1e-4 PR-T3D §5 grid gate."""
    n_effs = [grid_scan_results[(cl, Nm, Nq)]["N_eff"] for (Nm, Nq) in GRIDS]
    spread = max(n_effs) - min(n_effs)
    assert spread < GRID_GATE, (
        f"CL{cl} grid spread widened: {spread:.4e} > {GRID_GATE:.0e}"
    )


# ───────────────────────────────────────────────────────────────────
# §3. Per-CL anisotropy stability (canonical PR-T3D §5 gate)
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cl", [0, 1, 2, 3])
def test_anisotropy_n_eff_canonical_gate(envelope_results, cl):
    """Per-CL N_eff spread across Σ_H ∈ {0, 0.05, 0.10} at (4, 6)
    must pass the canonical < 1e-3 PR-T3D §5 anisotropy gate."""
    n_effs = [envelope_results[(cl, s)]["N_eff"] for s in (0.0, 0.05, 0.10)]
    spread = max(n_effs) - min(n_effs)
    assert spread < ANISO_GATE, (
        f"CL{cl} anisotropy spread widened: {spread:.4e} > {ANISO_GATE:.0e}"
    )


# ───────────────────────────────────────────────────────────────────
# §4. CL ordering at every σ (Y_p increases with CL)
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.10])
def test_cl_ordering_yp_monotone(envelope_results, sigma):
    """Y_p must increase monotonically with CL: CL0 < CL1 < CL2 < CL3.
    The Born → +Coulomb+Sirlin → +finite-mass weak-rate ladder
    increases the n→p rate at freeze-out, raising Y_p."""
    yp_by_cl = [envelope_results[(cl, sigma)]["Yp"] for cl in (0, 1, 2, 3)]
    assert yp_by_cl[0] < yp_by_cl[1] < yp_by_cl[2] < yp_by_cl[3], (
        f"σ={sigma}: CL ladder broke ordering: {yp_by_cl}"
    )


# ───────────────────────────────────────────────────────────────────
# §5. N_eff is CL-invariant at FLRW (Mangano gap is CL-independent)
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cl", [0, 1, 2, 3])
def test_n_eff_invariant_under_cl_at_flrw(envelope_results, cl):
    """The AP-form FLRW N_eff = 3.0345 ± 5e-4 holds for every CL —
    the Mangano gap +0.0095 is the same documented model
    approximation limit regardless of which weak-rate ladder is
    selected."""
    n_eff = envelope_results[(cl, 0.0)]["N_eff"]
    assert abs(n_eff - N_EFF_BENCHMARK) < N_EFF_TOL_VS_BENCHMARK, (
        f"CL{cl} FLRW N_eff drifted from AP-unified benchmark: "
        f"{n_eff:.6f} vs {N_EFF_BENCHMARK:.4f}"
    )


# ───────────────────────────────────────────────────────────────────
# §6. Public-authority retirement
# ───────────────────────────────────────────────────────────────────

def test_ap_unified_public_dispatch_is_retired():
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend="jax_ap_unified_tier3")


# ───────────────────────────────────────────────────────────────────
# §7. Retained component metadata
# ───────────────────────────────────────────────────────────────────

def test_full_boltzmann_component_capability_remains_non_dispatchable():
    from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY

    cap = CAPABILITY_BY_KEY["jax_typeI_full_boltzmann_tier3_preflight"]
    assert cap.max_correction_level >= 3
    assert cap.key not in {item.key for item in CAPABILITY_BY_BACKEND.values()}
