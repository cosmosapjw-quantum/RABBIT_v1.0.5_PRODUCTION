"""BD631 FB — resolved ℓ=2 neutrino collision in the JAX full-Boltzmann lane.

Battery T0–T7 (+ flag isolation) from ``audit/reference/FB_ELL2_DESIGN.md`` §E
with the mandatory devil fixes (``audit/telemetry/fb/fb_devil.json``):

  T0  FLAG-OFF BIT-IDENTITY — the ℓ=2 mode is a pure additive superset of the
      monopole (ap_unified) path; every non-transport sector is bit-identical.
  T1  jnp-twin parity vs the numpy kernel at the 28 T-grid nodes.
  T2  gather/scatter round-trip + P₂⟂P₀ (M1 Gram-Schmidt) at Σ ∈ {0, 0.1, 0.3}.
  T3  conservation: number + ℓ=0 energy unchanged; dT_* identical ON vs OFF.
  T4  dissipativity (diag A2 ≤ 0) + contraction + a realizable full solve.
  T5  magnitude anchor: κ₂_eff(2 MeV) ≈ 0.24·(5/3) (F-033 downward refinement).
  T6  Jacobian linearity in f2 (jacfwd == analytic A2) + AP diag ≤ 0.
  T7  flag isolation: mode ON changes ONLY the resolved ℓ=2 transport sector.

Env: PYTHONPATH=src:audit/pydeps ./venv/bin/python.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.jax.characteristic_rays_jax import (
    P2_jax,
    jacobian_jax,
    mu_current_jax,
)
from rabbit.jax.driver_typeI_full_boltzmann import (
    JAXFullBoltzmannConfig,
    _ELL2_KERNEL_MODE,
    _build_collision_bank_gather_ell2,
    _build_collision_bank_scatter_ell2,
    _collision_bank_ell2_core_jax,
    _ell2_perp_and_norm,
    _get_full_boltzmann_rhs,
    _initial_transport_state,
    _mode_enables_ell2_kernel,
    _unpack_transport,
    run_full_boltzmann_jax,
)
from rabbit.jax.collision_ell2_kernelized_jax import (
    a2_block_analytic_jnp,
    kernelized_C2_ell2_f2_jnp,
    load_ell2_kernel_tables,
    m2_gamma_jnp,
)
from rabbit.jax.thermo_provider_jax import hubble_3T_jax
from rabbit.collisions.kernelized_ell2 import (
    _MEV2S,
    kappa2_eff,
    kernelized_C2_ell2,
)


_MONOPOLE_BASELINE_MODE = "ap_unified_preflight"


# ─────────────────────────────────────────────────────────────────────────
# shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────

def _build(mode: str):
    """Build the (rhs_fn, jac_fn, layout, mu0, w0, x0, signs, q_nodes, ...) payload."""
    return _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode=mode,
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=12,
        N_q=20,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )


def _state(layout, *, Sp=0.1, quad_amp=0.05):
    """A thermal state with a nonzero P₂ quadrupole injected onto the rays."""
    y = np.zeros(layout["n_total"], dtype=np.float64)
    y[layout["i_Sp"]] = Sp
    y[layout["i_S"]] = Sp
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    tr = _initial_transport_state(12, 20)
    mu0 = np.polynomial.legendre.leggauss(12)[0]
    p2 = np.asarray(P2_jax(jnp.asarray(mu0)))
    tr = tr * (1.0 + quad_amp * p2[None, :, None])
    y[i0:i1] = tr.reshape(-1)
    y[layout["i_tg"]] = 3.0
    y[layout["i_tne"]] = 2.9
    y[layout["i_tnx"]] = 2.8
    y[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87])
    return y


def _transport_slice(layout):
    i0 = layout["i_transport"]
    return slice(i0, i0 + layout["n_transport"])


def _independent_ell2_transport_term(layout, mu0, w0, x0, signs, q_nodes, y, tables):
    """Reconstruct the additive ℓ=2 transport contribution the RHS should add."""
    S_val = float(y[layout["i_S"]])
    mu = mu_current_jax(x0, signs, jnp.asarray(S_val))
    J = jacobian_jax(x0, jnp.asarray(S_val), mu)
    G2 = _build_collision_bank_gather_ell2(mu=mu, jacobian_weights=J, w0=w0, layout=layout)
    S2 = _build_collision_bank_scatter_ell2(mu=mu, jacobian_weights=J, w0=w0, layout=layout)
    f2 = G2 @ jnp.asarray(y)
    T_gamma = jnp.asarray(y[layout["i_tg"]])
    T_nu_e = jnp.asarray(y[layout["i_tne"]])
    T_nu_x = jnp.asarray(y[layout["i_tnx"]])
    Sigma_sq = jnp.asarray(y[layout["i_Sp"]] ** 2 + y[layout["i_Sm"]] ** 2)
    H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq)
    dN = _collision_bank_ell2_core_jax(
        f2, T_nu_e=T_nu_e, T_nu_x=T_nu_x, H_MeV=H_MeV, q_nodes=q_nodes, ell2_tables=tables
    )
    return np.asarray(_unpack_transport(S2 @ dN, layout)).reshape(-1), np.asarray(dN)


# ─────────────────────────────────────────────────────────────────────────
# T0 — FLAG-OFF BIT-IDENTITY (promotion-safety gate)
# ─────────────────────────────────────────────────────────────────────────

def test_T0_flag_off_bit_identity():
    """The ℓ=2 mode is a pure additive superset of the monopole path.

    Non-transport sectors (3T + network) are bit-identical to the OFF path
    (``ap_unified_preflight``); the transport differs only by the independently
    reconstructed ℓ=2 term.  This is exactly what "OFF is bit-identical to
    pre-change" means: the monopole code path is untouched.
    """
    tables = load_ell2_kernel_tables()
    rhs_off, _jac_off, layout, mu0, w0, x0, signs, q_nodes, _ew = _build(_MONOPOLE_BASELINE_MODE)
    rhs_on, _jac_on, layout2, *_ = _build(_ELL2_KERNEL_MODE)
    y = _state(layout)

    d_off = np.asarray(rhs_off(jnp.asarray(0.0), jnp.asarray(y)))
    d_on = np.asarray(rhs_on(jnp.asarray(0.0), jnp.asarray(y)))

    tsl = _transport_slice(layout)
    mask = np.ones_like(d_off, dtype=bool)
    mask[tsl] = False

    # Non-transport (Σ, S, 3T, network) bit-identical between OFF and ON.
    assert np.array_equal(d_off[mask], d_on[mask]), "ℓ=2 perturbed a non-transport sector"

    # Determinism of the OFF path (bit-identical on repeat).
    d_off2 = np.asarray(rhs_off(jnp.asarray(0.0), jnp.asarray(y)))
    assert np.array_equal(d_off, d_off2)

    # Transport = monopole (OFF) + independently reconstructed ℓ=2 term.
    ell2_flat, _dN = _independent_ell2_transport_term(
        layout, mu0, w0, x0, signs, q_nodes, y, tables
    )
    expected = d_off[tsl] + ell2_flat
    np.testing.assert_allclose(d_on[tsl], expected, rtol=1e-11, atol=1e-14)

    # The default collisionless mode never reaches the ℓ=2 wiring.
    assert not _mode_enables_ell2_kernel("collisionless")
    assert not _mode_enables_ell2_kernel(_MONOPOLE_BASELINE_MODE)


# ─────────────────────────────────────────────────────────────────────────
# T1 — jnp twin parity vs numpy kernel at the 28 T-grid nodes
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("species,combo_attr", [("nue", "nue"), ("nux", "nux")])
def test_T1_twin_parity_numpy_at_T_nodes(species, combo_attr):
    """twin(f0·Ψ₂) == f0·numpy_C2(Ψ₂) to machine precision at every T-grid node."""
    tables = load_ell2_kernel_tables()
    npz = np.load(
        os.path.join(os.path.dirname(__file__), "..", "src", "rabbit", "data",
                     "ell2_kernel_tables_v1.npz"),
        allow_pickle=True,
    )
    q_nodes = np.asarray(npz["q_nodes"])
    q_weights = np.asarray(npz["q_weights"])
    T_grid = np.asarray(npz["T_grid"])
    nu2 = getattr(tables, f"{combo_attr}_nu2")
    R2 = getattr(tables, f"{combo_attr}_R2")

    rng = np.random.default_rng(1234)
    H_inv_sec = 1.0e3
    H_MeV = H_inv_sec / _MEV2S
    max_rel = 0.0
    for T in T_grid:
        Psi2 = rng.standard_normal(20)
        f0 = 1.0 / (np.exp(np.clip(q_nodes / T, -500, 500)) + 1.0)
        f2 = f0 * Psi2
        npy = np.asarray(
            kernelized_C2_ell2(Psi2, float(T), float(T), H_inv_sec, q_nodes, q_weights, species)
        )
        twin = np.asarray(
            kernelized_C2_ell2_f2_jnp(
                jnp.asarray(f2), float(T), float(H_MeV), tables.lnT, nu2, R2, tables.q_nodes
            )
        )
        rhs = f0 * npy
        keep = np.abs(rhs) > 1e-30
        if keep.any():
            rel = np.max(np.abs(twin[keep] - rhs[keep]) / np.abs(rhs[keep]))
            max_rel = max(max_rel, rel)
        # tail rows floored to zero in both.
        assert np.allclose(twin[~keep], 0.0, atol=1e-30)
    assert max_rel <= 1e-10, f"twin/numpy parity {max_rel:.2e} exceeds 1e-10 at T-nodes"


def test_T1_m2_gamma_exact_at_nodes():
    """M2_G(T) via jnp.interp reproduces the tabulated slice exactly at nodes."""
    tables = load_ell2_kernel_tables()
    npz = np.load(
        os.path.join(os.path.dirname(__file__), "..", "src", "rabbit", "data",
                     "ell2_kernel_tables_v1.npz"),
        allow_pickle=True,
    )
    T_grid = np.asarray(npz["T_grid"])
    nu2_tab = np.asarray(npz["nu_e__nue__nu2"])
    R2_tab = np.asarray(npz["nu_e__nue__R2"])
    for k, T in enumerate(T_grid):
        M = np.asarray(m2_gamma_jnp(float(T), tables.lnT, tables.nue_nu2, tables.nue_R2))
        expected = -np.diag(nu2_tab[k]) + R2_tab[k]
        np.testing.assert_allclose(M, expected, rtol=0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# T2 — gather/scatter round-trip + P₂⟂P₀ (M1)
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("S_val", [0.0, 0.1, 0.3])
def test_T2_p2perp_orthogonal_to_p0(S_val):
    """M1: Σ w0·J·P2_perp = 0 to machine precision at any shear (raw P2 leaks)."""
    _r, _j, layout, mu0, w0, x0, signs, q_nodes, _ew = _build(_ELL2_KERNEL_MODE)
    mu = mu_current_jax(x0, signs, jnp.asarray(S_val))
    J = jacobian_jax(x0, jnp.asarray(S_val), mu)
    P2_perp, N2 = _ell2_perp_and_norm(mu, J, w0)
    wj = np.asarray(w0) * np.asarray(J)
    leak_perp = float(np.sum(wj * np.asarray(P2_perp)))
    leak_raw = float(np.sum(wj * np.asarray(P2_jax(mu))))
    assert abs(leak_perp) < 1e-12, f"P2_perp leak {leak_perp:.2e} at S={S_val}"
    assert float(N2) > 0.0
    if S_val >= 0.3:
        # Documents WHY M1 is needed: the raw P2 leaks ~2e-3 at strong shear.
        assert abs(leak_raw) > 1e-4


def test_T2_gather_scatter_round_trip():
    """gather∘scatter == identity on the quadrupole moment (N2 consistency)."""
    _r, _j, layout, mu0, w0, x0, signs, q_nodes, _ew = _build(_ELL2_KERNEL_MODE)
    n_q = int(layout["N_q"])
    for S_val in (0.0, 0.1, 0.3):
        mu = mu_current_jax(x0, signs, jnp.asarray(S_val))
        J = jacobian_jax(x0, jnp.asarray(S_val), mu)
        G2 = _build_collision_bank_gather_ell2(mu=mu, jacobian_weights=J, w0=w0, layout=layout)
        S2 = _build_collision_bank_scatter_ell2(mu=mu, jacobian_weights=J, w0=w0, layout=layout)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(3 * n_q)
        rays = S2 @ jnp.asarray(x)             # (n_total,)
        back = np.asarray(G2 @ rays)           # (3 n_q,)
        np.testing.assert_allclose(back, x, rtol=1e-10, atol=1e-11)


# ─────────────────────────────────────────────────────────────────────────
# T3 — conservation (number + ℓ=0 energy) + 3T invariance
# ─────────────────────────────────────────────────────────────────────────

def test_T3_scatter_conserves_number_and_energy():
    """The ℓ=2 scatter has zero ℓ=0 (number) and zero ℓ=0-energy projection."""
    _r, _j, layout, mu0, w0, x0, signs, q_nodes, energy_weight = _build(_ELL2_KERNEL_MODE)
    n_mu = int(layout["N_mu"])
    n_q = int(layout["N_q"])
    stride = n_mu * n_q
    i0 = layout["i_transport"]
    for S_val in (0.0, 0.1, 0.3):
        mu = mu_current_jax(x0, signs, jnp.asarray(S_val))
        J = jacobian_jax(x0, jnp.asarray(S_val), mu)
        S2 = _build_collision_bank_scatter_ell2(mu=mu, jacobian_weights=J, w0=w0, layout=layout)
        rng = np.random.default_rng(99)
        moment = rng.standard_normal(3 * n_q)
        rays_flat = np.asarray(S2 @ jnp.asarray(moment))
        transport = rays_flat[i0 : i0 + 4 * stride].reshape(4, n_mu, n_q)
        wj = np.asarray(w0) * np.asarray(J)
        # number moment per (species, q): Σ_mu (1/2 w0 J) δf  — machine-zero
        # because Σ_mu w0·J·P2_perp = 0 by construction (M1 Gram-Schmidt).
        number = 0.5 * np.tensordot(wj, transport, axes=((0,), (1,)))   # (4, n_q)
        assert np.max(np.abs(number)) < 1e-12, f"number leak at S={S_val}"
        # ℓ=0 energy moment = Σ_q ew(q)·number(q); ew reaches ~4e6 (Gauss-Laguerre
        # q³ weights), so the leak inherits number's machine-zero scaled by ‖ew‖.
        # The physically meaningful metric is the RELATIVE leak → machine ε.
        ew = np.asarray(energy_weight)
        energy = np.tensordot(number, ew, axes=((1,), (0,)))            # (4,)
        energy_scale = np.max(np.abs(ew)) * np.max(np.abs(moment))
        assert np.max(np.abs(energy)) < 1e-13 * energy_scale, f"energy leak at S={S_val}"


def test_T3_3T_invariant_on_vs_off():
    """dT_γ, dT_νe, dT_νx are bit-identical with ℓ=2 ON vs OFF (traceless ℓ=2)."""
    rhs_off, _jo, layout, *_ = _build(_MONOPOLE_BASELINE_MODE)
    rhs_on, _jn, layout2, *_ = _build(_ELL2_KERNEL_MODE)
    y = _state(layout)
    d_off = np.asarray(rhs_off(jnp.asarray(0.0), jnp.asarray(y)))
    d_on = np.asarray(rhs_on(jnp.asarray(0.0), jnp.asarray(y)))
    for key in ("i_tg", "i_tne", "i_tnx"):
        assert d_off[layout[key]] == d_on[layout[key]], f"{key} changed by ℓ=2"


# ─────────────────────────────────────────────────────────────────────────
# T4 — dissipativity + contraction + realizable full solve
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("combo_attr", ["nue", "nux"])
def test_T4_dissipativity_diag_nonpositive(combo_attr):
    """diag(A2) ≤ 0 on all thermally-populated core nodes across the T grid."""
    tables = load_ell2_kernel_tables()
    nu2 = getattr(tables, f"{combo_attr}_nu2")
    R2 = getattr(tables, f"{combo_attr}_R2")
    q_nodes = tables.q_nodes
    T_grid = np.exp(np.asarray(tables.lnT))
    H_MeV = 1.0
    worst = -np.inf
    for T in T_grid:
        A2 = np.asarray(a2_block_analytic_jnp(float(T), H_MeV, tables.lnT, nu2, R2, q_nodes))
        f0 = 1.0 / (np.exp(np.clip(np.asarray(q_nodes) / T, -500, 500)) + 1.0)
        core = f0 >= 1e-6
        diag = np.diag(A2)[core]
        worst = max(worst, float(np.max(diag)) if diag.size else -np.inf)
    assert worst <= 1e-12, f"diag(A2) positive somewhere: max={worst:.2e}"


def test_T4_contraction_damps_quadrupole():
    """A flat positive Ψ₂ relaxes (dominant-core df2/dN < 0): damping toward isotropy."""
    tables = load_ell2_kernel_tables()
    q_nodes = np.asarray(tables.q_nodes)
    T = 2.0
    f0 = 1.0 / (np.exp(np.clip(q_nodes / T, -500, 500)) + 1.0)
    f2 = f0 * 1.0  # Ψ₂ = 1
    dN = np.asarray(
        kernelized_C2_ell2_f2_jnp(
            jnp.asarray(f2), T, 1.0, tables.lnT, tables.nue_nu2, tables.nue_R2, tables.q_nodes
        )
    )
    core = f0 >= 1e-6
    assert np.min(dN[core]) < 0.0, "no damping mode present"
    # the dominant (largest |rate|) core mode must be relaxing (negative).
    dom = np.argmax(np.abs(dN[core]))
    assert dN[core][dom] < 0.0


def test_T4_full_solve_realizable():
    """A full ℓ=2 solve succeeds with finite, physical yields (implicit stability)."""
    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.1,
        N_mu=12,
        N_q=20,
        correction_level=0,
        n_reactions=12,
        collision_mode=_ELL2_KERNEL_MODE,
        thermo_tier=2,
        rtol=1e-6,
        atol=1e-8,
        max_steps=512,
        event_refine_steps=12,
    )
    r = run_full_boltzmann_jax(cfg)
    assert r.success, "ℓ=2 full solve did not converge (realizability/stiffness)"
    assert np.isfinite(r.Yp) and 0.0 < r.Yp < 1.0
    assert np.isfinite(r.metadata["N_eff_measured"])


# ─────────────────────────────────────────────────────────────────────────
# T5 — magnitude anchor (F-033 downward refinement)
# ─────────────────────────────────────────────────────────────────────────

def test_T5_magnitude_anchor_kappa2_eff():
    """κ₂_eff(2 MeV) ≈ 0.24·(5/3) ≈ 0.40 — the kernel refines 5/3 DOWNWARD."""
    val = kappa2_eff(2.0, "nue")
    assert 0.35 < val < 0.45
    assert abs(val - 0.24 * (5.0 / 3.0)) < 0.02


# ─────────────────────────────────────────────────────────────────────────
# T6 — Jacobian linearity + AP diag ≤ 0
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("combo_attr,species_bank", [("nue", 0), ("nux", 2)])
def test_T6_jacfwd_equals_analytic_A2(combo_attr, species_bank):
    """jacfwd(twin wrt f2) == analytic A2 = -2·M2_G/H (linearity, machine precision)."""
    tables = load_ell2_kernel_tables()
    nu2 = getattr(tables, f"{combo_attr}_nu2")
    R2 = getattr(tables, f"{combo_attr}_R2")
    q_nodes = tables.q_nodes
    T = 2.0
    H_MeV = 3.0e-19
    f2 = jnp.asarray(np.random.default_rng(3).standard_normal(20))
    jac = np.asarray(
        jax.jacfwd(
            lambda x: kernelized_C2_ell2_f2_jnp(x, T, H_MeV, tables.lnT, nu2, R2, q_nodes)
        )(f2)
    )
    analytic = np.asarray(a2_block_analytic_jnp(T, H_MeV, tables.lnT, nu2, R2, q_nodes))
    np.testing.assert_allclose(jac, analytic, rtol=1e-11, atol=1e-14)


def test_T6_bank_core_jacobian_constant_in_f2():
    """The 3-bank ℓ=2 core Jacobian is state-independent (linear in f2)."""
    tables = load_ell2_kernel_tables()
    q_nodes = tables.q_nodes
    kw = dict(T_nu_e=jnp.asarray(2.9), T_nu_x=jnp.asarray(2.8), H_MeV=jnp.asarray(3.0e-19),
              q_nodes=q_nodes, ell2_tables=tables)
    fn = lambda f2b: _collision_bank_ell2_core_jax(f2b, **kw)
    rng = np.random.default_rng(11)
    J1 = np.asarray(jax.jacfwd(fn)(jnp.asarray(rng.standard_normal(60))))
    J2 = np.asarray(jax.jacfwd(fn)(jnp.asarray(rng.standard_normal(60))))
    np.testing.assert_allclose(J1, J2, rtol=0.0, atol=1e-30)
    # block-diagonal: nue/nuebar/nux banks do not cross-couple.
    assert np.max(np.abs(J1[:20, 20:])) < 1e-30
    assert np.max(np.abs(J1[20:40, :20])) < 1e-30


def test_T6_ap_diag_nonpositive():
    """The ℓ=2 AP diagonal (diag of the bank core Jacobian) is ≤ 0."""
    tables = load_ell2_kernel_tables()
    q_nodes = tables.q_nodes
    kw = dict(T_nu_e=jnp.asarray(2.9), T_nu_x=jnp.asarray(2.8), H_MeV=jnp.asarray(3.0e-19),
              q_nodes=q_nodes, ell2_tables=tables)
    fn = lambda f2b: _collision_bank_ell2_core_jax(f2b, **kw)
    J = np.asarray(jax.jacfwd(fn)(jnp.asarray(np.zeros(60))))
    assert np.max(np.diag(J)) <= 1e-12


# ─────────────────────────────────────────────────────────────────────────
# T7 — flag isolation
# ─────────────────────────────────────────────────────────────────────────

def test_T7_flag_isolation_only_ell2_sector():
    """Mode ON changes ONLY the resolved ℓ=2 transport sector.

    3T + network bit-identical; the transport difference has zero ℓ=0 (number)
    projection (conservation) and is nonzero (ℓ=2 is genuinely active).
    """
    rhs_off, _jo, layout, mu0, w0, x0, signs, q_nodes, _ew = _build(_MONOPOLE_BASELINE_MODE)
    rhs_on, _jn, layout2, *_ = _build(_ELL2_KERNEL_MODE)
    y = _state(layout)
    d_off = np.asarray(rhs_off(jnp.asarray(0.0), jnp.asarray(y)))
    d_on = np.asarray(rhs_on(jnp.asarray(0.0), jnp.asarray(y)))

    tsl = _transport_slice(layout)
    mask = np.ones_like(d_off, dtype=bool)
    mask[tsl] = False
    assert np.array_equal(d_off[mask], d_on[mask])

    diff = (d_on[tsl] - d_off[tsl])
    assert np.max(np.abs(diff)) > 1e-14, "ℓ=2 term is inert"

    # zero ℓ=0 projection of the ℓ=2 transport difference (conservation).
    n_mu = int(layout["N_mu"]); n_q = int(layout["N_q"])
    diff_t = diff.reshape(4, n_mu, n_q)
    S_val = float(y[layout["i_S"]])
    mu = mu_current_jax(x0, signs, jnp.asarray(S_val))
    J = jacobian_jax(x0, jnp.asarray(S_val), mu)
    wj = np.asarray(w0) * np.asarray(J)
    number = 0.5 * np.tensordot(wj, diff_t, axes=((0,), (1,)))
    assert np.max(np.abs(number)) < 1e-11
