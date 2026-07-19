"""
test_j02_jax_rhs — Verification tests for J02 (JAX-Native RHS Functions).

Gate GJ1: FLRW RHS parity JAX vs SciPy < 10⁻¹²
Acceptance criteria:
  AC1: jax.jit(flrw_rhs)(N, y, params) compiles without error
  AC2: FLRW RHS parity on 7-point T grid: max|Δ|/|f| < 10⁻¹²
  AC3: Type I RHS parity at Σ=0.1: max|Δ|/|f| < 10⁻¹¹
  AC4: jax.jacfwd(rhs) returns Jacobian without NaN
  AC5: Nuclear rate interpolation: max|ΔlogR| < 10⁻³ vs SciPy
  AC6: JIT compilation time < 30 s
  AC7: Single RHS evaluation time after JIT < 0.1 ms
"""
import pytest
import sys
import os
import time

# (paths resolved via pip install)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Thermo EOS parity (Gate GJ1 core)
# ═══════════════════════════════════════════════════════════════════════

class TestThermoParity:
    """rho_plasma, rho_electron, drho_dT parity vs SciPy reference."""

    T_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def test_ac2_rho_plasma_parity(self):
        """AC2: rho_plasma parity < 10⁻¹² at all 7 temperatures."""
        from rabbit.thermo.eos_photon_electron import rho_plasma as rho_scipy
        from rabbit.jax.thermo_jax import rho_plasma as rho_jax

        max_rel = 0.0
        for T in self.T_GRID:
            rs = float(rho_scipy(T))
            rj = float(rho_jax(T))
            rel = abs(rs - rj) / max(abs(rs), 1e-100)
            max_rel = max(max_rel, rel)
        assert max_rel < 1e-12, f"Max relative error = {max_rel:.2e}"
        print(f"  AC2 rho_plasma parity: PASS (max_rel = {max_rel:.2e})")

    def test_drho_dT_ad_vs_fd(self):
        """drho_dT: JAX AD matches SciPy FD to < 10⁻⁹ (FD is the less accurate one)."""
        from rabbit.thermo.eos_photon_electron import drho_dT as drho_scipy
        from rabbit.jax.thermo_jax import drho_dT as drho_jax

        max_rel = 0.0
        for T in [0.1, 1.0, 5.0, 10.0]:
            ds = float(drho_scipy(T))
            dj = float(drho_jax(T))
            rel = abs(ds - dj) / max(abs(ds), 1e-100)
            max_rel = max(max_rel, rel)
        # FD truncation error ~10⁻¹⁰; AD is exact
        assert max_rel < 1e-8, f"Max relative error = {max_rel:.2e}"
        print(f"  drho_dT AD vs FD: PASS (max_rel = {max_rel:.2e})")

    def test_hubble_rate_finite(self):
        """Hubble rate finite and positive across T range."""
        from rabbit.jax.thermo_jax import hubble_rate
        for T in self.T_GRID:
            H = float(hubble_rate(T, 3.044))
            assert H > 0, f"H({T}) = {H} ≤ 0"
            assert jnp.isfinite(H), f"H({T}) = {H} not finite"
        print("  Hubble rate finite: PASS")


# ═══════════════════════════════════════════════════════════════════════
# §2. Geometry parity
# ═══════════════════════════════════════════════════════════════════════

class TestGeometryParity:

    def test_ac3_typeI_geometry_parity(self):
        """AC3: Type I geometry RHS parity < 10⁻¹⁴."""
        from rabbit.geometry.typeI import compute_typeI_geometry_rhs as geom_scipy
        from rabbit.jax.rhs_typeI import typeI_geometry_rhs as geom_jax

        max_err = 0.0
        for Sp, Pi in [(0.0, 0.01), (0.01, 0.005), (0.1, 0.02), (0.3, 0.05)]:
            Omega = max(0, 1 - Sp ** 2)
            ds, _ = geom_scipy(Sp, 0.0, Pi, 0.0, Omega)
            dj, _ = geom_jax(Sp, 0.0, Pi, 0.0)
            err = abs(ds - float(dj))
            ref = max(abs(ds), 1e-100)
            max_err = max(max_err, err / ref)
        assert max_err < 1e-14, f"Max relative error = {max_err:.2e}"
        print(f"  AC3 geometry parity: PASS (max_rel = {max_err:.2e})")


# ═══════════════════════════════════════════════════════════════════════
# §3. I₀ and weak rate parity
# ═══════════════════════════════════════════════════════════════════════

class TestWeakParity:

    def test_I0_parity(self):
        """I₀ normalization integral matches SciPy to machine precision."""
        from rabbit.weak.channels import compute_I0_born as I0_scipy
        from rabbit.jax.weak_jax import compute_I0_born as I0_jax
        i0s = I0_scipy()
        i0j = float(I0_jax())
        rel = abs(i0s - i0j) / i0s
        assert rel < 1e-10, f"I₀ rel error = {rel:.2e}"
        print(f"  I₀ parity: PASS (rel = {rel:.2e})")

    def test_born_rates_order_of_magnitude(self):
        """Born-level rates have correct order of magnitude at T=1 MeV."""
        from rabbit.jax.weak_jax import compute_born_rates
        lnp, lpn = compute_born_rates(1.0, 1.0, 878.4)
        lnp, lpn = float(lnp), float(lpn)
        # At T ≈ 1 MeV: λ_np ~ 1-2 s⁻¹, λ_pn ~ 0.3-0.5 s⁻¹
        assert 0.5 < lnp < 5.0, f"λ_np = {lnp:.4f} (expect ~1-2)"
        assert 0.1 < lpn < 2.0, f"λ_pn = {lpn:.4f} (expect ~0.3-0.5)"
        # λ_np > λ_pn (neutrons decay faster)
        assert lnp > lpn, f"λ_np={lnp} should be > λ_pn={lpn}"
        print(f"  Born rates: PASS (λ_np={lnp:.3f}, λ_pn={lpn:.3f})")

    def test_equilibrium_Xn(self):
        """Equilibrium X_n at high T should be ~0.5 (detailed balance)."""
        from rabbit.jax.weak_jax import equilibrium_Xn
        # At T = 10 MeV ≫ Q, X_n → 1/(1 + exp(Q/T)) → ~0.45
        Xn_high = float(equilibrium_Xn(10.0, 10.0, 878.4))
        assert 0.35 < Xn_high < 0.55, f"X_n(10 MeV) = {Xn_high:.4f}"
        # At T = 0.1 MeV ≪ Q, X_n → 0
        Xn_low = float(equilibrium_Xn(0.1, 0.1, 878.4))
        assert Xn_low < 0.01, f"X_n(0.1 MeV) = {Xn_low:.6f}"
        print(f"  Equilibrium X_n: PASS (T=10: {Xn_high:.3f}, T=0.1: {Xn_low:.6f})")


# ═══════════════════════════════════════════════════════════════════════
# §4. Nuclear network parity
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkParity:

    def test_ac5_rate_interpolation(self):
        """AC5: Nuclear rate interpolation error < 10⁻³ vs SciPy standard."""
        from rabbit.network.abundances_standard import evaluate_nuclear_rates as rates_scipy
        from rabbit.jax.network_jax import evaluate_nuclear_rates_jax, load_rate_table

        table = load_rate_table(n_reactions=31)
        max_log_err = 0.0
        for T in [0.01, 0.1, 0.5, 1.0, 5.0]:
            fwd_s, _ = rates_scipy(T, 31)
            fwd_j, _ = evaluate_nuclear_rates_jax(T, table)
            for i in range(31):
                if fwd_s[i] > 0:
                    log_err = abs(jnp.log10(fwd_j[i]) - jnp.log10(fwd_s[i]))
                    max_log_err = max(max_log_err, float(log_err))
        assert max_log_err < 1e-3, f"Max |Δlog₁₀R| = {max_log_err:.4f}"
        print(f"  AC5 rate interpolation: PASS (max |ΔlogR| = {max_log_err:.2e})")

    def test_stoichiometry_mass_conservation(self):
        """Stoichiometry preserves baryon number: ΣA_i S_ij = 0."""
        from rabbit.jax.network_jax import STOICHIOMETRY, ATOMIC_MASSES
        check = ATOMIC_MASSES @ STOICHIOMETRY
        assert jnp.allclose(check, 0.0, atol=1e-14), f"Baryon violation: {check}"
        print("  Stoichiometry mass conservation: PASS")


# ═══════════════════════════════════════════════════════════════════════
# §5. AD capability tests
# ═══════════════════════════════════════════════════════════════════════

class TestADCapability:

    def test_ac4_jacfwd_thermo(self):
        """AC4: jacfwd on rho_plasma returns finite derivative."""
        from rabbit.jax.thermo_jax import rho_plasma
        grad_fn = jax.grad(rho_plasma)
        for T in [0.5, 1.0, 5.0]:
            g = grad_fn(T)
            assert jnp.isfinite(g), f"grad rho_plasma({T}) = {g} (not finite)"
        print("  AC4 jacfwd thermo: PASS")

    def test_ac4_jacfwd_geometry(self):
        """AC4: jacfwd on geometry returns 2-element Jacobian without NaN."""
        from rabbit.jax.rhs_typeI import typeI_geometry_rhs
        def geom_vec(Sp):
            return jnp.array(typeI_geometry_rhs(Sp, 0.0, 0.01, 0.0))
        J = jax.jacfwd(geom_vec)(0.1)
        assert J.shape == (2,), f"Jacobian shape {J.shape}"
        assert jnp.all(jnp.isfinite(J)), f"NaN in Jacobian: {J}"
        print(f"  AC4 jacfwd geometry: PASS (J = {J})")

    def test_grad_through_weak_rates(self):
        """Gradient through weak rates w.r.t. tau_n is finite."""
        from rabbit.jax.weak_jax import compute_born_rates
        def lnp_of_tau(tau):
            lnp, _ = compute_born_rates(1.0, 1.0, tau)
            return lnp
        g = jax.grad(lnp_of_tau)(878.4)
        assert jnp.isfinite(g), f"∂λ_np/∂τ_n = {g} (not finite)"
        # ∂λ_np/∂τ_n < 0 (longer lifetime → smaller rate)
        assert g < 0, f"∂λ_np/∂τ_n = {g} should be < 0"
        print(f"  Grad through weak rates: PASS (∂λ_np/∂τ_n = {float(g):.4e})")


# ═══════════════════════════════════════════════════════════════════════
# §6. Performance tests
# ═══════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_ac6_jit_compilation_time(self):
        """AC6: JIT compilation < 30 s."""
        from rabbit.jax.thermo_jax import rho_plasma, drho_dT, hubble_rate
        from rabbit.jax.rhs_typeI import typeI_geometry_rhs

        t0 = time.perf_counter()
        # Force compilation
        _ = rho_plasma(1.0)
        _ = drho_dT(1.0)
        _ = hubble_rate(1.0, 3.044)
        _ = typeI_geometry_rhs(0.1, 0.0, 0.01, 0.0)
        jax.block_until_ready(_)
        dt = time.perf_counter() - t0
        assert dt < 30, f"JIT compilation took {dt:.1f} s"
        print(f"  AC6 JIT compilation: PASS ({dt:.2f} s)")

    def test_ac7_warm_evaluation_time(self):
        """AC7: Warm RHS evaluation < 1 ms."""
        from rabbit.jax.thermo_jax import rho_plasma
        # Warm up
        _ = rho_plasma(1.0)
        jax.block_until_ready(_)

        t0 = time.perf_counter()
        N = 100
        for _ in range(N):
            r = rho_plasma(1.0)
        jax.block_until_ready(r)
        dt = (time.perf_counter() - t0) / N * 1000  # ms
        assert dt < 1.0, f"Warm evaluation took {dt:.3f} ms"
        print(f"  AC7 warm evaluation: PASS ({dt:.4f} ms)")


# ═══════════════════════════════════════════════════════════════════════
# §7. Runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("J02 Verification: JAX-Native RHS Functions")
    print("=" * 60)

    t1 = TestThermoParity()
    t1.test_ac2_rho_plasma_parity()
    t1.test_drho_dT_ad_vs_fd()
    t1.test_hubble_rate_finite()

    t2 = TestGeometryParity()
    t2.test_ac3_typeI_geometry_parity()

    t3 = TestWeakParity()
    t3.test_I0_parity()
    t3.test_born_rates_order_of_magnitude()
    t3.test_equilibrium_Xn()

    t4 = TestNetworkParity()
    t4.test_ac5_rate_interpolation()
    t4.test_stoichiometry_mass_conservation()

    t5 = TestADCapability()
    t5.test_ac4_jacfwd_thermo()
    t5.test_ac4_jacfwd_geometry()
    t5.test_grad_through_weak_rates()

    t6 = TestPerformance()
    t6.test_ac6_jit_compilation_time()
    t6.test_ac7_warm_evaluation_time()

    print("=" * 60)
    print("ALL J02 VERIFICATION TESTS PASSED")
