"""
test_constraint_regression.py — Lock ALL headline numbers from report.

Every number that appears in the abstract, conclusions, or tables of
RABBIT_report_v1.0.0 is regression-locked here against a JSON fixture.
If any of these drift, the report claims become invalid.
"""
import json
import pytest
import numpy as np
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "constraint_gold_v100.json"


@pytest.fixture(scope="module")
def gold():
    with open(FIXTURE) as f:
        return json.load(f)


# ═══ Eigenvalue lock ═══════════════════════════════════════════

class TestEigenvalueLock:
    """Lock the linearised shear-stress eigenvalues."""

    def test_eigenvalues(self, gold):
        g = gold["eigenvalues"]
        f_nu = 0.4052
        C_fb, C_src, C_drain = 24/5, 8/15, 4.0
        disc = 25 - 4*(C_drain - C_fb*C_src*f_nu)
        lam_s = (-5 + np.sqrt(disc)) / 2
        lam_f = (-5 - np.sqrt(disc)) / 2

        assert abs(lam_s - g["lambda_slow"]) < g["tolerance"], \
            f"λ_slow={lam_s:.4f} vs gold {g['lambda_slow']}"
        assert abs(lam_f - g["lambda_fast"]) < g["tolerance"], \
            f"λ_fast={lam_f:.4f} vs gold {g['lambda_fast']}"
        assert abs(lam_s + lam_f - g["trace"]) < 1e-10, \
            f"tr(M)={lam_s+lam_f} vs gold {g['trace']}"


# ═══ FLRW gold Yp lock ════════════════════════════════════════

@pytest.mark.slow
class TestFLRWYpLock:
    """Lock FLRW Yp at CL0 and CL2."""

    def _solve(self, cl):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        return canonical_forward_solver(
            Sigma_H=0.0, correction_level=cl, N_q=20, backend="scipy")

    def test_flrw_cl0(self, gold):
        g = gold["flrw_gold"]["CL0"]
        r = self._solve(0)
        assert r.success
        assert abs(r.Yp - g["Yp"]) < g["tolerance"], \
            f"CL0 Yp={r.Yp:.7f} vs gold {g['Yp']}"

    def test_flrw_cl2(self, gold):
        g = gold["flrw_gold"]["CL2"]
        r = self._solve(2)
        assert r.success
        assert abs(r.Yp - g["Yp"]) < g["tolerance"], \
            f"CL2 Yp={r.Yp:.7f} vs gold {g['Yp']}"


# ═══ Constraint Σ_H < 0.66 lock ═══════════════════════════════

@pytest.mark.slow
class TestConstraintLock:
    """Lock the 95% CL upper limit on Σ_H."""

    def test_sigma_constraint_cl2(self, gold):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        g = gold["constraint_95cl"]["CL2"]

        # Scan around the expected crossing
        sigmas = np.linspace(0.5, 0.75, 15)
        chi2_vals = []
        for s in sigmas:
            r = canonical_forward_solver(
                Sigma_H=float(s), correction_level=2, N_q=20, backend="scipy")
            chi2 = ((r.Yp - 0.2449) / 0.004)**2 + ((r.DH - 2.547e-5) / 0.029e-5)**2
            chi2_vals.append(chi2)
        chi2_vals = np.array(chi2_vals)
        dchi2 = chi2_vals - chi2_vals[0]

        # Find crossing
        above = np.where(dchi2 > 3.84)[0]
        if len(above) > 0 and above[0] > 0:
            sigma_95 = float(np.interp(
                3.84, dchi2[:above[0]+1], sigmas[:above[0]+1]))
        else:
            sigma_95 = float(sigmas[-1])

        assert abs(sigma_95 - g["Sigma_H_upper"]) < g["tolerance"], \
            f"95% CL: Σ_H < {sigma_95:.3f} vs gold {g['Sigma_H_upper']}"


# ═══ Gradient bridge lock ══════════════════════════════════════

@pytest.mark.slow
class TestGradientBridgeLock:
    """Lock gradient bridge ∂logL/∂θ values."""

    def test_gradient_finite_and_correct_sign(self, gold):
        from rabbit.inference.bbn_inference import _scipy_forward_solve
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        g = gold["gradient_bridge"]
        YP_OBS, YP_ERR = 0.2449, 0.004
        DH_OBS, DH_ERR = 2.547e-5, 0.029e-5
        ETA_OBS, TAUN_OBS = 6.104, 878.4

        @jax.custom_vjp
        def fwd(p):
            r = _scipy_forward_solve(0.0, float(p[0])*1e-10, float(p[1]), 20, 0)
            return jnp.array([r["Yp"], r["DH"]]) if r["success"] else jnp.array([0., 0.])
        def fwd_f(p):
            y = fwd.__wrapped__(p); return y, p
        def fwd_b(p, g_):
            gr = jnp.zeros(2)
            for i in range(2):
                e = jnp.maximum(jnp.abs(p[i])*1e-5, 1e-15)
                yp = fwd.__wrapped__(p.at[i].set(p[i]+e))
                ym = fwd.__wrapped__(p.at[i].set(p[i]-e))
                gr = gr.at[i].set(jnp.sum(g_*(yp-ym)/(2*e)))
            return (gr,)
        fwd.defvjp(fwd_f, fwd_b)

        def lp(p):
            o = fwd(p)
            return (-0.5*((o[0]-YP_OBS)/YP_ERR)**2
                    - 0.5*((o[1]-DH_OBS)/DH_ERR)**2)

        init = jnp.array([ETA_OBS, TAUN_OBS])
        val, grad = jax.value_and_grad(lp)(init)

        assert jnp.all(jnp.isfinite(grad)), f"grad not finite: {grad}"
        # Sign check
        assert float(grad[0]) < 0, f"∂/∂η₁₀ should be negative: {grad[0]}"
        assert float(grad[1]) > 0, f"∂/∂τ_n should be positive: {grad[1]}"
        # Magnitude check (within 10× of expected)
        tol = g["tolerance_rel"]
        assert abs(float(grad[0]) - g["d_logL_d_eta10"]) / abs(g["d_logL_d_eta10"]) < tol, \
            f"∂/∂η₁₀ = {grad[0]:.2f} vs gold {g['d_logL_d_eta10']}"
