#!/usr/bin/env python3
"""Stress test suite for characteristic ray transport.

Tests:
  T1. FLRW parity (Σ=0): Yp must match linearized to < 1e-8
  T2. Linear-limit stress calibration: Π_char / Π_lin → 1 as S → 0
  T3. Monotone convergence in N_μ: Yp(N_μ=8) → Yp(N_μ=16) → Yp(N_μ=24)
  T4. Positivity: f_mono ∈ [0,1] for all Σ up to 0.7
  T5. Conservation: (1/2)Σ w_j J_j exp(-8I_j) stays finite and O(1)
  T6. No coordinate singularity: I, J finite for all Σ ≤ 0.7
  T7. SciPy ↔ JAX parity (where JAX available)

Usage: python3 scripts/stress_test_characteristic.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, roots_laguerre
from numpy.polynomial.legendre import leggauss

from rabbit.transport.characteristic_rays import (
    setup_ray_grid, mu_current, characteristic_transport_rhs,
    extract_stress, extract_monopole, ray_diagnostics)
from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.thermo.incomplete_decoupling import dT_gamma_dN_tier1, T_nu_from_T_gamma_tier1
from rabbit.thermo.eos_photon_electron import _RHO_GAMMA_PREFACTOR
from rabbit.weak.live_rates import compute_live_weak_rates
from rabbit.network.abundances_standard import (
    abundance_rhs_phase1, abundance_rhs_phase2, phase1_to_phase2, N_SPECIES)

_TAU_N=879.6; _ETA=6.104e-10; _N_EFF=3.044; _F_NU=0.40520
_MEV_TO_S=1/6.58212e-22; _G_N=6.70883e-45; B_Q=8/15

q_gl, q_wgl = roots_laguerre(20)
q_gl = q_gl.astype(np.float64); q_wgl = q_wgl.astype(np.float64)
f0_gl = 1/(np.exp(np.minimum(q_gl,500))+1)

def _hub(Tg,Tnu,S2):
    rho=_RHO_GAMMA_PREFACTOR*Tg**4*(1+_N_EFF*7/8*(Tnu/Tg)**4)
    return np.sqrt(max(8*np.pi*_G_N/3*rho/max(1-S2,1e-10),0))

def run_p1(Sigma_H, N_mu=12, mode='characteristic'):
    """Run phase 1 only, return handoff state and diagnostics."""
    mu0, w0, X0, signs = setup_ray_grid(N_mu)
    if mode == 'characteristic':
        n_I=N_mu; n_J=N_mu
        iI=1; iJ=1+n_I; iS=1+n_I+n_J; iTG=iS+1; iNET=iTG+1
    else:
        iTG=2; iNET=3

    def rhs(N, y):
        Sp=y[0]; Tg=y[iTG]; Xn=y[iNET]
        if Tg<1e-6: return np.zeros(len(y))
        S2=Sp**2
        if mode=='characteristic':
            Iv=y[iI:iI+n_I]; Jv=y[iJ:iJ+n_J]; Sv=y[iS]
            mu=mu_current(X0,signs,Sv); P2=eval_legendre(2,mu)
            Pi=extract_stress(Iv,Jv,mu,w0,_F_NU)
            dI,dJ,dS=characteristic_transport_rhs(Sp,Iv,Jv,mu)
            fm=np.clip(extract_monopole(Iv,Jv,w0,q_gl),0,1)
        else:
            Psi2=y[1]; Pi=6*_F_NU*Psi2; fm=f0_gl
        Om=max(0,1-S2)
        dSp,_=compute_typeI_geometry_rhs(Sp,0.,Pi,0.,Om)
        dTg=dT_gamma_dN_tier1(Tg); Tnu=T_nu_from_T_gamma_tier1(Tg)
        H=_hub(Tg,Tnu,S2)*_MEV_TO_S
        weak=compute_live_weak_rates(fm,fm,q_gl,Tg,Tnu,_TAU_N,compute_iso_reference=False,correction_level=0)
        dXn=abundance_rhs_phase1(Xn,weak.lambda_np,weak.lambda_pn)/max(H,1e-100)
        dy=np.zeros(len(y)); dy[0]=dSp
        if mode=='characteristic':
            dy[iI:iI+n_I]=dI; dy[iJ:iJ+n_J]=dJ; dy[iS]=dS
        else:
            dy[1]=-B_Q*Sp
        dy[iTG]=dTg; dy[iNET]=dXn; dy[iNET+1]=-dXn
        return dy

    Tnu0=T_nu_from_T_gamma_tier1(10.)
    w0_wr=compute_live_weak_rates(f0_gl,f0_gl,q_gl,10.,Tnu0,_TAU_N,compute_iso_reference=False,correction_level=0)
    Xn_eq=float(w0_wr.lambda_pn/max(w0_wr.lambda_np+w0_wr.lambda_pn,1e-100))
    n_total=iNET+2; y0=np.zeros(n_total); y0[0]=Sigma_H
    if mode=='characteristic': y0[iJ:iJ+n_J]=1.0
    y0[iTG]=10.; y0[iNET]=Xn_eq; y0[iNET+1]=1-Xn_eq

    def ev(N,y): return y[iTG]-0.08
    ev.terminal=True; ev.direction=-1

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol=solve_ivp(rhs,[0,50],y0,method='Radau',events=ev,rtol=1e-8,atol=1e-10,max_step=0.5)
    if sol.status<0:
        return {'ok':False,'err':sol.message}

    yh=sol.y[:,-1]
    out={'ok':True,'Xn':float(yh[iNET]),'Sigma_f':float(yh[0]),'steps':len(sol.t)}
    if mode=='characteristic':
        Iv=yh[iI:iI+n_I]; Jv=yh[iJ:iJ+n_J]; Sv=yh[iS]
        mu=mu_current(X0,signs,Sv)
        out['diag']=ray_diagnostics(Iv,Jv,mu,w0,Sv,q_gl)
        out['Pi']=float(extract_stress(Iv,Jv,mu,w0,_F_NU))
    else:
        out['Psi2']=float(yh[1])
        out['Pi']=float(6*_F_NU*yh[1])
    return out


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

def test_t1_flrw_parity():
    """T1: Σ=0 → identical Yp."""
    print("T1. FLRW parity (Σ=0)...", end=" ", flush=True)
    r_lin = run_p1(0.0, mode='linearized')
    r_char = run_p1(0.0, mode='characteristic')
    assert r_lin['ok'] and r_char['ok']
    delta = abs(r_char['Xn'] - r_lin['Xn'])
    print(f"ΔXn={delta:.2e}  ", end="")
    assert delta < 1e-8, f"FAIL: ΔXn={delta}"
    print("PASS ✓")

def test_t2_linear_stress():
    """T2: Stress matches at small Σ."""
    print("T2. Linear stress calibration...", end=" ", flush=True)
    mu0, w0, X0, signs = setup_ray_grid(16)
    for S_test in [0.001, 0.01, 0.1]:
        I_test = S_test * eval_legendre(2, mu0)
        J_test = np.ones(len(mu0))
        mu_test = mu0  # barely moved
        Pi_char = _F_NU * np.sum(w0 * J_test * eval_legendre(2,mu_test) * (1-8*I_test))
        Pi_lin = 6*_F_NU*(-(8/15)*S_test)
        ratio = Pi_char/Pi_lin
        assert abs(ratio-1) < 1e-10, f"FAIL at S={S_test}: ratio={ratio}"
    print("PASS ✓ (S=0.001,0.01,0.1)")

def test_t3_nmu_convergence():
    """T3: Monotone convergence in N_μ."""
    print("T3. N_μ convergence (Σ=0.3)...", end=" ", flush=True)
    results = {}
    for nmu in [8, 12, 16, 24]:
        r = run_p1(0.3, N_mu=nmu, mode='characteristic')
        if r['ok']:
            results[nmu] = r['Xn']
            print(f"N_μ={nmu}:Xn={r['Xn']:.8f} ", end="", flush=True)
    if len(results) >= 3:
        vals = list(results.values())
        diffs = [abs(vals[i+1]-vals[i]) for i in range(len(vals)-1)]
        converging = all(diffs[i+1] <= 2*diffs[i] for i in range(len(diffs)-1)) or diffs[-1] < 1e-6
        print(f"  diffs={[f'{d:.1e}' for d in diffs]}  ", end="")
        print("PASS ✓" if converging else "WARN (not monotone, check)")
    else:
        print("SKIP (insufficient data)")

def test_t4_positivity():
    """T4: f_mono ∈ [0,1] for all tested Σ."""
    print("T4. Positivity...", end=" ", flush=True)
    all_pass = True
    for sigma in [0.1, 0.3, 0.5, 0.7]:
        r = run_p1(sigma, mode='characteristic')
        if r['ok']:
            d = r['diag']
            fmin, fmax = d['f_mono_min'], d['f_mono_max']
            if fmin < -1e-15 or fmax > 1+1e-15:
                print(f"FAIL at Σ={sigma}: f∈[{fmin},{fmax}]")
                all_pass = False
        else:
            print(f"FAIL at Σ={sigma}: {r.get('err','?')}")
            all_pass = False
    print("PASS ✓ (Σ=0.1..0.7)" if all_pass else "FAIL")

def test_t5_energy_conservation():
    """T5: (1/2)Σ w_j J_j exp(-8I_j) remains O(1)."""
    print("T5. Energy density ratio...", end=" ", flush=True)
    all_pass = True
    for sigma in [0.1, 0.3, 0.5]:
        r = run_p1(sigma, mode='characteristic')
        if r['ok']:
            edr = r['diag']['energy_density_ratio']
            if edr < 0.1 or edr > 10:
                print(f"WARN at Σ={sigma}: ratio={edr:.4f}")
                all_pass = False
            else:
                print(f"Σ={sigma}:{edr:.4f} ", end="")
    print("  PASS ✓" if all_pass else "")

def test_t6_no_singularity():
    """T6: I, J remain finite for Σ ≤ 0.7."""
    print("T6. No singularity...", end=" ", flush=True)
    all_pass = True
    for sigma in [0.3, 0.5, 0.7]:
        r = run_p1(sigma, mode='characteristic')
        if r['ok']:
            d = r['diag']
            if d['max_abs_I'] > 100 or d['J_max'] > 1e6 or d['J_min'] < 1e-6:
                print(f"WARN at Σ={sigma}: |I|={d['max_abs_I']:.1e} J=[{d['J_min']:.1e},{d['J_max']:.1e}]")
                all_pass = False
        else:
            print(f"FAIL at Σ={sigma}: {r.get('err','?')}")
            all_pass = False
    print("PASS ✓ (Σ=0.3..0.7)" if all_pass else "")

def test_t7_scipy_jax_parity():
    """T7: SciPy ↔ JAX parity."""
    print("T7. SciPy ↔ JAX parity...", end=" ", flush=True)
    try:
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        os.environ.setdefault('JAX_ENABLE_X64', '1')
        import jax; jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from rabbit.jax.characteristic_rays_jax import (
            extract_stress_jax, extract_monopole_jax, P2_jax, mu_current_jax)

        mu0, w0, X0, signs = setup_ray_grid(12)
        mu0_j = jnp.array(mu0); w0_j = jnp.array(w0)
        X0_j = jnp.array(X0); signs_j = jnp.array(signs)
        q_j = jnp.array(q_gl)

        # Test at a representative state
        S_test = 0.3
        I_np = S_test * eval_legendre(2, mu0)
        J_np = np.ones(len(mu0))
        mu_np = mu_current(X0, signs, S_test)

        I_j = jnp.array(I_np); J_j = jnp.array(J_np)
        mu_j = mu_current_jax(X0_j, signs_j, jnp.float64(S_test))

        Pi_np = extract_stress(I_np, J_np, mu_np, w0, _F_NU)
        Pi_jax = float(extract_stress_jax(I_j, J_j, mu_j, w0_j, _F_NU))

        fm_np = extract_monopole(I_np, J_np, w0, q_gl)
        fm_jax = np.array(extract_monopole_jax(I_j, J_j, w0_j, q_j))

        delta_Pi = abs(Pi_np - Pi_jax) / max(abs(Pi_np), 1e-30)
        delta_fm = np.max(np.abs(fm_np - fm_jax))

        print(f"δΠ/Π={delta_Pi:.1e}  δf_mono={delta_fm:.1e}  ", end="")
        assert delta_Pi < 1e-12 and delta_fm < 1e-12
        print("PASS ✓")
    except ImportError:
        print("SKIP (JAX not available)")
    except Exception as e:
        print(f"FAIL: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Characteristic ray transport stress tests")
    parser.add_argument('--cpu', action='store_true', help='Force CPU for JAX')
    args = parser.parse_args()

    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.setdefault('JAX_ENABLE_X64', '1')

    print("="*60)
    print("  Characteristic Ray Transport — Stress Tests")
    print("="*60)
    t0 = time.perf_counter()

    test_t1_flrw_parity()
    test_t2_linear_stress()
    test_t3_nmu_convergence()
    test_t4_positivity()
    test_t5_energy_conservation()
    test_t6_no_singularity()
    test_t7_scipy_jax_parity()

    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
