"""
rabbit.drivers.tilted_flrw_driver — Tilted FLRW BBN driver (v2).

MATURITY: EXPLORATORY
  Reduced model (FLRW background + tilt perturbation).
  No end-to-end validation against full coupled driver.

Three physics upgrades over v1 for proper tilted baseline:
  1. FULL NONLINEAR RHS: (2-q)(1-(γ-1)v²)/G replaces simplified bracket
  2. CORRECTION-AWARE WEAK RATES: correction_level=0/1/2/3
  3. BARYON-FRAME BOOST: neutrino FD boosted to plasma frame before
     monopole extraction for weak rate computation (Gate G_TILT)

Architecture (dual-frame, per Doc 6):
  - Geometry normal frame: tilt ODE, Hubble-normalized variables
  - Baryon/plasma frame: weak rate computation
  - Boost wrapper connects the two

References:
    Hewitt, Bridson, Wainwright (2001) GRG 33, 65
    Wainwright & Ellis (1997) Ch. 7
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

_G_N = 6.70883e-45; _MEV_TO_S = 1.519267447e21
_M_E = 0.5109989500; _Q_NP = 1.29333236

@dataclass
class TiltedFLRWConfig:
    v0: float = 0.0
    T_start: float = 10.0; T_handoff: float = 0.08; T_end: float = 0.005
    tau_n: float = 878.4; eta: float = 6.104e-10; N_eff: float = 3.044
    gamma: float = 4.0/3.0; rtol: float = 1e-10; atol: float = 1e-12
    correction_level: int = 0
    enable_baryon_boost: bool = False

def _rho_radiation(T_gamma, N_eff):
    rho_g = (np.pi**2/15.0)*T_gamma**4
    T_nu = T_gamma*(4.0/11.0)**(1.0/3.0)
    rho_nu = N_eff*(7.0/8.0)*(np.pi**2/15.0)*T_nu**4
    return rho_g + rho_nu

def _hubble_tilted(T_gamma, N_eff, v_sq):
    rho = _rho_radiation(T_gamma, N_eff)
    H_flrw = np.sqrt(max((8*np.pi*_G_N/3)*rho, 0))*_MEV_TO_S
    return H_flrw / np.sqrt(max(1-v_sq, 1e-10))

# ── Baryon-frame boost ──

def _boost_fd_monopole(T_nu_d, v, N_mu=16):
    """Plasma-frame monopole of neutrinos boosted from geometry frame.

    The neutrinos have temperature T_ν in the geometry frame (FD at T_ν).
    The plasma moves at velocity v relative to geometry frame.
    This function returns f₀^{plasma}(E^{pl}) — the angle-averaged
    occupation number as a function of PLASMA-FRAME energy.

    Input `eps` to the returned callable is plasma-frame energy in m_e units.
    The Laguerre quadrature nodes in the channel integrals use the
    geometry-frame T_ν as scale parameter — this is fine because the
    quadrature covers the full energy range regardless of scale choice.

    Physics: E_geo = Γ × E_pl × (1 + v × μ_pl), then
      f₀^{pl}(E_pl) = ½∫dμ_pl f_FD(E_geo(E_pl, μ_pl) / T_ν)
    """
    if abs(v) < 1e-12:
        def f0(eps):
            eps = np.asarray(eps, dtype=float)
            if T_nu_d < 1e-30: return np.zeros_like(eps)
            return 1.0/(np.exp(np.clip(eps/T_nu_d, 0, 500))+1)
        return f0
    from numpy.polynomial.legendre import leggauss
    mu_n, mu_w = leggauss(N_mu)
    Gamma = 1.0/np.sqrt(1-v**2)
    def f0(eps):
        eps = np.asarray(eps, dtype=float)
        result = np.zeros_like(eps)
        for mu, w in zip(mu_n, mu_w):
            E_g = Gamma * eps * (1 + v*mu)
            if T_nu_d > 1e-30:
                result += 0.5*w/(np.exp(np.clip(E_g/T_nu_d, 0, 500))+1)
            # else: stays 0
        return np.clip(result, 0, 1)
    return f0

# ── Weak rates with full correction stack ──

def _weak_rates_corrected(T_gamma, T_nu, tau_n, v=0.0,
                           correction_level=0, enable_boost=False):
    T_e = T_gamma/_M_E; T_nu_d = T_nu/_M_E; q = _Q_NP/_M_E
    from numpy.polynomial.laguerre import laggauss
    from numpy.polynomial.legendre import leggauss
    N_lag = 24; lag_n, lag_w = laggauss(N_lag)
    leg_n, leg_w = leggauss(24)

    # Neutrino monopole: boosted or geometry-frame
    if enable_boost and abs(v) > 1e-12:
        f_nu = _boost_fd_monopole(T_nu_d, v, N_mu=16)
    else:
        def f_nu(eps):
            eps = np.asarray(eps, dtype=float)
            if T_nu_d < 1e-30: return np.zeros_like(eps)
            return 1.0/(np.exp(np.clip(eps/T_nu_d, 0, 500))+1)

    def f_e(eps):
        eps = np.asarray(eps, dtype=float)
        if T_e < 1e-30: return np.zeros_like(eps)
        return 1.0/(np.exp(np.clip(eps/T_e, 0, 500))+1)

    # Correction factor
    corr_fns = {}
    if correction_level >= 1:
        from weak_corrections import weak_correction_factor
        ec = True; er = correction_level >= 2
        for ch in 'abcdef':
            _ch, _ec, _er = ch, ec, er
            corr_fns[ch] = lambda ee, enu, c=_ch, ec_=_ec, er_=_er: \
                weak_correction_factor(ee, c, enable_coulomb=ec_, enable_radiative=er_)
    if correction_level >= 3:
        from weak_finite_mass import finite_mass_scalar_correction
        base = dict(corr_fns)
        for ch in 'abcdef':
            prev = base.get(ch)
            _ch = ch
            if prev:
                corr_fns[ch] = lambda ee, enu, c=_ch, p=prev: \
                    p(ee, enu) * finite_mass_scalar_correction(ee, enu, c)
            else:
                corr_fns[ch] = lambda ee, enu, c=_ch: \
                    finite_mass_scalar_correction(ee, enu, c)

    def corr(ee, enu, ch):
        if ch in corr_fns: return corr_fns[ch](ee, enu)
        return np.ones_like(np.asarray(ee, dtype=float))

    # I₀ with matching corrections — MUST include ALL active corrections
    if correction_level == 0:
        I0 = 1.636100
    elif correction_level <= 2:
        from weak_corrections import compute_I0_corrected
        I0 = compute_I0_corrected(True, correction_level >= 2)
    else:
        # Level 3: Coulomb + Sirlin + FM/WM combined on I₀
        # Must use all three together for consistent normalization.
        from numpy.polynomial.legendre import leggauss as _lg
        from weak_corrections import (
            fermi_sommerfeld as _fs, radiative_correction_factor as _rc,
            W0_NEUTRON_DECAY as _W0,
        )
        from weak_finite_mass import (
            finite_mass_scalar_correction as _fm, _Q_DIMLESS as _qd,
        )
        _xq, _wq = _lg(64)
        _W = 0.5*(_qd-1)*_xq + 0.5*(_qd+1)
        _jac = 0.5*(_qd-1)
        _p = np.sqrt(np.maximum(_W**2-1, 0))
        _Enu = _qd - _W
        _integ = _W * _Enu**2 * _p
        _integ *= _fs(_W, Z=+1) * _rc(_W, _qd) * _fm(_W, _Enu, 'c')
        I0 = float(_jac * np.sum(_wq * _integ))

    # Explicit per-channel computation
    # Channel a: ν_e + n → p + e⁻, eps_nu = lag*T_nu, eps_e = eps_nu + q
    eps_nu_a = lag_n * T_nu_d; eps_e_a = eps_nu_a + q
    ma = eps_e_a > 1
    I_a = 0.0
    if np.any(ma):
        en, ee = eps_nu_a[ma], eps_e_a[ma]
        pe = np.sqrt(ee**2-1); ps = en**2*ee*pe
        I_a = T_nu_d*np.sum(lag_w[ma]*ps*f_nu(en)*(1-f_e(ee))*corr(ee,en,'a')*np.exp(lag_n[ma]))

    # Channel b: e⁺ + n → p + ν̄_e, eps_e = 1+lag*T_e, eps_nu = eps_e+q
    eps_e_b = 1+lag_n*T_e; eps_nu_b = eps_e_b+q
    pe_b = np.sqrt(eps_e_b**2-1); ps_b = eps_nu_b**2*eps_e_b*pe_b
    I_b = T_e*np.sum(lag_w*ps_b*f_e(eps_e_b)*(1-f_nu(eps_nu_b))*corr(eps_e_b,eps_nu_b,'b')*np.exp(lag_n))

    # Channel c: n → p + e⁻ + ν̄_e (bounded)
    eps_e_c = 0.5*(q-1)*leg_n+0.5*(q+1); eps_nu_c = q-eps_e_c
    mc = (eps_nu_c>0)&(eps_e_c>1); I_c = 0.0
    if np.any(mc):
        ec, nc = eps_e_c[mc], eps_nu_c[mc]
        I_c = 0.5*(q-1)*np.sum(leg_w[mc]*nc**2*ec*np.sqrt(ec**2-1)*(1-f_e(ec))*(1-f_nu(nc))*corr(ec,nc,'c'))

    # Channel d: p + e⁻ → n + ν_e, eps_e = q+lag*T_e, eps_nu = eps_e-q
    eps_e_d = q+lag_n*T_e; eps_nu_d = eps_e_d-q
    md = (eps_nu_d>0)&(eps_e_d>1); I_d = 0.0
    if np.any(md):
        ed, nd = eps_e_d[md], eps_nu_d[md]
        I_d = T_e*np.sum(lag_w[md]*nd**2*ed*np.sqrt(ed**2-1)*f_e(ed)*(1-f_nu(nd))*corr(ed,nd,'d')*np.exp(lag_n[md]))

    # Channel e: p + ν̄_e → n + e⁺, eps_nu = (q+1)+lag*T_nu, eps_e = eps_nu-q
    eps_nu_e = (q+1)+lag_n*T_nu_d; eps_e_e = eps_nu_e-q
    me = eps_e_e > 1; I_e = 0.0
    if np.any(me):
        ne, ee = eps_nu_e[me], eps_e_e[me]
        I_e = T_nu_d*np.sum(lag_w[me]*ne**2*ee*np.sqrt(ee**2-1)*f_nu(ne)*(1-f_e(ee))*corr(ee,ne,'e')*np.exp(lag_n[me]))

    # Channel f: p + e⁻ + ν̄_e → n (bounded)
    eps_e_f = 0.5*(q-1)*leg_n+0.5*(q+1); eps_nu_f = q-eps_e_f
    mf = (eps_nu_f>0)&(eps_e_f>1); I_f = 0.0
    if np.any(mf):
        ef, nf_ = eps_e_f[mf], eps_nu_f[mf]
        I_f = 0.5*(q-1)*np.sum(leg_w[mf]*nf_**2*ef*np.sqrt(ef**2-1)*f_e(ef)*f_nu(nf_)*corr(ef,nf_,'f'))

    lnp = max((I_a+I_b+I_c)/(I0*tau_n), 1/tau_n)
    lpn = max((I_d+I_e+I_f)/(I0*tau_n), 0.0)
    return lnp, lpn

# ── Full nonlinear tilt RHS ──

def _tilt_rhs_exact(v, gamma, q_decel=1.0):
    """dv/dN = v(1-v²)/G × [(2-q)(1-(γ-1)v²)/G − (3γ-4)]"""
    v_sq = v**2
    if v_sq >= 0.99: return 0.0
    G = 1+(gamma-1)*v_sq
    friction = (2-q_decel)*(1-(gamma-1)*v_sq)/G
    return v*(1-v_sq)/G*(friction-(3*gamma-4))

# ── ODE system ──

def _tilted_flrw_rhs_phase1(N, y, config):
    v, T_gamma, Xn = y[0], y[1], y[2]
    if T_gamma < 1e-6: return np.zeros(3)
    dv = _tilt_rhs_exact(v, config.gamma)
    dT = -T_gamma/3.0
    T_nu = T_gamma*(4/11)**(1/3)
    H = _hubble_tilted(T_gamma, config.N_eff, v**2)
    lnp, lpn = _weak_rates_corrected(T_gamma, T_nu, config.tau_n, v=v,
        correction_level=config.correction_level, enable_boost=config.enable_baryon_boost)
    dXn = (-lnp*Xn + lpn*(1-Xn))/max(H, 1e-100)
    return np.array([dv, dT, dXn])

# ── Result + entry point ──

@dataclass
class TiltedFLRWResult:
    Yp: float; v_final: float; v_max: float; amplification: float
    N_efolds: float; delta_Yp: float; tilt_eigenvalue: float
    correction_level: int; enable_boost: bool; trajectory: dict

def run_tilted_flrw(config=None):
    if config is None: config = TiltedFLRWConfig()
    eigenvalue = 7-4.5*config.gamma
    T_nu_i = config.T_start*(4/11)**(1/3)
    lnp0, lpn0 = _weak_rates_corrected(config.T_start, T_nu_i, config.tau_n,
        correction_level=config.correction_level, enable_boost=False)
    Xn_eq = lpn0/(lnp0+lpn0) if (lnp0+lpn0)>0 else 0.5
    y0 = np.array([config.v0, config.T_start, Xn_eq])
    def stop(N,y): return y[1]-config.T_handoff
    stop.terminal=True; stop.direction=-1
    sol = solve_ivp(lambda N,y: _tilted_flrw_rhs_phase1(N,y,config),
        [0,30], y0, events=stop, rtol=config.rtol, atol=config.atol, method='Radau')
    if not sol.success and len(sol.t)<2: raise RuntimeError(sol.message)
    v_f, Yp = float(sol.y[0,-1]), 2*float(sol.y[2,-1])
    # FLRW baseline at SAME correction level
    cfg0 = TiltedFLRWConfig(v0=0, T_start=config.T_start, T_handoff=config.T_handoff,
        tau_n=config.tau_n, eta=config.eta, N_eff=config.N_eff, gamma=config.gamma,
        correction_level=config.correction_level, enable_baryon_boost=False)
    y0f = np.array([0, config.T_start, Xn_eq])
    solf = solve_ivp(lambda N,y: _tilted_flrw_rhs_phase1(N,y,cfg0),
        [0,30], y0f, events=stop, rtol=config.rtol, atol=config.atol, method='Radau')
    Yp_f = 2*float(solf.y[2,-1])
    amp = abs(v_f/config.v0) if abs(config.v0)>1e-30 else 0
    return TiltedFLRWResult(Yp=Yp, v_final=v_f, v_max=float(np.max(np.abs(sol.y[0]))),
        amplification=amp, N_efolds=float(sol.t[-1]), delta_Yp=Yp-Yp_f,
        tilt_eigenvalue=eigenvalue, correction_level=config.correction_level,
        enable_boost=config.enable_baryon_boost,
        trajectory={'N':sol.t,'v':sol.y[0],'T_gamma':sol.y[1],'Xn':sol.y[2]})
