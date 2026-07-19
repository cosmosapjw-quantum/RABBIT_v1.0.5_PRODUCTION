"""
rabbit.network.abundances — Original 11-reaction BBN backbone.

Species (8): n, p, D, T, ³He, ⁴He, ⁷Li, ⁷Be
Reactions (11): PRIMAT backbone from primat_rates_12.json (v3.1 legacy).

This is the ORIGINAL module restored after accidental corruption.
For the canonical 12-reaction PRIMAT AC2024 network, use abundances_v2.py.
"""
from __future__ import annotations
import numpy as np
import json, os
from scipy.interpolate import interp1d

N_SPECIES = 8
N_REACTIONS = 11
SPECIES_NAMES = ('n', 'p', 'D', 'T', '3He', '4He', '7Li', '7Be')
ATOMIC_MASSES = np.array([1, 1, 2, 3, 3, 4, 7, 7], dtype=float)

_MEV_TO_T9 = 11.604519

# Stoichiometry: S[species, reaction]
STOICHIOMETRY = np.array([
    # R0   R1   R2   R3   R4   R5   R6   R7   R8   R9  R10
    [-1,   0,  +1,   0,  +1,   0,  -1,   0,   0,   0,  -1],  # n
    [-1,  -1,   0,  +1,   0,   0,  +1,  +1,   0,  -1,  +1],  # p
    [+1,  -1,  -2,  -2,  -1,   0,   0,  -1,   0,   0,   0],  # D
    [ 0,   0,   0,  +1,  -1,  -1,  +1,   0,   0,   0,   0],  # T
    [ 0,  +1,  +1,   0,   0,   0,  -1,  -1,  -1,   0,   0],  # ³He
    [ 0,   0,   0,   0,  +1,  -1,   0,  +1,  -1,  +2,   0],  # ⁴He
    [ 0,   0,   0,   0,   0,  +1,   0,   0,   0,  -1,  +1],  # ⁷Li
    [ 0,   0,   0,   0,   0,   0,   0,   0,  +1,   0,  -1],  # ⁷Be
], dtype=float)

_RATE_CACHE = None

def _load_rate_table():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(here, 'primat_rates_12.json'),
                 os.path.join(here, '..', '..', 'primat_rates_12.json'),
                 '/mnt/project/primat_rates_12.json']:
        if os.path.exists(path):
            with open(path) as f:
                jdata = json.load(f)
            def from_json(d):
                if isinstance(d, dict) and 'data' in d:
                    arr = np.array(d['data'], dtype=d.get('dtype', 'float64'))
                    if 'shape' in d: arr = arr.reshape(d['shape'])
                    return arr
                return np.array(d)
            return {k: from_json(jdata[k]) for k in
                    ['T9', 'log_T9', 'log_rates', 'sigmas', 'Q_MeV',
                     'rev_factor', 'T9_power']}
    raise FileNotFoundError("primat_rates_12.json not found")

def _get_rate_table():
    global _RATE_CACHE
    if _RATE_CACHE is None:
        _RATE_CACHE = _load_rate_table()
    return _RATE_CACHE

def evaluate_nuclear_rates(T_gamma_MeV):
    """Evaluate 11 forward + reverse rates at T_γ [MeV]."""
    tab = _get_rate_table()
    T9 = max(T_gamma_MeV * _MEV_TO_T9, 1e-30)
    lT9 = np.log(T9)
    fwd = np.zeros(N_REACTIONS)
    rev = np.zeros(N_REACTIONS)
    for i in range(N_REACTIONS):
        lr = np.interp(lT9, tab['log_T9'], tab['log_rates'][i])
        fwd[i] = np.exp(lr)
        Q = tab['Q_MeV'][i]; rf = tab['rev_factor'][i]; tp = tab['T9_power'][i]
        if rf > 0 and abs(Q) > 0:
            log_Keq = np.log(rf) + tp*np.log(T9) + (-Q/0.086173)/T9
            rev[i] = fwd[i] * np.exp(np.clip(log_Keq, -500, 500))
    return fwd, rev

def abundance_rhs_phase1(X_n, lambda_np, lambda_pn):
    return -lambda_np * X_n + lambda_pn * (1.0 - X_n)

def abundance_rhs_phase2(X, T_gamma, eta, lambda_np, lambda_pn):
    fwd, rev = evaluate_nuclear_rates(T_gamma)
    zeta3 = 1.202056903
    n_b_cgs = eta * 2.0*zeta3/np.pi**2 * T_gamma**3 / 197.3269804e-13**3
    rho_fac = n_b_cgs / 6.02214076e23
    Y = X / ATOMIC_MASSES
    flux = np.zeros(N_REACTIONS)
    flux[0]  = rho_fac*fwd[0]*Y[0]*Y[1] - rev[0]*Y[2]
    flux[1]  = rho_fac*fwd[1]*Y[2]*Y[1] - rev[1]*Y[4]
    flux[2]  = rho_fac*(0.5*fwd[2]*Y[2]**2 - rev[2]*Y[0]*Y[4])
    flux[3]  = rho_fac*(0.5*fwd[3]*Y[2]**2 - rev[3]*Y[1]*Y[3])
    flux[4]  = rho_fac*(fwd[4]*Y[3]*Y[2] - rev[4]*Y[0]*Y[5])
    flux[5]  = rho_fac*fwd[5]*Y[3]*Y[5] - rev[5]*Y[6]
    flux[6]  = rho_fac*(fwd[6]*Y[4]*Y[0] - rev[6]*Y[1]*Y[3])
    flux[7]  = rho_fac*(fwd[7]*Y[4]*Y[2] - rev[7]*Y[1]*Y[5])
    flux[8]  = rho_fac*fwd[8]*Y[4]*Y[5] - rev[8]*Y[7]
    flux[9]  = rho_fac*(fwd[9]*Y[6]*Y[1] - 0.5*rev[9]*Y[5]**2)
    flux[10] = rho_fac*(fwd[10]*Y[7]*Y[0] - rev[10]*Y[1]*Y[6])
    dX = (STOICHIOMETRY @ flux) * ATOMIC_MASSES
    dXn_weak = -lambda_np*X[0] + lambda_pn*X[1]
    dX[0] += dXn_weak; dX[1] -= dXn_weak
    return dX

def phase1_to_phase2(X_n):
    X = np.zeros(N_SPECIES)
    X[0] = X_n; X[1] = 1.0 - X_n
    X[2]=1e-25; X[3]=1e-25; X[4]=1e-25; X[5]=1e-25; X[6]=1e-30; X[7]=1e-30
    return X

def mass_conservation_residual(X):
    return abs(np.sum(X) - 1.0)
