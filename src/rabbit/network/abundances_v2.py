"""
rabbit.network.abundances_v2 — PRIMAT AC2024 12-reaction nuclear network.

12 reactions, 8 species (n, p, D, T, ³He, ⁴He, ⁷Li, ⁷Be).
All rates from PRIMAT BBNRatesAC2024_small_network.dat.

Reactions:
  R0:  n + p → D + γ
  R1:  D + p → ³He + γ
  R2:  D + D → n + ³He
  R3:  D + D → p + T
  R4:  T + D → n + ⁴He
  R5:  T + ⁴He → ⁷Li + γ
  R6:  ³He + n → p + T
  R7:  ³He + D → p + ⁴He
  R8:  ³He + ⁴He → ⁷Be + γ
  R9:  ⁷Li + p → ⁴He + ⁴He
  R10: ⁷Be + n → ⁷Li + p
  R11: T + p → ⁴He + γ  (NEW vs v1 backbone)
"""
import json, pathlib
import numpy as np
from scipy.interpolate import interp1d

N_SPECIES = 8
N_REACTIONS = 12
SPECIES_NAMES = ('n', 'p', 'D', 'T', '3He', '4He', '7Li', '7Be')
ATOMIC_MASSES = np.array([1, 1, 2, 3, 3, 4, 7, 7], dtype=float)

# Stoichiometry: S[species, reaction]
#   R0:  n+p→D+γ       R4: T+D→n+⁴He     R8:  ³He+⁴He→⁷Be+γ
#   R1:  D+p→³He+γ     R5: T+⁴He→⁷Li+γ   R9:  ⁷Li+p→⁴He+⁴He
#   R2:  D+D→n+³He     R6: ³He+n→p+T      R10: ⁷Be+n→⁷Li+p
#   R3:  D+D→p+T       R7: ³He+D→p+⁴He    R11: T+p→⁴He+γ
STOICHIOMETRY = np.array([
    # R0   R1   R2   R3   R4   R5   R6   R7   R8   R9   R10  R11
    [-1,   0,  +1,   0,  +1,   0,  -1,   0,   0,   0,  -1,   0],  # n
    [-1,  -1,   0,  +1,   0,   0,  +1,  +1,   0,  -1,  +1,  -1],  # p
    [+1,  -1,  -2,  -2,  -1,   0,   0,  -1,   0,   0,   0,   0],  # D
    [ 0,   0,   0,  +1,  -1,  -1,  +1,   0,   0,   0,   0,  -1],  # T
    [ 0,  +1,  +1,   0,   0,   0,  -1,  -1,  -1,   0,   0,   0],  # ³He
    [ 0,   0,   0,   0,  +1,  -1,   0,  +1,  -1,  +2,   0,  +1],  # ⁴He
    [ 0,   0,   0,   0,   0,  +1,   0,   0,   0,  -1,  +1,   0],  # ⁷Li
    [ 0,   0,   0,   0,   0,   0,   0,   0,  +1,   0,  -1,   0],  # ⁷Be
], dtype=float)

_MEV_TO_T9 = 11.6045  # 1 MeV = 11.6045 GK
_K_B_GK = 0.086173    # k_B in MeV/GK

# ═══════════════════════════════════════════════════════════════
# Rate table loading
# ═══════════════════════════════════════════════════════════════

_TABLE = None

def _load_table():
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    path = pathlib.Path(__file__).parent / 'data/primat_ac2024_12rxn.json'
    with open(path) as f:
        data = json.load(f)
    T9 = np.array(data['T9'])
    log_T9 = np.log(T9)
    lr = np.array(data['log_rates']['data']).reshape(data['log_rates']['shape'])
    # Build interpolators for each reaction
    interps = []
    for i in range(N_REACTIONS):
        interps.append(interp1d(log_T9, lr[i], kind='linear',
                                bounds_error=False,
                                fill_value=(lr[i, 0], lr[i, -1])))
    _TABLE = {
        'T9': T9, 'log_T9': log_T9, 'log_rates': lr, 'interps': interps,
        'Q': np.array(data['Q_MeV']),
        'rev': np.array(data['rev_factor']),
        'T9p': np.array(data['T9_power']),
        'gamma': np.array(data['gamma']),
    }
    return _TABLE


def evaluate_nuclear_rates(T_gamma_MeV):
    """Evaluate all 12 forward+reverse rates at given T_γ [MeV]."""
    tab = _load_table()
    T9 = max(T_gamma_MeV * _MEV_TO_T9, 1e-30)
    lT9 = np.log(T9)

    fwd = np.zeros(N_REACTIONS)
    rev = np.zeros(N_REACTIONS)

    for i in range(N_REACTIONS):
        fwd[i] = np.exp(float(tab['interps'][i](lT9)))
        Q = tab['Q'][i]
        rf = tab['rev'][i]
        tp = tab['T9p'][i]
        gm = tab['gamma'][i]
        if rf > 0 and abs(Q) > 0:
            log_Keq = np.log(rf) + tp * np.log(T9) + gm / T9
            rev[i] = fwd[i] * np.exp(np.clip(log_Keq, -500, 500))

    return fwd, rev


# ═══════════════════════════════════════════════════════════════
# Abundance evolution
# ═══════════════════════════════════════════════════════════════

def abundance_rhs_phase1(X_n, lambda_np, lambda_pn):
    """Phase 1: weak rates only, dX_n/dt."""
    return -lambda_np * X_n + lambda_pn * (1.0 - X_n)


def compute_fluxes(X, T_gamma, eta):
    """Net reaction fluxes for all 12 reactions."""
    fwd, rev = evaluate_nuclear_rates(T_gamma)

    zeta3 = 1.202056903
    n_gamma = 2.0 * zeta3 / np.pi**2 * T_gamma**3
    n_b = eta * n_gamma
    hbar_c_cm = 197.3269804e-13
    n_b_cgs = n_b / hbar_c_cm**3
    N_A = 6.02214076e23
    rho_fac = n_b_cgs / N_A

    Y = X / ATOMIC_MASSES
    flux = np.zeros(N_REACTIONS)

    flux[0]  = rho_fac * fwd[0]*Y[0]*Y[1] - rev[0]*Y[2]           # n+p→D
    flux[1]  = rho_fac * fwd[1]*Y[2]*Y[1] - rev[1]*Y[4]           # D+p→³He
    flux[2]  = rho_fac * (0.5*fwd[2]*Y[2]**2 - rev[2]*Y[0]*Y[4])  # D+D→n+³He
    flux[3]  = rho_fac * (0.5*fwd[3]*Y[2]**2 - rev[3]*Y[1]*Y[3])  # D+D→p+T
    flux[4]  = rho_fac * (fwd[4]*Y[3]*Y[2] - rev[4]*Y[0]*Y[5])    # T+D→n+⁴He
    flux[5]  = rho_fac * fwd[5]*Y[3]*Y[5] - rev[5]*Y[6]           # T+⁴He→⁷Li
    flux[6]  = rho_fac * (fwd[6]*Y[4]*Y[0] - rev[6]*Y[1]*Y[3])    # ³He+n→p+T
    flux[7]  = rho_fac * (fwd[7]*Y[4]*Y[2] - rev[7]*Y[1]*Y[5])    # ³He+D→p+⁴He
    flux[8]  = rho_fac * fwd[8]*Y[4]*Y[5] - rev[8]*Y[7]           # ³He+⁴He→⁷Be
    flux[9]  = rho_fac * (fwd[9]*Y[6]*Y[1] - 0.5*rev[9]*Y[5]**2)  # ⁷Li+p→2⁴He
    flux[10] = rho_fac * (fwd[10]*Y[7]*Y[0] - rev[10]*Y[1]*Y[6])  # ⁷Be+n→⁷Li+p
    flux[11] = rho_fac * fwd[11]*Y[3]*Y[1] - rev[11]*Y[5]         # T+p→⁴He

    return flux


def abundance_rhs_phase2(X, T_gamma, eta, lambda_np, lambda_pn):
    """Phase 2 RHS: 12-reaction network + weak rates."""
    flux = compute_fluxes(X, T_gamma, eta)
    dX = (STOICHIOMETRY @ flux) * ATOMIC_MASSES
    dXn_weak = -lambda_np * X[0] + lambda_pn * X[1]
    dX[0] += dXn_weak
    dX[1] -= dXn_weak
    return dX


def phase1_to_phase2(X_n):
    """Phase 2 IC from neutron fraction."""
    X = np.zeros(N_SPECIES)
    X[0] = X_n; X[1] = 1.0 - X_n
    X[2] = 1e-25; X[3] = 1e-25; X[4] = 1e-25; X[5] = 1e-25
    X[6] = 1e-30; X[7] = 1e-30
    return X


def mass_conservation_residual(X):
    """Mass-fraction conservation: |Σ X_i − 1|.

    The evolved network variables X_i are mass fractions, so baryon-number
    conservation is equivalent to Σ_i X_i = 1.
    """
    return abs(float(np.sum(X) - 1.0))


def normalization_residual(X):
    """Alias of the mass-fraction conservation residual."""
    return mass_conservation_residual(X)
