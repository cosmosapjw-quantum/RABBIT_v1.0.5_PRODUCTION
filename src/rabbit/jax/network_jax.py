"""
rabbit.jax.network_jax — JAX-native Standard BBN nuclear network.

9 species, 31 reactions.  ALL rates from PRIMAT AC2024 tabulated data.

Reverse-flux formula (general, for any n-body product):
  flux_rev = rho_fac^{n_prod−1} × sym_fac × rev[r] × Π Y_k^{S_k}
  where n_prod = total product particles, sym_fac = 1/Π(S_k!).

  n_prod=1 (radiative capture): flux_rev = rev × Y_product  (no rho_fac)
  n_prod=2 (2→2 reverse):      flux_rev = rho_fac × sym × rev × Y_prod
  n_prod≥3 (multi-body):        flux_rev = rho_fac² × sym × rev × Y_prod

Species (9): n, p, D, T, ³He, ⁴He, ⁷Li, ⁷Be, ⁶Li
"""
import json
import pathlib
import math
import jax
import jax.numpy as jnp
import numpy as _np

from rabbit.validation.truncation_guards import (
    validate_manual_network_truncation,
    validate_rate_table_temperature,
)

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
# §1. Constants
# ═══════════════════════════════════════════════════════════════════════

N_SPECIES = 9
N_REACTIONS_FULL = 31
N_REACTIONS_BACKBONE = 12

def _readonly_array(values, *, dtype=float):
    array = _np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


ATOMIC_MASSES = _readonly_array([1, 1, 2, 3, 3, 4, 7, 7, 6])

STOICHIOMETRY = _readonly_array([
    [ -1,  0, +1,  0,  0, +1,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1],
    [ -1, -1,  0, +1, -1,  0,  0, +1, +1,  0, +1, -1, -1,  0, +1,  0, -1, -1, +1,  0,  0,  0,  0,  0,  0,  0,  0, +2,  0, +2,  0],
    [ +1, -1, -2, -2,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0, +1, +1, +1, +1,  0,  0, -2,  0, -1],
    [  0,  0,  0, +1, -1, -1, -1, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1,  0,  0, -1, -1,  0,  0,  0,  0],
    [  0, +1, +1,  0,  0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, +1, -1,  0, -1,  0,  0, -1, -1,  0, +1, -1,  0, -2,  0],
    [  0,  0,  0,  0, +1, +1, -1,  0, +1, -1,  0, +2, +2, +2, +2, -1,  0, +1, +2, +2, +1, +1,  0,  0, +2, +2,  0, +2, +1, +1, +2],
    [  0,  0,  0,  0,  0,  0, +1,  0,  0,  0, +1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0, +1,  0, -1,  0, +1,  0,  0,  0, -1],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, -1,  0,  0, -1, -1,  0, +1,  0,  0,  0,  0, -1,  0, +1,  0, -1, -1, -1,  0,  0,  0],
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, -1, -1, -1, -1, +1, +1, -1, -1,  0,  0,  0,  0,  0,  0,  0],
])

REACTANT_PAIRS = _readonly_array([
    [0,1],[2,1],[2,2],[2,2],[3,1],[3,2],[3,5],[4,0],
    [4,2],[4,5],[7,0],[6,1],[6,1],[7,0],[7,2],[2,5],
    [8,1],[8,1],[8,4],[8,3],[6,4],[7,3],[8,3],[8,4],
    [6,4],[7,3],[7,3],[7,4],[2,2],[4,4],[6,2],
], dtype=_np.int32)

# Forward identical-particle factors
_if = _np.ones(N_REACTIONS_FULL)
_if[2] = 0.5; _if[3] = 0.5; _if[28] = 0.5; _if[29] = 0.5
IDENTICAL_FACTOR = _readonly_array(_if)

# ── Reverse-flux precomputed arrays ──
# Product stoichiometry (positive entries only)
_S_np = STOICHIOMETRY
PRODUCT_STOICH = _readonly_array(_np.where(STOICHIOMETRY > 0, STOICHIOMETRY, 0.0))

# n_product_particles = Σ max(S_k, 0) per reaction
_n_prod = _np.sum(_np.maximum(_S_np, 0), axis=0).astype(int)

# rho_fac power for reverse: max(n_prod − 1, 0)
REVERSE_RHO_POWER = _readonly_array(_np.maximum(_n_prod - 1, 0))

# Symmetry factor: 1 / Π_{S_k>0} S_k!
_sym = _np.ones(N_REACTIONS_FULL)
for _r in range(N_REACTIONS_FULL):
    for _k in range(N_SPECIES):
        _sk = max(int(_S_np[_k, _r]), 0)
        if _sk > 1:
            _sym[_r] /= math.factorial(_sk)
REVERSE_SYMMETRY = _readonly_array(_sym)

MEV_TO_T9 = 11.6045


# ═══════════════════════════════════════════════════════════════════════
# §2. Rate table
# ═══════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class RateTable:
    __slots__ = ('log_T9', 'log_rates', 'Q', 'rev_factor', 'T9_power',
                 'gamma', 'n_reactions', 'segment_slopes')

    def __init__(self, log_T9, log_rates, Q, rev_factor, T9_power, gamma,
                 n_reactions, segment_slopes):
        object.__setattr__(self, 'log_T9', jnp.array(log_T9))
        object.__setattr__(self, 'log_rates', jnp.array(log_rates))
        object.__setattr__(self, 'Q', jnp.array(Q))
        object.__setattr__(self, 'rev_factor', jnp.array(rev_factor))
        object.__setattr__(self, 'T9_power', jnp.array(T9_power))
        object.__setattr__(self, 'gamma', jnp.array(gamma))
        object.__setattr__(self, 'n_reactions', int(n_reactions))
        object.__setattr__(self, 'segment_slopes', jnp.array(segment_slopes))

    def __setattr__(self, name, value):
        raise AttributeError("RateTable is frozen.")

    def tree_flatten(self):
        children = (self.log_T9, self.log_rates, self.Q, self.rev_factor, self.T9_power, self.gamma, self.segment_slopes)
        aux = {'n_reactions': self.n_reactions}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        log_T9, log_rates, Q, rev_factor, T9_power, gamma, segment_slopes = children
        return cls(log_T9, log_rates, Q, rev_factor, T9_power, gamma, aux['n_reactions'], segment_slopes)


from rabbit.jax._cache_guard import _CACHE_LOCK, register_cache_clearer

_RATE_TABLE_CACHE = {}

def _clear_network_caches():
    _RATE_TABLE_CACHE.clear()

register_cache_clearer(_clear_network_caches)


def load_rate_table(n_reactions=31, path=None) -> RateTable:
    """Load PRIMAT rate table.

    Always loads from the 31rxn indexed JSON (same source as SciPy),
    then slices to n_reactions. This guarantees identical ordering.
    Manual truncations outside the documented 12/31 choices raise an explicit warning.
    """
    validate_manual_network_truncation(n_reactions, context="JAX rate-table load")
    with _CACHE_LOCK:
        if n_reactions in _RATE_TABLE_CACHE:
            return _RATE_TABLE_CACHE[n_reactions]

    # Always load from 31rxn table (canonical source, indexed by rxn['index'])
    json_name = 'primat_ac2024_31rxn.json'

    if path is None:
        candidates = [
            pathlib.Path(__file__).parent.parent / 'network' / 'data' / json_name,
            pathlib.Path(__file__).parent.parent / 'network' / json_name,
            pathlib.Path('.') / json_name,
        ]
        for p in candidates:
            if p.exists():
                path = p; break
        if path is None:
            raise FileNotFoundError(f"Cannot find {json_name}")

    with open(path) as f:
        data = json.load(f)

    T9 = _np.array(data['T9']); log_T9 = _np.log(T9)
    n_rxn_total = len(data['reactions']); n_T9 = len(T9)
    log_rates = _np.zeros((n_rxn_total, n_T9))
    Q = _np.zeros(n_rxn_total); rev_factor = _np.zeros(n_rxn_total)
    T9_power = _np.zeros(n_rxn_total); gamma = _np.zeros(n_rxn_total)
    for rxn in data['reactions']:
        i = rxn['index']
        log_rates[i] = rxn['log_rates']; Q[i] = rxn['Q_MeV']
        rev_factor[i] = rxn['rev_factor']; T9_power[i] = rxn['T9_power']
        gamma[i] = rxn['gamma']

    # Slice to requested n_reactions
    n_rxn = min(n_reactions, n_rxn_total)
    log_rates_slice = log_rates[:n_rxn]
    segment_slopes = (log_rates_slice[:, 1:] - log_rates_slice[:, :-1]) / (log_T9[1:] - log_T9[:-1])[None, :]
    table = RateTable(log_T9, log_rates_slice, Q[:n_rxn],
                      rev_factor[:n_rxn], T9_power[:n_rxn], gamma[:n_rxn],
                      n_rxn, segment_slopes)
    with _CACHE_LOCK:
        _RATE_TABLE_CACHE[n_reactions] = table
    return table


def validate_rate_table_window_jax(T_gamma_MeV: float, table: RateTable, *, context: str, strict: bool = False) -> None:
    """Plain-Python guard for PRIMAT table boundary use before entering JIT code."""
    validate_rate_table_temperature(float(T_gamma_MeV), context=context, strict=strict)

# ═══════════════════════════════════════════════════════════════════════
# §3. Rate evaluation
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def evaluate_nuclear_rates_jax(T_gamma_MeV, table):
    """Forward + reverse rates via cached piecewise-linear interpolation.

    This is algebraically identical to jnp.interp on the fixed PRIMAT grid,
    but avoids per-reaction Python loops and repeated slope recomputation.
    The pre-tabulated segment slopes help repeated statistical runs.
    """
    T9 = jnp.maximum(T_gamma_MeV * MEV_TO_T9, 1e-30)
    lT9 = jnp.log(T9)

    grid = table.log_T9
    idx = jnp.clip(jnp.searchsorted(grid, lT9, side="right") - 1, 0, grid.shape[0] - 2)
    base_x = grid[idx]
    log_rate_mid = table.log_rates[:, idx] + table.segment_slopes[:, idx] * (lT9 - base_x)
    log_rate = jnp.where(
        lT9 <= grid[0],
        table.log_rates[:, 0],
        jnp.where(lT9 >= grid[-1], table.log_rates[:, -1], log_rate_mid),
    )

    fwd = jnp.exp(log_rate)

    has_rev = (table.rev_factor > 0.0) & (jnp.abs(table.Q) > 0.0)
    log_Keq = (jnp.log(jnp.maximum(table.rev_factor, 1e-300))
               + table.T9_power * jnp.log(T9) + table.gamma / T9)
    log_Keq = jnp.clip(log_Keq, -500.0, 500.0)
    rev = jnp.where(has_rev, fwd * jnp.exp(log_Keq), 0.0)
    return fwd, rev


# ═══════════════════════════════════════════════════════════════════════
# §4. Flux computation
# ═══════════════════════════════════════════════════════════════════════

ZETA3 = 1.202056903


def compute_rho_factor(T_gamma, eta):
    PI2 = jnp.pi ** 2
    n_gamma = 2.0 * ZETA3 / PI2 * T_gamma ** 3
    n_b = eta * n_gamma
    hbar_c_cm = 197.3269804e-13
    n_b_cgs = n_b / hbar_c_cm ** 3
    N_A = 6.02214076e23
    return n_b_cgs / N_A


@jax.jit
def compute_fluxes_jax(X, T_gamma, eta, table):
    """Net reaction fluxes with vectorized reverse-flux treatment.

    Reverse flux = rho_fac^{n_prod−1} × sym_fac × rev × Π Y_k^{S_k}
    evaluated in log-space for numerical stability.
    """
    fwd, rev = evaluate_nuclear_rates_jax(T_gamma, table)
    rho_fac = compute_rho_factor(T_gamma, eta)
    Y = X / ATOMIC_MASSES
    log_Y_safe = jnp.log(jnp.maximum(Y, 1e-300))

    react_i = REACTANT_PAIRS[:table.n_reactions, 0]
    react_j = REACTANT_PAIRS[:table.n_reactions, 1]
    flux_fwd = rho_fac * IDENTICAL_FACTOR[:table.n_reactions] * fwd * Y[react_i] * Y[react_j]

    log_Y_prod = PRODUCT_STOICH[:, :table.n_reactions].T @ log_Y_safe
    Y_prod = jnp.exp(jnp.clip(log_Y_prod, -700.0, 700.0))

    rho_pow = REVERSE_RHO_POWER[:table.n_reactions]
    rho_factor_rev = jnp.where(
        rho_pow > 1.5,
        rho_fac * rho_fac,
        jnp.where(rho_pow > 0.5, rho_fac, 1.0),
    )
    flux_rev = jnp.where(
        rev > 0.0,
        rho_factor_rev * REVERSE_SYMMETRY[:table.n_reactions] * rev * Y_prod,
        0.0,
    )
    return flux_fwd - flux_rev


# ═══════════════════════════════════════════════════════════════════════
# §5. Abundance RHS
# ═══════════════════════════════════════════════════════════════════════

def abundance_rhs_phase1_jax(X_n, lambda_np, lambda_pn):
    return -lambda_np * X_n + lambda_pn * (1.0 - X_n)


@jax.jit
def abundance_rhs_phase2_jax(X, T_gamma, eta, lambda_np, lambda_pn, table):
    """Phase 2 RHS: nuclear network + weak rates."""
    n_rxn = table.n_reactions
    flux = compute_fluxes_jax(X, T_gamma, eta, table)
    S = STOICHIOMETRY[:, :n_rxn]
    dX = (S @ flux[:n_rxn]) * ATOMIC_MASSES

    dXn_weak = -lambda_np * X[0] + lambda_pn * X[1]
    dX = dX.at[0].add(dXn_weak)
    dX = dX.at[1].add(-dXn_weak)
    return dX


def phase1_to_phase2_jax(X_n):
    X = jnp.full(N_SPECIES, 1e-25)
    X = X.at[0].set(X_n)
    X = X.at[1].set(1.0 - X_n)
    X = X.at[6].set(1e-30)
    X = X.at[7].set(1e-30)
    return X


def mass_conservation_residual_jax(X):
    return jnp.abs(jnp.sum(X) - 1.0)
