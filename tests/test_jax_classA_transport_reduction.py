"""
Test: Class A transport κ-cascade reduction — candidate gate.

Validates:
  G1. Type I (κ=0): exact flat-streaming reduction
  G2. Type II (κ>0): curved monopole coupling active
  G3. κ formula: |mask * N_i| for all types
  G4. κ=0 for Type I regardless of N_i input (mask zeros them)
  G5. κ>0 for all curved types (II, VI₀, VII₀, VIII, IX)
"""
import math
import numpy as np
import jax.numpy as jnp
import pytest

from rabbit.jax.rhs_classA import (
    classA_transport_rhs, classA_transport_rhs_for_type,
    effective_kappa_from_curvature,
)
from rabbit.jax.transport_ops_jax import apply_flat_streaming_rhs
from rabbit.jax.geometry_classA_jax import build_type_mask
from rabbit.config.conventions import BianchiType


# ═══════════════════════════════════════════════════════════════
# G1. Type I: exact flat reduction
# ═══════════════════════════════════════════════════════════════

def test_classA_transport_reduces_exactly_to_flat_typeI_when_kappa_zero():
    N_q = 6
    n_ell = 2
    n_species = 6
    psi_flat = np.zeros(n_species * n_ell * N_q)
    psi_flat[1 * N_q:2 * N_q] = 0.02

    ref = np.asarray(apply_flat_streaming_rhs(0.08, 0.0, jnp.asarray(psi_flat), n_ell=n_ell, n_species=n_species))
    got = np.asarray(classA_transport_rhs(0.08, 0.0, jnp.asarray(psi_flat), n_ell=n_ell, n_species=n_species, kappa=jnp.asarray(0.0)))
    assert np.allclose(got, ref, rtol=0.0, atol=1e-15)


# ═══════════════════════════════════════════════════════════════
# G2. Type II: curved monopole coupling
# ═══════════════════════════════════════════════════════════════

def test_classA_transport_typeII_smoke_induces_monopole_coupling():
    N_q = 6
    n_ell = 2
    n_species = 6
    psi_flat = np.zeros(n_species * n_ell * N_q)
    for s in range(n_species):
        base = s * n_ell * N_q + 1 * N_q
        psi_flat[base:base + N_q] = 0.03

    got = np.asarray(classA_transport_rhs_for_type(
        0.08, 0.0, jnp.asarray(psi_flat), BianchiType.TYPE_II, 0.3, 0.0, 0.0, n_ell=n_ell, n_species=n_species
    ))
    assert np.max(np.abs(got[:N_q])) > 0.0
    assert np.isfinite(got).all()


# ═══════════════════════════════════════════════════════════════
# G3. κ formula: |mask * N_i|
# ═══════════════════════════════════════════════════════════════

def test_kappa_zero_for_typeI():
    mask = build_type_mask(BianchiType.TYPE_I)
    kappa = float(effective_kappa_from_curvature(0.1, 0.2, 0.3, mask))
    assert kappa == 0.0


def test_kappa_positive_for_typeII():
    mask = build_type_mask(BianchiType.TYPE_II)
    kappa = float(effective_kappa_from_curvature(0.3, 0.0, 0.0, mask))
    assert kappa > 0.0
    assert abs(kappa - 0.3) < 1e-12


def test_kappa_matches_expected_formula():
    mask = build_type_mask(BianchiType.TYPE_VIII)
    kappa = float(effective_kappa_from_curvature(0.1, 0.2, 0.15, mask))
    expected = math.sqrt(0.1**2 + 0.2**2 + 0.15**2)
    assert abs(kappa - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════
# G4-G5. κ for all 6 types
# ═══════════════════════════════════════════════════════════════

CURVED_TYPES = {
    BianchiType.TYPE_II:   (0.2, 0.0, 0.0),
    BianchiType.TYPE_VI0:  (0.0, 0.15, 0.10),
    BianchiType.TYPE_VII0: (0.0, 0.15, 0.15),
    BianchiType.TYPE_VIII: (0.1, 0.2, -0.05),
    BianchiType.TYPE_IX:   (0.08, 0.08, 0.08),
}


@pytest.mark.parametrize("btype", list(CURVED_TYPES.keys()),
                         ids=[b.value for b in CURVED_TYPES])
def test_kappa_positive_for_curved_types(btype):
    n1, n2, n3 = CURVED_TYPES[btype]
    mask = build_type_mask(btype)
    kappa = float(effective_kappa_from_curvature(n1, n2, n3, mask))
    assert kappa > 0, f"{btype.value}: κ={kappa}"


@pytest.mark.parametrize("btype", list(CURVED_TYPES.keys()),
                         ids=[b.value for b in CURVED_TYPES])
def test_transport_finite_for_curved_types(btype):
    """Transport RHS is finite and nonzero for all curved types."""
    N_q, n_ell, n_species = 6, 2, 6
    n1, n2, n3 = CURVED_TYPES[btype]
    psi = np.zeros(n_species * n_ell * N_q)
    for s in range(n_species):
        psi[s * n_ell * N_q + 1 * N_q: s * n_ell * N_q + 2 * N_q] = 0.02

    got = np.asarray(classA_transport_rhs_for_type(
        0.08, 0.0, jnp.asarray(psi), btype, n1, n2, n3, n_ell=n_ell, n_species=n_species))
    assert np.isfinite(got).all(), f"{btype.value}: non-finite transport RHS"
    assert np.max(np.abs(got)) > 0, f"{btype.value}: zero transport RHS"
