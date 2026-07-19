"""
Test: Class A full BBN (phase 1 + phase 2) candidate smoke.

Runs the complete BBN pipeline for all 6 Class A Bianchi types:
  Type I (flat + shear), Type II (strong + weak curvature),
  Type VI₀ (asymmetric), Type VII₀ (LRS symmetric), Type VIII (small N_i),
  Type IX (closed).

Matrix (8 cells, 6 types):
  Type I FLRW:       reduction parity vs TypeI driver (Y_p, D/H, N_eff)
  Type I shear:      Σ=0.05, shear propagates through full BBN
  Type II strong:    Σ=0.05, N₁=0.2 (larger curvature)
  Type II weak:      Σ=0.05, N₁=0.1 (smaller curvature)
  Type VI₀:          Σ₊=0.03, Σ₋=0.01, N₂=0.1, N₃=-0.05 (asymmetric, opposite-sign active curvature)
  Type VII₀:         Σ=0.04, N₂=N₃=0.15 (LRS, open)
  Type VIII:         Σ=0.01, N₁=-0.001, N₂=N₃=0.001 (small, stable regime)
  Type IX:           Σ₊=0.02, Σ₋=0.01, N₁=N₂=N₃=0.08 (closed, K<0)

Note: Type VIII is curvature-unstable for N_i > ~0.002; only small-N_i regime is tested.
"""
import math
import pytest

pytest.importorskip("jax", reason="JAX required")

PHASE2_METADATA_CONTRACT = [
    'backend', 'phase', 'bianchi_type',
    'correction_level', 'n_reactions',
    'transport_mode', 'transport_kappa_init', 'transport_kappa_final',
    'transport_exact_typeI',
    'curvature_K_init', 'curvature_K_final',
    'T_final', 'T_nu_e_final', 'T_nu_x_final',
    'Sigma_plus_final', 'Sigma_minus_final',
    'N1_final', 'N2_final', 'N3_final',
    'N_q', 'n_ell',
]


@pytest.fixture(scope="module")
def jax_setup():
    import jax
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def typeI_flrw(jax_setup):
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_I",
        Sigma_H_plus=0.0, N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeI_shear(jax_setup):
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_I",
        Sigma_H_plus=0.05, N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeII_full(jax_setup):
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_II",
        Sigma_H_plus=0.05, N1_init=0.2,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeII_weak(jax_setup):
    """Type II with weaker curvature (N₁=0.1 vs 0.2)."""
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_II",
        Sigma_H_plus=0.05, N1_init=0.1,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeVII0_full(jax_setup):
    """Type VII₀: symmetric pair N₂=N₃, LRS."""
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_VII0",
        Sigma_H_plus=0.04, N2_init=0.15, N3_init=0.15,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeVI0_full(jax_setup):
    """Type VI₀: N₁ masked, N₂≠N₃ (asymmetric curvature)."""
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_VI0",
        Sigma_H_plus=0.03, Sigma_H_minus=0.01,
        N2_init=0.1, N3_init=-0.05,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeIX_full(jax_setup):
    """Type IX: all three N_i nonzero (closed, positive curvature)."""
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_IX",
        Sigma_H_plus=0.02, Sigma_H_minus=0.01,
        N1_init=0.08, N2_init=0.08, N3_init=0.08,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeVIII_full(jax_setup):
    """Type VIII: mixed-sign N_i (N₁<0, N₂,N₃>0). Small N_i for stability."""
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return run_classA_jax(JAXClassAConfig(
        bianchi_type="TYPE_VIII",
        Sigma_H_plus=0.01,
        N1_init=-0.001, N2_init=0.001, N3_init=0.001,
        N_q=20, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeI_ref(jax_setup):
    from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
    r = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.0, N_q=20, correction_level=0, enable_teff=False))
    return r.observables


# ═══════════════════════════════════════════════════════════════
# §1. Type I FLRW: reduction parity
# ═══════════════════════════════════════════════════════════════

class TestTypeIReductionParity:

    def test_success(self, typeI_flrw):
        assert typeI_flrw.success

    def test_yp_matches_typeI_driver(self, typeI_flrw, typeI_ref):
        diff = abs(typeI_flrw.Yp - typeI_ref.Yp)
        assert diff < 1e-4, f"Y_p diff={diff:.2e}"

    def test_dh_matches_typeI_driver(self, typeI_flrw, typeI_ref):
        rel = abs(typeI_flrw.DH - typeI_ref.DH) / typeI_ref.DH
        assert rel < 0.01

    def test_neff_matches_typeI_driver(self, typeI_flrw, typeI_ref):
        diff = abs(typeI_flrw.N_eff - typeI_ref.N_eff)
        assert diff < 0.01

    def test_flat_curvature(self, typeI_flrw):
        assert abs(typeI_flrw.metadata['curvature_K_init']) < 1e-15
        assert abs(typeI_flrw.metadata['curvature_K_final']) < 1e-15

    def test_kappa_zero(self, typeI_flrw):
        assert typeI_flrw.metadata['transport_kappa_init'] == 0.0
        assert typeI_flrw.metadata['transport_exact_typeI'] is True


# ═══════════════════════════════════════════════════════════════
# §2. Type I with shear: full BBN anisotropy
# ═══════════════════════════════════════════════════════════════

class TestTypeIShear:

    def test_success(self, typeI_shear):
        assert typeI_shear.success

    def test_yp_physical(self, typeI_shear):
        assert 0.20 < typeI_shear.Yp < 0.30

    def test_shear_increases_yp(self, typeI_flrw, typeI_shear):
        """Σ=0.05 → Y_p should increase (expansion channel)."""
        assert typeI_shear.Yp >= typeI_flrw.Yp - 1e-6

    def test_dh_physical(self, typeI_shear):
        assert 1e-6 < typeI_shear.DH < 1e-3

    def test_metadata_type(self, typeI_shear):
        assert typeI_shear.metadata['bianchi_type'] == 'TYPE_I'
        assert typeI_shear.metadata['phase'] == 'full_bbn'


# ═══════════════════════════════════════════════════════════════
# §3. Type II full BBN: curved universe
# ═══════════════════════════════════════════════════════════════

class TestTypeIIFullBBN:

    def test_success(self, typeII_full):
        assert typeII_full.success

    def test_yp_physical(self, typeII_full):
        assert 0.20 < typeII_full.Yp < 0.30

    def test_dh_physical(self, typeII_full):
        assert 1e-6 < typeII_full.DH < 1e-3

    def test_neff_physical(self, typeII_full):
        assert 2.9 < typeII_full.N_eff < 3.15

    def test_curvature_present(self, typeII_full):
        assert abs(typeII_full.metadata['curvature_K_init']) > 1e-6

    def test_kappa_positive(self, typeII_full):
        assert typeII_full.metadata['transport_kappa_init'] > 0
        assert typeII_full.metadata['transport_exact_typeI'] is False

    def test_metadata_type(self, typeII_full):
        assert typeII_full.metadata['bianchi_type'] == 'TYPE_II'
        assert typeII_full.metadata['phase'] == 'full_bbn'

    def test_n_reactions(self, typeII_full):
        assert typeII_full.metadata['n_reactions'] == 12


# ═══════════════════════════════════════════════════════════════
# §3b. Type VII₀ full BBN: LRS symmetric pair
# ═══════════════════════════════════════════════════════════════

class TestTypeVII0FullBBN:

    def test_success(self, typeVII0_full):
        assert typeVII0_full.success

    def test_yp_physical(self, typeVII0_full):
        assert 0.20 < typeVII0_full.Yp < 0.30

    def test_dh_physical(self, typeVII0_full):
        assert 1e-6 < typeVII0_full.DH < 1e-3

    def test_neff_physical(self, typeVII0_full):
        assert 2.9 < typeVII0_full.N_eff < 3.15

    def test_kappa_positive(self, typeVII0_full):
        assert typeVII0_full.metadata['transport_kappa_init'] > 0

    def test_metadata_type(self, typeVII0_full):
        assert typeVII0_full.metadata['bianchi_type'] == 'TYPE_VII0'
        assert typeVII0_full.metadata['phase'] == 'full_bbn'

    def test_n1_stays_zero(self, typeVII0_full):
        """Type VII₀: N₁ should remain zero (masked)."""
        assert abs(typeVII0_full.metadata['N1_final']) < 1e-14


# ═══════════════════════════════════════════════════════════════
# §3c. Type VI₀ full BBN: asymmetric curvature (N₂≠N₃, N₁ masked)
# ═══════════════════════════════════════════════════════════════

class TestTypeVI0FullBBN:

    def test_success(self, typeVI0_full):
        assert typeVI0_full.success

    def test_yp_physical(self, typeVI0_full):
        assert 0.20 < typeVI0_full.Yp < 0.30

    def test_dh_physical(self, typeVI0_full):
        assert 1e-6 < typeVI0_full.DH < 1e-3

    def test_neff_physical(self, typeVI0_full):
        assert 2.9 < typeVI0_full.N_eff < 3.15

    def test_kappa_positive(self, typeVI0_full):
        assert typeVI0_full.metadata['transport_kappa_init'] > 0
        assert typeVI0_full.metadata['transport_exact_typeI'] is False

    def test_curvature_positive(self, typeVI0_full):
        """Type VI₀ has K > 0 (hyperbolic type)."""
        assert typeVI0_full.metadata['curvature_K_init'] > 0

    def test_metadata_type(self, typeVI0_full):
        assert typeVI0_full.metadata['bianchi_type'] == 'TYPE_VI0'
        assert typeVI0_full.metadata['phase'] == 'full_bbn'

    def test_n1_stays_zero(self, typeVI0_full):
        """Type VI₀: N₁ should remain zero (masked)."""
        assert abs(typeVI0_full.metadata['N1_final']) < 1e-14

    def test_n2_n3_asymmetric(self, typeVI0_full):
        """N₂≠N₃ should remain distinct."""
        n2 = typeVI0_full.metadata['N2_final']
        n3 = typeVI0_full.metadata['N3_final']
        assert abs(n2 - n3) > 1e-6


# ═══════════════════════════════════════════════════════════════
# §3d. Type IX full BBN: closed honesty cell (all three N_i nonzero)
# ═══════════════════════════════════════════════════════════════

class TestTypeIXFullBBN:

    def test_closed_ix_not_gold_locked(self, typeIX_full):
        assert (not typeIX_full.success) or (typeIX_full.metadata.get('final_state_ok') is False)

    def test_kappa_positive(self, typeIX_full):
        assert typeIX_full.metadata['transport_kappa_init'] > 0
        assert typeIX_full.metadata['transport_exact_typeI'] is False

    def test_curvature_negative(self, typeIX_full):
        """Type IX has K < 0 (closed type, positive spatial curvature)."""
        assert typeIX_full.metadata['curvature_K_init'] < 0

    def test_metadata_type(self, typeIX_full):
        assert typeIX_full.metadata['bianchi_type'] == 'TYPE_IX'
        assert typeIX_full.metadata['phase'] == 'full_bbn'

    def test_all_three_ni_nonzero(self, typeIX_full):
        """Type IX: all three N_i should be nonzero."""
        assert abs(typeIX_full.metadata['N1_final']) > 1e-6
        assert abs(typeIX_full.metadata['N2_final']) > 1e-6
        assert abs(typeIX_full.metadata['N3_final']) > 1e-6


# ═══════════════════════════════════════════════════════════════
# §3e. Type VIII full BBN: mixed-sign curvature (N₁<0)
# ═══════════════════════════════════════════════════════════════

class TestTypeVIIIFullBBN:
    """Type VIII requires small N_i (≤0.002) for curvature stability.
    Larger initial conditions cause curvature runaway (Ω→negative)."""

    def test_success(self, typeVIII_full):
        assert typeVIII_full.success

    def test_yp_physical(self, typeVIII_full):
        assert 0.20 < typeVIII_full.Yp < 0.30

    def test_dh_physical(self, typeVIII_full):
        assert 1e-6 < typeVIII_full.DH < 1e-3

    def test_neff_physical(self, typeVIII_full):
        assert 2.9 < typeVIII_full.N_eff < 3.15

    def test_kappa_nonneg(self, typeVIII_full):
        assert typeVIII_full.metadata['transport_kappa_init'] >= 0

    def test_metadata_type(self, typeVIII_full):
        assert typeVIII_full.metadata['bianchi_type'] == 'TYPE_VIII'
        assert typeVIII_full.metadata['phase'] == 'full_bbn'


# ═══════════════════════════════════════════════════════════════
# §4. Type II curvature sensitivity
# ═══════════════════════════════════════════════════════════════

class TestCurvatureSensitivity:

    def test_weak_curvature_succeeds(self, typeII_weak):
        assert typeII_weak.success

    def test_weak_curvature_physical(self, typeII_weak):
        assert 0.20 < typeII_weak.Yp < 0.30
        assert 1e-6 < typeII_weak.DH < 1e-3

    def test_weaker_curvature_has_less_K(self, typeII_full, typeII_weak):
        """N₁=0.1 → smaller |K| than N₁=0.2."""
        K_strong = abs(typeII_full.metadata['curvature_K_init'])
        K_weak = abs(typeII_weak.metadata['curvature_K_init'])
        assert K_weak < K_strong

    def test_lrs_typeii_amplitude_changes_yields(self, typeII_full, typeII_weak):
        """Type II now evolves its own curvature instead of delegating to Type I."""
        assert typeII_full.metadata['production_authority'] == 'candidate_classA_curved_transport'
        assert typeII_weak.metadata['production_authority'] == 'candidate_classA_curved_transport'
        assert typeII_full.metadata['transport_mode'] == 'kappa_cascade_lmax2'
        assert typeII_weak.metadata['transport_mode'] == 'kappa_cascade_lmax2'
        assert abs(typeII_full.Yp - typeII_weak.Yp) > 1e-4

    def test_typeii_curvature_evolves(self, typeII_full):
        assert math.isfinite(typeII_full.metadata['curvature_K_final'])
        assert typeII_full.metadata['N1_final'] != pytest.approx(0.2)
        assert typeII_full.metadata['curvature_K_final'] != pytest.approx(
            typeII_full.metadata['curvature_K_init']
        )

    def test_both_above_flat(self, typeI_flrw, typeII_full, typeII_weak):
        """Both curved results should have Y_p ≥ flat Type I."""
        yp_flat = typeI_flrw.Yp
        assert typeII_full.Yp >= yp_flat - 1e-4
        assert typeII_weak.Yp >= yp_flat - 1e-4


# ═══════════════════════════════════════════════════════════════
# §5. Metadata contract (all cells)
# ═══════════════════════════════════════════════════════════════

ALL_CELLS = [
    "typeI_flrw", "typeI_shear",
    "typeII_full", "typeII_weak",
    "typeVII0_full", "typeVI0_full", "typeIX_full", "typeVIII_full",
]


class TestMetadataContract:

    @pytest.mark.parametrize("fixture_name", ALL_CELLS)
    def test_contract(self, fixture_name, typeI_flrw, typeI_shear, typeII_full, typeII_weak, typeVII0_full, typeVI0_full, typeIX_full, typeVIII_full):
        results = {"typeI_flrw": typeI_flrw, "typeI_shear": typeI_shear,
                    "typeII_full": typeII_full, "typeII_weak": typeII_weak,
                    "typeVII0_full": typeVII0_full, "typeVI0_full": typeVI0_full,
                    "typeIX_full": typeIX_full, "typeVIII_full": typeVIII_full}
        r = results[fixture_name]
        for key in PHASE2_METADATA_CONTRACT:
            assert key in r.metadata, f"{fixture_name} missing: {key}"

    @pytest.mark.parametrize("fixture_name", ALL_CELLS)
    def test_backend(self, fixture_name, typeI_flrw, typeI_shear, typeII_full, typeII_weak, typeVII0_full, typeVI0_full, typeIX_full, typeVIII_full):
        results = {"typeI_flrw": typeI_flrw, "typeI_shear": typeI_shear,
                    "typeII_full": typeII_full, "typeII_weak": typeII_weak,
                    "typeVII0_full": typeVII0_full, "typeVI0_full": typeVI0_full,
                    "typeIX_full": typeIX_full, "typeVIII_full": typeVIII_full}
        assert results[fixture_name].metadata['backend'] == 'jax_classA_driver'


# ═══════════════════════════════════════════════════════════════
# §6. Cross-type ordering: curvature → Y_p
# ═══════════════════════════════════════════════════════════════

class TestCrossTypeOrdering:
    """Sanity: curvature modifications should show consistent effects."""

    def test_open_and_bounded_curved_types_physical(
        self, typeI_shear, typeII_full, typeVII0_full, typeVI0_full, typeVIII_full
    ):
        """Open/bounded cells should produce physical Y_p and succeed."""
        for name, r in [("I+shear", typeI_shear), ("II", typeII_full),
                        ("VII₀", typeVII0_full), ("VI₀", typeVI0_full),
                        ("VIII", typeVIII_full)]:
            assert 0.15 < r.Yp < 0.30, f"{name}: Y_p={r.Yp}"
            assert r.success, f"{name} failed"

    def test_six_types_covered(
        self, typeI_flrw, typeII_full, typeVII0_full, typeVI0_full, typeIX_full, typeVIII_full
    ):
        """Merge gate: 6 distinct Bianchi types in full-BBN."""
        types_covered = {
            typeI_flrw.metadata['bianchi_type'],
            typeII_full.metadata['bianchi_type'],
            typeVII0_full.metadata['bianchi_type'],
            typeVI0_full.metadata['bianchi_type'],
            typeIX_full.metadata['bianchi_type'],
            typeVIII_full.metadata['bianchi_type'],
        }
        assert len(types_covered) == 6, f"Only {len(types_covered)} types: {types_covered}"
        assert types_covered == {'TYPE_I', 'TYPE_II', 'TYPE_VII0', 'TYPE_VI0', 'TYPE_IX', 'TYPE_VIII'}
