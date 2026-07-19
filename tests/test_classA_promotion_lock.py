"""
Test: Class A full-BBN promotion blocker contract.

After the curvature-sign and initial-data audit, the old absolute gold anchors
from 2026-04-04 are no longer trustworthy. This file now locks the *supported*
contract instead of stale Y_p numbers:

§1. TYPE-AWARE VALIDATION — mislabeled VI0/VII0/IX inputs must be rejected
§2. TYPE-I REDUCTION — ClassA(TYPE_I) must agree with the Type-I tier-2 path
§3. OPEN/CURVED SAFE CELLS — II, VI0, VII0, VIII, small IX must solve with physical outputs
§4. LARGE CLOSED IX HONESTY — unstable high-curvature IX cell must not be gold-locked
§5. INFERENCE METADATA — backend metadata remains intact
"""
import json
from pathlib import Path

import pytest
import numpy as np

pytest.importorskip("jax", reason="JAX required")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jax_bbn_gold.json"


def _gold(name: str) -> dict:
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)[name]


@pytest.fixture(scope="module")
def jax_setup():
    import jax; jax.config.update("jax_enable_x64", True)


class TestTypeAwareValidation:

    def test_vi0_rejects_same_sign_active_curvature(self):
        from rabbit.jax.driver_classA import JAXClassAConfig
        with pytest.raises(ValueError, match="N₂N₃<0"):
            JAXClassAConfig(bianchi_type='TYPE_VI0', N2_init=0.1, N3_init=0.1)

    def test_vii0_rejects_opposite_sign_active_curvature(self):
        from rabbit.jax.driver_classA import JAXClassAConfig
        with pytest.raises(ValueError, match="N₂N₃≥0"):
            JAXClassAConfig(bianchi_type='TYPE_VII0', N2_init=0.1, N3_init=-0.1)

    def test_ix_rejects_mixed_sign_closed_data(self):
        from rabbit.jax.driver_classA import JAXClassAConfig
        with pytest.raises(ValueError, match="same-sign"):
            JAXClassAConfig(bianchi_type='TYPE_IX', N1_init=0.08, N2_init=0.08, N3_init=-0.08)


@pytest.fixture(scope="module")
def typeI_pair(jax_setup):
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
    classA = run_classA_jax(JAXClassAConfig(
        bianchi_type='TYPE_I', Sigma_H_plus=0.0, N_q=20, n_ell=2, correction_level=0))
    typeI = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.0, N_q=20, correction_level=0, enable_teff=False))
    return classA, typeI


class TestTypeIReduction:

    def test_both_succeed(self, typeI_pair):
        classA, typeI = typeI_pair
        assert classA.success
        assert typeI is not None

    def test_yp_matches_typeI_reference(self, typeI_pair):
        classA, typeI = typeI_pair
        assert abs(classA.Yp - typeI.observables.Yp) < 1e-4

    def test_dh_matches_typeI_reference(self, typeI_pair):
        classA, typeI = typeI_pair
        assert abs(classA.DH - typeI.observables.DH) < 1e-7

    def test_curvature_zero_for_typeI(self, typeI_pair):
        classA, _ = typeI_pair
        assert classA.metadata['curvature_K_init'] == 0.0
        assert classA.metadata['transport_kappa_init'] == 0.0


@pytest.fixture(scope="module")
def safe_envelope(jax_setup):
    from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
    return {
        'TYPE_II': run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_II', Sigma_H_plus=0.03, N1_init=0.08, N_q=20, n_ell=2, correction_level=0)),
        'TYPE_VI0': run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_VI0', Sigma_H_plus=0.03, N2_init=0.08, N3_init=-0.05, N_q=20, n_ell=2, correction_level=0)),
        'TYPE_VII0': run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_VII0', Sigma_H_plus=0.03, N2_init=0.05, N3_init=0.03, N_q=20, n_ell=2, correction_level=0)),
        'TYPE_VIII': run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_VIII', Sigma_H_plus=0.01, N1_init=-1e-4, N2_init=1e-4, N3_init=1e-4, N_q=20, n_ell=2, correction_level=0)),
        'TYPE_IX': run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_IX', Sigma_H_plus=0.01, N1_init=1e-4, N2_init=1e-4, N3_init=1e-4, N_q=20, n_ell=2, correction_level=0)),
    }


class TestSafeCurvedCells:

    @pytest.mark.parametrize('btype', ['TYPE_II', 'TYPE_VI0', 'TYPE_VII0', 'TYPE_VIII', 'TYPE_IX'])
    def test_success(self, safe_envelope, btype):
        assert safe_envelope[btype].success

    @pytest.mark.parametrize('btype', ['TYPE_II', 'TYPE_VI0', 'TYPE_VII0', 'TYPE_VIII', 'TYPE_IX'])
    def test_physical_observables(self, safe_envelope, btype):
        r = safe_envelope[btype]
        assert np.isfinite(r.Yp) and 0.15 < r.Yp < 0.35
        assert np.isfinite(r.DH) and 1e-7 < r.DH < 1e-3
        assert np.isfinite(r.N_eff) and 2.5 < r.N_eff < 4.0

    def test_curvature_nonzero(self, safe_envelope):
        assert safe_envelope['TYPE_II'].metadata['curvature_K_init'] > 0.0
        assert safe_envelope['TYPE_VI0'].metadata['curvature_K_init'] > 0.0
        assert safe_envelope['TYPE_VIII'].metadata['curvature_K_init'] > 0.0
        assert safe_envelope['TYPE_IX'].metadata['curvature_K_init'] < 0.0


class TestLargeClosedIXHonesty:

    def test_ix_representative_cell_not_gold_locked(self, jax_setup):
        from rabbit.jax.driver_classA import JAXClassAConfig, run_classA_jax
        r = run_classA_jax(JAXClassAConfig(
            bianchi_type='TYPE_IX', Sigma_H_plus=0.02, Sigma_H_minus=0.01,
            N1_init=0.08, N2_init=0.08, N3_init=0.08, N_q=20, n_ell=2, correction_level=0))
        assert (not r.success) or (r.metadata.get('final_state_ok') is False)


CLASSA_METADATA_REQUIRED = [
    'backend', 'phase', 'bianchi_type', 'correction_level', 'N_q', 'n_ell',
    'transport_mode', 'transport_kappa_init', 'curvature_K_init', 'Sigma_plus_final', 'T_final',
]


@pytest.mark.parametrize("backend", ["jax_classA", "jax_tilted"])
def test_former_public_classa_endpoints_are_retired(backend):
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.03, backend=backend)


@pytest.fixture(scope="module")
def inference_result(jax_setup):
    from rabbit.inference.forward_likelihood import canonical_forward_solver
    return canonical_forward_solver(
        Sigma_H=0.03, backend='jax_classA',
        bianchi_type='TYPE_II', N1_init=0.08, N_q=20, correction_level=0)


@pytest.mark.skip(reason="F06 retired the public Class-A inference endpoint")
class TestInferenceMetadata:

    def test_success(self, inference_result):
        assert inference_result.success

    @pytest.mark.parametrize('key', CLASSA_METADATA_REQUIRED)
    def test_required_keys_present(self, inference_result, key):
        assert key in inference_result.metadata

    def test_dispatch_backend(self, inference_result):
        assert inference_result.metadata['dispatch_backend'] == 'jax_classA'


@pytest.mark.skip(reason="F06 retired public Class-A/tilted endpoint parity")
class TestPublicClassARepresentativeParity:

    @pytest.mark.parametrize('bianchi_type,gold_key', [
        ('TYPE_II', 'classA_typeII_N1_1em1_sigma005'),
        ('TYPE_VI0', 'classA_typeVI0_N2_2em4_N3_m1p2em4_sigma003'),
        ('TYPE_VII0', 'classA_typeVII0_N2_5e3_N3_3e3_sigma003'),
        ('TYPE_VIII', 'classA_typeVIII_N1_m1em4_N2_1em4_N3_1em4_sigma001'),
        ('TYPE_IX', 'classA_typeIX_N1_1em4_N2_1em4_N3_1em4_sigma001'),
    ])
    def test_public_jax_classA_matches_representative_gold(self, jax_setup, bianchi_type, gold_key):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        gold = _gold(gold_key)
        prediction = canonical_forward_solver(
            Sigma_H=gold["Sigma_H"],
            backend='jax_classA',
            bianchi_type=bianchi_type,
            N1_init=gold.get("N1_init", 0.0),
            N2_init=gold.get("N2_init", 0.0),
            N3_init=gold.get("N3_init", 0.0),
            N_q=20,
            correction_level=0,
        )

        assert prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_classA'
        assert prediction.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
        assert prediction.DH == pytest.approx(gold["DH"], rel=1.0e-10)
        assert prediction.metadata["curvature_K_final"] == pytest.approx(
            gold["curvature_K_final"], rel=1.0e-10
        )
        assert prediction.metadata["transport_kappa_final"] == pytest.approx(
            gold["transport_kappa_final"], rel=1.0e-10
        )
        if "Omega_final" in gold:
            assert prediction.metadata["Omega_final"] == pytest.approx(
                gold["Omega_final"], rel=1.0e-10
            )

    @pytest.mark.parametrize('bianchi_type,gold_key', [
        ('TYPE_II', 'tilted_typeII_N1_1em1_sigma005_v1e7'),
        ('TYPE_VI0', 'tilted_typeVI0_N2_2em4_N3_m1p2em4_sigma003_v1e7'),
        ('TYPE_VII0', 'tilted_typeVII0_N2_5e3_N3_3e3_sigma003_v1e7'),
        ('TYPE_VIII', 'tilted_typeVIII_N1_m1em4_N2_1em4_N3_1em4_sigma001_v1e7'),
        ('TYPE_IX', 'tilted_typeIX_N1_1em4_N2_1em4_N3_1em4_sigma001_v1e7'),
    ])
    def test_public_tilted_classA_matches_representative_gold(self, jax_setup, bianchi_type, gold_key):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        gold = _gold(gold_key)
        prediction = canonical_forward_solver(
            Sigma_H=gold["Sigma_H"],
            backend='jax_tilted',
            bianchi_type=bianchi_type,
            v0=gold["v0"],
            N1_init=gold.get("N1_init", 0.0),
            N2_init=gold.get("N2_init", 0.0),
            N3_init=gold.get("N3_init", 0.0),
            N_q=6,
            correction_level=0,
        )

        assert prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_tilted'
        assert prediction.metadata['canonical_bianchi_type'] == bianchi_type
        assert prediction.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
        assert prediction.DH == pytest.approx(gold["DH"], rel=1.0e-10)
        assert prediction.metadata["curvature_K_final"] == pytest.approx(
            gold["curvature_K_final"], rel=1.0e-10
        )
        assert prediction.metadata["transport_kappa_final"] == pytest.approx(
            gold["transport_kappa_final"], rel=1.0e-10
        )
        if "Omega_final" in gold:
            assert prediction.metadata["Omega_final"] == pytest.approx(
                gold["Omega_final"], rel=1.0e-10
            )
