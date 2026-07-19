"""
Test: Class B reduced-mask family contract.

The old family envelope treated VI_h/VII_h as if arbitrary N2/N3 data
defined the h families. It does not. This test now enforces:

§1. FULL-BBN supported slice — V, IV, canonical III, h-locked VI_h/VII_h representatives, and canonical VI_{-1/9}
§2. FAMILY-LABEL VALIDATION — stale h-family probes using inactive/missing data must be rejected
§3. FLRW reduction for Type V
§4. Metadata/capability honesty
"""
import json
from pathlib import Path
from types import SimpleNamespace

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


class TestValidation:

    def test_typeV_rejects_curvature_coordinates(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='Type V requires'):
            JAXClassBConfig(bianchi_type='V', N1_init=1e-3, A_init=1e-4)

    def test_typeIII_rejects_inactive_N1_probe(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='deactivates N₁'):
            JAXClassBConfig(bianchi_type='III', N1_init=1e-2, A_init=1e-5)

    def test_typeIII_requires_canonical_h_minus_one_relation(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='N₂=-N₃'):
            JAXClassBConfig(
                bianchi_type='III',
                N2_init=1e-4,
                N3_init=-5e-5,
                A_init=1e-5,
            )

    def test_typeVI_m19_requires_canonical_h_minus_one_ninth_relation(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='N₃=-N₂/9'):
            JAXClassBConfig(
                bianchi_type='VI_m19',
                N2_init=1e-4,
                N3_init=-5e-5,
                A_init=1e-5,
            )

    def test_vih_requires_active_curvature_direction(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='requires at least one active curvature'):
            JAXClassBConfig(bianchi_type='VIh', h=-2.0, A_init=1e-5)

    def test_vih_requires_explicit_h(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='requires an explicit h'):
            JAXClassBConfig(
                bianchi_type='VIh',
                N2_init=1e-4,
                N3_init=-2e-4,
                A_init=1e-5,
            )

    def test_vih_rejects_wrong_h_relation(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        with pytest.raises(ValueError, match='N₃=h\\*N₂'):
            JAXClassBConfig(
                bianchi_type='VIh',
                h=-2.0,
                N2_init=1e-4,
                N3_init=-1e-4,
                A_init=1e-5,
            )


@pytest.fixture(scope="module")
def family(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax
    return {
        'V': run_classB_jax(JAXClassBConfig(
            bianchi_type='V', Sigma_H_plus=0.02, A_init=1e-4, N_q=6, n_ell=2, correction_level=0)),
        'IV': run_classB_jax(JAXClassBConfig(
            bianchi_type='IV', Sigma_H_plus=0.01, N1_init=1e-3, A_init=1e-5, N_q=6, n_ell=2, correction_level=0)),
        'III': run_classB_jax(JAXClassBConfig(
            bianchi_type='III', Sigma_H_plus=0.01, N2_init=1e-4, N3_init=-1e-4, A_init=1e-5, N_q=6, n_ell=2, correction_level=0)),
        'VI_m19': run_classB_jax(JAXClassBConfig(
            bianchi_type='VI_m19', Sigma_H_plus=0.01, N2_init=1e-4, N3_init=-1e-4 / 9.0, A_init=1e-5, N_q=6, n_ell=2, correction_level=0)),
        'VIh': run_classB_jax(JAXClassBConfig(
            bianchi_type='VIh', h=-2.0, Sigma_H_plus=0.01, N2_init=1e-4, N3_init=-2e-4, A_init=1e-5, N_q=6, n_ell=2, correction_level=0)),
        'VIIh': run_classB_jax(JAXClassBConfig(
            bianchi_type='VIIh', h=0.5, Sigma_H_plus=0.01, N2_init=5e-4, N3_init=2.5e-4, A_init=1e-5, N_q=6, n_ell=2, correction_level=0)),
    }


class TestSupportedFullBBNSlice:

    @pytest.mark.parametrize('btype', ['V', 'IV', 'III', 'VI_m19', 'VIh', 'VIIh'])
    def test_success(self, family, btype):
        assert family[btype].success

    @pytest.mark.parametrize('btype', ['V', 'IV', 'III', 'VI_m19', 'VIh', 'VIIh'])
    def test_physical_observables(self, family, btype):
        r = family[btype]
        assert np.isfinite(r.Yp) and 0.20 < r.Yp < 0.30
        assert np.isfinite(r.DH) and 1e-7 < r.DH < 1e-3
        assert np.isfinite(r.N_eff) and 2.5 < r.N_eff < 4.0

    def test_iv_differs_from_v(self, family):
        assert family['IV'].Yp != family['V'].Yp

    def test_iii_differs_from_v(self, family):
        assert family['III'].Yp != family['V'].Yp
        assert family['III'].metadata['N2_final'] == pytest.approx(-family['III'].metadata['N3_final'], rel=1e-12)

    def test_vi_m19_differs_from_v_with_exceptional_c_factor(self, family):
        assert family['VI_m19'].Yp != family['V'].Yp
        assert family['VI_m19'].metadata['c_factor'] == pytest.approx(15.0 / 4.0)
        assert family['VI_m19'].metadata['N3_init'] == pytest.approx(
            -family['VI_m19'].metadata['N2_init'] / 9.0, rel=1.0e-15)
        assert family['VI_m19'].metadata['N3_final'] == pytest.approx(
            family['VI_m19'].metadata['h_parameter'] * family['VI_m19'].metadata['N2_final'], rel=1.0e-12)

    @pytest.mark.parametrize('btype,h', [('VIh', -2.0), ('VIIh', 0.5)])
    def test_h_family_representatives_are_h_locked(self, family, btype, h):
        assert family[btype].Yp != family['V'].Yp
        assert family[btype].metadata['h_parameter'] == pytest.approx(h)
        assert family[btype].metadata['c_factor'] == pytest.approx(3.0)
        assert family[btype].metadata['N3_init'] == pytest.approx(h * family[btype].metadata['N2_init'], rel=1.0e-15)
        assert family[btype].metadata['N3_final'] == pytest.approx(h * family[btype].metadata['N2_final'], rel=1.0e-12)


class TestFLRWReduction:

    @pytest.fixture(scope='class')
    def flrw_ref(self, jax_setup):
        from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax
        return run_classB_jax(JAXClassBConfig(
            bianchi_type='V', Sigma_H_plus=0.0, A_init=1e-10, N_q=6, n_ell=2, correction_level=0))

    def test_flrw_success(self, flrw_ref):
        assert flrw_ref.success

    def test_typeV_near_flrw(self, family, flrw_ref):
        assert abs(family['V'].Yp - flrw_ref.Yp) < 0.01


CLASSB_METADATA_REQUIRED = [
    'backend', 'phase', 'bianchi_type', 'c_factor', 'A_init', 'A_final',
    'correction_level', 'N_q', 'T_final', 'N_eff', 'final_state_ok', 'final_state_reason',
]


class TestMetadataContract:

    @pytest.mark.parametrize('btype', ['V', 'IV', 'III', 'VI_m19', 'VIh', 'VIIh'])
    def test_required_keys(self, family, btype):
        meta = family[btype].metadata
        for key in CLASSB_METADATA_REQUIRED:
            assert key in meta, f"{btype}: missing {key}"

    def test_c_factor_correct(self, family):
        expected = {
            'V': 3.0,
            'IV': 3.0,
            'III': 3.0,
            'VI_m19': 15.0 / 4.0,
            'VIh': 3.0,
            'VIIh': 3.0,
        }
        for btype, r in family.items():
            assert r.metadata['c_factor'] == expected[btype]


@pytest.mark.parametrize("backend", ["jax_classB", "jax_tilted"])
def test_former_public_classb_endpoints_are_retired(backend):
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.02, backend=backend)


@pytest.mark.skip(reason="F06 retired public Class-B/tilted inference dispatch")
class TestInferenceDispatch:

    @pytest.fixture(scope='class')
    def inference_result(self, jax_setup):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        return canonical_forward_solver(
            Sigma_H=0.02, backend='jax_classB',
            bianchi_type='V', A_init=1e-4, N_q=6, correction_level=0)

    def test_success(self, inference_result):
        assert inference_result.success

    def test_dispatch_backend(self, inference_result):
        assert inference_result.metadata['dispatch_backend'] == 'jax_classB'

    @pytest.mark.parametrize('bianchi_type,gold_key', [
        ('TYPE_V', 'classB_typeV_A1e4'),
        ('TYPE_IV', 'classB_typeIV_N1e3_A1e5_sigma001'),
        ('TYPE_III', 'classB_typeIII_N2_1em4_N3_m1em4_A1e5_sigma001'),
        ('TYPE_VI_M19', 'classB_typeVI_m19_N2_1em4_N3_mN2over9_A1e5_sigma001'),
        ('TYPE_VIH', 'classB_typeVIh_hm2_N2_1em4_N3_m2em4_A1e5_sigma001'),
        ('TYPE_VIIH', 'classB_typeVIIh_h0p5_N2_5em4_N3_2p5em4_A1e5_sigma001'),
    ])
    def test_public_classB_dispatch_matches_representative_gold(self, jax_setup, bianchi_type, gold_key):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        gold = _gold(gold_key)
        prediction = canonical_forward_solver(
            Sigma_H=gold["Sigma_H"],
            backend='jax_classB',
            bianchi_type=bianchi_type,
            h=gold.get("h"),
            A_init=gold["A_init"],
            N1_init=gold.get("N1_init", 0.0),
            N2_init=gold.get("N2_init", 0.0),
            N3_init=gold.get("N3_init", 0.0),
            N_q=6,
            correction_level=0,
        )

        assert prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_classB'
        assert prediction.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
        assert prediction.DH == pytest.approx(gold["DH"], rel=1.0e-10)
        if "c_factor" in gold:
            assert prediction.metadata["c_factor"] == pytest.approx(gold["c_factor"])
        if "h" in gold:
            assert prediction.metadata["h_parameter"] == pytest.approx(gold["h"])
        if "curvature_K_final" in gold:
            assert prediction.metadata["curvature_K_final"] == pytest.approx(
                gold["curvature_K_final"], rel=1.0e-10
            )
        if "frame_cA_sq_final" in gold:
            assert prediction.metadata["frame_cA_sq_final"] == pytest.approx(
                gold["frame_cA_sq_final"], rel=1.0e-10
            )

    @pytest.mark.parametrize('bianchi_type,gold_key', [
        ('TYPE_V', 'tilted_typeV_A1e4_sigma002_v1e7'),
        ('TYPE_IV', 'tilted_typeIV_N1e3_A1e5_sigma001_v1e7'),
        ('TYPE_III', 'tilted_typeIII_N2_1em4_N3_m1em4_A1e5_sigma001_v1e7'),
        ('TYPE_VI_M19', 'tilted_typeVI_m19_N2_1em4_N3_mN2over9_A1e5_sigma001_v1e7'),
        ('TYPE_VIH', 'tilted_typeVIh_hm2_N2_1em4_N3_m2em4_A1e5_sigma001_v1e7'),
        ('TYPE_VIIH', 'tilted_typeVIIh_h0p5_N2_5em4_N3_2p5em4_A1e5_sigma001_v1e7'),
    ])
    def test_public_tilted_dispatch_matches_representative_gold(self, jax_setup, bianchi_type, gold_key):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        gold = _gold(gold_key)
        prediction = canonical_forward_solver(
            Sigma_H=gold["Sigma_H"],
            backend='jax_tilted',
            bianchi_type=bianchi_type,
            h=gold.get("h"),
            v0=gold["v0"],
            A_init=gold.get("A_init", 0.0),
            N1_init=gold.get("N1_init", 0.0),
            N2_init=gold.get("N2_init", 0.0),
            N3_init=gold.get("N3_init", 0.0),
            N_q=6,
            correction_level=0,
        )

        assert prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_tilted'
        assert prediction.Yp == pytest.approx(gold["Yp"], rel=1.0e-10)
        assert prediction.DH == pytest.approx(gold["DH"], rel=1.0e-10)
        if "c_factor" in gold:
            assert prediction.metadata["c_factor"] == pytest.approx(gold["c_factor"])
        if "h" in gold:
            assert prediction.metadata["h_parameter"] == pytest.approx(gold["h"])
        if "curvature_K_final" in gold:
            assert prediction.metadata["curvature_K_final"] == pytest.approx(
                gold["curvature_K_final"], rel=1.0e-10
            )
        if "frame_cA_sq_final" in gold:
            assert prediction.metadata["frame_cA_sq_final"] == pytest.approx(
                gold["frame_cA_sq_final"], rel=1.0e-10
            )

    @pytest.mark.parametrize(
        'bianchi_type,h,N2_init,N3_init',
        [('VIh', -2.0, 1.0e-4, -2.0e-4), ('VIIh', 0.5, 5.0e-4, 2.5e-4)],
    )
    def test_public_classB_dispatch_propagates_h_family_h(
        self, jax_setup, bianchi_type, h, N2_init, N3_init
    ):
        from rabbit.inference.forward_likelihood import make_canonical_forward_model

        forward_model = make_canonical_forward_model(
            backend='jax_classB',
            bianchi_type=bianchi_type,
            h=h,
            A_init=1.0e-5,
            N2_init=N2_init,
            N3_init=N3_init,
            N_q=6,
            correction_level=0,
        )
        prediction = forward_model.predict(Sigma_H=0.01)

        assert prediction.success
        assert prediction.params['h'] == pytest.approx(h)
        assert prediction.metadata['dispatch_backend'] == 'jax_classB'
        assert prediction.metadata['h_parameter'] == pytest.approx(h)
        assert prediction.metadata['N3_final'] == pytest.approx(
            h * prediction.metadata['N2_final'], rel=1.0e-12
        )

    def test_public_classB_dispatch_rejects_missing_vih_h(self, jax_setup):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        prediction = canonical_forward_solver(
            Sigma_H=0.01,
            backend='jax_classB',
            bianchi_type='VIh',
            A_init=1.0e-5,
            N2_init=1.0e-4,
            N3_init=-2.0e-4,
            N_q=6,
            correction_level=0,
        )

        assert not prediction.success
        assert prediction.metadata['backend'] == 'jax_classB_failed'
        assert 'requires an explicit h' in prediction.metadata['error']

    @pytest.mark.parametrize(
        'bianchi_type,h,N2_init,N3_init',
        [('VIh', -2.0, 1.0e-4, -2.0e-4), ('VIIh', 0.5, 5.0e-4, 2.5e-4)],
    )
    def test_public_tilted_dispatch_propagates_h_family_h(
        self, jax_setup, bianchi_type, h, N2_init, N3_init
    ):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        prediction = canonical_forward_solver(
            Sigma_H=0.01,
            backend='jax_tilted',
            bianchi_type=bianchi_type,
            h=h,
            v0=1.0e-7,
            A_init=1.0e-5,
            N2_init=N2_init,
            N3_init=N3_init,
            N_q=6,
            correction_level=0,
        )

        assert prediction.success
        assert prediction.params['h'] == pytest.approx(h)
        assert prediction.metadata['dispatch_backend'] == 'jax_tilted'
        assert prediction.metadata['h_parameter'] == pytest.approx(h)
        assert prediction.metadata['N3_final'] == pytest.approx(
            h * prediction.metadata['N2_final'], rel=1.0e-12
        )

    def test_public_tilted_dispatch_rejects_missing_vih_h(self, jax_setup):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        prediction = canonical_forward_solver(
            Sigma_H=0.01,
            backend='jax_tilted',
            bianchi_type='VIh',
            v0=1.0e-7,
            A_init=1.0e-5,
            N2_init=1.0e-4,
            N3_init=-2.0e-4,
            N_q=6,
            correction_level=0,
        )

        assert not prediction.success
        assert prediction.metadata['backend'] == 'jax_tilted_failed'
        assert 'requires an explicit h' in prediction.metadata['error']

    def test_public_classB_dispatch_propagates_driver_failure(self, jax_setup, monkeypatch):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        import rabbit.jax.driver_classB as driver_classB

        def fake_run_classB(_config):
            return SimpleNamespace(
                Yp=np.nan,
                DH=np.nan,
                success=False,
                metadata={
                    'backend': 'fake_classB_failure',
                    'transport_mode': 'kappa_cascade_lmax2',
                },
            )

        monkeypatch.setattr(driver_classB, 'run_classB_jax', fake_run_classB)
        prediction = canonical_forward_solver(
            Sigma_H=0.01,
            backend='jax_classB',
            bianchi_type='V',
            A_init=1.0e-5,
            N_q=6,
            correction_level=0,
        )

        assert not prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_classB'
        assert prediction.metadata['backend'] == 'fake_classB_failure'

    def test_public_tilted_dispatch_propagates_driver_failure(self, jax_setup, monkeypatch):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        import rabbit.jax.run_tilted_bbn as tilted_driver

        def fake_run_tilted(_config):
            return SimpleNamespace(
                Yp=np.nan,
                DH=np.nan,
                success=False,
                metadata={
                    'backend': 'fake_tilted_failure',
                    'transport_mode': 'tilted_kappa_cascade_lmax2',
                },
            )

        monkeypatch.setattr(tilted_driver, 'run_tilted_bbn', fake_run_tilted)
        prediction = canonical_forward_solver(
            Sigma_H=0.01,
            backend='jax_tilted',
            bianchi_type='I',
            v0=1.0e-7,
            N_q=6,
            correction_level=0,
        )

        assert not prediction.success
        assert prediction.metadata['dispatch_backend'] == 'jax_tilted'
        assert prediction.metadata['backend'] == 'fake_tilted_failure'
