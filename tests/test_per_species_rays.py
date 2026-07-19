"""Tests for per-species ray infrastructure."""
import pytest
import numpy as np


class TestPerSpeciesRayState:
    def test_import(self):
        from rabbit.transport.per_species_rays import PerSpeciesRayState
        assert PerSpeciesRayState is not None

    def test_from_shared(self):
        from rabbit.transport.per_species_rays import PerSpeciesRayState
        I = np.zeros(12)
        J = np.ones(12)
        state = PerSpeciesRayState.from_shared(I, J, 0.0, species='nue')
        assert state.species == 'nue'
        assert state.coupling_g > 0.5  # ν_e has stronger coupling

    def test_species_coupling_hierarchy(self):
        from rabbit.transport.per_species_rays import PerSpeciesRayState
        I, J = np.zeros(12), np.ones(12)
        nue = PerSpeciesRayState.from_shared(I, J, 0.0, 'nue')
        nux = PerSpeciesRayState.from_shared(I, J, 0.0, 'nux')
        # ν_e couples ~4.7× stronger than ν_x
        assert nue.coupling_g > 4 * nux.coupling_g

    def test_monopole_isotropic(self):
        from rabbit.transport.per_species_rays import PerSpeciesRayState
        from numpy.polynomial.legendre import leggauss
        mu, w = leggauss(12)
        I = np.zeros(12)
        J = np.ones(12)
        state = PerSpeciesRayState.from_shared(I, J, 0.0, 'nue')
        q = np.array([1.0, 3.0, 5.0, 7.0])
        mono = state.monopole(w, q)
        f_eq = 1.0 / (np.exp(q) + 1)
        np.testing.assert_allclose(mono, f_eq, atol=1e-10)

    def test_guard_tier1_silent(self):
        from rabbit.transport.per_species_rays import species_identical_guard
        assert species_identical_guard(1) == True

    def test_guard_tier2_warns(self):
        import warnings
        from rabbit.transport.per_species_rays import species_identical_guard
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = species_identical_guard(2)
            assert result == False
            assert len(w) == 1
            assert "species-identical" in str(w[0].message)
