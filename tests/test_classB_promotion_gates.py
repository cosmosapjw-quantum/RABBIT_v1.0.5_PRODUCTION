"""
tests/test_classB_promotion_gates.py — Class B single-slice promotion gates.
"""
import pytest


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassBGates:
    def test_config_constructible(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        cfg = JAXClassBConfig(bianchi_type="TYPE_V", A_init=0.001,
                               Sigma_H_plus=0.05, N_q=6)
        assert cfg.bianchi_type == "TYPE_V"
        assert cfg.A_init == 0.001

    def test_A_init_zero_valid(self):
        from rabbit.jax.driver_classB import JAXClassBConfig
        cfg = JAXClassBConfig(bianchi_type="TYPE_V", A_init=0.0, N_q=6)
        assert cfg.A_init == 0.0

    def test_public_dispatch_fails_closed(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        with pytest.raises(ValueError, match="retired"):
            canonical_forward_solver(backend='jax_classB')

    def test_public_capability_absent_but_metadata_preserved(self):
        from rabbit.config.backend_capabilities import (
            CAPABILITY_BY_BACKEND,
            CAPABILITY_BY_KEY,
        )
        assert 'jax_classB' not in CAPABILITY_BY_BACKEND
        assert 'jax_classB_driver' in CAPABILITY_BY_KEY

    def test_promotion_packet_exists(self):
        import os
        assert os.path.exists("docs/CLASSB_PROMOTION_PACKET.md")
