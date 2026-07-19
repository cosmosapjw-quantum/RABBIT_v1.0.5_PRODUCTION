"""PR-04: Class B Layered Truth Sync — TDD tests.

The current 6/6/6/6 layered support must be registry-driven truth,
not a manually maintained narrative.
"""
import pytest
import re

pytestmark = [pytest.mark.production, pytest.mark.release_smoke]


class TestLayeredRegistryTruth:
    """Layered counts must come from registry, not manual text."""

    def test_registry_has_layered_metadata(self):
        """Feature registry must have explicit layered counts."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        assert hasattr(f, 'layered_scope'), \
            "classB_bbn missing layered_scope metadata"

    def test_layered_counts_are_correct(self):
        """Layered counts must be 6/6/6/6 after h-family representative gold promotion."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        ls = f.layered_scope
        assert ls['geometry'] == 6
        assert ls['family_envelope'] == 6
        assert ls['bbn_smoke'] == 6
        assert ls['gold_locked'] == 6

    def test_layered_types_are_specified(self):
        """Each layer must list its constituent types."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        ls = f.layered_scope
        assert len(ls.get('geometry_types', [])) == 6
        assert len(ls.get('envelope_types', [])) == 6
        assert len(ls.get('smoke_types', [])) == 6
        assert ls.get('gold_type') == 'TYPE_V/TYPE_IV/TYPE_III/TYPE_VIH/TYPE_VIIH/TYPE_VI_M19'


class TestLayeredDocSync:
    """Generated docs must reflect layered registry truth."""

    def test_supported_capabilities_layers_generated(self):
        """SUPPORTED_CAPABILITIES Class B layers must be in a generated block."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert "<!-- BEGIN:CLASSB_LAYERS" in text, \
            "SUPPORTED_CAPABILITIES missing CLASSB_LAYERS generated block"

    def test_layer_counts_match_registry(self):
        """Doc layer counts must match registry layered_scope."""
        from rabbit.config.feature_capabilities import FEATURE_BY_KEY
        f = FEATURE_BY_KEY['classB_bbn']
        ls = f.layered_scope
        text = open("SUPPORTED_CAPABILITIES.md").read()
        assert f"{ls['geometry']} types" in text or f"{ls['geometry']}-type" in text, \
            f"SUPPORTED_CAPABILITIES geometry count {ls['geometry']} not found"

    def test_no_production_in_geometry_layer(self):
        """Geometry/envelope layers must never say 'production'."""
        text = open("SUPPORTED_CAPABILITIES.md").read()
        layers_start = text.find("BEGIN:CLASSB_LAYERS")
        layers_end = text.find("END:CLASSB_LAYERS")
        if layers_start == -1:
            pytest.skip("No CLASSB_LAYERS block")
        layers_text = text[layers_start:layers_end]
        for line in layers_text.split('\n'):
            if 'Geometry' in line or 'envelope' in line.lower():
                assert 'production' not in line.lower(), \
                    f"Non-production layer claims production: {line.strip()}"
