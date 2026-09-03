"""Tests for the post-seal JSON-normalized final verifier wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WRAPPER = (
    Path(__file__).parents[1]
    / "00_F10_PHYSICAL_PREFIX_DIAGNOSIS/verify_final_json_normalized.py"
)


def _load_wrapper():
    specification = importlib.util.spec_from_file_location(
        "f10_final_verify_wrapper", WRAPPER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_json_normalization_changes_only_json_container_representation():
    wrapper = _load_wrapper()
    payload = {"records": [{"legs": ("nu_e", "nu_mu"), "value": 1.0}]}

    normalized = wrapper.json_normalize(payload)

    assert normalized == {
        "records": [{"legs": ["nu_e", "nu_mu"], "value": 1.0}]
    }
    assert payload["records"][0]["legs"] == ("nu_e", "nu_mu")
