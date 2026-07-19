from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_test_counts.py"


def _load_sync_test_counts_module():
    spec = importlib.util.spec_from_file_location("sync_test_counts", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_test_counts_fails_closed_when_pytest_count_is_unparseable(monkeypatch):
    module = _load_sync_test_counts_module()

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="collection aborted before count summary",
            stderr="synthetic import error",
            returncode=2,
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Could not parse pytest collection count"):
        module.get_counts()
