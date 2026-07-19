"""BD616 — smoke test for the backend bake-off probe script.

Verifies the JSON schema and decision shape without running a full endpoint
solve (kernel+rhs only, tiny n_q, single rep). Marked jax because the kernel
level imports and compiles the JAX collision kernels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]


def test_bakeoff_probe_schema(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import probe_clean_core_backend_bakeoff as probe

    # The script does ``ROOT / args.out``; an absolute path is preserved as-is.
    out = tmp_path / "bakeoff.json"
    rc = probe.main([
        "--n-q", "8", "--repeat", "1", "--warmups", "1",
        "--skip-solver", "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text())
    for key in ("meta", "kernel", "rhs", "endpoint", "decision"):
        assert key in data, f"missing top-level key {key}"
    assert data["decision"]["verdict"] in {"proceed", "stop"}
    assert data["endpoint"] == []            # --skip-solver
    assert len(data["kernel"]) >= 2          # numpy + jax rows
    for row in data["kernel"]:
        for f in ("kernel", "backend", "n_q", "median_s", "min_s", "max_s"):
            assert f in row, f"kernel row missing {f}"
    assert "amdahl_bound_end_to_end" in data["decision"]
