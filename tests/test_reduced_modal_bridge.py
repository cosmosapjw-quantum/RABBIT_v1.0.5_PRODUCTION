from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rabbit.transport.reduced_modal_bank import load_runtime_bank, project_vector_onto_bank
import rabbit.transport.reduced_modal_bridge as rmb


@dataclass
class FakeResult:
    tangency_D2: float
    delta_rho_nu: float
    delta_I: np.ndarray
    C_monopole: np.ndarray
    f_monopole: np.ndarray
    theta_per_ray: np.ndarray


def _write_bank(tmp_path: Path) -> Path:
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)

    data = {
        "q_nodes": q.tolist(),
        "q_weights": w.tolist(),
        "modes": {
            "fd2": [1.0, 0.0, -1.0, 0.0],
            "shared_windowed": [
                [0.0, 1.0, 0.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
            ],
            "species_windowed": {
                "nue": [
                    [0.5, 0.5, -0.5, -0.5],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            },
            "branch_windowed": {},
        },
    }
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(data))
    return p


def test_project_vector_exact_reconstruction(tmp_path: Path):
    bank_path = _write_bank(tmp_path)
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)

    bank = load_runtime_bank(bank_path, "nue", q, w, n_shared=2, n_species=1, include_fd2=True)
    vec = 2.0 * bank.raw_basis[0] - 0.3 * bank.raw_basis[1] + 1.1 * bank.raw_basis[2] + 0.7 * bank.raw_basis[3]
    proj = project_vector_onto_bank(vec, bank)

    np.testing.assert_allclose(proj["recon"], vec, rtol=1e-10, atol=1e-10)
    assert proj["cos"] > 0.999999999
    assert proj["resid"] < 1e-12


def test_reduced_bridge_preserves_exact_combo(tmp_path: Path, monkeypatch):
    bank_path = _write_bank(tmp_path)
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)

    bank = load_runtime_bank(bank_path, "nue", q, w, n_shared=2, n_species=1, include_fd2=True)
    C = 1.2 * bank.raw_basis[0] - 0.4 * bank.raw_basis[1] + 0.8 * bank.raw_basis[2] + 0.6 * bank.raw_basis[3]

    fake = FakeResult(
        tangency_D2=0.0,
        delta_rho_nu=2.5,
        delta_I=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        C_monopole=C.copy(),
        f_monopole=np.ones_like(C),
        theta_per_ray=np.linspace(-1.0, 1.0, len(C)),
    )

    def fake_raw_apply(**kwargs):
        return fake

    monkeypatch.setattr(rmb, "_raw_apply_species_tagged_bridge", fake_raw_apply)

    out = rmb.apply_reduced_modal_species_bridge(
        species="nue",
        I=np.zeros(4),
        J=np.zeros(4),
        w0=np.ones(4) / 4.0,
        q_nodes=q,
        q_weights=w,
        T_gamma=1.0,
        T_nu_e=0.9,
        T_nu_x=0.8,
        H=1.0,
        bank_path=str(bank_path),
        n_shared=2,
        n_species=1,
        include_fd2=True,
    )

    np.testing.assert_allclose(out.C_monopole, C, rtol=1e-10, atol=1e-10)
    assert hasattr(out, "bridge_debug")
    assert out.bridge_debug["cos"] > 0.999999999
    assert out.bridge_debug["resid"] < 1e-12
    assert out.bridge_debug["authoritative_path"] == "raw_characteristic"
    assert out.bridge_debug["surrogate_path"] == "reduced_modal"
    assert out.bridge_debug["reduced_modal_mode"] == "offline_only"
    assert out.bridge_debug["reduced_modal_status"] == "accepted_research_only"
