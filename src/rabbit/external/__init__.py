"""rabbit.external — live cross-code validation harness.

Phase ζ deliverable (Plan §4.6 / §5). Replaces the static
``tests/fixtures/tier3_cross_code.json`` round-tripped numbers with
live subprocess calls to independent BBN codes. Each external wrapper:

- Detects availability gracefully (no hard import dependency).
- Returns ``None`` or raises a typed ``ExternalCodeUnavailable`` error
  when its backend is not installed; tests then skip with reason.
- Publishes a single CLI: ``python -m rabbit.external.run_cross_code
  --code <name> --eta <value> [--tau-n <value>] [--n-eff <value>]``.

Supported back-ends
-------------------
- ``nudec_bsm``      — NUDEC_BSM Python (Mangano-style FLRW N_eff)
- ``alterbbn``       — AlterBBN via Docker (FLRW Y_p, D/H, Li7/H)
- ``parthenope``     — PArthENoPE 3.0 via Docker (optional)

Conventions (see ``conventions.md``)
------------------------------------
- ``eta`` is the absolute baryon-to-photon ratio (≈ 6.104e-10), not eta_10.
- ``Y_p`` is the He-4 mass fraction (matches AlterBBN, PArthENoPE 3.0,
  PRIMAT-AC2024). Some legacy codes report number fraction; conversion
  applied inside the wrapper.
- ``D/H`` is by number ratio.
- ``N_eff`` includes incomplete-decoupling heating (SM benchmark 3.044
  per Mangano 2005 / Froustey 2020).
"""

from __future__ import annotations


class ExternalCodeUnavailable(RuntimeError):
    """Raised when an external cross-code backend cannot be reached.

    Tests should use ``pytest.importorskip`` or check
    ``has_<backend>()`` to convert this into a test skip.
    """

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(f"{backend} unavailable: {reason}")
        self.backend = backend
        self.reason = reason
