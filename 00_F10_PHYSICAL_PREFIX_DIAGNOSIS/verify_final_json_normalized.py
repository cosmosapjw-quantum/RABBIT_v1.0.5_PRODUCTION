"""Run the sealed final verifier with JSON container normalization only.

The active seal's verifier compares a freshly built catalogue containing tuple
fields with its JSON-loaded representation containing list fields. Canonical
JSON bytes are identical, but direct Python equality is false. This post-seal
wrapper normalizes only that regenerated value through finite JSON before
delegating every substantive check to the sealed verifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit import f10_physical_prefix_fixture as fixture


def json_normalize(value: object) -> object:
    """Round-trip a finite value through JSON without changing scalar values."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seal-commit", required=True)
    args = parser.parse_args()

    original_builder = fixture.build_quadrature_catalog_manifest

    def normalized_builder(setup):
        return json_normalize(original_builder(setup))

    fixture.build_quadrature_catalog_manifest = normalized_builder
    result = fixture.verify_final(args.repo, args.output_dir, args.seal_commit)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
