#!/usr/bin/env python3
"""Retired NUTS-3D demo retained as an explicit B-05 fail-closed entry.

The host Type-I forward solver is not JAX-traceable.  The former finite-
difference custom-VJP could turn failed stencil points into fabricated
gradients, so this script produces no samples or output artifact.  B-05 must
provide a canonical full-solver inference path before this entry can run.

The command exits nonzero with the same frozen error as the five BBN JAX
convenience wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE

    print(BBN_JAX_SAMPLER_UNAVAILABLE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
