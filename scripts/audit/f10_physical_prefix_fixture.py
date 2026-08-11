"""Build and verify the F-10 physical-prefix provenance fixture.

This branch-local audit utility does not move a public capability or gate.  It
keeps deterministic input bytes separate from physical receipt execution so a
Git commit can prospectively seal the latter.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np


def canonical_json_bytes(value: object) -> bytes:
    """Return UTF-8 JSON bytes with one stable, finite-number encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def float64_le_bytes(values: np.ndarray) -> bytes:
    """Return contiguous little-endian float64 bytes independent of host order."""

    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def write_deterministic_npz(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    """Write numeric NPY members in a name-sorted, timestamp-fixed ZIP."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object dtype is forbidden in a sealed NPZ fixture")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(
                entry,
                buffer.getvalue(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )


def load_numeric_npz(path: Path) -> dict[str, np.ndarray]:
    """Load a sealed NPZ without pickle and reject object-bearing members."""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("object dtype is forbidden in a sealed NPZ fixture")
    return arrays
