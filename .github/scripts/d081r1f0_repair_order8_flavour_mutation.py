#!/usr/bin/env python3
"""Repair the order-eight flavour-swap mutation metrology.

The packed JVP contains heterogeneous spectral, photon-temperature, and
elapsed-time rows. The previous mutation check normalized a spectral-block
permutation by the entire packed-vector scale, so the non-spectral rows could
hide a wrong flavour ordering. This test-only repair compares the permuted and
frozen values on the 3*n spectral output block, leaving every frozen numerical
threshold and production formula unchanged.
"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPOSITORY_ROOT / "native/rabbit_cpu/src/f10_packed_rhs_jvp_order8_tests.rs"

OLD = '''    let mut flavour_mutant = base.values.clone();
    for node in 0..grid.order {
        flavour_mutant.swap(node, grid.order + node);
    }
    assert!(global_relative(&flavour_mutant, &expected_jvp) > 5.0e-7);
'''

NEW = '''    let mut flavour_mutant = base.values.clone();
    for node in 0..grid.order {
        flavour_mutant.swap(node, grid.order + node);
    }
    let spectral_size = 3 * grid.order;
    assert!(
        global_relative(
            &flavour_mutant[..spectral_size],
            &expected_jvp[..spectral_size],
        ) > 5.0e-7,
        "flavour swap was hidden by heterogeneous non-spectral output scales"
    );
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if NEW.strip() in text:
        print("D-081R1F0 flavour mutation repair: ALREADY_APPLIED")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"expected one mutation block, found {count}")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("D-081R1F0 flavour mutation repair: CHANGED")


if __name__ == "__main__":
    main()
