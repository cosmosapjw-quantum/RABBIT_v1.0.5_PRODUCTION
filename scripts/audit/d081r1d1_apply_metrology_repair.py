#!/usr/bin/env python3
"""Apply the bounded D-081R1D1 spectral metrology repair.

This script changes only the focused Rust test module.  It preserves the
relative acceptance gate away from zero, adds a roundoff-sized absolute floor
for cancellation-dominated coefficients, and proves a 1e-8 material mutation
still fails.  It is idempotent so a source-publishing workflow can rerun safely.
"""

from __future__ import annotations

from pathlib import Path


TEST_PATH = Path("native/rabbit_cpu/src/f10_action_foundations_tests.rs")

OLD_HELPER = '''fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| scaled_residual(left, right))
        .fold(0.0_f64, f64::max);
    assert!(
        maximum <= tolerance,
        "maximum scaled residual {maximum:.17e} exceeds {tolerance:.17e}"
    );
}
'''

NEW_HELPER = '''fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let block_scale = actual
        .iter()
        .chain(expected)
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    // Keep the relative gate away from zero, while admitting only the
    // measured order-eight dot-product association floor near exact zero.
    let absolute_floor = 32.0 * f64::EPSILON * block_scale;
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| {
            let absolute = (left - right).abs();
            if absolute <= absolute_floor {
                0.0
            } else {
                scaled_residual(left, right)
            }
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum <= tolerance,
        "maximum scaled residual {maximum:.17e} exceeds {tolerance:.17e}; \
         near-zero absolute floor is {absolute_floor:.17e}"
    );
}
'''

TEST_MARKER = '''    fn foundations_fail_closed_and_mutations_are_detected() {
        assert!(F10ActionGrid::affine_legendre(7, 8.0).is_err());
'''

TEST_REPLACEMENT = '''    fn foundations_fail_closed_and_mutations_are_detected() {
        let roundoff_actual = [-8.899_131_431_761_020e-16, 11.313_708_498_984_761];
        let roundoff_expected = [8.881_784_197_001_252e-16, 11.313_708_498_984_763];
        assert_slice_close(&roundoff_actual, &roundoff_expected, 2.0e-12);

        let material_mutation = std::panic::catch_unwind(|| {
            assert_slice_close(&[0.0, 1.0], &[0.0, 1.0 + 1.0e-8], 2.0e-12);
        });
        assert!(material_mutation.is_err());

        assert!(F10ActionGrid::affine_legendre(7, 8.0).is_err());
'''


def apply() -> bool:
    text = TEST_PATH.read_text(encoding="utf-8")
    original = text

    if "near-zero absolute floor is" not in text:
        if text.count(OLD_HELPER) != 1:
            raise SystemExit("original assert_slice_close helper was not found exactly once")
        text = text.replace(OLD_HELPER, NEW_HELPER)

    if "let roundoff_actual" not in text:
        if text.count(TEST_MARKER) != 1:
            raise SystemExit("existing mutation test marker was not found exactly once")
        text = text.replace(TEST_MARKER, TEST_REPLACEMENT)

    if text != original:
        TEST_PATH.write_text(text, encoding="utf-8")
        return True
    return False


if __name__ == "__main__":
    changed = apply()
    print("CHANGED" if changed else "ALREADY_APPLIED")
