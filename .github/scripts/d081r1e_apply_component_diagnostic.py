#!/usr/bin/env python3
"""Add non-gating retained component diagnostics to the D-081R1E tests."""

from __future__ import annotations

from pathlib import Path


TESTS = Path("native/rabbit_cpu/src/f10_packed_rhs_tests.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TESTS.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '        "combined_action_native": canonical["arrays"]["total_native"].clone(),\n'
        '        "combined_action_modal": canonical["arrays"]["total_modal"].clone(),\n',
        '        "self_action_native": canonical["arrays"]["self_native"].clone(),\n'
        '        "electron_action_native": canonical["arrays"]["electron_native"].clone(),\n'
        '        "combined_action_native": canonical["arrays"]["total_native"].clone(),\n'
        '        "self_action_modal": canonical["arrays"]["self_modal"].clone(),\n'
        '        "electron_action_modal": canonical["arrays"]["electron_modal"].clone(),\n'
        '        "combined_action_modal": canonical["arrays"]["total_modal"].clone(),\n',
        "canonical component arrays",
    )

    helper_anchor = '''fn scaled_difference(actual: &[f64], reference: &[f64]) -> f64 {\n    maximum_absolute_difference(actual, reference)\n        / maximum_absolute(actual)\n            .max(maximum_absolute(reference))\n            .max(f64::MIN_POSITIVE)\n}\n\n#[cfg(test)]\n'''
    helper = '''fn scaled_difference(actual: &[f64], reference: &[f64]) -> f64 {\n    maximum_absolute_difference(actual, reference)\n        / maximum_absolute(actual)\n            .max(maximum_absolute(reference))\n            .max(f64::MIN_POSITIVE)\n}\n\nfn report_array_diagnostic(name: &str, actual: &[f64], expected: &[f64]) {\n    assert_eq!(actual.len(), expected.len());\n    let actual_scale = maximum_absolute(actual);\n    let expected_scale = maximum_absolute(expected);\n    let global_scale = actual_scale.max(expected_scale).max(f64::MIN_POSITIVE);\n    let mut max_abs = 0.0_f64;\n    let mut max_abs_index = 0_usize;\n    let mut max_local_relative = 0.0_f64;\n    let mut max_local_index = 0_usize;\n    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {\n        let difference = (observed - reference).abs();\n        if difference > max_abs {\n            max_abs = difference;\n            max_abs_index = index;\n        }\n        let local = difference\n            / observed\n                .abs()\n                .max(reference.abs())\n                .max(f64::MIN_POSITIVE);\n        if local > max_local_relative {\n            max_local_relative = local;\n            max_local_index = index;\n        }\n    }\n    eprintln!(\n        "D081R1E_DIAG {name}: max_abs={max_abs:.17e} global_relative={:.17e} max_abs_index={max_abs_index} actual_at_max_abs={:.17e} expected_at_max_abs={:.17e} max_local_relative={max_local_relative:.17e} max_local_index={max_local_index} actual_at_local={:.17e} expected_at_local={:.17e}",\n        max_abs / global_scale,\n        actual[max_abs_index],\n        expected[max_abs_index],\n        actual[max_local_index],\n        expected[max_local_index],\n    );\n}\n\nfn report_total_cancellation_diagnostic(\n    name: &str,\n    actual: &[f64],\n    expected: &[f64],\n    expected_left: &[f64],\n    expected_right: &[f64],\n) {\n    assert_eq!(actual.len(), expected.len());\n    assert_eq!(actual.len(), expected_left.len());\n    assert_eq!(actual.len(), expected_right.len());\n    let mut worst = 0.0_f64;\n    let mut worst_index = 0_usize;\n    for index in 0..actual.len() {\n        let difference = (actual[index] - expected[index]).abs();\n        let envelope = actual[index]\n            .abs()\n            .max(expected[index].abs())\n            .max(expected_left[index].abs() + expected_right[index].abs())\n            .max(f64::MIN_POSITIVE);\n        let residual = difference / envelope;\n        if residual > worst {\n            worst = residual;\n            worst_index = index;\n        }\n    }\n    eprintln!(\n        "D081R1E_DIAG {name}: cancellation_scaled={worst:.17e} index={worst_index} actual={:.17e} expected={:.17e} expected_left={:.17e} expected_right={:.17e}",\n        actual[worst_index],\n        expected[worst_index],\n        expected_left[worst_index],\n        expected_right[worst_index],\n    );\n}\n\n#[cfg(test)]\n'''
    text = replace_once(text, helper_anchor, helper, "diagnostic helpers")

    assertion_anchor = '''        let expected_action_native = bit_array(&value["combined_action_native"]);\n        let expected_action_modal = bit_array(&value["combined_action_modal"]);\n        let action_scale = maximum_absolute(&expected_action_native);\n'''
    diagnostic = '''        let expected_action_native = bit_array(&value["combined_action_native"]);\n        let expected_action_modal = bit_array(&value["combined_action_modal"]);\n        let expected_self_native = bit_array(&value["self_action_native"]);\n        let expected_electron_native = bit_array(&value["electron_action_native"]);\n        let expected_self_modal = bit_array(&value["self_action_modal"]);\n        let expected_electron_modal = bit_array(&value["electron_action_modal"]);\n        report_array_diagnostic(\n            "self_native",\n            &result.combined_action.self_action.native,\n            &expected_self_native,\n        );\n        report_array_diagnostic(\n            "electron_native",\n            &result.combined_action.electron_action.native,\n            &expected_electron_native,\n        );\n        report_array_diagnostic(\n            "total_native",\n            &result.combined_action.native_total,\n            &expected_action_native,\n        );\n        report_total_cancellation_diagnostic(\n            "total_native",\n            &result.combined_action.native_total,\n            &expected_action_native,\n            &expected_self_native,\n            &expected_electron_native,\n        );\n        report_array_diagnostic(\n            "self_modal",\n            &result.combined_action.self_action.modal,\n            &expected_self_modal,\n        );\n        report_array_diagnostic(\n            "electron_modal",\n            &result.combined_action.electron_action.modal,\n            &expected_electron_modal,\n        );\n        report_array_diagnostic(\n            "total_modal",\n            &result.combined_action.modal_total,\n            &expected_action_modal,\n        );\n        report_total_cancellation_diagnostic(\n            "total_modal",\n            &result.combined_action.modal_total,\n            &expected_action_modal,\n            &expected_self_modal,\n            &expected_electron_modal,\n        );\n        eprintln!(\n            "D081R1E_DIAG counters: actual_rejections={} expected_rejections={} actual_corrections={} expected_corrections={}",\n            result.diagnostics.whole_reaction_domain_rejections,\n            value["support_and_roundoff_metrology"]["whole_reaction_domain_rejections"]\n                .as_u64()\n                .expect("expected rejection count"),\n            result.diagnostics.matrix_roundoff_corrections,\n            value["support_and_roundoff_metrology"]["matrix_roundoff_corrections"]\n                .as_u64()\n                .expect("expected correction count"),\n        );\n        let action_scale = maximum_absolute(&expected_action_native);\n'''
    text = replace_once(text, assertion_anchor, diagnostic, "retained component diagnostic")

    TESTS.write_text(text, encoding="utf-8")
    print("D-081R1E non-gating component diagnostic applied")


if __name__ == "__main__":
    main()
