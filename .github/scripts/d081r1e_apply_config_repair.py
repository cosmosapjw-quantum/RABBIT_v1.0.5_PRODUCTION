#!/usr/bin/env python3
"""Apply the bounded retained-configuration and factor-two mutation repair.

This script changes only the D-081R1E tests produced by the GREEN driver.
It does not change collision physics, quadrature constants, fixture bytes,
parity tolerances, or the packed-RHS implementation.
"""

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
        "use crate::f10_action_grid::F10ActionGrid;\n",
        "use crate::f10_action_grid::F10ActionGrid;\n"
        "use crate::f10_action_kinematics::F10CollisionConfig;\n"
        "use crate::f10_combined_action::F10CombinedActionConfig;\n",
        "imports",
    )

    text = replace_once(
        text,
        '        "order": canonical["configuration"]["order"].clone(),\n',
        '        "order": canonical["configuration"]["order"].clone(),\n'
        '        "temperature_start_mev_bits": canonical["configuration"]["temperature_start_mev_bits"].clone(),\n'
        '        "incoming_polar_order": canonical["configuration"]["incoming_polar_order"].clone(),\n'
        '        "final_polar_order": canonical["configuration"]["final_polar_order"].clone(),\n'
        '        "final_azimuth_order": canonical["configuration"]["final_azimuth_order"].clone(),\n'
        '        "electron_radial_order": canonical["configuration"]["electron_radial_order"].clone(),\n',
        "canonical configuration view",
    )

    fixture_function = '''fn retained_fixture() -> Value {\n    let canonical: Value = serde_json::from_str(RETAINED_FIXTURE)\n        .expect("valid canonical retained packed-RHS fixture");\n    assert_eq!(\n        canonical["schema"],\n        "rabbit.d081r1e0.retained_packed_rhs_oracle.v1"\n    );\n'''
    if fixture_function not in text:
        raise SystemExit("canonical retained_fixture function not found")

    helper_anchor = '''}\n\nfn control_fixture() -> Value {\n'''
    helper = '''}\n\nfn retained_config(value: &Value) -> F10PackedRhsConfig {\n    let mut config = F10PackedRhsConfig::default();\n    config.t_start_mev = bits(&value["temperature_start_mev_bits"]);\n    config.combined_action = F10CombinedActionConfig {\n        collision: F10CollisionConfig {\n            incoming_polar_order: usize::try_from(\n                value["incoming_polar_order"].as_u64().expect("incoming order"),\n            )\n            .expect("incoming order fits usize"),\n            final_polar_order: usize::try_from(\n                value["final_polar_order"].as_u64().expect("final order"),\n            )\n            .expect("final order fits usize"),\n            final_azimuth_order: usize::try_from(\n                value["final_azimuth_order"].as_u64().expect("azimuth order"),\n            )\n            .expect("azimuth order fits usize"),\n            electron_radial_order: usize::try_from(\n                value["electron_radial_order"].as_u64().expect("radial order"),\n            )\n            .expect("radial order fits usize"),\n        },\n        ..F10CombinedActionConfig::default()\n    };\n    config\n}\n\nfn control_fixture() -> Value {\n'''
    text = replace_once(text, helper_anchor, helper, "retained config helper")

    text = replace_once(
        text,
        '''        let ln_a = bits(&value["ln_a_bits"]);\n        let result = evaluate_f10_packed_rhs(\n            &grid,\n            ln_a,\n            &state,\n            F10PackedRhsConfig::default(),\n        )\n''',
        '''        let ln_a = bits(&value["ln_a_bits"]);\n        let config = retained_config(&value);\n        assert_eq!(config.combined_action.collision.incoming_polar_order, 4);\n        assert_eq!(config.combined_action.collision.final_polar_order, 4);\n        assert_eq!(config.combined_action.collision.final_azimuth_order, 4);\n        assert_eq!(config.combined_action.collision.electron_radial_order, 24);\n        let result = evaluate_f10_packed_rhs(&grid, ln_a, &state, config)\n''',
        "retained primary configuration",
    )

    text = replace_once(
        text,
        '''        let repeated = evaluate_f10_packed_rhs(\n            &grid,\n            ln_a,\n            &state,\n            F10PackedRhsConfig::default(),\n        )\n''',
        '''        let repeated = evaluate_f10_packed_rhs(&grid, ln_a, &state, config)\n''',
        "retained repeat configuration",
    )

    text = replace_once(
        text,
        '''        assert!(scaled_difference(&no_half, correct) > 0.5);\n''',
        '''        let no_half_difference = scaled_difference(&no_half, correct);\n        assert!(\n            (no_half_difference - 0.5).abs() <= 64.0 * f64::EPSILON,\n            "removing the particle/antiparticle half factor must produce the analytic factor-two mutation; observed scaled difference={no_half_difference:.17e}"\n        );\n''',
        "factor-two mutation gate",
    )

    TESTS.write_text(text, encoding="utf-8")
    print("D-081R1E retained configuration and factor-two mutation repair applied")


if __name__ == "__main__":
    main()
