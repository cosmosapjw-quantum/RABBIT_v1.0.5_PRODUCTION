#!/usr/bin/env python3
"""Apply the RED-first F10 inner-quadrature and factor-two contract repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXTURE = Path(
    "native/rabbit_cpu/tests/fixtures/d081r1/f10_inner_quadrature_case.json"
)
PREFLIGHT = Path("native/rabbit_cpu/src/f10_packed_rhs_preflight_tests.rs")
KINEMATICS = Path("native/rabbit_cpu/src/f10_action_kinematics.rs")
PACKED_TESTS = Path("native/rabbit_cpu/src/f10_packed_rhs_tests.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def fixture_bits(value: dict[str, object], key: str) -> list[int]:
    encoded = value[key]
    if not isinstance(encoded, dict) or "bits" not in encoded:
        raise SystemExit(f"missing bit array {key}")
    bits = encoded["bits"]
    if not isinstance(bits, list):
        raise SystemExit(f"invalid bit array {key}")
    return [int(str(item), 16) for item in bits]


def rust_u64_const(name: str, bits: list[int]) -> str:
    lines = [f"const {name}: [u64; {len(bits)}] = ["]
    for start in range(0, len(bits), 4):
        chunk = bits[start : start + 4]
        lines.append("    " + ", ".join(f"0x{value:016x}" for value in chunk) + ",")
    lines.append("];")
    return "\n".join(lines)


def apply_red_test() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    if "f10_inner_gl12_gl48_rules_match_frozen_numpy_binary64_operator" in text:
        print("D-081R1E inner-quadrature RED test: NOOP")
        return

    text = replace_once(
        text,
        "use crate::f10_action_grid::F10ActionGrid;\n",
        "use crate::f10_action_grid::F10ActionGrid;\n"
        "use crate::f10_action_kinematics::{\n"
        "    F10CollisionConfig, angular_rule, electron_half_line_rule,\n"
        "};\n",
        "kinematic test imports",
    )
    text = replace_once(
        text,
        "const FIXTURE: &str = include_str!(\"../tests/fixtures/d081r1/retained_packed_rhs_case.json\");\n",
        "const FIXTURE: &str = include_str!(\"../tests/fixtures/d081r1/retained_packed_rhs_case.json\");\n"
        "const INNER_FIXTURE: &str = include_str!(\n"
        "    \"../tests/fixtures/d081r1/f10_inner_quadrature_case.json\",\n"
        ");\n",
        "inner fixture constant",
    )

    insertion = r'''

    #[test]
    fn f10_inner_gl12_gl48_rules_match_frozen_numpy_binary64_operator() {
        let value: Value =
            serde_json::from_str(INNER_FIXTURE).expect("valid frozen inner quadrature fixture");
        assert_eq!(
            value["schema"],
            "rabbit.d081r1e.f10_inner_quadrature.v1"
        );
        assert_eq!(value["numpy_version"], "2.4.4");
        assert_eq!(
            value["python_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );

        let angular = angular_rule(F10CollisionConfig::default()).unwrap();
        let expected_gl12_nodes = bit_array(&value["gl12_nodes"]);
        let expected_gl12_weights = bit_array(&value["gl12_weights"]);
        assert_eq!(
            angular
                .incoming_mu
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_nodes
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 incoming GL12 nodes differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .incoming_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 incoming GL12 weights differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .final_mu
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_nodes
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 final-state GL12 nodes differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .final_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 final-state GL12 weights differ from the frozen NumPy operator"
        );

        let temperature = bits(&value["temperature_probe_bits"]);
        let (electron_p2, electron_weights) =
            electron_half_line_rule(48, temperature).unwrap();
        let expected_p2 = bit_array(&value["electron_p2"]);
        let expected_electron_weights = bit_array(&value["electron_weights"]);
        assert_eq!(
            electron_p2
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_p2
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 GL48 electron momenta differ from the frozen NumPy operator"
        );
        assert_eq!(
            electron_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_electron_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 GL48 electron weights differ from the frozen NumPy operator"
        );
    }
'''
    closing = text.rfind("\n}")
    if closing < 0:
        raise SystemExit("preflight test module closing brace not found")
    PREFLIGHT.write_text(text[:closing] + insertion + text[closing:], encoding="utf-8")
    print("D-081R1E inner-quadrature RED test: CHANGED")


def apply_green(inner_decision: str) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if inner_decision not in {"PATCH_REQUIRED", "ALREADY_EXACT"}:
        raise SystemExit(f"invalid inner decision: {inner_decision}")

    packed = PACKED_TESTS.read_text(encoding="utf-8")
    old_half = "        assert!(scaled_difference(&no_half, correct) > 0.5);\n"
    new_half = '''        let doubled_correct: Vec<f64> =
            correct.iter().map(|value| 2.0 * value).collect();
        assert_hybrid_close(
            &no_half,
            &doubled_correct,
            maximum_absolute(&doubled_correct),
            1.0e-13,
        );
        let factor_two_residual = scaled_difference(&no_half, correct);
        assert!(
            (factor_two_residual - 0.5).abs() <= 512.0 * f64::EPSILON,
            "removing the particle/antiparticle one-half factor must give residual one-half; got {factor_two_residual:.17e}"
        );
'''
    if old_half in packed:
        packed = replace_once(packed, old_half, new_half, "factor-two mutation boundary")
        PACKED_TESTS.write_text(packed, encoding="utf-8")
    elif "factor_two_residual" not in packed:
        raise SystemExit("factor-two mutation assertion not found")

    if inner_decision == "ALREADY_EXACT":
        print("D-081R1E inner quadrature source: already exact")
        return

    source = KINEMATICS.read_text(encoding="utf-8")
    if "EXACT_F10_GL12_NODE_BITS" in source:
        print("D-081R1E inner quadrature source: NOOP")
        return

    constants = "\n\n".join(
        [
            rust_u64_const("EXACT_F10_GL12_NODE_BITS", fixture_bits(fixture, "gl12_nodes")),
            rust_u64_const(
                "EXACT_F10_GL12_WEIGHT_BITS", fixture_bits(fixture, "gl12_weights")
            ),
            rust_u64_const("EXACT_F10_GL48_NODE_BITS", fixture_bits(fixture, "gl48_nodes")),
            rust_u64_const(
                "EXACT_F10_GL48_WEIGHT_BITS", fixture_bits(fixture, "gl48_weights")
            ),
        ]
    )
    helper = f'''{constants}

fn decode_exact_rule<const N: usize>(
    node_bits: [u64; N],
    weight_bits: [u64; N],
) -> Vec<(f64, f64)> {{
    node_bits
        .into_iter()
        .zip(weight_bits)
        .map(|(node, weight)| (f64::from_bits(node), f64::from_bits(weight)))
        .collect()
}}

fn f10_gauss_legendre_rule(order: usize) -> Result<Vec<(f64, f64)>, &'static str> {{
    match order {{
        12 => Ok(decode_exact_rule(
            EXACT_F10_GL12_NODE_BITS,
            EXACT_F10_GL12_WEIGHT_BITS,
        )),
        48 => Ok(decode_exact_rule(
            EXACT_F10_GL48_NODE_BITS,
            EXACT_F10_GL48_WEIGHT_BITS,
        )),
        _ => gauss_legendre_rule(order),
    }}
}}
'''
    source = replace_once(
        source,
        "use crate::quadrature::gauss_legendre_rule;\n",
        "use crate::quadrature::gauss_legendre_rule;\n\n"
        "// The frozen Python comparator defines the admitted F10 inner rules\n"
        "// through NumPy 2.4.4. Preserve those finite-dimensional binary64\n"
        "// operators only for GL12 and GL48; all other orders retain the\n"
        "// generic Rust quadrature implementation.\n"
        + helper,
        "F10 exact inner-rule helper",
    )
    source = replace_once(
        source,
        "    let incoming = gauss_legendre_rule(config.incoming_polar_order)?;\n",
        "    let incoming = f10_gauss_legendre_rule(config.incoming_polar_order)?;\n",
        "incoming GL rule",
    )
    source = replace_once(
        source,
        "    let final_state = gauss_legendre_rule(config.final_polar_order)?;\n",
        "    let final_state = f10_gauss_legendre_rule(config.final_polar_order)?;\n",
        "final GL rule",
    )
    source = replace_once(
        source,
        "    for (coordinate, weight) in gauss_legendre_rule(order)? {\n",
        "    for (coordinate, weight) in f10_gauss_legendre_rule(order)? {\n",
        "electron GL rule",
    )
    KINEMATICS.write_text(source, encoding="utf-8")
    print("D-081R1E inner quadrature source: CHANGED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("red", "green"), required=True)
    parser.add_argument(
        "--inner-decision",
        choices=("PATCH_REQUIRED", "ALREADY_EXACT"),
    )
    args = parser.parse_args()
    if args.phase == "red":
        apply_red_test()
        return
    if args.inner_decision is None:
        raise SystemExit("--inner-decision is required for the green phase")
    apply_green(args.inner_decision)


if __name__ == "__main__":
    main()
